import pandas as pd
import numpy as np
import plotly.express as px
import logging
import multiprocessing
from joblib import Parallel, delayed
from .queries import select_query

auto_show_plots = True


def get_data(bigquery, start_time, end_time, gcp_project: str, sample_rate=1.0):
    client = bigquery.Client(project=gcp_project)

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "start_time", "TIMESTAMP", pd.Timestamp(start_time).to_pydatetime()
            ),
            bigquery.ScalarQueryParameter(
                "end_time", "TIMESTAMP", pd.Timestamp(end_time).to_pydatetime()
            ),
            bigquery.ScalarQueryParameter("sample_rate", "NUMERIC", sample_rate),
        ]
    )

    logging.getLogger().setLevel(logging.DEBUG)
    query_job = client.query(select_query(), job_config=job_config)
    logging.getLogger().setLevel(logging.INFO)
    return query_job.to_dataframe()


def extract_thirdparty(s):
    tp = s.split(".")[0].lower()
    if tp in (
    "datastore", "cache", "elastic", "task", "vision", "experiments", "analytics", "appstore", "firebaseauth"):
        return tp
    return np.nan


def postprocess(df):
    df["root_module"] = df.root_name.str.extract(r"/api/v1.0/[^/]+/([^/]+)/")
    df["route"] = df.root_name.str.extract(r"/api/v1.0/([^/]+/[^/]+.*)")
    df["type"] = df.root_name.str.extract(r"/api/v1.0/([^/]+)/[^/]+.*")
    df["request"] = df["method"] + " " + df["route"]
    df["start_time"] = df.start_time.dt.tz_convert("Europe/Paris")
    df["span_thirdparty"] = df["span_name"].apply(extract_thirdparty)

    return df


def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)

    percentile_.__name__ = f"percentile_{n}"
    return percentile_


class Mask:
    def __init__(self, filter, name):
        self._filter = filter
        self._name = name

    def apply(self, df):
        return df[self._filter(df)]

    def name(self):
        return self._name

    def __and__(self, other):
        return Mask(
            lambda x: self._filter(x) & other._filter(x), f"{self._name}, {other._name}"
        )

    def __or__(self, other):
        return Mask(
            lambda x: self._filter(x) | other._filter(x),
            f"({self._name} or {other._name})",
        )


def add_time_type(df, incident_range, baseline_range):
    dg = pd.DataFrame(df.copy())
    start_time = df.reset_index().start_time
    incident = (start_time >= incident_range[0]) & (start_time <= incident_range[1])
    baseline = (start_time >= baseline_range[0]) & (start_time <= baseline_range[1])
    time_type = np.where(incident, "incident", np.where(baseline, "baseline", "none"))
    dg["time_type"] = time_type
    return dg


def plot_durations(df, mask, x="root_start_time", sample_rate="5min", color=None, display_trace_rest=False,
                   normalize=False, y="duration_ms", **kwargs):
    title = f"Latencies for: {mask.name()}"
    if color is not None:
        title += f", grouped by {color} contributions"
    df = mask.apply(df)
    df = df.set_index(x)

    if normalize:
        title = title + ", normalized"

    def get_trace_duration(trace_spans):
        root_id = get_root_span_id(trace_spans)
        duration = trace_spans[trace_spans.span_id == root_id].duration_ms
        return duration

    def compute_timestep(x):
        if color is not None:
            sums = x.groupby(color)[y].sum()
            if display_trace_rest:
                traces_durations = x.groupby("trace_id").duration_ms.max()
                rest = traces_durations.sum(axis=None) - sums.sum(axis=None)
                sums["rest"] = rest
        else:
            sums = x[y].sum(axis=None)

        num_traces = x.trace_id.nunique()

        means = sums / num_traces

        out = means
        if normalize:
            out = means / means.sum(axis=None)

        return out

    df = df.groupby(pd.Grouper(freq=sample_rate)).apply(compute_timestep)
    if isinstance(df, pd.DataFrame):
        # sometimes the result is a dataframe... thanks pandas, we turn it into a 
        # Series again
        df = df.stack()
    df = df.rename(y).to_frame()

    fig = px.bar(
        df.reset_index(),
        x=x,
        y=y,
        title=title,
        color=color,
        **kwargs,
    )

    if auto_show_plots:
        fig.show()


def plot_category_repartition(df, mask):
    title = f"Number of {mask.name()}"
    nb_spans = mask.apply(df).groupby("time_type").size().rename("number")

    fig = px.pie(
        nb_spans.reset_index(),
        names="time_type",
        values="number",
        title=title,
        category_orders={"time_type": ["incident", "baseline"]},
        color_discrete_sequence=["red", "green"],
        hole=0.4,
    )

    if auto_show_plots:
        fig.show()


def latencies_distributions_by(df, group_by, mask=None, **kwargs):
    title = f"Latencies by {group_by}"
    if mask is not None:
        title += f" for: {mask.name()}"
        df = mask.apply(df)

    order = (
        df.groupby(group_by)
        .duration_ms.median()
        .sort_values(ascending=False)
        .index.to_list()
    )
    fig = px.box(
        df,
        y=group_by,
        x="duration_ms",
        category_orders={group_by: order},
        title=title,
        orientation="h",
        height=max(len(order) * 70, 500),
        **kwargs,
    )

    if auto_show_plots:
        fig.show()


def get_root_span_id(df):
    mask = ~df.parent_span_id.isin(df.span_id)
    d = df[mask]
    if len(d) != 1:
        raise RuntimeError("get_root_id expects a single root to be found")
    if "span_id" in d.columns:
        return d.span_id[0]
    return d.index.get_level_values("span_id")


def compute_proper_durations_by_field(group, field, duration_column, depth_column):
    def rec_compute(span_id, level):
        data.loc[span_id, "span_depth"] = level
        children_ids = data[data.parent_span_id == span_id].index

        for child_id in children_ids:
            rec_compute(child_id, level + 1)

        exp_children = data.loc[children_ids, "explained"].sum()
        data.loc[span_id, "explained_children"] = exp_children
        data.loc[span_id, "proper"] = data.loc[span_id, "duration_ms"] - exp_children

        field_values = data.loc[span_id, field]
        data.loc[span_id, "explained"] = np.where(
            pd.notna(field_values), data.loc[span_id, "duration_ms"], exp_children
        )

    data = group.copy().set_index("span_id", drop=False)
    data.loc[:, "span_depth"] = np.nan
    data.loc[:, "explained"] = np.nan
    data.loc[:, "proper"] = np.nan
    data.loc[:, "explained_children"] = np.nan
    rec_compute(get_root_span_id(data), 0)

    data.index = group.index
    data = data.rename(columns={"proper": duration_column, "span_depth": depth_column})
    data = data[[duration_column, depth_column]]

    out = pd.concat([group, data], axis=1)
    return out


def latencies_distributions_and_contributions(df, group_by, mask=None, color=None, bars=5, **kwargs):
    low_percentile, up_percentile = sorted([bars, 100 - bars])

    title = f"Latencies contributions by {group_by}"
    if mask is not None:
        title += f", {mask.name()}"
        df = mask.apply(df)
    title += f", bars={low_percentile}/{up_percentile}"

    groupby = [group_by]
    if color is not None:
        groupby.append(color)

        # Default kwargs
    kw = {
        "log_y": True
    }
    kw.update(kwargs)

    dg = df.groupby(groupby).duration_ms.aggregate(
        ["mean", "count", "max", "sum", percentile(low_percentile), percentile(up_percentile)]).reset_index()
    fig = px.scatter(
        dg, y="sum", x="mean",
        error_x_minus=f"percentile_{low_percentile}",
        error_x=f"percentile_{up_percentile}",
        hover_name=group_by,
        color=color,
        title=title,
        **kw
    )

    if auto_show_plots:
        fig.show()


def applyParallel(dfGrouped, func, *args, **kwargs):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(func)(group, *args, **kwargs) for name, group in dfGrouped)
    return pd.concat(retLst)


def get_extended_dataset(df, dump_path_extended):
    if dump_path_extended.exists():
        print("Reading the proper durations from the disk")
        frameproper = pd.read_parquet(dump_path_extended)
    else:
        print("Computing the proper durations")
        frameproper = applyParallel(df.groupby("trace_id"),
                                    compute_proper_durations_by_field,
                                    "span_thirdparty",
                                    "proper_ms",
                                    "depth"
                                    )
        frameproper.to_parquet(dump_path_extended, compression="gzip")
    return frameproper
