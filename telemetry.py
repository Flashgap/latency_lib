import pandas as pd
import numpy as np
import plotly.express as px
import logging

query_str = """
SELECT 
    samples.trace_id as trace_id,
    span.span_id as span_id,
    span.name as span_name,
    parent.span_id as parent_span_id,
    parent.name as parent_name,
    root.name as root_name,
    root.start_time as root_start_time,
    STRING(json_extract(root.attributes, "$['http.method']")) as method,
    span.start_time as start_time,
    span.duration_nano/1000000 as duration_ms
FROM (
  SELECT trace_id FROM (
  SELECT DISTINCT(trace_id) FROM traces.spans s
  WHERE contains_substr(name, "/api/") AND s.start_time BETWEEN @start_time AND @end_time
  ) 
  WHERE RAND()<=@sample_rate
) samples
INNER JOIN (
  SELECT * FROM `traces.spans`
  WHERE contains_substr(name, "/api/")
) root
ON samples.trace_id = root.trace_id
INNER JOIN `traces.spans` span
ON root.trace_id = span.trace_id
LEFT JOIN (
  SELECT * FROM `traces.spans`
) parent
ON parent.span_id = span.parent_span_id
ORDER BY span.start_time

"""


def get_data(start_time, end_time, sample_rate=1.0):
    client = bigquery.Client(project="fruitsy-tutty")

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
    query_job = client.query(query_str, job_config=job_config)
    logging.getLogger().setLevel(logging.INFO)
    return query_job.to_dataframe()


def extract_thirdparty(s):
    tp = s.split(".")[0].lower()
    if tp in ("datastore", "cache"):
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


def plot_durations(df, mask, sample_rate="5min", color=None, normalize=False, **kwargs):
    title = f"Latencies for: {mask.name()}"
    if color is not None:
        title += f", grouped by {color} contributions"
    df = mask.apply(df).set_index("start_time")

    if normalize:
        title = title + ", normalized"

    def fff(x):
        if color is not None:
            sums = x.groupby(color).duration_ms.sum()
        else:
            sums = x.duration_ms.sum(axis=None)
        sums = sums / len(x)

        if normalize:
            sums = sums / sums.sum(axis=None)

        return sums

    df = df.groupby(pd.Grouper(freq=sample_rate)).apply(fff)
    if isinstance(df, pd.DataFrame):
        # sometimes if there is only one group, it will be considered as a column...
        # so we just undo this
        assert len(df.columns) == 1
        df = df.stack()
    df = df.rename("duration_ms").to_frame()

    px.bar(
        df.reset_index(),
        x="start_time",
        y="duration_ms",
        title=title,
        color=color,
        **kwargs,
    ).show()


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
    fig.show()


def get_root_id(df):
    mask = ~df.parent_span_id.isin(df.span_id)
    d = df[mask].index
    if len(d) != 1:
        raise RuntimeError("get_root_id expects a single root to be found")
    return d[0]

def compute_proper_durations_by_field(group, field, duration_column, depth_column):
    def rec_compute(span_id, level):
        data.loc[span_id, "span_depth"] = level
        children_ids = data[data.parent_span_id == span_id].index

        for child_id in children_ids:
            rec_compute(child_id, level+1)
        
        exp_children = data.loc[children_ids, "explained"].sum()
        data.loc[span_id, "explained_children"] = exp_children
        data.loc[span_id, "proper"] = data.loc[span_id, "duration_ms"] - exp_children

        field_values = data.loc[span_id, field]
        data.loc[span_id, "explained"] = np.where(pd.notna(field_values), data.loc[span_id, "duration_ms"], exp_children)
        

    data = group.copy().set_index("span_id", drop=False)
    data.loc[:, "span_depth"] = np.nan
    data.loc[:, "explained"] = np.nan
    data.loc[:, "proper"] = np.nan
    data.loc[:, "explained_children"] = np.nan
    rec_compute(get_root_id(data), 0)
    
    data.index = group.index
    data = data.rename(columns={
        "proper": duration_column,
        "span_depth": depth_column
    })
    data = data[[duration_column, depth_column]]

    out = pd.concat([group, data], axis=1)
    return out
