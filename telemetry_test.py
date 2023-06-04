import pandas as pd
import numpy as np
import pytest 

from . import telemetry as t


@pytest.fixture(scope="module", autouse=True)
def dont_show_figs():
    """We do not want to show the figures in test mode"""
    t.auto_show_plots = False

def test_compute_proper():
    df = pd.DataFrame([
        {
            "groupfield": np.nan,
            "duration_ms": 32,
            "span_id": "A",
        },
        {
            "groupfield": "datastore",
            "duration_ms": 14,
            "span_id": "B",
            "parent_span_id": "A",
        },
        {
            "groupfield": np.nan,
            "duration_ms": 13   ,
            "span_id": "C",
            "parent_span_id": "A",
        },
        {
            "groupfield": "cache",
            "duration_ms": 6,
            "span_id": "D",
            "parent_span_id": "B",
        },
        {
            "groupfield": "google",
            "duration_ms": 4,
            "span_id": "E",
            "parent_span_id": "B",
        },
        {
            "groupfield": "datastore",
            "duration_ms": 6,
            "span_id": "F",
            "parent_span_id": "C",
        },
        {
            "groupfield": "cache",
            "duration_ms": 2,
            "span_id": "G",
            "parent_span_id": "F",
        },
        {
            "groupfield": "google",
            "duration_ms": 2,
            "span_id": "H",
            "parent_span_id": "F",
        },
        {
            "groupfield": "other",
            "duration_ms": 3,
            "span_id": "I",
            "parent_span_id": "C",
        },
    ])
    
    out = t.compute_proper_durations_by_field(df, "groupfield", "myproper_duration", "myspan_depth")
    
    assert out.myspan_depth.tolist() == [0, 1, 1, 2, 2, 2, 3, 3, 2]
    assert out.myproper_duration.tolist() == [9, 4, 4, 6, 4, 2, 2, 2, 3]
    


def test_plot_durations():
    df = pd.DataFrame([
        {
            "groupfield": np.nan,
            "start_time": "2023-2-1T00:00",
            "duration_ms": 32,
            "span_id": "A",
             
        },
        {
            "groupfield": "datastore",
            "start_time": "2023-2-1T00:06",
            "duration_ms": 14,
            "span_id": "B",
            "parent_span_id": "A",
        },
        {
            "groupfield": np.nan,
            "duration_ms": 13   ,
            "start_time": "2023-2-1T00:11",
            "span_id": "C",
            "parent_span_id": "A",
        },  
    ])
    df["start_time"] = pd.to_datetime(df["start_time"]) 

    mask = t.Mask(lambda x: pd.Series(True, index=x.index), "ALL")
    t.plot_durations(df, mask)


def test_plot_durations_colored():
    df = pd.read_json("real_test_data/extended.json", orient="table")
    mask = t.Mask(lambda x: pd.Series(True, index=x.index), "ALL")
    t.plot_durations(df, mask, sample_rate="1H", color="method")
    

def test_plot_durations_extended():
    df = pd.read_json("real_test_data/extended.json", orient="table")
    mask = t.Mask(lambda x: pd.Series(True, index=x.index), "ALL")
    t.plot_durations(df, mask, sample_rate="1H", color="span_thirdparty")
