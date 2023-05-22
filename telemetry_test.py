import pandas as pd
import numpy as np

from . import telemetry as t

def test_one():
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
    


