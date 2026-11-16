"""Dataset loaders, canonical records, and official scorers for failure attribution."""

from . import agentrx, download, longrca, trail, whowhen
from .agentrx import AGENTRX_CATEGORIES, load_agentrx, load_agentrx_traces, score_agentrx
from .common import (
    FailureTraceAdapter,
    StepRecord,
    TraceRecord,
    attach_context,
    bootstrap_ci,
    paired_bootstrap_diff,
)
from .longrca import load_longrca, load_longrca_traces, normalize_role, score_longrca
from .trail import TRAIL_CATEGORIES, flatten_trail_spans, load_trail, load_trail_traces, score_trail
from .whowhen import (
    load_whowhen,
    load_whowhen_traces,
    paired_diff_whowhen,
    parse_whowhen_predictions,
    score_whowhen,
    trivial_baselines,
    write_whowhen_predictions,
)

__all__ = [
    "AGENTRX_CATEGORIES",
    "FailureTraceAdapter",
    "StepRecord",
    "TRAIL_CATEGORIES",
    "TraceRecord",
    "agentrx",
    "attach_context",
    "bootstrap_ci",
    "download",
    "flatten_trail_spans",
    "load_agentrx",
    "load_agentrx_traces",
    "load_longrca",
    "load_longrca_traces",
    "load_trail",
    "load_trail_traces",
    "load_whowhen",
    "load_whowhen_traces",
    "longrca",
    "normalize_role",
    "paired_bootstrap_diff",
    "paired_diff_whowhen",
    "parse_whowhen_predictions",
    "score_agentrx",
    "score_longrca",
    "score_trail",
    "score_whowhen",
    "trail",
    "trivial_baselines",
    "whowhen",
    "write_whowhen_predictions",
]
