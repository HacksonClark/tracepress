"""TracePress Failure-Attribution Benchmark & Research Suite."""

from . import datasets, pipelines, tune
from .datasets import (
    FailureTraceAdapter,
    StepRecord,
    TraceRecord,
    attach_context,
    load_agentrx,
    load_longrca,
    load_trail,
    load_whowhen,
    score_agentrx,
    score_longrca,
    score_trail,
    score_whowhen,
)
from .pipelines import all_at_once, fold, map_reduce, staged, trail_pipeline
from .tune import StagedAttributionModule, attribution_metric, compile_staged

__all__ = [
    "FailureTraceAdapter",
    "StagedAttributionModule",
    "StepRecord",
    "TraceRecord",
    "all_at_once",
    "attach_context",
    "attribution_metric",
    "compile_staged",
    "datasets",
    "fold",
    "load_agentrx",
    "load_longrca",
    "load_trail",
    "load_whowhen",
    "map_reduce",
    "pipelines",
    "score_agentrx",
    "score_longrca",
    "score_trail",
    "score_whowhen",
    "staged",
    "trail_pipeline",
    "tune",
]
