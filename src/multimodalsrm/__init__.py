"""Native-time shared response models for multimodal observations."""

from ._version import __version__
from .alignment import temporal_isc, time_segment_matching
from .data import TimeSeries
from .estimator import MultimodalSRM
from .inspection import KernelEstimate
from .kernels import (
    BachSCR,
    DoubleGamma,
    Gamma,
    Gaussian,
    Identity,
    KernelPrior,
    Normal,
    Response,
    SampledKernel,
)
from .posterior_results import PosteriorSeriesResult
from .results import SeriesResult
from .selection import (
    ModelComparisonResult,
    ModelSelectionResult,
    NestedCrossValidationResult,
    compare_models,
    nested_cross_validate,
    select_model,
)
from .validation import CrossValidationResult, LeaveOneRunOut, cross_validate

__all__ = [
    "__version__",
    "TimeSeries",
    "temporal_isc",
    "time_segment_matching",
    "Identity",
    "Gaussian",
    "Gamma",
    "DoubleGamma",
    "BachSCR",
    "SampledKernel",
    "Response",
    "Normal",
    "KernelPrior",
    "SeriesResult",
    "PosteriorSeriesResult",
    "MultimodalSRM",
    "KernelEstimate",
    "LeaveOneRunOut",
    "cross_validate",
    "CrossValidationResult",
    "compare_models",
    "select_model",
    "nested_cross_validate",
    "ModelComparisonResult",
    "ModelSelectionResult",
    "NestedCrossValidationResult",
]
