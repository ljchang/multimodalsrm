"""Experimental Bayesian SRM; optional runtime is loaded only when fitting."""

from .blocks import ParameterSubspace
from .calibration import ParticipantCalibration
from .fitting import SamplerConfig, SearchConfig
from .model import BayesianMultimodalSRM
from .priors import BayesianPriors, Prior, lag_priors
from .problem import BayesianProblem
from .results import GaussianMixtureSeries, TrajectorySamples
from .spectral import SpectralConfig

__all__ = [
    "Prior",
    "BayesianPriors",
    "lag_priors",
    "BayesianProblem",
    "ParameterSubspace",
    "BayesianMultimodalSRM",
    "SearchConfig",
    "SamplerConfig",
    "GaussianMixtureSeries",
    "TrajectorySamples",
    "SpectralConfig",
    "ParticipantCalibration",
]
