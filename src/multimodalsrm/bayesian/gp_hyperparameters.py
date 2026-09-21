"""Opt-in shared timescale conventions, preserving fixed-GP defaults."""

import numpy as np

from .priors import Prior

LENGTH_SCALE = ("gp", "length_scale")


def length_scale_settings(value):
    """A number is fixed; a bounded physical Prior requests estimation.

    The prior median is only a representative adapter value. Likelihoods and
    queries use the physical coordinate, never that representative constant.
    """
    if isinstance(value, Prior):
        validate_length_scale_prior(value)
        return float(value.ppf(0.5)), value
    return value, None


def validate_length_scale_prior(prior):
    if prior is not None and (
        not isinstance(prior, Prior)
        or not np.isfinite([prior.lower, prior.upper]).all()
        or prior.lower <= 0
    ):
        raise ValueError("length_scale prior must have positive finite bounds")


def validate_length_scale_scope(
    prior,
    *,
    linear_algebra,
    response_quadrature_order,
    run_baseline_sd,
    noise_timescales,
    responses=None,
):
    validate_length_scale_prior(prior)
    if prior is not None and (
        linear_algebra not in ("dense", "grouped") or run_baseline_sd or noise_timescales
    ):
        raise ValueError(
            "learned length_scale requires dense/grouped inference "
            "with independent noise and no run baselines"
        )
    if prior is not None:
        from .response_scope import validate_posterior_responses

        validate_posterior_responses(responses, response_quadrature_order)
