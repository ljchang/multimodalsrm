"""Native observation preparation for Bayesian response families.

Covariance evaluation belongs to BayesianProblem. The separate continuous
reference estimator retains its original family restrictions.
"""

from .._observation_preparation import NativeObservationPreparation
from ..kernels import BatemanSCR, DoubleGamma, Gamma, Gaussian, Identity


class BayesianObservationAdapter(NativeObservationPreparation):
    _response_types = (Identity, Gaussian, Gamma, DoubleGamma, BatemanSCR)


def capacity_unit(linear_algebra, *, noise_prior=None, run_baseline_sd=None, noise_timescales=None):
    """Select what ``max_observations`` bounds for one run of this configuration.

    Dense algebra, spectral algebra, run baselines and correlated noise keep the
    scalar-observation count. Grouped algebra factorizes exact modality/time
    nodes times factors. Grouped state-space filtering, used when the noise
    prior excludes zero, advances once per node.
    """
    if run_baseline_sd or noise_timescales:
        return "scalars"
    if linear_algebra == "grouped":
        return "functionals"
    if (
        linear_algebra == "state_space"
        and noise_prior is not None
        and (noise_prior.family == "lognormal" or noise_prior.lower > 0)
    ):
        return "nodes"
    return "scalars"
