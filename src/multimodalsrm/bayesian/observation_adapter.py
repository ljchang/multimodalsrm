"""Native observation preparation for Bayesian response families.

Covariance evaluation belongs to BayesianProblem. The separate continuous
reference estimator retains its original family restrictions.
"""

from .._observation_preparation import NativeObservationPreparation
from ..kernels import BachSCR, BatemanSCR, DoubleGamma, Gamma, Gaussian, Identity


class BayesianObservationAdapter(NativeObservationPreparation):
    _response_types = (Identity, Gaussian, Gamma, DoubleGamma, BachSCR, BatemanSCR)
