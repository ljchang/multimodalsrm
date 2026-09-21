"""Conditional physical-density MAP coordinates for one new participant."""

import numpy as np

from ..data import readonly_array
from ._backend import runtime
from .problem import BayesianProblem


class CalibrationProblem(BayesianProblem):
    """Reuse scalar transforms/search with only the new participant free.

    The likelihood integrates the group-conditioned shared GP. Original group
    parameters are embedded by name at every evaluation, including boundary
    filter values; they never pass through an optimizer transform.
    """

    def __init__(self, joint, group, subject):
        self.joint = joint
        self.features = joint.features
        self.systems = joint.systems
        self.keys = [k for k in joint.keys if k[0] == subject]
        self.groups = [g for g in joint.groups if g[0] == subject]
        self.active = np.array(
            [
                i
                for i, n in enumerate(joint.names)
                if n[0] in ("loading", "offset", "noise") and n[1] == subject
            ]
        )
        self.names = [joint.names[i] for i in self.active]
        self.indices = {n: i for i, n in enumerate(self.names)}
        self.parameter_priors = [joint.parameter_priors[i] for i in self.active]
        self._prepare_parameter_priors()
        self.initial = joint.initial[self.active].copy()
        reference = joint.initial.copy()
        for name, value in zip(group.parameter_names_, group.map_parameters_):
            reference[joint.indices[name]] = value
        self.reference = readonly_array(reference)
        self.group_nll = float(group.problem_.nll(group.map_parameters_))
        # Generic Newton refinement works on the free vector. Full-model exact
        # variance profiling is intentionally unavailable for this subproblem.
        self.linear_algebra = "conditional"
        jax, _, _, _ = runtime()
        self._vg = jax.jit(jax.value_and_grad(self.objective))

    def expand(self, x):
        _, jnp, _, _ = runtime()
        return jnp.asarray(self.reference).at[self.active].set(x)

    def nll(self, x):
        return self.joint.nll(self.expand(x)) - self.group_nll
