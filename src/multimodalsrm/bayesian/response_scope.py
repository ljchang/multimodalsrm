"""Shared response admission and numerical-state checks for posterior workflows."""

from ..kernels import BachSCR, BatemanSCR, DoubleGamma, Gamma, Gaussian, Identity


def validate_posterior_responses(responses, order):
    """Preserve analytic defaults; other admitted families need explicit quadrature."""
    families = [type(r.initial_kernel()) for r in (responses or {}).values()]
    if any(
        f not in (Identity, Gaussian, Gamma, DoubleGamma, BachSCR, BatemanSCR) for f in families
    ):
        raise ValueError(
            "posterior responses support Identity/Gaussian/Gamma/DoubleGamma/BachSCR/BatemanSCR; "
            "sampled responses remain unsupported"
        )
    if order is None and any(f in (Gamma, DoubleGamma, BachSCR, BatemanSCR) for f in families):
        raise ValueError(
            "Gamma/DoubleGamma/BachSCR/BatemanSCR posterior responses require explicit "
            "response_quadrature_order"
        )
    if order is not None:
        from .response_quadrature import validate_order

        validate_order(order)


def quadrature_state(problem):
    """Expose prepared rules for comparison with an independently rebuilt target."""
    from .response_quadrature import ResponseQuadrature

    q = problem.response_quadrature
    if q is None:
        return None
    if type(q) is not ResponseQuadrature:
        raise ValueError("posterior response quadrature has an unsupported implementation")
    return vars(q)


def validate_posterior_response_target(problem):
    """Reject stale/mutated quadrature rules, domains and parameter indexing."""
    from . import _archive
    from .response_quadrature import ResponseQuadrature

    validate_posterior_responses(problem.responses, problem.response_quadrature_order)
    expected = None
    if problem.response_quadrature_order is not None:
        expected = vars(
            ResponseQuadrature(
                problem.responses,
                problem.indices,
                problem.length_scale,
                problem.response_quadrature_order,
                minimum_length_scale=(
                    None if problem.length_scale_prior is None else problem.length_scale_prior.lower
                ),
            )
        )
    if not _archive.same(quadrature_state(problem), expected):
        raise ValueError("posterior response quadrature differs from the declared target")
