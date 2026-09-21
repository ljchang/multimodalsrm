"""Training sign and time conventions, without an identifiability claim."""

from ..kernels import Gaussian, Identity


def resolve_anchor(keys, anchor, training_systems):
    """Select a sign convention from training evidence, never held-out donors."""
    observed = {key for system in training_systems.values() for key in system.keys}
    if anchor is None:
        eligible = sorted(observed.intersection(keys))
        if not eligible:
            raise ValueError("anchor requires support-eligible training observations")
        return eligible[0]
    if not isinstance(anchor, (tuple, list)) or len(anchor) != 3:
        raise ValueError("anchor must name an observed feature")
    anchor = tuple(anchor)
    if anchor not in keys:
        raise ValueError("anchor must name an observed feature")
    if anchor not in observed:
        raise ValueError("anchor requires support-eligible training observations")
    return anchor


def reference_convention(
    adapter,
    keys,
    anchor,
    reference_modality,
    training_systems,
    *,
    linear_algebra="dense",
):
    responses = adapter.responses_
    requested_anchor = anchor
    anchor = resolve_anchor(keys, anchor, training_systems)
    observed = {key for system in training_systems.values() for key in system.keys}
    if reference_modality is None:
        # Preserve the original explicit Identity-anchor clock. A sign anchor
        # in another response family imposes no temporal restriction.
        if requested_anchor is not None and type(responses[anchor[1]].initial_kernel()) is Identity:
            reference, mode = anchor[1], "identity_anchor"
        else:
            fixed = sorted(
                m
                for m, response in responses.items()
                if "lag" not in response.free_parameters and any(key[1] == m for key in observed)
            )
            reference = fixed[0] if fixed else None
            mode = "automatic_fixed_lag" if fixed else "relative_lags"
    else:
        if not isinstance(reference_modality, str) or reference_modality not in responses:
            raise ValueError("reference_modality must name an observed modality")
        reference = reference_modality
        mode = "explicit_fixed_lag"
    response = responses[reference] if reference is not None else None
    if response is not None and "lag" in response.free_parameters:
        raise ValueError(
            "reference_modality requires a fixed lag: set Response.fixed['lag'] or estimate=False"
        )
    if reference is not None and not any(key[1] == reference for key in observed):
        raise ValueError("reference_modality requires support-eligible training observations")
    kernel = response.initial_kernel() if response is not None else None
    result = dict(
        mode=mode,
        positive_loading_anchor=list(anchor),
        reference_modality=reference,
        reference_family=type(kernel).__name__ if kernel is not None else None,
        reference_lag_seconds=float(getattr(kernel, "lag", 0.0)) if kernel is not None else None,
        reference_width=("learned" if "width" in response.free_parameters else "fixed")
        if response is not None
        else None,
        latent_mean=0.0,
        latent_variance=1.0,
        fixed_length_scale_seconds=float(adapter.length_scale),
        lag_coordinates="model_clock_seconds",
        relative_lag_definition="modality_lag_minus_reference_lag"
        if reference is not None
        else "modality_lag_minus_mean_lag",
        absolute_physiological_timing_established=False,
        identification="convention_only; data_connectivity_and_information_not_verified",
    )
    if reference is None:
        result["common_lag_offset"] = (
            "prior_and_spectral_boundary_dependent"
            if linear_algebra == "spectral"
            else "prior_defined_not_likelihood_identified"
        )
        result["latent_query_clock"] = "model_clock_not_mean_centered"
    if kernel is not None and type(kernel) not in (Identity, Gaussian):
        result["reference_width"] = "not_applicable"
        result["reference_shape_parameters_estimated"] = [
            p for p in response.free_parameters if p != "lag"
        ]
    return result
