"""Descriptive response inspection; participant spread is not uncertainty."""

from dataclasses import dataclass

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .data import readonly_array, validate_times
from .kernels import Identity


@dataclass(frozen=True)
class KernelEstimate:
    """Fitted response and optional sampled curve, with the scale convention.

    Identity has ``impulse_mass=1`` and ``values=None`` even when query times
    are supplied: a Dirac impulse is not an ordinary finite-valued density.
    Nonidentity values and group differences are time-by-one arrays.
    """

    family: str
    parameters: dict
    support: tuple
    normalization: str
    reference: str | None
    fixed: dict
    learned: tuple
    pooling: str
    level: str
    subject: object
    modality: object
    times: np.ndarray | None
    values: np.ndarray | None
    difference_from_group: np.ndarray | None
    impulse_mass: float | None
    metadata: dict

    @property
    def valid(self):
        """Sampled curve validity; an analytic impulse has no sampled values."""
        if self.values is None:
            return None
        return readonly_array(np.isfinite(self.values), dtype=bool)


def kernel(model, modality, *, subject=None, level="subject", times=None):
    """Inspect an effective subject or group kernel without changing the fit."""
    check_is_fitted(model, ["subject_kernels_", "group_kernels_"])
    if level not in ("subject", "group"):
        raise ValueError("level must be 'subject' or 'group'.")
    if modality not in model.responses_:
        raise ValueError(f"Unknown response modality {modality!r}.")
    response = model.responses_[modality]
    if level == "group":
        if subject is not None:
            raise ValueError("A group query cannot also name a subject.")
        if modality not in model.group_kernels_:
            raise ValueError(
                "This modality has no fitted group kernel (independent or unfitted response)."
            )
        effective = model.group_kernels_[modality]
    else:
        if subject not in model.subject_kernels_ or modality not in model.subject_kernels_[subject]:
            raise ValueError(
                "A fitted participant/modality is required for a subject kernel query."
            )
        effective = model.subject_kernels_[subject][modality]
    query = None if times is None else validate_times(times, allow_empty=True)
    impulse = isinstance(effective, Identity)
    values = difference = None
    if query is not None and not impulse:
        values = readonly_array(effective.evaluate(query)[:, None])
        group = model.group_kernels_.get(modality)
        if level == "subject" and group is not None:
            difference = readonly_array(values - group.evaluate(query)[:, None])
    fixed = {
        name: value
        for name, value in effective.parameters.items()
        if not response.estimate or name in response.fixed
    }
    metadata = {
        **effective.metadata,
        "response": response.metadata,
        "support_envelope": response.support_envelope(),
        "spread_interpretation": "descriptive participant differences; not a confidence interval",
        "timing_interpretation": "additional reference-shape lag; conditional relative timing",
    }
    return KernelEstimate(
        type(effective).__name__,
        dict(effective.parameters),
        effective.support,
        effective.normalization,
        effective.reference,
        fixed,
        tuple(name for name in effective.parameters if name not in fixed),
        response.pooling,
        level,
        subject,
        modality,
        query,
        values,
        difference,
        1.0 if impulse else None,
        metadata,
    )


def plot_kernels(model, modality, *, show_subjects=True, ax=None):
    """Plot effective curves on a common seconds axis; return matplotlib axes."""
    check_is_fitted(model, ["subject_kernels_", "group_kernels_"])
    import matplotlib.pyplot as plt

    curves = [
        (subject, values[modality])
        for subject, values in model.subject_kernels_.items()
        if modality in values
    ]
    group = model.group_kernels_.get(modality)
    if not curves:
        raise ValueError(f"No fitted kernels for modality {modality!r}.")
    all_kernels = [value for _, value in curves] + ([group] if group is not None else [])
    low, high = (
        min(k.support[0] for k in all_kernels),
        max(k.support[1] for k in all_kernels),
    )
    if low == high:
        low, high = low - 1.0, high + 1.0
    times = np.linspace(low, high, 601)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    def draw(curve, label, color, alpha, width):
        if isinstance(curve, Identity):
            ax.vlines(
                0,
                0,
                1,
                label=label + " (unit-mass impulse)",
                color=color,
                alpha=alpha,
                linewidth=width,
            )
            ax.plot([0], [1], marker="o", color=color, alpha=alpha)
        else:
            ax.plot(
                times,
                curve.evaluate(times),
                label=label,
                color=color,
                alpha=alpha,
                linewidth=width,
            )

    if show_subjects:
        for j, (subject, curve) in enumerate(curves):
            draw(curve, str(subject), f"C{j % 10}", 0.6, 1.2)
    if group is not None:
        draw(group, "group", "black", 1.0, 2.4)
    elif not show_subjects:
        raise ValueError("Independent pooling has no group curve; set show_subjects=True.")
    ax.set(
        xlabel="Lag (seconds)",
        ylabel="L2-normalized response / identity impulse mass",
        title=f"{modality}: fitted response kernels",
    )
    ax.legend(frameon=False)
    return ax


def factor_report(model):
    """How much of each modality every fitted factor explains.

    Loadings alone cannot be compared across modalities: a response kernel has
    its own gain, so the same fitted signal needs different loading sizes under
    different responses. This report uses the fitted signal instead. For one
    participant, modality and feature the fit is ``sum_k w_k * u_k(t)``, with
    ``u_k`` the latent factor passed through the response. Observations are
    standardized per feature, so ``mean_t (w_k u_k(t))**2`` over observed
    samples is the share of that feature's variance carried by factor ``k``
    alone. Values are averaged over features, then over participants.

    Returns a dictionary:

    ``explained_variance[modality]``
        One value per factor, in the fitted factor coordinates.
    ``total_explained_variance[modality]``
        Variance of the complete fitted signal. It differs from the sum over
        factors when filtered factors are correlated.
    ``modality_share[modality]``
        Each factor's explained variance in this modality divided by its sum
        over modalities. A factor near 1 in one modality is specific to it.
    ``participants[subject][modality]``
        The same per-factor values for one participant.
    ``specificity[modality]``
        Only with ``latent_pooling="shared"``. Factors are identified up to
        rotation, so the table above depends on the fitted coordinates.
        ``most`` and ``least`` are the largest and smallest share of a factor's
        explained variance that this modality can take under any rotation, and
        ``shares`` lists all of them in decreasing order. ``most`` near 1 with
        ``least`` near 0 means the fit contains both a factor specific to this
        modality and one absent from it. They are exact when the filtered
        factors are uncorrelated with equal variance within each modality, and
        an approximation otherwise.

    These are descriptive training-fit quantities, not held-out prediction.
    """
    check_is_fitted(model, ["loadings_", "observation_blocks_", "_training_latent_arrays_"])
    latents, loadings = model._training_latent_arrays_, model.loadings_
    features = model.features
    modalities = [m for m in model.responses_ if any(m in mods for mods in loadings.values())]
    shared = model.latent_pooling == "shared"
    unwhiten = None
    if shared:
        stacked = np.concatenate(list(latents[model.subjects_[0]].values()))
        covariance = stacked.T @ stacked / len(stacked)
        values, vectors = np.linalg.eigh(covariance)
        if values[0] > 1e-12 * values[-1]:
            # Symmetric square root: latent = whitened @ root, rotations orthogonal.
            unwhiten = (vectors * np.sqrt(values)) @ vectors.T
    # (subject, modality) -> [per-factor sum of squares, samples per feature, complete sum of squares]
    own = {}
    quadratic = {m: np.zeros((features, features)) for m in modalities}
    for block in model.observation_blocks_:
        key = (block.subject, block.modality)
        weights = loadings[block.subject][block.modality]
        observed = block.mask & block.valid[:, None]
        n = observed.sum(0)
        usable = n > 0
        if not usable.any():
            continue
        filtered = np.asarray(block.H @ latents[block.subject][block.run])
        # Per-feature second moments of the filtered factors over observed samples.
        moments = np.einsum("tf,tk,tl->fkl", observed.astype(float), filtered, filtered)
        entry = own.setdefault(key, [np.zeros(features), np.zeros(len(weights)), 0.0])
        entry[0] += np.einsum("fk,fkk->k", weights**2, moments)
        entry[1] += n
        entry[2] += float(np.einsum("fk,fl,fkl->", weights, weights, moments))
        if unwhiten is not None:
            white = filtered @ np.linalg.inv(unwhiten)
            second = np.einsum("tf,tk,tl->fkl", observed.astype(float), white, white)
            scaled = weights @ unwhiten
            for f in np.flatnonzero(usable):
                vals, vecs = np.linalg.eigh(second[f])
                root = (vecs * np.sqrt(np.maximum(vals, 0.0))) @ vecs.T
                direction = root @ scaled[f]
                quadratic[block.modality] += np.outer(direction, direction)
    participants, by_modality, totals = {}, {m: [] for m in modalities}, {m: [] for m in modalities}
    samples = {m: 0.0 for m in modalities}
    for (subject, modality), (per_factor, n, complete) in own.items():
        # A feature's variance share is its summed squares over its sample count;
        # pooling numerator and denominator over features weights them by samples.
        denominator = float(n.sum())
        value = per_factor / denominator
        participants.setdefault(subject, {})[modality] = value.tolist()
        by_modality[modality].append(value)
        totals[modality].append(complete / denominator)
        samples[modality] += denominator
    explained = {m: np.mean(by_modality[m], axis=0) for m in modalities if by_modality[m]}
    summed = sum(explained.values())
    report = {
        "factors": features,
        "modalities": list(explained),
        "explained_variance": {m: v.tolist() for m, v in explained.items()},
        "total_explained_variance": {m: float(np.mean(totals[m])) for m in explained},
        "modality_share": {
            m: np.divide(v, summed, out=np.zeros(features), where=summed > 0).tolist()
            for m, v in explained.items()
        },
        "participants": participants,
        "coordinates": "fitted factors; rotation is not identified without further constraints",
        "specificity": None,
    }
    if unwhiten is not None and len(explained) > 1:
        from scipy.linalg import eigh

        normalized = {m: quadratic[m] / samples[m] for m in explained}
        everything = sum(normalized.values())
        scale = np.trace(everything) / features
        if scale > 0:
            regular = everything + 1e-10 * scale * np.eye(features)
            report["specificity"] = {}
            for m in explained:
                shares = np.clip(eigh(normalized[m], regular, eigvals_only=True)[::-1], 0.0, 1.0)
                report["specificity"][m] = {
                    "most": float(shares[0]),
                    "least": float(shares[-1]),
                    "shares": shares.tolist(),
                }
    return report
