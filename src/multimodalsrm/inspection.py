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
