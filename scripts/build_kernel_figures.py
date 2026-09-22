"""Rebuild documentation illustrations from the implemented response families.

Requires the package's plots extra. No research data or model fitting is used.
"""

from pathlib import Path
from tempfile import gettempdir

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from multimodalsrm import BachSCR, BatemanSCR, DoubleGamma, Gamma, Gaussian, SampledKernel

OUT = Path(__file__).resolve().parents[1] / "docs/assets/figures"
COLORS = ["#087e8b", "#9963a5", "#bd6329", "#3377a8", "#ae4766", "#61842b"]
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#263e4b",
        "text.color": "#263e4b",
        "axes.prop_cycle": plt.cycler(color=COLORS),
        "svg.fonttype": "none",
        "svg.hashsalt": "multimodalsrm-kernel-guide",
        "figure.facecolor": "white",
    }
)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", metadata={"Date": None}, bbox_inches="tight")
    fig.savefig(Path(gettempdir()) / f"{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def curve(ax, kernel, label=None, **kwargs):
    t = np.linspace(*kernel.support, 2001)
    ax.plot(t, kernel(t), label=label, **kwargs)
    ax.axhline(0, color="#9ba9af", lw=0.6, zorder=0)
    ax.set_xlabel("Response lag u (s)")
    ax.set_ylabel("h(u), unit energy")


def families():
    fig, axes = plt.subplots(4, 2, figsize=(8.4, 12.8), layout="constrained")
    ax = axes.flat[0]
    ax.annotate("", (0, 1), (0, 0), arrowprops={"arrowstyle": "->", "color": COLORS[0], "lw": 2})
    ax.axhline(0, color="#9ba9af", lw=0.6)
    ax.set(
        xlim=(-2, 2), ylim=(-0.05, 1.2), title="Identity()", xlabel="Response lag u (s)", yticks=[]
    )
    ax.text(0.15, 0.6, "Unit mass\n(symbolic)", fontsize=10)
    examples = [
        (Gaussian(), "Gaussian(width=1, lag=0)"),
        (Gamma(), "Gamma(shape=3, scale=1)"),
        (DoubleGamma(), "DoubleGamma()"),
        (BachSCR(), "BachSCR()"),
        (BatemanSCR(), "BatemanSCR(rise=0.7, decay=3)"),
        (
            SampledKernel(np.array([0.0, 1.0, 3.0, 5.0, 8.0]), np.array([0.0, 0.5, 1.0, 0.3, 0.0])),
            "SampledKernel (custom example)",
        ),
    ]
    for i, (kernel, title) in enumerate(examples, 1):
        ax = axes.flat[i]
        curve(ax, kernel, color=COLORS[(i - 1) % len(COLORS)])
        ax.set_title(title)
        if isinstance(kernel, (BachSCR, BatemanSCR)):
            ax.set_xlim(-1, 35)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0.05,
        0.75,
        "Same normalization,\ndifferent shapes.\n\nSCR support extends to 90 s.\nSampled curve is fixed.\nIdentity is an analytic impulse.",
        va="top",
        fontsize=11,
        linespacing=1.6,
    )
    save(fig, "kernel-families")


def drive(t):
    return np.exp(-0.5 * ((t - 10) / 0.55) ** 2) + 0.65 * np.exp(-0.5 * ((t - 28) / 1.4) ** 2)


def convolution():
    t = np.linspace(-5, 65, 1401)
    kernels = [Gaussian(width=1, lag=2), Gamma(), DoubleGamma(), BachSCR(), BatemanSCR()]
    labels = ["Gaussian(width=1, lag=2)", "Gamma()", "DoubleGamma()", "BachSCR()", "BatemanSCR()"]
    fig, axes = plt.subplots(
        6, 1, figsize=(8.4, 10), sharex=True, sharey=True, layout="constrained"
    )
    axes[0].plot(t, drive(t), color="#263e4b")
    axes[0].set_title("Shared synthetic drive = Identity output", loc="left")
    for ax, kernel, label, color in zip(axes[1:], kernels, labels, COLORS):
        u = np.linspace(*kernel.support, 3601)
        y = np.trapezoid(kernel(u)[:, None] * drive(t[None, :] - u[:, None]), u, axis=0)
        assert np.isfinite(y).all()
        ax.plot(t, y, color=color)
        ax.set_title(label, loc="left")
    for ax in axes:
        ax.axhline(0, color="#9ba9af", lw=0.6)
        for event in [10, 28]:
            ax.axvline(event, ls=":", color="#9ba9af", lw=1)
        ax.set_ylabel("Amplitude")
        ax.set_xlim(-2, 65)
    axes[-1].set_xlabel("Observation time t (s)")
    save(fig, "kernel-convolution")


def parameters():
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7), layout="constrained")
    for lag in [0, 2, 4]:
        curve(axes[0, 0], Gaussian(width=1, lag=lag), f"lag={lag}")
    for width in [0.5, 1, 2]:
        curve(axes[0, 1], Gaussian(width=width, lag=2), f"width={width}")
    for shape in [1, 3, 6]:
        curve(axes[1, 0], Gamma(shape=shape), f"shape={shape}")
    for decay in [1.5, 3, 6]:
        curve(axes[1, 1], BatemanSCR(decay=decay), f"decay={decay}")
    for ax, title in zip(
        axes.flat,
        [
            "Lag shifts the Gaussian",
            "Width spreads the Gaussian",
            "Gamma shape moves the peak",
            "Bateman decay extends the tail",
        ],
    ):
        ax.set_title(title)
        ax.legend(frameon=False)
    axes[1, 1].set_xlim(-1, 30)
    save(fig, "kernel-parameters")


def latent():
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 10), layout="constrained")
    delta = np.linspace(0, 15, 400)
    t = np.linspace(0, 30, 241)
    noise = np.random.default_rng(42).normal(size=len(t))
    for scale, color in zip([1.0, 3.0, 8.0], COLORS):
        r = np.sqrt(3) * delta / scale
        axes[0].plot(delta, (1 + r) * np.exp(-r), color=color, label=f"length_scale={scale:g}")
        rmat = np.sqrt(3) * np.abs(t[:, None] - t[None, :]) / scale
        cov = (1 + rmat) * np.exp(-rmat)
        sample = np.linalg.cholesky(cov + 1e-10 * np.eye(len(t))) @ noise
        axes[1].plot(t, sample, color=color, lw=1.2)
    axes[0].set(
        title="Latent covariance: Matérn-3/2",
        xlabel="Time separation (s)",
        ylabel="Prior correlation",
    )
    axes[0].legend(frameon=False)
    axes[1].set(title="Examples from those latent priors", xlabel="Time (s)", ylabel="Latent value")
    for width in [0.5, 1, 2]:
        curve(axes[2], Gaussian(width=width), f"width={width}")
    axes[2].set_title("A separate choice: response width")
    axes[2].legend(frameon=False)
    save(fig, "latent-timescale")


def support():
    fig, ax = plt.subplots(figsize=(11, 3.6), layout="constrained")
    cases = [
        ("Identity: [0, 0]", 0, 120),
        ("Gaussian envelope: [−6, 10]", 10, 114),
        ("SCR envelope: [0, 90]", 90, 120),
    ]
    for i, (label, lo, hi) in enumerate(cases):
        ax.barh(i, 120, left=0, height=0.42, color="#e5e9ec")
        ax.barh(i, hi - lo, left=lo, height=0.42, color=COLORS[i])
        ax.text(
            (lo + hi) / 2,
            i,
            f"usable: {lo}–{hi} s",
            va="center",
            ha="center",
            color="white",
            weight="bold",
        )
    ax.set_yticks(range(3), [x[0] for x in cases])
    ax.set(
        xlim=(0, 120),
        xlabel="Observation time within the run (s)",
        title="Whole response support must fit inside the run",
    )
    ax.invert_yaxis()
    save(fig, "kernel-support")


if __name__ == "__main__":
    families()
    convolution()
    parameters()
    latent()
    support()
    print(f"Wrote five kernel illustrations to {OUT}")
