"""Native-time multimodal SRM with stationary response kernels."""

import copy
import os
import warnings
from collections.abc import Mapping

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state

from ._parallel import ordered_tasks, worker_count
from .data import normalize_data
from .grouping import complete_affinity, latent_groups
from .initialization import spectral_latents
from .kernel_optimization import (
    KernelObjective,
    group_kernels,
    kernel_penalty,
    kernel_stationarity,
    optimize_kernels,
)
from .kernels import Identity, Response
from .objective import (
    PreparedBlocks,
    build_grids,
    complete_objective,
    fit_preprocessing,
    make_blocks,
    solve_latents,
    solve_loadings,
)
from .optimization import (
    balance_global_scale,
    objective_components,
    stationarity_diagnostics,
)
from .results import SeriesResult


def select_candidate(candidates, *, policy="objective"):
    """Choose a finite training state; convergence remains a separate diagnostic.

    The legacy ``converged`` rule exists only for controlled research comparisons.
    Stable restart identifiers break exact objective ties independent of order.
    """
    if policy not in ("objective", "converged"):
        raise ValueError("unknown restart selection policy")
    finite = [c for c in candidates if np.isfinite(c[0]["objective"])]
    if not finite:
        raise RuntimeError("no usable finite fit; inspect restart_diagnostics_")
    if policy == "converged":
        finite = [c for c in finite if c[0]["converged"]] or finite
    return min(finite, key=lambda c: (c[0]["objective"], c[0]["restart"]))


def validate_affinity(affinity, subjects):
    if not isinstance(affinity, Mapping) or set(affinity) != set(subjects):
        raise ValueError("affinity must name every fitted subject")
    graph = {s: {} for s in subjects}
    for s, row in affinity.items():
        if not isinstance(row, Mapping) or not set(row) <= set(subjects):
            raise ValueError("affinity contains unknown subject names")
        for t, value in row.items():
            if not np.isfinite(value) or value < 0 or (s == t and value != 0):
                raise ValueError("affinity must be nonnegative with zero diagonal")
            graph[s][t] = float(value)
    A = np.array([[graph[s].get(t, 0) for t in subjects] for s in subjects])
    if not np.allclose(A, A.T, rtol=0, atol=1e-12):
        raise ValueError("affinity must be symmetric")
    return graph


class MultimodalSRM(BaseEstimator):
    """Regularized multimodal factor model; arrays are time by feature.

    By default, one latent response per run is shared exactly across participants
    and modalities, with separate participant–modality maps and response kernels.
    ``affinity_`` represents this sharing as a complete graph with unit edges;
    exact equality is enforced independently of ``latent_strength``.

    Penalties and balancing conventions are recorded in ``configuration_``.
    Reaching ``max_iter`` does not constitute convergence. ``init="hybrid"``
    uses one deterministic spectral start followed by paired random starts;
    ``init="spectral"`` repeats the same start redundantly at every restart.

    ``n_jobs`` runs independent restarts in isolated processes (default 1;
    -1 uses available CPUs, capped by ``n_init``). Integer ``random_state``
    preserves restart seeds and selection across worker counts. Native parallel
    cross-validation suppresses inner restart workers to avoid oversubscription.
    """

    def __init__(
        self,
        features=10,
        *,
        latent_dt,
        responses=None,
        latent_pooling="shared",
        latent_strength=1.0,
        modality_weights=None,
        loading_ridge=1e-3,
        loading_penalty_scaling="pair",
        latent_ridge=1e-4,
        temporal_strength=1e-2,
        max_iter=100,
        tol=1e-5,
        n_init=1,
        n_jobs=1,
        init="random",
        random_state=None,
        gap_threshold=None,
        kernel_max_iter=50,
        balance_scale=True,
    ):
        self.features = features
        self.latent_dt = latent_dt
        self.responses = responses
        self.latent_pooling = latent_pooling
        self.latent_strength = latent_strength
        self.modality_weights = modality_weights
        self.loading_ridge = loading_ridge
        self.loading_penalty_scaling = loading_penalty_scaling
        self.latent_ridge = latent_ridge
        self.temporal_strength = temporal_strength
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.n_jobs = n_jobs
        self.init = init
        self.random_state = random_state
        self.gap_threshold = gap_threshold
        self.kernel_max_iter = kernel_max_iter
        self.balance_scale = balance_scale

    def __setstate__(self, state):
        """Restore pre-policy pickles with the historical pair-scaled ridge."""
        state = state.copy()
        if "n_jobs" not in state:
            state["_legacy_n_jobs_default"] = True
        state.setdefault("n_jobs", 1)
        configuration = state.get("configuration_")
        legacy = "loading_penalty_scaling" not in state and (
            not isinstance(configuration, dict) or "loading_penalty_scaling" not in configuration
        )
        state.setdefault("loading_penalty_scaling", "pair")
        if legacy:
            state["_legacy_loading_penalty_scaling_default"] = True
        super().__setstate__(state)

    def _validate_configuration(self, data, affinity=None):
        if self.loading_penalty_scaling not in ("pair", "feature"):
            raise ValueError("loading_penalty_scaling must be 'pair' or 'feature'")
        if self.init not in ("random", "spectral", "hybrid"):
            raise ValueError("init must be random, spectral, or hybrid")
        if not isinstance(self.balance_scale, (bool, np.bool_)):
            raise ValueError("balance_scale must be boolean")
        for name in ["features", "max_iter", "n_init", "kernel_max_iter"]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        worker_count(self.n_jobs, self.n_init)
        for name in ["latent_dt", "loading_ridge", "latent_ridge", "tol"]:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive finite")
        for name in ["latent_strength", "temporal_strength"]:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative finite")
        if self.gap_threshold is not None and (
            not np.isfinite(self.gap_threshold) or self.gap_threshold <= 0
        ):
            raise ValueError("gap_threshold must be positive finite")
        if self.latent_pooling not in [
            "shared",
            "population",
            "neighborhood",
            "components",
        ]:
            raise ValueError("invalid latent_pooling")
        modalities = {m for runs in data.values() for mods in runs.values() for m in mods}
        self.responses_ = (
            {m: Response(Identity(), estimate=False) for m in sorted(modalities)}
            if self.responses is None
            else dict(self.responses)
        )
        if not modalities <= set(self.responses_) or not all(
            isinstance(r, Response) for r in self.responses_.values()
        ):
            raise ValueError("responses must cover observed modalities with Response objects")
        configured = set(self.responses_)
        weights = (
            {m: 1 / len(configured) for m in self.responses_}
            if self.modality_weights is None
            else dict(self.modality_weights)
        )
        if set(weights) != configured or not all(
            np.isfinite(v) and v >= 0 for v in weights.values()
        ):
            raise ValueError(
                "modality_weights must cover configured modalities and be finite nonnegative"
            )
        if not np.isclose(sum(weights.values()), 1.0, rtol=0, atol=1e-8):
            raise ValueError("modality_weights must sum to one")
        self.modality_weights_ = weights
        self.subjects_ = list(data)
        if self.latent_pooling in ("neighborhood", "components"):
            self.affinity_ = validate_affinity(affinity, self.subjects_)
        elif self.latent_pooling == "shared":
            self.affinity_ = complete_affinity(self.subjects_)
        else:
            self.affinity_ = None

    def fit(self, data, y=None, *, affinity=None):
        data = normalize_data(data)
        self.__dict__.pop("_legacy_loading_penalty_scaling_default", None)
        self.__dict__.pop("_legacy_n_jobs_default", None)
        for name in list(vars(self)):
            if name.endswith("_"):
                delattr(self, name)
        self._validate_configuration(data, affinity)
        active = {
            s: {
                r: {m: ts for m, ts in mods.items() if self.modality_weights_[m] > 0}
                for r, mods in runs.items()
            }
            for s, runs in data.items()
        }
        if any(not mods for runs in active.values() for mods in runs.values()):
            raise ValueError("participant/run has no positively weighted usable data")
        self.run_grids_, self.run_domains_ = build_grids(active, self.latent_dt)
        self.preprocessing_ = fit_preprocessing(active, self.modality_weights_)
        initial = {
            s: {m: self.responses_[m].initial_kernel() for m in stats}
            for s, stats in self.preprocessing_.items()
        }
        self.configuration_ = {
            "responses": {m: r.metadata for m, r in self.responses_.items()},
            "run_domains": copy.deepcopy(self.run_domains_),
            "latent_dt": self.latent_dt,
            "gap_policy": "split above twice median usable interval; singleton median cell"
            if self.gap_threshold is None
            else self.gap_threshold,
            "loading_penalty_scaling": self.loading_penalty_scaling,
            "penalty_scaling": (
                "loadings: mean pair Frobenius"
                + (" / fitted feature count" if self.loading_penalty_scaling == "feature" else "")
                + "; latent: mean subject/run trapezoid duration; graph: edge sum / subject/run count; kernel: mean participant deviations; independent priors: mean participant"
            ),
            "grid_policy": "observed positive-weight domain, ceil duration/dt; actual-domain support exclusion",
            "restart_selection": "minimum_finite_training_objective",
            "balance_scale": bool(self.balance_scale),
            "init": self.init,
            "stopping_rule": "objective_decrement_and_kernel_optimizer_success",
        }

        blocks_for = PreparedBlocks(
            make_blocks(
                active,
                self.run_grids_,
                self.run_domains_,
                self.preprocessing_,
                initial,
                self.responses_,
                self.modality_weights_,
                self.gap_threshold,
            ),
            self.run_grids_,
            self.responses_,
        )

        rng = check_random_state(self.random_state)
        candidates = []
        self.restart_diagnostics_ = []
        seeds = [int(rng.randint(0, 2**31 - 1)) for _ in range(self.n_init)]
        workers = worker_count(self.n_jobs, self.n_init)
        self.execution_ = dict(
            requested_jobs=self.n_jobs,
            restart_workers=workers,
            backend="serial" if workers == 1 else "loky",
        )
        arguments = [(initial, blocks_for, restart, seed) for restart, seed in enumerate(seeds)]
        if workers == 1:
            # Append immediately: research hooks use this boundary to maintain
            # their serial per-restart state.
            for args in arguments:
                record, candidate = self._fit_restart(*args)
                self.restart_diagnostics_.append(record)
                if candidate is not None:
                    candidates.append(candidate)
        else:
            for record, candidate in ordered_tasks(
                self._fit_restart, [args + (False,) for args in arguments], workers
            ):
                self.restart_diagnostics_.append(record)
                if candidate is not None:
                    candidates.append(candidate)
        if not candidates:
            raise RuntimeError("no usable finite fit; inspect restart_diagnostics_")
        best = self._select_fit(candidates)
        legacy = select_candidate(candidates, policy="converged")
        self.selection_diagnostics_ = {
            "policy": "minimum_finite_training_objective",
            "selected_restart": best[0]["restart"],
            "selected_objective": best[0]["objective"],
            "legacy_selected_restart": legacy[0]["restart"],
            "legacy_objective_gap": legacy[0]["objective"] - best[0]["objective"],
            "finite_restarts": len(candidates),
        }
        (
            record,
            self.loadings_,
            z,
            self.subject_kernels_,
            self.objective_history_,
            self.observation_blocks_,
        ) = best
        if self.observation_blocks_ is None:
            self.observation_blocks_ = blocks_for(self.subject_kernels_)
        self.converged_ = record["converged"]
        self.n_iter_ = record["iterations"]
        self.best_restart_ = record["restart"]
        self.optimization_diagnostics_ = {
            "components": copy.deepcopy(record["objective_components"]),
            "stationarity": copy.deepcopy(record["stationarity"]),
            "stopping_rule": self.configuration_["stopping_rule"],
            "kernel_stationarity": copy.deepcopy(record["kernel_stationarity"]),
            "kernel_stationarity_checked": record["kernel_stationarity"]["success"],
        }
        self.group_kernels_ = group_kernels(self.subject_kernels_, self.responses_)
        from .inference import component_support, coverage_metadata

        self.training_latents_ = {}
        for subject, runs in z.items():
            self.training_latents_[subject] = {}
            for run, values in runs.items():
                grid = self.run_grids_[run]
                local, domain = component_support(
                    self,
                    active,
                    self.observation_blocks_,
                    subject,
                    run,
                    self.run_domains_[run],
                )
                self.training_latents_[subject][run] = SeriesResult(
                    values,
                    grid,
                    (grid >= domain[0] - 1e-10) & (grid <= domain[1] + 1e-10),
                    metadata=coverage_metadata(
                        local,
                        run,
                        grid,
                        domain,
                        gap_threshold=self.gap_threshold,
                    ),
                )
        self._training_latent_arrays_ = z
        self._refresh_latent_groups()
        if self.latent_pooling not in ("neighborhood", "components"):
            self.population_latents_ = {
                r: SeriesResult(
                    np.mean([z[s][r] for s in self.subjects_], axis=0),
                    self.run_grids_[r],
                    self.run_grids_[r] <= self.run_domains_[r][1] + 1e-10,
                    metadata=coverage_metadata(
                        self.observation_blocks_,
                        r,
                        self.run_grids_[r],
                        self.run_domains_[r],
                        gap_threshold=self.gap_threshold,
                    ),
                )
                for r in self.run_grids_
            }
        if not self.converged_:
            warnings.warn(
                "MultimodalSRM did not converge; returning best finite restart",
                ConvergenceWarning,
                stacklevel=2,
            )
        return self

    def _fit_restart(self, initial, blocks_for, restart, seed, return_blocks=True):
        candidate = None
        local_rng = np.random.RandomState(seed)
        kernels = copy.deepcopy(initial)
        blocks = blocks_for(kernels)
        # Boundary-excluded mappings cannot be fitted by ridge alone.
        used = {(b.subject, b.modality) for b in blocks}
        kernels = {
            s: {m: k for m, k in mods.items() if (s, m) in used} for s, mods in kernels.items()
        }
        W = {
            s: {
                m: local_rng.normal(
                    scale=1 / np.sqrt(self.features),
                    size=(len(self.preprocessing_[s][m]["mean"]), self.features),
                )
                for m in mods
            }
            for s, mods in kernels.items()
        }
        diagnostics = []
        history = []
        converged = False

        def objective(z, w, k, b):
            return complete_objective(
                b,
                self.run_grids_,
                z,
                w,
                self.latent_pooling,
                self.latent_strength,
                self.latent_ridge,
                self.temporal_strength,
                self.loading_ridge,
                self.affinity_,
                kernel_penalty(k, self.responses_),
                loading_penalty_scaling=self.loading_penalty_scaling,
            )

        initialization = dict(method="random", fallback_reason=None)
        try:
            if self.init == "spectral" or (self.init == "hybrid" and restart == 0):
                informed_z, initialization = spectral_latents(
                    blocks,
                    self.run_grids_,
                    self.subjects_,
                    self.features,
                    self.latent_pooling,
                    self.affinity_,
                )
                if informed_z is not None:
                    W = solve_loadings(
                        blocks,
                        informed_z,
                        self.loading_ridge,
                        self.loading_penalty_scaling,
                    )
            z, solver = solve_latents(
                blocks,
                self.run_grids_,
                self.subjects_,
                W,
                self.features,
                self.latent_pooling,
                self.latent_strength,
                self.latent_ridge,
                self.temporal_strength,
                self.affinity_,
            )
            history.append(objective(z, W, kernels, blocks))
            for iteration in range(self.max_iter):
                previous = history[-1]
                new_W = solve_loadings(blocks, z, self.loading_ridge, self.loading_penalty_scaling)
                if objective(z, new_W, kernels, blocks) <= previous + 1e-12:
                    W = new_W
                kernel_objective = (
                    KernelObjective(
                        kernels,
                        self.responses_,
                        blocks_for,
                        z,
                        W,
                        lambda k: objective(z, W, k, blocks_for(k)),
                    )
                    if any(r.free_parameters for r in self.responses_.values())
                    else (lambda k: objective(z, W, k, blocks_for(k)))
                )
                new_k, kdiag = optimize_kernels(
                    kernels,
                    self.responses_,
                    kernel_objective,
                    self.kernel_max_iter,
                )
                kernels = new_k
                blocks = blocks_for(kernels)
                new_z, solver = solve_latents(
                    blocks,
                    self.run_grids_,
                    self.subjects_,
                    W,
                    self.features,
                    self.latent_pooling,
                    self.latent_strength,
                    self.latent_ridge,
                    self.temporal_strength,
                    self.affinity_,
                )
                if objective(new_z, W, kernels, blocks) <= objective(z, W, kernels, blocks) + 1e-12:
                    z = new_z
                scale_diagnostic = {"accepted": False, "reason": "disabled"}
                if self.balance_scale:
                    z, W, scale_diagnostic = balance_global_scale(
                        blocks,
                        self.run_grids_,
                        z,
                        W,
                        self.latent_pooling,
                        self.latent_strength,
                        self.latent_ridge,
                        self.temporal_strength,
                        self.loading_ridge,
                        self.affinity_,
                        kernel_penalty(kernels, self.responses_),
                        self.loading_penalty_scaling,
                    )
                current = objective(z, W, kernels, blocks)
                diagnostics.append(
                    {
                        "iteration": iteration + 1,
                        "latent_solver": solver,
                        "kernel_optimizer": kdiag,
                        "scale_balance": scale_diagnostic,
                    }
                )
                if not np.isfinite(current) or current > previous + 1e-10:
                    raise RuntimeError("nonfinite or increasing alternating objective")
                history.append(current)
                if (
                    abs(previous - current) <= self.tol * max(1.0, abs(previous))
                    and kdiag["success"]
                ):
                    converged = True
                    break
            record = dict(
                restart=restart,
                seed=seed,
                converged=converged,
                objective=history[-1],
                iterations=len(history) - 1,
                diagnostics=diagnostics,
                status="converged"
                if converged
                else "maximum iterations or nonlinear optimizer failure",
            )
            final_args = (
                blocks,
                self.run_grids_,
                z,
                W,
                self.latent_pooling,
                self.latent_strength,
                self.latent_ridge,
                self.temporal_strength,
                self.loading_ridge,
                self.affinity_,
                kernel_penalty(kernels, self.responses_),
                self.loading_penalty_scaling,
            )
            record["objective_components"] = objective_components(*final_args)
            record["stationarity"] = stationarity_diagnostics(*final_args)
            record["kernel_stationarity"] = kernel_stationarity(
                kernels,
                self.responses_,
                lambda candidate: objective(z, W, candidate, blocks_for(candidate)),
            )
            candidate = (
                record,
                copy.deepcopy(W),
                copy.deepcopy(z),
                copy.deepcopy(kernels),
                history,
                blocks if return_blocks else None,
            )
        except (RuntimeError, np.linalg.LinAlgError) as error:
            record = dict(
                restart=restart,
                seed=seed,
                converged=False,
                objective=np.inf,
                status=str(error),
                diagnostics=diagnostics,
            )
        record["initialization"] = initialization
        record["worker_pid"] = os.getpid()
        return record, candidate

    def _refresh_latent_groups(self):
        """Refresh exact sharing metadata, including after frozen calibration."""
        self.latent_groups_ = latent_groups(self.subjects_, self.latent_pooling, self.affinity_)
        self.configuration_["latent_groups"] = copy.deepcopy(self.latent_groups_)
        if self.latent_pooling == "components":
            self.component_latents_ = {
                group[0]: self.training_latents_[group[0]] for group in self.latent_groups_
            }

    def _select_fit(self, candidates):
        """Internal selection boundary; candidate arrays are not retained on fit."""
        return select_candidate(candidates)

    def infer_latent(self, data, *, times=None):
        from .inference import infer_latent

        return infer_latent(self, data, times=times)

    def predict(self, data, *, targets, source="within", times=None):
        from .inference import predict

        return predict(self, data, targets=targets, source=source, times=times)

    def calibrate(self, data, **kwargs):
        from .calibration import calibrate

        return calibrate(self, data, **kwargs)

    def kernel(self, modality, subject=None, level="subject", times=None):
        from .inspection import kernel

        return kernel(self, modality, subject=subject, level=level, times=times)

    def plot_kernels(self, modality, show_subjects=True, ax=None):
        from .inspection import plot_kernels

        return plot_kernels(self, modality, show_subjects=show_subjects, ax=ax)
