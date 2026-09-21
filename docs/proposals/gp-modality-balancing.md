# Proposal: explicit modality balancing for GP-MSRM

**Status: documentation-only research proposal for `ljchang/multimodalsrm`.** Explore an explicit, opt-in alternative for GP MAP and posterior inference. No estimator, API, default, or empirical result is introduced here. The current empirical campaign retains the ordinary GP likelihood. Equal modality weighting in R-MSRM does not establish an equivalent GP objective.

## Repository and implementation boundary

As inspected on 2026-09-21, the destination repository is a distribution scaffold: `model-integration.json` says `awaiting-model-import`, with a null source commit. Its [model integration plan](https://github.com/ljchang/multimodalsrm/blob/main/docs/model-integration.md) requires a verified source handoff before implementation. This proposal does not authorize copying a moving research implementation or bypassing that handoff.

The current implementation reference is the source research repository, under `src/personalized_srm/multimodal/`. Freeze the actual source and integration commit before any numerical comparison. Relevant source paths and behaviors are:

| Source path | Current behavior relevant to the proposal |
| --- | --- |
| `bayesian/problem.py` | Constructs a joint observation covariance per run, including cross-modality shared-GP covariance; sums run negative marginal log likelihoods; adds physical parameter priors once. The sampling potential additionally includes the coordinate-transform Jacobian. |
| `bayesian/grouped.py`, `bayesian/grouped_score.py`, `bayesian/multifactor.py`, `bayesian/multifactor_score.py` | Exact shared-functional reduction, including observation-noise normalizers and custom score/gradient paths. Its `weights` variables are feature loadings, not modality-balancing powers. |
| `_observation_preparation.py`, `bayesian/observation_adapter.py` | Packs only masked-in, response-support-eligible scalar observations on native clocks. Support eligibility is fixed over the declared response bounds. |
| `bayesian/model.py`, `bayesian/prediction.py`, `bayesian/trajectories.py`, `bayesian/transform.py` | MAP and posterior share a marginalized target; predictions and latent trajectories condition on observations using its covariance. GP consumes supplied units; any external standardization needs its own training-only boundary. Independent transforms preserve participant isolation. |
| `bayesian/posterior_updates.py`, `bayesian/posterior_persistence.py`, `bayesian/updated_posterior_persistence.py` | Joint update targets preserve a concrete observation union and provenance. Weight semantics must become part of that target and survive archives. |

Source workflow references are `docs/GP_GROUPED_POSTERIOR.md` and `docs/GP_POSTERIOR_WORKFLOW.md`. Existing numerical or posterior qualification applies to its recorded target, not automatically to this proposed extension. Destination module names remain subject to the integration plan.

## Mathematical target to review

Let `O_r` contain the admissible scalar observations for run `r`. One exact shared latent process `z_r` generates all modalities through individual loadings and response operators. Write `mu_j(z_r, theta)` for the resulting observation mean and `v_j(theta)` for the independent Gaussian observation variance. In the current source, noise parameters are indexed by participant and modality, while loadings/offsets are indexed by participant, modality, and feature.

The ordinary marginal likelihood integrates the product of conditional Gaussian densities against **one** GP prior per run. A candidate generalized posterior instead uses fixed positive observation powers `a_j`:

```math
\widetilde L_a(\theta;y)
=\prod_r\int p(z_r\mid\theta)
  \prod_{j\in O_r}\mathcal N(y_j;\mu_j(z_r,\theta),v_j(\theta))^{a_j}\,dz_r,
\qquad
\pi_a(\theta\mid y)\propto\pi(\theta)\widetilde L_a(\theta;y).
```

MAP would optimize `-log L_a - log pi`; posterior sampling would target the same physical density with the existing coordinate Jacobian. Parameter priors and the latent GP prior are not multiplied by the observation powers. All `a_j=1` must recover the ordinary model exactly.

Apply powers **before shared-GP marginalization**. Separately weighting modality-only marginal likelihoods discards their shared cross-covariances and is a different model. Multiplying pieces of a joint log determinant, or powering the already marginalized likelihood, is also a different target. The loss-based interpretation is consistent with generalized Bayesian updating; it is not an automatic calibration guarantee. See [Bissiri, Holmes and Walker (2016)](https://academic.oup.com/jrsssb/article/78/5/1103/7040623).

### Powered likelihood versus normalized variance rescaling

For `a>0`, scalar variance `v`, and a mean `mu`, direct Gaussian algebra gives:

```math
\mathcal N(y;\mu,v)^a
=c(a,v)\mathcal N(y;\mu,v/a),
\qquad
\log c(a,v)=\frac{1-a}{2}\log(2\pi v)-\frac12\log a.
```

Consequently, exact dense/grouped Gaussian machinery can evaluate the proposed marginal target using effective conditioning variance `v_j/a_j`, **plus** `sum_j log c(a_j,v_j)` in its log score. In negative-log form, subtract that sum from the normalized Gaussian NLL. This identity is a proposed implementation route, not an implemented capability.

Simply replacing `v` by `v/a` and keeping the normalized Gaussian density defines another legitimate generative model. It is variance inflation when `a<1` and contraction when `a>1`. With fixed powers and fixed noise, both targets differ only by a parameter-independent constant and give the same parameter posterior. With learned noise, the correction depends on parameters and changes noise gradients, MAP estimates, and posterior draws. Retaining only a weighted residual term is incorrect for either full target unless its normalizers are derived consistently. Conditional latent distributions agree between the two formulations at the same fixed parameters; integrating over their different parameter posteriors generally breaks that agreement.

## What does “equal” mean?

Freeze one rule before implementation; do not conflate relative modality balance with total likelihood strength. A simple candidate uses `N_m`, the number of admissible scalar training observations in modality `m`, `M` active modalities, and a declared evidence scale `N_ref`:

```math
a_j=\tau\frac{N_{\mathrm{ref}}}{M N_{m(j)}}.
```

Each modality then receives total power `tau N_ref/M`. Here `tau` is an inverse temperature: larger values strengthen observation evidence relative to both latent and parameter priors. `N_ref=sum_m N_m` preserves the ordinary total power at `tau=1`; a fixed reference count or `N_ref=1` defines a different prior/evidence balance. Total-count scaling means duplicating a modality can change every modality's strength. These options are not interchangeable defaults.

| Decision | Alternatives that require explicit review |
| --- | --- |
| Modality unit | Average over all scalar observations, average over features first, or equalize participant/run blocks. Feature-count balancing alone does not remove native sampling-rate or duration imbalance. |
| Participant/run unit | Pooled counts give longer runs and more-observed participants more within-modality mass. A nested mean can use `a_j=tau N_ref/(M S_m R_sm N_srm)` over nonempty participant/run blocks, giving each participant equal modality mass and each of their runs equal mass. A run-first hierarchy differs when participation is unbalanced. |
| Masks and support | Count only retained scalar observations after target exclusion and fixed support eligibility; never count padded or masked values. Decide whether denominators are frozen at training or recomputed from an allowed conditioning set. Specify absent modalities, empty blocks, and missing features without division by zero. |
| Temperature | Predeclare `N_ref` and a small `tau` panel. Unit temperature does not imply ordinary information content after normalization. Do not learn temperature silently through reconstruction or choose it from protected test outcomes. |
| Zero weights | Prefer explicit observation exclusion for the first prototype. If zero powers are later supported, remove those factors rather than evaluating `v/0` or `log 0`. |

Equal total power does **not** imply equal influence on the shared latent response. Influence also depends on noise, loadings, response operators, temporal coverage, redundant features, and prior geometry. In the conditional linear-Gaussian problem, modality information enters through terms such as `A_m.T diag(a_j/v_j) A_m`; equal scalar sums do not make these matrices equal. Balancing is not evidence of timing, filter-shape, or mechanism recovery.

## Conditioning, prediction, and information boundaries

Remove held-out targets before any preprocessing fit, count calculation, initialization, optimization, and donor/participant conditioning. Never derive powers from target values or residuals. A known acquisition design may specify masks in advance, but masked entries contribute no likelihood terms. Altering hidden target values must leave all fitted parameters, weights, and predictions unchanged. Preserve native clocks, separate run processes, and independent participant transforms.

Weighted training followed by ordinary conditioning is a distinct hybrid method and must not happen implicitly. The proposed powered target implies donor conditioning with precision `a_j/v_j` for posterior means, marginal variances, and joint latent trajectories at every retained parameter draw. Independent MAP transforms and posterior donor/calibration/participant updates need the same declared policy. Initially reject unsupported operations rather than silently falling back.

For predictive observation intervals, distinguish physical measurement noise `v` from the effective conditioning variance `v/a`. If the chosen target is a generalized posterior for the original observation model, adding original `v` is a candidate predictive convention requiring coverage assessment. If it is the normalized variance-rescaled generative model, its observation noise is `v/a`. Record which distribution a predictive score assesses; weighted objective values alone are not comparable predictive evidence.

Recomputing modality denominators on a new observation union changes powers on old observations. Such an update is not the old posterior multiplied only by new likelihood factors. Decide whether to freeze historical powers or rebuild the declared full target; archive counts, powers, normalization, temperature, observation identities, and update policy. Preserve one GP prior per run and parameter priors once. Ordinary saved models must retain ordinary behavior.

## Bounded research milestones and stopping points

1. **Mathematical decision record; no fits.** Resolve powered versus normalized target, normalization hierarchy, reference count/temperature, predictive noise, and update policy. Freeze a clean implementation source after model handoff. Document explicit opt-in behavior; default remains ordinary. Scope the initial prototype to dense/grouped algebra, independent Gaussian noise, one/two factors, Identity/Gaussian responses. State-space, spectral, correlated residuals, run baselines, and wider response families require separate qualification.

2. **Small exact synthetic implementation screen.** At most six predeclared tiny fixtures spanning unequal feature counts/rates, feature masks, missing streams, two runs, and learned noise. Compare an independent dense Gaussian calculation with grouped objectives, gradients, and conditional latent moments at fixed interior parameter points (`atol=rtol=1e-7`, float64); check finite-difference gradients away from bounds. Verify the analytic normalizer correction and its learned-noise derivative. Include `a=1` default regressions for objectives, gradients, predictions, and serialization; permutation invariance; the declared duplication/count behavior; and hidden-target perturbation invariance. Exercise the custom gradient paths, not just forward objective values. Any unexplained failure stops this milestone.

3. **Bounded uncertainty screen.** First check a fixed-parameter Gaussian case with independently calculated conditional means/covariances. Then cap the pilot at 20 synthetic datasets per predeclared data-generating regime, using ordinary inference and one chosen balancing rule at `tau in {0.5,1,2}`. Include a balanced correctly specified regime, an unequal-count/rate regime, and one modest prespecified misspecification regime. Freeze truths/seeds, sampler budget, interval levels, estimands, and diagnostic gates before launch; retain failed fits and denominators. Report convergence, latent/noise/filter interval coverage, interval width, bias, and prediction separately, with binomial uncertainty and no calibration claim from a small pilot. Conventional prior-predictive SBC is an appropriate check for the ordinary model and for a fully specified normalized-rescaling generative model under their own generators. A powered generalized posterior under the original generator is not expected automatically to have uniform SBC ranks; distinguish its coverage assessment from sampler correctness. Temperature selection and assessment use separate simulations. Passing the pilot only motivates a separately reviewed larger confirmation.

4. **Protected predictive comparison, only after preceding gates.** Use matched splits and participant-isolated transforms; select normalization/temperature and any regularization using inner validation. The selection objective is **Balanced held-out prediction across all modalities**, with a declared macro-average and every modality reported separately. Compare ordinary GP, the selected GP extension, and the appropriate frozen R-MSRM baseline with the same permitted sources. Keep training reconstruction, alignment, latent recovery, timing, width/shape recovery, convergence, and empirical validity separate. Reused development movie splits remain exploratory; freeze choices before a distinct-movie/original-target confirmation. No new empirical sweep or default change is authorized by this proposal.

This PR contains only the proposal. Later implementation and synthetic evidence belong in separate reviewable changes after mathematical decisions and integration. Empirical observations, participant-level records, fitted archives, and research/LFS payloads remain outside this PR. No numerical screen described above has been run for this proposal.
