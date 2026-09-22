# Upcoming API changes and optimizations

This is a **September 22, 2026 review snapshot**, not a release announcement. The main documentation baseline is commit `b538c86` (including the learned-response grouping in PR #34). Pending PRs can still change. The generated reference always describes the source in the checkout, rather than treating a proposal as an installed option.

## Changes that affect configuration

| Change | State at review | What an analysis author should do |
| --- | --- | --- |
| [Remove BachSCR, #32](https://github.com/ljchang/multimodalsrm/pull/32) | Pending | Use Bateman for new SCR workflows. Preserve old environments or refit; do not rename an existing fit's kernel. |
| [R-based MAP initialization, #18](https://github.com/ljchang/multimodalsrm/pull/18) | Pending | Account for the proposed `SearchConfig.r_init=True` default when comparing new fits with historical starts. |
| [Fixed-response grouped state-space, #31](https://github.com/ljchang/multimodalsrm/pull/31) | Merged into this baseline | No new argument: eligible fixed-response state-space calculations group automatically. |
| [Learned-response grouped state-space, #34](https://github.com/ljchang/multimodalsrm/pull/34) | Merged into this baseline; replaces the closed stacked #30 | No new public selector: automatic grouping now also covers supported learned responses. |
| [Empirical performance report, #21](https://github.com/ljchang/multimodalsrm/pull/21) | Pending documentation | Treat the reported experiments as bounded evidence, not a new public inference API. |

## Bach removal is a breaking change

PR #32 removes `BachSCR` from public imports, response admission, specialized inference paths and the GP archive registry. Archives containing a Bach record are rejected with migration guidance before reconstruction. The remaining response identifiers and archive schema stay unchanged.

Bach and Bateman have different shapes and parameter meanings. The proposed migration is to reproduce historical fits in their original environment, or configure Bateman and **refit the original observations**. There is no automatic conversion of fitted parameters, predictions or warmup checkpoints. Release 0.1.0 supports Bach; the PR also names development snapshot `8eeb84f` for historical reproduction. The actual fitting environment remains authoritative.

The new kernel gallery therefore centers the continuing families and labels Bach as legacy. The API generator retains Bach only while it is exported, and the figure generator does not depend on it.

## R initialization changes a search default

PR #18 adds the Boolean field `SearchConfig.r_init`, proposed default `True`. It seeds the first full-training GP MAP restart with a bounded CPU R-MSRM fit. This includes the MAP search used to initialize posterior sampling. It is **MAP initialization**, not NUTS adaptation or an alternative posterior target.

The preliminary R fit uses one hybrid start with bounded iteration/grid budgets. It transfers supported loadings, offsets, residual variances and Gaussian response estimates into GP coordinates. Non-Gaussian responses stay fixed during the preliminary fit; the subsequent GP optimization still learns all requested parameters. Numerical initialization failures retain the historical start with a warning and a recorded reason.

The proposal preserves restart count/seeds, later prior-based starts, native observations, priors and convergence requirements. Conditional calibration and posterior updates skip the preliminary R fit. Archives without this setting restore it as `False`, preserving historical fit evidence.

After that PR is integrated, the public switch will be:

```text
SearchConfig(r_init=True)   # proposed new default: R-based first start
SearchConfig(r_init=False)  # historical initialization
```

These lines are shown as pending configuration, not runnable examples for the current baseline. The source-generated [configuration reference](api/configuration.md) will acquire the field automatically after integration and regeneration. A better start or fewer iterations does not guarantee a shorter complete fit, the same optimum, or response recovery.

## Two different uses of “grouped”

| Mechanism | How it is selected | What is preserved |
| --- | --- | --- |
| Grouped GP covariance algebra | `linear_algebra="grouped"` | The modeled covariance likelihood, reduced over repeated modality/time functionals |
| Grouped state-space filtering/smoothing | Automatic inside eligible `linear_algebra="state_space"` calculations | Scalar residual contributions, variances, likelihood normalization and cross-factor moments |

The merged fixed- and learned-response optimizations group **exactly matching** modality/time nodes when the noise prior's support is strictly positive. It does not bin near-coincident timestamps, merge different modalities, or introduce a new model prior. Priors admitting zero variance retain scalar updates. State-space posterior sampling remains unsupported.

The merged learned-response implementation extends this to supported widths, scales, ratios, Bateman constants and lags. Node membership stays fixed while parameter-dependent event times are shifted and sorted, including ties/crossings. It introduces no new public constructor argument. Existing response admission and approximation tolerances still apply.

See the [current grouped state-space implementation guide](grouped-state-space.md) for its scoped benchmarks. Synthetic speedups and prototype empirical results should not be quoted as general runtime guarantees. In particular, the blocked Gibbs prototype discussed in PR #21 does **not** add an `inference="gibbs"` choice to the supported API.

## Maintaining this page when the queue lands

After each relevant merge, regenerate `scripts/build_api_reference.py`, update the choice tables and capability text against the new source, and run the strict docs build. Retire or revise pending notes instead of leaving conflicting current/pending instructions. The API inventory detects unknown exports and changed defaults; it does not establish that a new option is scientifically qualified.
