---
search:
  exclude: true
---

# Model extraction implementation plan

Goal: port the two supported models from merged source commit `67128c7a562cab67499bdc0604357731a628aa9e` into the standalone MIT package, preserving numerical behavior and explicit capability limits.

Spec: `../model-integration.md` and the approved extraction readiness assessment. Work occurs on `feature/model-extraction`; research code, arrays, archives, and history stay in the source repository. Version remains a development version until release approval.

## Tasks

- [x] Extract the frozen 77-module dependency inventory plus the new warmup-checkpoint module and standalone quality diagnostics. Preserve relative imports and serialized format identifiers; create focused public exports. Record source/destination hashes and provenance in `model-source.json`.
- [x] Migrate synthetic tests for the selected implementations. Use explicit package-relative helper imports compatible with pytest importlib; exclude research experiment/legacy tests with an explicit inventory. Verify collection and run installed-package suites.
- [x] Add focused synthetic examples and capability/migration documentation. Explain original-environment R joblib replay and source-bound warmup restart. Preserve fitted Bayesian archive schemas.
- [x] Compare old/new R fits and Bayesian MAP/prediction behavior on identical synthetic inputs; replay old fitted Bayesian archives under the new namespace. Run migrated persistence/checkpoint tests. No new convergence campaign.
- [x] Integrate full core/Bayesian tests with bounded CI shards, built artifacts on Linux/macOS, and nonempty reports. Build wheel/sdist, check metadata and optional imports, run examples and tests outside source paths. Keep release tag/version gate.
- [ ] Independently review extraction boundaries, test coverage, compatibility and release gates. Address blocking findings, commit, and prepare a draft PR with explicit verification evidence and remaining limitations.

## Progress

Baseline: infrastructure suite passed 24 tests. Isolated worktree created from `dc781a8`; source checkpoint is committed and merged. The integration manifest stays pending until transferred suites exist; dev version prevents release even once model CI is active.

Verification in progress: installed core suite passed 251 tests; independent review confirmed unchanged production ASTs/import sets after permitted formatting/message edits. Source-generated GP archives for schemas 1–4 replay exactly. Fresh R/GP numerical parity passed; elapsed-time telemetry and identifiers incorporating it are excluded from fresh-run comparison. Full Bayesian suites are running. A reviewer identified a missing public posterior participant test; the full test and independent oracle have been restored.

Final local acceptance: 1,519 unique model cases and 30 infrastructure cases pass. The restored participant test file passes all 19 cases. Full core suite passes without JAX/NumPyro. Wheel/sdist examples, metadata, lint, workflow syntax and artifact boundary checks pass. Independent review cleared the restored coverage finding. Draft PR creation/hosted checks are the remaining handoff step; publication is not performed.
