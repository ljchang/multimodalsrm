# Distribution infrastructure implementation plan

**Goal:** Establish the public MIT-licensed `ljchang/multimodalsrm` repository with verified packaging, CI and gated TestPyPI/PyPI release workflows.

**Architecture:** Use a new source-layout package and fresh Git history. Model development remains in its existing repository until a verified handoff; a machine-readable integration manifest blocks package publication until that handoff and its test suites exist.

**Stack:** Python 3.12, setuptools, pytest, Ruff, build, Twine, GitHub Actions and PyPI Trusted Publishing.

**Approved design:** A focused package for R-MSRM and GP-MSRM, public visibility, MIT license, optional Bayesian dependencies, Linux/macOS CI and release-driven PyPI deployment. FIR is deferred. This milestone introduces infrastructure only.

## Constraints

- Do not transfer research arrays, fitted states or unrelated model histories.
- Do not modify the source repository or develop a second Bayesian implementation.
- Package version `0.1.0.dev0` denotes the unreleased scaffold.
- A green infrastructure check is not evidence of model implementation or recovery.
- PyPI uploads require the integrated model suites and a matching non-development release version.
- No credentials are stored in the repository; publisher registration is account-side configuration.

## Execution

1. [x] Add tests that exercise release-readiness validation: pending integration, missing model suites, malformed source identity and mismatched/development tags must block release. Verify the missing gate fails before implementing it.
2. [x] Implement focused package metadata, version module, MIT license, explicit artifact inclusion, integration manifest and release gate. Provide dependency extras without legacy Git URL requirements.
3. [x] Add reusable CI for lint/infrastructure tests, wheel/sdist builds, installed-artifact tests and conditional core/Bayesian suites. Add release workflow that reruns these checks on the resolved tag commit before upload.
4. [x] Document contributing, extraction mapping, capability handoff, tutorials and exact Trusted Publisher settings. Keep planned model interfaces clearly distinct from available exports.
5. [x] Build/install/test both distributions in isolated environments. Validate workflows with actionlint and review changes independently. Fix findings before publication.
6. [ ] Commit only the explicit new repository files, create the public GitHub repository, push and verify hosted CI. Configure release environments. Register publishers if an authenticated PyPI account is available; otherwise report the precise remaining account setup.

## Acceptance commands

```sh
python -m pytest tests/infrastructure
ruff check .
ruff format --check .
actionlint
python -m build
python -m twine check --strict dist/*
python scripts/check_release.py --allow-pending
# Expected rejection while the scaffold has no models:
python scripts/check_release.py --tag v0.1.0
```

Installed-artifact checks must run from outside the checkout after installing the wheel and the sdist separately. The GitHub matrix is the evidence for Linux/macOS portability. Initial publication means publishing the repository; no placeholder package is uploaded to PyPI.

## Local verification record

September 21: 24 infrastructure tests passed, including regressions requiring the expected source repository and rejecting entirely skipped model suites. Independent review reproduced all 24 passes and found no remaining actionable issues. Ruff and actionlint passed. Wheel and sdist both built and passed strict Twine checks; each installed successfully into a separate clean environment, passed dependency consistency checks and passed both package/import tests outside the checkout. Artifacts were inspected for unintended data/cache payloads.

The local macOS interpreter skipped editable-install `.pth` files marked hidden by the local environment; installing the built wheel resolved that environment-specific issue without adding source paths to tests. Hosted editable-install verification remains part of Linux infrastructure CI.
