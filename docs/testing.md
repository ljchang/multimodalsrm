# CI and test policy

Routine reviews use the fast `ci.yml` workflow. The complete `full-tests.yml` workflow runs on manual request and is mandatory for every release publication. Python 3.12 is the initial tested version. The metadata permits newer Python, but broader compatibility must be measured before being advertised.

| Change or event | Checks |
| --- | --- |
| Documentation-only PR or main push | Ruff, infrastructure/policy tests, integration metadata, wheel/sdist build, strict Twine checks and a strict documentation-site build; no numerical dependencies or model tests |
| Code, tests, examples, dependencies, workflows or mixed changes | All documentation checks, the full core suite and curated Bayesian smoke tests against the built wheel on Linux, plus both quickstarts |
| Manual full-suite run or release | Complete Linux/macOS core and Bayesian suites and clean wheel/sdist installation checks |

Documentation-only means changes entirely within `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `zensical.toml`, `requirements-docs.txt`, or recognized prose/image files under `docs/`. Unknown paths, executable documentation, mixed changes and unavailable history receive code checks. The workflow always reports **Required checks**; only documentation profiles may skip its model jobs. New commits cancel superseded fast runs.

The Bayesian smoke manifest is `tests/smoke-tests.json`. It exercises likelihood/prior contracts, factorization, response filters, learned GP timescales and a short sampler checkpoint/resume check. It is deliberately limited. For numerical changes, run the affected tests and request the complete suite before merging when broader coverage is needed. Passing fast CI does not imply the full suite passed.

## Complete suite

In GitHub Actions, select **Full test suite**, choose **Run workflow**, and select the branch to test. With the GitHub CLI:

```sh
gh workflow run full-tests.yml --ref YOUR_BRANCH
```

This runs tests without publishing. Release verification calls the same full workflow at the resolved release commit; fast PR checks cannot substitute for it.

| Job | What it verifies |
| --- | --- |
| Infrastructure and lint | Release gate regressions, package metadata/import checks and Ruff |
| Model integration status | Valid source commit and required integrated model suite files |
| Build distributions | Wheel/sdist creation and strict Twine metadata rendering checks |
| Installed wheel/sdist on Linux/macOS | Clean artifact installation, dependency consistency and import tests outside the checkout |
| Core/Bayesian models on Linux/macOS | Entire migrated suite directories against the built wheel; two core and eight Bayesian file shards per platform |
| Full suite required checks | Every required job passed |

Release verification requires integrated models and a matching non-development version tag. Each model suite must execute at least one passing test; an entirely skipped suite is rejected even if pytest exits successfully. Bayesian jobs enable CPU float64 and one BLAS/OpenMP thread. Keep ordinary suite execution under the workflow timeout; costly SBC, recovery and scalability studies should use separately reviewed manual workflows with explicit budgets. No research campaign is launched by package CI.

For local infrastructure tests:

```sh
python -m pip install -e '.[dev]'
python -m pytest tests/infrastructure
python scripts/check_release.py
```

The last command checks integration readiness. Publication still requires a matching non-development version tag. Installed-package checks are also run separately by CI, preventing an editable install from hiding missing distribution files.

The synthetic model suites check numerical contracts and execution paths. They do not establish general convergence, uncertainty calibration, physiological timing, response recovery or empirical usefulness. See `capabilities.md` for scope and `test-migration.json` for transferred tests and exclusions.

For installed model tests, build and install the wheel with `[test,bayesian]`, then run:

```sh
JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_model_tests.py core
JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_model_tests.py bayesian --smoke
# Complete Bayesian suite when needed:
JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_model_tests.py bayesian
```

The runner uses Python isolated mode and pytest importlib mode. Test-only reference integrators and synthetic fixtures live under `tests/reference`; they are not installed in the wheel. Every file belongs to exactly one CI shard, and each shard must execute passing tests. Source-repository research campaigns and legacy estimators are explicitly excluded from this package.
