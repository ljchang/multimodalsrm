# CI and test policy

The reusable `ci.yml` workflow runs on pull requests, main-branch pushes, manual requests and release verification. Python 3.12 is the initial tested version. The metadata permits newer Python, but broader compatibility must be measured before being advertised.

| Job | What it verifies |
| --- | --- |
| Infrastructure and lint | Release gate regressions, package metadata/import checks and Ruff |
| Model integration status | Honest pending status or valid source commit and required model suite files |
| Build distributions | Wheel/sdist creation and strict Twine metadata rendering checks |
| Installed wheel/sdist on Linux/macOS | Clean artifact installation, dependency consistency and import tests outside the checkout |
| Core/Bayesian models on Linux/macOS | Entire migrated suite directories against the built wheel; two core and eight Bayesian file shards per platform |
| Required checks | Every applicable job passed; model jobs may be skipped only while integration is pending |

Release verification requires integrated models and a matching non-development version tag, so the pending-model exception never permits an upload. Each model suite must execute at least one passing test; an entirely skipped suite is rejected even if pytest exits successfully. Optional Bayesian jobs enable CPU float64 and one BLAS/OpenMP thread. Keep ordinary suite execution under the workflow timeout; costly SBC, recovery and scalability studies should use separately reviewed manual workflows with explicit budgets. No research campaign is launched by package CI.

For local infrastructure tests:

```sh
python -m pip install -e '.[dev]'
python -m pytest tests/infrastructure
python scripts/check_release.py --allow-pending
```

After the handoff, the last command reports `ready=true`. Publication still requires a matching non-development version tag. Installed-package checks are also run separately by CI, preventing an editable install from hiding missing distribution files.

The synthetic model suites check numerical contracts and execution paths. They do not establish general convergence, uncertainty calibration, physiological timing, response recovery or empirical usefulness. See `capabilities.md` for scope and `test-migration.json` for transferred tests and exclusions.

For installed model tests, build and install the wheel with `[test,bayesian]`, then run:

```sh
JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_model_tests.py core
JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_model_tests.py bayesian
```

The runner uses Python isolated mode and pytest importlib mode. Test-only reference integrators and synthetic fixtures live under `tests/reference`; they are not installed in the wheel. Every file belongs to exactly one CI shard, and each shard must execute passing tests. Source-repository research campaigns and legacy estimators are explicitly excluded from this package.
