# CI and test policy

The reusable `ci.yml` workflow runs on pull requests, main-branch pushes, manual requests and release verification. Python 3.12 is the initial tested version. The metadata permits newer Python, but broader compatibility must be measured before being advertised.

| Job | What it verifies |
| --- | --- |
| Infrastructure and lint | Release gate regressions, package metadata/import checks and Ruff |
| Model integration status | Honest pending status or valid source commit and required model suite files |
| Build distributions | Wheel/sdist creation and strict Twine metadata rendering checks |
| Installed wheel/sdist on Linux/macOS | Clean artifact installation, dependency consistency and import tests outside the checkout |
| Core/Bayesian models on Linux/macOS | Entire migrated suite directories against the built wheel; pending until integration |
| Required checks | Every applicable job passed; model jobs may be skipped only while integration is pending |

Release verification requires integrated models and a matching non-development version tag, so the pending-model exception never permits an upload. Each model suite must execute at least one passing test; an entirely skipped suite is rejected even if pytest exits successfully. Optional Bayesian jobs enable CPU float64 and one BLAS/OpenMP thread. Keep ordinary suite execution under the workflow timeout; costly SBC, recovery and scalability studies should use separately reviewed manual workflows with explicit budgets. No such campaign is launched by this scaffold.

For local infrastructure tests:

```sh
python -m pip install -e '.[dev]'
python -m pytest tests/infrastructure
python scripts/check_release.py --allow-pending
```

The last command reports `ready=false` until the model handoff. Running without `--allow-pending` must fail at this stage. Installed-package checks are also run separately by CI, preventing an editable install from hiding missing distribution files.

The current tests establish packaging and release-control behavior. They do not establish model numerical accuracy, convergence, uncertainty calibration or scientific usefulness.
