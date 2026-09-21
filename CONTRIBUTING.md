# Contributing

The extracted production models live in this repository. Research campaigns should consume a pinned package source commit or release. See the [integration provenance](docs/model-integration.md) before moving code between repositories.

Use Python 3.12 or later and an isolated environment. Install the base development tools with `python -m pip install -e '.[dev]'`; use `'.[dev,bayesian]'` for Bayesian work. Before submitting a pull request, run the checks applicable to the change:

```sh
ruff check .
ruff format --check .
python -m pytest tests/infrastructure tests/core
JAX_ENABLE_X64=true JAX_PLATFORM_NAME=cpu python scripts/run_model_tests.py bayesian --smoke
python examples/r_quickstart.py
JAX_ENABLE_X64=true JAX_PLATFORM_NAME=cpu python examples/gp_map_quickstart.py
python -m build
python -m twine check --strict dist/*
```

For documentation-only edits, lint, infrastructure checks and a package build are sufficient. Code PRs run the full core suite and Bayesian smoke tests on Linux. Bayesian changes should also run the affected tests; request **Full test suite** in GitHub Actions for broader numerical or platform coverage before merging. Every release must pass that complete suite, including Linux/macOS and installed wheels/source distributions. See the [test policy](docs/testing.md) for routing rules and full-suite commands.

Bayesian tests and examples require the optional Bayesian extra. CI also checks installed artifacts. Validate workflow edits with `actionlint` when available. Tests should exercise observable behavior using synthetic data. Do not commit participant observations, fitted states, private data paths or research archive payloads. Keep the base package import independent of optional Bayesian/plotting backends and legacy comparator dependencies.

Do not change a statistical target, default, coordinate convention or archive schema as an incidental consequence of packaging work. Explain intentional changes and their validation in the pull request. Preserve target exclusion before preprocessing and participant isolation in independent-run inference. Keep serialization identifiers stable; see [migration](docs/migration.md).

Contributions are made under the repository's MIT license. Retain applicable authorship and third-party notices when transferring code. Report software checks, numerical qualification, fit convergence, uncertainty calibration, response recovery and empirical validity separately. A small synthetic example is not a recovery study.

Documentation lives in `docs/` and is published with Zensical. Install `requirements-docs.txt`, preview with `zensical serve`, and check links with `zensical build --clean --strict`. See the [documentation guide](docs/documentation.md) for navigation and deployment.
