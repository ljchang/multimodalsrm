# Contributing

Model integration is pending. Infrastructure contributions can proceed independently; changes to model algorithms should remain in the source development repository until the handoff described in [the integration plan](docs/model-integration.md).

Use Python 3.12 and install `.[dev]` into an isolated environment. Before submitting a pull request, run:

```sh
ruff check .
ruff format --check .
python -m pytest tests/infrastructure
python -m build
python -m twine check --strict dist/*
```

Validate workflow edits with `actionlint` when available. Tests should exercise observable behavior and use synthetic data. Do not commit participant observations, fitted states, private data paths or research archive payloads. Keep the base package import independent of optional Bayesian/plotting backends and legacy comparator dependencies.

Once models are integrated, run the relevant `tests/core` and `tests/bayesian` suites, with JAX float64 enabled for the latter. Do not change a statistical target, default, coordinate convention or archive schema as an incidental consequence of packaging work. Explain any intentional change and its validation in the pull request.

Contributions are made under the repository's MIT license. Retain applicable authorship and third-party notices when transferring code. Report scientific limitations separately from software bugs and reproducibility failures.
