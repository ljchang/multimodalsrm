# Releasing MultimodalSRM

Package publication is deliberately blocked while `model-integration.json` says `awaiting-model-import`. Do not upload a placeholder package to claim the name. The repository can be public while the implementation is being prepared.

## One-time Trusted Publisher registration

Configure a pending publisher in each account's publishing settings. The two package indices have separate accounts and configuration.

| Field | PyPI | TestPyPI |
| --- | --- | --- |
| Settings | https://pypi.org/manage/account/publishing/ | https://test.pypi.org/manage/account/publishing/ |
| PyPI project name | `multimodalsrm` | `multimodalsrm` |
| Owner | `ljchang` | `ljchang` |
| Repository | `multimodalsrm` | `multimodalsrm` |
| Workflow filename | `release.yml` | `release.yml` |
| Environment | `pypi` | `testpypi` |

Create the matching GitHub environments. The workflow uses OIDC with `id-token: write` only in publishing jobs; no long-lived API token is needed. Publisher registration is an account-side prerequisite and is not accomplished by committing the workflow. A pending publisher does not reserve the project name. See [PyPI's pending publisher guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/) and [publishing guide](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

## Release candidate

1. Complete the [model handoff](model-integration.md), required suites, clean installed-artifact checks and documentation.
2. Set the single version in `src/multimodalsrm/_version.py`, for example `0.1.0rc1`, and update the changelog. The version must not be a development or local build.
3. Commit reviewed changes, let fast PR CI pass and create the matching Git tag, for example `v0.1.0rc1`. You can also run **Full test suite** manually on a branch before tagging; it does not publish.
4. Manually run **Release** with that tag. Manual runs publish only to TestPyPI after resolving the tag to one commit and running `full-tests.yml`: the complete Linux/macOS model matrix and installed wheel/sdist checks. Fast PR CI never replaces this release gate.
5. Install the exact candidate from TestPyPI in a clean environment and exercise its workflows. Resolve dependencies from PyPI separately; avoid allowing TestPyPI to supply arbitrary runtime dependencies. Record the version, artifact hashes and results.

## Production

Publish a GitHub Release for the intended version tag. The Release workflow resolves the tagged commit, checks source readiness/version agreement, builds the distributions once, runs the full suite against those artifacts and uploads those same artifacts to PyPI only after the full workflow succeeds. A main-branch push alone never publishes. Prerelease GitHub Releases can publish genuine prerelease versions to PyPI; reserve manual runs for TestPyPI rehearsal.

Successful registration or CI does not prove an upload succeeded. Verify the published project version and clean installation after the first real deployment. The current scaffold has no model release and intentionally cannot exercise an actual upload yet.

## Failures

- **Model integration incomplete:** complete the handoff; do not bypass the gate.
- **Version/tag mismatch:** correct the versioned commit and use a new appropriate tag; never overwrite a published version.
- **Missing publisher:** check the exact repository owner, workflow filename and environment in the correct index's account settings.
- **Existing distribution:** investigate prior upload status. Versioned distributions must not be silently replaced.

GitHub action references are pinned to reviewed commits. Dependabot proposes action updates monthly. Bayesian runtime dependency updates require their own numerical/diagnostic qualification rather than automatic version bumps.
