# Contribute to the documentation

The site is built with [Zensical](https://zensical.org/) from the Markdown files in `docs/`. Navigation and theme settings live in `zensical.toml`. Edit the source through a pull request; do not commit generated `site/` files.

## Preview locally

From the repository root, in a Python environment:

```sh
python -m pip install -r requirements-docs.txt
zensical serve
```

Open the local URL printed by the server. Documentation dependencies are separate from the model dependencies, so writing prose does not require installing JAX or running model fits.

Before submitting, run the same strict build as CI:

```sh
zensical build --clean --strict
```

The build validates internal links and anchors. Add new reader-facing pages to the navigation in `zensical.toml`. Historical implementation plans remain available by direct link but are excluded from site search. Use relative links between documentation pages. Link to GitHub for files outside `docs/`, such as runnable examples and source manifests, so links work both in the repository and on the site.

## Review and deployment

Fast PR CI builds the site inside its required quality job. Prose and site-configuration changes use the lightweight documentation profile; model, test, and workflow changes retain code checks. See [test policy](testing.md).

The separate **Documentation** workflow rebuilds and deploys the site after a push to `main`. It can also be run manually on `main`. PRs and other branches cannot deploy. The deployment job alone receives Pages and OIDC write permissions, and GitHub's `github-pages` environment can add deployment protection rules.

The site is live at [ljchang.github.io/multimodalsrm](https://ljchang.github.io/multimodalsrm/). GitHub Pages uses **Settings → Pages → Build and deployment → Source: GitHub Actions**. Confirm the workflow and live URL after merging; a successful local build is not a deployment.

## Scope of this first site

The first version organizes installation, data preparation, a conceptual model comparison, existing runnable tutorials, capabilities, and contributor guidance. Complete API documentation, longer posterior tutorials, and versioned release documentation remain future work. The site follows `main`; it does not currently offer a version selector.
