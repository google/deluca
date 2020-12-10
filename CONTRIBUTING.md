# Contributing

## Setting up a development environment

[Insert typical language about using some python environment]. Within that
environment, run the following:

```python
# Install editable version of code
pip install -e .

# (Optional) Install testing dependencies
pip install -e .[dev]

# (Optional) Install documentation dependencies
pip install -e .[docs]

# (Optional) Install pre-commit hooks
pre-commit install
```

## Linting

We use the `pre-commit` tools to automatically lint (and fix, where possible)
for Python warnings, errors, and style. This runs automatically on `git commit`
and will either pass with no issue, make changes to files, and / or ask you to
make fixes. If the tests don't pass, the commit fails (a good thing! Keeps
history clean).

## Testing

We also use `pytest`, which we encourage you to run before contributing
(`pytest -n auto` for parallelized testing).

## Documentation

We use [Read the Docs](https://deluca.readthedocs.io/en/latest) to
automatically build our documentation from `docstrings` and `*.rst` files in the
`docs` directory.

## Releasing

We rely on [GitHub Actions](https://github.com/google/deluca/actions) to
build, release, and publish `deluca` (described in `.github/workflows`). We
trigger release by creating a new version via `bumpversion` (see below) and
pushing tags.

```bash
bumpversion [major|minor|patch]
```

To release, run `git push --tags`.

# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google/conduct/).
