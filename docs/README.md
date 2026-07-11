# Building the docs locally

Build the HTML docs with Sphinx:

```
uv run sphinx-build docs/source docs/_build
```

Then serve the output and view it in a browser:

```
uv run python -m http.server --directory docs/_build 8080
```

Open `http://localhost:8080` in a browser.

## Rebuilding after moving/renaming modules

`docs/source/_api` and `docs/_build` are gitignored, generated artifacts.
Sphinx only regenerates autosummary stubs for files it detects as changed, so a
stale `_api/*.rst` stub left over from a module rename or removal can point at
a module that no longer exists and break the build (e.g. an
`ImportExceptionGroup`/`ModuleNotFoundError` from autosummary). If the build
fails after restructuring `src/`, do a clean rebuild:

```
rm -rf docs/_build docs/source/_api
uv run sphinx-build docs/source docs/_build
```

(PowerShell: `Remove-Item -Recurse -Force docs/_build, docs/source/_api`)
