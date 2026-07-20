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

`docs/source/_generated` and `docs/_build` are gitignored, generated
artifacts. `_generated/api` holds autosummary stubs and `_generated/gallery`
holds sphinx-gallery output; both tools only regenerate files they detect as
changed, so a stale stub left over from a module rename/removal, or a stale
gallery page left over from renaming/removing an example script, can point at
something that no longer exists and break the build (e.g. an
`ImportExceptionGroup`/`ModuleNotFoundError` from autosummary). If the build
fails after restructuring `src/` or `examples/`, do a clean rebuild:

```
rm -rf docs/_build docs/source/_generated
uv run sphinx-build docs/source docs/_build
```

(PowerShell: `Remove-Item -Recurse -Force docs/_build, docs/source/_generated`)

## Manual overrides in `conf.py`

Unrelated bits of docs-generation logic, specific to one or a few classes, live directly in `conf.py` (connected via `setup(app)`) rather than as `_ext/` extensions.

- **`type_alias_value`** -- extracts the value of `Type` aliases to added to the docs as an `:annotation:` in `_templates/autosummary/attributes.rst`.
- **`suppress_enum_inherited_members`** -- hides the `int`/`IntEnum` and `str`/`StrEnum` stdlib
  members (`bit_length`, `from_bytes`, `numerator`, ...) that
  `:inherited-members:` in `_templates/autosummary/class.rst` pulls onto
  `enum`'s docs page, while leaving `:inherited-members:` in place
  globally. Implemented via  the `autodoc-process-docstring` event, which hands over the *same* `options` object autodoc later reads when deciding which members to render. We set `options.inherited_members = set()` and `options.members = ALL`. Unrelatedly, we also set `options.member_order = "bysource"` to keep the `enum`'s members in declaration order (rather than alphabetical). Two things to know before touching this:
  - **This event also fires for an unrelated internal call** (autosummary's
    own one-line blurb extraction for the parent module's class-listing
    table), where `options.inherited_members` is `None` rather than the
    directive's resolved `{"object"}`. The
    `options.inherited_members == {"object"}` guard exists specifically to
    skip that call -- mutating its (differently-shaped) options crashes the
    build.
  - **Sphinx 9.1.0 has two distinct `ALL` sentinel objects** in different
    internal modules (`sphinx.ext.autodoc.ALL` vs
    `sphinx.ext.autodoc._sentinels.ALL`) that are *not* the same object,
    despite both meaning "document all members". The class-rendering code
    path (`sphinx.ext.autodoc._generate`) checks identity against the
    `_sentinels` one, so `options.members` must be set to
    `from sphinx.ext.autodoc._sentinels import ALL` specifically -- the
    top-level `sphinx.ext.autodoc.ALL` re-export is a different (legacy)
    singleton and silently fails the `is ALL` check, which makes `want_all`
    false and the page render with **zero** members (own members included),
    not just fewer. **Failure mode on a Sphinx upgrade:** if `_sentinels.ALL`
    moves or the two sentinels get unified, either an `ImportError` (easy to
    spot) or the page reverting to showing every stdlib member again (or
    going blank) with no error -- re-check the relevant docs pages (e.g.
    `offline_particles.kernels.status.Status`) after any Sphinx version bump.


## Custom Sphinx extensions (`docs/source/_ext/`)

All verified against Sphinx 9.1.0, which `pyproject.toml` pins exactly
(`sphinx==9.1.0`), since these patches reach into
leading-underscore private modules with no cross-version compatibility
guarantee.

- **`napoleon_tables.py`** -- renders Napoleon's Parameters/Returns/Raises/etc.
  sections as tables instead of field lists. Napoleon has no supported hook
  for changing how a section renders, so this monkeypatches
  `GoogleDocstring._parse_parameters_section` and its sibling `_parse_*`
  methods directly (`NumpyDocstring` inherits them unchanged, so patching the
  base class covers both docstring styles). **Failure mode on a Sphinx
  upgrade:** if these private method names or signatures change, the patch
  either raises on import (attribute error at `setup()` time) or silently
  stops being applied and the affected sections fall back to Napoleon's
  default field-list rendering -- check that the generated tables still look
  like tables after upgrading.

- **`member_labels.py`** -- adds a `method`/`attribute` keyword prefix to
  plain methods/attributes, matching Sphinx's built-in `property`/
  `classmethod` prefixes. This one only uses **public, documented** Sphinx
  extension API: `PyMethod`/`PyAttribute` (`sphinx.domains.python`) are
  subclassed and re-registered via `app.add_directive_to_domain(..., override=True)`.
  Low risk -- the Python domain's directive classes and `get_signature_prefix`
  hook have been stable for many Sphinx releases.

- **`member_order.py`** -- reorders `:member-order: groupwise` so a class's
  members are grouped as attributes/properties, then methods, then
  staticmethods, then classmethods (each group alphabetical), instead of
  Sphinx's built-in groupwise order (classmethod, staticmethod, method,
  attribute/property last). The current ("dynamic") autodoc backend has **no
  public hook** for customising the groupwise sort key -- that only exists on
  the legacy `Documenter`-subclass backend (toggled by
  `autodoc_use_legacy_class_based`, not currently enabled here). So this
  monkeypatches the private `_groupwise_order_key` properties on
  `sphinx.ext.autodoc._property_types._FunctionDefProperties` and
  `_AssignStatementProperties`. This reaches into a leading-underscore
  internal module rather than a documented API. **Failure mode:** an
  `ImportError` on `_property_types` (build fails immediately, easy to spot)
  or, if the module survives but its internals change shape, a silent
  reversion to Sphinx's default groupwise order (classmethods first) with no
  error -- re-check member ordering on a representative class (e.g.
  `offline_particles.kernels.ParticleKernel`, which has properties, methods,
  staticmethods, and a classmethod) after any Sphinx version bump. If this
  breaks, switching to the legacy autodoc backend (which exposes
  `Documenter.sort_members` as a public override point) is the fallback
  option that was considered and deferred when this was written.

- **`property_types.py`** -- shows a property's return type (e.g. `property
  data: SpatialArray`) even though `autodoc_typehints = "none"` suppresses
  type hints everywhere else. A property's return-type annotation is already
  computed unconditionally by autodoc; it's discarded at render time by a
  single `autodoc_typehints` gate in `sphinx.ext.autodoc._renderer
  ._directive_header_lines` that's shared by every object kind (property,
  attribute, data, method, function) -- there's no public per-object-kind
  override (the `autodoc-process-signature` event only rewrites a function/
  method's signature line, it never reaches the property/attribute `:type:`
  line). So this monkeypatches `_directive_header_lines`, patched at the
  point of use in `sphinx.ext.autodoc._generate` (which imported the function
  by name into its own module namespace, rather than the point of definition
  in `_renderer`, where patching would have no effect). **Failure mode on a
  Sphinx upgrade:** an `AttributeError`/`ImportError` if `_generate` no longer
  imports `_directive_header_lines` under that name (build fails
  immediately), or a silent reversion to no property types (with no error) if
  the function's keyword arguments change shape but it still imports cleanly
  -- re-check a property-heavy class like `offline_particles.fields
  .StaticField` after any Sphinx version bump. Of the four extensions here,
  this and `member_order.py` are the most likely to break on an upgrade,
  since both reach into the same leading-underscore internal modules of the
  "dynamic" autodoc backend rather than a documented API.
