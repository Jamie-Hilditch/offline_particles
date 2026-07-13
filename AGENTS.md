# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project

`offline_particles` is a Lagrangian particle-tracking library for offline advection through ocean model output (in particular ROMS). Simulations advect particles through a `Fieldset` of gridded fields using composable, kernel-based physics. The package is in early development (`0.1.0`); breaking changes are expected.

Managed with `uv`. The package is pure Python, with `numba`-jitted kernel functions for performance-critical hot loops.

## Commands

All commands run through `uv` (uses `uv.lock`, Python >=3.12).

- Install/sync dev environment: `uv sync --frozen --group dev`
- Run all tests: `uv run pytest`
- Run a single test file: `uv run pytest tests/kernels/test_advection_kernels.py`
- Run a single test: `uv run pytest tests/kernels/test_advection_kernels.py::test_name`
- Doctests are part of the suite: pytest is configured with `testpaths = ["tests", "src"]` and `addopts = ["--doctest-modules"]`, so docstring examples in `src/` are collected and run automatically.
- Type check: `uv run ty check`
- Lint: `uv run ruff check --fix .`
- Format: `uv run ruff format .`
- Build docs: `uv run sphinx-build docs/source docs/build`
- Pre-commit runs ruff-check and ruff-format on `src|tests|docs/source` — install with `uv run pre-commit install`.

CI (`.github/workflows/ci.yml`) runs `ty check`, `ruff check`/`format`, and `pytest` as separate jobs — match these locally before pushing.

## Architecture

### Kernel system (`kernels/`)

The core abstraction. A **kernel function** (`KernelFunction`) takes three mappings — particle properties (`Mapping[str, NDArray]`), scalars (`Mapping[str, np.generic]`), and field data (`Mapping[str, FieldData]`) — and mutates particle properties in place. The `@kernel_function(particle_property_keys=..., scalar_keys=..., field_data_keys=...)` decorator (`kernels/_kernels.py`) unpacks these mappings into positional arguments so the wrapped implementation can be written as a plain (often `numba`-jitted) function.

A `ParticleKernel` wraps one or more kernel functions together with declarations of the inputs they require (`ParticlePropertyDeclaration`, `ScalarDeclaration`, `FieldDataDeclaration` — all subclasses of `KernelInputDeclaration`), which carry dtype/layout constraints used for validation. `ParticleKernel.chain`/`.chain_with` merge kernels, erroring on conflicting declarations.

A `BoundKernel` binds a `ParticleKernel`'s declared input names to concrete argument names (e.g. binding a kernel's generic `"u"` field input to a fieldset's `"u_velocity"` field). Bindings can be composed via `BoundKernel.chain`/`.chain_with`, and rebound via `.rebind()`.

Kernel construction is organized by physics domain as submodules under `kernels/`: `advection`, `buoyancy`, `relaxation`, `interpolation`, `timestepping` (Adams-Bashforth update kernels), `validation` (bounding-box / finite-index checks), `roms` (ROMS-specific z-coordinate kernels), `status`, `timed_activation`, `base`. Each exposes `construct_*_kernel(...)` factory functions returning `ParticleKernel`s rather than exposing raw kernel functions. `models/roms/__init__.py` (`roms_ab3_timestepper`) shows the top-level pattern: assemble domain kernels into tendency/AB-update/post-step kernel lists and hand them to an `ABTimestepper`.

Performance-critical kernel input/output is implemented as plain Python with `numba`-jitted hot loops (e.g. `kernels/status/`, `kernels/roms/vertical_coordinate/`), not compiled extension modules.

### Fields and data (`fields.py`, `fieldset.py`, `spatial_arrays.py`)

`Field` (`StaticField` / `TimeDependentField`) wraps an underlying array (often backed by `xarray`/`dask`/`zarr`) with an `ArrayLayout` describing axis order (`ArrayAxis.Z/Y/X`) and staggering (`Stagger.CENTER`/edge). `Fieldset` is a named collection of `Field`s plus scalar constants, with domain index bounds (`domain_bbox`) used to validate particle positions. Fields expose `get_field_data(time_index, bbox)` returning a `FieldData` (array + offsets) for a given bounding box in index space.

### Particles and launching (`particles.py`, `launcher.py`)

`Particles` holds per-particle-set property arrays (built via `Particles.build_from_kernels`, which inspects the kernels that will run to determine required properties/dtypes); `ParticlesView` is a read-only view. `Launcher` is responsible for actually executing a `BoundKernel`: it computes a padded bounding box around active particles (tracking history for stability), gathers the bound particle properties/scalars/field data, and calls the kernel. Scalars can be supplied dynamically via the `ScalarSource` descriptor protocol — objects (e.g. `Clock`, timesteppers, events) declare `ScalarSource` class attributes that the `Launcher` auto-discovers via `register_scalar_data_sources_from_object`.

### Timestepping (`timestepping.py`, `kernels/timestepping/`)

`Timestepper` is the base class (`ABTimestepper` for Adams-Bashforth, `RK2Timestepper` for explicit RK2); each organizes kernels into initialisation/validation/pre-step/step/post-step phases run by `Simulation.step()`. The initialisation phase runs every step (not just once at simulation start): it's gated on `Status.INITIALISING` so particles that transition to that status mid-simulation (e.g. via a timed-release kernel) get the same setup and status finalization as particles present at simulation start — see `Timestepper.run_initialisation`/`set_initial_status`. `Clock` tracks simulation time/iteration/dt and direction (forward/backward in time).

### Simulation orchestration (`simulation.py`)

`SimulationBuilder` is the entry point for assembling a run: register `ParticleSet`s (name, particle count, timestepper, property dtypes), attach events (`every_n`/`every_dt`/`at_iteration`/`at_time`) and output writers (`add_output_writer`), then call `build_simulation()`. This gathers all kernels required per particle set (from timesteppers, events, and outputs), builds `Particles`/`ParticlesView` for each, wires up output writers' recurring/static output events, and returns an immutable `Simulation`. `Simulation.run()`/`.step()` drive the main loop against configurable stopping conditions (iteration, time, or wall time).

### Events and output (`events/`, `output/`)

`Event` (with `SimulationState` snapshots) is triggered by one of four schedulers (recurring/one-shot × iteration/time) and can launch kernels plus run arbitrary code against the current `SimulationState`. `output/` defines `Output` (a named, kernel-backed output variable) and `AbstractOutputWriter`/`AbstractOutputWriterBuilder`, with `ZarrOutputBuilder` writing simulation output to Zarr stores; output writers are registered on the `SimulationBuilder` and their write events feed back into the same event-scheduling machinery.

## Code review guidance

Apply these when reviewing or authoring changes to `src` or `tests`:

- Docstrings of any modified function/method/class must stay accurate and follow **NumPy style** strictly (see `pydocstyle.convention = "numpy"` in `pyproject.toml`), and must render correctly with Sphinx + napoleon.
- The docstring `Raises` section must document every exception raised directly *or* indirectly (via another function/method/class in this codebase); exceptions from external libraries may optionally be documented.
- Any modified behaviour must be tested — write tests for new behaviour and fill gaps in existing coverage, not just for the lines touched.
- Classes should implement `__str__` and `__repr__` unless there's a good reason not to (see `ParticleKernel`, `BoundKernel`, `KernelInputDeclaration` for the established pattern).
- Consider adding a `description`/`summary` property to a class when a more detailed, end-user-facing description of an instance would be useful; this is in addition to, not a replacement for, a proper class-level docstring.
- This codebase is still in development, so breaking changes are acceptable — but if a change has downstream impact elsewhere in the codebase, call that out explicitly in review.
- Kernel implementations are plain Python with `numba`-jitted hot loops; the codebase has no Cython dependencies, and refactors/additions should keep it that way.
