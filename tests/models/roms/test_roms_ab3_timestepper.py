"""Tests for the ROMS AB3 timestepper constructor."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import offline_particles.models.roms as roms_models
import offline_particles.timestepping as timestepping_module
from offline_particles.fields import StaticField
from offline_particles.fieldset import Fieldset
from offline_particles.kernels.status import Status
from offline_particles.launcher import Launcher
from offline_particles.particles import Particles
from offline_particles.timestepping import ABTimestepper


def _assert_initialises_to_multistep_2(timestepper: ABTimestepper) -> None:
    status = np.array([np.uint8(Status.INITIALISING)], dtype=np.uint8)
    timestepper._initialise_status_kernel.kernel({"status": status}, {}, {})
    assert status[0] == np.uint8(Status.MULTISTEP_2)


def _install_constructor_spies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    patch_linear_damping: bool = True,
    patch_quadratic_damping: bool = True,
    patch_buoyancy: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    calls: dict[str, Any] = {}
    sentinels = {
        "x_adv": object(),
        "y_adv": object(),
        "z_adv": object(),
        "linear_damping": object(),
        "quadratic_damping": object(),
        "buoyancy": object(),
        "add_property": object(),
        "ab_update_x": object(),
        "ab_update_y": object(),
        "ab_update_z": object(),
        "ab_update_w_rel": object(),
        "compute_zidx": object(),
        "compute_z": object(),
        "ab_bump_status": object(),
    }

    def stub(name: str, return_value: object):
        def _call(*args, **kwargs):
            calls[name] = {"args": args, "kwargs": kwargs}
            return return_value

        return _call

    def call_list_stub(name: str, return_value: object):
        # construct_compute_zidx_kernel is used both for the post-step recompute and,
        # depending on `initialise_z`, the initialisation kernel - record every call rather
        # than just the last one.
        def _call(*args, **kwargs):
            calls.setdefault(name, []).append({"args": args, "kwargs": kwargs})
            return return_value

        return _call

    def advection_stub(*args, **kwargs):
        key = {"_dxidx0": "x_adv", "_dyidx0": "y_adv"}[args[0]]
        calls[key] = {"args": args, "kwargs": kwargs}
        return sentinels[key]

    def ab3_update_stub(prop: str, *args, **kwargs):
        calls["ab_update"] = {"args": (prop, *args), "kwargs": kwargs}
        return {
            "xidx": sentinels["ab_update_x"],
            "yidx": sentinels["ab_update_y"],
            "z": sentinels["ab_update_z"],
            "w_rel": sentinels["ab_update_w_rel"],
        }[prop]

    monkeypatch.setattr(roms_models, "construct_advection_kernel", advection_stub, raising=True)
    monkeypatch.setattr(
        roms_models, "construct_ZYX_interpolation_kernel", stub("z_adv", sentinels["z_adv"]), raising=True
    )
    if patch_linear_damping:
        monkeypatch.setattr(
            roms_models,
            "construct_linear_damping_kernel",
            stub("linear_damping", sentinels["linear_damping"]),
            raising=True,
        )
    if patch_quadratic_damping:
        monkeypatch.setattr(
            roms_models,
            "construct_quadratic_damping_kernel",
            stub("quadratic_damping", sentinels["quadratic_damping"]),
            raising=True,
        )
    if patch_buoyancy:
        monkeypatch.setattr(
            roms_models,
            "construct_buoyancy_force_kernel",
            stub("buoyancy", sentinels["buoyancy"]),
            raising=True,
        )
    monkeypatch.setattr(
        roms_models, "construct_add_property_kernel", stub("add_property", sentinels["add_property"]), raising=True
    )
    monkeypatch.setattr(roms_models, "construct_ab3_update_kernel", ab3_update_stub, raising=True)
    monkeypatch.setattr(
        timestepping_module,
        "construct_ab_bump_status_kernel",
        stub("ab_bump_status", sentinels["ab_bump_status"]),
        raising=True,
    )
    monkeypatch.setattr(
        roms_models,
        "construct_compute_zidx_kernel",
        call_list_stub("compute_zidx", sentinels["compute_zidx"]),
        raising=True,
    )
    monkeypatch.setattr(
        roms_models, "construct_compute_z_kernel", stub("compute_z", sentinels["compute_z"]), raising=True
    )

    return calls, sentinels


def test_roms_ab3_timestepper_default_constructor_wires_expected_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, sentinels = _install_constructor_spies(monkeypatch)

    timestepper = roms_models.roms_ab3_timestepper(hc=5.0, NZ=4)

    assert isinstance(timestepper, ABTimestepper)
    assert timestepper._order == 3
    assert timestepper.index_padding == 5

    assert calls["x_adv"]["args"] == ("_dxidx0", "u", "pm", ("Z", "Y", "X"), ("Y", "X"))
    assert calls["x_adv"]["kwargs"] == {"metric": True}
    assert calls["y_adv"]["args"] == ("_dyidx0", "v", "pn", ("Z", "Y", "X"), ("Y", "X"))
    assert calls["y_adv"]["kwargs"] == {"metric": True}
    assert calls["z_adv"]["args"] == ("_dz0", "w")
    assert calls["z_adv"]["kwargs"] == {"accumulate": True}
    assert calls["compute_zidx"][0]["args"] == ()
    assert calls["compute_zidx"][0]["kwargs"] == {
        "hc": 5.0,
        "NZ": 4,
        "h": "h",
        "zeta": "zeta",
        "C": "C",
    }
    assert calls["ab_bump_status"]["args"] == ()
    assert "linear_damping" not in calls
    assert "quadratic_damping" not in calls
    assert "buoyancy" not in calls
    assert "add_property" not in calls

    assert timestepper.initialisation_kernels == [sentinels["compute_zidx"]]
    _assert_initialises_to_multistep_2(timestepper)
    assert timestepper.post_step_kernels == [sentinels["compute_zidx"]]
    assert timestepper._tendency_kernels == [sentinels["x_adv"], sentinels["y_adv"], sentinels["z_adv"]]
    assert timestepper._ab_update_kernels == [
        sentinels["ab_update_x"],
        sentinels["ab_update_y"],
        sentinels["ab_update_z"],
    ]


def test_roms_ab3_timestepper_forwards_compute_zidx_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, sentinels = _install_constructor_spies(monkeypatch)

    timestepper = roms_models.roms_ab3_timestepper(
        vertical_velocity=False,
        hc=7.5,
        NZ=10,
        h="bathymetry",
        zeta="surface",
        C="stretching",
    )

    assert isinstance(timestepper, ABTimestepper)
    assert calls["compute_zidx"][0]["args"] == ()
    assert calls["compute_zidx"][0]["kwargs"] == {
        "hc": 7.5,
        "NZ": 10,
        "h": "bathymetry",
        "zeta": "surface",
        "C": "stretching",
    }
    assert timestepper.post_step_kernels == [sentinels["compute_zidx"]]
    assert timestepper.initialisation_kernels == [sentinels["compute_zidx"]]


def test_roms_ab3_timestepper_default_initialises_zidx_from_z(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, sentinels = _install_constructor_spies(monkeypatch)

    timestepper = roms_models.roms_ab3_timestepper(hc=5.0, NZ=4)

    # called once for the post-step recompute and once as the initialisation kernel
    assert len(calls["compute_zidx"]) == 2
    assert calls["compute_zidx"][1]["args"] == ()
    assert calls["compute_zidx"][1]["kwargs"] == {
        "hc": 5.0,
        "NZ": 4,
        "h": "h",
        "zeta": "zeta",
        "C": "C",
        "only_initialising": True,
    }
    assert "compute_z" not in calls
    assert timestepper.initialisation_kernels == [sentinels["compute_zidx"]]


def test_roms_ab3_timestepper_initialise_z_true_initialises_z_from_zidx(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, sentinels = _install_constructor_spies(monkeypatch)

    timestepper = roms_models.roms_ab3_timestepper(hc=5.0, NZ=4, initialise_z=True)

    # only the post-step recompute of zidx; z is initialised instead of zidx
    assert len(calls["compute_zidx"]) == 1
    assert calls["compute_z"]["args"] == ()
    assert calls["compute_z"]["kwargs"] == {
        "hc": 5.0,
        "NZ": 4,
        "h": "h",
        "zeta": "zeta",
        "C": "C",
        "only_initialising": True,
    }
    assert timestepper.initialisation_kernels == [sentinels["compute_z"]]
    assert timestepper.post_step_kernels == [sentinels["compute_zidx"]]


def test_roms_ab3_timestepper_with_buoyancy_and_damping_wires_optional_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, sentinels = _install_constructor_spies(monkeypatch)

    timestepper = roms_models.roms_ab3_timestepper(
        vertical_velocity=False,
        index_padding=9,
        u="u_field",
        v="v_field",
        w="w_field",
        pm="pm_field",
        pn="pn_field",
        h="bathymetry",
        zeta="surface",
        C="stretching",
        rho="density",
        hc=7.5,
        NZ=10,
        constant_linear_damping_coefficient=0.25,
        property_quadratic_damping_coefficient="quad_drag",
        scalar_buoyancy_coefficient="g_over_rho0",
    )

    assert isinstance(timestepper, ABTimestepper)
    assert timestepper._order == 3
    assert timestepper.index_padding == 9

    assert calls["x_adv"]["args"] == ("_dxidx0", "u_field", "pm_field", ("Z", "Y", "X"), ("Y", "X"))
    assert calls["y_adv"]["args"] == ("_dyidx0", "v_field", "pn_field", ("Z", "Y", "X"), ("Y", "X"))
    assert "z_adv" not in calls

    assert calls["linear_damping"]["args"] == ("_dw_rel0", "w_rel")
    assert calls["linear_damping"]["kwargs"] == {
        "constant_coefficient": 0.25,
        "property_coefficient": None,
        "scalar_coefficient": None,
    }
    assert calls["quadratic_damping"]["args"] == ("_dw_rel0", "w_rel")
    assert calls["quadratic_damping"]["kwargs"] == {
        "constant_coefficient": None,
        "property_coefficient": "quad_drag",
        "scalar_coefficient": None,
    }
    assert calls["buoyancy"]["args"] == ("_dw_rel0",)
    assert calls["buoyancy"]["kwargs"] == {
        "particle_density": "rho",
        "density_field": "density",
        "array_layout": roms_models._DENSITY_FIELD_ARRAY_LAYOUT,
        "constant_coefficient": None,
        "property_coefficient": None,
        "scalar_coefficient": "g_over_rho0",
    }
    assert calls["add_property"]["args"] == ("w_rel", "_dz0")
    assert calls["add_property"]["kwargs"] == {}
    assert calls["ab_update"]["args"] == ("w_rel", "_dw_rel0", "_dw_rel1", "_dw_rel2")
    assert calls["ab_bump_status"]["args"] == ()

    assert timestepper.initialisation_kernels == [sentinels["compute_zidx"]]
    _assert_initialises_to_multistep_2(timestepper)
    assert timestepper.post_step_kernels == [sentinels["compute_zidx"]]
    assert timestepper._tendency_kernels == [
        sentinels["x_adv"],
        sentinels["y_adv"],
        sentinels["linear_damping"],
        sentinels["quadratic_damping"],
        sentinels["buoyancy"],
        sentinels["add_property"],
    ]
    assert timestepper._ab_update_kernels == [
        sentinels["ab_update_x"],
        sentinels["ab_update_y"],
        sentinels["ab_update_z"],
        sentinels["ab_update_w_rel"],
    ]


def test_roms_ab3_timestepper_linear_scalar_damping_wires_scalar_coefficient(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, sentinels = _install_constructor_spies(monkeypatch)

    timestepper = roms_models.roms_ab3_timestepper(
        vertical_velocity=False,
        hc=5.0,
        NZ=4,
        constant_linear_damping_coefficient=None,
        property_linear_damping_coefficient=None,
        scalar_linear_damping_coefficient="linear_drag",
    )

    assert isinstance(timestepper, ABTimestepper)
    assert calls["linear_damping"]["args"] == ("_dw_rel0", "w_rel")
    assert calls["linear_damping"]["kwargs"] == {
        "constant_coefficient": None,
        "property_coefficient": None,
        "scalar_coefficient": "linear_drag",
    }
    assert calls["add_property"]["args"] == ("w_rel", "_dz0")
    assert timestepper._tendency_kernels == [
        sentinels["x_adv"],
        sentinels["y_adv"],
        sentinels["linear_damping"],
        sentinels["add_property"],
    ]


def test_roms_ab3_timestepper_quadratic_scalar_damping_wires_scalar_coefficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, sentinels = _install_constructor_spies(monkeypatch)

    timestepper = roms_models.roms_ab3_timestepper(
        vertical_velocity=False,
        hc=5.0,
        NZ=4,
        constant_quadratic_damping_coefficient=None,
        property_quadratic_damping_coefficient=None,
        scalar_quadratic_damping_coefficient="quadratic_drag",
    )

    assert isinstance(timestepper, ABTimestepper)
    assert calls["quadratic_damping"]["args"] == ("_dw_rel0", "w_rel")
    assert calls["quadratic_damping"]["kwargs"] == {
        "constant_coefficient": None,
        "property_coefficient": None,
        "scalar_coefficient": "quadratic_drag",
    }
    assert calls["add_property"]["args"] == ("w_rel", "_dz0")
    assert timestepper._tendency_kernels == [
        sentinels["x_adv"],
        sentinels["y_adv"],
        sentinels["quadratic_damping"],
        sentinels["add_property"],
    ]


@pytest.mark.parametrize(
    ("kwargs", "patch_linear_damping", "patch_quadratic_damping"),
    [
        (
            {
                "constant_linear_damping_coefficient": 0.5,
                "property_linear_damping_coefficient": "linear_drag",
            },
            False,
            True,
        ),
        (
            {
                "constant_linear_damping_coefficient": 0.5,
                "scalar_linear_damping_coefficient": "linear_drag",
            },
            False,
            True,
        ),
        (
            {
                "property_linear_damping_coefficient": "linear_drag",
                "scalar_linear_damping_coefficient": "linear_drag_scalar",
            },
            False,
            True,
        ),
    ],
)
def test_roms_ab3_timestepper_linear_damping_rejects_multiple_coefficient_selectors(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
    patch_linear_damping: bool,
    patch_quadratic_damping: bool,
) -> None:
    _install_constructor_spies(
        monkeypatch,
        patch_linear_damping=patch_linear_damping,
        patch_quadratic_damping=patch_quadratic_damping,
    )

    with pytest.raises(ValueError, match="Exactly one coefficient"):
        roms_models.roms_ab3_timestepper(hc=5.0, NZ=4, **kwargs)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("kwargs", "patch_linear_damping", "patch_quadratic_damping"),
    [
        (
            {
                "constant_quadratic_damping_coefficient": 0.5,
                "property_quadratic_damping_coefficient": "quadratic_drag",
            },
            True,
            False,
        ),
        (
            {
                "constant_quadratic_damping_coefficient": 0.5,
                "scalar_quadratic_damping_coefficient": "quadratic_drag",
            },
            True,
            False,
        ),
        (
            {
                "property_quadratic_damping_coefficient": "quadratic_drag",
                "scalar_quadratic_damping_coefficient": "quadratic_drag_scalar",
            },
            True,
            False,
        ),
    ],
)
def test_roms_ab3_timestepper_quadratic_damping_rejects_multiple_coefficient_selectors(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
    patch_linear_damping: bool,
    patch_quadratic_damping: bool,
) -> None:
    _install_constructor_spies(
        monkeypatch,
        patch_linear_damping=patch_linear_damping,
        patch_quadratic_damping=patch_quadratic_damping,
    )

    with pytest.raises(ValueError, match="Exactly one coefficient"):
        roms_models.roms_ab3_timestepper(hc=5.0, NZ=4, **kwargs)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "constant_buoyancy_coefficient": 0.5,
            "property_buoyancy_coefficient": "buoyancy_coeff",
        },
        {
            "constant_buoyancy_coefficient": 0.5,
            "scalar_buoyancy_coefficient": "buoyancy_coeff",
        },
        {
            "property_buoyancy_coefficient": "buoyancy_coeff",
            "scalar_buoyancy_coefficient": "buoyancy_coeff_scalar",
        },
    ],
)
def test_roms_ab3_timestepper_buoyancy_rejects_multiple_coefficient_selectors(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
) -> None:
    _install_constructor_spies(monkeypatch, patch_buoyancy=False)

    with pytest.raises(ValueError, match="Exactly one coefficient"):
        roms_models.roms_ab3_timestepper(hc=5.0, NZ=4, **kwargs)  # type: ignore[call-arg]


# --- end-to-end initialisation behaviour (issue #133), exercising the real kernels ---

_HC, _NZ, _Y, _X = 5.0, 4, 4, 4
_H_VALUE, _ZETA_VALUE = 50.0, 0.5
# a zidx=1.5, z=-24.75 pair for _HC/_NZ/_H_VALUE/_ZETA_VALUE below, computed with the real
# compute_z/compute_zidx functions (offline_particles.kernels.roms._vertical_coordinate); this
# choice of C makes C == sigma, so the S-coordinate transform is exactly invertible.
_ZIDX, _Z = 1.5, -24.75


def _make_vertical_coordinate_fieldset() -> Fieldset:
    C = (np.arange(_NZ, dtype=np.float64) + 0.5) / _NZ - 1.0
    fieldset = Fieldset(1, _NZ, _Y, _X)
    fieldset.add_field(
        "h", StaticField.from_numpy(np.full((_Y, _X), _H_VALUE), axes=("Y", "X"), staggers=("center", "center"))
    )
    fieldset.add_field(
        "zeta",
        StaticField.from_numpy(np.full((_Y, _X), _ZETA_VALUE), axes=("Y", "X"), staggers=("center", "center")),
    )
    fieldset.add_field("C", StaticField.from_numpy(C, axes=("Z",), staggers=("center",)))
    return fieldset


def test_roms_ab3_timestepper_run_initialisation_computes_zidx_from_z(make_clock) -> None:
    launcher = Launcher(_make_vertical_coordinate_fieldset(), history_size=1)

    particles = Particles(2, {"z": np.dtype(np.float64)})
    particles["status"][:] = np.array([Status.INITIALISING, Status.NORMAL], dtype=np.uint8)
    particles["yidx"][:] = 1.5
    particles["xidx"][:] = 1.5
    particles["z"][:] = _Z
    particles["zidx"][:] = np.array([-999.0, 7.0])  # sentinel vs. an already-correct value

    timestepper = roms_models.roms_ab3_timestepper(hc=_HC, NZ=_NZ)  # initialise_z=False (default)
    clock = make_clock(np.array([0.0, 1.0, 2.0], dtype=np.float64), 1.0)

    timestepper.run_initialisation(particles, launcher, clock)

    assert particles["zidx"][0] == pytest.approx(_ZIDX)
    assert particles["status"][0] == np.uint8(Status.MULTISTEP_2)
    # the already-active particle is untouched by the initialisation kernel
    assert particles["zidx"][1] == 7.0
    assert particles["status"][1] == np.uint8(Status.NORMAL)


def test_roms_ab3_timestepper_initialise_z_true_computes_z_from_zidx(make_clock) -> None:
    launcher = Launcher(_make_vertical_coordinate_fieldset(), history_size=1)

    particles = Particles(2, {"z": np.dtype(np.float64)})
    particles["status"][:] = np.array([Status.INITIALISING, Status.NORMAL], dtype=np.uint8)
    particles["yidx"][:] = 1.5
    particles["xidx"][:] = 1.5
    particles["zidx"][:] = _ZIDX
    particles["z"][:] = np.array([-999.0, 7.0])  # sentinel vs. an already-correct value

    timestepper = roms_models.roms_ab3_timestepper(hc=_HC, NZ=_NZ, initialise_z=True)
    clock = make_clock(np.array([0.0, 1.0, 2.0], dtype=np.float64), 1.0)

    timestepper.run_initialisation(particles, launcher, clock)

    assert particles["z"][0] == pytest.approx(_Z)
    assert particles["status"][0] == np.uint8(Status.MULTISTEP_2)
    # the already-active particle is untouched by the initialisation kernel
    assert particles["z"][1] == 7.0
    assert particles["status"][1] == np.uint8(Status.NORMAL)
