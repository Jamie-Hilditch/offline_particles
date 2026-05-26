"""Tests for the ROMS AB3 timestepper constructor."""

from __future__ import annotations

import pytest

import offline_particles.models.roms as roms_models
import offline_particles.timestepping as timestepping_module
from offline_particles.timestepping import ABTimestepper


def _install_constructor_spies(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    calls: dict[str, dict[str, object]] = {}
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
        "validation": object(),
        "ab_initialisation": object(),
        "ab_bump_status": object(),
    }

    def stub(name: str, return_value: object):
        def _call(*args, **kwargs):
            calls[name] = {"args": args, "kwargs": kwargs}
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
    monkeypatch.setattr(
        roms_models,
        "construct_linear_damping_kernel",
        stub("linear_damping", sentinels["linear_damping"]),
        raising=True,
    )
    monkeypatch.setattr(
        roms_models,
        "construct_quadratic_damping_kernel",
        stub("quadratic_damping", sentinels["quadratic_damping"]),
        raising=True,
    )
    monkeypatch.setattr(
        roms_models,
        "construct_buoyancy_force_accumulation_kernel",
        stub("buoyancy", sentinels["buoyancy"]),
        raising=True,
    )
    monkeypatch.setattr(
        roms_models, "construct_add_property_kernel", stub("add_property", sentinels["add_property"]), raising=True
    )
    monkeypatch.setattr(roms_models, "construct_ab3_update_kernel", ab3_update_stub, raising=True)
    monkeypatch.setattr(
        timestepping_module,
        "construct_ab_initialisation_kernel",
        stub("ab_initialisation", sentinels["ab_initialisation"]),
        raising=True,
    )
    monkeypatch.setattr(
        timestepping_module,
        "construct_ab_bump_status_kernel",
        stub("ab_bump_status", sentinels["ab_bump_status"]),
        raising=True,
    )
    monkeypatch.setattr(
        roms_models, "construct_compute_zidx_kernel", stub("compute_zidx", sentinels["compute_zidx"]), raising=True
    )
    monkeypatch.setattr(
        roms_models, "construct_validation_kernel", stub("validation", sentinels["validation"]), raising=True
    )

    return calls, sentinels


def test_roms_ab3_timestepper_default_constructor_wires_expected_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, sentinels = _install_constructor_spies(monkeypatch)

    timestepper = roms_models.roms_ab3_timestepper()

    assert isinstance(timestepper, ABTimestepper)
    assert timestepper._order == 3
    assert timestepper.index_padding == 5

    assert calls["x_adv"]["args"] == ("_dxidx0", "u", "pm", ("Z", "Y", "X"), ("Y", "X"))
    assert calls["x_adv"]["kwargs"] == {"metric": True}
    assert calls["y_adv"]["args"] == ("_dyidx0", "v", "pn", ("Z", "Y", "X"), ("Y", "X"))
    assert calls["y_adv"]["kwargs"] == {"metric": True}
    assert calls["z_adv"]["args"] == ("_dz0", "w")
    assert calls["z_adv"]["kwargs"] == {"accumulate": True}
    assert calls["validation"]["args"] == ()
    assert calls["compute_zidx"]["args"] == ()
    assert calls["ab_initialisation"]["args"] == (3,)
    assert calls["ab_bump_status"]["args"] == ()
    assert "linear_damping" not in calls
    assert "quadratic_damping" not in calls
    assert "buoyancy" not in calls
    assert "add_property" not in calls

    assert timestepper.initialisation_kernels == [sentinels["ab_initialisation"]]
    assert timestepper.pre_step_kernels == [sentinels["validation"]]
    assert timestepper.post_step_kernels == [sentinels["compute_zidx"]]
    assert timestepper._tendency_kernels == [sentinels["x_adv"], sentinels["y_adv"], sentinels["z_adv"]]
    assert timestepper._ab_update_kernels == [
        sentinels["ab_update_x"],
        sentinels["ab_update_y"],
        sentinels["ab_update_z"],
    ]


def test_roms_ab3_timestepper_with_buoyancy_and_damping_wires_optional_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, sentinels = _install_constructor_spies(monkeypatch)

    timestepper = roms_models.roms_ab3_timestepper(
        vertical_velocity=False,
        buoyant_particles=True,
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
        hc="hc_scalar",
        NZ="nz_scalar",
        g="gravity",
        rho0="rho0_scalar",
        constant_linear_damping_coefficient=0.25,
        property_quadratic_damping_coefficient="quad_drag",
    )

    assert isinstance(timestepper, ABTimestepper)
    assert timestepper._order == 3
    assert timestepper.index_padding == 9

    assert calls["x_adv"]["args"] == ("_dxidx0", "u_field", "pm_field", ("Z", "Y", "X"), ("Y", "X"))
    assert calls["y_adv"]["args"] == ("_dyidx0", "v_field", "pn_field", ("Z", "Y", "X"), ("Y", "X"))
    assert "z_adv" not in calls

    assert calls["linear_damping"]["args"] == ("w_rel", "_dw_rel0")
    assert calls["linear_damping"]["kwargs"] == {
        "constant_coefficient": 0.25,
        "property_coefficient": None,
        "scalar_coefficient": None,
    }
    assert calls["quadratic_damping"]["args"] == ("w_rel", "_dw_rel0")
    assert calls["quadratic_damping"]["kwargs"] == {
        "constant_coefficient": None,
        "property_coefficient": "quad_drag",
        "scalar_coefficient": None,
    }
    assert calls["buoyancy"]["args"] == ("_dw_rel0",)
    assert calls["buoyancy"]["kwargs"] == {
        "density_field": "density",
        "reference_density": "rho0_scalar",
        "gravity": "gravity",
    }
    assert calls["add_property"]["args"] == ("w_rel", "_dz0")
    assert calls["add_property"]["kwargs"] == {}
    assert calls["ab_update"]["args"] == ("w_rel", "_dw_rel0", "_dw_rel1", "_dw_rel2")
    assert calls["ab_initialisation"]["args"] == (3,)
    assert calls["ab_bump_status"]["args"] == ()

    assert timestepper.initialisation_kernels == [sentinels["ab_initialisation"]]
    assert timestepper.pre_step_kernels == [sentinels["validation"]]
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
