"""Tests for Adams-Bashforth timestepping kernel constructors."""

import numpy as np
import pytest

from offline_particles.kernels.status import Status
from offline_particles.kernels.timestepping import (
    construct_ab2_update_kernel,
    construct_ab3_update_kernel,
    construct_ab_bump_status_kernel,
    construct_ab_initialisation_kernel,
)


class TestConstructAB2UpdateKernel:
    def test_bindings_use_provided_property_names(self) -> None:
        kernel = construct_ab2_update_kernel("x", "x_d0", "x_d1")

        assert kernel.particle_property_bindings["prop"] == "x"
        assert kernel.particle_property_bindings["dprop_0"] == "x_d0"
        assert kernel.particle_property_bindings["dprop_1"] == "x_d1"

    def test_applies_ab2_update_and_shifts_tendencies(self) -> None:
        kernel = construct_ab2_update_kernel("x", "x_d0", "x_d1")

        status = np.array(
            [
                np.uint8(Status.MULTISTEP_1),
                np.uint8(Status.NORMAL),
                np.uint8(Status.INACTIVE),
            ],
            dtype=np.uint8,
        )
        prop = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        dprop_0 = np.array([2.0, 4.0, 8.0], dtype=np.float32)
        dprop_1 = np.array([0.5, 1.0, 1.0], dtype=np.float32)

        kernel.kernel(
            {
                "status": status,
                "prop": prop,
                "dprop_0": dprop_0,
                "dprop_1": dprop_1,
            },
            {"_dt": np.float64(0.1)},
            {},
        )

        np.testing.assert_allclose(prop, np.array([1.2, 2.55, 3.0], dtype=np.float32), rtol=1e-6)
        np.testing.assert_allclose(dprop_0, np.array([0.0, 0.0, 8.0], dtype=np.float32), rtol=1e-6)
        np.testing.assert_allclose(dprop_1, np.array([2.0, 4.0, 1.0], dtype=np.float32), rtol=1e-6)


class TestConstructAB3UpdateKernel:
    def test_bindings_use_provided_property_names(self) -> None:
        kernel = construct_ab3_update_kernel("x", "x_d0", "x_d1", "x_d2")

        assert kernel.particle_property_bindings["prop"] == "x"
        assert kernel.particle_property_bindings["dprop_0"] == "x_d0"
        assert kernel.particle_property_bindings["dprop_1"] == "x_d1"
        assert kernel.particle_property_bindings["dprop_2"] == "x_d2"

    def test_applies_ab3_update_and_shifts_tendencies(self) -> None:
        kernel = construct_ab3_update_kernel("x", "x_d0", "x_d1", "x_d2")

        status = np.array(
            [
                np.uint8(Status.MULTISTEP_1),
                np.uint8(Status.MULTISTEP_2),
                np.uint8(Status.NORMAL),
                np.uint8(Status.INACTIVE),
            ],
            dtype=np.uint8,
        )
        prop = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        dprop_0 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        dprop_1 = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        dprop_2 = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)

        kernel.kernel(
            {
                "status": status,
                "prop": prop,
                "dprop_0": dprop_0,
                "dprop_1": dprop_1,
                "dprop_2": dprop_2,
            },
            {"_dt": np.float64(1.0)},
            {},
        )

        np.testing.assert_allclose(prop, np.array([1.0, -7.0, 90.75, 0.0], dtype=np.float32), rtol=1e-6)
        np.testing.assert_allclose(dprop_0, np.array([0.0, 0.0, 0.0, 4.0], dtype=np.float32), rtol=1e-6)
        np.testing.assert_allclose(dprop_1, np.array([1.0, 2.0, 3.0, 40.0], dtype=np.float32), rtol=1e-6)
        np.testing.assert_allclose(dprop_2, np.array([1.0, 20.0, 30.0, 400.0], dtype=np.float32), rtol=1e-6)


class TestABStatusKernels:
    def test_bump_status_advances_multistep_flags(self) -> None:
        kernel = construct_ab_bump_status_kernel()

        status = np.array(
            [
                np.uint8(Status.MULTISTEP_1),
                np.uint8(Status.MULTISTEP_2),
                np.uint8(Status.NORMAL),
                np.uint8(Status.INACTIVE),
            ],
            dtype=np.uint8,
        )

        kernel.kernel({"status": status}, {}, {})

        np.testing.assert_array_equal(
            status,
            np.array(
                [
                    np.uint8(Status.NORMAL),
                    np.uint8(Status.MULTISTEP_1),
                    np.uint8(Status.NORMAL),
                    np.uint8(Status.INACTIVE),
                ],
                dtype=np.uint8,
            ),
        )

    @pytest.mark.parametrize(
        ("order", "expected_status"),
        [
            (2, np.uint8(Status.MULTISTEP_1)),
            (3, np.uint8(Status.MULTISTEP_2)),
        ],
    )
    def test_initialisation_sets_multistep_status_for_active_particles_only(self, order: int, expected_status: np.uint8) -> None:
        kernel = construct_ab_initialisation_kernel(order)

        status = np.array(
            [
                np.uint8(Status.NORMAL),
                np.uint8(Status.INACTIVE),
                np.uint8(Status.NONFINITE),
            ],
            dtype=np.uint8,
        )

        kernel.kernel({"status": status}, {}, {})

        np.testing.assert_array_equal(
            status,
            np.array(
                [
                    expected_status,
                    np.uint8(Status.INACTIVE),
                    np.uint8(Status.NONFINITE),
                ],
                dtype=np.uint8,
            ),
        )

    def test_initialisation_rejects_unsupported_order(self) -> None:
        with pytest.raises(ValueError, match="Unsupported Adams-Bashforth order"):
            construct_ab_initialisation_kernel(4)
