"""Tests for validation kernel constructors and behaviour."""

import numpy as np
import pytest

from offline_particles.kernels.status import Status
from offline_particles.kernels.validation import (
    construct_validation_kernel,
    construct_validation_kernel_from_bbox,
    finite_indices_kernel,
)
from offline_particles.kernels.validation._domain_bounds import construct_domain_bounds_kernel
from offline_particles.spatial_arrays import BBox


class TestConstructDomainBoundsKernel:
    @pytest.mark.parametrize(
        ("zmin", "zmax", "ymin", "ymax", "xmin", "xmax", "match"),
        [
            (1.0, 0.0, 0.0, 1.0, 0.0, 1.0, "zmin"),
            (0.0, 1.0, 2.0, 1.0, 0.0, 1.0, "ymin"),
            (0.0, 1.0, 0.0, 1.0, 2.0, 1.0, "xmin"),
        ],
    )
    def test_rejects_invalid_bounds(
        self,
        zmin: float,
        zmax: float,
        ymin: float,
        ymax: float,
        xmin: float,
        xmax: float,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            construct_domain_bounds_kernel(zmin, zmax, ymin, ymax, xmin, xmax)

    def test_marks_particles_outside_domain(self) -> None:
        kernel = construct_domain_bounds_kernel(0.0, 1.0, 10.0, 20.0, 100.0, 200.0)

        status = np.array(
            [
                np.uint8(Status.NORMAL),
                np.uint8(Status.NORMAL),
                np.uint8(Status.NORMAL),
                np.uint8(Status.NORMAL),
                np.uint8(Status.INACTIVE),
            ],
            dtype=np.uint8,
        )
        zidx = np.array([0.5, -0.1, 1.1, -0.1, 0.5], dtype=np.float64)
        yidx = np.array([15.0, 15.0, 15.0, 5.0, 5.0], dtype=np.float64)
        xidx = np.array([150.0, 150.0, 150.0, 50.0, 50.0], dtype=np.float64)

        kernel.kernel(
            {
                "status": status,
                "zidx": zidx,
                "yidx": yidx,
                "xidx": xidx,
            },
            {},
            {},
        )

        np.testing.assert_array_equal(
            status,
            np.array(
                [
                    np.uint8(Status.NORMAL),
                    np.uint8(Status.BELOW_BOTTOM),
                    np.uint8(Status.ABOVE_SURFACE),
                    np.uint8(Status.OUT_OF_DOMAIN),
                    np.uint8(Status.INACTIVE),
                ],
                dtype=np.uint8,
            ),
        )


class TestFiniteIndicesKernel:
    def test_marks_nonfinite_indices_and_skips_inactive_particles(self) -> None:
        status = np.array(
            [
                np.uint8(Status.NORMAL),
                np.uint8(Status.NORMAL),
                np.uint8(Status.NORMAL),
                np.uint8(Status.NORMAL),
                np.uint8(Status.INACTIVE),
            ],
            dtype=np.uint8,
        )
        zidx = np.array([0.0, np.nan, 1.0, 1.0, np.nan], dtype=np.float64)
        yidx = np.array([0.0, 1.0, np.inf, 1.0, np.nan], dtype=np.float64)
        xidx = np.array([0.0, 1.0, 2.0, np.nan, np.nan], dtype=np.float64)

        finite_indices_kernel.kernel(
            {
                "status": status,
                "zidx": zidx,
                "yidx": yidx,
                "xidx": xidx,
            },
            {},
            {},
        )

        np.testing.assert_array_equal(
            status,
            np.array(
                [
                    np.uint8(Status.NORMAL),
                    np.uint8(Status.NONFINITE),
                    np.uint8(Status.NONFINITE),
                    np.uint8(Status.NONFINITE),
                    np.uint8(Status.INACTIVE),
                ],
                dtype=np.uint8,
            ),
        )


class TestConstructValidationKernel:
    def test_combines_finite_and_domain_checks(self) -> None:
        kernel = construct_validation_kernel(0.0, 1.0, 10.0, 20.0, 100.0, 200.0)

        status = np.array(
            [
                np.uint8(Status.NORMAL),
                np.uint8(Status.NORMAL),
                np.uint8(Status.NORMAL),
                np.uint8(Status.INACTIVE),
            ],
            dtype=np.uint8,
        )
        zidx = np.array([0.5, np.nan, -0.1, np.nan], dtype=np.float64)
        yidx = np.array([15.0, 15.0, 5.0, np.nan], dtype=np.float64)
        xidx = np.array([150.0, 150.0, 50.0, np.nan], dtype=np.float64)

        kernel.kernel(
            {
                "status": status,
                "zidx": zidx,
                "yidx": yidx,
                "xidx": xidx,
            },
            {},
            {},
        )

        np.testing.assert_array_equal(
            status,
            np.array(
                [
                    np.uint8(Status.NORMAL),
                    np.uint8(Status.NONFINITE),
                    np.uint8(Status.OUT_OF_DOMAIN),
                    np.uint8(Status.INACTIVE),
                ],
                dtype=np.uint8,
            ),
        )


class TestConstructValidationKernelFromBBox:
    def test_uses_bbox_bounds(self) -> None:
        bbox = BBox(zmin=2.0, zmax=4.0, ymin=10.0, ymax=20.0, xmin=30.0, xmax=40.0)
        kernel = construct_validation_kernel_from_bbox(bbox)

        status = np.array([np.uint8(Status.NORMAL), np.uint8(Status.NORMAL)], dtype=np.uint8)
        zidx = np.array([3.0, 1.5], dtype=np.float64)
        yidx = np.array([15.0, 15.0], dtype=np.float64)
        xidx = np.array([35.0, 35.0], dtype=np.float64)

        kernel.kernel(
            {
                "status": status,
                "zidx": zidx,
                "yidx": yidx,
                "xidx": xidx,
            },
            {},
            {},
        )

        np.testing.assert_array_equal(
            status,
            np.array(
                [np.uint8(Status.NORMAL), np.uint8(Status.BELOW_BOTTOM)],
                dtype=np.uint8,
            ),
        )
