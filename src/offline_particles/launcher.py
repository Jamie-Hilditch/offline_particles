"""Submodule for particle kernel launchers."""

import collections
import logging
from collections.abc import Callable
from typing import TypeVar

import numba
import numpy as np
import numpy.typing as npt

from .fields import FieldData
from .fieldset import Fieldset
from .kernels import BoundKernel
from .kernels.status import INACTIVE_FLAG, Status
from .particles import Particles
from .spatial_arrays import BBox

__all__ = [
    "Launcher",
    "ScalarProvider",
    "ScalarSource",
    "Tinfo",
]

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=np.generic)
_INITIALISING = np.uint8(Status.INITIALISING)

# Named tuple for time information
Time_info = collections.namedtuple("Time_info", ["time", "tidx", "iteration"])
type Tinfo = Time_info[np.float64 | np.timedelta64, np.float64, int]


class ScalarSource[T]:
    """Descriptor declaring a scalar data source.

    The getter must have signature:
        (self, tinfo: Tinfo) -> np.generic
    """

    def __init__(
        self,
        name: str,
        getter: Callable[[object, Tinfo], T],
    ) -> None:
        self.name = name
        self._getter = getter

    def __get__(self, obj: object | None, owner: type | None = None):
        # Accessed on the class → return descriptor for discovery
        if obj is None:
            return self

        # Accessed on an instance → return bound callable
        def scalar_func(tinfo: Tinfo) -> T:
            return self._getter(obj, tinfo)

        return scalar_func


type ScalarProvider = Callable[[Tinfo], np.generic]

# -------------------------------
# Kernel Launcher
# -------------------------------


class Launcher:
    """Class to launch bound kernels."""

    def __init__(self, fieldset: Fieldset, history_size: int) -> None:
        self._scalar_data_sources: dict[str, ScalarProvider] = {}
        self._fieldset = fieldset
        self._index_padding = 0

        # bbox cache
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        self._zmin_history = collections.deque(maxlen=history_size)
        self._zmax_history = collections.deque(maxlen=history_size)
        self._ymin_history = collections.deque(maxlen=history_size)
        self._ymax_history = collections.deque(maxlen=history_size)
        self._xmin_history = collections.deque(maxlen=history_size)
        self._xmax_history = collections.deque(maxlen=history_size)

        # register constants attached to fieldset as scalar data sources
        for name, value in self._fieldset.constants.items():
            value_func = self.create_value_scalar_source(value)
            self.register_scalar_data_source(name, value_func)

    @staticmethod
    def create_value_scalar_source(value: np.generic) -> ScalarProvider:
        """Create a scalar data source that always returns the given value.

        Parameters
        ----------
        value : np.generic
            The constant value to return.

        Returns
        -------
        ScalarProvider
            A function that takes a Tinfo and returns the constant value.
        """

        def value_func(tinfo: Tinfo) -> np.generic:
            return value

        return value_func

    def register_scalar_data_source(self, name: str, source: ScalarProvider) -> None:
        """Register a scalar data source function.

        Parameters
        ----------
        name : str
            The name of the scalar data source.
        source : ScalarProvider
            A callable that takes a Tinfo and returns a scalar value.

        Raises
        ------
        ValueError
            If a scalar data source with the same name is already registered or if the name conflicts with a field in the fieldset.
        """
        if name in self._scalar_data_sources:
            raise ValueError(
                f"Scalar data source '{name}' is already registered. Deregister it before registering a new one."
            )
        if name in self._fieldset.fields:
            raise ValueError(f"Scalar data source '{name}' conflicts with a field in the fieldset.")

        self._scalar_data_sources[name] = source

    def deregister_scalar_data_source(self, name: str) -> None:
        """Deregister a scalar data source function.

        Parameters
        ----------
        name : str
            The name of the scalar data source to deregister.

        Raises
        ------
        ValueError
            If the scalar data source is not registered.
        """
        if name not in self._scalar_data_sources:
            raise ValueError(f"Scalar data source '{name}' is not registered.")
        del self._scalar_data_sources[name]

    def register_scalar_data_sources_from_object(self, obj: object):
        """Scan object for scalar data source functions and register them."""
        for attr_name in dir(type(obj)):
            attr = getattr(type(obj), attr_name)
            if isinstance(attr, ScalarSource):
                scalar_func = getattr(obj, attr_name)
                self.register_scalar_data_source(attr.name, scalar_func)

    @property
    def index_padding(self) -> int:
        """The index padding used by this launcher."""
        return self._index_padding

    def set_index_padding(self, index_padding: int, force: bool = False) -> None:
        """Set the index padding used by this launcher.

        Parameters
        ----------
        index_padding : int
            The new index padding to set. Must be non-negative.
        force : bool, optional
            If True, forcefully set the index padding to the given value.
            If False, increase the index padding if the new value is greater than the current value
            else leave it unchanged. Default is False.

        Raises
        ------
        ValueError
            If index_padding is negative.
        """
        if index_padding < 0:
            raise ValueError("Index padding must be non-negative.")
        if force or index_padding > self._index_padding:
            self._index_padding = index_padding

    def construct_bbox(
        self,
        particles: Particles,
    ) -> BBox | None:
        """Construct a bounding box around the given particles with index padding.

        Parameters
        ----------
        particles : Particles
            The particles to compute the bounding box for.

        Returns
        -------
        BBox | None
            The computed bounding box with index padding applied, or ``None`` if there
            are no active or initialising particles. The bbox history is left untouched
            in that case, since a call with no particles to process shouldn't count
            against the rolling smoothing window.
        """
        # compute bounds of active and initialising particles
        zmin, zmax, ymin, ymax, xmin, xmax, any_active = _compute_particle_bounds(
            particles["status"],
            particles["zidx"],
            particles["yidx"],
            particles["xidx"],
        )
        if not any_active:
            return None

        # update history
        self._zmin_history.append(zmin)
        self._zmax_history.append(zmax)
        self._ymin_history.append(ymin)
        self._ymax_history.append(ymax)
        self._xmin_history.append(xmin)
        self._xmax_history.append(xmax)

        return BBox(
            zmin=min(self._zmin_history) - self._index_padding,
            zmax=max(self._zmax_history) + self._index_padding,
            ymin=min(self._ymin_history) - self._index_padding,
            ymax=max(self._ymax_history) + self._index_padding,
            xmin=min(self._xmin_history) - self._index_padding,
            xmax=max(self._xmax_history) + self._index_padding,
        )

    def get_field_data(self, name: str, time_index: float, bbox: BBox) -> FieldData:
        """Get the field data at a given time index.

        Parameters
        ----------
        time_index : float
            Time index.
        bbox : BBox
            Bounding box to extract data from defined in terms of centered grid indices.

        Returns
        -------
        FieldData
            Tuple containing the field data array and offsets.
        """
        return self._fieldset[name].get_field_data(time_index, bbox)

    def launch_kernel(self, bound_kernel: BoundKernel, particles: Particles, tinfo: Tinfo) -> None:
        """Launch a kernel."""
        particle_properties = {
            name: particles[binding] for name, binding in bound_kernel.particle_property_bindings.items()
        }
        scalars = {
            name: self._scalar_data_sources[binding](tinfo) for name, binding in bound_kernel.scalar_bindings.items()
        }

        # only construct the bbox and fetch field data if the kernel actually needs it - this
        # also skips the kernel launch if there are no active particles to build a bbox around
        field_data: dict[str, FieldData] = {}
        if bound_kernel.field_data_bindings:
            bbox = self.construct_bbox(particles)
            if bbox is None:
                logger.debug("launch_kernel: skipping %r - no active particles", bound_kernel)
                return
            field_data = {
                name: self.get_field_data(binding, tinfo.tidx, bbox)
                for name, binding in bound_kernel.field_data_bindings.items()
            }

        # call the kernel
        bound_kernel.kernel(particle_properties, scalars, field_data)


@numba.njit(fastmath=True, nogil=True)
def _compute_particle_bounds(
    status: npt.NDArray[np.uint8],
    zidx: npt.NDArray[np.float64],
    yidx: npt.NDArray[np.float64],
    xidx: npt.NDArray[np.float64],
) -> tuple[float, float, float, float, float, float, bool]:
    """Compute the bounding box of active particles.

    Particles with status ``Status.INITIALISING`` are included despite carrying the ``INACTIVE``
    bit: they're processed by initialisation kernels in the same
    :meth:`~offline_particles.timestepping.Timestepper.run_initialisation` call (e.g. a
    ROMS consistency-fix kernel interpolating field data at the particle's location), so the
    bounding box used to gather that field data must cover them.

    Parameters
    ----------
    status : npt.NDArray[np.uint8]
        Array of particle statuses.
    zidx : npt.NDArray[np.float64]
        Array of particle z indices.
    yidx : npt.NDArray[np.float64]
        Array of particle y indices.
    xidx : npt.NDArray[np.float64]
        Array of particle x indices.

    Returns
    -------
    tuple[float, float, float, float, float, float, bool]
        Bounding box of active particles in the form (zmin, zmax, ymin, ymax, xmin, xmax),
        followed by whether any particle was included in that bounding box. If no particle
        was included, the bounds are left at their ``+inf``/``-inf`` sentinel values.
    """
    zmin = np.inf
    zmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    xmin = np.inf
    xmax = -np.inf
    any_active = False

    for i in range(status.size):
        if (status[i] & INACTIVE_FLAG) and status[i] != _INITIALISING:  # inactive, but not initialising
            continue

        any_active = True
        z = zidx[i]
        y = yidx[i]
        x = xidx[i]

        zmin = min(zmin, z)
        zmax = max(zmax, z)
        ymin = min(ymin, y)
        ymax = max(ymax, y)
        xmin = min(xmin, x)
        xmax = max(xmax, x)

    return zmin, zmax, ymin, ymax, xmin, xmax, any_active
