"""Submodules defining kernel data and kernel data sources."""

import collections
from typing import Callable, Iterable

from .spatial_arrays import BBox

"""The KernelData type for passing data into kernels."""
KernelData = collections.namedtuple("KernelData", ("array", "dmask", "offsets"))

type KernelDataFunction = Callable[[float, BBox], KernelData]


class KernelDataSource:
    """Base class for sources of KernelData."""

    def __init_subclass__(cls):
        super().__init_subclass__()

        # Inherit parent declarations
        inherited = {}
        for base in cls.__mro__[1:]:
            inherited.update(getattr(base, "__kernel_data_declarations__", {}))

        # Merge child declarations (override parent if duplicate)
        declared = getattr(cls, "__kernel_data_declarations__", {})
        inherited.update(declared)

        cls.__kernel_data_declarations__ = inherited

    def __init__(self):
        # Instance-level resolved functions
        self._kernel_data_sources = {}

        # Bind class-level declarations into instance
        for name, method_name in self.__kernel_data_declarations__.items():
            method = getattr(self, method_name)
            self.register_kernel_data_function(name, method)

    def kernel_data_keys(self) -> Iterable[str]:
        """Get the keys of the KernelData sources registered with this source."""
        return self._kernel_data_sources.keys()

    def kernel_data_values(self) -> Iterable[KernelDataFunction]:
        """Get the KernelData functions registered with this source."""
        return self._kernel_data_sources.values()

    def kernel_data_items(self) -> Iterable[tuple[str, KernelDataFunction]]:
        """Get the (key, KernelDataFunction) pairs registered with this source."""
        return self._kernel_data_sources.items()

    def register_kernel_data_function(
        self, name: str, func: KernelDataFunction
    ) -> None:
        """Register a KernelDataFunction with this source.

        Parameters
        ----------
        name : str
            Name to register the function under.
        func : KernelDataFunction
            The function to register.
        """
        if name in self._kernel_data_sources:
            raise ValueError(
                f"Kernel data source {name} already registered. Please deregister first."
            )
        self._kernel_data_sources[name] = func

    def deregister_kernel_data_function(self, name: str) -> None:
        """Deregister a KernelDataFunction from this source.

        Parameters
        ----------
        name : str
            Name of the function to deregister.
        """
        if name not in self._kernel_data_sources:
            raise ValueError(f"Kernel data source {name} not registered.")
        del self._kernel_data_sources[name]


# -------------------------------
# KernelData descriptor
# -------------------------------


class register_kernel_data:
    """
    Decorator for marking a method as a kernel-data provider.
    Registration is handled in KernelDataSource.__init_subclass__.
    """

    def __init__(self, name: str):
        self.name = name
        self.method_name = None

    def __set_name__(self, owner, method_name: str):
        self.method_name = method_name
        # Record the declaration; actual registration is done in __init_subclass__
        if not hasattr(owner, "__kernel_data_declarations__"):
            owner.__kernel_data_declarations__ = {}
        owner.__kernel_data_declarations__[self.name] = method_name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        # Return bound method
        return getattr(instance, self.method_name)
