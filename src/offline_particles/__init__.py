"""Offline line particle advection.

Top level exports
=================

The following classes and submodules are reexported at the top level of the package:


Submodules
~~~~~~~~~~

.. autosummary::
   :nosignatures:

   kernels
   output
   ~models.roms

Classes
~~~~~~~

.. autosummary::
   :nosignatures:

   ~events.Event
   ~events.SimulationState
   ~fields.StaticField
   ~fields.TimeDependentField
   ~fieldset.Fieldset
   ~kernels.BoundKernel
   ~kernels.ParticleKernel
   ~kernels.status.Status
   ~output.Output
   ~output.ZarrOutputBuilder
   ~simulation.ParticleSet
   ~simulation.Simulation
   ~simulation.SimulationBuilder
   ~spatial_arrays.ArrayAxis
   ~spatial_arrays.ArrayLayout
   ~spatial_arrays.Stagger
   ~timestepping.ABTimestepper
   ~timestepping.Clock
   ~timestepping.RK2Timestepper
   ~timestepping.Timestepper
"""

from . import kernels, output
from .events import Event, SimulationState
from .fields import StaticField, TimeDependentField
from .fieldset import Fieldset
from .kernels import BoundKernel, ParticleKernel, Status
from .models import roms
from .output import Output, ZarrOutputBuilder
from .simulation import ParticleSet, Simulation, SimulationBuilder
from .spatial_arrays import ArrayAxis, ArrayLayout, Stagger
from .timestepping import ABTimestepper, Clock, RK2Timestepper, Timestepper

__all__ = [
    "ABTimestepper",
    "ArrayAxis",
    "ArrayLayout",
    "BoundKernel",
    "Clock",
    "Event",
    "Fieldset",
    "Output",
    "ParticleKernel",
    "ParticleSet",
    "RK2Timestepper",
    "Simulation",
    "SimulationBuilder",
    "SimulationState",
    "Stagger",
    "StaticField",
    "Status",
    "TimeDependentField",
    "Timestepper",
    "ZarrOutputBuilder",
    "kernels",
    "output",
    "roms",
]
