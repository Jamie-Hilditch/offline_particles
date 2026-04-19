"""Offline line particle advection.

The following classes and submodules are reexported at the top level of the package:

Submodules
~~~~~~~~~~

- :py:mod:`~offline_particles.kernels`: Kernels for implementing particle behaviour.
- :py:mod:`~offline_particles.output`: Output tools for saving simulation results.
- :py:mod:`~offline_particles.models.roms`: Tools for working with ROMS output.

Classes
~~~~~~~

- :py:class:`~offline_particles.events.Event`: Class for defining an event that occurs during a simulation.
- :py:class:`~offline_particles.events.SimulationState`: The state of a simulation at a given time.
- :py:class:`~offline_particles.fields.StaticField`: A field that does not change over time.
- :py:class:`~offline_particles.fields.TimeDependentField`: A field that changes over time.
- :py:class:`~offline_particles.fieldset.Fieldset`: A collection of fields and constants for use in a simulation.
- :py:class:`~offline_particles.kernels.BoundKernel`: A kernel with name bindings.
- :py:class:`~offline_particles.kernels.ParticleKernel`: A kernel implementing particle behaviour.
- :py:class:`~offline_particles.kernels.status.Status`: An enumeration of possible particle statuses.
- :py:class:`~offline_particles.output.Output`: Class for defining an output variable.
- :py:class:`~offline_particles.output.ZarrOutputBuilder`: An output builder for writing simulation results to a Zarr store.
- :py:class:`~offline_particles.simulation.ParticleSet`: A collection of particles.
- :py:class:`~offline_particles.simulation.Simulation`: Central class for managing a particle advection simulation.
- :py:class:`~offline_particles.simulation.SimulationBuilder`: A builder for constructing a Simulation.
- :py:class:`~offline_particles.timestepping.ABTimestepper`: A timestepper implementing an Adams-Bashforth method.
- :py:class:`~offline_particles.timestepping.Clock`: A clock for keeping track of simulation time.
- :py:class:`~offline_particles.timestepping.RK2Timestepper`: A timestepper implementing an explicit second-order Runge-Kutta method.
- :py:class:`~offline_particles.timestepping.Timestepper`: Base class for defining timesteppers.
"""

from . import kernels, output
from .events import Event, SimulationState
from .fields import StaticField, TimeDependentField
from .fieldset import Fieldset
from .kernels import BoundKernel, ParticleKernel, Status
from .models import roms
from .output import Output, ZarrOutputBuilder
from .simulation import ParticleSet, Simulation, SimulationBuilder
from .timestepping import ABTimestepper, Clock, RK2Timestepper, Timestepper

__all__ = [
    "kernels",
    "output",
    "Event",
    "SimulationState",
    "StaticField",
    "TimeDependentField",
    "Fieldset",
    "BoundKernel",
    "ParticleKernel",
    "Status",
    "roms",
    "Output",
    "ZarrOutputBuilder",
    "ParticleSet",
    "Simulation",
    "SimulationBuilder",
    "ABTimestepper",
    "Clock",
    "RK2Timestepper",
    "Timestepper",
]
