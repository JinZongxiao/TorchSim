# API Reference

Welcome to the TorchSim API Reference. This section provides detailed documentation for all classes, methods, and functions in the TorchSim package.

## Modules

- [core](core/index.md): Core molecular dynamics engine
  - [force](core/force.md): Force field implementations
  - [integrator](core/integrator.md): Integrator implementations
  - [neighbor_search](core/neighbor_search.md): Neighbor searching algorithms
  - [md_model](core/md_model.md): Model definition
  - [md_simulation](core/md_simulation.md): Simulation management
  - [energy_minimizer](core/energy_minimizer.md): Energy minimization tools
  - [parameter_manager](core/parameter_manager.md): Parameter handling utilities

- [io_utils](io_utils.md): Input/Output utilities
  - [reader](io_utils/reader.md): File readers for molecular structures
  - [writer](io_utils/writer.md): File writers for simulation outputs
  - [file_converter](io_utils/file_converter.md): File format converters

- [machine_learning_potentials](ml_potentials/index.md): Machine learning potentials
  - [machine_learning_force](ml_potentials/ml_force.md): ML force field implementation
  - [model](ml_potentials/model.md): Neural network model architectures

- [graph_diffusion](graph_diffusion/index.md): Graph-based diffusion models

## Core Classes

For quick reference, here are the most commonly used classes in TorchSim:

- `AtomFileReader`: Reads molecular structure files and prepares data for simulation
- `LennardJonesForce`: Implements the Lennard-Jones potential for atomic interactions
- `VerletIntegrator`: Implements the velocity Verlet integration algorithm
- `BaseModel`: Base class for molecular dynamics models
- `MDSimulator`: Main simulation controller class
- `MachineLearningForce`: Implements machine learning potentials for simulations 