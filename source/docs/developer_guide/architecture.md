# Architecture Overview

This document provides a high-level overview of the TorchSim architecture, explaining how the different components interact to create a molecular dynamics simulation framework.

## Core Components

TorchSim is built around several key components that work together:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  AtomReader │────▶│  MDModel    │────▶│ MDSimulator │
└─────────────┘     └─────────────┘     └─────────────┘
                         ▲   ▲
                         │   │
               ┌─────────┘   └─────────┐
               │                       │
        ┌─────────────┐         ┌─────────────┐
        │ ForceFields │         │ Integrators │
        └─────────────┘         └─────────────┘
```

### AtomReader

The `AtomFileReader` class is responsible for loading molecular structures from files and preparing the data for simulation. It handles:
- Reading atomic positions and types
- Setting up the simulation box
- Providing data structures for forces and integrators

### Force Fields

Force fields calculate the potential energy and forces between atoms. TorchSim provides:
- `LennardJonesForce`: Classic Lennard-Jones potential
- `MachineLearningForce`: Neural network-based potentials
- Custom force fields through the `BaseForce` interface

### Integrators

Integrators update atomic positions and velocities according to the equations of motion. TorchSim includes:
- `VerletIntegrator`: Velocity Verlet algorithm with NVE/NVT support
- Custom integrators through the `BaseIntegrator` interface

### MDModel

The `BaseModel` class combines force fields and integrators into a coherent simulation model. It:
- Manages the flow of data between components
- Provides a unified interface for the simulator

### MDSimulator

The `MDSimulator` class controls the execution of simulations and collects results. It:
- Runs the simulation for a specified number of steps
- Collects and saves trajectory and energy data
- Provides visualization and analysis tools

## Data Flow

1. `AtomFileReader` loads atomic positions and types
2. `ForceField` components calculate forces based on positions
3. `Integrator` updates positions and velocities using forces
4. `MDModel` coordinates these operations for each time step
5. `MDSimulator` manages the overall simulation and collects results

## Extension Points

TorchSim is designed to be easily extended:

1. **Custom Force Fields**: Extend `BaseForce` and implement `compute_energy` and `compute_forces`
2. **Custom Integrators**: Extend `BaseIntegrator` and implement `step`
3. **Machine Learning Models**: Create new neural network architectures in `machine_learning_potentials/model`
4. **Analysis Tools**: Add new analysis functions to `MDSimulator` or create standalone tools

## PyTorch Integration

TorchSim leverages PyTorch for:
- GPU acceleration through CUDA
- Automatic differentiation for force calculation
- Neural network models for machine learning potentials
- Tensor operations for efficient computation

This design allows TorchSim to combine traditional molecular dynamics with modern machine learning approaches in a unified framework. 