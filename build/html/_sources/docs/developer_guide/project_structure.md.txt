# Project Structure

This document describes the overall structure of the TorchSim codebase, to help you understand how the different components are organized and how to add your own extensions.

## Directory Structure

```
torchsim/
├── core/                        # Core molecular dynamics engine
│   ├── force/                   # Force field implementations
│   │   ├── __init__.py
│   │   ├── base_force.py        # Base class for all force fields
│   │   ├── lennard_jones_force.py
│   │   └── ...
│   ├── integrator/              # Integrator implementations
│   │   ├── __init__.py
│   │   ├── base_integrator.py   # Base class for all integrators
│   │   ├── integrator.py        # Verlet integrator implementation
│   │   └── ...
│   ├── neighbor_search/         # Neighbor search algorithms
│   │   ├── __init__.py
│   │   ├── neighbor_list.py     # Neighbor list implementation
│   │   └── ...
│   ├── __init__.py
│   ├── md_model.py              # Model definitions
│   ├── md_simulation.py         # Simulation controller
│   ├── energy_minimizer.py      # Energy minimization tools
│   ├── parameter_manager.py     # Parameter handling utilities
│   └── element_info.py          # Element properties
│
├── io_utils/                    # Input/Output utilities
│   ├── __init__.py
│   ├── reader.py                # Structure file readers
│   ├── writer.py                # Output file writers
│   ├── file_converter.py        # File format converters
│   └── ...
│
├── machine_learning_potentials/ # Machine learning potentials
│   ├── model/                   # Neural network model architectures
│   │   ├── __init__.py
│   │   ├── base_model.py        # Base class for ML models
│   │   └── ...
│   ├── __init__.py
│   └── machine_learning_force.py # ML force field implementation
│
├── graph_diffusion/             # Graph-based diffusion models
│   ├── __init__.py
│   └── ...
│
├── run_scripts/                 # Example run scripts
│   ├── lj_run.py                # Lennard-Jones simulation script
│   ├── lj_run.json              # Configuration for LJ simulation
│   ├── mlps_run.py              # ML potential simulation script
│   ├── mlps_run.json            # Configuration for ML simulation
│   └── ...
│
├── __init__.py                  # Package initialization
└── setup.py                     # Package installation script
```

## Core Components

### core/

The `core` directory contains the main molecular dynamics engine:

- `force/`: Force field implementations
  - `base_force.py`: Abstract base class for all force fields
  - `lennard_jones_force.py`: Lennard-Jones potential implementation
  
- `integrator/`: Integrator implementations
  - `base_integrator.py`: Abstract base class for all integrators
  - `integrator.py`: Velocity Verlet integrator with NVE/NVT support
  
- `neighbor_search/`: Efficient algorithms for finding neighboring atoms
  - `neighbor_list.py`: Cell list implementation for neighbor search
  
- `md_model.py`: Model definitions that combine force fields and integrators
  - `BaseModel`: Standard MD model
  - `SumBackboneInterface`: Combines multiple force fields
  
- `md_simulation.py`: Simulation controller that runs simulations and collects results
  - `MDSimulator`: Main simulation class
  
- `energy_minimizer.py`: Tools for minimizing the energy of molecular structures
  - `EnergyMinimizer`: Energy minimization algorithms
  
- `parameter_manager.py`: Utilities for handling simulation parameters

### io_utils/

The `io_utils` directory contains tools for reading and writing data:

- `reader.py`: Structure file readers
  - `AtomFileReader`: Reads XYZ and other file formats
  
- `writer.py`: Output file writers
  - `TrajectoryWriter`: Writes trajectory data in various formats
  
- `file_converter.py`: Utilities for converting between different file formats

### machine_learning_potentials/

The `machine_learning_potentials` directory contains machine learning-based force fields:

- `model/`: Neural network model architectures
  - `base_model.py`: Base class for all ML models
  
- `machine_learning_force.py`: Implementation of ML-based force fields
  - `MachineLearningForce`: Adapter for using ML models as force fields

### run_scripts/

The `run_scripts` directory contains example scripts and configurations:

- `lj_run.py` / `lj_run.json`: Lennard-Jones simulation example
- `mlps_run.py` / `mlps_run.json`: Machine learning potential example
- `user_defined_run.py` / `user_defined_run.json`: Custom simulation example

## Adding New Components

### Adding a New Force Field

1. Create a new file in `core/force/` (e.g., `my_force.py`)
2. Implement your force field by extending `BaseForce`
3. Add your class to `core/force/__init__.py`

### Adding a New Integrator

1. Create a new file in `core/integrator/` (e.g., `my_integrator.py`)
2. Implement your integrator by extending `BaseIntegrator`
3. Add your class to `core/integrator/__init__.py`

### Adding a New ML Model

1. Create a new file in `machine_learning_potentials/model/` (e.g., `my_model.py`)
2. Implement your model by extending `BaseMLModel`
3. Update `MachineLearningForce._load_model()` to support your model type

## Coding Standards

TorchSim follows these coding standards:

1. **PEP 8**: Follow Python style guidelines
2. **Type Hints**: Use type hints for function signatures
3. **Docstrings**: Use Google-style docstrings for all classes and functions
4. **Unit Tests**: Write tests for new functionality
5. **Modularity**: Keep components loosely coupled
6. **Error Handling**: Use appropriate error handling and validation

## Testing

Tests are located in the `tests/` directory and can be run with pytest:

```bash
pytest
```

When adding new functionality, create corresponding tests in the appropriate test module. 