# Quick Start Guide

This guide will help you get started with TorchSim by running basic molecular dynamics simulations.

## Basic Lennard-Jones Simulation

Here's a minimal example to run a Lennard-Jones simulation:

```python
from torchsim.io_utils.reader import AtomFileReader
from torchsim.core.force.lennard_jones_force import LennardJonesForce
from torchsim.core.integrator.integrator import VerletIntegrator
from torchsim.core.md_model import BaseModel, SumBackboneInterface
from torchsim.core.md_simulation import MDSimulator

# Read the structure data
atom_reader = AtomFileReader("your_structure.xyz", 
                             box_length=10.0, 
                             cutoff=2.5)

# Set up force field
lj_force = LennardJonesForce(atom_reader)
force_backbone = SumBackboneInterface([lj_force], atom_reader)

# Set up integrator
integrator = VerletIntegrator(atom_reader, 
                              dt=0.001,
                              force_field=lj_force,
                              ensemble='NVT',
                              temperature=300,
                              gamma=0.1)

# Create model and simulator
md_model = BaseModel(force_backbone, integrator, atom_reader)
simulator = MDSimulator(md_model, num_steps=1000, print_interval=100)

# Run simulation
results = simulator.run()

# Save the trajectory and energy curve
simulator.save_xyz_trajectory("trajectory.xyz", atom_types=atom_reader.atom_types)
simulator.save_energy_curve("energy_curve.png")
```

## Machine Learning Potential Simulation

To use a machine learning potential instead:

```python
from torchsim.machine_learning_potentials.machine_learning_force import MachineLearningForce

# Initialize ML potential
ml_force = MachineLearningForce(atom_reader, model_path="your_model.pt")

# Use ML potential in the force backbone
force_backbone = SumBackboneInterface([ml_force], atom_reader)

# The rest of the setup is the same as the basic example
```

## Using Configuration Files

TorchSim supports JSON configuration files for simulations:

1. Create a JSON configuration file (e.g., `lj_run.json`):

```json
{
  "data_path_xyz": "structures/water.xyz",
  "box_length": 10.0,
  "cut_off": 2.5,
  "dt": 0.001,
  "temperature": 300,
  "gamma": 0.1,
  "num_steps": 1000,
  "print_interval": 100,
  "output_save_path": "./output",
  "pair_parameter": {
    "epsilon": 0.1,
    "sigma": 3.4
  }
}
```

2. Run the simulation with the configuration file:

```bash
python lj_run.py --config lj_run.json
```

For more details, please refer to the [User Guide](user_guide/index.md) and [Examples](examples/index.md). 