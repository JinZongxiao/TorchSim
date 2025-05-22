# TorchSim

torchsim is a PyTorch-based molecular dynamics simulation framework that integrates traditional molecular force fields with machine learning potential models.

## Features
- **GPU Acceleration**: Leverages PyTorch's CUDA support
- **Hybrid Force Fields**: LJ potential + ML potentials
- **Flexible Integrators**: Verlet algorithm with NVT ensemble
- **GNN Support**: Integrated with PyTorch Geometric

## Installation
```bash
# From source
git clone https://github.com/JinZongxiao/torchsim.git
cd torchsim
pip install -e .

# PyPI
pip install torchsim
```

## Quick Start


### 1. Basic LJ Simulation

```python
from torchsim.io_utils.reader import AtomFileReader
from torchsim.core.force.lennard_jones_force import LennardJonesForce
from torchsim.core.integrator.integrator import VerletIntegrator
from torchsim.core.md_model import BaseModel, SumBackboneInterface
from torchsim.core.md_simulation import MDSimulator

# read the structure data
atom_reader = AtomFileReader("your_structure.xyz", 
                             box_length=10.0, 
                             cutoff=2.5)

# set force field
lj_force = LennardJonesForce(atom_reader)
force_backbone = SumBackboneInterface([lj_force], atom_reader)

# set integrator
integrator = VerletIntegrator(atom_reader, 
                              dt=0.001,
                              force_field=lj_force,
                              ensemble='NVT',
                              temperature=300,
                              gamma=0.1)

# create modle and simulator
md_model = BaseModel(force_backbone, integrator, atom_reader)
simulator = MDSimulator(md_model, num_steps=1000, print_interval=100)

# run simulation
results = simulator.run()

# save the info
simulator.save_xyz_trajectory("trajectory.xyz", atom_types=atom_reader.atom_types)
simulator.save_energy_curve("energy_curve.png")
```

### 2. ML Potential


```python
from torchsim.machine_learning_potentials.machine_learning_force import MachineLearningForce

# initialze mlps
ml_force = MachineLearningForce(atom_reader, model_path="your_model.pt")

# same as the porcess above only change the field module
force_backbone = SumBackboneInterface([ml_force], atom_reader)
```

## Configuration Example

You also can use JSON file to run the simulation：

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

and run the script use the command as：

```bash
python lj_run.py --config lj_run.json
```

## Project Structure

```
torchsim/
├── core/                  # core engine
│   ├── force/             # force fileds
│   ├── integrator/        # integrators
│   ├── neighbor_search/   # neighbor search algorithms
│   ├── md_model.py        # model definition
│   └── md_simulation.py   # simulator
├── io_utils/               
├── machine_learning_potentials/ 
├── graph_diffusion/       
└── run/                   # scripts
```

## Documentation
[Read the doc](https://google.com)

## Cite
```
@software{TorchSim,
  author = {Author Name},
  title = {TorchSim: Hybrid MD Simulator},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/JinZongxiao/torchsim}}
}
```