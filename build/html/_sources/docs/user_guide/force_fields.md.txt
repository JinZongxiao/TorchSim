# Force Fields

TorchSim provides various force field implementations for molecular dynamics simulations. This page describes the available force fields and how to configure them.

## Lennard-Jones Force Field

The Lennard-Jones potential is a mathematical model that approximates the interaction between a pair of neutral atoms or molecules:

### Basic Usage

```python
from torchsim.core.force.lennard_jones_force import LennardJonesForce
from torchsim.io_utils.reader import AtomFileReader

# Initialize with default parameters
atom_reader = AtomFileReader("structure.xyz", box_length=10.0, cutoff=2.5)
lj_force = LennardJonesForce(atom_reader)

# Initialize with custom parameters
lj_force = LennardJonesForce(
    atom_reader,
    epsilon=0.1,      # Energy parameter in kcal/mol
    sigma=3.4,        # Distance parameter in Angstroms
    use_cutoff=True,  # Whether to use distance cutoff
    shifted=True      # Whether to use shifted potential
)
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `atom_reader` | AtomFileReader instance | Required |
| `epsilon` | Energy parameter in kcal/mol | 0.1 |
| `sigma` | Distance parameter in Angstroms | 3.4 |
| `use_cutoff` | Whether to use distance cutoff | True |
| `shifted` | Whether to use shifted potential | True |

## Machine Learning Potentials

TorchSim supports machine learning potentials through the `MachineLearningForce` class:

```python
from torchsim.machine_learning_potentials.machine_learning_force import MachineLearningForce

ml_force = MachineLearningForce(
    atom_reader,
    model_path="path/to/model.pt",  # Path to trained PyTorch model
    model_type="SchNet"              # Model architecture type
)
```

For more information on machine learning potentials, see the [Machine Learning Potentials](ml_potentials.md) page.

## Combining Force Fields

TorchSim allows combining multiple force fields using the `SumBackboneInterface`:

```python
from torchsim.core.md_model import SumBackboneInterface

# Combine Lennard-Jones and ML potentials
force_backbone = SumBackboneInterface([lj_force, ml_force], atom_reader)
```

## Custom Force Fields

You can create custom force fields by extending the `BaseForce` class:

```python
from torchsim.core.force.base_force import BaseForce
import torch

class CustomForce(BaseForce):
    def __init__(self, atom_reader, **kwargs):
        super().__init__(atom_reader)
        # Initialize your parameters here
        
    def compute_energy(self, positions):
        # Compute and return the potential energy
        # positions: tensor of shape (n_atoms, 3)
        energy = torch.tensor(0.0, device=self.device)
        # Your energy calculation logic here
        return energy
        
    def compute_forces(self, positions):
        # Compute and return the forces
        # positions: tensor of shape (n_atoms, 3)
        forces = torch.zeros_like(positions)
        # Your force calculation logic here
        return forces
``` 