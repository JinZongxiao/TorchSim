# Machine Learning Integration

This guide explains how to integrate machine learning models into TorchSim for use as potential energy functions.

## Overview

TorchSim supports the use of machine learning models as potential energy functions through the `MachineLearningForce` class. This allows you to:

1. Use pre-trained neural network models for molecular dynamics simulations
2. Combine traditional force fields with machine learning potentials
3. Develop and test new machine learning architectures for molecular simulation

## Basic Integration

To use a machine learning model as a force field:

```python
from torchsim.machine_learning_potentials.machine_learning_force import MachineLearningForce
from torchsim.io_utils.reader import AtomFileReader

atom_reader = AtomFileReader("structure.xyz", box_length=10.0, cutoff=2.5)

# Initialize ML force with a pre-trained model
ml_force = MachineLearningForce(
    atom_reader,
    model_path="path/to/model.pt",
    model_type="SchNet"  # or other supported model type
)

# Use it like any other force field
energy = ml_force.compute_energy(atom_reader.positions)
forces = ml_force.compute_forces(atom_reader.positions)
```

## Supported Model Types

TorchSim currently supports several types of machine learning models:

1. **SchNet**: Graph neural network for molecular properties
2. **PhysNet**: Physics-inspired neural network potential
3. **Custom**: User-defined models that follow the required interface

## Creating a Custom ML Model

To create a custom machine learning model for use with TorchSim:

1. Create a new class in `torchsim/machine_learning_potentials/model/`
2. Implement the required interface methods
3. Register your model type in `MachineLearningForce`

Here's an example of a simple custom model:

```python
import torch
import torch.nn as nn
from torchsim.machine_learning_potentials.model.base_model import BaseMLModel

class SimpleMLModel(BaseMLModel):
    def __init__(self, n_atom_types, cutoff=5.0):
        super().__init__()
        self.cutoff = cutoff
        self.n_atom_types = n_atom_types
        
        # Define your neural network layers
        self.embedding = nn.Embedding(n_atom_types, 32)
        self.mlp = nn.Sequential(
            nn.Linear(32 * 2 + 1, 64),  # atom features + distance
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, positions, atom_types):
        """
        Forward pass to compute energy.
        
        Parameters:
        -----------
        positions : torch.Tensor
            Atomic positions with shape (n_atoms, 3)
        atom_types : torch.Tensor
            Atom type indices with shape (n_atoms,)
            
        Returns:
        --------
        torch.Tensor
            Scalar tensor containing the potential energy
        """
        n_atoms = positions.shape[0]
        device = positions.device
        
        # Embed atom types
        atom_features = self.embedding(atom_types)
        
        # Compute pairwise interactions
        energy = torch.tensor(0.0, device=device)
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                # Calculate distance
                r_ij = torch.norm(positions[i] - positions[j])
                
                # Skip if beyond cutoff
                if r_ij > self.cutoff:
                    continue
                    
                # Combine atom features and distance
                features_i = atom_features[i]
                features_j = atom_features[j]
                pair_input = torch.cat([features_i, features_j, r_ij.unsqueeze(0)])
                
                # Compute pair energy contribution
                pair_energy = self.mlp(pair_input)
                energy += pair_energy
                
        return energy
```

## Registering Your Model

To register your custom model with the `MachineLearningForce` class:

1. Add your model class to the appropriate module
2. Update the `_load_model` method in `MachineLearningForce`:

```python
def _load_model(self, model_path, model_type):
    if model_type == "SchNet":
        from torchsim.machine_learning_potentials.model.schnet import SchNet
        return SchNet.load_from_checkpoint(model_path)
    elif model_type == "PhysNet":
        from torchsim.machine_learning_potentials.model.physnet import PhysNet
        return PhysNet.load_from_checkpoint(model_path)
    elif model_type == "SimpleML":  # Your custom model
        from torchsim.machine_learning_potentials.model.simple_ml import SimpleMLModel
        return SimpleMLModel.load_from_checkpoint(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

## Training ML Models

TorchSim doesn't directly provide training functionality, but you can use PyTorch Lightning or other frameworks to train your models:

```python
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchsim.machine_learning_potentials.model.simple_ml import SimpleMLModel

# Define a PyTorch Lightning module for training
class MLTrainer(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, positions, atom_types):
        return self.model(positions, atom_types)
        
    def training_step(self, batch, batch_idx):
        positions, atom_types, target_energy, target_forces = batch
        
        # Forward pass with gradient tracking
        positions.requires_grad_(True)
        energy = self.model(positions, atom_types)
        
        # Compute forces as negative gradient of energy
        forces = -torch.autograd.grad(energy, positions, create_graph=True)[0]
        
        # Compute losses
        energy_loss = torch.nn.functional.mse_loss(energy, target_energy)
        forces_loss = torch.nn.functional.mse_loss(forces, target_forces)
        
        # Combined loss
        loss = energy_loss + forces_loss
        
        self.log('train_loss', loss)
        return loss
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Create and train the model
model = SimpleMLModel(n_atom_types=10, cutoff=5.0)
trainer = MLTrainer(model)

# Train the model (assuming you have a DataLoader)
pl_trainer = pl.Trainer(max_epochs=100)
pl_trainer.fit(trainer, train_dataloader)

# Save the trained model
torch.save(model.state_dict(), "trained_simple_ml.pt")
```

## Using Hybrid Potentials

You can combine traditional force fields with machine learning potentials using the `SumBackboneInterface`:

```python
from torchsim.core.md_model import SumBackboneInterface
from torchsim.core.force.lennard_jones_force import LennardJonesForce

# Create both force fields
lj_force = LennardJonesForce(atom_reader)
ml_force = MachineLearningForce(atom_reader, model_path="model.pt", model_type="SchNet")

# Combine them
force_backbone = SumBackboneInterface([lj_force, ml_force], atom_reader)

# Use in simulation
integrator = VerletIntegrator(atom_reader, dt=0.001, force_field=force_backbone)
```

This allows you to leverage the strengths of both approaches: the physical accuracy of traditional potentials and the flexibility of machine learning models. 