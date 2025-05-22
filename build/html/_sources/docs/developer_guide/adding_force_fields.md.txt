# Adding New Force Fields

This guide explains how to implement custom force fields in TorchSim by extending the `BaseForce` class.

## Force Field Interface

All force fields in TorchSim must implement the `BaseForce` interface, which requires two main methods:

1. `compute_energy(positions)`: Calculate the potential energy of the system
2. `compute_forces(positions)`: Calculate the forces on all atoms

## Basic Implementation

Here's a template for implementing a custom force field:

```python
from torchsim.core.force.base_force import BaseForce
import torch

class MyCustomForce(BaseForce):
    def __init__(self, atom_reader, **kwargs):
        super().__init__(atom_reader)
        # Initialize your parameters here
        self.param1 = kwargs.get('param1', default_value)
        self.param2 = kwargs.get('param2', default_value)
        
    def compute_energy(self, positions):
        """
        Compute the potential energy of the system.
        
        Parameters:
        -----------
        positions : torch.Tensor
            Atomic positions with shape (n_atoms, 3)
            
        Returns:
        --------
        torch.Tensor
            Scalar tensor containing the potential energy
        """
        # Your energy calculation logic here
        energy = torch.tensor(0.0, device=self.device)
        
        # Example calculation
        # distances = self.compute_distances(positions)
        # energy = self.param1 * torch.sum(distances**2)
        
        return energy
        
    def compute_forces(self, positions):
        """
        Compute the forces on all atoms.
        
        Parameters:
        -----------
        positions : torch.Tensor
            Atomic positions with shape (n_atoms, 3)
            
        Returns:
        --------
        torch.Tensor
            Forces on all atoms with shape (n_atoms, 3)
        """
        # Your force calculation logic here
        forces = torch.zeros_like(positions)
        
        # Example calculation
        # For simple cases, you can use autograd to calculate forces
        # positions_with_grad = positions.clone().detach().requires_grad_(True)
        # energy = self.compute_energy(positions_with_grad)
        # energy.backward()
        # forces = -positions_with_grad.grad
        
        return forces
```

## Using PyTorch Autograd

For many force fields, you can use PyTorch's automatic differentiation to calculate forces from the energy:

```python
def compute_forces(self, positions):
    # Create a copy of positions that requires gradient
    positions_with_grad = positions.clone().detach().requires_grad_(True)
    
    # Compute energy with the positions that require gradient
    energy = self.compute_energy(positions_with_grad)
    
    # Compute gradient of energy with respect to positions
    energy.backward()
    
    # Forces are negative gradient of energy
    forces = -positions_with_grad.grad
    
    return forces
```

## Neighbor Lists

For efficiency, many force fields use neighbor lists to avoid calculating interactions between distant atoms:

```python
from torchsim.core.neighbor_search.neighbor_list import NeighborList

def __init__(self, atom_reader, **kwargs):
    super().__init__(atom_reader)
    self.cutoff = kwargs.get('cutoff', 2.5)
    self.neighbor_list = NeighborList(atom_reader, self.cutoff)
    
def compute_energy(self, positions):
    # Update neighbor list if needed
    self.neighbor_list.update(positions)
    
    # Get neighbor indices and distances
    indices, distances = self.neighbor_list.get_neighbors(positions)
    
    # Use these for efficient energy calculation
    # ...
```

## Registering Your Force Field

To make your force field available to the rest of the system, add it to the appropriate module:

1. Create a new file in the `torchsim/core/force/` directory
2. Implement your force field class
3. Import your force field in `torchsim/core/force/__init__.py`

## Example: Simple Harmonic Potential

Here's a complete example of a simple harmonic potential:

```python
from torchsim.core.force.base_force import BaseForce
import torch

class HarmonicForce(BaseForce):
    def __init__(self, atom_reader, k=1.0, r0=1.0):
        super().__init__(atom_reader)
        self.k = k    # Spring constant
        self.r0 = r0  # Equilibrium distance
        
    def compute_energy(self, positions):
        n_atoms = positions.shape[0]
        energy = torch.tensor(0.0, device=self.device)
        
        # Loop through all pairs of atoms
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                # Calculate distance between atoms
                r_ij = torch.norm(positions[i] - positions[j])
                
                # Harmonic potential: E = 0.5 * k * (r - r0)^2
                energy += 0.5 * self.k * (r_ij - self.r0)**2
                
        return energy
        
    def compute_forces(self, positions):
        positions_with_grad = positions.clone().detach().requires_grad_(True)
        energy = self.compute_energy(positions_with_grad)
        energy.backward()
        return -positions_with_grad.grad
```

## Testing Your Force Field

It's important to test your force field to ensure it behaves correctly:

1. Create a simple system with known behavior
2. Calculate energy and forces
3. Verify conservation of energy in NVE simulations
4. Compare with analytical solutions if available

```python
# Example test
atom_reader = AtomFileReader("test_structure.xyz", box_length=10.0, cutoff=2.5)
my_force = MyCustomForce(atom_reader, param1=1.0)

# Test energy calculation
energy = my_force.compute_energy(atom_reader.positions)
print(f"Energy: {energy.item()}")

# Test force calculation
forces = my_force.compute_forces(atom_reader.positions)
print(f"Forces sum: {torch.sum(forces, dim=0)}")  # Should be close to zero
``` 