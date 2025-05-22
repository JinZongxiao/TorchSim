# Adding New Integrators

This guide explains how to implement custom integrators in TorchSim by extending the `BaseIntegrator` class.

## Integrator Interface

All integrators in TorchSim must extend the `BaseIntegrator` class and implement the `step()` method, which:

1. Updates atomic positions and velocities
2. Returns the potential energy of the system

## Basic Implementation

Here's a template for implementing a custom integrator:

```python
from torchsim.core.integrator.base_integrator import BaseIntegrator
import torch

class MyCustomIntegrator(BaseIntegrator):
    def __init__(self, atom_reader, dt, force_field, **kwargs):
        """
        Initialize the integrator.
        
        Parameters:
        -----------
        atom_reader : AtomFileReader
            Reader containing atomic positions and other data
        dt : float
            Time step in picoseconds
        force_field : BaseForce
            Force field to use for force calculations
        **kwargs : dict
            Additional parameters specific to this integrator
        """
        super().__init__(atom_reader, dt, force_field, **kwargs)
        # Initialize custom parameters
        self.param1 = kwargs.get('param1', default_value)
        self.param2 = kwargs.get('param2', default_value)
        
    def step(self):
        """
        Perform one integration step.
        
        Returns:
        --------
        float
            Potential energy of the system
        """
        # Calculate forces at current positions
        forces = self.force_field.compute_forces(self.positions)
        
        # Calculate potential energy
        energy = self.force_field.compute_energy(self.positions)
        
        # Update positions and velocities according to your integration scheme
        # Example (simple Euler integration):
        self.velocities += forces * self.dt
        self.positions += self.velocities * self.dt
        
        # Apply periodic boundary conditions if needed
        if self.pbc:
            self.positions = self.apply_pbc(self.positions)
        
        return energy
```

## Example: Leap-Frog Integrator

Here's an example implementation of the leap-frog integration algorithm:

```python
from torchsim.core.integrator.base_integrator import BaseIntegrator
import torch

class LeapFrogIntegrator(BaseIntegrator):
    def __init__(self, atom_reader, dt, force_field, **kwargs):
        super().__init__(atom_reader, dt, force_field, **kwargs)
        # Initialize velocities at t - dt/2 for leap-frog
        forces = self.force_field.compute_forces(self.positions)
        self.velocities -= 0.5 * self.dt * forces  # v(t-dt/2)
        
    def step(self):
        # Calculate forces at current positions
        forces = self.force_field.compute_forces(self.positions)
        
        # Update velocities to t + dt/2
        self.velocities += self.dt * forces  # v(t+dt/2)
        
        # Update positions to t + dt
        self.positions += self.dt * self.velocities
        
        # Apply periodic boundary conditions if needed
        if self.pbc:
            self.positions = self.apply_pbc(self.positions)
        
        # Calculate potential energy
        energy = self.force_field.compute_energy(self.positions)
        
        return energy
```

## Implementing Thermostats

To implement a thermostat for temperature control (NVT ensemble), you need to modify the velocities at each step:

```python
class NVTIntegrator(BaseIntegrator):
    def __init__(self, atom_reader, dt, force_field, temperature=300.0, gamma=0.1, **kwargs):
        super().__init__(atom_reader, dt, force_field, **kwargs)
        self.temperature = temperature  # Target temperature in Kelvin
        self.gamma = gamma              # Friction coefficient
        self.kB = 8.3145e-3             # Boltzmann constant in appropriate units
        
    def step(self):
        # Calculate forces
        forces = self.force_field.compute_forces(self.positions)
        
        # First part of velocity update
        self.velocities += 0.5 * self.dt * forces
        
        # Apply thermostat (Langevin dynamics)
        c1 = torch.exp(-self.gamma * self.dt)
        c2 = torch.sqrt((1.0 - c1**2) * self.kB * self.temperature / self.masses)
        
        # Random thermal noise
        noise = torch.randn_like(self.velocities, device=self.device)
        
        # Update velocities with thermostat
        self.velocities = c1 * self.velocities + c2 * noise
        
        # Update positions
        self.positions += self.dt * self.velocities
        
        # Apply periodic boundary conditions
        if self.pbc:
            self.positions = self.apply_pbc(self.positions)
        
        # Calculate new forces and finish velocity update
        forces = self.force_field.compute_forces(self.positions)
        self.velocities += 0.5 * self.dt * forces
        
        # Calculate energy
        energy = self.force_field.compute_energy(self.positions)
        
        return energy
```

## Registering Your Integrator

To make your integrator available to the rest of the system:

1. Create a new file in the `torchsim/core/integrator/` directory
2. Implement your integrator class
3. Import your integrator in `torchsim/core/integrator/__init__.py`

## Testing Your Integrator

It's important to test your integrator to ensure it behaves correctly:

1. Test energy conservation for NVE integrators
2. Test temperature stability for NVT integrators
3. Compare with analytical solutions for simple systems
4. Check long-term stability with extended simulations

```python
# Example test
atom_reader = AtomFileReader("test_structure.xyz", box_length=10.0, cutoff=2.5)
lj_force = LennardJonesForce(atom_reader)
my_integrator = MyCustomIntegrator(atom_reader, dt=0.001, force_field=lj_force)

# Run a short simulation
energies = []
for _ in range(1000):
    energy = my_integrator.step()
    energies.append(energy.item())

# Check energy conservation (for NVE)
energy_std = torch.std(torch.tensor(energies))
print(f"Energy standard deviation: {energy_std}")  # Should be small for NVE 