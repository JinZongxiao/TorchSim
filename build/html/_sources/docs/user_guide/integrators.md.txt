# Integrators

Integrators in TorchSim are responsible for updating the atomic positions and velocities according to the equations of motion. This page describes the available integrators and their parameters.

## Verlet Integrator

The Verlet integrator is the most commonly used integrator in molecular dynamics simulations. TorchSim implements the velocity Verlet algorithm, which provides good energy conservation and stability.

### Basic Usage

```python
from torchsim.core.integrator.integrator import VerletIntegrator
from torchsim.core.force.lennard_jones_force import LennardJonesForce
from torchsim.io_utils.reader import AtomFileReader

atom_reader = AtomFileReader("structure.xyz", box_length=10.0, cutoff=2.5)
lj_force = LennardJonesForce(atom_reader)

# Create the integrator
integrator = VerletIntegrator(
    atom_reader,
    dt=0.001,              # Time step in picoseconds
    force_field=lj_force,  # Force field to use
    ensemble='NVT',        # Ensemble type: 'NVE' or 'NVT'
    temperature=300,       # Target temperature in Kelvin (for NVT)
    gamma=0.1              # Friction coefficient (for NVT)
)
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `atom_reader` | AtomFileReader instance | Required |
| `dt` | Time step in picoseconds | Required |
| `force_field` | Force field instance | Required |
| `ensemble` | Ensemble type ('NVE' or 'NVT') | 'NVE' |
| `temperature` | Target temperature in Kelvin | 300 |
| `gamma` | Friction coefficient for NVT | 0.1 |
| `seed` | Random seed for thermostat | None |

## NVE Ensemble

The NVE ensemble (microcanonical ensemble) conserves the total energy, volume, and number of particles. It's used for simulations where energy conservation is important.

```python
# NVE ensemble
integrator = VerletIntegrator(
    atom_reader,
    dt=0.001,
    force_field=lj_force,
    ensemble='NVE'  # No thermostat, energy is conserved
)
```

## NVT Ensemble

The NVT ensemble (canonical ensemble) controls the temperature using a thermostat. TorchSim uses a Langevin thermostat for NVT simulations.

```python
# NVT ensemble with Langevin thermostat
integrator = VerletIntegrator(
    atom_reader,
    dt=0.001,
    force_field=lj_force,
    ensemble='NVT',      # Use Langevin thermostat
    temperature=300,     # Target temperature in Kelvin
    gamma=0.1           # Friction coefficient
)
```

## Custom Integrators

You can create custom integrators by extending the `BaseIntegrator` class:

```python
from torchsim.core.integrator.base_integrator import BaseIntegrator
import torch

class CustomIntegrator(BaseIntegrator):
    def __init__(self, atom_reader, dt, force_field, **kwargs):
        super().__init__(atom_reader, dt, force_field, **kwargs)
        # Initialize custom parameters here
        
    def step(self):
        # Implement your integration step here
        # Update self.positions and self.velocities
        # Return the potential energy
        
        # Example:
        forces = self.force_field.compute_forces(self.positions)
        energy = self.force_field.compute_energy(self.positions)
        
        # Update positions and velocities using your custom scheme
        self.positions += self.velocities * self.dt + 0.5 * forces * self.dt**2
        self.velocities += forces * self.dt
        
        return energy
``` 