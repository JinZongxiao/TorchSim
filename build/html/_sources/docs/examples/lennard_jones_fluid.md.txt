# Lennard-Jones Fluid Example

This example demonstrates how to simulate a simple Lennard-Jones fluid using TorchSim.

## Overview

The Lennard-Jones potential is a simple mathematical model that approximates the interaction between neutral atoms or molecules. It's commonly used to simulate noble gases like argon.

In this example, we'll:
1. Set up a system of 100 particles
2. Run a molecular dynamics simulation
3. Visualize the results

## Complete Code

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchsim.io_utils.reader import AtomFileReader
from torchsim.core.force.lennard_jones_force import LennardJonesForce
from torchsim.core.integrator.integrator import VerletIntegrator
from torchsim.core.md_model import BaseModel, SumBackboneInterface
from torchsim.core.md_simulation import MDSimulator

# Create a random initial configuration of 100 particles in a box
def create_random_xyz(num_particles=100, box_length=10.0, filename="lj_fluid.xyz"):
    with open(filename, "w") as f:
        f.write(f"{num_particles}\n")
        f.write("Lennard-Jones fluid\n")
        
        for i in range(num_particles):
            # Random position within the box
            x = np.random.uniform(0, box_length)
            y = np.random.uniform(0, box_length)
            z = np.random.uniform(0, box_length)
            
            # All particles are of type 'Ar' (argon)
            f.write(f"Ar {x:.6f} {y:.6f} {z:.6f}\n")
    
    return filename

# Create the input file
input_file = create_random_xyz(num_particles=100, box_length=10.0)

# Read the structure data
box_length = 10.0
cutoff = 2.5
atom_reader = AtomFileReader(input_file, box_length=box_length, cutoff=cutoff)

# Set up force field
epsilon = 0.1    # Energy parameter in kcal/mol
sigma = 3.4      # Distance parameter in Angstroms
lj_force = LennardJonesForce(atom_reader, epsilon=epsilon, sigma=sigma)
force_backbone = SumBackboneInterface([lj_force], atom_reader)

# Set up integrator
dt = 0.001  # Time step in picoseconds
temperature = 100.0  # Temperature in Kelvin
gamma = 0.1   # Friction coefficient for Langevin thermostat
integrator = VerletIntegrator(atom_reader, 
                             dt=dt,
                             force_field=lj_force,
                             ensemble='NVT',
                             temperature=temperature,
                             gamma=gamma)

# Create model and simulator
md_model = BaseModel(force_backbone, integrator, atom_reader)
num_steps = 10000
print_interval = 1000
simulator = MDSimulator(md_model, num_steps=num_steps, print_interval=print_interval)

# Run simulation
print(f"Running simulation with {num_steps} steps...")
results = simulator.run()

# Save the trajectory and energy curve
output_prefix = "lj_fluid"
simulator.save_xyz_trajectory(f"{output_prefix}_trajectory.xyz", atom_types=atom_reader.atom_types)
simulator.save_energy_curve(f"{output_prefix}_energy.png")
simulator.save_temperature_curve(f"{output_prefix}_temperature.png")

# Plot the radial distribution function
def compute_rdf(positions, box_length, num_bins=100, r_max=None):
    if r_max is None:
        r_max = box_length / 2
    
    n_atoms = positions.shape[0]
    r_edges = torch.linspace(0, r_max, num_bins+1, device=positions.device)
    r_centers = 0.5 * (r_edges[1:] + r_edges[:-1])
    dr = r_edges[1] - r_edges[0]
    
    # Initialize histogram
    hist = torch.zeros(num_bins, device=positions.device)
    
    # Loop over all atom pairs
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            # Calculate minimum image distance
            rij = positions[i] - positions[j]
            rij = rij - box_length * torch.round(rij / box_length)
            dist = torch.norm(rij)
            
            # Add to histogram if within range
            if dist < r_max:
                bin_idx = int(dist / dr)
                if bin_idx < num_bins:
                    hist[bin_idx] += 2  # Count each pair twice
    
    # Normalize by ideal gas RDF
    volume = box_length**3
    density = n_atoms / volume
    norm = 4 * np.pi * r_centers**2 * dr * density * n_atoms
    rdf = hist / norm
    
    return r_centers.cpu().numpy(), rdf.cpu().numpy()

# Compute RDF from the last snapshot
final_positions = results['trajectory'][-1]
r, g_r = compute_rdf(final_positions, box_length)

# Plot RDF
plt.figure(figsize=(10, 6))
plt.plot(r, g_r)
plt.xlabel('r (Å)')
plt.ylabel('g(r)')
plt.title('Radial Distribution Function')
plt.grid(True)
plt.savefig(f"{output_prefix}_rdf.png")
plt.close()

print(f"Simulation completed. Results saved with prefix '{output_prefix}'")
```

## Step-by-Step Explanation

### 1. Setting Up the System

First, we create a random initial configuration of particles:

```python
def create_random_xyz(num_particles=100, box_length=10.0, filename="lj_fluid.xyz"):
    # ... (function body as shown above)
```

This function generates a random distribution of particles within a cubic box and saves it in XYZ format.

### 2. Initializing TorchSim Components

We then set up the simulation components:

```python
# Read the structure data
atom_reader = AtomFileReader(input_file, box_length=box_length, cutoff=cutoff)

# Set up force field
lj_force = LennardJonesForce(atom_reader, epsilon=epsilon, sigma=sigma)

# Set up integrator with Langevin thermostat (NVT ensemble)
integrator = VerletIntegrator(atom_reader, dt=dt, force_field=lj_force,
                             ensemble='NVT', temperature=temperature, gamma=gamma)
```

### 3. Running the Simulation

With all components set up, we create the model and simulator, then run the simulation:

```python
md_model = BaseModel(force_backbone, integrator, atom_reader)
simulator = MDSimulator(md_model, num_steps=num_steps, print_interval=print_interval)
results = simulator.run()
```

### 4. Analyzing the Results

After the simulation, we save the trajectory and analyze the results:

```python
# Save trajectory and energy/temperature curves
simulator.save_xyz_trajectory(f"{output_prefix}_trajectory.xyz")
simulator.save_energy_curve(f"{output_prefix}_energy.png")
simulator.save_temperature_curve(f"{output_prefix}_temperature.png")

# Calculate and plot the radial distribution function
# ... (RDF calculation and plotting code)
```

The radial distribution function (RDF) shows the local structure of the fluid by measuring how the particle density varies as a function of distance from a reference particle.

## Expected Results

After running this simulation, you should see:

1. **Energy Curve**: The potential energy should decrease and stabilize as the system equilibrates
2. **Temperature Curve**: The temperature should fluctuate around the target value (100K)
3. **Radial Distribution Function**: Should show characteristic peaks corresponding to the shells of neighboring particles

## Variations

You can modify this example to explore different aspects of Lennard-Jones fluids:

- Change the temperature to observe phase transitions
- Modify the density by changing the box size or number of particles
- Adjust the Lennard-Jones parameters (epsilon and sigma) to model different substances
- Use different ensembles (NVE instead of NVT) by changing the integrator configuration 