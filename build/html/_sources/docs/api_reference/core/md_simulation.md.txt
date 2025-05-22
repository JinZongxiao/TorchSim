# MDSimulation

The `MDSimulator` class is the main simulation controller in TorchSim. It manages the execution of molecular dynamics simulations and collects the resulting data.

## Class Definition

```python
class MDSimulator:
    def __init__(self, md_model, num_steps, print_interval=100, save_interval=None, output_dir="./output"):
        """
        Initialize the simulator.
        
        Parameters:
        -----------
        md_model : BaseModel
            The molecular dynamics model to simulate
        num_steps : int
            Number of simulation steps to run
        print_interval : int, optional
            Interval for printing simulation progress, by default 100
        save_interval : int, optional
            Interval for saving trajectory data, by default None (no saving during simulation)
        output_dir : str, optional
            Directory to save output files, by default "./output"
        """
```

## Methods

### run

```python
def run(self):
    """
    Run the molecular dynamics simulation.
    
    Returns:
    --------
    dict
        Dictionary containing simulation results including:
        - 'trajectory': Position history
        - 'energy': Potential energy history
        - 'temperature': Temperature history
        - 'time': Elapsed time
    """
```

### save_xyz_trajectory

```python
def save_xyz_trajectory(self, filename, atom_types=None):
    """
    Save the trajectory in XYZ format.
    
    Parameters:
    -----------
    filename : str
        Path to save the XYZ file
    atom_types : list, optional
        List of atom types, by default None (uses atom types from reader)
    """
```

### save_energy_curve

```python
def save_energy_curve(self, filename):
    """
    Save an energy plot showing the energy evolution during simulation.
    
    Parameters:
    -----------
    filename : str
        Path to save the plot image
    """
```

### save_temperature_curve

```python
def save_temperature_curve(self, filename):
    """
    Save a temperature plot showing temperature evolution during simulation.
    
    Parameters:
    -----------
    filename : str
        Path to save the plot image
    """
```

## Example Usage

```python
from torchsim.core.md_model import BaseModel, SumBackboneInterface
from torchsim.core.force.lennard_jones_force import LennardJonesForce
from torchsim.core.integrator.integrator import VerletIntegrator
from torchsim.io_utils.reader import AtomFileReader
from torchsim.core.md_simulation import MDSimulator

# Set up components
atom_reader = AtomFileReader("structure.xyz", box_length=10.0, cutoff=2.5)
lj_force = LennardJonesForce(atom_reader)
force_backbone = SumBackboneInterface([lj_force], atom_reader)
integrator = VerletIntegrator(atom_reader, dt=0.001, force_field=lj_force, ensemble='NVT', temperature=300)

# Create model and simulator
md_model = BaseModel(force_backbone, integrator, atom_reader)
simulator = MDSimulator(md_model, num_steps=1000, print_interval=100, save_interval=10, output_dir="./sim_output")

# Run simulation and save results
results = simulator.run()
simulator.save_xyz_trajectory("trajectory.xyz")
simulator.save_energy_curve("energy.png")
simulator.save_temperature_curve("temperature.png")
```

## Properties

- `trajectory`: List of position tensors for each saved step
- `energy_history`: List of potential energy values for each step
- `temperature_history`: List of temperature values for each step
- `elapsed_time`: Total simulation time in seconds