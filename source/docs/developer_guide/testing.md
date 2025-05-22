# Testing

This guide explains how to write and run tests for TorchSim to ensure code quality and correctness.

## Testing Framework

TorchSim uses [pytest](https://docs.pytest.org/) for testing. The test directory structure mirrors the main package structure:

```
tests/
├── core/
│   ├── force/
│   │   ├── test_lennard_jones_force.py
│   │   └── ...
│   ├── integrator/
│   │   ├── test_verlet_integrator.py
│   │   └── ...
│   └── ...
├── io_utils/
│   ├── test_reader.py
│   └── ...
├── machine_learning_potentials/
│   ├── test_machine_learning_force.py
│   └── ...
└── ...
```

## Running Tests

To run all tests:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/core/force/test_lennard_jones_force.py
```

To run a specific test:

```bash
pytest tests/core/force/test_lennard_jones_force.py::test_compute_energy
```

## Writing Tests

When adding new features to TorchSim, you should also add tests to verify their correctness. Here are some guidelines for writing tests:

### Test File Structure

Each test file should focus on a specific module or class:

```python
import torch
import pytest
from torchsim.core.force.lennard_jones_force import LennardJonesForce
from torchsim.io_utils.reader import AtomFileReader

# Test fixtures (reusable test components)
@pytest.fixture
def atom_reader():
    # Create a simple test system
    # ... (setup code)
    return reader

@pytest.fixture
def lj_force(atom_reader):
    return LennardJonesForce(atom_reader, epsilon=0.1, sigma=3.4)

# Test functions
def test_compute_energy(lj_force, atom_reader):
    energy = lj_force.compute_energy(atom_reader.positions)
    # Assert that energy is a scalar tensor
    assert energy.dim() == 0
    # Assert that energy is a reasonable value
    assert energy.item() > 0

def test_compute_forces(lj_force, atom_reader):
    forces = lj_force.compute_forces(atom_reader.positions)
    # Assert that forces have the right shape
    assert forces.shape == atom_reader.positions.shape
    # Assert that forces approximately sum to zero (conservation of momentum)
    assert torch.allclose(torch.sum(forces, dim=0), torch.zeros(3, device=forces.device), atol=1e-6)
```

### Test Categories

#### Unit Tests

Unit tests focus on testing individual functions or methods in isolation. They should:
- Be fast and independent
- Test a single functionality
- Use mocks or fixtures for dependencies

Example:

```python
def test_lennard_jones_energy_formula(lj_force):
    # Test with a simple pair of atoms at a known distance
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.4]], device=lj_force.device)
    energy = lj_force.compute_energy(positions)
    
    # At r = sigma, the energy should be 0
    assert torch.isclose(energy, torch.tensor(0.0, device=lj_force.device), atol=1e-6)
```

#### Integration Tests

Integration tests verify that different components work together correctly:

```python
def test_lj_simulation_energy_conservation():
    # Create a simple system
    atom_reader = AtomFileReader("test_data/small_system.xyz", box_length=10.0, cutoff=2.5)
    lj_force = LennardJonesForce(atom_reader)
    integrator = VerletIntegrator(atom_reader, dt=0.001, force_field=lj_force, ensemble='NVE')
    md_model = BaseModel(SumBackboneInterface([lj_force], atom_reader), integrator, atom_reader)
    
    # Run a short simulation
    simulator = MDSimulator(md_model, num_steps=100)
    results = simulator.run()
    
    # Check energy conservation
    energies = torch.tensor(results['energy'])
    energy_std = torch.std(energies)
    
    # Energy should be conserved in NVE ensemble
    assert energy_std / torch.mean(energies) < 0.01  # Less than 1% variation
```

#### Regression Tests

Regression tests ensure that bugs, once fixed, don't reappear:

```python
def test_pbc_edge_case():
    # This test reproduces a previous bug with periodic boundary conditions
    # ... (test code reproducing the bug scenario)
    
    # Assert that the bug is fixed
    assert correct_behavior == observed_behavior
```

### Test Fixtures

Use pytest fixtures to set up reusable test components:

```python
@pytest.fixture
def small_system():
    # Create a small test system with known properties
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0],
        [0.0, 3.0, 0.0],
        [3.0, 0.0, 0.0]
    ])
    atom_types = ["Ar", "Ar", "Ar", "Ar"]
    box_length = 10.0
    cutoff = 5.0
    
    # Create a mock AtomFileReader
    reader = MagicMock(spec=AtomFileReader)
    reader.positions = positions
    reader.atom_types = atom_types
    reader.n_atoms = len(positions)
    reader.box_length = box_length
    reader.cutoff = cutoff
    reader.device = "cpu"
    
    return reader
```

### Testing GPU Code

For code that runs on GPU, make sure to test both CPU and GPU versions:

```python
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_compatibility(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU test")
    
    # Create components on the specified device
    atom_reader = AtomFileReader("test_data/small_system.xyz", box_length=10.0, 
                                 cutoff=2.5, device=device)
    lj_force = LennardJonesForce(atom_reader)
    
    # Test operations
    energy = lj_force.compute_energy(atom_reader.positions)
    forces = lj_force.compute_forces(atom_reader.positions)
    
    # Verify results are on the correct device
    assert energy.device.type == device
    assert forces.device.type == device
```

## Test Data

Store test data files in the `tests/test_data/` directory:

```
tests/
├── test_data/
│   ├── small_system.xyz
│   ├── water_box.xyz
│   └── ...
```

## Continuous Integration

TorchSim uses GitHub Actions for continuous integration. Every pull request is automatically tested to ensure it doesn't break existing functionality.

The CI configuration runs:
- Linting checks (black, isort, flake8)
- Unit tests
- Integration tests
- Documentation build

## Code Coverage

To check code coverage:

```bash
pytest --cov=torchsim
```

Aim for high test coverage, especially for critical components:
- Force fields
- Integrators
- Core simulation classes 