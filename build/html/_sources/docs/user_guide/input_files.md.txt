# Input File Formats

TorchSim supports several input file formats for molecular structures. This page details how to prepare and use these files.

## XYZ Format

The XYZ file format is the primary structure format supported by TorchSim. It is a simple text format that contains:
- The number of atoms in the first line
- A comment line (usually blank or a title)
- One line per atom with the format: `<atom_type> <x> <y> <z>`

Example:
```
3
Water molecule
O   0.000   0.000   0.000
H   0.758   0.586   0.000
H  -0.758   0.586   0.000
```

To load an XYZ file in TorchSim:

```python
from torchsim.io_utils.reader import AtomFileReader

atom_reader = AtomFileReader(
    "your_structure.xyz", 
    box_length=10.0,  # Simulation box size in Angstroms
    cutoff=2.5        # Cutoff distance for interactions
)
```

## Custom File Formats

To support custom file formats, you can extend the `AtomFileReader` class:

```python
from torchsim.io_utils.reader import AtomFileReader

class CustomFormatReader(AtomFileReader):
    def __init__(self, file_path, box_length, cutoff):
        super().__init__(file_path, box_length, cutoff)
        
    def _parse_file(self, file_path):
        # Your custom parsing logic here
        # Should set:
        # - self.positions: torch tensor of shape (n_atoms, 3)
        # - self.atom_types: list of atom types (strings)
        # - self.n_atoms: number of atoms
        pass
```

## Input File Parameters

When reading input files, the following parameters can be specified:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `file_path` | Path to the structure file | Required |
| `box_length` | Length of the simulation box in Angstroms | Required |
| `cutoff` | Cutoff distance for interactions in Angstroms | Required |
| `pbc` | Whether to use periodic boundary conditions | True |
| `device` | PyTorch device to use (e.g., 'cpu', 'cuda') | 'cuda' if available, else 'cpu' |

## Converting Between Formats

TorchSim provides utilities to convert between different file formats:

```python
from torchsim.io_utils.file_converter import convert_xyz_to_pdb

convert_xyz_to_pdb("structure.xyz", "structure.pdb")
``` 