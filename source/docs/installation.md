# Installation

## Dependencies

TorchSim requires the following dependencies:
- Python 3.8 or later
- PyTorch 1.9.0 or later
- PyTorch Geometric 2.0.0 or later
- NumPy 1.19.0 or later
- SciPy 1.6.0 or later
- Matplotlib 3.3.0 or later

## Installation Methods

### From PyPI

The easiest way to install TorchSim is through pip:

```bash
pip install torchsim
```

### From Source

For the latest development version, you can install directly from the source code:

```bash
git clone https://github.com/JinZongxiao/torchsim.git
cd torchsim
pip install -e .
```

The `-e` flag installs the package in development mode, allowing you to modify the source code and immediately see the effects without reinstalling.

### Development Installation

If you plan to contribute to TorchSim, install with development dependencies:

```bash
pip install -e ".[dev]"
```

This will install additional packages like pytest, black, and isort for testing and code formatting.

## Verifying Installation

After installation, you can verify that TorchSim is correctly installed by running:

```python
import torchsim
print(f"TorchSim version: {torchsim.__version__}")
``` 