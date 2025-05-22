.. TorchSim documentation master file, created by
   sphinx-quickstart on Thu May 22 15:45:09 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TorchSim Documentation
======================

TorchSim is a molecular dynamics simulation framework based on PyTorch, integrating traditional molecular force fields and machine learning potential models.

Main features
--------

- **GPU acceleration**：Use PyTorch's CUDA support for high-performance simulation
- **Hybrid force fields**：Combine Lennard-Jones potential and machine learning potential
- **Flexible integrators**：Support Verlet algorithm and NVT ensemble
- **GNN support**：Integrate PyTorch Geometric for graph neural networks
- **Easy to extend**：Modular design allows easy addition of new force fields and integrators

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   docs/index.md
   docs/installation.md
   docs/quickstart.md
   
.. toctree::
   :maxdepth: 2
   :caption: User Guide:
   
   docs/user_guide/index.md
   docs/user_guide/input_files.md
   docs/user_guide/force_fields.md
   docs/user_guide/integrators.md
   
.. toctree::
   :maxdepth: 2
   :caption: API Reference:
   
   docs/api_reference/index.md
   docs/api_reference/core/md_simulation.md
   
.. toctree::
   :maxdepth: 2
   :caption: Developer Guide:
   
   docs/developer_guide/index.md
   docs/developer_guide/architecture.md
   docs/developer_guide/adding_force_fields.md
   docs/developer_guide/adding_integrators.md
   docs/developer_guide/ml_integration.md
   docs/developer_guide/project_structure.md
   docs/developer_guide/contributing.md
   docs/developer_guide/testing.md
   
.. toctree::
   :maxdepth: 2
   :caption: Examples:
   
   docs/examples/index.md
   docs/examples/lennard_jones_fluid.md

