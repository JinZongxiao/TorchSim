# Contributing to TorchSim

Thank you for considering contributing to TorchSim! This document provides guidelines and instructions for contributing to the project.

## Getting Started

1. **Fork the repository**: Start by forking the [TorchSim repository](https://github.com/JinZongxiao/torchsim)
2. **Clone your fork**: `git clone https://github.com/YOUR-USERNAME/torchsim.git`
3. **Install in development mode**: `pip install -e ".[dev]"`
4. **Create a branch**: `git checkout -b feature/your-feature-name`

## Development Environment

To set up a development environment:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Check code style
black .
isort .

# Run tests
pytest
```

## Project Structure

Please familiarize yourself with the [project structure](project_structure.md) before contributing.

## Contribution Guidelines

### Code Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines with a few modifications:

- Use 4 spaces for indentation
- Maximum line length is 88 characters (compatible with Black)
- Use Google-style docstrings
- Use type hints for function signatures

You can automatically format your code using:

```bash
black .
isort .
```

### Pull Request Process

1. **Update your fork**: Make sure your fork is up to date with the main repository
2. **Develop your feature**: Make your changes, following the coding guidelines
3. **Write tests**: Add tests for your new functionality
4. **Run tests**: Ensure all tests pass by running `pytest`
5. **Submit a pull request**: Create a pull request against the main repository

### Pull Request Checklist

Before submitting your pull request, make sure:

- [ ] Your code follows the project's style guide
- [ ] You've added tests for your changes
- [ ] All tests pass
- [ ] Your code includes appropriate documentation
- [ ] You've updated any relevant documentation files
- [ ] You've added your change to the CHANGELOG.md file

## Adding New Features

### Force Fields

When adding a new force field:

1. Create a new file in `torchsim/core/force/`
2. Extend the `BaseForce` class
3. Implement `compute_energy` and `compute_forces` methods
4. Add tests in `tests/core/force/`
5. Update documentation in `docs/user_guide/force_fields.md`

### Integrators

When adding a new integrator:

1. Create a new file in `torchsim/core/integrator/`
2. Extend the `BaseIntegrator` class
3. Implement the `step` method
4. Add tests in `tests/core/integrator/`
5. Update documentation in `docs/user_guide/integrators.md`

### Machine Learning Models

When adding a new machine learning model:

1. Create a new file in `torchsim/machine_learning_potentials/model/`
2. Extend the appropriate base class
3. Update `MachineLearningForce` to support your model
4. Add tests in `tests/machine_learning_potentials/`
5. Update documentation in `docs/user_guide/ml_potentials.md`

## Reporting Bugs

When reporting bugs, please include:

1. A clear description of the bug
2. Steps to reproduce the issue
3. Expected behavior
4. Actual behavior
5. System information (OS, Python version, TorchSim version, PyTorch version)
6. If possible, a minimal code example that reproduces the issue

## Requesting Features

When requesting features, please include:

1. A clear description of the proposed feature
2. The motivation behind the feature
3. If possible, suggestions for implementation

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

### Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable behavior and are expected to take appropriate and fair corrective action in response to any instances of unacceptable behavior.

## License

By contributing to TorchSim, you agree that your contributions will be licensed under the project's MIT License. 