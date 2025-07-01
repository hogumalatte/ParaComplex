# ParaComplex

ParaComplex is a Python library for deep learning research and experiments based on the para-complex number system. It provides various neural network layers and models using complex and para-complex numbers.

## Features
- Implementation of layers and models based on para-complex numbers
- Neural network structures compatible with PyTorch
- Utilities for research and experiments

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/your-username/ParaComplex.git
cd ParaComplex
pip install -e .
```

## Usage Example

```python
from ParaComplex import layers, models

# Example: Create a para-complex linear layer
layer = layers.ParaComplexLinear(in_features=128, out_features=64)

# Example: Create a model
model = models.ParaComplexMLP(input_dim=128, hidden_dim=64, output_dim=10)
```

## Directory Structure
- `layers.py`: Para-complex and complex layer implementations
- `models.py`: Various neural network model implementations
- `architectures.py`: Neural network architecture definitions
- `core.py`: Core operations and number system definitions
- `utils.py`: Utility functions
- `examples/`: Example usage and experiment code

## License

This project is licensed under the MIT License. See the LICENSE file for details. 