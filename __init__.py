"""
ParaComplex: Complex-Valued Neural Networks with Model Parallelism
=============================================================

A library for model parallelism of complex-valued neural networks for PyTorch.

Key Features:
- Tensor parallelism for complex-valued operations
- Implementation of complex versions of ResNet, EfficientNet, etc.
- GPU memory efficient model parallelism
"""

from .core import *
from .layers import *
from .blocks import *
from .models import *
from .utils import *

__version__ = "0.1.0"
__author__ = "ParaComplex Team"
__license__ = "MIT"

__all__ = [
    # Core
    'SharedParameterToDevice',
    'complex_relu_modelparallel',
    'complex_silu_modelparallel',
    
    # Basic Layers
    'ComplexConv2dModelParallel',
    'ComplexLinearModelParallel',
    'ComplexDropoutModelParallel',
    
    # Normalization
    'NaiveComplexBatchNorm2dModelParallel',
    'ComplexLayerRMSNormModelParallel',
    
    # Containers
    'ComplexSequentialModelParallel',
    
    # Utils
    'round_filters',
    'round_repeats',
    'exists',
]

