"""
ParaComplex: A library for complex-valued neural networks with tensor parallelism
"""

# Core utilities
from .core import (
    SharedParameterToDevice,
    complex_cross_entropy_loss,
    set_devices,
    check_complex_input,
    get_magnitude_output
)

# Complex layers for tensor parallelism
from .layers import (
    ComplexConv2dTensorParallel,
    ComplexLinearTensorParallel,
    NaiveComplexBatchNorm2dTensorParallel,
    ComplexDropoutTensorParallel,
    ComplexSequentialTensorParallel,
    complex_relu_tensorparallel,
    complex_silu_tensorparallel
)

# Model building blocks
from .models import (
    ComplexBasicBlockTensorParallel,
    ComplexSEBlockTensorParallel,
    ComplexMBConvBlockTensorParallel
)

# Complete model architectures
from .architectures import (
    ComplexResNetTensorParallel,
    TensorParallelEfficientNetB6,
    complex_resnet34_tensor_parallel
)

# Utility functions
from .utils import (
    round_filters,
    round_repeats
)

__version__ = "1.0.0"
__author__ = "ParaComplex Team"

__all__ = [
    # Core
    'SharedParameterToDevice',
    'complex_cross_entropy_loss',
    'set_devices', 
    'check_complex_input',
    'get_magnitude_output',
    
    # Layers
    'ComplexConv2dTensorParallel',
    'ComplexLinearTensorParallel', 
    'NaiveComplexBatchNorm2dTensorParallel',
    'ComplexDropoutTensorParallel',
    'ComplexSequentialTensorParallel',
    'complex_relu_tensorparallel',
    'complex_silu_tensorparallel',
    
    # Models
    'ComplexBasicBlockTensorParallel',
    'ComplexSEBlockTensorParallel',
    'ComplexMBConvBlockTensorParallel',
    
    # Architectures
    'ComplexResNetTensorParallel',
    'TensorParallelEfficientNetB6',
    'complex_resnet34_tensor_parallel',
    
    # Utils
    'round_filters',
    'round_repeats'
]
