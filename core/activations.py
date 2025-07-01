
import torch
import torch.nn.functional as F

def complex_relu_modelparallel(x_tuple, device_real=None, device_imag=None):
    """Complex ReLU activation function (model parallel version)"""
    x_r, x_i = x_tuple
    if not (torch.is_tensor(x_r) and torch.is_tensor(x_i)):
        raise TypeError("Input must be a tuple of tensors (real, imag)")
    relu_r = F.relu(x_r.to(device_real, non_blocking=True))
    relu_i = F.relu(x_i.to(device_imag, non_blocking=True))
    return relu_r, relu_i

def complex_silu_modelparallel(x_tuple, device_real=None, device_imag=None):
    """Complex SiLU activation function (model parallel version)"""
    x_r, x_i = x_tuple
    if not (torch.is_tensor(x_r) and torch.is_tensor(x_i)):
        raise TypeError("Input must be a tuple of tensors (real, imag)")
    silu_r = F.silu(x_r.to(device_real, non_blocking=True))
    silu_i = F.silu(x_i.to(device_imag, non_blocking=True))
    return silu_r, silu_i

