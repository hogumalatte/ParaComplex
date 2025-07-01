import torch
import torch.nn as nn
import torch.nn.functional as F
from .core import SharedParameterToDevice

class ComplexConv2dTensorParallel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, groups=1, device_real=None, device_imag=None):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.device_real = device_real
        self.device_imag = device_imag

        # Convert kernel_size to tuple if it's an int
        if isinstance(kernel_size, int):
            kernel_size_tuple = (kernel_size, kernel_size)
        else:
            kernel_size_tuple = kernel_size

        # Both weights are stored on device_real
        self.weight_r = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size_tuple, device=device_real))
        self.weight_i = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size_tuple, device=device_real))
        nn.init.kaiming_normal_(self.weight_r, nonlinearity='relu')
        nn.init.kaiming_normal_(self.weight_i, nonlinearity='relu')

        if bias:
            # Both biases stored on device_real
            self.bias_r = nn.Parameter(torch.zeros(out_channels, device=device_real))
            self.bias_i = nn.Parameter(torch.zeros(out_channels, device=device_real))
        else:
            self.register_parameter('bias_r', None)
            self.register_parameter('bias_i', None)

    def forward(self, x_tuple): 
        x_r, x_i = x_tuple
        x_r = x_r.to(self.device_real, non_blocking=True)
        x_i = x_i.to(self.device_imag, non_blocking=True)

        # Real part computation (on device_real)
        x_i_for_real_calc = x_i.to(self.device_real, non_blocking=True)
        weight_i_on_real = SharedParameterToDevice.apply(self.weight_i, self.device_real)
        
        out_r = F.conv2d(x_r, self.weight_r, None, self.stride, self.padding, groups=self.groups) - \
                F.conv2d(x_i_for_real_calc, weight_i_on_real, None, self.stride, self.padding, groups=self.groups)
        if self.bias_r is not None:
            out_r = out_r + self.bias_r.view(1, -1, 1, 1)

        # Imaginary part computation (on device_imag)
        x_r_for_imag_calc = x_r.to(self.device_imag, non_blocking=True)
        weight_r_on_imag = SharedParameterToDevice.apply(self.weight_r, self.device_imag)
        weight_i_on_imag = SharedParameterToDevice.apply(self.weight_i, self.device_imag)
        
        out_i = F.conv2d(x_r_for_imag_calc, weight_i_on_imag, None, self.stride, self.padding, groups=self.groups) + \
                F.conv2d(x_i, weight_r_on_imag, None, self.stride, self.padding, groups=self.groups)
        if self.bias_i is not None:
            out_i = out_i + SharedParameterToDevice.apply(self.bias_i, self.device_imag).view(1, -1, 1, 1)
            
        return out_r, out_i

class ComplexLinearTensorParallel(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device_real=None, device_imag=None):
        super().__init__()
        self.device_real = device_real
        self.device_imag = device_imag
        self.in_features = in_features
        self.out_features = out_features

        # Both weights stored on device_real
        self.weight_r = nn.Parameter(torch.randn(out_features, in_features, device=device_real))
        self.weight_i = nn.Parameter(torch.randn(out_features, in_features, device=device_real))
        nn.init.kaiming_normal_(self.weight_r, nonlinearity='relu')
        nn.init.kaiming_normal_(self.weight_i, nonlinearity='relu')

        if bias:
            # Both biases stored on device_real
            self.bias_r = nn.Parameter(torch.zeros(out_features, device=device_real))
            self.bias_i = nn.Parameter(torch.zeros(out_features, device=device_real))
        else:
            self.register_parameter('bias_r', None)
            self.register_parameter('bias_i', None)

    def forward(self, x_tuple):
        x_r, x_i = x_tuple
        x_r = x_r.to(self.device_real, non_blocking=True)
        x_i = x_i.to(self.device_imag, non_blocking=True)

        # Fix: Correct flattening - preserve batch dimension
        needs_flattening = x_r.dim() > 2
        if needs_flattening:
            x_r = x_r.view(x_r.size(0), -1)  # (batch_size, -1)
            x_i = x_i.view(x_i.size(0), -1)  # (batch_size, -1)
        
        # Partial computations
        # On device_real:
        term_rr = F.linear(x_r, self.weight_r, None)  # Re(X)Re(W)
        term_ri = F.linear(x_r, self.weight_i, None)  # Re(X)Im(W)

        # On device_imag:
        weight_r_on_imag = SharedParameterToDevice.apply(self.weight_r, self.device_imag)
        weight_i_on_imag = SharedParameterToDevice.apply(self.weight_i, self.device_imag)
        term_ii = F.linear(x_i, weight_i_on_imag, None)  # Im(X)Im(W)
        term_ir = F.linear(x_i, weight_r_on_imag, None)  # Im(X)Re(W)

        # Final computation
        out_r = term_rr - term_ii.to(self.device_real, non_blocking=True)
        if self.bias_r is not None:
            out_r = out_r + self.bias_r
        
        out_i = term_ri.to(self.device_imag, non_blocking=True) + term_ir
        if self.bias_i is not None:
            bias_i_on_imag = SharedParameterToDevice.apply(self.bias_i, self.device_imag)
            out_i = out_i + bias_i_on_imag

        return out_r, out_i

class NaiveComplexBatchNorm2dTensorParallel(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device_real=None, device_imag=None):
        super().__init__()
        self.device_real = device_real
        self.device_imag = device_imag
        # bn_r created on device_real
        self.bn_r = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats).to(device_real)
        # bn_i created independently on device_imag (device-specific parameter)
        self.bn_i = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats).to(device_imag)
        
    def forward(self, x_tuple):
        x_r, x_i = x_tuple
        if not (torch.is_tensor(x_r) and torch.is_tensor(x_i)):
            raise TypeError("Input to NaiveComplexBatchNorm2dTensorParallel must be a tuple of tensors (real, imag)")
        
        # Real part processed on real device
        out_r = self.bn_r(x_r.to(self.device_real, non_blocking=True))
        
        # Imaginary part processed independently on imag device
        out_i = self.bn_i(x_i.to(self.device_imag, non_blocking=True))
            
        return out_r, out_i

class ComplexDropoutTensorParallel(nn.Module):
    def __init__(self, p=0.5, device_real=None, device_imag=None):
        super().__init__()
        self.p = p
        self.device_real = device_real
        self.device_imag = device_imag
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1")

    def forward(self, x_tuple):
        x_r, x_i = x_tuple
        x_r = x_r.to(self.device_real, non_blocking=True)
        x_i = x_i.to(self.device_imag, non_blocking=True)

        if not self.training or self.p == 0:
            return x_r, x_i

        keep_prob = 1 - self.p
        # binary_mask generated on device_real with same type as x_r
        binary_mask = (torch.rand_like(x_r) < keep_prob)
        
        scale = 1.0 / keep_prob if keep_prob > 0 else 0.0
        
        shared_mask_r = binary_mask.type_as(x_r) * scale  # on device_real
        # Copy shared_mask_r to device_imag with x_i's type
        shared_mask_i = shared_mask_r.to(self.device_imag, non_blocking=True).type_as(x_i) 
        
        out_r = x_r * shared_mask_r
        out_i = x_i * shared_mask_i
        return out_r, out_i

class ComplexSequentialTensorParallel(nn.Module):
    """Container to sequentially apply complex-valued modules with tensor parallelism."""
    def __init__(self, *args):
        super().__init__()
        self.modules_list = nn.ModuleList(args)

    def forward(self, x_tuple):
        x_r, x_i = x_tuple
        for module in self.modules_list:
            x_r, x_i = module((x_r, x_i))
        return (x_r, x_i)

def complex_relu_tensorparallel(x_tuple, device_real=None, device_imag=None):
    x_r, x_i = x_tuple
    if not (torch.is_tensor(x_r) and torch.is_tensor(x_i)):
        raise TypeError("Input to complex_relu_tensorparallel must be a tuple of tensors (real, imag)")
    relu_r = F.relu(x_r.to(device_real, non_blocking=True))
    relu_i = F.relu(x_i.to(device_imag, non_blocking=True))
    return relu_r, relu_i

def complex_silu_tensorparallel(x_tuple, device_real=None, device_imag=None):
    x_r, x_i = x_tuple
    if not (torch.is_tensor(x_r) and torch.is_tensor(x_i)):
        raise TypeError("Input to complex_silu_tensorparallel must be a tuple of tensors (real, imag)")
    silu_r = F.silu(x_r.to(device_real, non_blocking=True))
    silu_i = F.silu(x_i.to(device_imag, non_blocking=True))
    return silu_r, silu_i
