
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.autograd import SharedParameterToDevice

class ComplexConv2dModelParallel(nn.Module):
    """Complex 2D Convolution Layer for Model Parallelism"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, bias=True, groups=1, device_real=None, device_imag=None):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.device_real = device_real
        self.device_imag = device_imag

        if isinstance(kernel_size, int):
            kernel_size_tuple = (kernel_size, kernel_size)
        else:
            kernel_size_tuple = kernel_size

        self.weight_r = nn.Parameter(torch.randn(
            out_channels, in_channels // groups, *kernel_size_tuple, device=device_real))
        self.weight_i = nn.Parameter(torch.randn(
            out_channels, in_channels // groups, *kernel_size_tuple, device=device_imag))
        
        nn.init.kaiming_normal_(self.weight_r, nonlinearity=\'relu\')
        nn.init.kaiming_normal_(self.weight_i, nonlinearity=\'relu\')

        if bias:
            self.bias_r = nn.Parameter(torch.zeros(out_channels, device=device_real))
            self.bias_i = nn.Parameter(torch.zeros(out_channels, device=device_imag))
        else:
            self.register_parameter(\'bias_r\', None)
            self.register_parameter(\'bias_i\', None)

    def forward(self, x_tuple):
        x_r, x_i = x_tuple
        x_r = x_r.to(self.device_real, non_blocking=True)
        x_i = x_i.to(self.device_imag, non_blocking=True)

        # Real part operation
        x_i_for_real_calc = x_i.to(self.device_real, non_blocking=True)
        weight_i_on_real = SharedParameterToDevice.apply(self.weight_i, self.device_real)
        
        out_r = F.conv2d(x_r, self.weight_r, None, self.stride, self.padding, groups=self.groups) - \
                F.conv2d(x_i_for_real_calc, weight_i_on_real, None, self.stride, self.padding, groups=self.groups)
        
        if self.bias_r is not None:
            out_r = out_r + self.bias_r.view(1, -1, 1, 1)

        # Imaginary part operation
        x_r_for_imag_calc = x_r.to(self.device_imag, non_blocking=True)
        weight_r_on_imag = SharedParameterToDevice.apply(self.weight_r, self.device_imag)
        
        out_i = F.conv2d(x_r_for_imag_calc, self.weight_i, None, self.stride, self.padding, groups=self.groups) + \
                F.conv2d(x_i, weight_r_on_imag, None, self.stride, self.padding, groups=self.groups)
        
        if self.bias_i is not None:
            out_i = out_i + self.bias_i.view(1, -1, 1, 1)
            
        return out_r, out_i

class ComplexLinearModelParallel(nn.Module):
    """Complex Linear Layer for Model Parallelism"""
    def __init__(self, in_features, out_features, bias=True, device_real=None, device_imag=None):
        super().__init__()
        self.device_real = device_real
        self.device_imag = device_imag
        self.in_features = in_features
        self.out_features = out_features

        self.weight_r = nn.Parameter(torch.randn(out_features, in_features, device=device_real))
        self.weight_i = nn.Parameter(torch.randn(out_features, in_features, device=device_real))
        nn.init.kaiming_normal_(self.weight_r, nonlinearity=\'relu\')
        nn.init.kaiming_normal_(self.weight_i, nonlinearity=\'relu\')

        if bias:
            self.bias_r = nn.Parameter(torch.zeros(out_features, device=device_real))
            self.bias_i = nn.Parameter(torch.zeros(out_features, device=device_real))
        else:
            self.register_parameter(\'bias_r\', None)
            self.register_parameter(\'bias_i\', None)

    def forward(self, x_tuple):
        x_r, x_i = x_tuple
        x_r = x_r.to(self.device_real, non_blocking=True)
        x_i = x_i.to(self.device_imag, non_blocking=True)

        original_shape_r = x_r.shape
        needs_flattening = x_r.dim() > 2
        if needs_flattening:
            x_r = x_r.reshape(-1, x_r.size(-1))
            x_i = x_i.reshape(-1, x_i.size(-1))
        
        # Partial calculation
        term_rr = F.linear(x_r, self.weight_r, None)
        term_ri = F.linear(x_r, self.weight_i, None)

        weight_r_on_imag = SharedParameterToDevice.apply(self.weight_r, self.device_imag)
        weight_i_on_imag = SharedParameterToDevice.apply(self.weight_i, self.device_imag)
        term_ii = F.linear(x_i, weight_i_on_imag, None)
        term_ir = F.linear(x_i, weight_r_on_imag, None)

        # Final calculation
        out_r = term_rr - term_ii.to(self.device_real, non_blocking=True)
        if self.bias_r is not None:
            out_r = out_r + self.bias_r
        
        out_i = term_ri.to(self.device_imag, non_blocking=True) + term_ir
        if self.bias_i is not None:
            bias_i_on_imag = SharedParameterToDevice.apply(self.bias_i, self.device_imag)
            out_i = out_i + bias_i_on_imag
        
        if needs_flattening:
            out_r = out_r.view(*original_shape_r[:-1], -1)
            out_i = out_i.view(*original_shape_r[:-1], -1)

        return out_r, out_i

class ComplexDropoutModelParallel(nn.Module):
    """Complex Dropout Layer for Model Parallelism"""
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
        binary_mask = (torch.rand_like(x_r) < keep_prob)
        
        scale = 1.0 / keep_prob if keep_prob > 0 else 0.0
        
        shared_mask_r = binary_mask.type_as(x_r) * scale
        shared_mask_i = shared_mask_r.to(self.device_imag, non_blocking=True).type_as(x_i)
        
        out_r = x_r * shared_mask_r
        out_i = x_i * shared_mask_i
        return out_r, out_i

