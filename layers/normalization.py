
import torch
import torch.nn as nn
from ..core.autograd import SharedParameterToDevice

class NaiveComplexBatchNorm2dModelParallel(nn.Module):
    """Complex BatchNorm2d Layer for Model Parallelism"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, 
                 track_running_stats=True, device_real=None, device_imag=None):
        super().__init__()
        self.device_real = device_real
        self.device_imag = device_imag
        
        self.bn_r = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats).to(device_real)
        self.bn_i = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats).to(device_imag)
        
    def forward(self, x_tuple):
        x_r, x_i = x_tuple
        if not (torch.is_tensor(x_r) and torch.is_tensor(x_i)):
            raise TypeError("Input must be a tuple of tensors (real, imag)")
        
        out_r = self.bn_r(x_r.to(self.device_real, non_blocking=True))
        out_i = self.bn_i(x_i.to(self.device_imag, non_blocking=True))
            
        return out_r, out_i

class ComplexLayerRMSNormModelParallel(nn.Module):
    """Complex RMS Normalization Layer for Model Parallelism"""
    def __init__(self, dim, eps=1e-6, device_real=None, device_imag=None):
        super().__init__()
        self.eps = eps
        self.device_real = device_real
        self.device_imag = device_imag
        self.scale_factor = dim ** -0.5

        self.gamma_r = nn.Parameter(torch.ones(dim, device=device_real))
        self.gamma_i = nn.Parameter(torch.zeros(dim, device=device_imag))

    def forward(self, x_tuple):
        x_r, x_i = x_tuple
        x_r = x_r.to(self.device_real, non_blocking=True)
        x_i = x_i.to(self.device_imag, non_blocking=True)

        # RMS calculation
        x_i_sq_on_real = (x_i**2).to(self.device_real, non_blocking=True)
        abs_sq = x_r**2 + x_i_sq_on_real 
        
        rms_val_sq = torch.mean(abs_sq, dim=-1, keepdim=True) 
        rms_val = torch.sqrt(rms_val_sq + self.eps)
        
        # Normalization
        normed_r = x_r / rms_val
        rms_val_on_imag = rms_val.to(self.device_imag, non_blocking=True)
        normed_i = x_i / rms_val_on_imag

        # Apply gamma
        normed_i_on_real = normed_i.to(self.device_real, non_blocking=True)
        gamma_i_on_real = SharedParameterToDevice.apply(self.gamma_i, self.device_real)
        
        out_r = (normed_r * self.gamma_r - normed_i_on_real * gamma_i_on_real) * self.scale_factor

        normed_r_on_imag = normed_r.to(self.device_imag, non_blocking=True)
        gamma_r_on_imag = SharedParameterToDevice.apply(self.gamma_r, self.device_imag)

        out_i = (normed_r_on_imag * self.gamma_i + normed_i * gamma_r_on_imag) * self.scale_factor
        
        return out_r, out_i

