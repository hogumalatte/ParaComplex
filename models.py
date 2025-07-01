import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *
from .core import SharedParameterToDevice

class ComplexBasicBlockTensorParallel(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, 
                 device_real=None, device_imag=None, bias=False):
        super().__init__()
        self.device_real = device_real
        self.device_imag = device_imag
        self.stride = stride
        self.downsample = downsample

        # First conv layer parameters (all stored on device_real)
        self.weight1r = nn.Parameter(torch.randn(out_channels, in_channels, 3, 3, device=device_real))
        self.weight1i = nn.Parameter(torch.randn(out_channels, in_channels, 3, 3, device=device_real))
        nn.init.kaiming_normal_(self.weight1r, nonlinearity='relu')
        nn.init.kaiming_normal_(self.weight1i, nonlinearity='relu')
        
        self.bias1r = nn.Parameter(torch.zeros(out_channels, device=device_real)) if bias else None
        self.bias1i = nn.Parameter(torch.zeros(out_channels, device=device_real)) if bias else None
        
        # Second conv layer parameters (all stored on device_real)
        self.weight2r = nn.Parameter(torch.randn(out_channels, out_channels, 3, 3, device=device_real))
        self.weight2i = nn.Parameter(torch.randn(out_channels, out_channels, 3, 3, device=device_real))
        nn.init.kaiming_normal_(self.weight2r, nonlinearity='relu')
        nn.init.kaiming_normal_(self.weight2i, nonlinearity='relu')
        
        self.bias2r = nn.Parameter(torch.zeros(out_channels, device=device_real)) if bias else None
        self.bias2i = nn.Parameter(torch.zeros(out_channels, device=device_real)) if bias else None

        # Batch normalization layers (device-specific parameters)
        self.bn1_r = nn.BatchNorm2d(out_channels).to(device_real)
        self.bn1_i = nn.BatchNorm2d(out_channels).to(device_imag)
        self.bn2_r = nn.BatchNorm2d(out_channels).to(device_real)
        self.bn2_i = nn.BatchNorm2d(out_channels).to(device_imag)

    def forward(self, x_tuple):
        x_r, x_i = x_tuple
        identity_r, identity_i = x_r, x_i

        # First convolution block (optimized computation from original code)
        # Partial Computations
        x_r_real = x_r.to(self.device_real, non_blocking=True)
        term11 = F.conv2d(x_r_real, self.weight1r, None, stride=self.stride, padding=1)
        term13 = F.conv2d(x_r_real, self.weight1i, None, stride=self.stride, padding=1)

        x_i_imag = x_i.to(self.device_imag, non_blocking=True)
        weight1r_imag = SharedParameterToDevice.apply(self.weight1r, self.device_imag)
        weight1i_imag = SharedParameterToDevice.apply(self.weight1i, self.device_imag)
        term12 = F.conv2d(x_i_imag, weight1i_imag, None, stride=self.stride, padding=1)  # Im(X)*Im(W1)
        term14 = F.conv2d(x_i_imag, weight1r_imag, None, stride=self.stride, padding=1)  # Im(X)*Re(W1)

        # Exchange & Final Computation
        re_z1 = term11 - term12.to(self.device_real, non_blocking=True)
        im_z1 = term13.to(self.device_imag, non_blocking=True) + term14

        # Bias1
        if self.bias1r is not None: 
            re_z1 = re_z1 + self.bias1r.view(1, -1, 1, 1)
        if self.bias1i is not None: 
            im_z1 = im_z1 + SharedParameterToDevice.apply(self.bias1i, self.device_imag).view(1, -1, 1, 1)
        
        # BN1 & Activation
        y_r, y_i = self.bn1_r(re_z1), self.bn1_i(im_z1)
        y_r, y_i = F.relu(y_r), F.relu(y_i)

        # Second convolution block (optimized computation from original code)
        # Partial Computations
        y_r_real = y_r.to(self.device_real, non_blocking=True)
        term21 = F.conv2d(y_r_real, self.weight2r, None, stride=1, padding=1)
        term23 = F.conv2d(y_r_real, self.weight2i, None, stride=1, padding=1)

        y_i_imag = y_i.to(self.device_imag, non_blocking=True)
        weight2r_imag = SharedParameterToDevice.apply(self.weight2r, self.device_imag)
        weight2i_imag = SharedParameterToDevice.apply(self.weight2i, self.device_imag)
        term22 = F.conv2d(y_i_imag, weight2i_imag, None, stride=1, padding=1)  # Im(Y)*Im(W2)
        term24 = F.conv2d(y_i_imag, weight2r_imag, None, stride=1, padding=1)  # Im(Y)*Re(W2)

        # Exchange & Final Computation
        re_z2 = term21 - term22.to(self.device_real, non_blocking=True)
        im_z2 = term23.to(self.device_imag, non_blocking=True) + term24

        # Bias2
        if self.bias2r is not None: 
            re_z2 = re_z2 + self.bias2r.view(1, -1, 1, 1)
        if self.bias2i is not None: 
            im_z2 = im_z2 + SharedParameterToDevice.apply(self.bias2i, self.device_imag).view(1, -1, 1, 1)

        # BN2 & Shortcut & Final Activation
        out_r_pre_shortcut, out_i_pre_shortcut = self.bn2_r(re_z2), self.bn2_i(im_z2)

        # Shortcut
        if self.downsample is not None:
            identity_r_processed, identity_i_processed = self.downsample((identity_r, identity_i))
        else:
            identity_r_processed = identity_r.to(self.device_real, non_blocking=True)
            identity_i_processed = identity_i.to(self.device_imag, non_blocking=True)

        # Add shortcut
        out_r_final = out_r_pre_shortcut + identity_r_processed
        out_i_final = out_i_pre_shortcut + identity_i_processed

        # Final Activation
        out_r_final, out_i_final = F.relu(out_r_final), F.relu(out_i_final)
        return out_r_final, out_i_final

class ComplexSEBlockTensorParallel(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25, device_real=None, device_imag=None):
        super().__init__()
        reduced_channels = max(1, int(in_channels * se_ratio))
        self.fc1 = ComplexLinearTensorParallel(in_channels, reduced_channels, device_real=device_real, device_imag=device_imag)
        self.fc2 = ComplexLinearTensorParallel(reduced_channels, in_channels, device_real=device_real, device_imag=device_imag)
        self.device_real = device_real
        self.device_imag = device_imag
        
    def forward(self, x_tuple):
        x_r, x_i = x_tuple
        b, c, _, _ = x_r.shape
        se_r = x_r.mean(dim=(2, 3))
        se_i = x_i.mean(dim=(2, 3))
        se_r, se_i = self.fc1((se_r, se_i))
        se_r, se_i = complex_silu_tensorparallel((se_r, se_i), device_real=self.device_real, device_imag=self.device_imag)
        se_r, se_i = self.fc2((se_r, se_i))
        
        # Calculate magnitude for scaling (both computed on device_real)
        se_r_calc = se_r.to(self.device_real, non_blocking=True)
        se_i_calc = se_i.to(self.device_real, non_blocking=True)
        se_mag = torch.sqrt(se_r_calc ** 2 + se_i_calc ** 2 + 1e-9)
        se_mag = torch.sigmoid(se_mag)
        se_mag = se_mag.view(b, c, 1, 1)
        
        # Apply magnitude to both real and imaginary parts
        out_r = x_r * se_mag.to(self.device_real, non_blocking=True)
        out_i = x_i * se_mag.to(self.device_imag, non_blocking=True)
        
        return (out_r, out_i)

class ComplexMBConvBlockTensorParallel(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride, kernel_size=3, se_ratio=0.25, drop_connect_rate=0.0, device_real=None, device_imag=None, bias=False):
        super().__init__()
        mid_channels = in_channels * expansion
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.device_real = device_real
        self.device_imag = device_imag
        self.drop_connect_rate = drop_connect_rate
        self.expansion = expansion
        self.stride = stride
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2

        # Expansion Conv Parameters (1x1) - all weights on device_real
        if expansion != 1:
            self.weight_expand_r = nn.Parameter(torch.randn(mid_channels, in_channels, 1, 1, device=device_real))
            self.weight_expand_i = nn.Parameter(torch.randn(mid_channels, in_channels, 1, 1, device=device_real))
            nn.init.kaiming_normal_(self.weight_expand_r, nonlinearity='relu')
            nn.init.kaiming_normal_(self.weight_expand_i, nonlinearity='relu')
            self.bias_expand_r = nn.Parameter(torch.randn(mid_channels, device=device_real)) if bias else None
            self.bias_expand_i = nn.Parameter(torch.randn(mid_channels, device=device_real)) if bias else None
            self.bn0 = NaiveComplexBatchNorm2dTensorParallel(mid_channels, device_real=device_real, device_imag=device_imag)
        else:
            self.weight_expand_r, self.weight_expand_i = None, None
            self.bias_expand_r, self.bias_expand_i = None, None
            self.bn0 = nn.Identity()

        # Depthwise Conv Parameters (k x k) - all weights on device_real
        self.dw_padding = padding
        self.dw_groups = mid_channels
        self.weight_dw_r = nn.Parameter(torch.randn(mid_channels, 1, kernel_size, kernel_size, device=device_real))
        self.weight_dw_i = nn.Parameter(torch.randn(mid_channels, 1, kernel_size, kernel_size, device=device_real))
        nn.init.kaiming_normal_(self.weight_dw_r, nonlinearity='relu')
        nn.init.kaiming_normal_(self.weight_dw_i, nonlinearity='relu')
        self.bias_dw_r = nn.Parameter(torch.randn(mid_channels, device=device_real)) if bias else None
        self.bias_dw_i = nn.Parameter(torch.randn(mid_channels, device=device_real)) if bias else None
        self.bn1 = NaiveComplexBatchNorm2dTensorParallel(mid_channels, device_real=device_real, device_imag=device_imag)

        # SE Block
        self.se = ComplexSEBlockTensorParallel(mid_channels, se_ratio=se_ratio, device_real=device_real, device_imag=device_imag)

        # Projection Conv Parameters (1x1) - all weights on device_real
        self.weight_proj_r = nn.Parameter(torch.randn(out_channels, mid_channels, 1, 1, device=device_real))
        self.weight_proj_i = nn.Parameter(torch.randn(out_channels, mid_channels, 1, 1, device=device_real))
        nn.init.kaiming_normal_(self.weight_proj_r, nonlinearity='relu')
        nn.init.kaiming_normal_(self.weight_proj_i, nonlinearity='relu')
        self.bias_proj_r = nn.Parameter(torch.randn(out_channels, device=device_real)) if bias else None
        self.bias_proj_i = nn.Parameter(torch.randn(out_channels, device=device_real)) if bias else None
        self.bn2 = NaiveComplexBatchNorm2dTensorParallel(out_channels, device_real=device_real, device_imag=device_imag)

    def _optimized_conv_step(self, x_r, x_i, weight_r, weight_i, bias_r, bias_i, stride, padding, groups):
        # Helper for optimized convolution (from original code)
        x_r_real = x_r.to(self.device_real, non_blocking=True)
        term_rr = F.conv2d(x_r_real, weight_r, None, stride=stride, padding=padding, groups=groups)
        term_ri = F.conv2d(x_r_real, weight_i, None, stride=stride, padding=padding, groups=groups)

        x_i_imag = x_i.to(self.device_imag, non_blocking=True)
        weight_r_imag = SharedParameterToDevice.apply(weight_r, self.device_imag)
        weight_i_imag = SharedParameterToDevice.apply(weight_i, self.device_imag)
        term_ii = F.conv2d(x_i_imag, weight_i_imag, None, stride=stride, padding=padding, groups=groups)
        term_ir = F.conv2d(x_i_imag, weight_r_imag, None, stride=stride, padding=padding, groups=groups)

        out_r = term_rr - term_ii.to(self.device_real, non_blocking=True)
        out_i = term_ri.to(self.device_imag, non_blocking=True) + term_ir

        if bias_r is not None: 
            out_r = out_r + bias_r.view(1, -1, 1, 1)
        if bias_i is not None: 
            out_i = out_i + SharedParameterToDevice.apply(bias_i, self.device_imag).view(1, -1, 1, 1)
        
        return out_r, out_i

    def _drop_connect(self, x_r, x_i, drop_prob):
        if not self.training or drop_prob == 0.:
            return x_r, x_i
        
        keep_prob = 1. - drop_prob
        shape = (x_r.shape[0],) + (1,) * (x_r.ndim - 1)  # (B, 1, 1, 1)

        # Generate mask on real device
        random_tensor_r = keep_prob + torch.rand(shape, dtype=x_r.dtype, device=self.device_real)
        random_tensor_r.floor_()  # Bernouilli
        
        # Scaled real part output
        out_r = x_r.div(keep_prob) * random_tensor_r
        
        # Copy same mask to imag device
        random_tensor_i = random_tensor_r.to(self.device_imag, non_blocking=True)
        # Scaled imaginary part output
        out_i = x_i.div(keep_prob) * random_tensor_i
        
        return out_r, out_i

    def forward(self, x_tuple):
        x_r, x_i = x_tuple
        identity_r, identity_i = x_r, x_i
        
        # Expansion Phase
        if self.expansion != 1:
            out_r, out_i = self._optimized_conv_step(x_r, x_i, 
                                                   self.weight_expand_r, self.weight_expand_i, 
                                                   self.bias_expand_r, self.bias_expand_i, 
                                                   stride=1, padding=0, groups=1)
            out_r, out_i = self.bn0((out_r, out_i))
            out_r, out_i = complex_silu_tensorparallel((out_r, out_i), device_real=self.device_real, device_imag=self.device_imag)
        else:
            out_r, out_i = x_r, x_i  # Pass through if no expansion

        # Depthwise Conv Phase
        out_r, out_i = self._optimized_conv_step(out_r, out_i,
                                               self.weight_dw_r, self.weight_dw_i,
                                               self.bias_dw_r, self.bias_dw_i,
                                               stride=self.stride, padding=self.dw_padding, groups=self.dw_groups)
        out_r, out_i = self.bn1((out_r, out_i))
        out_r, out_i = complex_silu_tensorparallel((out_r, out_i), device_real=self.device_real, device_imag=self.device_imag)

        # Squeeze-and-Excitation Phase
        out_r, out_i = self.se((out_r, out_i))

        # Projection Phase
        out_r, out_i = self._optimized_conv_step(out_r, out_i,
                                               self.weight_proj_r, self.weight_proj_i,
                                               self.bias_proj_r, self.bias_proj_i,
                                               stride=1, padding=0, groups=1)
        out_r, out_i = self.bn2((out_r, out_i))
        # No activation after projection

        # Residual Connection
        if self.use_residual:
            out_r, out_i = self._drop_connect(out_r, out_i, self.drop_connect_rate)
            out_r = out_r + identity_r.to(self.device_real, non_blocking=True)
            out_i = out_i + identity_i.to(self.device_imag, non_blocking=True)
            
        return (out_r, out_i)
