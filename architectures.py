import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import *
from .layers import *
from .utils import round_filters, round_repeats
from .core import get_magnitude_output, check_complex_input

class ComplexResNetTensorParallel(nn.Module):
    """Complex-valued ResNet architecture with tensor parallelism."""
    def __init__(self, block, layers, num_classes=10, input_channels=3, device_real=None, device_imag=None):
        super().__init__()
        self.in_channels = 64
        self.device_real = device_real
        self.device_imag = device_imag

        # Initial convolution
        self.conv1 = ComplexConv2dTensorParallel(
            input_channels, 64, kernel_size=3, stride=1, padding=1, bias=True,
            device_real=device_real, device_imag=device_imag
        )
        self.bn1_r = nn.BatchNorm2d(64).to(device_real)
        self.bn1_i = nn.BatchNorm2d(64).to(device_imag)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = ComplexLinearTensorParallel(
            512 * block.expansion, num_classes, 
            device_real=device_real, device_imag=device_imag
        )

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = ComplexSequentialTensorParallel(
                ComplexConv2dTensorParallel(
                    self.in_channels, out_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=True,
                    device_real=self.device_real, device_imag=self.device_imag
                ),
                NaiveComplexBatchNorm2dTensorParallel(
                    out_channels * block.expansion, 
                    device_real=self.device_real, device_imag=self.device_imag
                )
            )

        layers = []
        layers.append(block(
            self.in_channels, out_channels, stride, downsample,
            device_real=self.device_real, device_imag=self.device_imag, bias=False
        ))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(
                self.in_channels, out_channels,
                device_real=self.device_real, device_imag=self.device_imag, bias=False
            ))

        return ComplexSequentialTensorParallel(*layers)

    def forward(self, x):
        x_r, x_i = check_complex_input(x)

        x_r, x_i = self.conv1((x_r, x_i))
        
        # Apply BN and ReLU after initial conv
        x_r = self.bn1_r(x_r.to(self.device_real, non_blocking=True))
        x_i = self.bn1_i(x_i.to(self.device_imag, non_blocking=True))
        x_r, x_i = F.relu(x_r), F.relu(x_i)

        x_r, x_i = self.layer1((x_r, x_i))
        x_r, x_i = self.layer2((x_r, x_i))
        x_r, x_i = self.layer3((x_r, x_i))
        x_r, x_i = self.layer4((x_r, x_i))

        x_r = self.avgpool(x_r)
        x_i = self.avgpool(x_i)

        out_r, out_i = self.fc((x_r, x_i))

        return get_magnitude_output(out_r, out_i, self.device_real)

def complex_resnet34_tensor_parallel(num_classes=10, input_channels=3, 
                                   device_real=None, device_imag=None):
    """Create Complex ResNet34 with tensor parallelism"""
    return ComplexResNetTensorParallel(
        ComplexBasicBlockTensorParallel, [3, 4, 6, 3], 
        num_classes, input_channels, device_real, device_imag
    )

class TensorParallelEfficientNetB6(nn.Module):
    def __init__(self, num_classes=10, width_coefficient=1.8, depth_coefficient=2.6, 
                 head_dropout_rate=0.5, block_drop_connect_rate=0.0, 
                 device_real=None, device_imag=None, bias=False):
        super().__init__()
        self.device_real = device_real
        self.device_imag = device_imag
        
        # EfficientNet-B0 base settings
        b0_settings = [
            # k, s, e, o, n, se
            [3, 1, 1,  16, 1, 0.25],  # Stage 0 -> 1
            [3, 2, 6,  24, 2, 0.25],  # Stage 1 -> 2
            [5, 2, 6,  40, 2, 0.25],  # Stage 2 -> 3
            [3, 2, 6,  80, 3, 0.25],  # Stage 3 -> 4
            [5, 1, 6, 112, 3, 0.25],  # Stage 4 -> 5
            [5, 2, 6, 192, 4, 0.25],  # Stage 5 -> 6
            [3, 1, 6, 320, 1, 0.25],  # Stage 6 -> 7
        ]
        
        # Stem (Optimized Conv) - all weights on device_real
        stem_out_channels = round_filters(32, width_coefficient)
        self.stem_weight_r = nn.Parameter(torch.randn(stem_out_channels, 3, 3, 3, device=device_real))
        self.stem_weight_i = nn.Parameter(torch.randn(stem_out_channels, 3, 3, 3, device=device_real))
        nn.init.kaiming_normal_(self.stem_weight_r, nonlinearity='relu')
        nn.init.kaiming_normal_(self.stem_weight_i, nonlinearity='relu')
        # Stem usually doesn't have bias before BN
        self.stem_bias_r = None 
        self.stem_bias_i = None 
        self.stem_bn = NaiveComplexBatchNorm2dTensorParallel(stem_out_channels, device_real=device_real, device_imag=device_imag)
        
        # Blocks
        in_channels = stem_out_channels
        layers = []
        for k_size, stride_val, exp_ratio, base_out_c, base_num_b, se_r in b0_settings:
            out_channels_block = round_filters(base_out_c, width_coefficient)
            num_blocks_stage = round_repeats(base_num_b, depth_coefficient)
            for i in range(num_blocks_stage):
                s = stride_val if i == 0 else 1
                layers.append(ComplexMBConvBlockTensorParallel(
                    in_channels=(in_channels if i == 0 else out_channels_block),
                    out_channels=out_channels_block,
                    expansion=exp_ratio,
                    stride=s,
                    kernel_size=k_size,
                    se_ratio=se_r,
                    drop_connect_rate=block_drop_connect_rate,
                    device_real=device_real,
                    device_imag=device_imag,
                    bias=bias
                ))
                in_channels = out_channels_block
        self.blocks = nn.Sequential(*layers)
        
        # Head (Optimized Conv) - all weights on device_real
        head_in_channels = round_filters(b0_settings[-1][3], width_coefficient)
        head_intermediate_channels = round_filters(1280, width_coefficient)
        self.head_weight_r = nn.Parameter(torch.randn(head_intermediate_channels, head_in_channels, 1, 1, device=device_real))
        self.head_weight_i = nn.Parameter(torch.randn(head_intermediate_channels, head_in_channels, 1, 1, device=device_real))
        nn.init.kaiming_normal_(self.head_weight_r, nonlinearity='relu')
        nn.init.kaiming_normal_(self.head_weight_i, nonlinearity='relu')
        # Head conv usually doesn't have bias before BN
        self.head_bias_r = None 
        self.head_bias_i = None 
        self.head_bn = NaiveComplexBatchNorm2dTensorParallel(head_intermediate_channels, device_real=device_real, device_imag=device_imag)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = ComplexDropoutTensorParallel(head_dropout_rate, device_real=device_real, device_imag=device_imag) if head_dropout_rate > 0 else nn.Identity()
        self.fc = ComplexLinearTensorParallel(head_intermediate_channels, num_classes, device_real=device_real, device_imag=device_imag, bias=True)

    def _optimized_conv_step(self, x_r, x_i, weight_r, weight_i, bias_r, bias_i, stride, padding, groups=1):
        # Reuse the optimized convolution step from original code
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

    def forward(self, x):
        x_r, x_i = check_complex_input(x)
        x_r = x_r.to(self.device_real)
        x_i = x_i.to(self.device_imag)
            
        # Stem
        out_r, out_i = self._optimized_conv_step(x_r, x_i, self.stem_weight_r, self.stem_weight_i, 
                                               self.stem_bias_r, self.stem_bias_i, stride=2, padding=1, groups=1)
        out_r, out_i = self.stem_bn((out_r, out_i))
        out_r, out_i = complex_silu_tensorparallel((out_r, out_i), device_real=self.device_real, device_imag=self.device_imag)
        
        # Blocks
        out_r, out_i = self.blocks((out_r, out_i))
        
        # Head
        out_r, out_i = self._optimized_conv_step(out_r, out_i, self.head_weight_r, self.head_weight_i, 
                                               self.head_bias_r, self.head_bias_i, stride=1, padding=0, groups=1)
        out_r, out_i = self.head_bn((out_r, out_i))
        out_r, out_i = complex_silu_tensorparallel((out_r, out_i), device_real=self.device_real, device_imag=self.device_imag)
        
        # Pooling and classification
        out_r = self.avg_pool(out_r)
        out_i = self.avg_pool(out_i)
        out_r = out_r.view(out_r.size(0), -1)
        out_i = out_i.view(out_i.size(0), -1)
        out_r, out_i = self.dropout((out_r, out_i))
        out_r, out_i = self.fc((out_r, out_i))
        
        return get_magnitude_output(out_r, out_i, self.device_real)
