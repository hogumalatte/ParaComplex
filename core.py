import torch
import torch.nn.functional as F

class SharedParameterToDevice(torch.autograd.Function):
    """Custom autograd function for sharing parameters across devices"""
    @staticmethod
    def forward(ctx, param, target_device):
        if param is None:
            return None
        ctx.input_device = param.device
        return param.to(target_device)
        
    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None
        # Gradient accumulation for shared parameters happens automatically in PyTorch
        # when .backward() is called on the final loss.
        # Returning grad_output.to(ctx.input_device) ensures the gradient computed
        # on device_imag is added to the .grad attribute of the parameter on device_real.
        return grad_output.to(ctx.input_device), None

def complex_cross_entropy_loss(output_real, output_imag, target, device_real):
    """Complex cross entropy loss using magnitude"""
    calculation_device = device_real
    output_real_calc = output_real.to(calculation_device, non_blocking=True)
    output_imag_calc = output_imag.to(calculation_device, non_blocking=True)
    target_calc = target.to(calculation_device, non_blocking=True)
    magnitude = torch.sqrt(output_real_calc**2 + output_imag_calc**2 + 1e-9)
    loss = F.cross_entropy(magnitude, target_calc)
    return loss

def set_devices(device_real="cuda:0", device_imag="cuda:1"):
    """Set default devices for tensor parallelism"""
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device("cpu"), torch.device("cpu")
    
    if torch.cuda.device_count() < 2:
        print("Less than 2 GPUs available, using single GPU")
        return torch.device("cuda:0"), torch.device("cuda:0")
    
    return torch.device(device_real), torch.device(device_imag)

def check_complex_input(x):
    """Convert input to complex format if needed"""
    if torch.is_complex(x):
        return x.real, x.imag
    else:
        return x, torch.zeros_like(x)

def get_magnitude_output(output_real, output_imag, device_real):
    """Calculate magnitude from complex output for final prediction"""
    output_real_calc = output_real.to(device_real, non_blocking=True)
    output_imag_calc = output_imag.to(device_real, non_blocking=True)
    return torch.sqrt(output_real_calc**2 + output_imag_calc**2 + 1e-9)
