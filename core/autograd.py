
import torch

class SharedParameterToDevice(torch.autograd.Function):
    """
    Custom autograd function for transferring shared parameters to different devices
    in model parallelism.
    """
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
        return grad_output.to(ctx.input_device), None

