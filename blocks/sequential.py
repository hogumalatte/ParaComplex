
import torch.nn as nn

class ComplexSequentialModelParallel(nn.Module):
    """Sequential container for complex model parallelism"""
    def __init__(self, *args):
        super().__init__()
        self.modules_list = nn.ModuleList(args)

    def forward(self, x_tuple):
        x_r, x_i = x_tuple
        for module in self.modules_list:
            x_r, x_i = module((x_r, x_i))
        return (x_r, x_i)


