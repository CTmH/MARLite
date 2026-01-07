import torch
import torch.distributed as dist

class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_input = torch.zeros_like(grad_outputs[0])
        dist.reduce_scatter(grad_input, list(grad_outputs))
        return grad_input