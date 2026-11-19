"""One explicit reduction path for local and torchrun learners."""

import torch
import torch.distributed as dist


class ExecutionBackend:
    """Losses use local sums divided by global counts; gradients use SUM.

    No model replicas or algorithm logic live here. All ranks must call the
    same collectives even when their local batch has no valid labels.
    """

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

    @property
    def distributed(self):
        return dist.is_available() and dist.is_initialized()

    @property
    def rank(self):
        return dist.get_rank() if self.distributed else 0

    @property
    def world_size(self):
        return dist.get_world_size() if self.distributed else 1

    def sum(self, value):
        result = torch.as_tensor(value, device=self.device).detach().clone()
        if self.distributed:
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result

    def count(self, mask):
        return self.sum(mask.sum().to(device=self.device, dtype=torch.float32))

    def reduce_gradients(self, parameters):
        parameters = tuple(dict.fromkeys(parameters))
        if not self.distributed or not parameters:
            return
        used = torch.tensor(
            [p.grad is not None for p in parameters], device=self.device, dtype=torch.int32
        )
        dist.all_reduce(used, op=dist.ReduceOp.MAX)
        for index, parameter in enumerate(parameters):
            if used[index]:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            else:
                parameter.grad = None

    def synchronize(self, module):
        if self.distributed:
            for value in module.state_dict().values():
                dist.broadcast(value, src=0)

    def broadcast_object(self, value):
        if not self.distributed:
            return value
        values = [value if self.rank == 0 else None]
        dist.broadcast_object_list(values, src=0)
        return values[0]

    def gather_objects(self, value):
        if not self.distributed:
            return [value]
        result = [None] * self.world_size
        dist.all_gather_object(result, value)
        return result

    def barrier(self):
        if self.distributed:
            dist.barrier()
