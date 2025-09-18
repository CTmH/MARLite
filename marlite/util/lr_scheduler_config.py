import torch.optim.lr_scheduler as lr_scheduler

registered_lr_scheduler = {
    "StepLR": lr_scheduler.StepLR,
    "OneCycleLR": lr_scheduler.OneCycleLR,
    "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,
}

class LRSchedulerConfig:
    def __init__(self, **kwargs):
        self.lr_scheduler_type = kwargs.pop('type')
        self.lr_scheduler_kwargs = kwargs

    def get_lr_scheduler(self, optimizer) -> lr_scheduler.LRScheduler:
        if self.lr_scheduler_type in registered_lr_scheduler:
            optim_class = registered_lr_scheduler[self.lr_scheduler_type]
            return optim_class(optimizer, **self.lr_scheduler_kwargs)
        else:
            raise ValueError(f"Unsupported scheduler type: {self.lr_scheduler_type}")
