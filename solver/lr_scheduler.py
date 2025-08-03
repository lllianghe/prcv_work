from bisect import bisect_right
from math import cos, pi

from torch.optim.lr_scheduler import _LRScheduler


class LRSchedulerWithWarmup(_LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        mode="step",
        warmup_factor=1.0 / 3,
        warmup_method="linear",
        warmup_epochs=10,
        annealing_epochs=-1, # 新的参数, epoch=step
        min_lr=1e-7,
        total_epochs=100,
        target_lr=0,
        power=0.9,
        step_size=2000,
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of"
                " increasing integers. Got {}".format(milestones),
            )
        if mode not in ("step", "exp", "poly", "cosine", "linear"):
            raise ValueError(
                "Only 'step', 'exp', 'poly' or 'cosine' learning rate scheduler accepted"
                "got {}".format(mode)
            )
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.mode = mode
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        self.total_epochs = total_epochs
        self.target_lr = target_lr
        self.power = power
        self.annealing_epochs=annealing_epochs
        self.min_lr=min_lr
        self.step_size = step_size
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # 热身
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_epochs
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        # step退火
        elif self.mode == "step":
            return [
                base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs
            ]
        
        if self.mode == "exp":
            return [base_lr * self.power ** ((self.last_epoch-self.warmup_epochs)/self.step_size) for base_lr in self.base_lrs]

        # 全程退火
        elif self.annealing_epochs <= self.warmup_epochs:
            epoch_ratio = (self.last_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )

            if self.mode == "linear":
                factor = 1 - epoch_ratio
                return [base_lr * factor for base_lr in self.base_lrs]

            if self.mode == "poly":
                factor = 1 - epoch_ratio
                return [
                    self.target_lr + (base_lr - self.target_lr) * self.power ** factor
                    for base_lr in self.base_lrs
                ]
            if self.mode == "cosine":
                factor = 0.5 * (1 + cos(pi * epoch_ratio))
                return [
                    self.target_lr + (base_lr - self.target_lr) * factor
                    for base_lr in self.base_lrs
                ]
            raise NotImplementedError
        
        # 三段式退火
        elif  self.last_epoch < self.warmup_epochs + self.annealing_epochs:
            epoch_ratio = (self.last_epoch - self.warmup_epochs) / self.annealing_epochs

            if self.mode == "exp":
                factor = epoch_ratio
                return [base_lr * self.power ** factor for base_lr in self.base_lrs]
            if self.mode == "linear":
                factor = 1 - epoch_ratio
                return [base_lr * factor for base_lr in self.base_lrs]

            if self.mode == "poly":
                factor = 1 - epoch_ratio
                return [
                    self.target_lr + (base_lr - self.target_lr) * self.power ** factor
                    for base_lr in self.base_lrs
                ]
            if self.mode == "cosine":
                factor = 0.5 * (1 + cos(pi * epoch_ratio))
                return [
                    self.target_lr + (base_lr - self.target_lr) * factor
                    for base_lr in self.base_lrs
                ]
            raise NotImplementedError
        else:
            return [self.min_lr for _ in self.base_lrs]

            
    def step(self, steps=1):
        self.last_epoch += steps - 1
        super(LRSchedulerWithWarmup, self).step()