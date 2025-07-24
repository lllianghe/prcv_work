from bisect import bisect_right
from math import cos, pi

from torch.optim.lr_scheduler import _LRScheduler


class LRSchedulerWithWarmup(_LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones, # 不能用了
        gamma=0.1,
        mode="cosine",
        warmup_factor=1.0 / 3,
        warmup_method="linear",
        warmup_steps=800,
        annealing_steps=4000,
        target_lr=0,
        power=0.9,
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of"
                " increasing integers. Got {}".format(milestones),
            )
        if mode not in ("step", "exp", "poly", "cosine", "linear", "cosine_warm"):
            raise ValueError(
                "Only 'step', 'exp', 'poly', 'cosine', 'linear' or 'cosine_warm' learning rate scheduler accepted"
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
        self.warmup_steps = warmup_steps
        self.warmup_method = warmup_method
        self.annealing_steps = annealing_steps
        self.target_lr = target_lr
        self.power = power

        # parameters for cosine annealing with warm restarts
        self.t_0 = annealing_steps
        self.t_mult = 1
        self.t_cur = 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_steps
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]


        if self.mode == "cosine_warm":
            if self.t_cur >= self.t_0:
                self.t_cur = self.t_cur - self.t_0
                self.t_0 = self.t_0 * self.t_mult
            
            factor = 0.5 * (1 + cos(pi * self.t_cur / self.t_0))

            return [
                self.target_lr + (base_lr - self.target_lr) * factor
                for base_lr in self.base_lrs
            ]
        elif self.last_epoch < self.warmup_steps + self.annealing_steps:
            if self.mode == "step":
                return [
                    base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                    for base_lr in self.base_lrs
                ]

            step_ratio = (self.last_epoch - self.warmup_steps) / self.annealing_steps

            if self.mode == "exp":
                factor = step_ratio
                return [base_lr * self.power ** factor for base_lr in self.base_lrs]
            if self.mode == "linear":
                factor = 1 - step_ratio
                return [base_lr * factor for base_lr in self.base_lrs]

            if self.mode == "poly":
                factor = 1 - step_ratio
                return [
                    self.target_lr + (base_lr - self.target_lr) * self.power ** factor
                    for base_lr in self.base_lrs
                ]
            if self.mode == "cosine":
                factor = 0.5 * (1 + cos(pi * step_ratio))
                return [
                    self.target_lr + (base_lr - self.target_lr) * factor
                    for base_lr in self.base_lrs
                ]
            raise NotImplementedError
        
        return [base_lr / 5 for base_lr in self.base_lrs]

    def step(self, steps=1):
        self.t_cur += steps
        self.last_epoch += steps - 1
        super(LRSchedulerWithWarmup, self).step()
