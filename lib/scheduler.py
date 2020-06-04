import torch
import math


class TransformerScheduler:
    def __init__(self, d_model, warmup_steps):
        self.inverse_sqrt_d_model = 1 / math.sqrt(d_model)
        self.inverse_warmup_steps_power_onehalf = 1 / (warmup_steps ** 1.5)
        pass

    def compute_lr(self, stepnum):
        inverse_sqrt_stepnum = 1 / math.sqrt(stepnum)
        return self.inverse_sqrt_d_model * min(inverse_sqrt_stepnum, stepnum * self.inverse_warmup_steps_power_onehalf)

    def __call__(self, stepnum):
        return self.compute_lr(stepnum)
