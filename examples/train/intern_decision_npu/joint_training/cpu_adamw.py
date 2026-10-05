"""Single-device FP32 AdamW with optimizer parameters and moments on CPU."""
import torch
from swift.optimizers import OptimizerCallback, optimizers_map


class CPUAdamW(torch.optim.Optimizer):
    def __init__(self, groups, defaults):
        super().__init__(groups, defaults)
        cpu_groups = []
        for group in self.param_groups:
            if any(p.dtype != torch.float32 for p in group['params']):
                raise ValueError('FP32 trainable parameters are required')
            copied = {k: v for k, v in group.items() if k != 'params'}
            copied.update(params=[torch.nn.Parameter(p.detach().to('cpu', copy=True)) for p in group['params']],
                          foreach=False, fused=False, capturable=False)
            cpu_groups.append(copied)
        self.cpu_optimizer = torch.optim.AdamW(cpu_groups, foreach=False, fused=False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for live, cpu in zip(self.param_groups, self.cpu_optimizer.param_groups):
            for key in ('lr', 'betas', 'eps', 'weight_decay', 'maximize'):
                cpu[key] = live[key]
            for parameter, copy in zip(live['params'], cpu['params']):
                copy.grad = None if parameter.grad is None else parameter.grad.detach().to('cpu', copy=True)
        self.cpu_optimizer.step()
        for live, cpu in zip(self.param_groups, self.cpu_optimizer.param_groups):
            for parameter, copy in zip(live['params'], cpu['params']):
                if parameter.grad is not None:
                    parameter.copy_(copy, non_blocking=False)
                copy.grad = None
        return loss

    def state_dict(self):
        return self.cpu_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        # Accelerate may move the serialized dictionary to the device before
        # calling this method. AdamW casts it back to its CPU parameters.
        self.cpu_optimizer.load_state_dict(state_dict)
        for live, cpu in zip(self.param_groups, self.cpu_optimizer.param_groups):
            for key, value in cpu.items():
                if key != 'params':
                    live[key] = value


class CPUAdamWCallback(OptimizerCallback):
    def create_optimizer(self, model=None):
        original = super().create_optimizer(model)
        optimizer = CPUAdamW(original.param_groups, original.defaults)
        self.trainer.optimizer = optimizer
        return optimizer


optimizers_map['decision_cpu_adamw'] = CPUAdamWCallback
