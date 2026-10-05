"""Single-device FP32 AdamW with optimizer parameters and moments on CPU."""
import ctypes
import os
import sys

import torch
from swift.optimizers import OptimizerCallback, optimizers_map


class CPUAdamW(torch.optim.Optimizer):
    def __init__(self, groups, defaults):
        if os.environ.get('DECISION_DISABLE_THP') == '1':
            if not sys.platform.startswith('linux'):
                raise RuntimeError('Per-process THP control requires Linux')
            # This affects only this process, never the host-wide THP policy.
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.prctl(41, 1, 0, 0, 0) != 0:  # PR_SET_THP_DISABLE
                raise OSError(ctypes.get_errno(), 'Cannot disable process THP')
        super().__init__(groups, defaults)
        self._grad_buffers = {}
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
                if parameter.grad is None:
                    copy.grad = None
                else:
                    if parameter not in self._grad_buffers:
                        self._grad_buffers[parameter] = torch.empty_like(copy)
                    buffer = self._grad_buffers[parameter]
                    buffer.copy_(parameter.grad.detach(), non_blocking=False)
                    copy.grad = buffer
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
