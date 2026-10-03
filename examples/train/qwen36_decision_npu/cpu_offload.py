"""Use pageable CPU offload when large NPU pinned allocations are unavailable."""
from functools import wraps

from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp import CPUOffloadPolicy

_original = FullyShardedDataParallelPlugin.set_cpu_offload


@wraps(_original)
def set_cpu_offload(self):
    _original(self)
    if self.fsdp_version == 2 and isinstance(self.cpu_offload, CPUOffloadPolicy):
        self.cpu_offload = CPUOffloadPolicy(pin_memory=False)


FullyShardedDataParallelPlugin.set_cpu_offload = set_cpu_offload
