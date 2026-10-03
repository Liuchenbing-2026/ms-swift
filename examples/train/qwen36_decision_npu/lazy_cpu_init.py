"""Keep non-source FSDP initialization zero-filled without touching every CPU page.

Anonymous mappings are initially zero by the OS contract. Rank zero's complete
state is still broadcast by Accelerate before training. This changes allocation,
not the initialized values, and applies only to non-source FSDP2 workers.
"""
from functools import wraps
import mmap
import os

import torch
import transformers.modeling_utils as modeling

_original = modeling.PreTrainedModel._move_missing_keys_from_meta_to_device


def lazy_cpu_zeros(tensor):
    if tensor.numel() == 0 or tensor.layout != torch.strided:
        return torch.zeros_like(tensor, device='cpu')
    storage = mmap.mmap(-1, tensor.numel() * tensor.element_size(),
                        flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    # frombuffer retains the mmap owner; do not close it while the tensor lives.
    return torch.frombuffer(storage, dtype=tensor.dtype, count=tensor.numel()).reshape(tensor.shape)


@wraps(_original)
def move_missing(self, missing_keys, device_map, device_mesh, hf_quantizer):
    if (os.environ.get('FSDP_VERSION') == '2' and modeling.is_fsdp_enabled()
            and not modeling.is_local_dist_rank_0() and hf_quantizer is None):
        for key, param in self.named_parameters():
            modeling._load_parameter_into_model(self, key, lazy_cpu_zeros(param))
        for key, buffer in self.named_buffers():
            modeling._load_parameter_into_model(self, key, lazy_cpu_zeros(buffer))
        return
    return _original(self, missing_keys, device_map, device_mesh, hf_quantizer)


modeling.PreTrainedModel._move_missing_keys_from_meta_to_device = move_missing
