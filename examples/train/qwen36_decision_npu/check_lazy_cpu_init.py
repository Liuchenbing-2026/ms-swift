"""CPU-only value, lifetime, writability, and allocation checks."""
import gc
import resource

import torch
from lazy_cpu_init import lazy_cpu_zeros

for dtype in (torch.float32, torch.bfloat16, torch.float16, torch.int64, torch.bool):
    x = lazy_cpu_zeros(torch.empty((3, 7), dtype=dtype, device='meta'))
    gc.collect()
    assert torch.equal(x, torch.zeros((3, 7), dtype=dtype))
    x[1, 2] = 1
    assert x[1, 2].item() == 1 and x[0, 2].item() == 0
assert lazy_cpu_zeros(torch.empty(0, device='meta')).numel() == 0
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
x = lazy_cpu_zeros(torch.empty(35_000_000_000, device='meta', dtype=torch.float32))
assert all(x[i].item() == 0 for i in (0, 7, 1_000_000, 34_999_999_999))
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
# Linux reports ru_maxrss in KiB; this recipe targets the Linux training image.
assert after - before < 64 * 1024
print('Lazy CPU zero values, ownership, writes, and bounded resident allocation passed')
