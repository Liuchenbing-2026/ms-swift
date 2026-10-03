"""Wait for sufficient free storage before starting a complete checkpoint write."""
from datetime import timedelta
from functools import wraps
import json
from pathlib import Path
import shutil
import time

import torch
import torch.distributed as dist
from transformers import Trainer

_original = Trainer._save_checkpoint
_group = None


@wraps(_original)
def save_with_space_check(self, *args, **kwargs):
    global _group
    distributed = dist.is_initialized()
    if distributed and _group is None:
        _group = dist.new_group(backend='gloo', timeout=timedelta(minutes=20))
    main = not distributed or dist.get_rank() == 0
    total = sum(p.numel() for p in self.model.parameters())
    trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    required = int((total * 4 + trainable * 8) * 1.05) + 10 * 1024**3
    output = Path(self.args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    status = output / 'checkpoint-storage.json'
    while True:
        available = shutil.disk_usage(output).free
        ready = torch.tensor(int(available >= required) if main else 0, dtype=torch.int32)
        if distributed:
            dist.broadcast(ready, src=0, group=_group)
        if main:
            status.write_text(json.dumps({'step': self.state.global_step,
                'status': 'ready' if ready.item() else 'waiting_for_storage',
                'required_free_bytes': required, 'available_bytes': available,
                'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, indent=2))
        if ready.item():
            break
        time.sleep(15)
    return _original(self, *args, **kwargs)


Trainer._save_checkpoint = save_with_space_check
