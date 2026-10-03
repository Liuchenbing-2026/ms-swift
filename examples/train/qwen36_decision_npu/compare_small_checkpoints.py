"""Compare same-seed small-model checkpoints after an initialization change."""
import json
import math
from pathlib import Path
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

def read(stage):
    path=Path('/workspace/outputs')/stage/'checkpoint-2/pytorch_model_fsdp_0'
    reader=dcp.FileSystemReader(path)
    meta=reader.read_metadata()
    assert sum(math.prod(value.size) for value in meta.state_dict_metadata.values() if isinstance(value,TensorStorageMetadata)) < 100_000_000, 'Use only the small regression fixture'
    state={key:torch.empty(value.size,dtype=value.properties.dtype) for key,value in meta.state_dict_metadata.items() if isinstance(value,TensorStorageMetadata)}
    dcp.load(state,storage_reader=reader)
    return state

left=read('tiny-control');right=read('tiny-lazy')
assert left.keys()==right.keys()
results=[]
for name,a in left.items():
    b=right[name]
    error=(a.float()-b.float()).abs()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    results.append({'name':name,'elements':a.numel(),'max_abs':float(error.max()) if error.numel() else 0,
                    'mean_abs':float(error.mean()) if error.numel() else 0,'bitwise_equal':torch.equal(a,b)})
record={'tensor_count':len(results),'max_abs':max(row['max_abs'] for row in results),
        'mean_abs':sum(row['mean_abs']*row['elements'] for row in results)/sum(row['elements'] for row in results),
        'equal_tensors':sum(row['bitwise_equal'] for row in results),
        'worst_tensors':sorted(results,key=lambda row:row['max_abs'],reverse=True)[:5]}
Path('/workspace/results').mkdir(exist_ok=True)
Path('/workspace/results/control-lazy-checkpoint-comparison.json').write_text(json.dumps(record,indent=2))
assert record['max_abs'] <= 1e-6 and record['mean_abs'] <= 1e-8
print(json.dumps(record))
