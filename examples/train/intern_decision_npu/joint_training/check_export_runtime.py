"""Audit stored tensors and compare canonical reload with explicit cast/autocast."""
import argparse
from collections import Counter
import json
import hashlib
from pathlib import Path
import torch
import torch_npu
from safetensors import safe_open
from swift.model import get_model_processor
from joint_validation import evaluate

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);args=p.parse_args()
out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
torch.manual_seed(42);torch.npu.set_device(0)
model,processor=get_model_processor(args.checkpoint,model_type='qwen3_5',torch_dtype=torch.bfloat16,device_map='npu:0',attn_impl='sdpa',new_special_tokens=['<decision>'])
model.eval();state=model.state_dict();records=[]
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
checkpoint_hashes={p.name:digest(p) for p in sorted(Path(args.checkpoint).iterdir()) if p.is_file()}
for path in sorted(Path(args.checkpoint).glob('*.safetensors')):
 with safe_open(path,framework='pt',device='cpu') as saved:
  for key in saved.keys():
   expected=saved.get_tensor(key);actual=state[key].detach().cpu()
   equal=actual.shape==expected.shape and torch.equal(actual,expected.to(actual.dtype))
   records.append({'name':key,'saved_dtype':str(expected.dtype),'loaded_dtype':str(actual.dtype),'equal':equal})
   if not equal:raise ValueError('Stored tensor mismatch: '+key)
report={'checkpoint_sha256':checkpoint_hashes,'tensor_count':len(records),'all_stored_tensors_exact':True,'dtype_pairs':dict(Counter(x['saved_dtype']+' -> '+x['loaded_dtype'] for x in records)),'dtype_changes':[x for x in records if x['saved_dtype']!=x['loaded_dtype']]}
(out/'weights.json').write_text(json.dumps(report,indent=2));print('WEIGHTS',json.dumps(report),flush=True)
evaluate(model,processor.tokenizer,args.data,out/'canonical-repeat.json')
buffers={name:buf.detach().cpu().clone() for name,buf in model.named_buffers()}
model.to(dtype=torch.bfloat16)
changes=[]
for name,buf in model.named_buffers():
 before=buffers[name];after=buf.detach().cpu()
 if before.dtype!=after.dtype or not torch.equal(before.float(),after.float()):
  changes.append({'name':name,'before_dtype':str(before.dtype),'after_dtype':str(after.dtype),'persistent':name in state,'max_abs_delta':float((before.float()-after.float()).abs().max())})
(out/'buffers.json').write_text(json.dumps(changes,indent=2))
evaluate(model,processor.tokenizer,args.data,out/'cast-all-bf16.json')
with torch.autocast(device_type='npu',dtype=torch.bfloat16):
 evaluate(model,processor.tokenizer,args.data,out/'cast-all-autocast.json')
(out/'complete.json').write_text(json.dumps({'status':'complete','scope':'Validation diagnostic only; no training or benchmark selection'}))
