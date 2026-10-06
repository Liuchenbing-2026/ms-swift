"""Read-only candidate-scored evaluation after SWIFT restores FSDP state."""
from collections.abc import Mapping
from datetime import timedelta
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import re

import torch
import torch.distributed as dist
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import TrainOutput
from swift.callbacks import callbacks_map


class EvaluationComplete(Exception):
    """Exit the prepared training loop before its first optimizer update."""


class CheckpointEvaluation(TrainerCallback):
    def __init__(self, args, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        config = json.loads(Path(os.environ['DECISION_EVAL_CONFIG']).read_text())
        rank, world = dist.get_rank(), dist.get_world_size()
        group = dist.new_group(backend='gloo', timeout=timedelta(minutes=30))
        model, tokenizer = self.trainer.model, self.trainer.template.tokenizer
        model.eval();tokenizer.padding_side='right'
        marker=tokenizer.encode('<decision>',add_special_tokens=False);assert len(marker)==1
        root=Path(args.output_dir);reports={}
        train=[json.loads(x) for x in Path(config['training_data']).read_text().splitlines()]
        train_ids={r['case_id'] for r in train}
        train_messages={json.dumps(r['messages'],sort_keys=True,ensure_ascii=False) for r in train}
        for suite in config['suites']:
            path=Path(suite['data']);assert hashlib.sha256(path.read_bytes()).hexdigest()==suite['sha256']
            rows=[json.loads(x) for x in path.read_text().splitlines()]
            assert len(rows)==suite['expected_rows']
            keys=[(r['case_id'],f) for r in rows for f in r['fields']]
            assert len(keys)==len(set(keys))==suite['expected_decisions']
            assert not ({r['case_id'] for r in rows}&train_ids)
            jobs=[]
            for row in rows:
                assert json.dumps(row['messages'],sort_keys=True,ensure_ascii=False) not in train_messages
                fields=row['fields'];assert list(json.loads(row['messages'][-1]['content']))==fields
                assert all(v=='<decision>' for v in json.loads(row['messages'][-1]['content']).values())
                ids=tokenizer.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=False,enable_thinking=False)
                if isinstance(ids,Mapping):ids=ids['input_ids']
                positions=[i-1 for i,x in enumerate(ids) if x==marker[0]]
                assert len(positions)==len(fields) and min(positions)>=0 and len(ids)<=8192
                tokens={f:[tokenizer.encode(s,add_special_tokens=False) for s in row['candidates'][f]] for f in fields}
                assert all(len(x)==1 for values in tokens.values() for x in values)
                assert all(row['targets'][f] in row['candidates'][f] for f in fields)
                jobs.append((ids,positions,{f:[x[0] for x in values] for f,values in tokens.items()}))
            batch_size=suite['batch_size'];local=[]
            with torch.no_grad():
                for start in range(0,len(jobs),batch_size*world):
                    indices=[start+rank*batch_size+j for j in range(batch_size)]
                    selected=[jobs[i if i<len(jobs) else 0] for i in indices]
                    batch=tokenizer.pad({'input_ids':[j[0] for j in selected]},padding=True,pad_to_multiple_of=128,return_tensors='pt').to(args.device)
                    keep=sorted({position for j in selected for position in j[1]})
                    logits=model(**batch,use_cache=False,logits_to_keep=torch.tensor(keep,device=args.device)).logits
                    for offset,index in enumerate(indices):
                        if index>=len(jobs):continue
                        row=rows[index];job=selected[offset]
                        input_hash=hashlib.sha256(json.dumps(job[0]).encode()).hexdigest()
                        for f,position in zip(row['fields'],job[1]):
                            scores=logits[offset,keep.index(position),job[2][f]].float().cpu()
                            assert torch.isfinite(scores).all()
                            local.append({'index':index,'case_id':row['case_id'],'field':f,'gold':row['targets'][f],
                                'prediction':row['candidates'][f][int(scores.argmax())],
                                'symbols':row['candidates'][f],'probabilities':scores.softmax(-1).tolist(),'input_sha256':input_hash})
                    if rank==0 and start%(batch_size*world*25)==0:
                        print(suite['name'],min(start+batch_size*world,len(rows)),'/',len(rows),flush=True)
            name=suite['name'];(root/f'{name}-rank{rank}.json').write_text(json.dumps(local))
            dist.monitored_barrier(group=group,timeout=timedelta(minutes=30))
            if rank==0:
                merged=[x for r in range(world) for x in json.loads((root/f'{name}-rank{r}.json').read_text())]
                assert sorted((x['case_id'],x['field']) for x in merged)==sorted(keys)
                correct=sum(x['prediction']==x['gold'] for x in merged)
                reports[name]={'rows':len(rows),'correct':correct,'total':len(merged),'accuracy':correct/len(merged),
                    'data_sha256':suite['sha256'],'batch_size':batch_size,
                    'input_hash':hashlib.sha256(json.dumps([(x['case_id'],x['field'],x['input_sha256']) for x in sorted(merged,key=lambda x:(x['index'],x['field']))]).encode()).hexdigest()}
                (root/'suite-progress.json').write_text(json.dumps(reports,indent=2))
            dist.monitored_barrier(group=group,timeout=timedelta(minutes=30))
        if rank==0:
            official=[v for k,v in reports.items() if k.startswith('official-')]
            broad=[v for k,v in reports.items() if k.startswith('broad-')]
            report={'status':'complete','checkpoint_step':state.global_step,'scope':config['scope'],
                'optimizer_updates_performed':0,'suites':reports,'test_use':'report_only',
                'official_macro_accuracy':sum(x['accuracy'] for x in official)/len(official) if official else None,
                'broad_accuracy':sum(x['correct'] for x in broad)/sum(x['total'] for x in broad) if broad else None,
                'limitation':'BF16 FSDP runtime; calibration metrics and bitwise backend equivalence not claimed'}
            (root/'evaluation.json').write_text(json.dumps(report,indent=2))
        dist.monitored_barrier(group=group,timeout=timedelta(minutes=30))
        raise EvaluationComplete()


if os.environ.get("DECISION_EVAL_ONLY") == "1":
    original_train = Trainer.train

    @wraps(original_train)
    def evaluate_in_prepared_loop(self, *args, **kwargs):
        try:
            return original_train(self, *args, **kwargs)
        except EvaluationComplete:
            return TrainOutput(self.state.global_step, 0.0, {"checkpoint_evaluation_only": True})

    Trainer.train = evaluate_in_prepared_loop
    callbacks_map["decision_checkpoint_eval"] = CheckpointEvaluation
