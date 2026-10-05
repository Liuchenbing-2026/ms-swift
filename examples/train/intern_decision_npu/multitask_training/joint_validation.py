"""Candidate accuracy on frozen validation cases, before and after training."""
import hashlib
import json
import os
import math
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import TrainerCallback
from swift.callbacks import callbacks_map
from decision_schema import compile_row, _options, _answer_value


def evaluate(model, tokenizer, data, destination, limit=0):
    cases = [json.loads(line) for line in Path(data).read_text().splitlines()]
    if limit:
        cases = cases[:limit]
    marker = tokenizer.encode('<decision>', add_special_tokens=False)
    assert len(marker) == 1
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    results = {}
    with torch.inference_mode():
        for mode in ('joint', 'single'):
            jobs = []
            for case in cases:
                questions = case['questions']
                groups = [questions] if mode == 'joint' else [{field: question} for field, question in questions.items()]
                for group in groups:
                    clean = {'state': case['state'], 'questions': group}
                    compiled = compile_row(clean, include_targets=False)
                    text = tokenizer.apply_chat_template(compiled.messages, tokenize=False,
                        add_generation_prompt=False, enable_thinking=False)
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    positions = [i-1 for i, token in enumerate(ids) if token == marker[0]]
                    assert len(positions) == len(compiled.fields) and min(positions) >= 0 and len(ids) <= 8192
                    fields = []
                    for field in compiled.fields:
                        symbols = [tokenizer.encode(s, add_special_tokens=False) for s in compiled.symbols[field]]
                        assert all(len(s) == 1 for s in symbols)
                        choices = [k for k, _ in _options(group[field])]
                        gold = _answer_value(group[field], case['targets'][field])
                        assert gold in choices
                        fields.append((field, [s[0] for s in symbols], choices, gold))
                    jobs.append((ids, positions, fields, case['id']))
            predictions = []
            tokenizer.padding_side = 'right'
            for start in range(0, len(jobs), 4):
                batch_jobs = jobs[start:start+4]
                batch = tokenizer.pad({'input_ids': [job[0] for job in batch_jobs]},
                    padding=True, pad_to_multiple_of=128, return_tensors='pt').to(device)
                keep = sorted({position for job in batch_jobs for position in job[1]})
                logits = model(**batch, use_cache=False,
                    logits_to_keep=torch.tensor(keep, device=device)).logits
                for index, job in enumerate(batch_jobs):
                    for position, (field, tokens, choices, gold) in zip(job[1], job[2]):
                        scores = logits[index, keep.index(position), tokens].float().cpu()
                        assert torch.isfinite(scores).all()
                        predictions.append({'case_id': job[3], 'field': field, 'gold': gold,
                            'prediction': choices[int(scores.argmax())], 'probabilities': scores.softmax(-1).tolist(),
                            'input_hash': hashlib.sha256(json.dumps(job[0]).encode()).hexdigest()})
                if start % 100 == 0:
                    print(f'validation {mode}: {min(start+4, len(jobs))}/{len(jobs)}', flush=True)
            assert len(predictions) == sum(len(case['questions']) for case in cases)
            correct = sum(p['prediction'] == p['gold'] for p in predictions)
            results[mode] = {'correct': correct, 'total': len(predictions), 'accuracy': correct / len(predictions),
                             'predictions': predictions}
    model.train(was_training)
    results['data_sha256'] = hashlib.sha256(Path(data).read_bytes()).hexdigest()
    Path(destination).write_text(json.dumps(results, indent=2))
    return results


class JointValidation(TrainerCallback):
    def __init__(self, args, trainer):
        self.trainer = trainer

    def run(self, args, name):
        evaluate(self.trainer.model, self.trainer.template.tokenizer,
            os.environ['JOINT_VALIDATION_DATA'], Path(args.output_dir) / (name + '.json'),
            int(os.environ.get('JOINT_VALIDATION_LIMIT', '0')))

    def on_train_begin(self, args, state, control, **kwargs):
        # Avoid a parameter-sized foreach temporary during the AdamW step.
        for group in kwargs['optimizer'].param_groups:
            group['foreach'] = False
        if os.environ.get('JOINT_INITIAL_VALIDATION') == '1':
            self.run(args, 'validation-before')

    def on_log(self, args, state, control, logs=None, **kwargs):
        for key in ('loss', 'grad_norm'):
            if key in (logs or {}) and not math.isfinite(float(logs[key])):
                raise ValueError('Nonfinite training metric: ' + key)
        (Path(args.output_dir) / 'progress.json').write_text(json.dumps({
            'step': state.global_step, 'max_steps': state.max_steps, 'metrics': logs,
            'observed_utc': datetime.now(timezone.utc).isoformat()}, default=str, indent=2))

    def on_train_end(self, args, state, control, **kwargs):
        # Export inference weights after the last optimizer update. FP32 training
        # parameters are preserved until training ends; no optimizer resume is claimed.
        from export_dtype import cast_parameters_for_export
        cast_parameters_for_export(self.trainer.model)
        self.run(args, 'validation-after')
        destination = Path(args.output_dir) / ('checkpoint-' + str(state.global_step))
        self.trainer.save_model(str(destination))
        self.trainer.template.processor.save_pretrained(destination)
        (Path(args.output_dir) / 'export.json').write_text(json.dumps({
            'step': state.global_step, 'checkpoint': str(destination),
            'dtype': 'bfloat16', 'optimizer_state_saved': False}, indent=2))


callbacks_map['joint_validation'] = JointValidation


if __name__ == '__main__':
    import argparse
    from swift.model import get_model_processor
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=0)
    options = parser.parse_args()
    if Path(options.output).exists():
        raise FileExistsError('Use a fresh validation output')
    torch.npu.set_device(0)
    model, processor = get_model_processor(options.checkpoint, model_type='qwen3_5',
        torch_dtype=torch.bfloat16, device_map='npu:0', attn_impl='sdpa',
        new_special_tokens=['<decision>'])
    evaluate(model, processor.tokenizer, options.data, options.output, options.limit)
