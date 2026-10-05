"""Freeze a balanced training intervention using already audited train splits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def select(rows, count, seed):
    if len(rows) < count:
        raise ValueError('Insufficient unique source cases')
    rows = sorted(rows, key=lambda row: row['id'])
    random.Random(seed).shuffle(rows)
    return rows[:count]


def build(business_train, business_validation, multitask_train, multitask_validation,
          per_source=320, seed=42):
    if per_source % 2:
        raise ValueError('Safety classes require an even per-source count')
    all_train = business_train + multitask_train
    all_validation = business_validation + multitask_validation
    train_ids = [row['id'] for row in all_train]
    validation_ids = [row['id'] for row in all_validation]
    if len(set(train_ids)) != len(train_ids) or len(set(validation_ids)) != len(validation_ids):
        raise ValueError('Duplicate input identity')
    if set(train_ids) & set(validation_ids):
        raise ValueError('Training/validation identity overlap')
    tools = [r for r in multitask_train if r['source'] == 'ToolACE']
    safety = {label: [r for r in multitask_train if r['source'] == 'Aegis-2.0'
                     and r['targets']['unsafe']['label'] == label] for label in ('yes', 'no')}
    chosen = {
        'business': select(business_train, per_source, seed),
        'tool': select(tools, per_source, seed + 1),
        'safety': select(safety['yes'], per_source // 2, seed + 2)
                  + select(safety['no'], per_source // 2, seed + 3),
    }
    random.Random(seed + 4).shuffle(chosen['safety'])
    # Equal case exposure, not equal token counts. Business cases retain all fields.
    mixed = [dict(chosen[source][i], mixture_source=source)
             for i in range(per_source) for source in ('business', 'tool', 'safety')]
    validation = {'business': business_validation,
                  'tool': [r for r in multitask_validation if r['source'] == 'ToolACE'],
                  'safety': [r for r in multitask_validation if r['source'] == 'Aegis-2.0']}
    if any(not rows for rows in validation.values()):
        raise ValueError('Every source requires independent validation')
    return mixed, validation


def main():
    p = argparse.ArgumentParser()
    for name in ('business_train', 'business_validation', 'multitask_train', 'multitask_validation', 'output', 'schema_dir'):
        p.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    p.add_argument('--per-source', type=int, default=320)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    sys.path.insert(0, str(a.schema_dir))
    from decision_schema import compile_row
    paths = {k: getattr(a, k) for k in ('business_train', 'business_validation', 'multitask_train', 'multitask_validation')}
    rows, validation = build(**{k: read(path) for k, path in paths.items()}, per_source=a.per_source, seed=a.seed)
    a.output.mkdir(parents=True, exist_ok=False)
    compiled = []
    for row in rows:
        item = compile_row(row)
        assert item.messages == compile_row(row, include_targets=False).messages
        compiled.append({'case_id': row['id'], 'messages': item.messages,
                         'decision_targets': [item.targets[f] for f in item.fields]})
    for name, values in [('train', compiled), ('train-raw', rows)] + [
            ('validation-' + name, values) for name, values in validation.items()]:
        (a.output / (name + '.jsonl')).write_text(''.join(json.dumps(row, ensure_ascii=True) + '\n' for row in values))
    manifest = {'seed': a.seed, 'case_ratio': '1:1:1 business/tool/safety',
                'safety_class_ratio': '1:1 unsafe/safe', 'train_cases': len(rows),
                'train_decisions': sum(len(row['questions']) for row in rows),
                'unique_case_ids': len({row['id'] for row in rows}),
                'source_counts': dict(Counter(row['mixture_source'] for row in rows)),
                'validation': {k: {'cases': len(v), 'decisions': sum(len(r['questions']) for r in v)} for k, v in validation.items()},
                'input_sha256': {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in paths.items()},
                'output_sha256': {v.name: hashlib.sha256(v.read_bytes()).hexdigest() for v in a.output.glob('*.jsonl')},
                'test_data_used': False, 'token_audit': 'required before training'}
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
