"""Compile decision records and reject invalid/overlength rows without truncation."""
import argparse
from collections import Counter
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--schema-dir', type=Path, required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-tokens', type=int, default=8192)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    sys.path.insert(0, str(args.schema_dir))
    from decision_schema import compile_row
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    marker = tokenizer.encode('<decision>', add_special_tokens=False)
    assert len(marker) == 1
    args.output.mkdir(parents=True)
    manifest = {'max_tokens': args.max_tokens, 'no_truncation': True, 'splits': {}}
    for split in ('train', 'validation'):
        path = args.prepared / (split + '.jsonl')
        counts = Counter(); lengths = []; rejected = []; raw = []; compiled = []
        for line in path.read_text().splitlines():
            row = json.loads(line); counts['input_rows'] += 1
            try:
                example = compile_row(row)
                assert example.messages == compile_row(row, include_targets=False).messages
                targets = [example.targets[field] for field in example.fields]
                ids = tokenizer.apply_chat_template(example.messages, tokenize=True,
                          add_generation_prompt=False, enable_thinking=False)
                if isinstance(ids, Mapping):
                    ids = ids['input_ids']
                assert ids.count(marker[0]) == len(targets) > 0
                assert all(len(tokenizer.encode(t, add_special_tokens=False)) == 1 for t in targets)
                if len(ids) > args.max_tokens:
                    counts['overlength'] += 1; rejected.append({'id': row['id'], 'reason': 'overlength', 'tokens': len(ids)});continue
            except (ValueError, AssertionError, KeyError) as error:
                counts['invalid'] += 1; rejected.append({'id': row['id'], 'reason': str(error)});continue
            lengths.append(len(ids)); counts['retained'] += 1; counts[row['source']] += 1
            counts['decision_labels'] += len(targets);raw.append(row)
            compiled.append({'case_id': row['id'], 'messages': example.messages, 'decision_targets': targets})
        for name, values in [(split + '.jsonl', compiled), (split + '-raw.jsonl', raw), (split + '-rejected.jsonl', rejected)]:
            (args.output / name).write_text(''.join(json.dumps(r, ensure_ascii=True) + '\n' for r in values))
        manifest['splits'][split] = {**counts, 'min_tokens': min(lengths, default=0), 'max_tokens': max(lengths, default=0),
            'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'output_sha256': hashlib.sha256((args.output / (split + '.jsonl')).read_bytes()).hexdigest()}
    (args.output / 'token-audit.json').write_text(json.dumps(manifest, indent=2));print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
