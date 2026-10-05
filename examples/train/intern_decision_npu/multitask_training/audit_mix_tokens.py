"""Verify prepared messages, masked labels and actual tokenizer lengths on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping


def main():
    p = argparse.ArgumentParser();p.add_argument('--data', type=Path, required=True)
    p.add_argument('--tokenizer', required=True);a = p.parse_args()
    from transformers import AutoTokenizer
    from decision_schema import compile_row
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    marker = tok.encode('<decision>', add_special_tokens=False);assert len(marker) == 1
    results = {}
    for name in ('train-raw', 'validation-business', 'validation-tool', 'validation-safety'):
        path = a.data / (name + '.jsonl');rows = [json.loads(s) for s in path.read_text().splitlines()]
        lengths = [];compiled = []
        for row in rows:
            example = compile_row(row);assert example.messages == compile_row(row, include_targets=False).messages
            ids = tok.apply_chat_template(example.messages, tokenize=True, add_generation_prompt=False, enable_thinking=False)
            if isinstance(ids, Mapping):ids = ids['input_ids']
            assert len(ids) <= 8192 and ids.count(marker[0]) == len(example.fields)
            targets = [example.targets[f] for f in example.fields]
            assert all(len(tok.encode(t, add_special_tokens=False)) == 1 for t in targets)
            lengths.append(len(ids));compiled.append({'case_id': row['id'], 'messages': example.messages, 'decision_targets': targets})
        if name == 'train-raw':
            assert compiled == [json.loads(s) for s in (a.data / 'train.jsonl').read_text().splitlines()]
        results[name] = {'rows': len(rows), 'max_tokens': max(lengths), 'min_tokens': min(lengths),
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (a.data / 'token-audit.json').write_text(json.dumps({'status': 'complete', 'no_truncation': True, 'splits': results}, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
