"""Prepare joint-field training without changing the frozen case split."""
import argparse
import hashlib
import json
from pathlib import Path
import random

from decision_schema import compile_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(exist_ok=False, parents=True)
    seen, manifest = set(), {}
    for split in ('train', 'validation', 'calibration', 'test'):
        path = Path(args.prepared) / (split + '.jsonl')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        ids = {row['id'] for row in rows}
        assert len(ids) == len(rows) and not ids & seen
        seen |= ids
        manifest[split] = {'cases': len(rows), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if split not in ('train', 'validation'):
            continue  # Only identity auditing; no test examples compiled for training.
        if split == 'train':
            random.Random(42).shuffle(rows)
        compiled = []
        for row in rows:
            assert len(row['questions']) == 5
            clean = {key: row[key] for key in ('state', 'questions', 'targets')}
            example = compile_row(clean)
            assert example.messages == compile_row(clean, include_targets=False).messages
            compiled.append({'case_id': row['id'], 'messages': example.messages,
                             'decision_targets': [example.targets[field] for field in example.fields]})
        destination = output / (split + '.jsonl')
        destination.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in compiled))
        manifest[split].update(decisions=5 * len(compiled), compiled_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
