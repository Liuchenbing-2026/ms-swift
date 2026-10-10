"""Select the complete seven suites for a zero-update base-model evaluation."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    config = json.loads(args.source.read_text())
    config['scope'] = 'official7_base_no_decision_training'
    config['suites'] = [dict(s, batch_size=args.batch_size)
                        for s in config['suites'] if s['name'].startswith('official-')]
    expected = {'official-easy': (48, 48), 'official-original': (72, 72),
                'official-hard': (111, 111), 'official-agnews': (7600, 7600),
                'official-toolace': (310, 310), 'official-typed': (400, 2000),
                'official-wildjailbreak': (2210, 2210)}
    actual = {s['name']: (s['expected_rows'], s['expected_decisions'])
              for s in config['suites']}
    if len(config['suites']) != 7 or actual != expected:
        raise ValueError('Expected all seven suites with their exact denominators')
    with args.output.open('x') as stream:
        json.dump(config, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
