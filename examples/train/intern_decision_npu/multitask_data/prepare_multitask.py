"""Build local decision data from public training splits; never train on test rows."""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def normalized(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class ExclusionIndex:
    """Exact normalization plus >=80% overlap of the shorter word-8-gram set.

    This is a lexical screen, not a guarantee against semantic contamination.
    """
    def __init__(self):
        self.exact = set()
        self.grams = []
        self.inverted = defaultdict(set)

    @staticmethod
    def shingles(text):
        words = normalized(text).split()
        return {tuple(words[i:i + 8]) for i in range(max(0, len(words) - 7))}

    def add(self, text):
        self.exact.add(normalized(text))
        grams = self.shingles(text)
        idx = len(self.grams)
        self.grams.append(grams)
        for gram in grams:
            self.inverted[gram].add(idx)

    def contains(self, text):
        if normalized(text) in self.exact:
            return True
        grams = self.shingles(text)
        if len(grams) < 8:
            return False
        counts = Counter(idx for gram in grams for idx in self.inverted.get(gram, ()))
        return any(min(len(grams), len(self.grams[idx])) >= 8 and
                   count / min(len(grams), len(self.grams[idx])) >= 0.8
                   for idx, count in counts.items())


def tool_case(row):
    marker = 'Here is a list of functions in JSON format that you can invoke:'
    payload = row['system'].split(marker, 1)[1].strip()
    tools, end = json.JSONDecoder().raw_decode(payload)
    if payload[end:].strip() and not payload[end:].strip().startswith('. \nShould you decide'):
        raise ValueError('Unrecognized text after tools')
    names = [tool['name'] for tool in tools]
    if not 2 <= len(names) <= 62 or len(set(names)) != len(names):
        raise ValueError('Candidate count or duplicate names')
    conversation = row['conversations']
    if len(conversation) < 2 or [x['from'] for x in conversation[:2]] != ['user', 'assistant']:
        raise ValueError('First pair must be user then assistant')
    text, answer = conversation[0]['value'], conversation[1]['value'].strip()
    # Tokenize a leading tool name only; never rewrite strings/argument values.
    if not answer.startswith('[') or not answer.endswith(']'):
        raise ValueError('Non-call response')
    call = answer[1:-1].strip()
    matches = [name for name in names if call.startswith(name + '(')]
    if len(matches) != 1:
        raise ValueError('Unknown or ambiguous function')
    name = matches[0]
    parsed = ast.parse('_tool' + call[len(name):], mode='eval').body
    if not isinstance(parsed, ast.Call) or not isinstance(parsed.func, ast.Name) or parsed.func.id != '_tool':
        raise ValueError('Require a single tool call')
    # Arguments are irrelevant to tool selection, but reject executable structures.
    for node in list(parsed.args) + [kw.value for kw in parsed.keywords]:
        ast.literal_eval(node)
    key = digest(normalized(text))
    return {'id': 'toolace-' + key, 'state': {'request': text, 'tools': tools},
            'questions': {'tool': {'type': 'choice', 'instructions': 'Which one of the available tools should be called for this request?',
                                   'criteria': {name: str(tool.get('description', '')) for name, tool in zip(names, tools)}}},
            'targets': {'tool': {'label': name}}, 'source': 'ToolACE', 'group': key,
            '_text': text, '_label': name}


def safety_case(row):
    if row['prompt_label_source'] != 'human' or row['prompt_label'] not in ('safe', 'unsafe'):
        raise ValueError('Require human prompt safety label')
    text = row['prompt']
    if not isinstance(text, str) or not normalized(text):
        raise ValueError('Empty prompt')
    key = digest(normalized(text))
    label = 'yes' if row['prompt_label'] == 'unsafe' else 'no'
    return {'id': 'aegis-' + key, 'state': {'prompt': text},
            'questions': {'unsafe': {'type': 'noul', 'instructions': 'Does this user prompt request or promote harmful or unsafe conduct?',
                                     'criteria': {'yes': 'The prompt is unsafe.', 'no': 'The prompt is safe.'}}},
            'targets': {'unsafe': {'label': label}}, 'source': 'Aegis-2.0', 'group': key,
            '_text': text, '_label': label}


def unique_cases(rows, converter, counts):
    groups = defaultdict(list)
    for row in rows:
        try:
            result = converter(row)
            groups[result['group']].append(result)
        except (ValueError, KeyError, TypeError, IndexError, SyntaxError):
            counts['invalid_or_unsupported'] += 1
    accepted = []
    for items in groups.values():
        # Different candidate sets for the same request also make tool gold ambiguous.
        signatures = {json.dumps((r['state'], r['targets']), sort_keys=True) for r in items}
        if len(signatures) != 1:
            counts['conflicting_or_ambiguous_group_rows'] += len(items)
            continue
        accepted.append(items[0]); counts['duplicate_rows'] += len(items) - 1
    return sorted(accepted, key=lambda x: x['id'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--heldout-inputs', type=Path, required=True, help='Input text only; no test targets')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    index = ExclusionIndex()
    heldout = json.loads(args.heldout_inputs.read_text())
    for rows in heldout.values():
        for row in rows:
            assert set(row) <= {'id', 'text'}, 'Test labels must not enter this preparation interface'
            index.add(row['text'])
    manifest = {'protocol': 'lexical exclusion, frozen source validation, no test labels', 'sources': {},
                'heldout_input_sha256': digest(args.heldout_inputs.read_text()), 'token_audit': 'pending; not ready for training'}
    splits = {'train': [], 'validation': []}
    for filename, converter, fixed_split in [('aegis-validation.json', safety_case, 'validation'),
                                            ('toolace-data.json', tool_case, None),
                                            ('aegis-train.json', safety_case, 'train')]:
        path = args.sources / filename; raw = path.read_text(); rows = json.loads(raw)
        counts = Counter(raw_rows=len(rows))
        cases = unique_cases(rows, converter, counts)
        staged = []
        for case in cases:
            split = fixed_split or ('validation' if int(case['group'][:8], 16) % 10 == 0 else 'train')
            staged.append((split, case))
        # Validation first so similar train prompts are excluded, never re-assigned.
        staged.sort(key=lambda item: (item[0] != 'validation', item[1]['id']))
        for split, case in staged:
            if index.contains(case['_text']):
                counts['excluded_overlap'] += 1
                continue
            index.add(case['_text'])
            case.pop('_text'); case.pop('_label')
            splits[split].append(case); counts[split + '_retained'] += 1
        manifest['sources'][filename] = {'sha256': digest(raw), **counts}
    assert not {r['group'] for r in splits['train']} & {r['group'] for r in splits['validation']}
    for split, rows in splits.items():
        if {r['source'] for r in rows} != {'ToolACE', 'Aegis-2.0'}:
            raise ValueError('Every split must contain both supported training sources')
    args.output.mkdir(parents=True)
    for split, rows in splits.items():
        rows.sort(key=lambda row: row['id'])
        content = ''.join(json.dumps(row, ensure_ascii=True) + '\n' for row in rows)
        (args.output / (split + '.jsonl')).write_text(content)
        manifest[split] = {'rows': len(rows), 'sha256': digest(content),
                           'sources': dict(Counter(r['source'] for r in rows))}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
