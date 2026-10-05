import json
import unittest
from prepare_multitask import ExclusionIndex, safety_case, tool_case, unique_cases
from collections import Counter


class PreparationTests(unittest.TestCase):
    def tool(self, answer):
        return {'system': 'Here is a list of functions in JSON format that you can invoke:' + json.dumps([
            {'name': 'Market API', 'description': 'market'}, {'name': 'other', 'description': 'other'}]) + '. \nShould you decide to return the function call(s).',
                'conversations': [{'from': 'user', 'value': 'Find market trends'}, {'from': 'assistant', 'value': answer}]}

    def test_single_non_identifier_name(self):
        case = tool_case(self.tool('[Market API(region="US")]'))
        self.assertEqual(case['targets']['tool']['label'], 'Market API')
        self.assertIn('instructions', case['questions']['tool'])

    def test_ambiguous_and_executable_calls_rejected(self):
        for answer in ['[Market API(), other()]', '[Market API(x=__import__("os"))]', 'No tool required']:
            with self.assertRaises((ValueError, SyntaxError)):
                tool_case(self.tool(answer))

    def test_prompt_gold_never_derived_from_response(self):
        case = safety_case({'prompt': 'Hello', 'prompt_label_source': 'human', 'prompt_label': 'safe',
                            'response_label': 'unsafe', 'response': 'not an input'})
        self.assertEqual(case['targets']['unsafe']['label'], 'no')
        self.assertEqual(case['state'], {'prompt': 'Hello'})

    def test_conflicting_labels_drop_entire_group(self):
        rows = [{'prompt': 'identical prompt', 'prompt_label_source': 'human', 'prompt_label': label}
                for label in ['safe', 'unsafe']]
        self.assertEqual(unique_cases(rows, safety_case, Counter()), [])

    def test_exact_and_near_exclusion(self):
        index = ExclusionIndex(); text = ' '.join('word' + str(i) for i in range(50)); index.add(text)
        self.assertTrue(index.contains(text.upper() + '!'))
        self.assertTrue(index.contains('wrapper prefix ' + text + ' appended suffix'))
        self.assertFalse(index.contains('unrelated short request'))


if __name__ == '__main__':
    unittest.main()
