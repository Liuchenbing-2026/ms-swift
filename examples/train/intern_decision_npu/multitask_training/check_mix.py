import unittest
from prepare_mix import build


def case(identity, source='business', label='no'):
    return {'id': identity, 'source': source, 'targets': {'unsafe': {'label': label}}}


class MixChecks(unittest.TestCase):
    def fixture(self):
        return {'business_train': [case('b' + str(i)) for i in range(4)],
                'business_validation': [case('bv')],
                'multitask_train': [case('t' + str(i), 'ToolACE') for i in range(4)] +
                    [case(label + str(i), 'Aegis-2.0', label) for label in ('yes', 'no') for i in range(4)],
                'multitask_validation': [case('tv', 'ToolACE'), case('sv', 'Aegis-2.0')]}

    def test_balanced_unique_reproducible(self):
        rows, validation = build(**self.fixture(), per_source=4)
        self.assertEqual(rows, build(**self.fixture(), per_source=4)[0])
        self.assertEqual(len({row['id'] for row in rows}), 12)
        self.assertEqual([r['mixture_source'] for r in rows], ['business', 'tool', 'safety'] * 4)
        self.assertEqual(sum(r['targets']['unsafe']['label'] == 'yes' for r in rows), 2)

    def test_overlap_and_short_source_rejected(self):
        data = self.fixture();data['business_validation'] = data['business_train'][:1]
        with self.assertRaises(ValueError):
            build(**data, per_source=4)
        with self.assertRaises(ValueError):
            build(**self.fixture(), per_source=6)


if __name__ == '__main__':
    unittest.main()
