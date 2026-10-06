import copy
import unittest
from final_evaluation import ready, compare

class Checks(unittest.TestCase):
    def test_waits_for_both_and_rejects_failure(self):
        complete = {'status': 'training_finished_pending_independent_evaluation'}
        self.assertFalse(ready(complete, {'status':'relocating'}))
        self.assertTrue(ready(complete, {'status':'complete'}))
        with self.assertRaises(RuntimeError):ready({'status':'failed'}, {'status':'complete'})
    def test_protocol_and_no_update_gate(self):
        record = {'status':'complete', 'optimizer_updates_performed':0,
            'suites': {'test': {'correct':1, 'total':2, 'role':'test_report_only', 'data_sha256':'a','input_hash':'b'}}}
        self.assertEqual(compare(record, copy.deepcopy(record))['test']['total'],2)
        changed=copy.deepcopy(record);changed['suites']['test']['input_hash']='different'
        with self.assertRaises(AssertionError):compare(record,changed)
        changed=copy.deepcopy(record);changed['optimizer_updates_performed']=1
        with self.assertRaises(AssertionError):compare(record,changed)

if __name__=='__main__':unittest.main()
