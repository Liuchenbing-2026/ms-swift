import copy
import unittest
from prepare_acceptance import pack, normalize
class Checks(unittest.TestCase):
    def test_targets_never_change_inputs(self):
        r={'id':'fixture','state':'evidence','questions':{'x':{'type':'choice','instructions':'pick','criteria':{'left':'L','right':'R'}},'y':{'type':'noul','instructions':'valid?','criteria':{}}},'targets':{'x':{'label':'left'},'y':{'label':'yes'}}}
        before=pack(r);other=copy.deepcopy(r);other['targets']['x']['label']='right';after=pack(other)
        self.assertEqual(before['messages'],after['messages']);self.assertNotEqual(before['targets'],after['targets'])
        self.assertEqual(before['fields'],['x','y']);self.assertEqual(before['messages'][-1]['content'].count('<decision>'),2)
        r['questions']['x']['answer']={'label':'right'}
        self.assertEqual(pack(r)['messages'],before['messages'])
    def test_schema_and_limits(self):
        q=normalize({'type':'choice','instructions':{'text':'pick'},'criteria':['a','b']})
        self.assertEqual(list(q['criteria']),['a','b'])
        with self.assertRaises(ValueError):pack({'id':'x','state':'s','questions':{'x':{'type':'choice','instructions':'pick','criteria':{str(i):'' for i in range(63)}}},'targets':{'x':{'label':'0'}}})
if __name__=='__main__':unittest.main()
