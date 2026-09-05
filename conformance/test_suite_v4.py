"""Synthetic final-state checks for buffer role exchanges, not GPU evidence."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from compare import validate_capture, CaptureError
from test_compare import synthetic_report

class RebindingSuiteTests(unittest.TestCase):
    def setUp(self):
        raw=Path(__file__).with_name('suite-v4.json').read_bytes()
        self.suite=json.loads(raw); self.digest=hashlib.sha256(raw).hexdigest()
        self.report=synthetic_report(self.suite,self.digest)

    def test_all_backends_report_complete_ever_written_pool(self):
        for backend in ('vulkan','native-metal','native-metal-provider'):
            validate_capture(self.suite,self.digest,synthetic_report(self.suite,self.digest,backend),backend)
        self.assertEqual(len(self.suite['cases']),4)
        self.assertEqual(len(self.suite['cases'][-1]['expected_writebacks']),2)

    def test_simulate_each_dispatch_by_view_identity(self):
        for case in self.suite['cases']:
            pool={b['view']:bytearray.fromhex(b['initial_hex']) for b in case['buffers']}
            for dispatch in case['dispatches']:
                mapping=dispatch['bindings']
                if case['entry']=='copy_word':
                    pool[mapping[1]][:]=pool[mapping[0]]
                else:
                    a,b,c=mapping;bias=int.from_bytes(pool[b],'little')
                    for offset in range(0,120,4):
                        value=(int.from_bytes(pool[a][offset:offset+4],'little')+bias)&0xffffffff
                        pool[a][offset:offset+4]=value.to_bytes(4,'little')
                        pool[c][offset:offset+4]=(value^0xa5a55a5a).to_bytes(4,'little')
            for expected in case['expected_writebacks']:
                self.assertEqual(pool[expected['view']].hex(),expected['bytes_hex'])

    def test_copy_first_read_only_view_must_be_returned_after_later_write(self):
        self.report['results'][-1]['writebacks'].pop(0)
        with self.assertRaisesRegex(CaptureError,'writable set mismatch'):
            validate_capture(self.suite,self.digest,self.report)

    def test_same_table_instead_of_rebinding_cannot_match_transform(self):
        previous=json.loads(Path(__file__).with_name('suite-v3.json').read_text())
        for index in range(3):
            report=copy.deepcopy(self.report)
            report['results'][index]['writebacks']=previous['cases'][index]['expected_writebacks']
            with self.assertRaisesRegex(CaptureError,'differing byte'):
                validate_capture(self.suite,self.digest,report)

    def test_duplicate_unknown_or_missing_view_mapping_rejected(self):
        for mapping in ([410,420,410],[410,420,999],[410,420]):
            suite=copy.deepcopy(self.suite)
            suite['cases'][0]['dispatches'][1]['bindings']=mapping
            with self.assertRaisesRegex(CaptureError,'permute every resource'):
                validate_capture(suite,self.digest,self.report)

    def test_short_scalar_cannot_be_rebound_to_array_slot(self):
        self.suite['cases'][0]['dispatches'][1]['bindings']=[420,410,400]
        with self.assertRaisesRegex(CaptureError,'binding extent'):
            validate_capture(self.suite,self.digest,self.report)

if __name__=='__main__': unittest.main()
