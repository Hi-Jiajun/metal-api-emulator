"""Synthetic multi-shader reference calculations; no GPU evidence generated."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from compare import validate_capture, CaptureError
from test_compare import synthetic_report

class MixedPipelineTests(unittest.TestCase):
    def setUp(self):
        raw=Path(__file__).with_name('suite-v5.json').read_bytes()
        self.suite=json.loads(raw);self.digest=hashlib.sha256(raw).hexdigest()
        self.report=synthetic_report(self.suite,self.digest)

    def test_all_three_backends_accept_complete_multi_program_results(self):
        for backend in ('native-metal','native-metal-provider','vulkan'):
            validate_capture(self.suite,self.digest,synthetic_report(self.suite,self.digest,backend),backend)

    def test_independent_reference_applies_selected_shader_in_order(self):
        for case in self.suite['cases']:
            pool={b['view']:bytearray.fromhex(b['initial_hex']) for b in case['buffers']}
            for d in case['dispatches']:
                a,b,c=d['bindings'];bias=int.from_bytes(pool[b],'little')
                entry=case['programs'][d['program']]['entry']
                for offset in range(0,120,4):
                    old=int.from_bytes(pool[a][offset:offset+4],'little')
                    if entry=='transform_3d':
                        value=(old+bias)&0xffffffff; output=value^0xa5a55a5a
                    elif entry=='mix_3d':
                        value=((old^bias)*3+1)&0xffffffff; output=(value+0x10203040)&0xffffffff
                    else: self.fail(entry)
                    pool[a][offset:offset+4]=value.to_bytes(4,'little')
                    pool[c][offset:offset+4]=output.to_bytes(4,'little')
            for expected in case['expected_writebacks']:
                self.assertEqual(pool[expected['view']].hex(),expected['bytes_hex'])

    def test_using_only_previous_shader_cannot_match_expected_output(self):
        v4=json.loads(Path(__file__).with_name('suite-v4.json').read_text())
        for index in range(3):
            wrong=copy.deepcopy(self.report)
            wrong['results'][index]['writebacks']=v4['cases'][index]['expected_writebacks']
            with self.assertRaisesRegex(CaptureError,'differing byte'):
                validate_capture(self.suite,self.digest,wrong)

    def test_missing_or_out_of_range_program_selection_is_rejected(self):
        for program in (None,2,-1,True):
            suite=copy.deepcopy(self.suite);suite['cases'][0]['dispatches'][1]['program']=program
            with self.assertRaisesRegex(CaptureError,'dispatch.program'):
                validate_capture(suite,self.digest,self.report)

    def test_unused_or_duplicate_programs_are_rejected(self):
        self.suite['cases'][0]['dispatches'][1]['program']=0
        with self.assertRaisesRegex(CaptureError,'unused program'):
            validate_capture(self.suite,self.digest,self.report)
        self.suite['cases'][0]['programs'][1]=copy.deepcopy(self.suite['cases'][0]['programs'][0])
        with self.assertRaisesRegex(CaptureError,'duplicate program'):
            validate_capture(self.suite,self.digest,self.report)

if __name__=='__main__':unittest.main()
