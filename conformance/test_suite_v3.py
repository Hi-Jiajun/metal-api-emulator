"""Synthetic checks of serial-pass expectations, not evidence of a GPU run."""

import copy
import hashlib
import json
from pathlib import Path
import unittest

from compare import CaptureError, validate_capture
from test_compare import synthetic_report


class SerialSuiteTests(unittest.TestCase):
    def setUp(self):
        raw = Path(__file__).with_name('suite-v3.json').read_bytes()
        self.suite = json.loads(raw)
        self.digest = hashlib.sha256(raw).hexdigest()
        self.report = synthetic_report(self.suite, self.digest)

    def test_all_three_backend_report_roles_support_serial_results(self):
        for backend in ('native-metal', 'native-metal-provider', 'vulkan'):
            report = synthetic_report(self.suite, self.digest, backend)
            validate_capture(self.suite, self.digest, report, backend)
        self.assertEqual([len(c['dispatches']) for c in self.suite['cases']], [2, 3, 8])

    def test_final_values_include_every_read_modify_write_dispatch(self):
        for case in self.suite['cases']:
            initial = bytes.fromhex(case['buffers'][0]['initial_hex'])
            amount = int.from_bytes(bytes.fromhex(case['buffers'][1]['initial_hex']), 'little')
            values = [int.from_bytes(initial[i:i+4], 'little') for i in range(0, len(initial), 4)]
            for dispatch in case['dispatches']:
                self.assertEqual(dispatch['grid'], [5, 3, 2])
                values = [(value + amount) & 0xffffffff for value in values]
            for write in case['expected_writebacks']:
                expected = values if write['allocation'] == 310 else [v ^ 0xa5a55a5a for v in values]
                self.assertEqual(bytes.fromhex(write['bytes_hex']), b''.join(v.to_bytes(4,'little') for v in expected))

    def test_reuploading_initial_bytes_between_passes_cannot_pass(self):
        v2 = json.loads(Path(__file__).with_name('suite-v2.json').read_text())
        single_pass = next(c for c in v2['cases'] if c['id'] == 'transform_tail')
        for index in range(3):
            report = copy.deepcopy(self.report)
            report['results'][index]['writebacks'] = copy.deepcopy(single_pass['expected_writebacks'])
            with self.assertRaisesRegex(CaptureError, 'differing byte'):
                validate_capture(self.suite, self.digest, report)

    def test_previous_completion_cannot_satisfy_a_longer_sequence(self):
        self.report['results'][1]['writebacks'] = copy.deepcopy(self.report['results'][0]['writebacks'])
        with self.assertRaisesRegex(CaptureError, 'differing byte'):
            validate_capture(self.suite, self.digest, self.report)

    def test_per_pass_duplicate_writebacks_are_not_final_outputs(self):
        self.report['results'][0]['writebacks'] *= 2
        with self.assertRaisesRegex(CaptureError, 'duplicate writeback'):
            validate_capture(self.suite, self.digest, self.report)

    def test_previous_suite_cannot_be_reused_as_v3_evidence(self):
        self.report['suite'] = 'compute-buffer-v2'
        with self.assertRaisesRegex(CaptureError, 'suite name mismatch'):
            validate_capture(self.suite, self.digest, self.report)


if __name__ == '__main__':
    unittest.main()
