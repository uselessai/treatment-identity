"""Counterexamples found by the 2026-09-08 manuscript review."""
import unittest
import numpy as np
from treatment_identity import (ContentContract, StreamContract, Sample,
    check_sample_value_range, check_sample_channel_content, check_separability,
    check_geometry, check_content_scale)

class ReviewRegressions(unittest.TestCase):
    def test_same_units_do_not_create_false_equality(self):
        r = check_separability(lambda: np.full((8,8,3), .5),
                               lambda: np.full((8,8,3), 127.5))
        self.assertEqual(r.status, 'PASS')
        self.assertAlmostEqual(r.evidence['max_abs_delta_0_255'], 127.)

    def test_explicit_different_units_can_establish_identity(self):
        r = check_separability(lambda: np.full((8,8,3), .5),
             lambda: np.full((8,8,3), 127.5), declared='identical',
             value_range_a=(0,1), value_range_b=(0,255))
        self.assertEqual(r.status, 'PASS')

    def test_missing_content_declarations_are_not_inapplicability(self):
        sample = Sample(lq=np.zeros((8,8,3)), gt=None)
        contract = ContentContract(lq=StreamContract())
        for check in (check_sample_value_range, check_sample_channel_content):
            self.assertEqual(check(sample, contract).status, 'UNDECL')

    def test_source_matching_count_does_not_fill_a_missing_target_contract(self):
        import tempfile
        from pathlib import Path
        from selftest import ReferenceLoader
        from treatment_identity import check_target_transforms
        loader = ReferenceLoader()
        with tempfile.TemporaryDirectory() as tmp:
            r = check_target_transforms(loader, Path(tmp), loader.recorder,
                                        "sharpen", expected=1, declared=False)
        self.assertEqual(r.status, 'UNDECL')

    def test_no_comparison_contract_does_not_attribute_a_defect(self):
        r = check_separability(lambda: np.zeros((8,8)),
                               lambda: np.zeros((8,8)), declared=None)
        self.assertEqual(r.status, 'UNDECL')
        self.assertEqual(r.evidence['draws_distinguishable'], 0)

    def test_common_seeds_do_not_supply_exchangeability(self):
        r = check_separability(lambda s: np.full((8,8), s),
             lambda s: np.full((8,8), s+1), seeds=range(4), distributional=True)
        self.assertEqual(r.status, 'SKIP')
        self.assertNotIn('energy_test', r.evidence)

    def test_uniform_resize_keeps_aspect_and_records_scale(self):
        r = check_geometry((256,256), (512,512))
        self.assertEqual(r.status, 'PASS')
        self.assertEqual(r.evidence['scale_ratio_x'], .5)
        self.assertEqual(r.evidence['scale_ratio_y'], .5)

    @staticmethod
    def pattern(h=256, w=256, px=8, py=16):
        y,x = np.mgrid[:h,:w]
        return 100 + 30*np.cos(2*np.pi*x/px) + 30*np.cos(2*np.pi*y/py)

    def test_content_scale_conforming_crop(self):
        r = check_content_scale([(self.pattern(),8,16,16/9)], declared='crop')
        self.assertEqual(r.status, 'PASS', r.message)

    def test_content_scale_detects_anisotropic_resize(self):
        r = check_content_scale([(self.pattern(px=16,py=16),8,16,16/9)], declared='crop')
        self.assertEqual(r.status, 'FAIL', r.message)

    def test_content_scale_missing_contract_is_undeclared_even_for_crop(self):
        r = check_content_scale([(self.pattern(),8,16,16/9)])
        self.assertEqual(r.status, 'UNDECL', r.message)

    def test_content_scale_refuses_unreadable_probe(self):
        r = check_content_scale([(np.ones((256,256)),8,16,16/9)], declared='crop')
        self.assertEqual(r.status, 'SKIP', r.message)

if __name__ == '__main__':
    unittest.main()
