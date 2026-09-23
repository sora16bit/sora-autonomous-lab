from __future__ import annotations

import numpy as np
import pandas as pd
import unittest

from sora_synth.audit import _perm_beta, _roc_tpr, evaluate_aia, evaluate_aia_suite, evaluate_mia, evaluate_shadow_mia
from sora_synth.data import SUBMITTED
from sora_synth.generator import _sample_correlated_normals, _weighted_corr, _weighted_layer_stats


class SoraTests(unittest.TestCase):
    def test_ties_are_one_operating_point(self):
        y = np.array([1, 0, 0, 1, 0, 0])
        score = np.ones(len(y))
        self.assertEqual(_roc_tpr(y, score, 0.01), 0.01)


    def test_permutation_beta_is_reproducible(self):
        y = np.array([1, 0, 0, 0, 1, 0, 0, 0])
        rare = np.array([True, False, False, False, True, False, False, False])
        score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        left = _perm_beta(y, rare, score, 25, 12)
        right = _perm_beta(y, rare, score, 25, 12)
        for key in left:
            if np.isnan(left[key]):
                self.assertTrue(np.isnan(right[key]))
            else:
                self.assertEqual(left[key], right[key])
        self.assertGreaterEqual(left["a_mia"], 0.0)
        self.assertLessEqual(left["a_mia"], 1.0)


    def test_submitted_schema_excludes_record_id(self):
        self.assertNotIn("record_id", SUBMITTED)
        self.assertEqual(len(SUBMITTED), 13)

    def test_aia_proxy_returns_one_value_per_query(self):
        c = pd.DataFrame({
            "age": [50, 60, 70], "sex": [0, 1, 0], "prefecture": [1, 2, 1],
            "BMI": [22.0, 25.0, 27.0], "SBP": [120.0, 130.0, 140.0],
            "TG": [100.0, 120.0, 140.0], "HDL": [60.0, 55.0, 50.0],
            "ALT": [20.0, 25.0, 30.0], "smoking": [0, 1, 0], "FPG": [90.0, 100.0, 110.0],
            "time": [1.0, 2.0, 3.0], "onset": [0, 1, 0], "death": [0, 0, 1],
        })
        result = evaluate_aia(c, c, c)
        self.assertEqual(set(result), {"p_target", "p_control", "R"})
        self.assertEqual(result["p_target"], result["p_control"])

    def test_aia_suite_uses_only_c_as_attack_reference(self):
        c = pd.DataFrame({
            "age": [50, 60, 70], "sex": [0, 1, 0], "prefecture": [1, 2, 1],
            "BMI": [22.0, 25.0, 27.0], "SBP": [120.0, 130.0, 140.0],
            "TG": [100.0, 120.0, 140.0], "HDL": [60.0, 55.0, 50.0],
            "ALT": [20.0, 25.0, 30.0], "smoking": [0, 1, 0], "FPG": [90.0, 100.0, 110.0],
            "time": [1.0, 2.0, 3.0], "onset": [0, 1, 0], "death": [0, 0, 1],
        })
        target = c.copy()
        control = c.copy()
        target["time"] = [99.0, 99.0, 99.0]
        result = evaluate_aia_suite(c, target, control)
        self.assertEqual(result["reference"], "C_only")
        self.assertIn("knn1", result["methods"])
        self.assertIn("conservative_R", result)
        custom = evaluate_aia_suite(c, target, control, attack_specs=[
            {"attack_id": "custom_aia_knn", "family": "aia_knn", "params": {"k": 2}},
        ])
        self.assertEqual(set(custom["methods"]), {"custom_aia_knn"})
        self.assertEqual(custom["conservative_R"], custom["methods"]["custom_aia_knn"]["R"])

    def test_weighted_layer_stats_use_prior_weights_for_categories(self):
        b = pd.DataFrame({
            "BMI": [20.0, 21.0], "SBP": [110.0, 111.0], "sex": [0, 0],
            "smoking": [0, 0], "prefecture": [1, 1],
        })
        prior = pd.DataFrame({
            "BMI": [30.0, 31.0], "SBP": [140.0, 141.0], "sex": [1, 1],
            "smoking": [1, 1], "prefecture": [2, 2],
        })
        model = _weighted_layer_stats(
            b, prior, ["BMI", "SBP"], tau=2.0, component=0,
            b_mask=np.array([True, True]), p_mask=np.array([True, True]),
        )
        sex_values, sex_probs = model.cat_probs["sex"][0]
        self.assertEqual(set(sex_values.tolist()), {0, 1})
        self.assertAlmostEqual(float(sex_probs.sum()), 1.0)
        self.assertAlmostEqual(float(sex_probs[0]), 0.5)

    def test_weighted_corr_is_finite_and_positive_semidefinite(self):
        values = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
        corr = _weighted_corr(values, np.array([0.8, 0.1, 0.1]))
        self.assertTrue(np.isfinite(corr).all())
        self.assertTrue(np.all(np.linalg.eigvalsh(corr) > 0))

    def test_sobol_normals_are_reproducible_with_rng_seed(self):
        corr = np.array([[1.0, 0.25], [0.25, 1.0]])
        left = _sample_correlated_normals(corr, 16, np.random.default_rng(123), quasi=True)
        right = _sample_correlated_normals(corr, 16, np.random.default_rng(123), quasi=True)
        self.assertTrue(np.allclose(left, right))

    def test_shadow_mia_requires_independent_pairs(self):
        c = pd.DataFrame({
            "age": [50, 60, 70], "BMI": [22.0, 25.0, 27.0], "SBP": [120.0, 130.0, 140.0],
            "TG": [100.0, 120.0, 140.0], "HDL": [60.0, 55.0, 50.0], "ALT": [20.0, 25.0, 30.0],
            "FPG": [90.0, 100.0, 110.0], "time": [1.0, 2.0, 3.0], "onset": [0, 1, 0], "death": [0, 0, 1],
        })
        with self.assertRaises(ValueError):
            evaluate_shadow_mia(c, [], [], c, c)
        result = evaluate_shadow_mia(c, [c], [c], c, c, n_perm=5)
        self.assertIn("a_mia", result)

    def test_mia_can_use_holdout_rare_truth_labels(self):
        def frame(offset):
            n = 6
            return pd.DataFrame({
                "age": np.arange(50, 50 + n) + offset, "BMI": np.arange(22, 22 + n) + offset,
                "SBP": np.arange(120, 120 + n) + offset, "TG": np.arange(100, 100 + n) + offset,
                "HDL": np.arange(60, 60 - n, -1) - offset, "ALT": np.arange(20, 20 + n) + offset,
                "FPG": np.arange(90, 90 + n) + offset, "time": np.arange(1, 1 + n) + offset,
                "onset": np.tile([0, 1], 3), "death": np.tile([0, 0, 1], 2),
                "sex": np.tile([0, 1], 3), "prefecture": np.tile([1, 2], 3),
            })
        result = evaluate_mia(frame(0), frame(10), frame(20), reference=frame(40),
                              rare_labels=np.array([True, False, True, False, False, False,
                                                    False, False, False, False, False, False]),
                              n_perm=5, seed=3)
        self.assertEqual(result["_meta"]["rare_kind"], "provided_labels")
        self.assertEqual(result["_meta"]["rare_n"], 2)
