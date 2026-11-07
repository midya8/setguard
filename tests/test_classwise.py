import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from setguard.core import fit, predict, evaluate, threshold, corrected_rank
from setguard.cli import main


def data():
    return {'class_names': ['a', 'b', 'c'],
            'probabilities': [[.9, .08, .02]] * 9 + [[.2, .7, .1]],
            'labels': [0] * 9 + [1]}


def unlabeled():
    return {'class_names': ['a', 'b', 'c'], 'probabilities': [[.8, .19, .01]]}


class TestClasswise(unittest.TestCase):
    def test_separate_counts_ranks_and_thresholds(self):
        m = fit(data(), .1, mode='classwise')
        self.assertEqual(m['class_counts'], [9, 1, 0])
        self.assertEqual(m['ranks'], [9, 2, 1])
        self.assertAlmostEqual(m['thresholds'][0], .1)
        self.assertEqual(m['thresholds'][1:], [None, None])

    def test_rare_and_absent_class_always_included(self):
        self.assertEqual(predict(fit(data(), mode='classwise'), unlabeled()), [[1, 2]])

    def test_default_still_version_one(self):
        self.assertEqual(fit(data())['version'], 1)
        self.assertEqual(fit(data())['method'], 'split-conformal-1-minus-p')

    def test_json_roundtrip_and_no_mutation(self):
        d = data(); saved = copy.deepcopy(d)
        m = fit(d, mode='classwise'); frozen = copy.deepcopy(m)
        self.assertEqual(predict(json.loads(json.dumps(m)), unlabeled()), predict(m, unlabeled()))
        self.assertEqual(d, saved); self.assertEqual(m, frozen)

    def test_count_and_shape_corruption_rejected(self):
        for key, value in [('class_counts', [True, 1, 0]), ('class_counts', [9, 0, 0]),
                           ('ranks', [9, 1, 1]), ('thresholds', [.1, .1, None]),
                           ('thresholds', [.1, None]), ('method', 'other')]:
            m = fit(data(), mode='classwise'); m[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):predict(m, unlabeled())

    def test_class_order_rejected(self):
        u = unlabeled(); u['class_names'].reverse()
        with self.assertRaises(ValueError):predict(fit(data(), mode='classwise'), u)

    def test_group_score_uses_only_its_true_class(self):
        d = data(); a = fit(d, .5, 'classwise')
        d['probabilities'][-1] = [.1, .3, .6]
        b = fit(d, .5, 'classwise')
        self.assertEqual(a['thresholds'][0], b['thresholds'][0])
        self.assertNotEqual(a['thresholds'][1], b['thresholds'][1])

    def test_groupwise_rank_invariant_with_ties(self):
        d = data(); m = fit(d, .5, 'classwise')
        for k, n in enumerate(m['class_counts']):
            if n:
                scores = [1-row[y] for row,y in zip(d['probabilities'], d['labels']) if y==k]
                self.assertEqual(m['thresholds'][k], threshold(scores, .5))
                self.assertEqual(m['ranks'][k], corrected_rank(n, .5))

    def test_cli_and_evaluation_method(self):
        out = io.StringIO()
        with patch('sys.stdin', io.StringIO(json.dumps(data()))), redirect_stdout(out):
            self.assertEqual(main(['fit','--input','-','--mode','classwise']),0)
        m=json.loads(out.getvalue())
        self.assertEqual(evaluate(m,data())['method'], 'classwise-split-conformal-1-minus-p')

    def test_demo_reports_below_target_class_honestly(self):
        from setguard.demo import run_demo
        result = run_demo()["classwise_same_generator"]
        self.assertEqual(result["coverage"], .9285)
        self.assertEqual(result["mean_set_size"], 3.1095)
        self.assertLess(result["per_class"][3]["coverage"], .9)
        self.assertIn("true label", result["target_scope"])

    def test_unknown_mode_fails(self):
        with self.assertRaises(ValueError):fit(data(), mode='anything')
