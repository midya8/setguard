import copy
import itertools
import json
import unittest

from setguard.core import fit, predict, evaluate, corrected_rank, threshold, rank_audit
from setguard.demo import run_demo


def data():
    return {'class_names': ['cat', 'dog'],
            'probabilities': [[.9, .1], [.8, .2], [.7, .3], [.6, .4]],
            'labels': [0, 0, 0, 0]}


def unlabeled(rows):
    return {'class_names': ['cat', 'dog'], 'probabilities': rows}


class TestCore(unittest.TestCase):
    def test_decimal_rank_boundary(self):
        self.assertEqual(corrected_rank(99, .7), 30)
        self.assertEqual(corrected_rank(9, .1), 9)
        self.assertEqual(corrected_rank(4, .01), 5)

    def test_threshold_order_statistic_not_interpolation(self):
        self.assertEqual(threshold([.8, .1, .6, .2], .4), .6)

    def test_all_labels_boundary_is_json_null(self):
        model = fit(data(), .01)
        self.assertIsNone(model['threshold'])
        self.assertEqual(predict(model, unlabeled([[.99, .01]])), [[0, 1]])
        self.assertEqual(json.loads(json.dumps(model)), model)

    def test_inclusive_ties_and_empty_sets(self):
        model = fit(data(), .5)
        q = model['threshold']
        self.assertIn(0, predict(model, unlabeled([[.7, .3]]))[0])
        self.assertEqual(predict(model, unlabeled([[.5, .5]])), [[]])
        self.assertAlmostEqual(q, .3)

    def test_nested_sets_as_alpha_decreases(self):
        rows = unlabeled([[.5, .5], [.8, .2], [.3, .7]])
        small = predict(fit(data(), .6), rows)
        large = predict(fit(data(), .2), rows)
        for a, b in zip(small, large):
            self.assertTrue(set(a) <= set(b))

    def test_serialized_roundtrip_and_no_mutation(self):
        calibration = data(); saved = copy.deepcopy(calibration)
        model = json.loads(json.dumps(fit(calibration, .2)))
        rows = unlabeled([[.8, .2]]); frozen = copy.deepcopy(rows)
        self.assertEqual(predict(model, rows), [[0]])
        self.assertEqual(calibration, saved); self.assertEqual(rows, frozen)

    def test_class_order_required(self):
        rows = unlabeled([[.8, .2]]); rows['class_names'].reverse()
        with self.assertRaises(ValueError):predict(fit(data()), rows)

    def test_no_label_access_during_prediction(self):
        with self.assertRaises(ValueError): predict(fit(data()), data())

    def test_bad_probabilities_rejected(self):
        for row in [[.4, .4], [1.1, -.1], [float('nan'), .5], [True, 0], [1], []]:
            d = data(); d['probabilities'][0] = row
            with self.subTest(row=row), self.assertRaises(ValueError):fit(d)

    def test_bad_labels_rejected(self):
        for y in [True, -1, 2, .5, '0']:
            d = data(); d['labels'][0] = y
            with self.subTest(y=y), self.assertRaises(ValueError):fit(d)

    def test_bad_alpha_rejected(self):
        for a in [0, 1, -.1, float('nan'), float('inf'), True, '.1']:
            with self.subTest(a=a), self.assertRaises(ValueError): fit(data(), a)

    def test_bad_class_names_rejected(self):
        for names in [['x', 'x'], ['', 'y'], ['x'], [True, 'y']]:
            d = data(); d['class_names'] = names
            with self.subTest(names=names), self.assertRaises(ValueError):fit(d)

    def test_invalid_artifacts_rejected(self):
        for key, value in [('version', True), ('method', 'other'), ('rank', 1),
                           ('threshold', float('inf')), ('calibration_n', 0)]:
            model = fit(data(), .5); model[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): predict(model, unlabeled([[.5, .5]]))
        model = fit(data(), .01); model['threshold'] = 1
        with self.assertRaises(ValueError):predict(model, unlabeled([[.5, .5]]))

    def test_absent_class_coverage_is_null(self):
        result = evaluate(fit(data(), .2), data())
        self.assertIsNone(result['per_class'][1]['coverage'])
        self.assertEqual(result['size_histogram'], {'1': 4})
        self.assertEqual(result['coverage'], 1.)

    def test_rank_bound_exhaustive_tied_multisets(self):
        for scores in itertools.product([0., .5, 1.], repeat=5):
            for alpha in [.1, .2, .4, .7, .9]:
                audit = rank_audit(list(scores), alpha)
                self.assertGreaterEqual(audit['coverage'], audit['rank_lower_bound'])

    def test_rank_audit_nontrivial_exact_count(self):
        audit = rank_audit([i / 20 for i in range(21)], .1)
        self.assertEqual(audit['covered'], 19)
        self.assertEqual(audit['coverage'], audit['rank_lower_bound'])

    def test_demo_regression_and_shift_limit(self):
        r = run_demo()
        self.assertEqual(r['same_generator']['coverage'], .9235)
        self.assertEqual(r['same_generator']['mean_set_size'], 3.017)
        self.assertEqual(r['adversarial_label_shift']['coverage'], .076)
        self.assertEqual(r['same_generator']['size_histogram'], r['adversarial_label_shift']['size_histogram'])
