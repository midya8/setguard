"""Finite input-symmetry checks, not a conformal-coverage theorem."""
import copy
import itertools
import unittest

from setguard.core import fit, predict
from setguard.ledger_fixtures import cases


def relabel(data, order):
    """New column j contains old column order[j]; remap true indices too."""
    inverse = {old: new for new, old in enumerate(order)}
    result = {
        'class_names': [data['class_names'][old] for old in order],
        'probabilities': [[row[old] for old in order] for row in data['probabilities']],
    }
    if 'labels' in data:
        result['labels'] = [inverse[y] for y in data['labels']]
    return result


def named_sets(model, candidates):
    return [frozenset(model['class_names'][j] for j in row)
            for row in predict(model, candidates)]


class InputSymmetryTests(unittest.TestCase):
    def test_all_class_permutations(self):
        for case in cases():
            calibration, candidates = case['calibration'], case['candidates']
            before = copy.deepcopy(case)
            width = len(calibration['class_names'])
            for alpha in [i / 20 for i in range(1, 20)]:
                for mode in ('pooled', 'classwise'):
                    original = fit(calibration, alpha, mode)
                    expected = named_sets(original, candidates)
                    for order in itertools.permutations(range(width)):
                        with self.subTest(case=case['id'], alpha=alpha, mode=mode, order=order):
                            changed_calibration = relabel(calibration, order)
                            changed_candidates = relabel(candidates, order)
                            frozen = copy.deepcopy((changed_calibration, changed_candidates))
                            changed = fit(changed_calibration, alpha, mode)
                            artifact = copy.deepcopy(original)
                            artifact['class_names'] = [original['class_names'][i] for i in order]
                            if mode == 'classwise':
                                for field in ('class_counts', 'ranks', 'thresholds'):
                                    artifact[field] = [original[field][i] for i in order]
                            self.assertEqual(changed, artifact)
                            self.assertEqual(named_sets(changed, changed_candidates), expected)
                            self.assertEqual((changed_calibration, changed_candidates), frozen)
            self.assertEqual(case, before)

    def test_calibration_row_order_and_query_row_order(self):
        for case in cases():
            calibration, candidates = case['calibration'], case['candidates']
            count = len(calibration['labels'])
            for alpha in [i / 20 for i in range(1, 20)]:
                for mode in ('pooled', 'classwise'):
                    original = fit(calibration, alpha, mode)
                    for order in (list(reversed(range(count))), list(range(1, count)) + [0]):
                        changed = copy.deepcopy(calibration)
                        changed['probabilities'] = [calibration['probabilities'][i] for i in order]
                        changed['labels'] = [calibration['labels'][i] for i in order]
                        with self.subTest(case=case['id'], alpha=alpha, mode=mode, order=order):
                            self.assertEqual(fit(changed, alpha, mode), original)
                    changed = copy.deepcopy(candidates)
                    changed['probabilities'].reverse()
                    self.assertEqual(named_sets(original, changed),
                                     list(reversed(named_sets(original, candidates))))

    def test_alpha_nesting_including_infinite_thresholds(self):
        saw_infinite = saw_finite = saw_empty = saw_strict = False
        for case in cases():
            for mode in ('pooled', 'classwise'):
                previous = None
                for alpha in [i / 20 for i in range(1, 20)]:
                    model = fit(case['calibration'], alpha, mode)
                    current = named_sets(model, case['candidates'])
                    thresholds = model['thresholds'] if mode == 'classwise' else [model['threshold']]
                    saw_infinite |= any(q is None for q in thresholds)
                    saw_finite |= any(q is not None for q in thresholds)
                    saw_empty |= any(not row for row in current)
                    if previous is not None:
                        for old, new in zip(previous, current):
                            with self.subTest(case=case['id'], mode=mode, alpha=alpha):
                                self.assertLessEqual(new, old)
                                saw_strict |= new < old
                    previous = current
        self.assertTrue(saw_infinite and saw_finite and saw_empty and saw_strict)


if __name__ == '__main__':
    unittest.main()
