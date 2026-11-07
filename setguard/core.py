"""Split-conformal classification with score 1 - probability of true class.

Algorithm: Angelopoulos and Bates, arXiv:2107.07511, Section 1.
Implementation is dependency-free and original to this repository.
"""
import math
from fractions import Fraction


def _alpha(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
        raise ValueError('alpha must be finite and strictly between 0 and 1')
    return Fraction(str(value))


def corrected_rank(n, alpha):
    """Exact ceiling for the decimal representation of alpha, not float product."""
    if type(n) is not int or n < 1:
        raise ValueError('calibration size must be a positive integer')
    value = (n + 1) * (1 - _alpha(alpha))
    return (value.numerator + value.denominator - 1) // value.denominator


def threshold(scores, alpha):
    if not isinstance(scores, list) or not scores or any(
        type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in scores
    ):
        raise ValueError('scores must be a nonempty list in [0,1]')
    k = corrected_rank(len(scores), alpha)
    return None if k > len(scores) else sorted(scores)[k - 1]


def _names(names):
    if not isinstance(names, list) or len(names) < 2 or any(
        not isinstance(n, str) or not n.strip() for n in names
    ) or len(set(names)) != len(names):
        raise ValueError('class_names must be distinct nonempty strings, at least two')


def validate(data, labeled):
    fields = {'class_names', 'probabilities'} | ({'labels'} if labeled else set())
    if not isinstance(data, dict) or set(data) != fields:
        raise ValueError('input fields must be exactly ' + ', '.join(sorted(fields)))
    _names(data['class_names'])
    rows, width = data['probabilities'], len(data['class_names'])
    if not isinstance(rows, list) or not rows:
        raise ValueError('probabilities must be a nonempty list')
    for row in rows:
        if not isinstance(row, list) or len(row) != width or any(
            type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in row
        ) or not math.isclose(math.fsum(row), 1., rel_tol=0, abs_tol=1e-9):
            raise ValueError('rows must match class_names with finite [0,1] probabilities summing to one')
    if labeled:
        ys = data['labels']
        if not isinstance(ys, list) or len(ys) != len(rows) or any(
            type(y) is not int or not 0 <= y < width for y in ys
        ):
            raise ValueError('labels must be aligned integer class indices')
    return rows


def fit(calibration, alpha=.1):
    rows = validate(calibration, True)
    scores = [1 - row[y] for row, y in zip(rows, calibration['labels'])]
    return {'version': 1, 'method': 'split-conformal-1-minus-p',
            'alpha': alpha, 'calibration_n': len(rows),
            'rank': corrected_rank(len(rows), alpha),
            'class_names': list(calibration['class_names']), 'threshold': threshold(scores, alpha)}


def validate_model(model):
    fields = {'version', 'method', 'alpha', 'calibration_n', 'rank', 'class_names', 'threshold'}
    if not isinstance(model, dict) or set(model) != fields:
        raise ValueError('invalid artifact fields')
    if type(model['version']) is not int or model['version'] != 1 or model['method'] != 'split-conformal-1-minus-p':
        raise ValueError('unsupported artifact version or method')
    _names(model['class_names'])
    k = corrected_rank(model['calibration_n'], model['alpha'])
    if type(model['rank']) is not int or model['rank'] != k:
        raise ValueError('artifact rank mismatch')
    q = model['threshold']
    if k > model['calibration_n']:
        if q is not None:
            raise ValueError('all-label boundary requires null threshold')
    elif type(q) not in (int, float) or not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError('threshold must be finite in [0,1]')


def predict(model, data):
    validate_model(model)
    rows = validate(data, False)
    if data['class_names'] != model['class_names']:
        raise ValueError('class_names must match artifact in the same order')
    q = model['threshold']
    return [[j for j, p in enumerate(row) if q is None or 1 - p <= q] for row in rows]


def evaluate(model, test):
    rows = validate(test, True)
    sets = predict(model, {'class_names': test['class_names'], 'probabilities': rows})
    ys = test['labels']; n = len(ys)
    hits = [y in labels for y, labels in zip(ys, sets)]
    sizes = [len(labels) for labels in sets]
    per_class = []
    for k, name in enumerate(test['class_names']):
        indices = [i for i, y in enumerate(ys) if y == k]
        per_class.append({'class_name': name, 'n': len(indices),
                          'coverage': sum(hits[i] for i in indices) / len(indices) if indices else None})
    return {'n': n, 'target_marginal_coverage': 1 - model['alpha'],
            'coverage': sum(hits) / n, 'mean_set_size': sum(sizes) / n,
            'empty_rate': sizes.count(0) / n, 'singleton_rate': sizes.count(1) / n,
            'size_histogram': {str(size): sizes.count(size) for size in sorted(set(sizes))},
            'argmax_accuracy': sum(max(range(len(row)), key=row.__getitem__) == y
                                   for row, y in zip(rows, ys)) / n,
            'per_class': per_class,
            'scope': 'Empirical metrics on these rows, not a guarantee for this fixed split, '
                     'each class/input, shifted data or deployment. Marginal coverage requires '
                     'exchangeable calibration/test scores and a predictor fixed independently of calibration.'}


def rank_audit(scores, alpha):
    """Exact leave-one-out rank check for a finite score multiset, ties included.

    Each position is held out once. This is a deterministic rank diagnostic,
    not an empirical proof that a dataset is exchangeable.
    """
    if len(scores) < 2:
        raise ValueError('rank audit needs at least two scores')
    covered = sum(q is None or value <= q for i, value in enumerate(scores)
                  for q in [threshold(scores[:i] + scores[i + 1:], alpha)])
    k = corrected_rank(len(scores) - 1, alpha)
    return {'positions': len(scores), 'covered': covered, 'coverage': covered / len(scores),
            'rank_lower_bound': k / len(scores)}
