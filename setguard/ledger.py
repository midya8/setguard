"""Bounded transparent witnesses around the unchanged conformal implementation."""
from .core import fit, predict, validate, threshold, corrected_rank, rank_audit


def _group(scores, indices, alpha, name):
    ordered = sorted(indices, key=lambda i: (scores[i], i))
    n = len(ordered)
    rank = corrected_rank(n, alpha) if n else 1
    q = threshold([scores[i] for i in ordered], alpha) if n else None
    rows = [{'position': k + 1, 'row': i, 'score': scores[i],
             'at_threshold': q is not None and scores[i] == q,
             'not_above_threshold': q is None or scores[i] <= q}
            for k, i in enumerate(ordered)]
    deletions = []
    if n >= 2:
        for i in indices:
            remaining = [scores[j] for j in indices if j != i]
            held_q = threshold(remaining, alpha)
            deletions.append({'held_row': i, 'held_score': scores[i],
                              'remaining_n': n - 1,
                              'rank': corrected_rank(n - 1, alpha),
                              'threshold': held_q,
                              'covered': held_q is None or scores[i] <= held_q})
        summary = rank_audit([scores[i] for i in indices], alpha)
    else:
        summary = None
    return {'name': name, 'n': n, 'rank': rank, 'threshold': q,
            'all_label_boundary': q is None,
            'boundary_rows': [r['row'] for r in rows if r['at_threshold']],
            'ordered': rows, 'rank_check': summary, 'leave_one_position_out': deletions}


def audit(calibration, candidates, alpha=.1):
    """Audit both modes, all calibration ranks and all candidate inclusions.

    Candidate data are unlabeled. Deletion checks cycle calibration positions,
    not labeled test examples, and do not prove exchangeability or uncertainty.
    """
    rows = validate(calibration, True)
    queries = validate(candidates, False)
    if calibration['class_names'] != candidates['class_names']:
        raise ValueError('candidate class names/order must match calibration')
    if len(rows) > 256 or len(queries) > 32 or len(calibration['class_names']) > 16:
        raise ValueError('audit supports <=256calibration rows, <=32candidates and <=16classes')
    names = calibration['class_names']
    scores = [1 - row[y] for row, y in zip(rows, calibration['labels'])]
    pooled = _group(scores, list(range(len(rows))), alpha, 'All calibration rows')
    classwise = [_group(scores, [i for i, y in enumerate(calibration['labels']) if y == k],
                        alpha, name) for k, name in enumerate(names)]
    models = {mode: fit(calibration, alpha, mode) for mode in ['pooled', 'classwise']}
    output = []
    for mode, model in models.items():
        qs = [model['threshold']] * len(names) if mode == 'pooled' else model['thresholds']
        sets = predict(model, candidates)
        output.append({'mode': mode, 'artifact': model, 'candidates': [
            {'row': i, 'prediction_set': sets[i], 'labels': [
                {'class_index': k, 'class_name': name, 'probability': row[k],
                 'score': 1 - row[k], 'threshold': qs[k],
                 'included': k in sets[i],
                 'boundary': qs[k] is not None and 1 - row[k] == qs[k],
                 'reason': 'insufficient calibration: always included' if qs[k] is None
                           else 'score <= threshold' if k in sets[i] else 'score > threshold'}
                for k, name in enumerate(names)]} for i, row in enumerate(queries)]})
    return {'alpha': alpha, 'calibration_rows': [
                {'row': i, 'label': y, 'class_name': names[y],
                 'true_probability': rows[i][y], 'score': scores[i]}
                for i, y in enumerate(calibration['labels'])],
            'pooled_group': pooled, 'classwise_groups': classwise, 'modes': output,
            'scope': 'Deterministic score/rank/inclusion witness. No claim of coverage for this '
                     'fixed calibration split, candidate, class rate or under shift. Calibration '
                     'position rank checks do not test exchangeability. Predictor/score must be '
                     'fixed independently of calibration; classwise validity needs exchangeability '
                     'within true label. Null thresholds are infinity sentinels, not missing results.'}
