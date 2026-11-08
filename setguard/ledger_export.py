"""Portable compact numerical witnesses; no hidden estimator implementation."""
import json
import math
from pathlib import Path
from .ledger_fixtures import bundle, witnesses


def project(a):
    def group(g):
        return [g['n'], g['rank'], g['threshold'], g['all_label_boundary'], g['boundary_rows'],
                [[r['position'], r['row'], r['score'], r['at_threshold'], r['not_above_threshold']]
                 for r in g['ordered']], g['rank_check'],
                [[r['held_row'], r['held_score'], r['remaining_n'], r['rank'], r['threshold'], r['covered']]
                 for r in g['leave_one_position_out']]]
    return {'alpha': a['alpha'],
            'rows': [[r['row'], r['label'], r['true_probability'], r['score']] for r in a['calibration_rows']],
            'pooled': group(a['pooled_group']), 'classwise': [group(g) for g in a['classwise_groups']],
            'modes': [{'artifact': m['artifact'], 'candidates': [
                [r['row'], r['prediction_set'], [[l['class_index'], l['probability'], l['score'],
                     l['threshold'], l['included'], l['boundary']] for l in r['labels']]]
                for r in m['candidates']]} for m in a['modes']]}


def verify(actual, expected, path='root'):
    """Exact structure/nonfloat types; finite float leaves allow1e-12abs/rel."""
    if type(actual) is not type(expected):
        raise ValueError(path+': type mismatch')
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(path+': key mismatch')
        for key in expected:
            verify(actual[key], expected[key], path+'.'+key)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(path+': length mismatch')
        for i, (a, b) in enumerate(zip(actual, expected)):
            verify(a,b,path+'['+str(i)+']')
    elif isinstance(expected, float):
        if not math.isfinite(actual) or not math.isfinite(expected) or not math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError(path+': nonfinite or different float')
    elif actual != expected:
        raise ValueError(path+': value mismatch')


def write(root=None):
    root = Path(root) if root else Path(__file__).parent.parent
    artifacts = root / 'setguard' / 'artifacts'
    artifacts.mkdir(parents=True, exist_ok=True)
    data = bundle()
    (artifacts / 'rank-meta.json').write_text(json.dumps(data, separators=(',', ':'), allow_nan=False)+'\n')
    (root / 'docs').mkdir(exist_ok=True)
    (root / 'docs' / 'data.js').write_text('globalThis.SET_DATA='+json.dumps(data, separators=(',', ':'), allow_nan=False)+';\n')
    ws = witnesses()
    for i in range(19):
        chunk = [{'case': w['case'], 'alpha': w['alpha'], 'witness': project(w['audit'])}
                 for w in ws if w['alpha'] == (i+1)/20]
        (artifacts / ('rank-%02d.json' % i)).write_text(json.dumps(chunk, separators=(',', ':'), allow_nan=False)+'\n')
    return len(ws)


if __name__ == '__main__':
    print(write(), 'rank witnesses exported')
