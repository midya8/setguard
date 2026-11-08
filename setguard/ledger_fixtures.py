"""Fixed public synthetic cases. No new training or benchmark claim."""
from .demo import sample, run_demo
from .ledger import audit


def _data(names, rows, labels=None):
    value = {'class_names': names, 'probabilities': rows}
    if labels is not None:
        value['labels'] = labels
    return value


def cases():
    names = ['a', 'b', 'c']
    original = sample(200, 11)
    query = sample(6, 29)
    return [
        {'id': 'original', 'name': 'Original 200-row calibration',
         'description': 'Original fixed overconfident predictor and seeded calibration retained; six original same-generator candidate rows. Selection is illustrative, not a new test sample or benchmark.',
         'calibration': original, 'candidates': _data(query['class_names'], query['probabilities'])},
        {'id': 'ties', 'name': 'Ties at the threshold',
         'description': 'Repeated scores expose deterministic rank order versus inclusive <= membership. Ties can make the finite rank check conservative.',
         'calibration': _data(names, [[.5,.25,.25],[.25,.5,.25],[.25,.25,.5],
                                       [.5,.25,.25],[.25,.5,.25],[.25,.25,.5]], [0,1,2,0,1,2]),
         'candidates': _data(names, [[.5,.25,.25],[.25,.5,.25],[.25,.25,.5],[1/3]*3])},
        {'id': 'absent', 'name': 'Rare and absent classes',
         'description': 'Class c is absent and class b has one calibration row. Classwise thresholds can be infinite and force inclusion. There is no pooled fallback.',
         'calibration': _data(names, [[.9,.05,.05],[.8,.1,.1],[.7,.2,.1],[.6,.2,.2],[.1,.8,.1]], [0,0,0,0,1]),
         'candidates': _data(names, [[.9,.09,.01],[.01,.98,.01],[.1,.1,.8],[1/3]*3])},
        {'id': 'empty', 'name': 'Empty sets are possible',
         'description': 'Low calibration scores and uniform candidate probabilities can produce empty pooled sets. This finite example is not input-conditional coverage.',
         'calibration': _data(names, [[.99,.005,.005],[.01,.98,.01],[.015,.015,.97]]*3, [0,1,2]*3),
         'candidates': _data(names, [[1/3]*3,[.99,.005,.005],[.01,.98,.01],[.015,.015,.97]])},
        {'id': 'one', 'name': 'One calibration row',
         'description': 'The corrected rank can exceed calibration size. Null threshold means infinity and all classes included, not a missing computation. A leave-one-position check needs at least two positions.',
         'calibration': _data(names, [[.8,.1,.1]], [0]),
         'candidates': _data(names, [[.01,.98,.01],[1/3]*3])},
    ]


def bundle():
    return {'cases': cases(), 'original_demo': run_demo()}


def witnesses():
    return [{'case': c['id'], 'alpha': i/20,
             'audit': audit(c['calibration'], c['candidates'], i/20)}
            for c in cases() for i in range(1,20)]
