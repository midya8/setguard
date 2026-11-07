"""Selected development fixture, not an untouched model benchmark."""
import random

from .core import fit, evaluate, rank_audit


def sample(n, seed, shifted=False):
    rng = random.Random(seed)
    rows, ys = [], []
    for _ in range(n):
        raw = [rng.gammavariate(.7, 1) for _ in range(5)]
        total = sum(raw); true = [p / total for p in raw]
        u = rng.random(); cumulative = 0; y = 4
        for j, p in enumerate(true):
            cumulative += p
            if u < cumulative:
                y = j; break
        # Fixed overconfident predictor, independent of calibration labels.
        distorted = [p ** 3 for p in true]; scale = sum(distorted)
        rows.append([p / scale for p in distorted])
        ys.append(min(range(5), key=true.__getitem__) if shifted else y)
    return {'class_names': ['a', 'b', 'c', 'd', 'e'], 'probabilities': rows, 'labels': ys}


def run_demo():
    calibration = sample(200, 11)
    model = fit(calibration, alpha=.1)
    classwise = fit(calibration, alpha=.1, mode='classwise')
    return {'artifact': model,
            'classwise_artifact': classwise,
            'classwise_same_generator': evaluate(classwise, sample(2000, 29)),
            'same_generator': evaluate(model, sample(2000, 29)),
            'adversarial_label_shift': evaluate(model, sample(2000, 29, shifted=True)),
            'rank_audit': rank_audit([i / 20 for i in range(21)], .1),
            'fixture_scope': 'Development demonstration: independent seeded samples from one '
                             'hand-coded generator; no model training or real benchmark. Shift arm '
                             'deliberately assigns the least-probable class; not a realistic drift estimate.'}
