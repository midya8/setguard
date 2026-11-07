#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m setguard demo
python - <<'PY'
from setguard.demo import run_demo
r = run_demo()
assert r['same_generator']['coverage'] >= .85
assert r['same_generator']['mean_set_size'] < 4
assert r['classwise_artifact']['class_counts'] == [43, 37, 50, 35, 35]
assert r['classwise_same_generator']['mean_set_size'] < 4
assert r['rank_audit']['coverage'] >= r['rank_audit']['rank_lower_bound']
try:
    assert r['adversarial_label_shift']['coverage'] >= .9
except AssertionError:
    print('shifted coverage gate fails as expected')
else:
    raise SystemExit('ERROR: shifted coverage unexpectedly passed')
PY
