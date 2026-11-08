#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
python -m pip wheel --no-deps "$root" --wheel-dir "$tmp/wheels" >/dev/null
python -m venv "$tmp/venv"
"$tmp/venv/bin/python" -m pip install --no-index "$tmp"/wheels/*.whl >/dev/null
cd "$tmp"
"$tmp/venv/bin/python" - <<'PY'
import importlib,json,pathlib
for name in ['core','demo','cli','ledger','ledger_fixtures','ledger_export']:importlib.import_module('setguard.'+name)
from setguard.ledger_fixtures import bundle,witnesses
from setguard.ledger_export import project,verify
import setguard
root=pathlib.Path(setguard.__file__).parent/'artifacts'
assert bundle()==json.loads((root/'rank-meta.json').read_text())
a=[r for p in sorted(root.glob('rank-[0-9]*.json')) for r in json.loads(p.read_text())]
b=[{'case':w['case'],'alpha':w['alpha'],'witness':project(w['audit'])}for w in witnesses()]
key=lambda w:(w['case'],w['alpha'])
assert len(a)==95
verify(sorted(a,key=key),sorted(b,key=key))
c=bundle()['cases'][1]
pathlib.Path('audit-input.json').write_text(json.dumps({'calibration':c['calibration'],'candidates':c['candidates']}))
print('6 modules import, original report and95rank witnesses reproduce outside checkout')
PY
"$tmp/venv/bin/python" -m setguard demo > demo.json
"$tmp/venv/bin/python" -m setguard rank-demo > rank-demo.json
"$tmp/venv/bin/python" -m setguard fit --input "$root/examples/calibration.json" > model.json
"$tmp/venv/bin/python" -m setguard predict --model model.json --input "$root/examples/unlabeled.json" > predict.json
"$tmp/venv/bin/python" -m setguard evaluate --model model.json --input "$root/examples/calibration.json" > evaluate.json
"$tmp/venv/bin/python" -m setguard audit --input audit-input.json --alpha .5 > audit.json
"$tmp/venv/bin/python" - <<'PY'
import json
assert json.load(open('demo.json'))==json.load(open('rank-demo.json'))['original_demo']
assert json.load(open('predict.json'))['prediction_sets']
assert json.load(open('evaluate.json'))['n']>0
assert json.load(open('audit.json'))['pooled_group']['rank']==4
print('All6commands execute; audit and prediction outputs checked')
PY
