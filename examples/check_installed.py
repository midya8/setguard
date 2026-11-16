"""Run every test and compare complete CLI reports from an installed wheel."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]
ENV = os.environ.copy()
ENV.pop('PYTHONPATH', None)
MODULES = ['setguard', 'setguard.__main__', 'setguard.cli', 'setguard.core',
           'setguard.demo', 'setguard.ledger', 'setguard.ledger_export',
           'setguard.ledger_fixtures']


def call(args, cwd, env=ENV):
    return subprocess.check_output(args, cwd=cwd, env=env, text=True)


with tempfile.TemporaryDirectory() as directory:
    work = Path(directory)
    wheels = work / 'wheels'
    wheels.mkdir()
    call([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '-w', str(wheels),
          str(ROOT)], work)
    built = list(wheels.glob('*.whl'))
    assert len(built) == 1, built
    venv.create(work / 'env', with_pip=True)
    py = str(work / 'env/bin/python')
    call([py, '-m', 'pip', 'install', '--no-deps', str(built[0])], work)
    outside = work / 'outside'
    outside.mkdir()
    for name in ['tests', 'examples']:
        shutil.copytree(ROOT / name, outside / name)
    # This directory supplies archive data only, never importable module source.
    shutil.copytree(ROOT / 'setguard/artifacts', outside / 'setguard/artifacts')
    suite = '''
import importlib,importlib.util,json,pathlib,sys,unittest
outside=pathlib.Path.cwd().resolve()
sys.path.insert(0,str(outside))
names=json.loads(sys.argv[1])
source=pathlib.Path(sys.argv[2]).resolve()
prefix=pathlib.Path(sys.prefix).resolve()
for name in names:
    if name != 'setguard.__main__':importlib.import_module(name)
def check():
    for name in names:
        origin=pathlib.Path(importlib.util.find_spec(name).origin).resolve()
        assert origin.is_relative_to(prefix),(name,str(origin))
        assert not origin.is_relative_to(source),(name,str(origin))
        assert 'site-packages' in origin.parts,(name,str(origin))
check()
tests=unittest.defaultTestLoader.discover(str(outside/'tests'))
result=unittest.TextTestRunner(verbosity=2).run(tests)
assert result.testsRun>0 and result.wasSuccessful()
check()
print(json.dumps(dict(tests=result.testsRun,installed_origins=len(names),
                     checked_before_and_after=True)))
'''
    print(call([py, '-I', '-c', suite, json.dumps(MODULES), str(ROOT)], outside))
    reference = work / 'reference'
    reference.mkdir()
    shutil.copytree(ROOT / 'examples', reference / 'examples')
    source_env = ENV.copy()
    source_env['PYTHONPATH'] = str(ROOT)
    source = [sys.executable, '-m', 'setguard']
    installed = [py, '-I', '-m', 'setguard']
    reports = 0

    def compare(args):
        global reports
        expected = json.loads(call(source + args, reference, source_env))
        text = call(installed + args, outside)
        assert json.loads(text) == expected, args
        reports += 1
        return text

    compare(['demo'])
    bundle = json.loads(compare(['rank-demo']))
    control = bundle['cases'][1]
    audit_input = json.dumps({key: control[key] for key in ['calibration', 'candidates']})
    for folder in [reference, outside]:
        (folder / 'audit-input.json').write_text(audit_input)
    compare(['audit', '--input', 'audit-input.json', '--alpha', '.5'])
    for mode in ['pooled', 'classwise']:
        model = compare(['fit', '--input', 'examples/calibration.json',
                         '--alpha', '.2', '--mode', mode])
        for folder in [reference, outside]:
            (folder / 'model.json').write_text(model)
        compare(['predict', '--model', 'model.json', '--input', 'examples/unlabeled.json'])
        compare(['evaluate', '--model', 'model.json', '--input', 'examples/calibration.json'])
    print(json.dumps(dict(complete_reports=reports, command_kinds=6,
                          modes=['pooled', 'classwise'], entrypoint='python -m setguard')))
