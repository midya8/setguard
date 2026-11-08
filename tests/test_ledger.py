import copy
import itertools
import json
import math
import random
import subprocess
import sys
import unittest
from fractions import Fraction
from pathlib import Path
from setguard.core import fit, predict, rank_audit
from setguard.ledger import audit
from setguard.ledger_fixtures import cases, bundle, witnesses
from setguard.ledger_export import project, verify


class LedgerTests(unittest.TestCase):
    def test_original_models_and_inputs(self):
        for c in cases():
            before = copy.deepcopy(c)
            a = audit(c['calibration'], c['candidates'])
            for m in a['modes']:
                model = fit(c['calibration'], .1, m['mode'])
                self.assertEqual(m['artifact'], model)
                self.assertEqual([r['prediction_set'] for r in m['candidates']], predict(model, c['candidates']))
            self.assertEqual(c, before)

    def test_independent_fraction_rank160(self):
        rng = random.Random(351)
        for _ in range(160):
            n = rng.randrange(1, 31); alpha = rng.randrange(1, 100)/100
            ps = [rng.randrange(1, 100)/100 for i in range(n)]
            c = {'class_names': ['a','b'], 'probabilities': [[p,1-p] for p in ps], 'labels': [0]*n}
            q = {'class_names': ['a','b'], 'probabilities': [[.5,.5]]}
            a = audit(c,q,alpha);g = a['pooled_group']
            exact = Fraction(n+1)*(1-Fraction(str(alpha)))
            k = -(-exact.numerator//exact.denominator)
            self.assertEqual(g['rank'],k)
            ordered = sorted((1-p,i) for i,p in enumerate(ps))
            expected = ordered[k-1][0] if k<=n else None
            self.assertEqual(g['threshold'],expected)
            self.assertEqual([r['row'] for r in g['ordered']],[i for p,i in ordered])
            for r in g['leave_one_position_out']:
                scores = sorted(1-p for i,p in enumerate(ps) if i!=r['held_row'])
                k2f = Fraction(n)*(1-Fraction(str(alpha)));k2 = -(-k2f.numerator//k2f.denominator)
                held = scores[k2-1] if k2<=n-1 else None
                self.assertEqual(r['threshold'],held)
                self.assertEqual(r['covered'],held is None or r['held_score']<=held)

    def test_decimal_boundary49rows(self):
        c=cases()[1];c['calibration']={'class_names':['a','b'],'probabilities':[[.5,.5]]*49,'labels':[0]*49}
        c['candidates']={'class_names':['a','b'],'probabilities':[[.5,.5]]}
        self.assertEqual(audit(c['calibration'],c['candidates'],.58)['pooled_group']['rank'],21)

    def test_tie_boundary_inclusive_not_just_rankprefix(self):
        c=cases()[1];a=audit(c['calibration'],c['candidates'],.5)
        self.assertEqual(len(a['pooled_group']['boundary_rows']),6)
        self.assertTrue(all(r['not_above_threshold'] for r in a['pooled_group']['ordered']))
        self.assertEqual(a['pooled_group']['rank_check']['coverage'],1)
        self.assertEqual(a['modes'][0]['candidates'][0]['prediction_set'],[0])

    def test_absent_one_and_empty(self):
        cs={c['id']:c for c in cases()}
        c=cs['absent'];a=audit(c['calibration'],c['candidates'],.1)
        self.assertEqual(a['classwise_groups'][2]['n'],0)
        self.assertIsNone(a['classwise_groups'][2]['threshold'])
        self.assertTrue(all(2 in r['prediction_set'] for r in a['modes'][1]['candidates']))
        c=cs['one'];a=audit(c['calibration'],c['candidates'],.1)
        self.assertIsNone(a['pooled_group']['rank_check'])
        self.assertTrue(all(r['prediction_set']==[0,1,2] for r in a['modes'][0]['candidates']))
        c=cs['empty'];a=audit(c['calibration'],c['candidates'],.1)
        self.assertEqual(a['modes'][0]['candidates'][0]['prediction_set'],[])

    def test_delete_position_check_agrees_existing(self):
        for c in cases():
            a=audit(c['calibration'],c['candidates'],.35)
            for g in [a['pooled_group']]+a['classwise_groups']:
                if g['n']>=2:
                    s=[r['score'] for r in g['ordered']]
                    self.assertEqual(g['rank_check'],rank_audit(s,.35))
                    self.assertEqual(sum(r['covered'] for r in g['leave_one_position_out']),g['rank_check']['covered'])

    def test_reject_bounds_and_mismatch(self):
        c=cases()[1]
        for alpha in [0,1,True,math.nan,math.inf,'0.1']:
            with self.assertRaises((ValueError,TypeError)):audit(c['calibration'],c['candidates'],alpha)
        bad=copy.deepcopy(c['candidates']);bad['class_names']=['b','a','c']
        with self.assertRaises(ValueError):audit(c['calibration'],bad)
        bad=copy.deepcopy(c['candidates']);bad['probabilities']*=9
        with self.assertRaises(ValueError):audit(c['calibration'],bad)

    def test95_archive_reproduces(self):
        root=Path(__file__).resolve().parents[1]/'setguard/artifacts'
        actual=[r for p in sorted(root.glob('rank-[0-9]*.json')) for r in json.loads(p.read_text())]
        expected=[{'case':w['case'],'alpha':w['alpha'],'witness':project(w['audit'])} for w in witnesses()]
        key=lambda x:(x['case'],x['alpha'])
        self.assertEqual(len(actual),95)
        verify(sorted(actual,key=key),sorted(expected,key=key))

    def test_cli_audit_rankdemo_and_invalid(self):
        c=cases()[1];payload=json.dumps({'calibration':c['calibration'],'candidates':c['candidates']})
        p=subprocess.run([sys.executable,'-m','setguard','audit','--input','-','--alpha','.5'],input=payload,text=True,capture_output=True)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(json.loads(p.stdout),audit(c['calibration'],c['candidates'],.5))
        p=subprocess.run([sys.executable,'-m','setguard','rank-demo'],text=True,capture_output=True)
        self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout),bundle())
        for bad in ['{"calibration":1,"calibration":2,"candidates":3}', '{"calibration":1}']:
            p=subprocess.run([sys.executable,'-m','setguard','audit','--input','-'],input=bad,text=True,capture_output=True)
            self.assertEqual(p.returncode,2);self.assertEqual(p.stdout,'')

    def test_archive_structure_types_and_nonfinite(self):
        expected={'rank':1,'values':[.5,None,True]}
        verify(copy.deepcopy(expected),expected)
        for bad in [{'rank':True,'values':[.5,None,True]},
                    {'rank':1,'values':[.5,None]},
                    {'rank':1,'values':[math.nan,None,True]},
                    {'rank':1,'values':[.5,None,1]},
                    {'rank':1,'values':[.5,None,True],'extra':0}]:
            with self.assertRaises(ValueError):verify(bad,expected)


if __name__=='__main__':unittest.main()
