"""JSON stdout; errors go to stderr with exit 2."""
import argparse
import json
import sys
from .core import fit, predict, evaluate
from .demo import run_demo
from .ledger import audit
from .ledger_fixtures import bundle


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: '+key)
        result[key] = value
    return result


def load(path, strict=False):
    if path == '-':
        return json.load(sys.stdin, object_pairs_hook=_unique if strict else dict)
    with open(path, encoding='utf-8') as source:
        return json.load(source, object_pairs_hook=_unique if strict else dict)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='setguard')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('demo')
    sub.add_parser('rank-demo')
    a = sub.add_parser('audit')
    a.add_argument('--input', required=True)
    a.add_argument('--alpha', type=float, default=.1)
    f = sub.add_parser('fit')
    f.add_argument('--input', required=True)
    f.add_argument('--alpha', type=float, default=.1)
    f.add_argument('--mode', choices=['pooled', 'classwise'], default='pooled')
    for name in ['predict', 'evaluate']:
        p = sub.add_parser(name)
        p.add_argument('--model', required=True)
        p.add_argument('--input', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'demo':
            result = run_demo()
        elif args.command == 'rank-demo':
            result = bundle()
        elif args.command == 'audit':
            payload = load(args.input, strict=True)
            if not isinstance(payload, dict) or set(payload) != {'calibration', 'candidates'}:
                raise ValueError('audit input requires exactly calibration and candidates')
            result = audit(payload['calibration'], payload['candidates'], args.alpha)
        elif args.command == 'fit':
            result = fit(load(args.input), args.alpha, args.mode)
        else:
            if args.model == '-' and args.input == '-':
                raise ValueError('model and input cannot both use stdin')
            model = load(args.model); data = load(args.input)
            result = {'prediction_sets': predict(model, data)} if args.command == 'predict' else evaluate(model, data)
        body = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    except (ValueError, TypeError, OSError, OverflowError) as exc:
        print('setguard input error: ' + str(exc), file=sys.stderr)
        return 2
    print(body)
    return 0
