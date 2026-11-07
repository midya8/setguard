import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch
from setguard.cli import main


class TestCLI(unittest.TestCase):
    def test_fit_then_unlabeled_prediction(self):
        calibration = {'class_names': ['a', 'b'], 'probabilities': [[.9, .1]] * 9, 'labels': [0] * 9}
        out = io.StringIO()
        with patch('sys.stdin', io.StringIO(json.dumps(calibration))), redirect_stdout(out):
            self.assertEqual(main(['fit', '--input', '-', '--alpha', '.1']), 0)
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as artifact:
            artifact.write(out.getvalue()); artifact.flush()
            out = io.StringIO()
            test = {'class_names': ['a', 'b'], 'probabilities': [[.9, .1]]}
            with patch('sys.stdin', io.StringIO(json.dumps(test))), redirect_stdout(out):
                self.assertEqual(main(['predict', '--model', artifact.name, '--input', '-']), 0)
            self.assertEqual(json.loads(out.getvalue())['prediction_sets'], [[0]])

    def test_error_two_and_no_partial_json(self):
        for body in ['{', '{}']:
            out, err = io.StringIO(), io.StringIO()
            with patch('sys.stdin', io.StringIO(body)), redirect_stdout(out), redirect_stderr(err):
                self.assertEqual(main(['fit', '--input', '-']), 2)
            self.assertEqual(out.getvalue(), '')
            self.assertIn('input error', err.getvalue())

    def test_double_stdin_rejected(self):
        err = io.StringIO()
        with redirect_stderr(err):self.assertEqual(main(['predict', '--model', '-', '--input', '-']), 2)
        self.assertIn('both', err.getvalue())
