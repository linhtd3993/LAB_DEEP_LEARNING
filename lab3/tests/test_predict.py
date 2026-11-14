import contextlib
import io
import json
import sys
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from predict import main


class PredictTests(unittest.TestCase):
    def setUp(self):
        test_dir = Path(__file__).resolve().parent
        self.root = test_dir / ('.predict-test-' + uuid.uuid4().hex)
        assert self.root.resolve().is_relative_to(test_dir)
        self.root.mkdir()
        self.addCleanup(self.remove_fixture)
        self.model = self.root / 'model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{}', encoding='utf-8')
        (self.root / 'config.json').write_text('{"max_length": 64}', encoding='utf-8')
        self.classifier = Mock(return_value=[{'label': 'POSITIVE', 'score': 0.9}])
        self.pipeline = Mock(return_value=self.classifier)
        self.modules = {
            'torch': SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
            'transformers': SimpleNamespace(pipeline=self.pipeline),
        }

    def remove_fixture(self):
        test_dir = Path(__file__).resolve().parent
        assert self.root.resolve().is_relative_to(test_dir)
        assert self.root.name.startswith('.predict-test-')
        shutil.rmtree(self.root)

    def run_prediction(self, *extra):
        args = ['predict.py', '--model', str(self.model), '--text', 'Great movie!', *extra]
        with patch.object(sys, 'argv', args), patch.dict(sys.modules, self.modules):
            with contextlib.redirect_stdout(io.StringIO()):
                main()

    def test_saved_length_cpu_and_json_output(self):
        output = self.root / 'results' / 'predictions.json'
        self.run_prediction('--cpu', '--output', str(output))
        self.assertEqual(self.pipeline.call_args.kwargs['device'], -1)
        self.assertEqual(self.classifier.call_args.kwargs['max_length'], 64)
        self.assertEqual(json.loads(output.read_text(encoding='utf-8')), [
            {'text': 'Great movie!', 'label': 'POSITIVE', 'score': 0.9}
        ])

    def test_explicit_length_overrides_saved_config(self):
        self.run_prediction('--max-length', '32')
        self.assertEqual(self.classifier.call_args.kwargs['max_length'], 32)
        self.assertEqual(self.pipeline.call_args.kwargs['device'], 0)

    def test_invalid_length_rejected_before_loading_model(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                self.run_prediction('--max-length', '0')
        self.assertEqual(error.exception.code, 2)
        self.pipeline.assert_not_called()


if __name__ == '__main__':
    unittest.main()
