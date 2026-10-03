import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from music_genre.data import GENRES, build_manifest, validate_manifest, verify_images, image_slices, SpectrogramSlices
from music_genre.metrics import evaluate
from music_genre.model import GenreCNN


def make_images(root):
    rng = np.random.default_rng(123)
    for genre in GENRES:
        (root / genre).mkdir(parents=True)
        for i in range(5):
            pixels = rng.integers(0, 220, (40, 100, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(root / genre / f'{genre}{i:05d}.png')


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        make_images(self.root)
        self.manifest = build_manifest(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_split_is_deterministic_and_track_disjoint(self):
        self.assertEqual(self.manifest, build_manifest(self.root))
        groups = {s: {r['track_id'] for r in self.manifest['records'] if r['split'] == s}
                  for s in ['train', 'validation', 'test']}
        self.assertEqual([len(groups[s]) for s in groups], [30, 10, 10])
        self.assertFalse(groups['train'] & groups['test'])
        self.assertFalse(groups['train'] & groups['validation'])
        self.assertFalse(groups['validation'] & groups['test'])
        self.assertNotEqual(self.manifest['records'], build_manifest(self.root, 9)['records'])

    def test_duplicate_content_rejected(self):
        a, b = list((self.root / 'blues').glob('*.png'))[:2]
        b.write_bytes(a.read_bytes())
        with self.assertRaisesRegex(ValueError, 'Duplicate image'):
            build_manifest(self.root)

    def test_duplicate_track_rejected(self):
        broken = copy.deepcopy(self.manifest)
        broken['records'].append(broken['records'][0])
        with self.assertRaisesRegex(ValueError, 'Duplicate source track'):
            validate_manifest(broken)

    def test_changed_image_rejected(self):
        verify_images(self.root, self.manifest)
        (self.root / self.manifest['records'][0]['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Missing or changed'):
            verify_images(self.root, self.manifest)

    def test_inference_and_training_preprocessing_match(self):
        record = self.manifest['records'][0]
        dataset = SpectrogramSlices(self.root, [record], slices=4, size=64)
        with Image.open(self.root / record['path']) as image:
            batched = image_slices(image, slices=4, size=64)
        for i in range(4):
            torch.testing.assert_close(dataset[i][0], batched[i])
            self.assertEqual(dataset[i][2], record['track_id'])
        with self.assertRaisesRegex(ValueError, 'all-white'):
            image_slices(Image.new('RGB', (100, 40), 'white'))

    def test_selected_model_forward_and_backward(self):
        model = GenreCNN()
        logits = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(logits.shape), (2, 10))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 9])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_evaluation_is_deterministic_and_restores_training_mode(self):
        dataset = SpectrogramSlices(self.root, self.manifest['records'][:2], slices=2)
        loader = DataLoader(dataset, batch_size=2)
        model = GenreCNN()
        model.train()
        first = evaluate(model, loader, list(GENRES))
        second = evaluate(model, loader, list(GENRES))
        self.assertEqual(first, second)
        self.assertTrue(model.training)
        self.assertEqual(first['slice']['count'], 4)
        self.assertEqual(first['track']['count'], 2)

    def test_track_aggregation_uses_mean_scores(self):
        class Identity(torch.nn.Module):
            def forward(self, x): return x
        loader = [(torch.tensor([[3., 0.], [0., 2.], [0., 4.]]),
                   torch.tensor([0, 0, 1]), ['a', 'a', 'b'])]
        result = evaluate(Identity(), loader, ['one', 'two'])
        self.assertAlmostEqual(result['slice']['accuracy'], 2/3)
        self.assertEqual(result['track']['accuracy'], 1.)


if __name__ == '__main__':
    unittest.main()
