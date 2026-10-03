"""Exercise the installed CLI on disposable synthetic data, never a benchmark."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
from music_genre.data import GENRES


def main():
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    def cli(*args, succeeds=True):
        run = subprocess.run([sys.executable, '-m', 'music_genre.cli', *map(str, args)],
                             cwd=repo, env=env, capture_output=True, text=True)
        if succeeds and run.returncode:
            raise RuntimeError(run.stdout + run.stderr)
        if not succeeds and run.returncode == 0:
            raise AssertionError('Expected rejection of altered experiment provenance')
        return run
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        rng = np.random.default_rng(42)
        rows = []
        for label, genre in enumerate(GENRES):
            (root / 'images' / genre).mkdir(parents=True)
            for i in range(5):
                Image.fromarray(rng.integers(0, 220, (40, 100, 3), dtype=np.uint8)).save(
                    root / 'images' / genre / f'{genre}{i:05d}.png')
                for segment in range(2):
                    rows.append({'filename': f'{genre}.{i:05d}.{segment}.wav', 'label': genre,
                                 'length': 100, 'feature_a': float(rng.random()), 'feature_b': float(label)})
        pd.DataFrame(rows).to_csv(root / 'features.csv', index=False)
        config = json.loads((repo / 'configs/model_set13.json').read_text())
        config.update(epochs=1, batch_size=8, slices=2)
        config['model'].update(conv_layers=2, initial_channels=4, hidden=[16])
        (root / 'config.json').write_text(json.dumps(config))
        cli('prepare', '--images', root/'images', '--out', root/'manifest.json')
        cli('train', '--images', root/'images', '--manifest', root/'manifest.json',
            '--config', root/'config.json', '--out', root/'run')
        cli('evaluate', '--images', root/'images', '--manifest', root/'manifest.json',
            '--checkpoint', root/'run/best.pt', '--out', root/'test.json')
        result = json.loads((root/'test.json').read_text())
        assert result['slice']['count'] == 20 and result['track']['count'] == 10
        prediction = json.loads(cli('predict', '--image', root/'images/blues/blues00000.png',
                                    '--checkpoint', root/'run/best.pt').stdout)
        assert len(prediction['scores']) == 10
        assert abs(sum(prediction['scores'].values()) - 1) < 1e-5
        cli('baseline', '--csv', root/'features.csv', '--manifest', root/'manifest.json',
            '--out', root/'baseline.json', '--trees', 5)
        baseline = json.loads((root/'baseline.json').read_text())
        assert baseline['test']['track']['count'] == 10
        with (root/'manifest.json').open('a') as f: f.write('\n')
        cli('evaluate', '--images', root/'images', '--manifest', root/'manifest.json',
            '--checkpoint', root/'run/best.pt', '--out', root/'invalid.json', succeeds=False)
    print('PASS: CLI prepare, train, checkpoint reload, test evaluation, prediction, baseline, and manifest-tamper rejection.')
    print('Synthetic fixtures only; no GTZAN performance was measured.')


if __name__ == '__main__':
    main()
