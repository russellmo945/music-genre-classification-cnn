"""Optional audio-to-image conversion; no audio download or streaming access."""

from pathlib import Path

import numpy as np


def spectrogram(audio_path, output, offset=0., duration=30.):
    import librosa
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if offset < 0 or duration <= 0:
        raise ValueError("Offset must be nonnegative and duration must be positive")
    samples, sr = librosa.load(audio_path, sr=None, mono=True, offset=offset, duration=duration)
    if len(samples) < int(sr * duration):
        raise ValueError("Audio is shorter than the requested interval; choose a shorter duration")
    mel = librosa.feature.melspectrogram(y=samples, sr=sr, n_mels=128, fmax=min(8000, sr / 2))
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # Direct image export has no plot axes or white margins. See docs/data.md for domain differences.
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output, log_mel, origin="lower", cmap="magma", vmin=-80, vmax=0)
    return output
