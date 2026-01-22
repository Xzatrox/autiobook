"""audio processing utilities."""

import numpy as np

from .config import SAMPLE_RATE


def concatenate_audio(
    audio_chunks: list[np.ndarray],
    sample_rate: int = SAMPLE_RATE,
    pause_duration_ms: int = 500,
) -> np.ndarray:
    """concatenate audio chunks with pauses between them."""
    if not audio_chunks:
        return np.array([], dtype=np.float32)

    pause_samples = int(sample_rate * pause_duration_ms / 1000)
    pause = np.zeros(pause_samples, dtype=np.float32)

    result = []
    for i, chunk in enumerate(audio_chunks):
        result.append(chunk)
        if i < len(audio_chunks) - 1:
            result.append(pause)

    return np.concatenate(result)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """normalize audio levels to prevent clipping."""
    if audio.size == 0:
        return audio

    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.95

    return audio
