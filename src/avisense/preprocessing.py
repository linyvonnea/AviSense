from pathlib import Path
import librosa
import soundfile as sf

from avisense.utils import safe_filename


def load_audio(audio_path, target_sr=22050):
    """
    Load audio as mono and resample.
    """
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    return y, sr


def normalize_audio(y):
    """
    Normalize audio amplitude.
    """
    if len(y) == 0:
        return y

    max_val = max(abs(y))

    if max_val == 0:
        return y

    return y / max_val


def trim_silence(y, top_db=30):
    """
    Trim leading and trailing silence.
    """
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed


def fix_duration(y, sr, duration_seconds):
    """
    Pad or truncate audio to fixed duration.
    """
    target_length = int(sr * duration_seconds)
    y_fixed = librosa.util.fix_length(y, size=target_length)
    return y_fixed


def convert_to_wav(
    input_path,
    output_path,
    target_sr=22050,
    duration_seconds=None,
    trim=True,
    normalize=True,
):
    """
    Convert an audio file to standardized WAV format.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y, sr = load_audio(input_path, target_sr=target_sr)

    if trim:
        y = trim_silence(y)

    if normalize:
        y = normalize_audio(y)

    if duration_seconds is not None:
        y = fix_duration(y, sr, duration_seconds)

    sf.write(output_path, y, sr)

    return output_path


def make_wav_output_path(raw_audio_path, raw_data_dir, output_dir):
    """
    Create WAV output path while preserving species folder name.
    """
    raw_audio_path = Path(raw_audio_path)
    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)

    relative_path = raw_audio_path.relative_to(raw_data_dir)
    species = relative_path.parts[0]

    filename = safe_filename(raw_audio_path.stem) + ".wav"

    return output_dir / species / filename