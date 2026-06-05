"""Real audio + reference transcripts for the ASR-RL infra (LibriSpeech dummy).

A tiny, standard ASR set so the WER reward is meaningful (unlike synthetic
tones). Provides both:
  - clips on disk + base64 (for serve+eval, where audio rides in the chat body)
  - an `AudioASRDataset` (MultimodalDataset) for the slime training rollout

For wispr's real data, swap `load_clips()` for their {audio, text} manifest.
"""

from __future__ import annotations

import base64
from pathlib import Path

from modal_training_gym import MultimodalDataset

# <audio> placeholder is where slime injects the media (same convention as <image>
# for VLMs). One placeholder per audio item in the row's audios column.
INSTRUCTION = "<audio>\nTranscribe the speech to text. Respond with only the transcript."
_HF_REPO = "hf-internal-testing/librispeech_asr_dummy"


def load_clips(out_dir: str | Path, n: int = 8) -> list[dict]:
    """Write n LibriSpeech clips to wav; return [{audio_path, label}]."""
    import io

    import soundfile as sf
    from datasets import Audio, load_dataset

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(_HF_REPO, "clean", split="validation")
    ds = ds.select(range(min(n, len(ds))))
    # decode=False avoids the torchcodec dependency; we decode bytes with soundfile.
    ds = ds.cast_column("audio", Audio(decode=False))

    clips = []
    for i, ex in enumerate(ds):
        audio = ex["audio"]
        data = audio["bytes"] if audio.get("bytes") else Path(audio["path"]).read_bytes()
        arr, sr = sf.read(io.BytesIO(data))
        path = out / f"clip_{i:03d}.wav"
        sf.write(str(path), arr, sr, format="WAV")
        clips.append({"audio_path": str(path), "label": ex["text"].lower().strip()})
    return clips


def b64_wav(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def audio_chat_messages(audio_b64: str, instruction: str = INSTRUCTION) -> list[dict]:
    """OpenAI-style multimodal messages for an audio chat-completions request."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
            ],
        }
    ]


class AudioASRDataset(MultimodalDataset):
    """Training dataset: prompt + audio column + transcript label.

    `media` items are audio references the rollout engine resolves. Inline
    base64 keeps the demo self-contained; for scale, switch to volume paths/URLs.
    """

    modality = "audio"

    @classmethod
    def from_clips(cls, clips: list[dict]) -> "AudioASRDataset":
        rows = [
            {
                "prompt": INSTRUCTION,
                "media": [f"data:audio/wav;base64,{b64_wav(c['audio_path'])}"],
                "label": c["label"],
            }
            for c in clips
        ]
        return cls(rows=rows, modality="audio")


class LibriSpeechASRDataset(MultimodalDataset):
    """LibriSpeech ASR dataset whose audio is base64'd inside ``prepare()``.

    Audio bytes don't live on the in-memory object (would blow Modal's 64 KiB
    function-serialization limit) — only the HF coordinates do. prepare()
    runs in a Modal container that downloads + base64-encodes the audio and
    writes a self-contained jsonl to the data volume.
    """

    modality = "audio"
    hf_repo: str = "hf-internal-testing/librispeech_asr_dummy"
    hf_config: str = "clean"
    hf_split: str = "validation"
    n_rows: int = 8
    # Re-materialize each run so prompt changes (e.g. <audio> placeholder) take effect
    # instead of being shadowed by a stale jsonl on the data volume.
    always_prepare: bool = True
    # Keep sample.prompt a conversation list (don't collapse to a templated string)
    # so the audio data-URI survives in the message content for transcription_rollout.
    # (slime's process_vision_info drops audio from multimodal_inputs.)
    apply_chat_template: bool = False

    def __init__(self, **kwargs):
        super().__init__(rows=[], **kwargs)

    def _build_rows(self) -> list[dict]:
        import base64 as b64
        import io as _io
        import soundfile as _sf
        from datasets import Audio as _Audio, load_dataset as _ld

        ds = _ld(self.hf_repo, self.hf_config, split=self.hf_split)
        ds = ds.select(range(min(self.n_rows, len(ds))))
        ds = ds.cast_column("audio", _Audio(decode=False))
        rows = []
        for ex in ds:
            audio = ex["audio"]
            data = audio["bytes"] if audio.get("bytes") else open(audio["path"], "rb").read()
            arr, sr = _sf.read(_io.BytesIO(data))
            buf = _io.BytesIO()
            _sf.write(buf, arr, sr, format="WAV")
            data_uri = "data:audio/wav;base64," + b64.b64encode(buf.getvalue()).decode("ascii")
            rows.append({
                self.input_key: INSTRUCTION,
                self.media_column: [data_uri],
                self.label_key: ex["text"].lower().strip(),
            })
        return rows

    def load(self) -> list[dict]:
        return self._build_rows()

    def prepare(self, path, eval_paths=None):
        rows = self._build_rows()
        self._write_jsonl(rows, path)
        if eval_paths:
            for ep in eval_paths.values():
                self._write_jsonl(rows, ep)
