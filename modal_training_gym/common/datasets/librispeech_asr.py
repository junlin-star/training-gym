"""LibriSpeech ASR dataset (the canonical audio demo set).

A tiny, standard ASR set so a −WER reward is meaningful. Audio bytes are NOT held
on the in-memory object (that would blow Modal's function-serialization limit) —
only the HF coordinates are. ``prepare()`` runs in a Modal container that
downloads + base64-encodes the audio and writes a self-contained jsonl to the
data volume.
"""

from __future__ import annotations

from modal_training_gym.common.dataset import MultimodalDataset

# The <audio> placeholder is where slime injects the media (same convention as
# <image> for VLMs): one placeholder per audio item in the row's media column.
INSTRUCTION = "<audio>\nTranscribe the speech to text. Respond with only the transcript."


class LibriSpeechASRDataset(MultimodalDataset):
    """LibriSpeech ASR rows (prompt + audio data-URI + transcript label)."""

    modality = "audio"
    hf_repo: str = "hf-internal-testing/librispeech_asr_dummy"
    hf_config: str = "clean"
    hf_split: str = "validation"
    n_rows: int = 8
    # Re-materialize each run so prompt changes (e.g. the <audio> placeholder) take
    # effect instead of being shadowed by a stale jsonl on the data volume.
    always_prepare: bool = True
    # Keep sample.prompt a conversation list (don't collapse to a templated string)
    # so the audio data-URI survives in the message content for the transcription
    # rollout. (slime's process_vision_info drops audio from multimodal_inputs.)
    apply_chat_template: bool = False

    def __init__(self, **kwargs):
        super().__init__(rows=[], **kwargs)

    def _build_rows(self) -> list[dict]:
        import base64 as b64
        import io as _io

        import soundfile as _sf
        from datasets import Audio as _Audio
        from datasets import load_dataset as _ld

        ds = _ld(self.hf_repo, self.hf_config, split=self.hf_split)
        ds = ds.select(range(min(self.n_rows, len(ds))))
        # decode=False avoids the torchcodec dependency; decode bytes with soundfile.
        ds = ds.cast_column("audio", _Audio(decode=False))
        rows = []
        for ex in ds:
            audio = ex["audio"]
            data = audio["bytes"] if audio.get("bytes") else open(audio["path"], "rb").read()
            arr, sr = _sf.read(_io.BytesIO(data))
            buf = _io.BytesIO()
            _sf.write(buf, arr, sr, format="WAV")
            data_uri = "data:audio/wav;base64," + b64.b64encode(buf.getvalue()).decode("ascii")
            rows.append(
                {
                    self.input_key: INSTRUCTION,
                    self.media_column: [data_uri],
                    self.label_key: ex["text"].lower().strip(),
                }
            )
        return rows

    def load(self) -> list[dict]:
        return self._build_rows()

    def prepare(self, path, eval_paths=None):
        rows = self._build_rows()
        self._write_jsonl(rows, path)
        if eval_paths:
            for ep in eval_paths.values():
                self._write_jsonl(rows, ep)
