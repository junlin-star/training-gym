"""Mini audio GRPO on Qwen3-ASR-1.7B via the gym.

Drives Qwen/Qwen3-ASR-1.7B end-to-end through the gym's TrainConfig(...).train()
API: LibriSpeech audio (the gym's multimodal passthrough) -> slime rollout served
by SGLang on /v1/audio/transcriptions -> -WER reward -> GRPO. Runs on upstream
training-gym `main` (native sglang 0.5.12 + megatron-bridge 0.5.0); the few
upstream gaps it works around are documented in _native_qwen3asr_compat.py.

  uv run python train_qwen3_asr.py

Everything is kept small (few clips, few rollouts, single 2×H100 node) so the
-WER reward signal is quick and bounded.
"""

from __future__ import annotations

from pathlib import Path

from modal_training_gym import SlimeRecipe, TrainConfig
from modal_training_gym.common.wandb import WandbConfig

from asr_rollout import transcription_rollout
from audio_data import LibriSpeechASRDataset
from qwen3_asr_model import Qwen3ASR
from reward import wer_reward

N_CLIPS = 8


def build_train_config(
    *,
    n_clips: int = N_CLIPS,
    actor_num_gpus_per_node: int = 2,
    num_rollout: int = 8,
    rollout_batch_size: int = 4,
    global_batch_size: int = 8,
    micro_batch_size: int = 1,
    exp_name: str = "qwen3-asr-grpo-audio-long-cosine",
) -> TrainConfig:
    dataset = LibriSpeechASRDataset(n_rows=n_clips)

    return TrainConfig(
        model=Qwen3ASR(),
        dataset=dataset,  # multimodal_keys={"audio":"audios"} -> --multimodal-keys
        recipe=SlimeRecipe(
            custom_rm_function=wer_reward,
            # Stream reward (−WER) + pg_loss/grad_norm to W&B for a live curve in
            # the demo (uses the existing `wandb-secret` Modal secret).
            wandb=WandbConfig(
                project="qwen3-asr-rl",
                group="gym-demo",
                exp_name=exp_name,
            ),
            # Qwen3-ASR is served on /v1/audio/transcriptions (never chat
            # completions), so always drive it through our transcription rollout.
            custom_generate_function=transcription_rollout,
            gpu_type="H100",
            colocate=True,
            actor_num_nodes=1,
            # TP=1 + colocate: the 1.7B serves on one GPU, so each GPU is a
            # data-parallel rank for the colocated actor + rollout engines. Defaults
            # to 2 for the demo; the scale variant passes 8.
            actor_num_gpus_per_node=actor_num_gpus_per_node,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            rollout_num_gpus_per_engine=1,
            # Audio-conditioned GRPO confirmed working (reward real, gradient steps
            # clean). Qwen3-ASR is near-deterministic on clean LibriSpeech, so to get
            # NONZERO within-group variance (→ advantages ≠ 0 → a moving curve) we
            # use many samples/prompt + higher temperature.
            num_rollout=num_rollout,
            rollout_batch_size=rollout_batch_size,
            n_samples_per_prompt=8,
            global_batch_size=global_batch_size,
            # Cosine LR decay over the (longer) run → LR 1e-6 → ~0, so the reward
            # curve settles instead of jittering at constant LR. Longer run = a
            # richer wandb curve for the demo.
            lr_decay_style="cosine",
            rollout_max_response_len=128,
            rollout_temperature=1.0,
            # Audio conditioning made prompts ~5-10x longer (expanded <audio_pad>)
            # and added the frozen audio tower, so free sglang memory to give the
            # colocated actor headroom (text-only run used 0.78; here 0.45).
            sglang_mem_fraction_static=0.45,
            # The native megatron-bridge Qwen3-ASR forward doesn't implement THD
            # sequence packing (raises "packed_seq_params is not supported"). slime
            # only builds packed_seq_params when qkv_format="thd" (its default), so
            # switch to padded "bshd" batches → packed_seq_params=None → the bridge's
            # raise is skipped. bshd requires dynamic batching OFF + an explicit
            # micro_batch_size. micro_batch_size>1 also averages several sequences per
            # backward, which (without packing) smooths per-sample gradient spikes
            # that can otherwise blow up grad norm at scale — see the scale variant.
            use_dynamic_batch_size=False,
            extra_config={"qkv_format": "bshd", "micro_batch_size": micro_batch_size},
            save_interval=1000,
            # "bridge" loads the HF checkpoint into Megatron at startup AND drives the
            # gym's post-training megatron→HF export. slime's hand-written converter
            # has no qwen3_asr entry (it falls to the qwen2 path and can't map the
            # audio tower), so the compat shim routes the export through the native
            # megatron-bridge's AutoBridge — which maps the full model incl. the audio
            # tower — so the trained model lands as a standard HF checkpoint.
            megatron_to_hf_mode="bridge",
            # The native stack (sglang 0.5.12 + megatron-bridge 0.5.0) supports
            # Qwen3-ASR, so no source overlays / custom bridge / processor shim are
            # needed — just the reward/audio deps + our example modules. Three small
            # upstream gaps remain (bridge config validate-order, slime processor
            # loading, bridge pg_collection); one idempotent compat script works
            # around all three at image build — see _native_qwen3asr_compat.py.
            image_overlay=lambda image: image.uv_pip_install(
                "jiwer", "librosa", "soundfile"
            ).add_local_file(
                str(Path(__file__).parent / "_native_qwen3asr_compat.py"),
                "/tmp/_native_qwen3asr_compat.py",
                copy=True,
            ).run_commands(
                "python /tmp/_native_qwen3asr_compat.py"
            ).add_local_python_source(
                "audio_data", "qwen3_asr_model", "asr_rollout", "reward", copy=True
            ),
        ),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Audio GRPO on Qwen3-ASR-1.7B (native training-gym stack)."
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Quick 2xH100 demo instead of the default full 8xH100 node "
        "(fewer GPUs + a shorter run; same example).",
    )
    args = parser.parse_args()

    config = (
        build_train_config(
            actor_num_gpus_per_node=2,
            num_rollout=8,
            exp_name="qwen3-asr-grpo-audio-demo",
        )
        if args.minimal
        else build_train_config(
            actor_num_gpus_per_node=8,
            num_rollout=50,
            exp_name="qwen3-asr-grpo-8gpu",
        )
    )
    result = config.train()
    print("training_run_id:", result.training_run_id)
