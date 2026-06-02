# Miles smoke copy of tutorials/rl/000_rl_basics/000_rl_basics.py.
# It builds a Training Gym Modal app with MilesConfig without running the full tutorial.

import re

from modal_training_gym import HuggingFaceDataset, MilesConfig, Qwen3_4B, TrainConfig

_cmudict_cache = {}


def _get_cmudict() -> dict:
    if not _cmudict_cache:
        import nltk
        from nltk.corpus import cmudict

        nltk.download("cmudict", quiet=True)
        _cmudict_cache.update(cmudict.dict())
    return _cmudict_cache


def _count_syllables(text: str) -> int:
    cmu = _get_cmudict()
    total = 0
    for word in re.findall(r"[a-zA-Z]+", text):
        phones = cmu.get(word.lower())
        if phones:
            total += sum(p[-1].isdigit() for p in phones[0])
        else:
            count = len(re.findall(r"[aeiouy]+", word.lower()))
            if word.lower().endswith("e") and count > 1:
                count -= 1
            total += max(count, 1)
    return total


def score_haiku(response: str) -> float:
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
    if len(lines) != 3:
        return -10
    total_diff = sum(
        abs(_count_syllables(line) - target)
        for line, target in zip(lines, [5, 7, 5])
    )
    return -float(total_diff)


class HaikuDataset(HuggingFaceDataset):
    hf_repo = "statworx/haiku"
    input_column = "keywords"
    output_column = "text"
    output_format = "jsonl"
    apply_chat_template = True
    system_prompt = (
        "You are a haiku poet. Write a haiku about the given topic. "
        "Use the 5-7-5 syllable format across three lines."
    )
    prompt_template = "Write a haiku about {input}."
    always_prepare = True


async def haiku_rm(args, sample, **kwargs) -> float:
    return score_haiku(sample.response)


def build_training_config() -> TrainConfig:
    base_model = Qwen3_4B()
    train_dataset = HaikuDataset(n_rows=10)
    return TrainConfig(
        model=base_model,
        dataset=train_dataset,
        recipe=MilesConfig(
            custom_rm_function=haiku_rm,
            gpu_type="H100",
            colocate=True,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            rollout_num_gpus_per_engine=1,
            num_rollout=10,
            rollout_batch_size=16,
            rollout_max_response_len=4096,
            rollout_temperature=1.0,
            save_interval=5,
            apply_chat_template_kwargs={"enable_thinking": False},
            image_overlay=lambda image: image.run_commands(
                "uv pip install --system aiohttp nltk>=3.8.0",
                "python -c \"import nltk; nltk.download('cmudict', quiet=True)\"",
            ),
        ),
    )


training_run = build_training_config()
app = training_run._build_app()


if __name__ == "__main__":
    print(type(training_run.recipe).__name__)
    print(sorted(app.registered_functions))
    print(training_run.recipe.cli_args(dataset=training_run.dataset, model=training_run.model)[:12])
