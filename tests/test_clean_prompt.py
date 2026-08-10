"""Display-time prompt cleaning.

``_clean_prompt`` runs on every rollout sample of every framework, so the
Gemma-specific role headers it strips ("model", "thought") must not eat ordinary
prompt text that happens to be one of those words.
"""

from modal_training_gym.common.training_rollout import _clean_prompt


def test_gemma_role_headers_are_stripped():
    prompt = "<|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n"
    cleaned = _clean_prompt(prompt)

    assert "What is 2+2?" in cleaned
    assert "model" not in cleaned
    assert "user" not in cleaned


def test_gemma_thought_header_is_stripped():
    cleaned = _clean_prompt("<|turn>user\nHi<turn|>\n<|channel>thought\n")
    assert cleaned == "Hi"


def test_plain_prompt_keeps_a_line_saying_model():
    """No Gemma markers, so "model" is content, not a header."""
    prompt = "Classify the noun.\n\nmodel\n\nAnswer:"
    assert _clean_prompt(prompt) == prompt


def test_plain_prompt_keeps_a_line_saying_thought():
    prompt = "Pick the best word:\n\nthought\n\nDone."
    assert _clean_prompt(prompt) == prompt


def test_generic_role_headers_still_go_without_gemma_markers():
    """system/user/assistant stripping predates the Gemma headers."""
    cleaned = _clean_prompt("<|im_start|>user\nHello<|im_end|>")
    assert cleaned == "Hello"


def test_gemma_prompt_still_keeps_model_inside_a_sentence():
    prompt = "<|turn>user\nThe model predicts rain.<turn|>\n<|turn>model\n"
    assert "The model predicts rain." in _clean_prompt(prompt)
