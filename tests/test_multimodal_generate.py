"""ModelDeployment.generate multimodal passthrough: when images are supplied the
request sends OpenAI-style structured content (text + image_url parts) so vision
models see the image; text-only requests keep posting a plain prompt string. This
is the read-side path the computer-use (Qwen3-VL) eval_fn relies on.
"""

import requests

from modal_training_gym.common.deployment import DeploymentConfig, ModelDeployment
from modal_training_gym.common.models import Qwen3VL_8B


class _FakeResponse:
    status_code = 200

    def __init__(self, body: dict):
        self._body = body

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._body


def _deployment() -> ModelDeployment:
    cfg = DeploymentConfig(model=Qwen3VL_8B(), served_model_name="vl")
    return ModelDeployment(deployment_id="t", deployment_config=cfg, url="http://test")


def _capture_post(monkeypatch) -> dict:
    captured: dict = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["body"] = json
        return _FakeResponse({"choices": [{"message": {"content": "(0.5, 0.5)"}}]})

    monkeypatch.setattr(requests, "post", fake_post)
    return captured


def test_generate_with_images_sends_structured_content(monkeypatch):
    captured = _capture_post(monkeypatch)
    out = _deployment().generate(
        "click submit",
        ensure_ready=False,
        images=["data:image/png;base64,AAA", "https://x/y.png"],
    )
    assert out == "(0.5, 0.5)"
    content = captured["body"]["messages"][0]["content"]
    assert content == [
        {"type": "text", "text": "click submit"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
    ]


def test_generate_text_only_sends_plain_string(monkeypatch):
    captured = _capture_post(monkeypatch)
    _deployment().generate("hello", ensure_ready=False)
    assert captured["body"]["messages"][0]["content"] == "hello"


def test_generate_empty_images_is_text_only(monkeypatch):
    captured = _capture_post(monkeypatch)
    _deployment().generate("hi", ensure_ready=False, images=[])
    assert captured["body"]["messages"][0]["content"] == "hi"
