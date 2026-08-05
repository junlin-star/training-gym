from __future__ import annotations

from tutorials.generate_tutorial import (
    _DEFAULT_REQUIRED_MODAL_SECRETS,
    _NOTEBOOK_GPU_NOTE_MARKDOWN,
    Cell,
    _inject_secret_check,
    _py_globals_used_by_definitions,
    _required_modal_secrets,
    _split_py_code_cell,
)


def _scopes(*sources: str) -> list[list[tuple[str, str]]]:
    cells = [
        Cell(kind="code", source=source, targets={"py", "ipynb"}) for source in sources
    ]
    globals_used = _py_globals_used_by_definitions(cells)
    return [
        _split_py_code_cell(source, globals_used_by_defs=globals_used)
        for source in sources
    ]


def _scope_of(scoped: list[tuple[str, str]], needle: str) -> str:
    for scope, statement in scoped:
        if needle in statement:
            return scope
    raise AssertionError(f"{needle!r} not found in {scoped!r}")


def test_default_argument_constant_from_an_earlier_cell_is_hoisted() -> None:
    constant_cell, function_cell = _scopes(
        "CHECK_MAX_CONCURRENCY = 4",
        "def run_check(url, max_concurrency: int = CHECK_MAX_CONCURRENCY):\n"
        "    return url, max_concurrency",
    )

    assert _scope_of(constant_cell, "CHECK_MAX_CONCURRENCY = 4") == "module"
    assert _scope_of(function_cell, "def run_check") == "module"


def test_decorator_and_annotation_constants_are_hoisted() -> None:
    constants_cell, function_cell = _scopes(
        "CACHE = staticmethod\nALIAS = str",
        "@CACHE\ndef labelled(value: ALIAS) -> ALIAS:\n    return value",
    )

    assert _scope_of(constants_cell, "CACHE = staticmethod") == "module"
    assert _scope_of(constants_cell, "ALIAS = str") == "module"
    assert _scope_of(function_cell, "def labelled") == "module"


def test_base_class_from_an_earlier_cell_is_hoisted() -> None:
    base_cell, subclass_cell = _scopes(
        "BASE = dict",
        "class Rows(BASE):\n    pass",
    )

    assert _scope_of(base_cell, "BASE = dict") == "module"
    assert _scope_of(subclass_cell, "class Rows") == "module"


def test_runtime_values_no_definition_needs_stay_in_main() -> None:
    setup_cell, function_cell = _scopes(
        "train_result = train()",
        "def report(run_id):\n    return run_id",
    )

    assert _scope_of(setup_cell, "train_result = train()") == "main"
    assert _scope_of(function_cell, "def report") == "module"


def test_explicit_empty_required_secrets_overrides_default() -> None:
    assert _required_modal_secrets({}) == _DEFAULT_REQUIRED_MODAL_SECRETS
    assert _required_modal_secrets({"required_modal_secrets": []}) == ()


def test_empty_required_secrets_still_injects_notebook_gpu_note() -> None:
    cells = [
        Cell(kind="markdown", source="# Intro", targets=frozenset({"py", "notebook"})),
        Cell(kind="code", source="x = 1", targets=frozenset({"py", "notebook"})),
    ]

    injected = _inject_secret_check(cells, ())

    assert len(injected) == 3
    assert injected[0].kind == "markdown"
    assert injected[0].source == "# Intro"
    assert injected[1].kind == "markdown"
    assert injected[1].source == _NOTEBOOK_GPU_NOTE_MARKDOWN
    assert injected[1].targets == frozenset({"notebook"})
    assert injected[2].kind == "code"
    assert injected[2].source == "x = 1"
    assert not any("Modal Secret" in cell.source for cell in injected)
