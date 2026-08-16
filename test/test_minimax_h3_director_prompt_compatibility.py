"""Regression tests for Director prompt migration and standard prompt output."""
from nodes.helper_minimax_h3_prompt_builder import (
    build_prompt,
    default_builder_state,
    migrate_legacy_prompt,
)
from nodes.nodes_minimax_h3_director import MiniMaxH3Director


def test_legacy_widget_prompt_is_preserved_losslessly():
    prompt = "Detailed old prompt\n<d>[English] Exact dialogue.</d>"
    builder = default_builder_state("FL2VA")

    assert migrate_legacy_prompt(builder, {}, prompt)
    assert builder["prompt_mode"] == "simple"
    assert build_prompt(builder) == prompt


def test_director_uses_legacy_prompt_when_builder_is_absent():
    prompt = "Former video generation prompt"
    guide = MiniMaxH3Director().build_guide(
        "T2VA", prompt, 1344, 768, 5, "match", "{}", "",
    )[0]

    assert guide["resolved_prompt"] == prompt
    assert guide["builder_state"]["simple_prompt"] == prompt


def test_frontend_keeps_standard_prompt_widget_serialized():
    source = open("js/minimax_h3_director.js", encoding="utf-8").read()

    assert "promptWidget.value = resolved" in source
    assert "promptWidget.callback?.(resolved)" in source
    assert "state.resolved_prompt = resolved" in source
