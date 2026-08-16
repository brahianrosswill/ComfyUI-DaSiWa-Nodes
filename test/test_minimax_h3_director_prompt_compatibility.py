"""Regression tests for preserving pre-builder MiniMax Director prompts."""
from nodes.helper_minimax_h3_prompt_builder import (
    build_prompt,
    default_builder_state,
    migrate_legacy_prompt,
)
from nodes.nodes_minimax_h3_director import MiniMaxH3Director


def test_migrates_legacy_widget_prompt_losslessly():
    builder = default_builder_state("FL2VA")
    prompt = "A complex scene\n<d>[English] Keep every token.</d>"

    assert migrate_legacy_prompt(builder, {}, prompt)
    assert builder["prompt_mode"] == "simple"
    assert builder["simple_prompt"] == prompt
    assert build_prompt(builder) == prompt


def test_prefers_exact_prior_resolved_prompt_over_legacy_parts():
    builder = default_builder_state("REF2VA")
    state = {
        "resolved_prompt": "Exact former generation prompt\nwith all sections intact",
        "prompt": "short old prompt",
        "prompt_blocks": [{"text": "old block"}],
    }

    assert migrate_legacy_prompt(builder, state, "widget prompt")
    assert build_prompt(builder) == state["resolved_prompt"]


def test_legacy_prompt_blocks_are_ordered_and_disabled_blocks_excluded():
    builder = default_builder_state("T2VA")
    state = {
        "prompt": "Global prompt",
        "prompt_blocks": [
            {"text": "Second", "start": 2, "order": 1},
            {"text": "Disabled", "start": 0, "enabled": False},
            {"text": "First", "start": 1, "order": 0},
        ],
    }

    assert migrate_legacy_prompt(builder, state)
    assert build_prompt(builder) == "Global prompt\nFirst\nSecond"


def test_never_overwrites_current_builder_content():
    builder = default_builder_state("FL2VA")
    builder["imd"] = "Current authored prompt"

    assert not migrate_legacy_prompt(builder, {"resolved_prompt": "Old prompt"}, "Old widget")
    assert "simple_prompt" not in builder


def test_director_execution_preserves_legacy_prompt_in_guide():
    prompt = "Detailed legacy prompt\n<d>[English] Exact dialogue.</d>"
    guide = MiniMaxH3Director().build_guide(
        "T2VA", prompt, 1344, 768, 5, "match", "{}", "",
    )[0]

    assert guide["resolved_prompt"] == prompt
    assert guide["builder_state"]["simple_prompt"] == prompt