"""Unit tests for apply_delta()."""
import yaml
from pathlib import Path

from conftest import FIXTURES
from sweep import apply_delta


def load_baseline_recipe() -> dict:
    return yaml.safe_load((FIXTURES / "baseline-recipe.yaml").read_text())


def test_command_append_adds_flags():
    recipe = load_baseline_recipe()
    delta = {"command_append": ["--flag-a", "--flag-b value"]}
    new = apply_delta(recipe, delta)
    assert "--flag-a" in new["command"]
    assert "--flag-b value" in new["command"]
    # Original unchanged
    assert "--flag-a" not in recipe["command"]


def test_command_append_preserves_existing_lines():
    recipe = load_baseline_recipe()
    original_lines = recipe["command"].count("--")
    delta = {"command_append": ["--new-flag"]}
    new = apply_delta(recipe, delta)
    new_lines = new["command"].count("--")
    assert new_lines == original_lines + 1


def test_command_replace_swaps_existing():
    recipe = load_baseline_recipe()
    # Baseline has --max-num-seqs 16
    delta = {"command_replace": {"--max-num-seqs": "--max-num-seqs 32"}}
    new = apply_delta(recipe, delta)
    assert "--max-num-seqs 32" in new["command"]
    # No duplicate occurrences of the old flag
    assert new["command"].count("--max-num-seqs ") == 1


def test_command_replace_appends_when_missing():
    recipe = load_baseline_recipe()
    delta = {"command_replace": {"--brand-new-flag": "--brand-new-flag value"}}
    new = apply_delta(recipe, delta)
    assert "--brand-new-flag value" in new["command"]


def test_env_add_merges_with_existing():
    recipe = load_baseline_recipe()
    # Baseline has SAFETENSORS_FAST_GPU=1
    delta = {"env_add": {"FOO": "1", "BAR": "2"}}
    new = apply_delta(recipe, delta)
    assert new["env"]["FOO"] == "1"
    assert new["env"]["BAR"] == "2"
    assert new["env"]["SAFETENSORS_FAST_GPU"] == "1"


def test_env_add_creates_env_block_if_missing():
    recipe = load_baseline_recipe()
    recipe["env"] = None  # simulate missing env
    delta = {"env_add": {"X": "Y"}}
    new = apply_delta(recipe, delta)
    assert new["env"] == {"X": "Y"}


def test_empty_delta_is_identity():
    recipe = load_baseline_recipe()
    new = apply_delta(recipe, {})
    assert new == recipe


def test_delta_does_not_mutate_input():
    recipe = load_baseline_recipe()
    original_cmd = recipe["command"]
    original_env = dict(recipe.get("env", {}))
    apply_delta(recipe, {"command_append": ["--x"], "env_add": {"K": "V"}})
    assert recipe["command"] == original_cmd
    assert recipe.get("env", {}) == original_env


def test_command_append_then_replace_works():
    recipe = load_baseline_recipe()
    # Apply T1.4-like compilation-config first via append
    r1 = apply_delta(recipe, {
        "command_append": ['--compilation-config \'{"cudagraph_mode":"PIECEWISE"}\'']
    })
    # Now T2.1 replaces it
    r2 = apply_delta(r1, {
        "command_replace": {
            "--compilation-config":
                '--compilation-config \'{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}\''
        }
    })
    assert "fuse_minimax_qk_norm" in r2["command"]
    assert r2["command"].count("--compilation-config") == 1


def test_combined_delta_command_and_env():
    recipe = load_baseline_recipe()
    delta = {
        "command_append": ["--attention-backend FLASHINFER"],
        "env_add": {"VLLM_USE_FLASHINFER_MOE_FP16": "1"},
    }
    new = apply_delta(recipe, delta)
    assert "--attention-backend FLASHINFER" in new["command"]
    assert new["env"]["VLLM_USE_FLASHINFER_MOE_FP16"] == "1"


def test_delta_with_json_value_renders_correctly():
    """JSON-valued flags must survive str.format() rendering.

    Recipes use str.format() for {port}, {host}, etc. Without brace escaping,
    a delta like '--compilation-config '{"mode":3}'' would crash render with
    KeyError: '"mode"'.
    """
    from sweep import render_recipe_command

    recipe = load_baseline_recipe()
    delta = {
        "command_append": [
            "--compilation-config '{\"cudagraph_mode\":\"PIECEWISE\"}'",
        ]
    }
    new = apply_delta(recipe, delta)
    rendered = render_recipe_command(new)
    # Single braces survive in the rendered output
    assert "{\"cudagraph_mode\":\"PIECEWISE\"}" in rendered
    assert "--compilation-config" in rendered


def test_command_replace_with_json_value_renders():
    from sweep import render_recipe_command

    recipe = load_baseline_recipe()
    # First add a JSON config, then replace it
    r1 = apply_delta(recipe, {
        "command_append": ["--compilation-config '{\"a\":1}'"]
    })
    r2 = apply_delta(r1, {
        "command_replace": {
            "--compilation-config":
                "--compilation-config '{\"b\":2,\"c\":{\"nested\":true}}'"
        }
    })
    rendered = render_recipe_command(r2)
    assert "{\"b\":2,\"c\":{\"nested\":true}}" in rendered
    assert rendered.count("--compilation-config") == 1
