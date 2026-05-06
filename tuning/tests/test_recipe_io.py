"""Unit tests for recipe load/save/render."""
import yaml
from pathlib import Path

from conftest import FIXTURES
from sweep import load_recipe, save_recipe, render_recipe_command, apply_delta


def test_recipe_round_trip_preserves_content(tmp_path: Path):
    recipe = load_recipe(FIXTURES / "baseline-recipe.yaml")
    out = tmp_path / "out.yaml"
    save_recipe(recipe, out)
    reloaded = load_recipe(out)
    assert reloaded == recipe


def test_recipe_command_renders_with_defaults():
    recipe = load_recipe(FIXTURES / "baseline-recipe.yaml")
    rendered = render_recipe_command(recipe)
    # Defaults from baseline include port=8000, host=0.0.0.0, tp=2, gmu=0.78, mml=196608
    assert "--port 8000" in rendered
    assert "--host 0.0.0.0" in rendered
    assert "-tp 2" in rendered
    assert "--gpu-memory-utilization 0.78" in rendered
    assert "--max-model-len 196608" in rendered
    # Static flags from T0/T1.2 should be present
    assert "--max-num-seqs 16" in rendered
    assert "--max-num-batched-tokens 16384" in rendered
    assert "--prefix-caching-hash-algo sha256_cbor" in rendered


def test_recipe_renders_after_delta_application():
    recipe = load_recipe(FIXTURES / "baseline-recipe.yaml")
    delta = {
        "command_append": [
            "--long-prefill-token-threshold 4096",
            "--max-num-partial-prefills 2",
        ]
    }
    new = apply_delta(recipe, delta)
    rendered = render_recipe_command(new)
    assert "--long-prefill-token-threshold 4096" in rendered
    assert "--max-num-partial-prefills 2" in rendered
    # Originals still present
    assert "--port 8000" in rendered


def test_render_raises_on_missing_template_var():
    recipe = load_recipe(FIXTURES / "baseline-recipe.yaml")
    # Inject a reference to an undefined variable
    recipe["command"] = recipe["command"].rstrip() + " --extra {undefined_var}\n"
    try:
        render_recipe_command(recipe)
    except KeyError as e:
        assert "undefined_var" in str(e)
        return
    raise AssertionError("Expected KeyError for missing template var")


def test_save_recipe_atomic_no_partial_file(tmp_path: Path):
    recipe = load_recipe(FIXTURES / "baseline-recipe.yaml")
    out = tmp_path / "atomic.yaml"
    save_recipe(recipe, out)
    # No leftover .tmp file
    assert not (tmp_path / "atomic.yaml.tmp").exists()
    assert out.exists()
