"""Unit tests for parse_bench_output()."""
from conftest import FIXTURES
from sweep import parse_bench_output


def load_sample() -> str:
    return (FIXTURES / "sample-bench-output.txt").read_text()


def test_parse_real_t12_output():
    text = load_sample()
    parsed = parse_bench_output(text)
    # These are the actual T1.2 c=1 numbers
    assert parsed["output_tps"] == 38.88
    assert parsed["total_tps"] == 76.10
    assert parsed["tpot_p50"] == 24.73
    assert parsed["tpot_p95"] == 25.42
    assert parsed["ttft_p50"] == 206.13
    assert parsed["ttft_p95"] == 439.53
    assert parsed["interactive_tps"] == 40.4
    assert parsed["total_input_tokens"] == 102498
    assert parsed["duration_s"] == 2753.44


def test_parse_handles_garbage():
    parsed = parse_bench_output("this string contains no metrics whatsoever")
    assert parsed == {}


def test_parse_handles_partial():
    text = """
    Output token throughput (tok/s):         50.5
    P50 TPOT (ms):                           20.0
    """
    parsed = parse_bench_output(text)
    assert parsed["output_tps"] == 50.5
    assert parsed["tpot_p50"] == 20.0
    assert parsed["interactive_tps"] == 50.0  # 1000/20.0


def test_parse_does_not_set_interactive_when_no_tpot():
    text = "Output token throughput (tok/s):         50.5"
    parsed = parse_bench_output(text)
    assert parsed["output_tps"] == 50.5
    assert "interactive_tps" not in parsed


def test_parse_handles_zero_tpot_safely():
    text = "P50 TPOT (ms):                           0"
    parsed = parse_bench_output(text)
    assert parsed["tpot_p50"] == 0.0
    assert "interactive_tps" not in parsed
