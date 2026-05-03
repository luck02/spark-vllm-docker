"""Unit tests for compare()."""
from sweep import compare, DEFAULT_THRESHOLDS


def _baseline(c1_tps=40.0, c1_tpot=25.0):
    return {1: {"output_tps": c1_tps, "tpot_p50": c1_tpot}}


def test_clear_throughput_win_accepts():
    baseline = _baseline()
    results = {1: {"output_tps": 50.0, "tpot_p50": 25.0}}  # +25%
    decision, reason = compare(results, baseline)
    assert decision == "ACCEPTED"
    assert "+25.0%" in reason


def test_tpot_regression_rejects():
    baseline = _baseline()
    results = {1: {"output_tps": 50.0, "tpot_p50": 35.0}}  # +10ms tpot regression
    decision, reason = compare(results, baseline)
    assert decision == "REJECTED"
    assert "TPOT" in reason


def test_throughput_drop_rejects():
    baseline = _baseline()
    results = {1: {"output_tps": 35.0, "tpot_p50": 25.0}}  # -12.5%
    decision, _ = compare(results, baseline)
    assert decision == "REJECTED"


def test_noise_band_review():
    baseline = _baseline()
    results = {1: {"output_tps": 41.0, "tpot_p50": 25.0}}  # +2.5%, below 3% win threshold
    decision, reason = compare(results, baseline)
    assert decision == "REVIEW"
    assert "noise band" in reason.lower()


def test_exact_clear_win_threshold_accepts():
    baseline = _baseline()
    results = {1: {"output_tps": 41.2, "tpot_p50": 25.0}}  # +3.0% exactly
    decision, _ = compare(results, baseline)
    assert decision == "ACCEPTED"


def test_acceptable_throughput_drop_below_min():
    baseline = _baseline()
    # -1% is within the -2% min, no clear win, so REVIEW
    results = {1: {"output_tps": 39.6, "tpot_p50": 25.0}}
    decision, _ = compare(results, baseline)
    assert decision == "REVIEW"


def test_throughput_drop_at_min_threshold_rejects():
    baseline = _baseline()
    # -2.5% is below the -2% min
    results = {1: {"output_tps": 39.0, "tpot_p50": 25.0}}
    decision, _ = compare(results, baseline)
    assert decision == "REJECTED"


def test_multi_concurrency_one_clear_win_accepts():
    baseline = {
        1: {"output_tps": 40.0, "tpot_p50": 25.0},
        4: {"output_tps": 80.0, "tpot_p50": 50.0},
    }
    results = {
        1: {"output_tps": 41.0, "tpot_p50": 25.0},  # +2.5% — noise band
        4: {"output_tps": 100.0, "tpot_p50": 50.0},  # +25% — clear win
    }
    decision, _ = compare(results, baseline)
    assert decision == "ACCEPTED"


def test_multi_concurrency_one_regression_rejects():
    baseline = {
        1: {"output_tps": 40.0, "tpot_p50": 25.0},
        4: {"output_tps": 80.0, "tpot_p50": 50.0},
    }
    results = {
        1: {"output_tps": 50.0, "tpot_p50": 25.0},  # +25% win
        4: {"output_tps": 80.0, "tpot_p50": 60.0},  # +10ms tpot regression
    }
    decision, reason = compare(results, baseline)
    assert decision == "REJECTED"
    assert "c=4" in reason


def test_custom_thresholds_apply():
    baseline = _baseline()
    results = {1: {"output_tps": 41.5, "tpot_p50": 25.0}}  # +3.75%
    # Stricter threshold: clear win requires +5%
    decision, _ = compare(results, baseline,
                          {"throughput_clear_win_pct": 5.0,
                           "tpot_p50_max_regression_ms": 5.0,
                           "throughput_min_improvement_pct": -2.0})
    assert decision == "REVIEW"


def test_missing_baseline_concurrency_skipped():
    baseline = {1: {"output_tps": 40.0, "tpot_p50": 25.0}}
    results = {
        1: {"output_tps": 41.0, "tpot_p50": 25.0},
        99: {"output_tps": 200.0, "tpot_p50": 25.0},  # not in baseline
    }
    decision, reason = compare(results, baseline)
    assert decision == "REVIEW"
    assert "c=99" not in reason
