import numpy as np
import pytest


def test_zero_and_perfect_success_counts_can_actually_be_plotted():
    import closed_loop_report as report
    assert hasattr(report, "success_error_bars"), "binomial plot boundary handling is missing"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    errors = report.success_error_bars([0, 50, 100], 100)
    assert errors.shape == (2, 3)
    assert (errors >= 0).all()
    assert errors[0, 0] == 0 and errors[1, 2] == 0
    assert errors[1, 0] == pytest.approx(3.69935, abs=.001)
    fig, ax = plt.subplots()
    ax.bar([0, 1, 2], [0, 50, 100], yerr=errors)
    fig.canvas.draw()
    plt.close(fig)


def test_audit_rejects_tampered_latency_or_query_averages():
    import closed_loop_report as report
    assert hasattr(report, "validate_metrics"), "summary measurements are not checked against episodes"
    rows = [{"steps": 10, "policy_queries": 2, "query_latency_ms": [1., 3.]},
            {"steps": 20, "policy_queries": 4, "query_latency_ms": [2., 4., 5., 6.]}]
    summary = {"mean_control_steps": 15., "mean_policy_queries": 3.,
               "query_latency_ms_p50": 3.5, "query_latency_ms_p95": 5.75}
    report.validate_metrics(summary, rows)
    for key in summary:
        with pytest.raises(ValueError):
            report.validate_metrics({**summary, key: summary[key] + 1}, rows)
