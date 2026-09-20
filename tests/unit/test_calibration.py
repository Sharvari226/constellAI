"""Unit tests for Expected Calibration Error."""

from constellai.models.tgnn.evaluation import expected_calibration_error


def test_perfectly_calibrated_predictions_have_zero_ece():
    """10 predictions all at confidence 0.7, with EXACTLY 7 out of 10
    positive -- avg_confidence (0.7) equals observed_frequency (0.7)
    exactly, so this must be zero error, not approximately zero."""
    scores = [0.7] * 10
    labels = [1] * 7 + [0] * 3
    ece = expected_calibration_error(scores, labels, n_bins=10)
    assert ece == pytest_approx(0.0)


def pytest_approx(x, tol=1e-9):
    class _Approx:
        def __eq__(self, other):
            return abs(other - x) < tol
    return _Approx()


def test_confident_and_wrong_scores_worse_than_honestly_uncertain():
    """Same accuracy (50%), very different calibration: always
    predicting 0.95 when actually right 50% of the time is badly
    miscalibrated; predicting 0.5 when right 50% of the time is
    perfectly calibrated -- ECE must distinguish these, plain accuracy
    cannot."""
    labels = [1, 0] * 10

    overconfident_scores = [0.95] * 20
    honest_scores = [0.5] * 20

    ece_overconfident = expected_calibration_error(overconfident_scores, labels, n_bins=10)
    ece_honest = expected_calibration_error(honest_scores, labels, n_bins=10)

    assert ece_overconfident > ece_honest
    assert ece_honest < 0.05


def test_empty_input_returns_zero():
    assert expected_calibration_error([], [], n_bins=10) == 0.0