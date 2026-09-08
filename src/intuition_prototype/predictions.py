"""Shared declared prediction criteria; these are not calibrated probabilities."""

import math

from intuition_prototype.records import Intervention, Metrics, Prediction


def make_prediction(
    record_id: str, hypothesis_id: str, baseline: Metrics, intervention: Intervention,
) -> Prediction:
    if intervention == Intervention.DISABLE_RETRIES:
        return Prediction(
            record_id, hypothesis_id,
            "No fewer unique completions, at least 30% less queue growth "
            "(and strictly less), and zero new retries.",
            baseline.completions,
            min(baseline.queue_growth - 1, math.floor(baseline.queue_growth * 0.7)),
            0,
        )
    return Prediction(
        record_id, hypothesis_id,
        "At least 20% more unique completions AND strictly lower queue growth than baseline.",
        max(1, math.ceil(baseline.completions * 1.2)),
        baseline.queue_growth - 1,
    )
