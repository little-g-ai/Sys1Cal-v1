from __future__ import annotations

import math
import random
from typing import Any


def simulate_calibration(rows: list[dict[str, Any]], seed: int = 123, bins: int = 10) -> dict[str, Any]:
    rng = random.Random(seed)
    binary = [row for row in rows if not row.get("error") and len(row["gold_distribution"]) == 2]
    records = []
    for row in binary:
        outcomes = list(row["gold_distribution"])
        positive = outcomes[0]
        y = 1 if rng.random() < row["gold_distribution"][positive] else 0
        q = row["predicted_distribution"][positive]
        records.append((q, y))
    if not records:
        return {"bins": [], "ece": 0.0, "empirical_brier": 0.0, "empirical_log_loss": 0.0}
    bucketed = []
    ece = 0.0
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        bucket = [(q, y) for q, y in records if lo <= q < hi or (i == bins - 1 and q == 1.0)]
        if not bucket:
            continue
        mean_q = sum(q for q, _ in bucket) / len(bucket)
        freq = sum(y for _, y in bucket) / len(bucket)
        ece += len(bucket) / len(records) * abs(freq - mean_q)
        bucketed.append({"lo": lo, "hi": hi, "count": len(bucket), "mean_predicted": mean_q, "empirical_frequency": freq})
    eps = 1e-12
    brier = sum((q - y) ** 2 for q, y in records) / len(records)
    log_loss = -sum(y * math.log(max(eps, q)) + (1 - y) * math.log(max(eps, 1 - q)) for q, y in records) / len(records)
    return {"bins": bucketed, "ece": ece, "empirical_brier": brier, "empirical_log_loss": log_loss}

