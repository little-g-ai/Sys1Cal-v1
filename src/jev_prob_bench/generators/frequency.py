from __future__ import annotations

import math
import random

from jev_prob_bench.generators.base import FamilyGenerator, normalize
from jev_prob_bench.schemas import LatentProblem, probability_band


class FrequencyGenerator(FamilyGenerator):
    family = "frequency"
    difficulty = 2
    representations = ("counts", "scaled_counts", "ratio", "table", "prose", "distractor")

    def latent(self, index: int, rng: random.Random) -> LatentProblem:
        if index % 4 == 0:
            total = rng.choice([10, 20, 50, 100])
            a = rng.randint(1, total - 1)
            counts = {"zor": a, "nif": total - a}
        else:
            k = rng.choice([3, 4, 5])
            raw = [rng.randint(1, 200) for _ in range(k)]
            labels = [f"class_{chr(65 + i)}" for i in range(k)]
            counts = dict(zip(labels, raw))
        total = sum(counts.values())
        gold = {label: count / total for label, count in counts.items()}
        first_p = next(iter(gold.values()))
        gcd = math.gcd(*counts.values())
        return LatentProblem(
            benchmark_version=self.version,
            latent_instance_id=f"{self.family}__seed_{self.seed + index:06d}",
            family=self.family,
            difficulty=self.difficulty,
            seed=self.seed + index,
            state={"counts": counts, "sampling": "uniform"},
            prompt="An item is sampled uniformly from the collection. What is the probability distribution over item labels?",
            outcomes=list(counts),
            gold_distribution=gold,
            generator_metadata={"parameters": {"counts": counts, "reduced_counts": {k: v // gcd for k, v in counts.items()}}},
            probability_band=probability_band(first_p),
        )

