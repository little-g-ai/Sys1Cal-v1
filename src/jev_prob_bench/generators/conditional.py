from __future__ import annotations

import random

from jev_prob_bench.generators.base import FamilyGenerator
from jev_prob_bench.schemas import LatentProblem, probability_band


class ConditionalGenerator(FamilyGenerator):
    family = "conditional"
    difficulty = 4
    representations = ("table", "nested", "direct", "prose", "distractor")

    def latent(self, index: int, rng: random.Random) -> LatentProblem:
        a_b = rng.randint(1, 90)
        a_not_b = rng.randint(1, 90)
        not_a_b = rng.randint(1, 90)
        not_a_not_b = rng.randint(1, 90)
        p = a_b / (a_b + not_a_b)
        state = {
            "contingency": {
                "A": {"B": a_b, "not_B": a_not_b},
                "not_A": {"B": not_a_b, "not_B": not_a_not_b},
            },
            "query": "P(A|B)",
            "sampling": "uniform among records satisfying B",
            "temporal_filter": "all records observed before cutoff T",
            "boundary_rule": "include records exactly at cutoff T",
        }
        return LatentProblem(
            benchmark_version=self.version,
            latent_instance_id=f"{self.family}__seed_{self.seed + index:06d}",
            family=self.family,
            difficulty=self.difficulty,
            seed=self.seed + index,
            state=state,
            prompt="Using the specified conditioning population B, what is the probability distribution over A versus not_A?",
            outcomes=["A", "not_A"],
            gold_distribution={"A": p, "not_A": 1.0 - p},
            generator_metadata={"parameters": state},
            probability_band=probability_band(p),
        )

