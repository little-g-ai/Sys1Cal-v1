from __future__ import annotations

import random

from jev_prob_bench.generators.base import FamilyGenerator, stratified_probability
from jev_prob_bench.schemas import LatentProblem, probability_band


class ExplicitProbabilityGenerator(FamilyGenerator):
    family = "explicit_probability"
    difficulty = 1
    representations = ("direct", "ratio", "prose", "distractor")

    def latent_with_count(self, index: int, count: int, rng: random.Random) -> LatentProblem:
        if count <= 1:
            p = 0.5
        else:
            p = round(index / (count - 1), 12)
        return self._problem(index, p)

    def latent(self, index: int, rng: random.Random) -> LatentProblem:
        p = stratified_probability(index, rng)
        return self._problem(index, p)

    def _problem(self, index: int, p: float) -> LatentProblem:
        outcomes = ["A", "not_A"]
        return LatentProblem(
            benchmark_version=self.version,
            latent_instance_id=f"{self.family}__seed_{self.seed + index:06d}",
            family=self.family,
            difficulty=self.difficulty,
            seed=self.seed + index,
            state={"p_A": p, "p_not_A": 1.0 - p},
            prompt="Given the fully specified state, what is the probability that event A occurs?",
            outcomes=outcomes,
            gold_distribution={"A": p, "not_A": 1.0 - p},
            generator_metadata={"parameters": {"p_A": p, "p_not_A": 1.0 - p}},
            probability_band=probability_band(p),
        )

