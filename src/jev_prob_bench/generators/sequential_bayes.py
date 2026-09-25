from __future__ import annotations

import random

from jev_prob_bench.generators.base import FamilyGenerator, stratified_probability
from jev_prob_bench.schemas import LatentProblem, probability_band


class SequentialBayesGenerator(FamilyGenerator):
    family = "sequential_bayes"
    difficulty = 6
    representations = ("direct", "table", "prose")

    def latent(self, index: int, rng: random.Random) -> LatentProblem:
        prior = stratified_probability(index, rng)
        steps = rng.randint(3, 6)
        likelihoods: list[dict[str, float]] = []
        posterior = prior
        trajectory = [posterior]
        for step in range(steps):
            like_h = stratified_probability(index + step + 2, rng)
            like_not = stratified_probability(index + step + 7, rng)
            denom = like_h * posterior + like_not * (1.0 - posterior)
            posterior = (like_h * posterior) / denom
            likelihoods.append({"E_given_H": like_h, "E_given_not_H": like_not})
            trajectory.append(posterior)
        state = {
            "prior_H": prior,
            "evidence_sequence": likelihoods,
            "observed_prefix_length": steps,
            "conditional_independence": "Evidence items are conditionally independent given H or not_H.",
            "posterior_trajectory": trajectory,
        }
        return LatentProblem(
            benchmark_version=self.version,
            latent_instance_id=f"{self.family}__seed_{self.seed + index:06d}",
            family=self.family,
            difficulty=self.difficulty,
            seed=self.seed + index,
            state=state,
            prompt="After all listed conditionally independent evidence items are observed, what is the posterior distribution over H versus not_H?",
            outcomes=["H", "not_H"],
            gold_distribution={"H": posterior, "not_H": 1.0 - posterior},
            generator_metadata={"parameters": state, "posterior_trajectory": trajectory},
            probability_band=probability_band(posterior),
        )

