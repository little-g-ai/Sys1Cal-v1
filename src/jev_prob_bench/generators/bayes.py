from __future__ import annotations

import random

from jev_prob_bench.generators.base import FamilyGenerator, normalize, stratified_probability
from jev_prob_bench.schemas import LatentProblem, probability_band


class BayesGenerator(FamilyGenerator):
    family = "bayes"
    difficulty = 5
    representations = ("direct", "counts", "table", "prose", "distractor")

    def latent(self, index: int, rng: random.Random) -> LatentProblem:
        if index % 3 == 0:
            prior_h = stratified_probability(index, rng)
            like_h = stratified_probability(index + 2, rng)
            like_not = stratified_probability(index + 5, rng)
            denom = like_h * prior_h + like_not * (1.0 - prior_h)
            posterior_h = (like_h * prior_h) / denom
            outcomes = ["H", "not_H"]
            gold = {"H": posterior_h, "not_H": 1.0 - posterior_h}
            state = {"priors": {"H": prior_h, "not_H": 1.0 - prior_h}, "likelihood_event": {"H": like_h, "not_H": like_not}, "observed_event": True}
        else:
            k = rng.choice([3, 4, 5])
            outcomes = [f"H{i + 1}" for i in range(k)]
            priors_list = normalize([rng.random() + 0.05 for _ in range(k)])
            likelihoods = [stratified_probability(index + i, rng) for i in range(k)]
            weights = [p * l for p, l in zip(priors_list, likelihoods)]
            posteriors = normalize(weights)
            gold = dict(zip(outcomes, posteriors))
            state = {"priors": dict(zip(outcomes, priors_list)), "likelihood_event": dict(zip(outcomes, likelihoods)), "observed_event": True}
        first_p = gold[outcomes[0]]
        return LatentProblem(
            benchmark_version=self.version,
            latent_instance_id=f"{self.family}__seed_{self.seed + index:06d}",
            family=self.family,
            difficulty=self.difficulty,
            seed=self.seed + index,
            state=state,
            prompt="Given the prior probabilities and likelihoods, what is the posterior distribution after observing the event?",
            outcomes=outcomes,
            gold_distribution=gold,
            generator_metadata={"parameters": state},
            probability_band=probability_band(first_p),
        )

