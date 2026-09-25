from __future__ import annotations

import math
import random

from jev_prob_bench.generators.base import FamilyGenerator, stratified_probability
from jev_prob_bench.schemas import LatentProblem, probability_band


class CompoundGenerator(FamilyGenerator):
    family = "compound"
    difficulty = 3
    representations = ("direct", "table", "prose")

    def latent(self, index: int, rng: random.Random) -> LatentProblem:
        subtask = ["complement", "intersection", "union", "repeated_exact", "hypergeometric"][index % 5]
        if subtask == "complement":
            p = stratified_probability(index, rng)
            target = 1.0 - p
            state = {"subtask": subtask, "p_A": p, "query": "not_A"}
            prompt = "Event A has the supplied probability. What is the probability that A does not occur?"
        elif subtask == "intersection":
            p_a = stratified_probability(index, rng)
            p_b = stratified_probability(index + 3, rng)
            target = p_a * p_b
            state = {"subtask": subtask, "p_A": p_a, "p_B": p_b, "independent": True, "query": "A_and_B"}
            prompt = "Events A and B are independent. What is the probability that both occur?"
        elif subtask == "union":
            p_a = stratified_probability(index, rng)
            p_b = stratified_probability(index + 4, rng)
            target = p_a + p_b - p_a * p_b
            state = {"subtask": subtask, "p_A": p_a, "p_B": p_b, "independent": True, "query": "A_or_B"}
            prompt = "Events A and B are independent. What is the probability that at least one occurs?"
        elif subtask == "repeated_exact":
            n = rng.randint(3, 10)
            k = rng.randint(1, n - 1)
            p = stratified_probability(index, rng)
            target = math.comb(n, k) * (p**k) * ((1.0 - p) ** (n - k))
            state = {"subtask": subtask, "n": n, "k": k, "success_probability": p, "query": "exactly_k_successes"}
            prompt = "Independent Bernoulli trials use the supplied success probability. What is the probability of exactly k successes?"
        else:
            good = rng.randint(2, 20)
            bad = rng.randint(2, 20)
            draws = rng.randint(1, min(6, good + bad - 1))
            k = rng.randint(0, min(draws, good))
            if draws - k > bad:
                k = draws - bad
            target = math.comb(good, k) * math.comb(bad, draws - k) / math.comb(good + bad, draws)
            state = {"subtask": subtask, "good": good, "bad": bad, "draws_without_replacement": draws, "k_good": k}
            prompt = "A sample is drawn without replacement. What is the probability of exactly k good items?"
        return LatentProblem(
            benchmark_version=self.version,
            latent_instance_id=f"{self.family}__seed_{self.seed + index:06d}",
            family=self.family,
            difficulty=self.difficulty,
            seed=self.seed + index,
            state=state,
            prompt=prompt,
            outcomes=["target", "not_target"],
            gold_distribution={"target": target, "not_target": 1.0 - target},
            generator_metadata={"parameters": state},
            probability_band=probability_band(target),
        )

