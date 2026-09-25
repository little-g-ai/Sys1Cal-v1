from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Iterable

from jev_prob_bench.schemas import BenchmarkItem, LatentProblem, argmax, validate_distribution
from jev_prob_bench.renderers.base import render_problem


PRESET_COUNTS = {"tiny": 20, "dev": 100, "standard": 500, "full": 1000}


class FamilyGenerator(ABC):
    family: str
    difficulty: int
    representations: tuple[str, ...]

    def __init__(self, version: str, seed: int) -> None:
        self.version = version
        self.seed = seed

    @abstractmethod
    def latent(self, index: int, rng: random.Random) -> LatentProblem:
        raise NotImplementedError

    def generate(self, count: int) -> Iterable[BenchmarkItem]:
        for index in range(count):
            rng = random.Random(self.seed * 1000003 + index)
            if hasattr(self, "latent_with_count"):
                problem = self.latent_with_count(index, count, rng)
            else:
                problem = self.latent(index, rng)
            validate_distribution(problem.gold_distribution, problem.outcomes)
            for representation in self.representations:
                rendered_state, rendered_prompt = render_problem(problem, representation)
                item = BenchmarkItem(
                    benchmark_version=problem.benchmark_version,
                    instance_id=f"{problem.latent_instance_id}__repr_{representation}",
                    latent_instance_id=problem.latent_instance_id,
                    family=problem.family,
                    difficulty=problem.difficulty,
                    seed=problem.seed,
                    representation=representation,
                    state=rendered_state,
                    prompt=rendered_prompt,
                    outcomes=problem.outcomes,
                    gold_distribution=problem.gold_distribution,
                    gold_argmax=argmax(problem.gold_distribution),
                    generator_metadata=problem.generator_metadata,
                    probability_band=problem.probability_band,
                )
                yield item


def stratified_probability(index: int, rng: random.Random) -> float:
    bands = [
        (0.01, 0.10),
        (0.10, 0.30),
        (0.30, 0.45),
        (0.45, 0.55),
        (0.55, 0.70),
        (0.70, 0.90),
        (0.90, 0.99),
    ]
    lo, hi = bands[index % len(bands)]
    return round(rng.uniform(lo, hi), 6)


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    return [value / total for value in values]

