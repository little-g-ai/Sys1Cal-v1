from jev_prob_bench.generators.bayes import BayesGenerator
from jev_prob_bench.generators.compound import CompoundGenerator
from jev_prob_bench.generators.conditional import ConditionalGenerator
from jev_prob_bench.generators.explicit_probability import ExplicitProbabilityGenerator
from jev_prob_bench.generators.frequency import FrequencyGenerator
from jev_prob_bench.generators.sequential_bayes import SequentialBayesGenerator


GENERATOR_REGISTRY = {
    "explicit_probability": ExplicitProbabilityGenerator,
    "frequency": FrequencyGenerator,
    "compound": CompoundGenerator,
    "conditional": ConditionalGenerator,
    "bayes": BayesGenerator,
    "sequential_bayes": SequentialBayesGenerator,
}

