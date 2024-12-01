import numpy as np
from dataclasses import dataclass
from typing import NamedTuple

class BinomialExperiment(NamedTuple): 
    k: int
    n: int

@dataclass
class ExperimentGen:
    rng: np.random.Generator

    def experiment(self, n: int, epsilon):
        k = self.rng.binomial(n, 0.5 + epsilon)
        return BinomialExperiment(k, n)
