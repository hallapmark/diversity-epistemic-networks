from typing import Callable, TypeAlias
import numpy as np

# Takes number of priors and an rng generator as input. Outputs a priors distribution
Priors_Func: TypeAlias = Callable[[int, np.random.Generator], list[float]]

def uniform_priors(n: int, rng: np.random.Generator) -> list[float]:
    # Excluding 0 and 1 (uniform auto-excludes 1). Cf. sep-sen where 0 is included
    return rng.uniform(low = 0.001, size = n).tolist()