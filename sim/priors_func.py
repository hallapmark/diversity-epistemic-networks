import math
from typing import Callable, NamedTuple, Optional, TypeAlias
import numpy as np

# Excluding 0 and 1 (uniform auto-excludes 1). In some models 0 is included
UNIFORM_LOW = 0.001

class ConfidentStartConfig(NamedTuple):
    min_prop_confident: float # Proportion of agents for whom cr >= high_credence
    high_credence: float = 0.991 # > .99

class PriorSetup(NamedTuple):
    uniform_low: float = UNIFORM_LOW
    # If set, most agents will start with a high credence in the true theory, B
    
# Takes as input number of priors, an rng generator as input, and a PriorSetup config.
# Outputs a priors distribution
Priors_Func: TypeAlias = Callable[[int, np.random.Generator, PriorSetup], list[float]]

def uniform_priors(pop: int, rng: np.random.Generator, priorsetup: PriorSetup) -> list[float]:
    return rng.uniform(low = priorsetup.uniform_low, size = pop).tolist()

