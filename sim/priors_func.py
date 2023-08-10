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
    confident_start_config: Optional[ConfidentStartConfig] = None
    # If set, most agents will start with a high credence in the true theory, B
    admissions_low_priors_ceiling: Optional[float] = None

# Takes as input number of priors, an rng generator as input, and a PriorSetup config.
# Outputs a priors distribution
Priors_Func: TypeAlias = Callable[[int, np.random.Generator, PriorSetup], list[float]]

def uniform_priors(pop: int, rng: np.random.Generator, priorsetup: PriorSetup) -> list[float]:
    return rng.uniform(low = priorsetup.uniform_low, size = pop).tolist()

# Start off with most having confidence in the better theory
def confident_priors(pop: int, rng: np.random.Generator, priorsetup: PriorSetup) -> list[float]:
    """ Start off with much of the population having high confidence in the true view.
    The typical use case is to have everyone in the population start off with .991
    confidence in the true view (minus any potential skeptics)."""
    if not priorsetup.confident_start_config:
        raise ValueError("Must define ConfidentStartConfig.")
    # We round up: we never assign cr == high_credence to less than proportion_confident of the agents
    # E.g. 0.8 x 13 = 10.4 -> 11 agents get credences at high_credence
    n_confident = math.ceil(priorsetup.confident_start_config.min_prop_confident * pop)
    priors = [priorsetup.confident_start_config.high_credence] * n_confident
    if n_confident < pop:
        priors.extend(rng.uniform(low = UNIFORM_LOW, size = pop - n_confident).tolist())
    # The others get a uniform distribution
    # NOTE: This is ordered. Which does not matter in a fully connected network. But it
    # potentially would for a cycle. So mix it if you ever want to use this on a cycle network.
    return priors

def low_priors(pop: int, rng: np.random.Generator, priorsetup: PriorSetup) -> list[float]:
    ceiling = priorsetup.admissions_low_priors_ceiling
    high = ceiling if ceiling else 0.25
    return rng.uniform(low = 0.001, high = high, size = pop).tolist()
