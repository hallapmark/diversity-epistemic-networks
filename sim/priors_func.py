from typing import Callable, NamedTuple, TypeAlias
import numpy as np

# Excluding 0 and 1 (uniform auto-excludes 1). 
UNIFORM_LOW = 0.001

class ConfidentStartConfig(NamedTuple):
    min_prop_confident: float # Proportion of agents for whom cr >= high_credence
    high_credence: float = 0.991 # > .99

Priors_Func: TypeAlias = Callable[[int, np.random.Generator], list[float]]

def uniform_priors(pop: int, rng: np.random.Generator) -> list[float]:
    """ 
    Takes as input the desired number of priors and an rng generator.

    Parameters:
    pop (int): The number of priors to generate.
    rng (np.random.Generator): The random number generator.

    Returns: 
    list[float]: a priors distribution that is uniform between 0.001 and 1.
    """
    return rng.uniform(low = UNIFORM_LOW, size = pop).tolist()

def confident_priors(pop: int, rng: np.random.Generator) -> list[float]:
    """ 
    Start off with the population (minus any potential skeptics) having 
    high confidence in the true view.
    
    Parameters:
    pop (int): The number of priors to generate.
    rng (np.random.Generator): The random number generator [ignored]. 
    
    Returns:
    list[float]: A list of priors with high confidence.
    """
    return [.991] * pop