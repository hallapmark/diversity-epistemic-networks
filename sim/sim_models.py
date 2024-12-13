from typing import Optional, NamedTuple, List
from enum import Enum, auto

from sim.priors_func import *

## SETUP
class ENParams(NamedTuple):
    pop_size: int # How many agents are in the network
    epsilon: float # How much better theory B is in fact. pB = 0.5 + epsilon. pA = 0.5
    m: float # how distrustful agents are of others' evidence (larger m means more distrustful)
    # See O'Connor and Weatherall 2018, Scientific Polarization

    skeptic_count: int = 0
    alternator: bool = False 
    # If True, any agents with .5 credence such as the skeptic will take action A 50% of the time 
    # and action B 50% of the time. If false, agents with .5 credence will always take action B.

    propagandist: bool = False
    # If True, replace one non-skeptic scientist in the network with a Propagandist

    trials: int = 5 # How many experiments, or 'coin flips', per round an agent will conduct

    max_rounds: int = 1000 # When we terminate the simulation
    rounds_to_new_agent: int = 10 
    # A new agent will appear and an existing agent will retire every x rounds

    admissions_priors_func: Priors_Func = uniform_priors
    # This is the prior distribution for newly admitted agents

    init_priors_func: Priors_Func = confident_priors
    # Controls priors distribution for the initial network

## RESULTS
class ENSingleSimResults(NamedTuple):
    ## SCORING
    sim_brier_penalty_total: float
    # sum(penalty of all agents in network in round i...n)
    # I.e. Total penalty from round 1 + total penalty from round 2 ... to n.
    sim_brier_penalty_ratio: float
    # The ratio of the total brier penalty obtained to the maximum possible penalty.
    # Maximum penalty would be obtained if all agents were maximally distant
    # from the truth every round of the game

    sim_non_skeptic_brier_ratio: Optional[float] = None
    # Only measured in lifecycle sims, and then only when skeptics are present.
    # The previous metric looked at the performance of the network as such.
    # Here we separately track the performance of the non-skeptics
    # to gauge the effect of the skeptic on the non-skeptics

    av_retired_brier_penalty: Optional[float] = None
    # Only measured in lifecycle sims. A snapshot providing the mean brier score of
    # agents just as they retire. We can think of this as being indicative of what 
    # kind of agents the network ultimately produces, or as what kind of transformation 
    # the network performs on the agent. Of course, we do not only care about what 
    # the agents' beliefs look like when they retire, we also care about what their 
    # beliefs are throughout their time in the network, and this metric does not tell
    # us anything about that

    prop_retired_confident: Optional[float] = None
    # Only measured in lifecycle sims.
    # A snapshot identifying the proportion of all retired agents who retired confident 
    # in the true view.

    unstable_conclusion_round: Optional[int] = None
    # Stable here means irrevocably polarized, research abandonment or
    # consensus on truth.
    # Lifecycle sims can go indefinitely until a hardcoded cutoff.
    # Non-lifecycle sims can also sometimes conclude in an unstable state.
    # This happens rarely in practice, but can happen if the number of
    # rounds is small.

    ## OTHER
    n_all_agents: Optional[float] = None
    # How many agents were in the network over all time (how many were touched
    # by the network). Only tracked for lifecycle sims
    # For stable sims, this would be equal to the initial population

class ENLifecycleAnalyzedResults(NamedTuple):
    ## METRICS FOR LIFECYCLE SIMS
    # Results taken as the mean value from all sim runs

    sims_av_n_all_agents: str
    # On average, the total number of agents that were present in the network over all time
    sims_av_total_brier_penalty: str # Av total penalty from all agents over all rounds
    sims_av_brier_ratio: str # Av ratio of obtained total penalty to obtainable penalty
    sims_sd_av_brier_ratio: str # Standard deviation
    sims_av_retired_brier_penalty: str
    # Take the total brier score of agents at the point they retire, and
    # divide by the maximum obtainable brier score for those agents at the point of retirement
    sims_av_prop_retired_confident: str
    sims_av_non_skeptic_brier_ratio: str
    
    
class ENSimsSummary(NamedTuple):
    params: ENParams
    results_summary: ENLifecycleAnalyzedResults
    # In the output csv, we record both the initial params
    # the sims were run with and the results

class ENResultsCSVWritableSummary(NamedTuple):
    headers: List[str]
    sim_data: List[str]
