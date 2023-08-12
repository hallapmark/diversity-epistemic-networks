from typing import Optional, NamedTuple, List
from enum import Enum, auto

from sim.priors_func import *

## SETUP
class ENetworkType(Enum):
   # The shape of the network (see Zollman 2007)
   COMPLETE = auto()
   CYCLE = auto()

class SkepticalAgentsSetup(NamedTuple):
    n_skeptical: Optional[int]
    # Add n skeptical agents, if set
    min_cr: float
    max_cr: float

class LifeCycleSetup(NamedTuple):
    rounds_to_new_agent: int
    # A new agent will appear and an existing agent will retire every x rounds
    admissions_priors_func: Priors_Func = low_priors
    # This is the prior distribution for newly admitted agents
    # The initial population is configured separately, see below

class ENParams(NamedTuple):
    scientist_init_popcount: int
    network_type: ENetworkType # Controls connectedness of network
    n_per_round: int # How many experiments, or 'coin flips', per round an agent will conduct
    epsilon: float # How much better theory B is in fact. pB = 0.5 + epsilon. pA = 0.5
    low_stop: float # The cr. threshold at which the informative action will no longer be taken
    max_research_rounds: int # When we terminate the simulation, if it has not already stopped
    stable_confidence_threshold: Optional[float]
    # For simulations where we presume a stable outcome, what level of confidence is required
    # of all agents before we conclude that the network has converged on the true view
    m: float
    # m: how distrustful agents are of others' evidence (larger m means more distrustful)
    # See O'Connor and Weatherall 2018, Scientific Polarization
    priors_func: Priors_Func = uniform_priors
    # Controls priors distribution for the initial network
    priorsetup: PriorSetup = PriorSetup()
    # Additional settings for the prior distribution that get passed to the priors func
    lifecyclesetup: Optional[LifeCycleSetup] = None
    # Setup for lifecycle networks with admissions and retirings
    skeptical_agents_setup: Optional[SkepticalAgentsSetup] = None
    # Setup for networks with skeptics

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

    prop_agents_confident_in_true_view: Optional[float] = None
    # Only measured for non-lifecycle sims. And this will only really have an informative
    # value for stable polarization outcomes.
    # If we had a stable consensus outcome, everyone holds the true view confidently.
    # If research was abandoned, then nobody holds the true view.
    # In lifecycle sims, it is not clear what we would be measuring with this. It would
    # be a snapshot, but the result would depend on exactly when the snapshot was taken
    # – e.g. we might get a different result depending on whether it was just before
    # or after a new agent was introduced

    ## TYPE OF AND ROUND OF CONCLUSION
    ## The following three are for non-lifecycle sims only
    consensus_round: Optional[int] = None
    research_abandoned_round: Optional[int] = None
    # Zollman 2007 always exits sim in one of these two ways in practice

    stable_pol_round: Optional[int] = None
    # O'Connor and Weatherall 2018, Scientific Polarization,
    # can additionally exit sim via stable polarization

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
    
class ENResultsSummary(NamedTuple):
    ## METRICS FOR NON-LIFECYCLE SIMS
    # Results taken as the mean value from all sim runs unless otherwise noted

    sims_proportion_consensus_reached: str
    sims_av_consensus_round: str
    sims_proportion_polarization: str
    sims_av_polarization_round: str
    sims_proportion_research_abandoned: str
    sims_av_research_abandonment_round: str
    sims_av_prop_agents_confident_in_true_view: str
    sims_sd_av_prop_agents_confident: str
    sims_unstable_count: str
    # Stable here means we have reached irrevocable
    # consensus on truth, research abandonment or polarization.
    # Theoretically, Zollman 2007, as well as O'Connor and Weatherall 2018
    # can end as an unstable sim. It is just extremely unlikely to happen
    # if there are enough rounds of play. Just in case, this tracks the
    # number of unstable sims.
    # Lifecycle sims are always unstable in the sense described above.
    # Which is not to say that they will not have stable cyclical patterns
    # emerge

    sims_av_total_brier_penalty: str # Av total penalty from all agents over all rounds
    sims_av_brier_ratio: str # Av ratio of obtained total penalty to obtainable penalty
    sims_sd_av_brier_ratio: str # Standard deviation

class ENLifecycleResultsSummary(NamedTuple):
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
    results_summary: ENResultsSummary | ENLifecycleResultsSummary
    # In the output csv, we record both the initial params
    # the sims were run with and the results

class ENResultsCSVWritableSummary(NamedTuple):
    headers: List[str]
    sim_data: List[str]