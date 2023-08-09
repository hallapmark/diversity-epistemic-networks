from typing import Optional, NamedTuple, List
from enum import Enum, auto

from sim.priors_func import *

class ENetworkType(Enum):
   COMPLETE = auto()
   CYCLE = auto()

class SkepticalAgentsSetup(NamedTuple):
    # Add n skeptical agents
    n_skeptical: Optional[int]
    min_cr: float
    max_cr: float

class LifeCycleSetup(NamedTuple):
    admissions_priors_func: Priors_Func = low_priors

class ENParams(NamedTuple):
    scientist_init_popcount: int
    network_type: ENetworkType
    n_per_round: int
    epsilon: float # How much better theory B is. pB = 0.5 + epsilon
    low_stop: float # The threshold at which the informative action will no longer be taken
    max_research_rounds: int # When we terminate the simulation, if it has not already stopped
    # For simulations where we presume a stable outcome, what level of confidence is required
    stable_confidence_threshold: Optional[float]
    m: float
    priors_func: Priors_Func = uniform_priors
    priorsetup: PriorSetup = PriorSetup()
    lifecyclesetup: Optional[LifeCycleSetup] = None
    skeptical_agents_setup: Optional[SkepticalAgentsSetup] = None

class ENSingleSimResults(NamedTuple):
    consensus_round: Optional[int]
    research_abandoned_round: Optional[int]
    stable_pol_round: Optional[int]
    unstable_conclusion_round: Optional[int]
    # Final round for an unstable network (where we left off, manual max rounds threshold reached)
    prop_agents_confident_in_true_view: float
    sim_brier_penalty_total: float
    sim_brier_penalty_ratio_to_max: float
    # The ratio of the brier penalty obtained to the maximum possible penalty
    # Maximum penalty would be obtained if all agents were maximally distant
    # from the truth every round of the game
    sim_game_exit_snapshot_brier: float
    # Mean brier score of all agents at the point of exit from the game
    # An agent exits if they retire
    # An agent exits if the game is over
    prop_retired_confident: Optional[float] = None
    prop_working_confident: Optional[float] = None
    n_all_agents: Optional[float] = None
    
class ENResultsSummary(NamedTuple):
    sims_proportion_consensus_reached: str
    sims_av_consensus_round: str
    sims_proportion_polarization: str
    sims_av_polarization_round: str
    sims_proportion_research_abandoned: str
    sims_av_research_abandonment_round: str
    sims_unstable_count: str
    av_prop_agents_confident_in_true_view: str
    sd: str
    cv: str
    sims_mean_brier_penalty_total: str
    sims_mean_brier_penalty_ratio_to_max: str
    sims_mean_snapshot_brier: str
    av_n_all_agents: str = "N/A"
    av_prop_working_confident: str = "N/A"
    av_prop_retired_confident: str = "N/A"
    
class ENSimsSummary(NamedTuple):
    params: ENParams
    results_summary: ENResultsSummary

class ENResultsCSVWritableSummary(NamedTuple):
    headers: List[str]
    sim_data: List[str]