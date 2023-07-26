from typing import Optional, NamedTuple, List
from enum import Enum, auto

from sim.priors_func import *

class ENetworkType(Enum):
   COMPLETE = auto()
   CYCLE = auto()

class ENParams(NamedTuple):
    scientist_popcount: int
    network_type: ENetworkType
    n_per_round: int
    epsilon: float
    low_stop: float # The threshold at which the informative action will no longer be taken
    max_research_rounds: int
    consensus_threshold: float
    m: float
    priors_func: Priors_Func = uniform_priors

class ENSimulationRawResults(NamedTuple):
    consensus_round: Optional[int]
    research_abandoned_round: Optional[int]
    stable_pol_round: Optional[int]
    
class ENResultsSummary(NamedTuple):
    sims_proportion_consensus_reached: str
    sims_avg_consensus_round: str
    sims_proportion_polarization: str
    sims_avg_polarization_round: str
    sims_proportion_research_abandoned: str
    sims_avg_research_abandonment_round: str
    

class ENSimsSummary(NamedTuple):
    params: ENParams
    results_summary: ENResultsSummary

class ENResultsCSVWritableSummary(NamedTuple):
    headers: List[str]
    sim_data: List[str]