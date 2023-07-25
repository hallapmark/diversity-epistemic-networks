from typing import Optional, NamedTuple,  List
from enum import Enum, auto

class ENetworkType(Enum):
   COMPLETE = auto()
   CYCLE = auto()

class ENParams(NamedTuple):
    scientist_pop_count: int 
    network_type: ENetworkType
    binom_n_per_round: int
    epsilon: float
    scientist_stop_threshold: float
    max_research_rounds_allowed: int
    consensus_threshold: float
    m: float

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