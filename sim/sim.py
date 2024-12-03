from sim.scientist import Scientist
from sim.network import ENetwork
import numpy as np
from typing import Optional
from sim.sim_models import ENParams, ENSingleSimResults

class ENSimulation():
    def __init__(self,
                 epistemic_network: ENetwork,
                 params: ENParams):
        self.epistemic_network = epistemic_network
        self.params = params
        self._sim_round = 0
        self.results: Optional[ENSingleSimResults] = None
        self.metrics = SimMetrics()
    
    def run_sim(self):
        for i in range(1, self.params.max_research_rounds + 1):
            if self.results:
                break
            self._sim_action(i)
            if i == self.params.max_research_rounds:
                self.results = self._lifecycle_results(i)

    def _sim_action(self, sim_round: int):
        if self.results:
            return
        self._sim_round = sim_round
        self.epistemic_network.enetwork_play_round()
        self.metrics.update_brier_stats(self)
    
    def _lifecycle_results(self, sim_round: int) -> ENSingleSimResults:
        en = self.epistemic_network
        working = en.scientists

        nsbr = self.metrics.non_skeptic_brier_ratio() if self.params.skeptic_n > 0 else None
        av_retired_brier_penalty = self.metrics.mean_brier_score(en.retiree_credences)
        retired_confidently = self.metrics.prop_truth_confidently(en.retiree_credences)
        return ENSingleSimResults(
            sim_brier_penalty_total=self.metrics.brier_penalty_total,
            sim_brier_penalty_ratio=self.metrics.brier_ratio(),
            sim_non_skeptic_brier_ratio=nsbr,
            av_retired_brier_penalty=av_retired_brier_penalty,
            prop_retired_confident=retired_confidently,
            unstable_conclusion_round=sim_round,
            n_all_agents=len(working + en.retiree_credences))

class SimMetrics:
        def __init__(self):
            self.max_obtainable_brier_penalty: float = 0
            self.brier_penalty_total: float = 0
            # Tally the total obtained brier penalties. I.e. the sum of the obtained 
            # round penalties. The round penalty is given by the sum of the penalties obtained
            # by all agents in that round
            
            self.non_skep_obtainable_brier_penalty: float = 0
            # Tally the total obtainable brier penalties. I.e. the penalty that would 
            # obtain if each round everyone had a maximum penalty

            self.non_skep_brier_penalty_total: float = 0
            # Same as above, but excludes skeptics

        def update_brier_stats(self, simulation: ENSimulation):
            en = simulation.epistemic_network
            round_briers = [self.brier_score(s.credence) for s in en.scientists]
            self.brier_penalty_total += sum(round_briers)
            round_non_skeptic_briers = [self.brier_score(s.credence) for s in en.scientists if not s.is_skeptic]
            self.non_skep_brier_penalty_total += sum(round_non_skeptic_briers)
            self.max_obtainable_brier_penalty += len(en.scientists)
            self.non_skep_obtainable_brier_penalty += len([s for s in en.scientists if not s.is_skeptic])

        def brier_score(self, credence: float) -> float:
            return (credence - 1)**2

        def mean_brier_score(self, credences: list[float]) -> float:
            return float(np.mean([self.brier_score(cr) for cr in credences]))

        def brier_ratio(self) -> float:
            return self.brier_penalty_total / self.max_obtainable_brier_penalty

        def non_skeptic_brier_ratio(self) -> float:
            return self.non_skep_brier_penalty_total / self.non_skep_obtainable_brier_penalty

        def prop_truth_confidently(self, credences: list[float]) -> float:
            return float(np.mean([cr > 0.99 for cr in credences]))