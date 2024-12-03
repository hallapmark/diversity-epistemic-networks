from sim.scientist import Scientist
from sim.network import ENetwork
import numpy as np
from typing import Optional
from sim.sim_models import ENParams, ENSingleSimResults

class EpistemicNetworkSimulation():

    def __init__(self,
                 epistemic_network: ENetwork,
                 params: ENParams):
        self.epistemic_network = epistemic_network
        self.params = params
        self._sim_round = 0
        self.results: Optional[ENSingleSimResults] = None
        self.max_obtainable_brier_penalty: float = 0
        # Tally the total obtainable brier penalties.
        # I.e. the penalty that would obtain if each round had a maximum
        # penalty
        self.brier_penalty_total: float = 0
        # Tally the total obtained brier penalties.
        # I.e. the sum of the obtained round penalties.
        # The round penalty is given by the sum of the penalties obtained
        # by all agents in that round
        self.non_skep_obtainable_brier_penalty: float = 0
        self.non_skep_brier_penalty_total: float = 0
        # Same as above, but excludes skeptics
    
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
        self._save_brier_stats()

        if self.params.lifecyclesetup:
            self._lifecycle_sim_action()
        else:
            raise NotImplementedError(
                "params.lifecyclesetup missing")

    def _lifecycle_sim_action(self):
        self.epistemic_network.enetwork_play_round(lifecycle_sim=True)
    
    def _save_brier_stats(self):
        en = self.epistemic_network
        round_briers = [self._brier_score(s.credence) for s in en.scientists]
        self.brier_penalty_total += sum(round_briers)
        round_non_skeptic_briers = [self._brier_score(s.credence) for s in en.scientists if not s.is_skeptic]
        self.non_skep_brier_penalty_total += sum(round_non_skeptic_briers)
        self.max_obtainable_brier_penalty += len(en.scientists)
        # This keeps track of the maximum brier penalty from the round, which
        # is given by 1 * n (the maximum penalty from a single agent is 1)
        self.non_skep_obtainable_brier_penalty += len([s for s in en.scientists if not s.is_skeptic])
        # Same for non-skeptics only

    ## CALCULATION HELPERS
    def _brier_score(self, credence: float) -> float:
        # Credence is the credence in the true view, theory B: pB = 0.5 + e
        """ We calculate the brier score not on predictions but on
        the credences to measure distance from truth."""
        return (credence - 1)**2
    
    def _mean_brier_score(self, credences: list[float]) -> float:
        return float(np.mean([self._brier_score(cr) for cr in credences]))
    
    def _brier_ratio(self) -> float:
        return self.brier_penalty_total / self.max_obtainable_brier_penalty

    def _non_skeptic_brier_ratio(self) -> float:
        return self.non_skep_brier_penalty_total / self.non_skep_obtainable_brier_penalty
    
    def _prop_truth_confidently(self, credences: list[float]) -> float:
        return float(np.mean([cr > 0.99 for cr in credences]))
    
    def _lifecycle_results(self, sim_round: int) -> ENSingleSimResults:
        en = self.epistemic_network
        working = en.scientists

        nsbr = self._non_skeptic_brier_ratio() if self.params.skeptic_n > 0 else None
        av_retired_brier_penalty = self._mean_brier_score(en.retiree_credences)
        retired_confidently = self._prop_truth_confidently(en.retiree_credences)
        return ENSingleSimResults(
            sim_brier_penalty_total=self.brier_penalty_total,
            sim_brier_penalty_ratio=self._brier_ratio(),
            sim_non_skeptic_brier_ratio=nsbr,
            av_retired_brier_penalty=av_retired_brier_penalty,
            prop_retired_confident=retired_confidently,
            unstable_conclusion_round=sim_round,
            n_all_agents=len(working + en.retiree_credences))
    