from agents.scientist import Scientist
from network.network import ENetwork
import numpy as np
from typing import Optional
from sim.sim_models import ENParams, ENSimulationRawResults

class EpistemicNetworkSimulation():

    def __init__(self,
                 epistemic_network: ENetwork,
                 params: ENParams):
        self.epistemic_network = epistemic_network
        self.params = params
        self._rounds_played = 0
        self.results: Optional[ENSimulationRawResults] = None
    
    def run_sim(self):
        for i in range(1, self.params.max_research_rounds + 1):
            if self.results:
                break
            self._sim_action(i)
            if i == self.params.max_research_rounds:
                en = self.epistemic_network
                all_conf = self._prop_truth_confidently(en.scientists + en.retired)
                if self.params.lifecycle:
                    working_conf = self._prop_truth_confidently(en.scientists)
                    retired_conf = self._prop_truth_confidently(en.retired)
                    self.results = ENSimulationRawResults(
                        consensus_round=None,
                        research_abandoned_round=None,
                        stable_pol_round=None,
                        unstable_conclusion_round=i,
                        prop_agents_confident_in_true_view=all_conf,
                        prop_retired_confident=retired_conf,
                        prop_working_confident=working_conf)
                    break
                self.results = ENSimulationRawResults(None, None, None, i, all_conf)
    
    def _prop_truth_confidently(self, agents: list[Scientist]):
        # TODO: Parametrize?
        return float(np.mean([s.credence > 0.99 for s in agents]))

    def _stable_polarization(self, scientists: list[Scientist], confidence_threshold: float) -> bool:
        # We return True by default. The loop finds conditions under
        # which we are NOT stably polarized
        for scientist in scientists:
            if scientist.credence > confidence_threshold:
                continue
            if scientist.low_stop < scientist.credence <= confidence_threshold:
                # Someone is taking the informative action and is not yet at 0.99. Network not 
                # stable
                return False
            # Scientist's credence is below 0.5. Check if scientist's d * m >= 1 with everyone
            # whose credence is over .99. Otherwise, the scientist will be in their sphere of
            # influence, and can still be pulled up.
            high_rollers = [s for s in scientists if s.credence > 0.99]
            for high_roller in high_rollers:
                if scientist.dm(high_roller) < 1:
                    # Someone below 0.5 is still influenced by those at .99, and can still be 
                    # pulled up.
                    return False
        return True

    def _sim_action(self, sim_round: int):
        # if sim_round % 300 == 0:
        #     print(f"A sim has reached round {sim_round}")
        if self.results:
            return
        self._rounds_played = sim_round
        # Check if we are in a config where we do not presume a stable outcome.
        if not self.params.stable_confidence_threshold:
            self.unstable_sim_action()
            return

        # We are in a config where we are looking for a stable outcome. Check options for
        # a stable outcome, one by one.
        # TODO: Make this a func. self.stable_sim_action()
        scientists = self.epistemic_network.scientists
        credences = np.array([s.credence for s in scientists])
        # low-stops AT 0.5.
        if all(credences <= self.params.low_stop):
            # Everyone's credence in B is at or below 0.5. Abandon further research
            prop_truth_confidently = self._prop_truth_confidently(scientists)
            self.results = ENSimulationRawResults(None, sim_round, None, None, prop_truth_confidently)
            return
         # high-stops if ABOVE 0.99
        if all(credences > self.params.stable_confidence_threshold):
            # Everyone's credence in B is above .99. Scientific consensus reached
            prop_truth_confidently = self._prop_truth_confidently(scientists)
            self.results = ENSimulationRawResults(sim_round,
                                                  None,
                                                  None,
                                                  None,
                                                  prop_truth_confidently)
            return
        if self._stable_polarization(scientists, self.params.stable_confidence_threshold):
            prop_truth_confidently = self._prop_truth_confidently(scientists)
            self.results = ENSimulationRawResults(None,
                                                  None,
                                                  sim_round,
                                                  None,
                                                  prop_truth_confidently)
            return
        self.epistemic_network.enetwork_play_round()

    def unstable_sim_action(self):
        self.epistemic_network.enetwork_play_round(unstable_sim=True)

