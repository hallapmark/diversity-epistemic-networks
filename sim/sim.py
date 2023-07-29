from agents.scientist import Scientist
from network.network import ENetwork
import numpy as np
from typing import Optional
from sim.sim_models import ENSimulationRawResults

class EpistemicNetworkSimulation():

    def __init__(self,
                 epistemic_network: ENetwork,
                 maxrounds: int,
                 low_stop: float,
                 stable_confidence_threshold: float):
        self.epistemic_network = epistemic_network
        self._low_stop = low_stop
        self._maxrounds = maxrounds
        self.stable_confidence_threshold = stable_confidence_threshold
        self._sim_round = 0
        self.results: Optional[ENSimulationRawResults] = None
    
    def run_sim(self):
        for i in range(1, self._maxrounds + 1):
            if self.results:
                break
            self._sim_action(i)
            if i == self._maxrounds:
                prop_truth_confidently = self._prop_truth_confidently()
                self.results = ENSimulationRawResults(None, None, None, i, prop_truth_confidently)
    
    def _prop_truth_confidently(self):
        en = self.epistemic_network
        all_scientists = en.scientists# + en.intransigent_scientists
        # TODO: Parametrize?
        return float(np.mean([s.credence > 0.99 for s in all_scientists]))

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
        if self.results:
            return
        self._sim_round = sim_round
        scientists = self.epistemic_network.scientists
        credences = np.array([s.credence for s in scientists])
        # low-stops AT 0.5.
        if all(credences <= self._low_stop):
            # Everyone's credence in B is at or below 0.5. Abandon further research
            prop_truth_confidently = self._prop_truth_confidently()
            self.results = ENSimulationRawResults(None, sim_round, None, None, prop_truth_confidently)
            return
         # high-stops if ABOVE 0.99
        if all(credences > self.stable_confidence_threshold):
            # Everyone's credence in B is above .99. Scientific consensus reached
            prop_truth_confidently = self._prop_truth_confidently()
            self.results = ENSimulationRawResults(sim_round,
                                                  None,
                                                  None,
                                                  None,
                                                  prop_truth_confidently)
            return
        if self._stable_polarization(scientists, self.stable_confidence_threshold):
            prop_truth_confidently = self._prop_truth_confidently()
            self.results = ENSimulationRawResults(None,
                                                  None,
                                                  sim_round,
                                                  None,
                                                  prop_truth_confidently)
            return
        self.epistemic_network.enetwork_play_round()
