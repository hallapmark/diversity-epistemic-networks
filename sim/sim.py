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
                 full_confidence_threshold: float):
        self.epistemic_network = epistemic_network
        self._low_stop = low_stop
        self._maxrounds = maxrounds
        self.full_confidence_threshold = full_confidence_threshold
        self._sim_round = 0
        self.results: Optional[ENSimulationRawResults] = None
    
    def run_sim(self):
        for i in range(1, self._maxrounds + 1):
            if self.results:
                break
            self._sim_action(i)
    
    def _stable_polarization(self, scientists: list[Scientist]) -> bool:
        # We return True by default. The loop finds conditions under
        # which we are NOT stably polarized
        for scientist in scientists:
            if scientist.credence > self.full_confidence_threshold:
                continue
            if scientist.low_stop < scientist.credence <= self.full_confidence_threshold:
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
            self.results = ENSimulationRawResults(None, sim_round, None)
            return
         # high-stops if ABOVE 0.99
        if all(credences > self.full_confidence_threshold):
            # Everyone's credence in B is above .99. Scientific consensus reached
            self.results = ENSimulationRawResults(sim_round,
                                                  None,
                                                  None)
            return
        if self._stable_polarization(scientists):
            self.results = ENSimulationRawResults(None,
                                                  None,
                                                  sim_round)
            return
        self.epistemic_network.enetwork_play_round()
