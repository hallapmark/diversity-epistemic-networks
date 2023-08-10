from agents.scientist import Scientist
from network.network import ENetwork
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
        self.max_possible_brier_penalty: float = 0
        # Tally the total brier penalties from each round
        self.brier_penalty_total: float = 0
        self.non_skeptic_max_possible_brier_penalty: float = 0
        self.non_skeptic_brier_penalty_total: float = 0
    
    def run_sim(self):
        for i in range(1, self.params.max_research_rounds + 1):
            if self.results:
                break
            self._sim_action(i)
            if i == self.params.max_research_rounds:
                en = self.epistemic_network
                # In non-lifecycle sims, the latter two will be empty
                all_scientists = en.scientists + en.skeptics + en.retired
                working_scientists = en.scientists + en.skeptics
                all_conf = self._prop_truth_confidently(all_scientists)
                n_all_agents = len(all_scientists)
                brier_ratio = self.brier_penalty_total / self.max_possible_brier_penalty
                non_skeptic_brier_ratio = self.non_skeptic_brier_penalty_total / self.non_skeptic_max_possible_brier_penalty
                game_exit_snapshot_brier = self._round_mean_brier_score([s.credence for s in working_scientists])
                if self.params.lifecyclesetup:
                    working_conf = self._prop_truth_confidently(working_scientists)
                    retired_conf = self._prop_truth_confidently(en.retired)
                    self.results = ENSingleSimResults(
                        consensus_round=None,
                        research_abandoned_round=None,
                        stable_pol_round=None,
                        unstable_conclusion_round=i,
                        prop_agents_confident_in_true_view=all_conf,
                        sim_brier_penalty_total=self.brier_penalty_total,
                        sim_brier_penalty_ratio_to_max=brier_ratio,
                        sim_non_skeptic_brier_ratio=non_skeptic_brier_ratio,
                        sim_game_exit_snapshot_brier=game_exit_snapshot_brier,
                        prop_retired_confident=retired_conf,
                        prop_working_confident=working_conf,
                        n_all_agents=n_all_agents)
                    break
                self.results = ENSingleSimResults(
                    consensus_round=None,
                    research_abandoned_round=None,
                    stable_pol_round=None,
                    unstable_conclusion_round=i,
                    prop_agents_confident_in_true_view=all_conf,
                    sim_brier_penalty_total=self.brier_penalty_total,
                    sim_brier_penalty_ratio_to_max=brier_ratio,
                    sim_non_skeptic_brier_ratio=non_skeptic_brier_ratio,
                    sim_game_exit_snapshot_brier=game_exit_snapshot_brier)
                break
    
    def _round_mean_brier_score(self, credences: list[float]) -> float:
        #e.g. H1: 0.6 is true value, H2: it is  .4.
        # In fact, H1 is true
        # E.g. My credence is .8
        # ((.8-1) + (1-.8))**2 -> (.2 + .2)**2 = .16
        # 1/n * .16 = 0.08
        return float(np.mean([self._brier_score(cr) for cr in credences]))

    def _brier_score(self, credence: float) -> float:
        return (credence - 1)**2

    def _prop_truth_confidently(self, agents: list[Scientist]) -> float:
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
        # if sim_round % 500 == 0:
        #     print(f"A sim has reached round {sim_round}")
        #     for scientist in self.epistemic_network.scientists:
        #         print(scientist)
        #     print("Skeptic:")
        #     for scientist in self.epistemic_network.skeptics:
        #         print(scientist)
        if self.results:
            return
        self._sim_round = sim_round
        # Check if we are in a config where we do not presume a stable outcome.
        if not self.params.stable_confidence_threshold:
            self._unstable_sim_action()
        else:
            self._stable_sim_action(sim_round, self.params.stable_confidence_threshold)
        en = self.epistemic_network
        working_scientists = en.scientists + en.skeptics
        round_briers = [self._brier_score(s.credence) for s in working_scientists]
        self.brier_penalty_total += sum(round_briers)
        round_non_skeptic_briers = [self._brier_score(s.credence) for s in en.scientists]
        self.non_skeptic_brier_penalty_total += sum(round_non_skeptic_briers)
        n = len(working_scientists)
        # This keeps track of the maximum brier penalty from the round, which
        # is given by 1 * n (the maximum penalty from a single agent is 1)
        self.max_possible_brier_penalty += n
        self.non_skeptic_max_possible_brier_penalty += len(en.scientists)

    def _stable_sim_action(self, sim_round: int, stable_confidence_threshold: float):
        # We are in a config where we are looking for a stable outcome. Check options for
        # a stable outcome, one by one.
        # TODO: Make this a func. self.stable_sim_action()
        scientists = self.epistemic_network.scientists
        credences = np.array([s.credence for s in scientists])
        game_exit_snapshot_brier = self._round_mean_brier_score([s.credence for s in scientists])
        # low-stops AT 0.5.
        if all(credences <= self.params.low_stop):
            # Everyone's credence in B is at or below 0.5. Abandon further research
            prop_truth_confidently = self._prop_truth_confidently(scientists)
            brier_ratio = self.brier_penalty_total / self.max_possible_brier_penalty
            non_skeptic_brier_ratio = self.non_skeptic_brier_penalty_total / self.non_skeptic_max_possible_brier_penalty
            self.results = ENSingleSimResults(
                consensus_round=None,
                research_abandoned_round=sim_round,
                stable_pol_round=None,
                unstable_conclusion_round=None,
                prop_agents_confident_in_true_view=prop_truth_confidently,
                sim_brier_penalty_total=self.brier_penalty_total,
                sim_brier_penalty_ratio_to_max=brier_ratio,
                sim_non_skeptic_brier_ratio=non_skeptic_brier_ratio,
                sim_game_exit_snapshot_brier=game_exit_snapshot_brier)
            return
         # high-stops if ABOVE 0.99
        if all(credences > self.params.stable_confidence_threshold):
            # Everyone's credence in B is above .99. Scientific consensus reached
            prop_truth_confidently = self._prop_truth_confidently(scientists)
            brier_ratio = self.brier_penalty_total / self.max_possible_brier_penalty
            non_skeptic_brier_ratio = self.non_skeptic_brier_penalty_total / self.non_skeptic_max_possible_brier_penalty
            self.results = ENSingleSimResults(
                consensus_round=sim_round,
                research_abandoned_round=None,
                stable_pol_round=None,
                unstable_conclusion_round=None,
                prop_agents_confident_in_true_view=prop_truth_confidently,
                sim_brier_penalty_total=self.brier_penalty_total,
                sim_brier_penalty_ratio_to_max=brier_ratio,
                sim_non_skeptic_brier_ratio=non_skeptic_brier_ratio,
                sim_game_exit_snapshot_brier=game_exit_snapshot_brier)
            return
        if self._stable_polarization(scientists, stable_confidence_threshold):
            prop_truth_confidently = self._prop_truth_confidently(scientists)
            brier_ratio = self.brier_penalty_total / self.max_possible_brier_penalty
            non_skeptic_brier_ratio = self.non_skeptic_brier_penalty_total / self.non_skeptic_max_possible_brier_penalty
            self.results = ENSingleSimResults(
                consensus_round=None,
                research_abandoned_round=None,
                stable_pol_round=sim_round,
                unstable_conclusion_round=None,
                prop_agents_confident_in_true_view=prop_truth_confidently,
                sim_brier_penalty_total=self.brier_penalty_total,
                sim_brier_penalty_ratio_to_max=brier_ratio,
                sim_non_skeptic_brier_ratio=non_skeptic_brier_ratio,
                sim_game_exit_snapshot_brier=game_exit_snapshot_brier)
            return
        self.epistemic_network.enetwork_play_round()

    def _unstable_sim_action(self):
        self.epistemic_network.enetwork_play_round(unstable_sim=True)

