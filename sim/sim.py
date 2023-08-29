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

        ## VARIOUS SCORING MECHANISMS
        self.max_obtainable_brier_penalty: float = 0
        # Tally the total obtainable brier penalties.
        # I.e. the penalty that would obtain if each round had a maximum
        # penalty
        # It is equal to k * rounds

        self.brier_penalty_total: float = 0
        # Tally the total obtained brier penalties.
        # I.e. the sum of the obtained round penalties.
        # The round penalty is given by the sum of the penalties obtained
        # by all agents in that round
        self.non_skep_obtainable_brier_penalty: float = 0
        self.non_skep_brier_penalty_total: float = 0
        # Same as above, but excludes skeptics

        ## The following stats are all only tracked for agents whose credence is 
        # below .5 i.e. we want to measure the skeptic's influence on those taking 
        # the uninformative action – we want to see whether the skeptic is helping
        # pull them up.
        self.skeptic_credence_influence: float = 0
        # Credence that the skeptic moved
        # through their influence, whichever direction.
        self.skeptic_credence_directional_influence: float = 0
        # + is good
        # Credence moved in the positive direction (toward truth) minus
        # credence moved in the negative direction on account of the skeptic's
        # influence.

        self.skeptic_brier_influence: float = 0
        self.skeptic_total_brier_directional_influence: float = 0 # + is bad
        self.br_ki: float = 0 
        # Sum(skeptic round brier directional influence times no agents w. <.5 credence 
        # for whom influence attempted)
        self.k_skeptic_influenced: int = 0
        # Sum(number of agents w. <.5 credence skeptic attempted to influence in round)
    
    def run_sim(self):
        for i in range(1, self.params.max_research_rounds + 1):
            if self.results:
                break
            self._sim_action(i)
            if i == self.params.max_research_rounds:
                if self.params.lifecyclesetup:
                    self.results = self._lifecycle_results(i)
                    breakpoint()
                    break

                # This should not typically be reached, but it is possible when running
                # Zollman or O'Connor and Weatherall sims with low numbers of rounds
                self.results = self._stability_not_reached_results(i)
                break

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
        self._save_brier_stats()
        # We save the brier penalty even before the first round action.
        # If the game ends at round 1 in a stable sim because everyone
        # came in with such low credences that nobody ever takes the
        # informative action, then we can still calculate a brier score.
        # It will be the total brier score of the agents as they came in.
        # This enables us to penalize networks that
        # fail to transform a mass of agents with low credences.
        # The alternative conceptualization would be that such networks
        # simply fail to apply to these sets of agents.
        # But this would lead to the following counterintuitive result:
        # we could in principle have two configurations such
        # that: say, over 10 simulations, one config gets an average total
        # score of 50. The other configuration also gets a score of 50 but
        # over 8 simulations: two simulation's brier penalty would be
        # N/A because the agents immediately exited the game. But the
        # populations of the latter configuration would still be worse off.

        if self.params.lifecyclesetup:
            self._lifecycle_sim_action()
        elif self.params.stable_confidence_threshold:
            self._static_sim_action(sim_round, self.params.stable_confidence_threshold)
        else:
            raise NotImplementedError(
                "Stable/non-lifecycle sims currently assume stable_confidence_threshold")
        if self.params.skeptical_agents_setup:
            self._save_skeptic_influence_stats()

    # This is used for simulating Zollman 2007, and for simulating O'Connor & Weatherall 2018
    def _static_sim_action(self, sim_round: int, stable_confidence_threshold: float):
        # We are in a config where network membership does not change. Check options for
        # a stable outcome, one by one.
        # NOTE: No skeptics here. And no retired folk
        scientists = self.epistemic_network.scientists
        credences = np.array([s.credence for s in scientists])
        # low-stops AT 0.5
        if all(credences <= self.params.low_stop):
            # Everyone's credence in B is at or below 0.5. Abandon further research
            self.results = self._abandoned_research_results(sim_round)
            return
        # high-stops if ABOVE 0.99 (.99 is the default value in the literature, but
        # this can in principle be configured)
        if all(credences > self.params.stable_confidence_threshold):
            # Everyone's credence in B is above .99. Scientific consensus reached
            self.results = self._consensus_results(sim_round)
            return
        if self._stable_polarization(scientists, stable_confidence_threshold):
            self.results = self._polarization_results(sim_round)
            return
        # Not stable, continue playing
        self.epistemic_network.enetwork_play_round()

    def _lifecycle_sim_action(self):
        self.epistemic_network.enetwork_play_round(lifecycle_sim=True)
    
    def _save_brier_stats(self):
        en = self.epistemic_network
        working_scientists = en.scientists + en.skeptics
        round_briers = [self._brier_score(s.credence) for s in working_scientists]
        self.brier_penalty_total += sum(round_briers)
        round_non_skeptic_briers = [self._brier_score(s.credence) for s in en.scientists]
        self.non_skep_brier_penalty_total += sum(round_non_skeptic_briers)
        self.max_obtainable_brier_penalty += len(working_scientists)
        # This keeps track of the maximum brier penalty from the round, which
        # is given by 1 * n (the maximum penalty from a single agent is 1)
        self.non_skep_obtainable_brier_penalty += len(en.scientists)
        # Same for non-skeptics only
    
    def _save_skeptic_influence_stats(self):
        scientists = self.epistemic_network.scientists
        self.skeptic_credence_influence += sum(
            [a.skeptic_credence_influence for a in scientists])
        self.skeptic_credence_directional_influence += sum(
            [a.skeptic_credence_directional_influence for a in scientists])
        self.skeptic_brier_influence += sum(
            [a.skeptic_brier_influence for a in scientists])
        round_k = sum(
            [1 for s in scientists if s.below_min_and_skeptic_attempted_influence])
        round_br_dir_inf = sum(
            [a.skeptic_brier_directional_influence for a in scientists])
        self.skeptic_total_brier_directional_influence += round_br_dir_inf
        self.k_skeptic_influenced += round_k
        self.br_ki += round_br_dir_inf * round_k
    
    def _skeptic_av_brier_directional_influence(self, br_ki: float, ki: float) -> float:
        return br_ki / ki

    def _stable_polarization(self, scientists: list[Scientist], confidence_threshold: float) -> bool:
        credences = np.array([s.credence for s in scientists])
        if all(credences > confidence_threshold):
            return False
        if all(credences <= self.params.low_stop):
            return False
        
        # We return True by default. The loop finds conditions under
        # which we are NOT stably polarized.
        # We are not stably polarized if someone is taking the informative action
        # and is under the confidence threshold.
        # We are not stably polarized if someone is taking the informative action,
        # under or over the confidence threshold, and someone else who is not taking
        # the informative action is within their sphere of influence
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
            high_rollers = [s for s in scientists if s.credence > confidence_threshold]
            for high_roller in high_rollers:
                if scientist.dm(high_roller) < 1:
                    # Someone below 0.5 is still influenced by those at .99, and can still be
                    # pulled up.
                    return False
        return True

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
    
    def _prop_truth_confidently(self, agents: list[Scientist]) -> float:
        # TODO: Parametrize (but note that conceptually
        # this is not necessarily the same as stable_confidence_threshold)
        return float(np.mean([s.credence > 0.99 for s in agents]))

    ## RESULTS
    def _abandoned_research_results(self, sim_round: int) -> ENSingleSimResults:
        return ENSingleSimResults(
            sim_brier_penalty_total=self.brier_penalty_total,
            sim_brier_penalty_ratio=self._brier_ratio(),
            prop_agents_confident_in_true_view=0,
            research_abandoned_round=sim_round)
    
    def _consensus_results(self, sim_round: int) -> ENSingleSimResults:
        return ENSingleSimResults(
            sim_brier_penalty_total=self.brier_penalty_total,
            sim_brier_penalty_ratio=self._brier_ratio(),
            prop_agents_confident_in_true_view=1,
            consensus_round=sim_round)
    
    def _polarization_results(self, sim_round: int) -> ENSingleSimResults:
        scientists = self.epistemic_network.scientists
        return ENSingleSimResults(
            sim_brier_penalty_total=self.brier_penalty_total,
            sim_brier_penalty_ratio=self._brier_ratio(),
            prop_agents_confident_in_true_view=self._prop_truth_confidently(scientists),
            stable_pol_round=sim_round)
    
    def _lifecycle_results(self, sim_round: int) -> ENSingleSimResults:
        en = self.epistemic_network
        retired = en.retired
        working = en.scientists + en.skeptics

        nsbr = self._non_skeptic_brier_ratio() if self.params.skeptical_agents_setup else None
        av_retired_brier_penalty = self._mean_brier_score([s.credence for s in retired])
        retired_confidently = self._prop_truth_confidently(retired)

        print(f"skeptic_credence_influence: {self.skeptic_credence_influence}")
        print(f"skeptic_credence_directional_influence: {self.skeptic_credence_directional_influence}")
        print(f"skeptic_brier_influence: {self.skeptic_brier_influence}")
        print(f"skeptic_total_brier_directional_influence: {self.skeptic_total_brier_directional_influence}")
        print(f"Skeptic influenced agent-rounds: {self.k_skeptic_influenced}")
        # Keep in mind k is agent-rounds, not agents
        print()

        skeptic_av_credence_influence = self.skeptic_credence_influence \
            / self.k_skeptic_influenced
        print(f"skeptic_av_credence_influence: {skeptic_av_credence_influence}")
        skeptic_av_credence_directional_influence = self.skeptic_credence_directional_influence \
            / self.k_skeptic_influenced
        print(f"skeptic_av_credence_directional_influence: {skeptic_av_credence_directional_influence}")
        skeptic_av_brier_influence = self.skeptic_brier_influence \
            / self.k_skeptic_influenced
        print(f"skeptic_av_brier_influence: {skeptic_av_brier_influence}")
        skeptic_av_brier_directional_influence = self.skeptic_total_brier_directional_influence \
            / self.k_skeptic_influenced
        print(f"skeptic_av_brier_directional_influence calc 1: {skeptic_av_brier_directional_influence}")
        print(f"skeptic_av_brier_directional_influence calc 2: {self._skeptic_av_brier_directional_influence(self.br_ki, self.k_skeptic_influenced)}")
        agents = retired + working
        av_skeptic_lifetime_brier_directional_influence = float(np.mean([
            a.skeptic_lifetime_brier_directional_influence for a in agents if a.skeptic_lifetime_brier_directional_influence is not None]))
        print(f"av_skeptic_lifetime_brier_directional_influence: {av_skeptic_lifetime_brier_directional_influence}")
        breakpoint()
        return ENSingleSimResults(
            sim_brier_penalty_total=self.brier_penalty_total,
            sim_brier_penalty_ratio=self._brier_ratio(),
            sim_non_skeptic_brier_ratio=nsbr,
            av_retired_brier_penalty=av_retired_brier_penalty,
            prop_retired_confident=retired_confidently,
            unstable_conclusion_round=sim_round,
            skeptic_credence_influence=self.skeptic_credence_influence,
            skeptic_credence_directional_influence=self.skeptic_credence_directional_influence,
            skeptic_brier_influence=self.skeptic_brier_influence,
            skeptic_total_brier_directional_influence=self.skeptic_total_brier_directional_influence,
            skeptic_av_brier_directional_influence=self._skeptic_av_brier_directional_influence(self.br_ki, self.k_skeptic_influenced),
            n_all_agents=len(working + retired),)
    
    def _stability_not_reached_results(self, sim_round: int) -> ENSingleSimResults:
        """ Unstable results for a sim that would typically be expected to reach
        a stable state (research abandoment, polarization or consensus on truth).
        This is rare but can happen if the number of maximum rounds is small."""
        en = self.epistemic_network
        scientists = en.scientists
        return ENSingleSimResults(
            sim_brier_penalty_total=self.brier_penalty_total,
            sim_brier_penalty_ratio=self._brier_ratio(),
            prop_agents_confident_in_true_view=self._prop_truth_confidently(scientists),
            unstable_conclusion_round=sim_round)
    