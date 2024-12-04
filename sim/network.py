from sim.scientist import Scientist
from sim.sim_models import *
import numpy as np
from typing import List

class ENetwork():
    def __init__(self,
                 rng: np.random.Generator,
                 params: ENParams):
        self.rng = rng
        self.params = params
        priors = params.priors_func(params.scientist_init_popcount, rng)
        self._rounds_played = 0
        self.scientists = [Scientist(prior, params, rng, False) for prior in priors]
        if not len(self.scientists) == params.scientist_init_popcount:
            raise ValueError(
                "Something went wrong. !(len(self.scientists) == scientist_popcount)")
        for _ in range(params.skeptic_n):
            prior = .5
            non_skeptics = [s for s in self.scientists if not s.is_skeptic]
            skeptic_to_become: Scientist = np.random.choice(non_skeptics)
            skeptic_to_become.__init__(prior, params, rng, True)
        for s in self.scientists:
            # We assume that the network we start off with has some experience
            s.rounds_of_experience = 20 
        self._structure_scientific_network(self.scientists)
        self.retiree_credences: list[float] = []

    ## Init helpers
    def _structure_scientific_network(self, scientists: List[Scientist]):
        for scientist in scientists:
            scientist.influencers = []
            indices = np.arange(len(scientists))
            self.rng.shuffle(indices)
            shuffled_influencers: list[Scientist] = []
            for i in indices:
                shuffled_influencers.append(scientists[i])
            self._add_all_influencers_for_updater(scientist, shuffled_influencers)

    ## Interface
    def enetwork_play_round(self):
        self._standard_round_actions()
        self._lifecycle_round_actions()
        self._rounds_played += 1
        
    ## Private methods
    def _standard_round_actions(self):
        for scientist in self.scientists:
            scientist.round_binomial_experiment = None # reset to None before new round starts
            # Whether 'tis nobler to experiment
            scientist.decide_round_research_action()
        for scientist in self.scientists:
            scientist.jeffrey_update_credence()
        for scientist in self.scientists:
            scientist.rounds_of_experience += 1

    def _lifecycle_round_actions(self):
        """ Every x rounds, a scientist exits and a new one enters """
        if not self.params.lifecyclesetup:
            raise NotImplementedError("Lifecycle actions requested but lifecyclesetup missing from params.")
        if not self._rounds_played % self.params.lifecyclesetup.rounds_to_new_agent == 0:
            return
        
        # Do not retire if no experienced scientist is found
        experienced_scientists = [s for s in self.scientists if s.rounds_of_experience >= 20]
        if not experienced_scientists:
            return
        
        params = self.params
        retiree: Scientist = np.random.choice(experienced_scientists)
        if not retiree:
            raise ValueError("We should not reach this. No retiree agent found.")
        if retiree.is_skeptic:
            prior = .5
        else:
            prior = params.lifecyclesetup.admissions_priors_func(1, self.rng)[0]
        # re-initialize retiree to new agent
        self.retiree_credences.append(retiree.credence)
        retiree.__init__(prior, params, self.rng, retiree.is_skeptic)
        self._structure_scientific_network(self.scientists)
        # Idea for future:
        # Conversion from incentive structure
        # if approx_consensus_reached:
        #     # Convert a non-radical scientist to a contrarian to the emerging consensus
        #     non_radicals = [s for s in scientists if s.credence > 0.2]
        #     indices: np.ndarray = np.arange(len(non_radicals))
        #     s: Scientist = non_radicals[self.rng.choice(indices)]
        #     s.credence = self.rng.uniform(UNIFORM_LOW, 0.2)
        #     intransigent_scientists.append(s)
        #     scientists.remove(s)

    def _add_all_influencers_for_updater(self,
                                         updater: Scientist,
                                         influencers: List[Scientist]):
        for influencer in influencers:
            updater.add_jeffrey_influencer(influencer)
