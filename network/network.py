import random
from agents.scientist import Scientist
from agents.updaters.jeffreyupdater import JeffreyUpdater
from sim.sim_models import *
import numpy as np
from typing import List

class ENetwork():
    def __init__(self,
                 rng: np.random.Generator,
                 scientist_popcount: int,
                 scientist_network_type: ENetworkType,
                 n_per_round: int,
                 epsilon: float,
                 low_stop: float,
                 m: float,
                 priors_func: Priors_Func,
                 priorsetup: PriorSetup):
        self.scientist_popcount = scientist_popcount
        self.scientist_network_type = scientist_network_type
        priors = priors_func(scientist_popcount, rng, priorsetup)
        self.rng = rng
        self.n_per_round = n_per_round
        self.epsilon = epsilon
        self.low_stop = low_stop
        self.m = m
        self.scientists = [Scientist(
            rng,
            n_per_round,
            epsilon,
            low_stop,
            prior,
            m
            ) for prior in priors]
        if not len(self.scientists) == scientist_popcount:
            raise ValueError(
                "Something went wrong. !(len(self.scientists) == scientist_popcount)")
        self.retired: list[Scientist] = []
        self._structure_scientific_network(self.scientists, scientist_network_type)

    ## Init helpers
    def _structure_scientific_network(self,
                                      scientists: List[Scientist],
                                      network_type: ENetworkType):
        match network_type:
            case ENetworkType.COMPLETE:
                for scientist in scientists:
                    self._add_all_influencers_for_updater(scientist, scientists)
            case ENetworkType.CYCLE:
                for i, scientist in enumerate(scientists):
                    self._add_cycle_influencers_for_updater(scientist, i, scientists)
            case _:
                print("Invalid. All ENetworkType need to be specifically matched.")
                raise NotImplementedError

    ## Interface
    def enetwork_play_round(self, diversity_sim: bool = False):
        #if diversity_sim:
            #self._diversity_sim_actions()
        self._standard_round_actions()

    ## Private methods
    def _standard_round_actions(self):
        for scientist in self.scientists:
            # Whether 'tis nobler to experiment
            scientist.decide_round_research_action()
        for scientist in self.scientists:
            scientist.jeffrey_update_credence()
        for scientist in self.scientists:
            scientist.round_binomial_experiment = None
        
    ## Private methods
    def _add_all_influencers_for_updater(self,
                                           updater: JeffreyUpdater,
                                           influencers: List[Scientist]):
        for influencer in influencers:
            updater.add_jeffrey_influencer(influencer)
    
    def _add_cycle_influencers_for_updater(self, 
                                           updater: JeffreyUpdater,
                                           i: int,
                                           influencers: List[Scientist]):
        updater.add_jeffrey_influencer(influencers[i-1])
        updater.add_jeffrey_influencer(influencers[i])
        updater.add_jeffrey_influencer(influencers[(i + 1) % self.scientist_popcount])
