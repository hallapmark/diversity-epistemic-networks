from agents.scientist import Scientist
from agents.updaters.jeffreyupdater import JeffreyUpdater
from sim.sim_models import *
import numpy as np
from typing import List

class ENetwork():
    def __init__(self,
                 rng: np.random.Generator,
                 params: ENParams):
        self.rng = rng
        self.params = params
        priors = params.priors_func(params.scientist_init_popcount, rng, params.priorsetup)
        self._rounds_played = 0
        self.scientists = [Scientist(
            rng,
            params.n_per_round,
            params.epsilon,
            params.low_stop,
            prior,
            params.m
            ) for prior in priors]
        #self.intransigent_scientists: list[Scientist] = []
        if not len(self.scientists) == params.scientist_init_popcount:
            raise ValueError(
                "Something went wrong. !(len(self.scientists) == scientist_popcount)")
        self.retired: list[Scientist] = []
        self._structure_scientific_network(self.scientists, params.network_type)

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
    def enetwork_play_round(self, unstable_sim: bool = False):
        if unstable_sim:
            self._unstable_sim_actions()
        self._standard_round_actions()
        self._rounds_played += 1

    ## Private methods
    def _standard_round_actions(self):
        for scientist in self.scientists:
            # Whether 'tis nobler to experiment
            scientist.decide_round_research_action()
        for scientist in self.scientists:
            scientist.jeffrey_update_credence()
        for scientist in self.scientists:
            scientist.previous_binomial_experiment = scientist.round_binomial_experiment
            scientist.round_binomial_experiment = None
            scientist.rounds_of_experience += 1

    def _unstable_sim_actions(self):
        if not self.params.network_type == ENetworkType.COMPLETE:
            raise NotImplementedError(
                """
                For unstable sims, currently only a complete network is supported
                """)
        if self.params.lifecycle:
            self._lifecycle()
            self._standard_round_actions()
        else:
            raise NotImplementedError(
                """
                For unstable sims, currently only lifecycle configurations are supported.
                """)
        #intransigent_scientists = self.intransigent_scientists
        #all_scientists = self.scientists + self.intransigent_scientists
        #credences = np.array([s.credence for s in all_scientists])
        #l = [c > 0.99 for c in credences]
        #if not scientists:
            #raise ValueError("No truth-seeking scientists left. Something weird is afoot.")
        #approx_consensus_reached = np.mean(l) >= 0.8

        # (1) Conversion from incentive structure
        # if approx_consensus_reached:
        #     # Convert a non-radical scientist to a contrarian to the emerging consensus
        #     non_radicals = [s for s in scientists if s.credence > 0.2]
        #     indices: np.ndarray = np.arange(len(non_radicals))
        #     s: Scientist = non_radicals[self.rng.choice(indices)]
        #     s.credence = self.rng.uniform(UNIFORM_LOW, 0.2)
        #     intransigent_scientists.append(s)
        #     scientists.remove(s)

        # (2)
        # Add diversity-preserving structure IF corresponding flag set (we also need to run without
        # this to have a comparison class)

    def _lifecycle(self):
        self._retire()
        self._admissions()

    def _retire(self):
        scientists = self.scientists
        # There is a chance that an existing scientist retires. They stop
        # experimenting and updating credence.
        # Nobody exits before 10 rounds of experience.
        # And there is a limit on network contraction.
        if self.rng.uniform() < 0.25:
            # TODO: Parametrize?
            if len(scientists) <= math.ceil(self.params.scientist_init_popcount * 0.7):
                return
            experienced_scientists = [s for s in scientists if s.rounds_of_experience > 10]
            if not experienced_scientists:
                return
            indices: np.ndarray = np.arange(len(experienced_scientists))
            s: Scientist = experienced_scientists[self.rng.choice(indices)]
            self.retired.append(s)
            self.scientists.remove(s)
            # print(f"Retired: {s}. Round: {self._rounds_played + 1}")
            # print(f"len(scientists): {len(scientists)}")

    def _admissions(self):
        if not self._rounds_played > 10:
            return
        scientists = self.scientists
        # There is a chance that a new scientist enters the discourse (e.g. someone enters
        # PhD program and starts research on the topic)
        if self.rng.uniform() < 0.25:
            # Set a limit on network expansion
            # TODO: Parametrize?
            if len(scientists) > math.floor(self.params.scientist_init_popcount * 1.3):
                return
            cr = self.rng.uniform(UNIFORM_LOW)
            new_s = Scientist(self.rng,
                              self.params.n_per_round,
                              self.params.epsilon,
                              self.params.low_stop,
                              cr,
                              self.params.m,
                              True)
            for existing_scientist in scientists:
                new_s.add_jeffrey_influencer(existing_scientist)
                existing_scientist.add_jeffrey_influencer(new_s)
            scientists.append(new_s)

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
        updater.add_jeffrey_influencer(influencers[(i + 1) % len(influencers)])
