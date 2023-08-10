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
        if not len(self.scientists) == params.scientist_init_popcount:
            raise ValueError(
                "Something went wrong. !(len(self.scientists) == scientist_popcount)")
        self.skeptics: list[Scientist] = []
        if params.skeptical_agents_setup:
            if params.stable_confidence_threshold:
                raise NotImplementedError("Skeptics are only supported in unstable networks.")
            setup = params.skeptical_agents_setup
            if setup.n_skeptical:
                skeptic_priors = rng.uniform(low = setup.min_cr, high = setup.max_cr, size = setup.n_skeptical).tolist()
                for prior in skeptic_priors:
                    self.skeptics.append(Scientist(rng,
                                                   params.n_per_round,
                                                   params.epsilon,
                                                   params.low_stop,
                                                   prior,
                                                   params.m))
        for _ in range(len(self.skeptics)):
            indices: np.ndarray = np.arange(len(self.scientists))
            self.scientists.pop(rng.choice(indices))
        for s in self.scientists + self.skeptics:
            # We assume that the network we start off with has some experience
            s.rounds_of_experience = 20 
        self.retired: list[Scientist] = []
        self._structure_scientific_network(self.scientists, params.network_type)

    ## Init helpers
    def _structure_scientific_network(self,
                                      scientists: List[Scientist],
                                      network_type: ENetworkType):
        match network_type:
            case ENetworkType.COMPLETE:
                for scientist in scientists:
                    self._add_all_influencers_for_updater(scientist, scientists + self.skeptics)

            case ENetworkType.CYCLE:
                for i, scientist in enumerate(scientists):
                    # NOTE: Skeptics are unsupported for the cycle network
                    self._add_cycle_influencers_for_updater(scientist, i, scientists)
            case _:
                raise NotImplementedError("ENetworkType needs to be specifically matched.")

    ## Interface
    def enetwork_play_round(self, unstable_sim: bool = False):
        if unstable_sim:
            self._unstable_sim_actions()
        self._standard_round_actions()
        self._rounds_played += 1

    ## Private methods
    def _standard_round_actions(self):
        for scientist in self.scientists + self.skeptics:
            # Whether 'tis nobler to experiment
            scientist.decide_round_research_action()
        for scientist in self.scientists:
            scientist.jeffrey_update_credence()
        for scientist in self.scientists + self.skeptics:
            scientist.previous_binomial_experiment = scientist.round_binomial_experiment
            scientist.round_binomial_experiment = None
            scientist.rounds_of_experience += 1

    def _unstable_sim_actions(self):
        if not self.params.network_type == ENetworkType.COMPLETE:
            raise NotImplementedError(
                """
                For unstable sims, currently only a complete network is supported
                """)
        if self.params.lifecyclesetup:
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
        skeptic_retired = self._retire()
        self._admissions(skeptic_retired)

    def _retire(self) -> bool:
        """
        Retire an agent. Returns True if the retired agent was a skeptic, False
        if somebody else or nobody was retired.
        """
        if not self.params.lifecyclesetup:
            raise ValueError("Attempted lifecycle action but lifecycle setup not set.")
        scientists = self.scientists
        # Every x rounds, a scientist exits
        if not self._rounds_played % self.params.lifecyclesetup.rounds_to_new_agent == 0:
            return False
        # unless a limit on network contraction is hit
        working_scientists = scientists + self.skeptics
        if len(working_scientists) <= math.ceil(self.params.scientist_init_popcount * 0.5):
            return False
        # and unless no experienced scientist is found
        experienced_scientists = [s for s in working_scientists if s.rounds_of_experience >= 20]
        if not experienced_scientists:
            return False
        indices: np.ndarray = np.arange(len(experienced_scientists))
        s: Scientist = experienced_scientists[self.rng.choice(indices)]
        self.retired.append(s)
        if s in self.scientists:
            self.scientists.remove(s)
            return False
        if s in self.skeptics:
            self.skeptics.remove(s)
            return True
        return False
    
    def _admissions(self, skeptic_needed: bool):
        # Every x rounds, a new scientist enters the discourse (e.g. someone enters
        # PhD program and starts research on the topic)
        if not self._rounds_played % 8 == 0:
            return
        scientists = self.scientists
        all_scientists = scientists + self.skeptics
        
        # unless a limit on network expansion is hit
        if len(all_scientists) > math.floor(self.params.scientist_init_popcount * 1.5):
            return
        
        params = self.params
        if not params.lifecyclesetup:
            raise ValueError("Attempted admissions but lifecyclesetup is not set.")
        if skeptic_needed:
            skep_setup = params.skeptical_agents_setup
            if skep_setup:
                prior = self.rng.uniform(low = skep_setup.min_cr,
                                         high = skep_setup.max_cr)
                skeptic = Scientist(self.rng,
                                                params.n_per_round,
                                                params.epsilon,
                                                params.low_stop,
                                                prior,
                                                params.m)
                self.skeptics.append(skeptic)
                for existing_scientist in scientists:
                    existing_scientist.add_jeffrey_influencer(skeptic)
                return
        cr = params.lifecyclesetup.admissions_priors_func(1, self.rng, params.priorsetup)[0]
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
        for existing_skeptic in self.skeptics:
            new_s.add_jeffrey_influencer(existing_skeptic)
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
