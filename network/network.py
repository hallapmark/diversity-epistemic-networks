from agents.scientist import Scientist
from agents.updaters.jeffreyupdater import JeffreyUpdater
from sim.sim_models import *
import numpy as np
from typing import List

class RetireResponse(NamedTuple):
    agent_retired: bool
    skeptic_retired: bool

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
            params.m,
            id_code
            ) for id_code, prior in enumerate(priors)]
        if not len(self.scientists) == params.scientist_init_popcount:
            raise ValueError(
                "Something went wrong. !(len(self.scientists) == scientist_popcount)")
        self.used_ids = len(self.scientists) - 1
        self.skeptics: list[Scientist] = []
        if params.skeptical_agents_setup:
            if params.stable_confidence_threshold:
                raise NotImplementedError("Skeptics are only supported in unstable networks.")
            setup = params.skeptical_agents_setup
            skeptic_priors = rng.uniform(low = setup.min_cr, high = setup.max_cr, size = setup.n_skeptical).tolist()
            for prior in skeptic_priors:
                self.skeptics.append(Scientist(rng,
                                               params.n_per_round,
                                               params.epsilon,
                                               params.low_stop,
                                               prior,
                                               params.m,
                                               self.used_ids + 1))
                self.used_ids += 1
        # We keep the total network size the same if skeptics are present
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
        for s in scientists:
            s.jeffrey_influencers = []
        match network_type:
            case ENetworkType.COMPLETE:
                for scientist in scientists:
                    influencers = scientists + self.skeptics
                    indices = np.arange(len(influencers))
                    self.rng.shuffle(indices)
                    shuffled_influencers: list[Scientist] = []
                    for i in indices:
                        shuffled_influencers.append(influencers[i])
                    self._add_all_influencers_for_updater(scientist, shuffled_influencers)
            case ENetworkType.CYCLE:
                for i, scientist in enumerate(scientists):
                    # NOTE: Skeptics are unsupported for the cycle network
                    self._add_cycle_influencers_for_updater(scientist, i, scientists)
            case _:
                raise NotImplementedError("ENetworkType needs to be specifically matched.")

    ## Interface
    def enetwork_play_round(self, lifecycle_sim: bool = False):
        self._standard_round_actions()
        if lifecycle_sim:
            self._lifecycle_round_actions()
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

    def _lifecycle_round_actions(self):
        """ Every x rounds, a scientist exits and a new one enters """
        if not self.params.network_type == ENetworkType.COMPLETE:
            raise NotImplementedError(
                """
                For unstable sims, currently only a complete network is supported
                """)
        if not self.params.lifecyclesetup:
            raise NotImplementedError(
                """
                For unstable sims, currently only lifecycle configurations are supported.
                """)
        if not self._rounds_played % self.params.lifecyclesetup.rounds_to_new_agent == 0:
            return
        self._admissions(self._retire())
        self._structure_scientific_network(self.scientists, self.params.network_type)
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

    def _retire(self) -> RetireResponse:
        """
        Retire an agent. Returns a RetireResponse.
        """
        working_scientists = self.scientists + self.skeptics
        # Do not retire if no experienced scientist is found
        experienced_scientists = [s for s in working_scientists if s.rounds_of_experience >= 20]
        if not experienced_scientists:
            return RetireResponse(agent_retired=False, skeptic_retired=False)
        indices: np.ndarray = np.arange(len(experienced_scientists))
        retiree: Scientist = experienced_scientists[self.rng.choice(indices)]
        self.retired.append(retiree)
        retiree.round_binomial_experiment = None
        for scientist in self.scientists:
            if retiree.id_code == scientist.id_code:
                self.scientists.remove(scientist)
                return RetireResponse(agent_retired=True, skeptic_retired=False)
        for skeptic in self.skeptics:
            if retiree.id_code == skeptic.id_code:
                self.skeptics.remove(skeptic)
                return RetireResponse(agent_retired=True, skeptic_retired=True)
        raise NotImplementedError("We can only retire scientists and skeptics.")
    
    def _admissions(self, retire_response: RetireResponse):
        # Every x rounds, a new scientist enters the discourse (e.g. someone enters
        # PhD program and starts research on the topic)
        scientists = self.scientists
        params = self.params
        if not params.lifecyclesetup:
            raise ValueError("Attempted admissions but lifecyclesetup is not set.")
        if not retire_response.agent_retired:
            # We only admit a new agent if someone just retired
            return
        if retire_response.skeptic_retired:
            skep_setup = params.skeptical_agents_setup
            if not skep_setup:
                raise ValueError("Attempted skeptic admission but skeptic setup not set.")
            prior = self.rng.uniform(low = skep_setup.min_cr,
                                     high = skep_setup.max_cr)
            skeptic = Scientist(self.rng,
                                params.n_per_round,
                                params.epsilon,
                                params.low_stop,
                                prior,
                                params.m,
                                self.used_ids + 1)
            self.used_ids += 1
            self.skeptics.append(skeptic)
            return
        # A regular agent retired – replace with regular agent.
        cr = params.lifecyclesetup.admissions_priors_func(1, self.rng, params.priorsetup)[0]
        new_s = Scientist(self.rng,
                          self.params.n_per_round,
                          self.params.epsilon,
                          self.params.low_stop,
                          cr,
                          self.params.m,
                          self.used_ids + 1)
        self.used_ids += 1
        index = self.rng.choice(np.arange(len(self.scientists) + 1))
        scientists.insert(index, new_s)

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
