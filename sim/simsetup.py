import numpy as np
import timeit
from multiprocessing import Pool, cpu_count
from sim.network import *
from sim.output_processor import OutputProcessor
from sim.sim import *
from sim.sim_models import *
from typing import Optional, List
from enum import Enum, auto

LIFECYCLE_FILENAME = "lifecycle_effect_of_m_1000r.csv"
LIFECYCLE_W_SKEPTICS_FILENAME = "lifecycle_w_skep_effect_of_m_1000r.csv"
LIFECYCLE_W_ALTERNATOR_SKEPTICS_FILENAME = "lifecycle_w_alt_skep_effect_of_m_1000r.csv"
LIFECYCLE_W_PROPAGANDIST_FILENAME = "lifecycle_w_propagandist_effect_of_m_1000r.csv"
LIFECYCLE_W_PROPAGANDIST_N_SKEPTIC_FILENAME = "lifecycle_w_propagandist_n_skeptic_effect_of_m_1000r.csv"

pop_VALS = (10, 20, 50)
e_VALS = (0.01, 0.05, 0.1)
m_VALS = (0, 1, 1.1, 1.5, 2, 2.5, 3)

class ENSimType(Enum):
    LIFECYCLE = auto()
    LIFECYCLE_W_SKEPTIC = auto()
    LIFECYCLE_W_ALTERNATOR_SKEPTIC = auto()
    LIFECYCLE_W_PROPAGANDIST = auto()
    LIFECYCLE_W_PROPAGANDIST_N_SKEPTIC = auto()

class ENSimSetup():
    def __init__(self,
                 sim_count: int,
                 sim_type: Optional[ENSimType]):
        self.sim_count = sim_count
        self.sim_type = sim_type
        self.output_processor = OutputProcessor()
    
    def quick_setup(self):
        """ Setup sims from pre-defined templates, e.g. ENSimType.LIFECYCLE. 
        Use func run_configs instead if you need to customize the parameters."""
        if not self.sim_type:
            raise ValueError("Quick setup can only be called if you have specified ENSimType")
        match self.sim_type:
            case ENSimType.LIFECYCLE:
                configs = [ENParams(
                    pop, n, e, rounds, m, 
                    LifeCycleSetup(rounds_to_new_agent, uniform_priors), 
                    confident_priors
                    )   for pop in (pop_VALS) # 10, 20, 50
                        for e in (e_VALS) # 0.01, 0.05, 0.1
                        for m in (m_VALS) # 0, 1, 1.1, 1.5, 2, 2.5, 3
                        for n in (5,)
                        for rounds in (1000,)
                        for rounds_to_new_agent in (10,)]
                self.run_configs(configs, LIFECYCLE_FILENAME)
            case ENSimType.LIFECYCLE_W_SKEPTIC:
                configs = [ENParams(
                    pop, n, e, rounds, m, 
                    LifeCycleSetup(rounds_to_new_agent, uniform_priors),
                    confident_priors, 
                    1
                    )   for pop in (pop_VALS)
                        for e in (e_VALS)
                        for m in (m_VALS)
                        for n in (5,)
                        for rounds in (1000,)
                        for rounds_to_new_agent in (10,)]
                self.run_configs(configs, LIFECYCLE_W_SKEPTICS_FILENAME)
            case ENSimType.LIFECYCLE_W_ALTERNATOR_SKEPTIC:
                configs = [ENParams(
                    pop, n, e, rounds, m, 
                    LifeCycleSetup(rounds_to_new_agent, uniform_priors),
                    confident_priors, 
                    1, True
                    )   for pop in (pop_VALS)
                        for e in (e_VALS)
                        for m in (m_VALS)
                        for n in (5,)
                        for rounds in (1000,)
                        for rounds_to_new_agent in (10,)]
                self.run_configs(configs, LIFECYCLE_W_ALTERNATOR_SKEPTICS_FILENAME)
            case ENSimType.LIFECYCLE_W_PROPAGANDIST:
                configs = [ENParams(
                    pop, n, e, rounds, m, 
                    LifeCycleSetup(rounds_to_new_agent, uniform_priors),
                    confident_priors, 
                    0, False, True
                    )   for pop in (20,)
                        for e in (e_VALS)
                        for m in (m_VALS)
                        for n in (5,)
                        for rounds in (1000,)
                        for rounds_to_new_agent in (10,)]
                self.run_configs(configs, LIFECYCLE_W_PROPAGANDIST_FILENAME)
            case ENSimType.LIFECYCLE_W_PROPAGANDIST_N_SKEPTIC:
                configs = [ENParams(
                    pop, n, e, rounds, m, 
                    LifeCycleSetup(rounds_to_new_agent, uniform_priors),
                    confident_priors, 
                    1, False, True
                    )   for pop in (20,)
                        for e in (e_VALS)
                        for m in (m_VALS)
                        for n in (5,)
                        for rounds in (1000,)
                        for rounds_to_new_agent in (10,)]
                self.run_configs(configs, LIFECYCLE_W_PROPAGANDIST_N_SKEPTIC_FILENAME)

    def run_configs(self, configs: List[ENParams], output_filename: str):
        # We need to be careful when passing rng instances to starmap. If we do not set independent seeds, 
        # we will get the *same* binomial experiments each simulation since the subprocesses share the
        # parent's initial rng state.
        # https://numpy.org/doc/stable/reference/random/parallel.html
        child_seeds = [np.random.SeedSequence(253 + i).spawn(self.sim_count) for i in range(len(configs))]
        for i, param_config in enumerate(configs):
            print(f'Running config: {param_config}')
            print('...')
            rng_streams = [np.random.default_rng(s) for s in child_seeds[i]]
            start_time = timeit.default_timer()
            results_summary = self.run_sims_for_param_config(param_config, rng_streams)
            time_elapsed = timeit.default_timer() - start_time
            print(f'Time elapsed: {time_elapsed}s')
            print()
            csv_data = self.output_processor.data_for_writing(results_summary, self.sim_count, time_elapsed)
            self.output_processor.record_sim(csv_data, output_filename)

    def run_sims_for_param_config(self, params: ENParams, rng_streams: List[np.random.Generator]) -> ENSimsSummary:
        if not rng_streams:
            raise ValueError("There needs to be at least one rng.")
        pool = Pool(processes=max(cpu_count() - 1, 1))
        results_from_sims = pool.starmap(self.run_sim,
                                        [(rng, params) for rng in rng_streams])
        pool.close()
        pool.join()
        #Commented code is for testing a single run with breakpoints
        # results_from_sims = [self.run_sim(rng_streams[0], params)]
        if None in results_from_sims:
            raise Warning("Failed to get results from at least one simulation.")
        results: list[ENSingleSimResults] = [r for r in results_from_sims if r is not None]
        sims_summary = self.output_processor.process_sims_results(results, params)
        return ENSimsSummary(params, sims_summary)

    def run_sim(self,
                rng: np.random.Generator,
                params: ENParams) -> Optional[ENSingleSimResults]:
        network = ENetwork(rng, params)
        simulation = ENSimulation(network, params)
        simulation.run_sim()
        return simulation.results
