import numpy as np
import timeit
from multiprocessing import Pool
from network.network import *
from sim.output_processor import OutputProcessor
from sim.sim import *
from sim.sim_models import *
from typing import Optional, List
from enum import Enum, auto

class ENSimType(Enum):
    ZOLLMAN_COMPLETE = auto()
    ZOLLMAN_CYCLE = auto()
    POLARIZATION = auto()
    LIFECYCLE = auto()
    LIFECYCLE_W_SKEPTICS = auto()

class ENSimSetup():
    def __init__(self,
                 sim_count: int,
                 sim_type: Optional[ENSimType]):
        self.sim_count = sim_count
        self.sim_type = sim_type
        self.output_processor = OutputProcessor()
    
    def quick_setup(self):
        """ Setup sims from pre-defined templates, e.g. ENSimType.ZOLLMAN_COMPLETE. 
        Use setup_sims instead if you need to customize the parameters."""
        if not self.sim_type:
            raise ValueError("Quick setup can only be called if you have specified ENSimType")
        match self.sim_type:
            # A reproduction of Zollman 2007, https://philpapers.org/rec/ZOLTCS
            case ENSimType.ZOLLMAN_COMPLETE:
                configs = [ENParams(pop, ENetworkType.COMPLETE, 1000, 0.001, 0.5, 3000, 0.99, 0) for pop in range(4, 12)]
                self.run_configs(configs, "zollman2007.csv")
            case ENSimType.ZOLLMAN_CYCLE:
                configs = [ENParams(pop, ENetworkType.CYCLE, 1000, 0.001, 0.5, 3000, 0.99, 0) for pop in range(4, 12)]
                self.run_configs(configs, "zollman2007.csv")
            # A reproduction of O'Connor & Weatherall 2018, https://philpapers.org/rec/OCOSP
            case ENSimType.POLARIZATION: 
                configs = [ENParams(pop, ENetworkType.COMPLETE, n, e, 0.5, 10000, 0.99, m) for pop in (6, 10, 20) # 2, 6, 10, 20
                                                                                                for e in (0.2,) # 0.01, 0.05, 0.1, 0.15, 0.2
                                                                                                for m in np.arange(1.0, 3.1, 0.1).tolist() 
                                                                                                for n in (50,)] # 1, 5, 10, 20, 50, 100
                self.run_configs(configs, "oconnor2018.csv")
            case ENSimType.LIFECYCLE:
                CT = 0.99
                configs = [ENParams(
                    pop, ENetworkType.COMPLETE, n, e, 0.5, rounds, None, m, confident_priors,
                    PriorSetup(confident_start_config=ConfidentStartConfig(1, CT)),
                    LifeCycleSetup(rounds_to_new_agent, uniform_priors)
                    )   for pop in (20,) # 6, 10, 20, 50)
                        for e in (0.05,) #0.01, 0.05, 0.1, 0.15
                        for m in (1, 1.1, 1.5, 2, 2.5, 3) # 1, 1.1, 1.5, 2, 2.5, 3)]
                        for n in (5,) # 1, 5, 10, 20, 50, 100]
                        for rounds in (2000,)
                        for rounds_to_new_agent in (10,)]
                self.run_configs(configs, "lifecycle_uniform_admissions.csv")
            case ENSimType.LIFECYCLE_W_SKEPTICS:
                CT = 0.99
                configs = [ENParams(
                    pop, ENetworkType.COMPLETE, n, e, 0.5, rounds, None, m, confident_priors,
                    PriorSetup(confident_start_config=ConfidentStartConfig(1, CT)),
                    LifeCycleSetup(rounds_to_new_agent, uniform_priors),
                    SkepticalAgentsSetup(skep_n, 0.501, 0.502)
                    )   for pop in (20,) # 6, 10, 20, 50)
                        for e in (0.05,) #0.01, 0.05, 0.1, 0.15
                        for m in (1, 1.1, 1.5, 2, 2.5, 3) # 1, 1.1, 1.5, 2, 2.5)]
                        for n in (5,) # 1, 5, 10, 20, 50, 100
                        for skep_n in (1,)
                        for rounds in (2000,)
                        for rounds_to_new_agent in (10,)]
                self.run_configs(configs, "lifecycle_uniform_admissions_w_skep.csv")
    def run_configs(self, configs: List[ENParams], output_filename: str):
        # We need to be careful when passing rng instances to starmap. If we do not set independent seeds, 
        # we will get the *same* binomial experiments each simulation since the subprocesses share the parent's initial 
        # rng state.
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
        pool = Pool()
        #Commented code is for testing a single run with breakpoints
        # results_from_sims = [self.run_sim(rng_streams[0], params)]
        results_from_sims = pool.starmap(self.run_sim,
                                        [(rng, params) for rng in rng_streams])
        pool.close()
        pool.join()
        if None in results_from_sims:
            raise Warning("Failed to get results from at least one simulation.")
        results = [r for r in results_from_sims if r is not None]
        sims_summary = self.output_processor.process_sims_results(results, params)
        return ENSimsSummary(params, sims_summary)

    def run_sim(self,
                rng: np.random.Generator,
                params: ENParams) -> Optional[ENSingleSimResults]:
        network = ENetwork(rng, params)
        simulation = EpistemicNetworkSimulation(network, params)
        simulation.run_sim()
        return simulation.results
