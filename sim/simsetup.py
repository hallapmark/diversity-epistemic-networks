from math import sqrt
from statistics import stdev
import numpy as np
import timeit
from multiprocessing import Pool
from network.network import *
from sim.sim import *
from sim.sim_models import *
from typing import Optional, List
from enum import Enum, auto
import os
import csv

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
    
    def quick_setup(self):
        """ Setup sims from pre-defined templates, e.g. ENSimType.ZOLLMAN_COMPLETE. 
        Use setup_sims instead if you need to customize the parameters."""
        if not self.sim_type:
            raise ValueError("Quick setup can only be called if you have specified ENSimType")
        match self.sim_type:
            # A reproduction of Zollman 2007, https://philpapers.org/rec/ZOLTCS
            case ENSimType.ZOLLMAN_COMPLETE:
                configs = [ENParams(pop, ENetworkType.COMPLETE, 1000, 0.001, 0.5, 3000, 0.99, 0) for pop in range(4, 12)]
                self.setup_sims(configs, "zollman2007.csv")
            case ENSimType.ZOLLMAN_CYCLE:
                configs = [ENParams(pop, ENetworkType.CYCLE, 1000, 0.001, 0.5, 3000, 0.99, 0) for pop in range(4, 12)]
                self.setup_sims(configs, "zollman2007.csv")
            # A reproduction of O'Connor & Weatherall 2018, https://philpapers.org/rec/OCOSP
            case ENSimType.POLARIZATION: 
                configs = [ENParams(pop, ENetworkType.COMPLETE, n, e, 0.5, 10000, 0.99, m) for pop in (6, 10, 20) # 2, 6, 10, 20
                                                                                                for e in (0.2,) # 0.01, 0.05, 0.1, 0.15, 0.2
                                                                                                for m in np.arange(1.0, 3.1, 0.1).tolist() 
                                                                                                for n in (50,)] # 1, 5, 10, 20, 50, 100
                self.setup_sims(configs, "oconnor2018.csv")
            case ENSimType.LIFECYCLE:
                CT = 0.99
                configs = [ENParams(
                    pop, ENetworkType.COMPLETE, n, e, 0.5, 600, None, m, confident_priors,
                    PriorSetup(confident_start_config=ConfidentStartConfig(c, CT)), True
                    )   for pop in (10, 20, 50) # 6, 10, 20, 50)
                        for e in (0.05, 0.1) #0.01, 0.05, 0.1, 0.15
                        for m in (2, 2.5, 3) # 1, 1.1, 1.5, 2, 2.5, 3)]
                        for n in (5, 20) # 1, 5, 10, 20, 50, 100
                        for c in (1,)]
                self.setup_sims(configs, "lifecycle_every8.csv")
            case ENSimType.LIFECYCLE_W_SKEPTICS:
                CT = 0.99
                configs = [ENParams(
                    pop, ENetworkType.COMPLETE, n, e, 0.5, 600, None, m, confident_priors,
                    PriorSetup(confident_start_config=ConfidentStartConfig(c, CT)), True,
                    SkepticalAgentsSetup(skep_n, 0.501, 0.8)
                    )   for pop in (10, 20, 50) # 6, 10, 20, 50)
                        for e in (0.05, 0.1,) #0.01, 0.05, 0.1, 0.15
                        for m in (2, 2.5, 3) # 1, 1.1, 1.5, 2, 2.5)]
                        for n in (5, 20) # 1, 5, 10, 20, 50, 100
                        for c in (1,)
                        for skep_n in (1,2,3)
]
                self.setup_sims(configs, "lifecycle_w_skeptics_every8.csv")
    def setup_sims(self, configs: List[ENParams], output_filename: str):
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
            csv_data = self.data_for_writing(results_summary, self.sim_count, time_elapsed)
            print()
            self.record_sim(csv_data, output_filename)

    def run_sims_for_param_config(self, params: ENParams, rng_streams: List[np.random.Generator]) -> ENSimsSummary:
        if not rng_streams:
            raise ValueError("There needs to be at least one rng.")
        pool = Pool()
        # Commented code is for testing a single run with breakpoints
        # results_from_sims = [self.run_sim(rng_streams[0], params)]
        results_from_sims = pool.starmap(self.run_sim,
                                        [(rng, params) for rng in rng_streams])
        pool.close()
        pool.join()
        if None in results_from_sims:
            raise Warning("Failed to get results from at least one simulation.")
        results = [r for r in results_from_sims if r is not None]
        sims_summary = self.process_sims_results(results, params)
        return ENSimsSummary(params, sims_summary)
    
    def process_sims_results(self, results: list[ENSimulationRawResults], params: ENParams) -> ENResultsSummary:
        cons_sims = [res for res in results if res.consensus_round]
        polarized_sims = [res for res in results if res.stable_pol_round]
        abandon_sims = [res for res in results if res.research_abandoned_round]
        unstable_sims = [res for res in results if res.unstable_conclusion_round]
        cons_count = len(cons_sims)
        polarized_count = len(polarized_sims)
        abandoned_count = len(abandon_sims)
        unstable_count = str(len(unstable_sims))
        prop_cons = str(round(cons_count / len(results), 3))
        prop_pol = str(round(polarized_count / len(results), 3))
        prop_aband = str(round(abandoned_count / len(results), 3))
        av_c_r = av_p_r = av_a_r = "N/A"
        if cons_sims:
            av_c_r = np.mean([res.consensus_round for res in cons_sims if res.consensus_round])
            av_c_r = str(round(float(av_c_r), 3))
        if polarized_sims:
            av_p_r = np.mean(
                [res.stable_pol_round for res in polarized_sims if res.stable_pol_round])
            av_p_r = str(round(float(av_p_r), 3))
        if abandon_sims:
            av_a_r = np.mean(
                [res.research_abandoned_round for res in abandon_sims if res.research_abandoned_round])
            av_a_r = str(round(float(av_a_r), 3))
        props_confident = [res.prop_agents_confident_in_true_view for res in results]
        av_prop_confident_in_true_view = round(float(np.mean(props_confident)), 3)
        sd = stdev(props_confident)
        cv = round(sd / av_prop_confident_in_true_view, 3) # Coefficient of variation
        if params.lifecycle:
            av_prop_working_confident = str(round(float(
                np.mean([res.prop_working_confident for res in results if res.prop_working_confident])),
                3))
            av_prop_retired_confident = str(round(float(
                np.mean([res.prop_retired_confident for res in results if res.prop_retired_confident])),
                3))
            av_n_all_agents = str(round(float(np.mean(
                [res.n_all_agents for res in results if res.n_all_agents])), 3))
            return ENResultsSummary(
                sims_proportion_consensus_reached=prop_cons,
                sims_avg_consensus_round=av_c_r,
                sims_proportion_polarization=prop_pol,
                sims_avg_polarization_round=av_p_r,
                sims_proportion_research_abandoned=prop_aband,
                sims_avg_research_abandonment_round=av_a_r,
                sims_unstable_count=unstable_count,
                av_prop_agents_confident_in_true_view=str(av_prop_confident_in_true_view),
                sd=str(round(sd, 3)),
                cv=str(cv),
                av_n_all_agents=av_n_all_agents,
                av_prop_working_confident=av_prop_working_confident,
                av_prop_retired_confident=av_prop_retired_confident)
        return ENResultsSummary(
            prop_cons, av_c_r, prop_pol, av_p_r, prop_aband, av_a_r, unstable_count, 
            str(av_prop_confident_in_true_view), str(sd), str(cv))

    def run_sim(self,
                rng: np.random.Generator,
                params: ENParams) -> Optional[ENSimulationRawResults]:
        network = ENetwork(rng, params)
        simulation = EpistemicNetworkSimulation(network, params)
        simulation.run_sim()
        return simulation.results

    def record_sim(self, results: ENResultsCSVWritableSummary, path: str):
        file_exists = os.path.isfile(path)
        # res_dir = "/results"
        # Path(res_dir).mkdir(parents=True, exist_ok=True)
        # filename = Path(res_dir, filename).with_suffix('.csv')
        with open(path, newline='', mode = 'a') as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                writer.writerow(results.headers)
            writer.writerow(results.sim_data)

    def data_for_writing(self, 
                        sims_summary: ENSimsSummary, 
                        sim_count: int, 
                        time_elapsed: float) -> ENResultsCSVWritableSummary:
        headers = ['Sim count']
        headers.extend([param_name for param_name in sims_summary.params._asdict().keys()])
        headers.append('sim time (s)')
        summary_fields = [field for field in sims_summary.results_summary._asdict().keys()]
        headers.extend(summary_fields)

        sim_data = [str(sim_count)]
        parameter_vals: list[str] = []
        for parameter_val in sims_summary.params:
            try:
                # Get a *function* name e.g. the name of the priors func used
                parameter_vals.append(parameter_val.__name__)  # type: ignore
            except AttributeError:
                # Get the value of any other parameter
                parameter_vals.append(str(parameter_val))
        sim_data.extend(parameter_vals)
        sim_data.append(str(round(time_elapsed, 1)))
        result_str_list = [r for r in sims_summary.results_summary]
        print(f'Summary fields: {summary_fields}')
        print(f'Results from config: {result_str_list}')
        sim_data.extend(result_str_list)
        summary = ENResultsCSVWritableSummary(headers, sim_data)
        return summary
