import csv
import os
from statistics import stdev
import numpy as np

from sim.sim_models import *


class OutputProcessor():
    def process_sims_results(self, results: list[ENSingleSimResults], params: ENParams) -> ENLifecycleAnalyzedResults:
        # In the following, "is not None" is verbose but helps prevent a bug where "if x" would evaluate
        # to False when the value is 0 (for some parameters, 0 can be a legitimate result)
        sims_av_total_brier_penalty = str(
            round(float(np.mean([res.sim_brier_penalty_total for res in results])), 3))
        brier_penalty_ratios = [res.sim_brier_penalty_ratio for res in results]
        sims_av_brier_ratio = str(
            round(float(np.mean(brier_penalty_ratios)), 3))
        sims_sd_av_brier_ratio = str(round(stdev(brier_penalty_ratios), 3))
        sims_av_retired_brier_penalty = str(round(float(
            np.mean([res.av_retired_brier_penalty for res in results if res.av_retired_brier_penalty is not None])),
            3))
        if params.skeptic_count > 0:
            sims_av_non_skeptic_brier_ratio = str(
                round(float(
                np.mean([res.sim_non_skeptic_brier_ratio for res in results if res.sim_non_skeptic_brier_ratio is not None])),
                3))
        else:
            sims_av_non_skeptic_brier_ratio = "N/A"
        av_prop_retired_confident = str(round(float(
            np.mean([res.prop_retired_confident for res in results if res.prop_retired_confident is not None])),
            3))
        av_n_all_agents = str(round(float(np.mean(
            [res.n_all_agents for res in results if res.n_all_agents is not None])), 3))
        return ENLifecycleAnalyzedResults(
            sims_av_n_all_agents=av_n_all_agents,
            sims_av_total_brier_penalty=sims_av_total_brier_penalty,
            sims_av_brier_ratio=sims_av_brier_ratio,
            sims_sd_av_brier_ratio=sims_sd_av_brier_ratio,
            sims_av_retired_brier_penalty=sims_av_retired_brier_penalty,
            sims_av_prop_retired_confident=av_prop_retired_confident,
            sims_av_non_skeptic_brier_ratio=sims_av_non_skeptic_brier_ratio
        )
    
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
        headers = ['sim_count']
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
        print()
        sim_data.extend(result_str_list)
        summary = ENResultsCSVWritableSummary(headers, sim_data)
        return summary
