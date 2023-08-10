import csv
import os
from statistics import stdev
import numpy as np

from sim.sim_models import *


class OutputProcessor():
    def process_sims_results(self, results: list[ENSingleSimResults], params: ENParams) -> ENResultsSummary:
        # In all the following, "is not None" is verbose but helps prevent a bug where "if x" would evaluate
        # to False when the value is 0 (for some parameters, 0 can be a legitimate result)
        cons_sims = [res for res in results if res.consensus_round is not None]
        polarized_sims = [res for res in results if res.stable_pol_round is not None]
        abandon_sims = [res for res in results if res.research_abandoned_round is not None]
        unstable_sims = [res for res in results if res.unstable_conclusion_round is not None]
        cons_count = len(cons_sims)
        polarized_count = len(polarized_sims)
        abandoned_count = len(abandon_sims)
        unstable_count = str(len(unstable_sims))
        prop_cons = str(round(cons_count / len(results), 3))
        prop_pol = str(round(polarized_count / len(results), 3))
        prop_aband = str(round(abandoned_count / len(results), 3))
        av_c_r = av_p_r = av_a_r = "N/A"
        if cons_sims:
            av_c_r = np.mean([res.consensus_round for res in cons_sims if res.consensus_round is not None])
            av_c_r = str(round(float(av_c_r), 3))
        if polarized_sims:
            av_p_r = np.mean(
                [res.stable_pol_round for res in polarized_sims if res.stable_pol_round is not None])
            av_p_r = str(round(float(av_p_r), 3))
        if abandon_sims:
            av_a_r = np.mean(
                [res.research_abandoned_round for res in abandon_sims if res.research_abandoned_round is not None])
            av_a_r = str(round(float(av_a_r), 3))
        props_confident = [res.prop_agents_confident_in_true_view for res in results]
        av_prop_confident_in_true_view = str(round(float(np.mean(props_confident)), 3))
        sd_prop_confident = str(round(stdev(props_confident), 3))
        sims_snapshot_brier = str(
            round(float(np.mean([res.sim_game_exit_snapshot_brier for res in results])), 3))
        av_sim_brier_penalty_total = str(
            round(float(np.mean([res.sim_brier_penalty_total for res in results])), 3))
        brier_penalty_ratios = [res.sim_brier_penalty_ratio_to_max for res in results]
        av_sim_brier_total_to_max_possible = str(
            round(float(np.mean(brier_penalty_ratios)), 3))
        sims_sd_av_ratio_brier = str(round(stdev(brier_penalty_ratios), 3))
        sims_av_non_skeptic_brier_ratio = str(
            round(float(np.mean([res.sim_non_skeptic_brier_ratio for res in results])), 3))
        if params.lifecyclesetup:
            av_prop_working_confident = str(round(float(
                np.mean([res.prop_working_confident for res in results if res.prop_working_confident is not None])),
                3))
            av_prop_retired_confident = str(round(float(
                np.mean([res.prop_retired_confident for res in results if res.prop_retired_confident is not None])),
                3))
            av_n_all_agents = str(round(float(np.mean(
                [res.n_all_agents for res in results if res.n_all_agents is not None])), 3))
            return ENResultsSummary(
                sims_proportion_consensus_reached=prop_cons,
                sims_av_consensus_round=av_c_r,
                sims_proportion_polarization=prop_pol,
                sims_av_polarization_round=av_p_r,
                sims_proportion_research_abandoned=prop_aband,
                sims_av_research_abandonment_round=av_a_r,
                sims_unstable_count=unstable_count,
                sims_prop_agents_confident_in_true_view=av_prop_confident_in_true_view,
                sims_sd_prop_agents=sd_prop_confident,
                sims_av_total_brier_penalty=av_sim_brier_penalty_total,
                sims_av_ratio_brier_to_max_possible=av_sim_brier_total_to_max_possible,
                sims_av_non_skeptic_brier_ratio=sims_av_non_skeptic_brier_ratio,
                sims_sd_av_ratio_brier=sims_sd_av_ratio_brier,
                sims_av_exit_snapshot_brier=sims_snapshot_brier,
                sims_av_n_all_agents=av_n_all_agents,
                sims_av_prop_working_confident=av_prop_working_confident,
                sims_av_prop_retired_confident=av_prop_retired_confident)
        return ENResultsSummary(
            sims_proportion_consensus_reached=prop_cons,
            sims_av_consensus_round=av_c_r,
            sims_proportion_polarization=prop_pol,
            sims_av_polarization_round=av_p_r,
            sims_proportion_research_abandoned=prop_aband,
            sims_av_research_abandonment_round=av_a_r,
            sims_unstable_count=unstable_count,
            sims_prop_agents_confident_in_true_view=av_prop_confident_in_true_view,
            sims_sd_prop_agents=sd_prop_confident,
            sims_av_total_brier_penalty=av_sim_brier_penalty_total,
            sims_av_ratio_brier_to_max_possible=av_sim_brier_total_to_max_possible,
            sims_av_non_skeptic_brier_ratio=sims_av_non_skeptic_brier_ratio,
            sims_sd_av_ratio_brier=sims_sd_av_ratio_brier,
            sims_av_exit_snapshot_brier=sims_snapshot_brier)
    
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
