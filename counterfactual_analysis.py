import pandas as pd
from sim.simsetup import *

# Define constants for column names
SKEPTIC_BRIER_ADVANTAGE_KEY = 'skeptic_brier_advantage'
COUNTERFACTUAL_BRIER_RATIO_KEY = 'counterfactual_brier_ratio_base_model_w_skeptic_credence'
COUNTERFACTUAL_DIFF_KEY = 'counterfactual_diff'
SKEP_MODEL_BR_DIFF_KEY = 'skep_model_br_diff'
IMPROVEMENT_KEY = 'proportion_improvement_responsible_due_credence'

DF2_KEY = '_df2'

class CounterfactualAnalysis:
    def analyze_credence_spiking_vs_didactic(self, 
                                             df1_baseline: pd.DataFrame, 
                                             df2: pd.DataFrame, 
                                             output_basename: str):
        df1 = df1_baseline

        # Calculate skeptic_brier_advantage and add to a new dataframe
        # Positive value means AABM (average agent in baseline model) has higher (worse) score
        # (skeptic has advantage). Negative value means AABM has lower (better) score.
        df1[SKEPTIC_BRIER_ADVANTAGE_KEY] = (df1['sims_av_brier_ratio'] - 0.25)
        df1[COUNTERFACTUAL_BRIER_RATIO_KEY] = (
            ((df1['scientist_init_popcount'] - 1) * df1['sims_av_brier_ratio'] + 0.25) / df1['scientist_init_popcount']
        )
        df1[COUNTERFACTUAL_DIFF_KEY] = (
            df1[COUNTERFACTUAL_BRIER_RATIO_KEY] - df1['sims_av_brier_ratio']
        ) # Positive value means brier ratio would increase when an agent with a skeptic's credence is added,
        # negative value means brier ratio would decrease (score improved).

        # Define the relevant parameters
        relevant_params = ['sim_count', 'pop_size', 'trials', 'epsilon', 'max_rounds', 'm']

        # Merge df1 and df2 on relevant parameters
        merged_df1_df2 = pd.merge(df1, df2, on=relevant_params, suffixes=(None, DF2_KEY))
        pd.options.display.max_columns = None
        h = merged_df1_df2[merged_df1_df2.epsilon == 0.05].iloc[1]
        print(h)

        # Add missing columns to df2
        col_list = [SKEPTIC_BRIER_ADVANTAGE_KEY, COUNTERFACTUAL_BRIER_RATIO_KEY, 
                    COUNTERFACTUAL_DIFF_KEY, SKEP_MODEL_BR_DIFF_KEY, IMPROVEMENT_KEY]

        for col in col_list:
            if col not in df2.columns:
                df2[col] = None

        for index, row in merged_df1_df2.iterrows():
            df2.loc[
            (df2[relevant_params] == row[relevant_params]).all(axis=1), 
            [SKEPTIC_BRIER_ADVANTAGE_KEY]
            ] = row[SKEPTIC_BRIER_ADVANTAGE_KEY]


            df2.loc[
            (df2[relevant_params] == row[relevant_params]).all(axis=1), 
            [COUNTERFACTUAL_BRIER_RATIO_KEY]
            ] = row[COUNTERFACTUAL_BRIER_RATIO_KEY]


            df2.loc[
            (df2[relevant_params] == row[relevant_params]).all(axis=1), 
            [COUNTERFACTUAL_DIFF_KEY]
            ] = row[COUNTERFACTUAL_DIFF_KEY]
            
            df2.loc[
                (df2[relevant_params] == row[relevant_params]).all(axis=1), 
                [SKEP_MODEL_BR_DIFF_KEY]
            ] = (row['sims_av_brier_ratio' + DF2_KEY] - row['sims_av_brier_ratio'])

        # Calculate proportion_improvement_responsible_due_credence and save it to df2
        df2[IMPROVEMENT_KEY] = df2.apply(
            lambda row: (round(row[COUNTERFACTUAL_DIFF_KEY] / row[SKEP_MODEL_BR_DIFF_KEY], 4) 
                         if row[SKEP_MODEL_BR_DIFF_KEY] < 0 and row[COUNTERFACTUAL_DIFF_KEY] < 0 else "N/A"), 
                         axis=1
        ) # N/A if the score did not improve (lower) in the skep model,
        # or if the counterfactual score did not improve the score in the base model.

        # Convert columns to numeric before rounding
        for col in col_list:
            try:
                df2[col] = pd.to_numeric(df2[col])
            except ValueError:
                pass 
            # We write "N/A" to the improvement column on some rows, so we can't convert it to numeric
                                     
        df2 = df2.round(4)
        output_filename = output_basename.replace('.csv', '_w_counterfactual.csv')
        df2.to_csv(output_filename, index=False)

if __name__ == '__main__':
    ca = CounterfactualAnalysis()
    df1 = pd.read_csv(LIFECYCLE_FILENAME)
    df2_name = LIFECYCLE_W_SKEPTICS_FILENAME
    df3_name = LIFECYCLE_W_ALTERNATOR_SKEPTICS_FILENAME
    df2 = pd.read_csv(df2_name)
    df3 = pd.read_csv(df3_name)
    ca.analyze_credence_spiking_vs_didactic(df1, df2, df2_name)
    ca.analyze_credence_spiking_vs_didactic(df1, df3, df3_name)

    # Propagandist and centrist
    df4_name = LIFECYCLE_W_PROPAGANDIST_FILENAME
    df5_name = LIFECYCLE_W_PROPAGANDIST_N_SKEPTIC_FILENAME
    df4 = pd.read_csv(df4_name)
    df5 = pd.read_csv(df5_name)
    ca.analyze_credence_spiking_vs_didactic(df4, df5, df5_name) 
