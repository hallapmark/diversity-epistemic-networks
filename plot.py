import pandas as pd
import matplotlib.pyplot as plt

from sim.simsetup import *

df1 = pd.read_csv(LIFECYCLE_FILENAME)
df2 = pd.read_csv(LIFECYCLE_W_SKEPTICS_FILENAME)
df3 = pd.read_csv(LIFECYCLE_W_ALTERNATOR_SKEPTICS_FILENAME)
df4 = pd.read_csv(LIFECYCLE_W_PROPAGANDIST_FILENAME)
df5 = pd.read_csv(LIFECYCLE_W_PROPAGANDIST_N_SKEPTIC_FILENAME)

def plot_brier_ratio(scientist_init_popcount, epsilon, df1, df2, df3,
                     df2_label, df3_label, title):
    # Load the CSV files
    # Filter the data
    df1_filtered = df1[(df1['scientist_init_popcount'] == scientist_init_popcount) & (df1['epsilon'] == epsilon)]
    df2_filtered = df2[(df2['scientist_init_popcount'] == scientist_init_popcount) & (df2['epsilon'] == epsilon)]
    df3_filtered = df3[(df3['scientist_init_popcount'] == scientist_init_popcount) & (df3['epsilon'] == epsilon)]

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(df1_filtered['m'], df1_filtered['sims_av_brier_ratio'], label='Baseline', marker='o')
    plt.plot(df2_filtered['m'], df2_filtered['sims_av_brier_ratio'], label=df2_label, marker='s')
    plt.plot(df3_filtered['m'], df3_filtered['sims_av_brier_ratio'], label=df3_label, marker='v')

    # Add labels and title
    plt.xlabel('m (distrust multiplier)')
    plt.ylabel('Simulation Brier ratio')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

epsilons = [0.01, 0.05, 0.1]
# plot_brier_ratio(10, 0.01)
# plot_brier_ratio(10, 0.05)
#plot_brier_ratio(10, 0.1, df1, df2, df3)
# plot_brier_ratio(20, 0.01)
# plot_brier_ratio(20, 0.05)
# plot_brier_ratio(20, 0.1)
# plot_brier_ratio(50, 0.01)
# plot_brier_ratio(50, 0.05)
# plot_brier_ratio(50, 0.1)

#Propagandist
# plot_brier_ratio(10, 0.01, df1, df4, df5, 'With propagandist', 'With propagandist and centrist',
#                  'Propagandist and centrist')
# plot_brier_ratio(20, 0.01, df1, df4, df5, 'With propagandist', 'With propagandist and centrist',
#                  'Propagandist and centrist') # Centrist only slightly helps against propagandist (credence spikign only?)
# plot_brier_ratio(20, 0.05, df1, df4, df5, 'With propagandist', 'With propagandist and centrist',
#                  'Propagandist and centrist') # Centrist balances out propagandist
# plot_brier_ratio(20, 0.1, df1, df4, df5, 'With propagandist', 'With propagandist and centrist',
#                  'Propagandist and centrist') # Propagandist and centrist outperforms baseline

plot_brier_ratio(20, 0.1, df1, df2, df5, 'Centrist', 'With propagandist and centrist',
                 'Propagandist and centrist')