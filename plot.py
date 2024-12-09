import pandas as pd
import matplotlib.pyplot as plt

from sim.simsetup import LIFECYCLE_FILENAME, LIFECYCLE_W_ALTERNATOR_SKEPTICS_FILENAME, LIFECYCLE_W_SKEPTICS_FILENAME

def plot_brier_ratio(scientist_init_popcount, epsilon, title = 'Effect of alternator centrist on network performance'):
    # Load the CSV files
    df1 = pd.read_csv(LIFECYCLE_FILENAME)
    df2 = pd.read_csv(LIFECYCLE_W_SKEPTICS_FILENAME)
    df3 = pd.read_csv(LIFECYCLE_W_ALTERNATOR_SKEPTICS_FILENAME)

    # Filter the data
    df1_filtered = df1[(df1['scientist_init_popcount'] == scientist_init_popcount) & (df1['epsilon'] == epsilon)]
    df2_filtered = df2[(df2['scientist_init_popcount'] == scientist_init_popcount) & (df2['epsilon'] == epsilon)]
    df3_filtered = df3[(df3['scientist_init_popcount'] == scientist_init_popcount) & (df3['epsilon'] == epsilon)]

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(df1_filtered['m'], df1_filtered['sims_av_brier_ratio'], label='No centrist', marker='o')
    plt.plot(df2_filtered['m'], df2_filtered['sims_av_brier_ratio'], label='With centrist', marker='s')
    plt.plot(df3_filtered['m'], df3_filtered['sims_av_brier_ratio'], label='With alternator centrist', marker='v')

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
plot_brier_ratio(10, 0.1)
# plot_brier_ratio(20, 0.01)
# plot_brier_ratio(20, 0.05)
# plot_brier_ratio(20, 0.1)
# plot_brier_ratio(50, 0.01)
# plot_brier_ratio(50, 0.05)
# plot_brier_ratio(50, 0.1)