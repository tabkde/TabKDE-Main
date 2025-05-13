import os
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def _2_DCR_dist_comparison_noninteractive(to_real, to_test, show = True, title=None, legend_title= None, save_path=None):
    # Estimate KDE
    kde_real = gaussian_kde(to_real)
    kde_test = gaussian_kde(to_test)

    x = np.linspace(min(min(to_real), min(to_test)), max(max(to_real), max(to_test)), 1000)
    y_real = kde_real(x)
    y_test = kde_test(x)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, y_real, label='Synth_to_Real', color='blue')
    ax.fill_between(x, y_real, alpha=0.2, color='blue')

    ax.plot(x, y_test, label='Synth_to_Test', color='red')
    ax.fill_between(x, y_test, alpha=0.2, color='red')

    ax.set_title(title)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Density")
    ax.legend(title= legend_title)

    # Save if a path is given
    if save_path:
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')

    # Show the plot in all cases
    if show: 
        plt.show()
    plt.close(fig)

def extract_distributions(base_path, data_names, method_files):
    """
    Extracts numeric scores from text files.

    Args:
        base_path (str): The base directory path.
        data_names (list of str): List of dataset folder names.
        method_files (list of str): List of method names (each corresponds to a .txt file).

    Returns:
        pd.DataFrame: A DataFrame with datasets as rows, methods as columns, containing the extracted scores.
    """
    all_dists = {}

    for data_name in data_names:
        all_dists[data_name] = {}
        for method_file in method_files:
            file_path = os.path.join(base_path, data_name, f"distributions/{method_file}.csv")
            df = pd.read_csv(file_path)
            all_dists[data_name][method_file] = df

    return all_dists

base_path =  '/tabkde-main/eval/DCR_scores'


methods_1 = [  'smote', 'tabsyn', 'coreset', 'TabKDE'  ]
data_names = ['adult_equal', 'default_equal', 'shoppers_equal', 'magic_equal', 'beijing_equal', 'news_equal']


all_distributions = extract_distributions(base_path, data_names, method_files =  methods_1)

DCRs_all_data ={} 
names_distributions = {}
for name in data_names:
    DCRs_all_data[name] = []
    names_distributions[name] = []
    for m in methods_1:
        DCRs_all_data[name].append(all_distributions[name][m]['syn_to_real'].to_numpy())
        names_distributions[name].append(f'syn_to_real: {name[:-6]}-{m}' )
        DCRs_all_data[name].append(all_distributions[name][m]['syn_to_test'].to_numpy())
        names_distributions[name].append(f'syn_to_test: {name[:-6]}-{m}' )


for d in range(6):
    for j, i in enumerate([0, 4, 6]):
        _2_DCR_dist_comparison_noninteractive(DCRs_all_data[keys[d]][i], 
                                              DCRs_all_data[keys[d]][i+1], 
                                              legend_title = f'{data_sets[d][:-6}', 
                                              show = True, 
                                              save_path = f'{data_sets[d]}/{Title_of_plots[j]}.png'
                                             )