import os
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


def _2_DCR_dist_comparison_noninteractive(to_real, to_test, show=True, title=None, legend_title=None, save_path=None):
    # Estimate KDE
    kde_real = gaussian_kde(to_real)
    kde_test = gaussian_kde(to_test)

    x = np.linspace(min(min(to_real), min(to_test)), max(max(to_real), max(to_test)), 1000)
    y_real = kde_real(x)
    y_test = kde_test(x)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, y_real, label='Synth_to_Train', color='blue')
    ax.fill_between(x, y_real, alpha=0.2, color='blue')

    ax.plot(x, y_test, label='Synth_to_Test', color='red')
    ax.fill_between(x, y_test, alpha=0.2, color='red')

    ax.set_title(title)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Density")
    ax.legend(title=legend_title)

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
    all_dists = {}

    for data_name in data_names:
        all_dists[data_name] = {}
        for method_file in method_files:
            file_path = os.path.join(base_path, data_name, f"distributions/{method_file}.csv")
            print(file_path)
            df = pd.read_csv(file_path)
            all_dists[data_name][method_file] = df

    return all_dists


# === Configuration ===
base_path = '/tabkde-main/eval/DCR_scores'

methods_1 = ['smote', 'tabsyn', 'coreset', 'TabKDE']
data_names = ['adult_equal', 'default_equal', 'shoppers_equal', 'magic_equal', 'beijing_equal', 'news_equal']
plot_titles = ['SMOTE', 'TabSYN', 'CoresetTabKDE', 'TabKDE']

# === Load Distributions ===
all_distributions = extract_distributions(base_path, data_names, method_files=methods_1)
# print(all_distributions)
DCRs_all_data = {}
names_distributions = {}

# for name in data_names:
#     DCRs_all_data[name] = []
#     names_distributions[name] = []
#     for m in methods_1:
#         DCRs_all_data[name].append(all_distributions[name][m]['syn_to_real'].to_numpy())
#         names_distributions[name].append(f'syn_to_real: {name[:-6]}-{m}')
#         DCRs_all_data[name].append(all_distributions[name][m]['syn_to_test'].to_numpy())
#         names_distributions[name].append(f'syn_to_test: {name[:-6]}-{m}')


# # === Generate Plots ===
# for d in range(len(data_names)):
#     for j, i in enumerate([0, 2, 4, 6]):  
#         save_path = os.path.join("DCR_plots", f"{data_names[d]}", f"{plot_titles[j]}.png")
#         _2_DCR_dist_comparison_noninteractive(
#             DCRs_all_data[data_names[d]][i],
#             DCRs_all_data[data_names[d]][i + 1],
#             legend_title=data_names[d],
#             title=f"{plot_titles[j]} - {data_names[d]}",
#             save_path=save_path,
#             show=False
#         )


# === Prepare DCR Data and Labels ===
for name in data_names:
    print(name)
    DCRs_all_data[name], names_distributions[name] = [], []
    for method in methods_1:
        print(method)
        for dist_type in ['syn_to_real', 'syn_to_test']:
            DCRs_all_data[name].append(all_distributions[name][method][dist_type].to_numpy())
            tag = dist_type.replace('_', ' ')
            names_distributions[name].append(f'{tag}: {name}-{method}')

name_map = {
'adult_equal': 'Adult',
'default_equal': 'Default',
'shoppers_equal': 'Shoppers',
'magic_equal': 'Magic',
'news_equal': 'News',
'beijing_equal': 'Beijing'
}

# === Generate and Save Plots ===
for name in data_names:
    for j, idx in enumerate(range(0, len(DCRs_all_data[name]), 2)):
        save_path = os.path.join("DCR_plots", name, f"{plot_titles[j]}.png")
        print(save_path)
        _2_DCR_dist_comparison_noninteractive(
            DCRs_all_data[name][idx],
            DCRs_all_data[name][idx + 1],
            legend_title=name_map[name],
            title=f"{plot_titles[j]} - {name_map[name]}",
            save_path=save_path,
            show=False
        )

