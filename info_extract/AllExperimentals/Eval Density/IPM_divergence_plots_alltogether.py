import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import json
import matplotlib.gridspec as gridspec


def load_files(data_path, *file_names):
    dataframes = {}
    print(f'Loading data from {data_path}')

    for file_name in file_names:
        file_path = os.path.join(data_path, file_name)
        try:
            dataframes[file_name[:-4]] = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            dataframes[file_name[:-4]] = None

    return dataframes


def load_divergence_matrix(df):
    features = sorted(set(df["Column 1"].unique()).union(set(df["Column 2"].unique())))
    feature_map = {feature: idx for idx, feature in enumerate(features)}
    size = len(features)
    matrix = np.zeros((size, size))

    for _, row in df.iterrows():
        i, j = feature_map[row["Column 1"]], feature_map[row["Column 2"]]
        value = 1 - row["Score"]
        matrix[i, j] = value
        matrix[j, i] = value

    return matrix


def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def plot_all_heatmaps_grid(All_Divergences, features_name, Datas_name, Methods_name,
                           vmin=0, vmax=1, show_numbers=False, save_path=None, data_names_to_show=None):
    if data_names_to_show is None:
        data_names_to_show = Datas_name

    n_rows = len(Datas_name)
    n_cols = len(Methods_name)

    fig = plt.figure(figsize=(4 * n_cols + 1, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols + 1, width_ratios=[1] * n_cols + [0.05],
                           wspace=0.1, hspace=0.1)

    for i, data_name in enumerate(Datas_name):
        for j, method in enumerate(Methods_name):
            ax = fig.add_subplot(gs[i, j])
            if method in All_Divergences[data_name]:
                div = All_Divergences[data_name][method]
                feature_names = features_name[data_name]
                if len(feature_names) == div.shape[0]:
                    div = pd.DataFrame(div, index=feature_names, columns=feature_names)

                sns.heatmap(div, ax=ax, cmap="Greens", cbar=False,
                            vmin=vmin, vmax=vmax,
                            annot=show_numbers,
                            fmt=".2f" if show_numbers else None,
                            square=True, linewidths=0.5, linecolor='white')
            else:
                ax.axis('off')

            if i == 0:
                ax.set_title(method, fontsize=12)
            if j == 0:
                # ax.set_ylabel(data_names_to_show[i], fontsize=12)
                ax.set_ylabel('IBM', fontsize=12)

            ax.set_xticks([])
            ax.set_yticks([])

    # Shared colourbar
    cbar_ax = fig.add_subplot(gs[:, -1])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap="Greens", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')

    plt.close()


def main():
    Methods_name = ['Tabsyn', 'PGE-Tabsyn', 'CopulaDiff (no modelling trick)','CopulaDiff', 'CoresetTabKDE', 'TabKDE']
    Folders_name = ['tabsyn', 'Ordinal_tabsyn', 'copulaDiff_no_trick',  'copulaDiff', 'coreset', 'TabKDE']
    Datas_name = ['ibm_func']
    # Datas_name = ['adult', 'default', 'shoppers', 'magic', 'beijing', 'news', 'ibm_func']

    base_path = '/tabkde-main/eval/density/'
    show_numbers = False

    All_Divergences = {name: {} for name in Datas_name}
    features_name = {}

    for data_name in Datas_name:
        meta_path = os.path.join(base_path, data_name, 'tabsyn', 'metadata.json')
        features_name[data_name] = load_json(meta_path)["column_names"]

        for i, folder in enumerate(Folders_name):
            path = os.path.join(base_path, data_name, folder, 'trend.csv')
            print(path)
            if os.path.exists(path):
                All_Divergences[data_name][Methods_name[i]] = load_divergence_matrix(pd.read_csv(path))
            else:
                print(f'{path} does not exist')

    save_path = os.path.join('all_in_one_plots', 'IBM.png')
    print(save_path)
    plot_all_heatmaps_grid(All_Divergences,
                           features_name,
                           Datas_name,
                           Methods_name,
                           vmin=0,
                           vmax=1,
                           show_numbers=show_numbers,
                           save_path=save_path
                          )


if __name__ == "__main__":
    main()
