import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import json


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


def heat_map_plot(divergence, show_numbers=False, vmin=0, vmax=1, title=None,
                  feature_names=None, path_to_save=None, show_axis_name=True, show_colorbar=True):

    if feature_names:
        if len(feature_names) != divergence.shape[0]:
            raise ValueError("Length of feature_names must match the dimensions of divergence matrix.")
        divergence = pd.DataFrame(divergence, index=feature_names, columns=feature_names)

    plt.figure(figsize=(10, 8))
    sns.heatmap(divergence, annot=show_numbers, cmap="Greens", cbar=show_colorbar,
                fmt=".2f" if show_numbers else None, linewidths=0.5, linecolor='white',
                vmin=vmin, vmax=vmax)

    if title != 'empty':
        plt.title(f"Absolute Divergence in Pairwise Correlations\n({title})" if title else "Absolute Divergence in Pairwise Correlations")

    if not show_axis_name:
        plt.xticks([])
        plt.yticks([])

    if path_to_save:
        os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
        plt.savefig(path_to_save, bbox_inches='tight')

    plt.close()


def main():
    # Methods_name = ['Tabsyn', 'Smote', 'CopulaDiff', 'VAETabKDE', 'SimpleKDE', 'TabKDE']
    # Folders_name = ['test', 'tabsyn', 'smote', 'diffusion_on_copula', 'KDE_VAE_encoding', 'simple_KDE', 'TabKDE']
    # Datas_name = ['adult', 'default', 'shoppers', 'magic', 'beijing', 'news', 'ibm_func']

    # Methods_name = ['Tabsyn', 'CopulaDiff', 'TabKDE']
    # Folders_name = ['tabsyn', 'copulaDiff', 'TabKDE']
    # Datas_name = ['adult', 'default', 'shoppers', 'magic', 'beijing', 'news', 'ibm_func']


    Methods_name = ['CoDi', 'TabDDPM', 'Tabsyn', 'CopulaDiff', 'CoresetTabKDE', 'TabKDE']
    Folders_name = ['codi', 'tabddpm', 'tabsyn', 'copulaDiff', 'coreset', 'TabKDE']
    # Datas_name = ['ibm_func']
    Datas_name = ['adult', 
                  'default', 
                  'shoppers', 
                  'magic', 
                  'beijing', 
                  # 'news', 
                  'ibm_func'
                 ]

    
    # Datas_name = ['default']
    show_numbers = True
    show_axis_name = True
    

    base_path = '/tabkde-main/eval/density/'
        
    All_Divergences = {name: {} for name in Datas_name}
    features_name = {}

    for data_name in Datas_name:
        meta_path = os.path.join(base_path, data_name, 'tabsyn', 'metadata.json')
        features_name[data_name] = load_json(meta_path)["column_names"]

        for i, folder in enumerate(Folders_name):
            path = os.path.join(base_path, data_name, folder, 'trend.csv')
            if os.path.exists(path):
                All_Divergences[data_name][Methods_name[i]] = load_divergence_matrix(pd.read_csv(path))
            else:
                print(f'{path} does not exist')
        save_path = os.path.join('all_in_one_plots', 'all.png')
    print(save_path)
    for data_name in Datas_name:
        for method in Methods_name:
            if method in All_Divergences[data_name]:
                show_colorbar = True if method == 'TabKDE' else False
                save_path = os.path.join(data_name, 'GPU/with_names' if show_axis_name else 'GPU/no_names', 
                                         'plot_divergences', f'{method}.png')
                heat_map_plot(All_Divergences[data_name][method],
                              show_numbers=show_numbers,
                              vmin=0,
                              vmax=1,
                              title='empty',
                              feature_names=features_name[data_name],
                              path_to_save=save_path,
                              show_axis_name=show_axis_name,
                              show_colorbar=True
                             )


if __name__ == "__main__":
    main()
