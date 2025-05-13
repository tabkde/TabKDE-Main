#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# Import project modules
from tabkde.tools_1 import *
from tabkde.tools import *

def load_files(data_path, *file_names):
    dataframes = {}
    print(f'📂 Loading data from: {data_path}')
    for file_name in file_names:
        file_path = os.path.join(data_path, file_name)
        try:
            dataframes[file_name[:-4]] = pd.read_csv(file_path)
        except Exception as e:
            print(f"❌ Error loading {file_name}: {e}")
            dataframes[file_name[:-4]] = None
    return dataframes

def load_model(name, path):
    return joblib.load(os.path.join(path, f"{name}.joblib"))

def plot_marginals(dataframes, categorical_features, df_names=None, columns_to_plot=None,
                   plot_type='kde', title='Marginal Plots', alpha=0.6, alpha_shade=0.3,
                   save_path=None, shade=True):

    if not isinstance(dataframes, list):
        raise ValueError("dataframes must be a list.")
    if df_names and len(df_names) != len(dataframes):
        raise ValueError("Length of df_names must match number of dataframes.")

    if df_names is None:
        df_names = [f'DF {i+1}' for i in range(len(dataframes))]

    all_columns = dataframes[0].columns
    columns_to_plot = columns_to_plot or all_columns

    if save_path:
        os.makedirs(save_path, exist_ok=True)

    num_features = list(set(columns_to_plot) - set(categorical_features))
    colour_palette = sns.color_palette(n_colors=len(dataframes))

    for feature in columns_to_plot:
        plt.figure(figsize=(8, 5))

        if feature in categorical_features:
            bar_width = 0.35 / len(dataframes)
            categories = dataframes[0][feature].dropna().unique()
            positions = np.arange(len(categories))

            for idx, df in enumerate(dataframes):
                value_counts = df[feature].value_counts(normalize=True).reindex(categories, fill_value=0)
                bar_pos = positions + (idx * bar_width) - (bar_width * len(dataframes) / 2)
                plt.bar(bar_pos, value_counts.values, width=bar_width, alpha=alpha,
                        label=df_names[idx], color=colour_palette[idx])
            plt.xticks(positions, categories)

        elif feature in num_features:
            for idx, df in enumerate(dataframes):
                data = df[feature].dropna()
                colour = colour_palette[idx]
                if plot_type == 'kde':
                    kde = gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 500)
                    y_vals = kde(x_range)
                    if shade and idx == 0:
                        plt.fill_between(x_range, y_vals, color=colour, alpha=alpha_shade)
                    plt.plot(x_range, y_vals, color=colour, alpha=alpha, label=df_names[idx])
                elif plot_type == 'hist':
                    plt.hist(data, bins=30, density=True, alpha=alpha,
                             label=df_names[idx], color=colour)
                else:
                    raise ValueError("plot_type must be 'kde' or 'hist'.")

        plt.title(f'{feature} - {title}')
        plt.xlabel(feature)
        plt.ylabel('Density' if feature in num_features else 'Frequency')
        plt.legend()
        plt.tight_layout()

        if save_path:
            clean_feature = feature.lstrip()  # Remove leading whitespace
            file_name = os.path.join(save_path, f'{clean_feature}.png')
            plt.savefig(file_name)

        plt.close()


if __name__ == "__main__":
    data_path = '/synthetic/'
    save_base_path = '/simple_vs_tabkde_marginals/'

    datasets = ['adult', 'magic', 'news', 'shoppers', 'beijing', 'default', 'ibm_func']
    display_names = {
        'adult': 'Adult', 'magic': 'Magic', 'news': 'News',
        'shoppers': 'Shoppers', 'beijing': 'Beijing', 'default': 'Default', 'ibm_func': 'IBM'
    }

    data_files = [
        'real.csv', 'simple_KDE.csv', 'TabKDE.csv'
    ]
    


    for dataname in datasets:
        print(f'\n📊 Generating plots for: {display_names[dataname]}')
        loaded_data = load_files(data_path +f'{dataname}/', *data_files)
        real = loaded_data.get("real")
        simplekde = loaded_data.get("simple_KDE")
        tabkde = loaded_data.get("TabKDE")
        dataframes = [real, simplekde, tabkde]
        df_names = ["Real", "SimpleKDE", "TabKDE"]

        
        model_path = f'/tabkde/copula_encoding/ckpt/{dataname}/'
        encoder = load_model(name='draw_aware_model', path=model_path)
        columns, cat_columns, num_columns = encoder.get_columns()

        save_path = os.path.join(save_base_path, dataname)
        plot_marginals(
            dataframes=dataframes,
            categorical_features=cat_columns,
            df_names=df_names,
            columns_to_plot=num_columns + cat_columns,
            plot_type='kde',
            title=display_names[dataname],
            alpha=1,
            alpha_shade=0.3,
            shade=True,
            save_path=save_path
        )
