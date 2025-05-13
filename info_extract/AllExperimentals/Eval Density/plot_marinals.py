import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde



import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import joblib

# Set up path
parent_dir = os.path.abspath("..")
sys.path.append(parent_dir)

# Import project modules
from tabsyn.tools_1 import *
from tabsyn.tools import *


def load_files(data_path, *file_names):
    """
    Load multiple CSV files into DataFrames.

    Parameters:
    - data_path (str): Base directory path where the files are stored.
    - file_names (str): Variable number of file names to be loaded.

    Returns:
    - dict: A dictionary where keys are file names and values are DataFrames. If loading fails, the value is None for that file.
    """
    dataframes = {}
    print(f'loading data from {data_path}')

    for file_name in file_names:
        file_path = data_path + file_name
        try:
            dataframes[file_name[:-4]] = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            dataframes[file_name[:-4]] = None

    return dataframes

def load_model(name, path):
    full_path = os.path.join(path, f"{name}.joblib")
    return joblib.load(full_path)

# def load_all_models(dataname, parent_dir):
#     model_path = f'{parent_dir}/tabsyn/copula_encoding/ckpt/{dataname}/'
#     cprint(f'Downloading the models from {model_path}', color='green')
#     copula_model = load_model(name='copula_model', path=model_path)
#     initial_encoder_model = load_model(name='ordinal_encoder_model', path=model_path)
#     mix_gaussian_model = load_model(name='gauss_model', path=model_path)
#     return initial_encoder_model, copula_model, mix_gaussian_model

    

def plot_marginals(dataframes, categorical_features, df_names=None, columns_to_plot=None,
                   plot_type='kde', title='Marginal Plots', alpha=0.6, alpha_shade=0.3,
                   save_path=None, shade=True):
    """
    Plot marginal distributions for numerical and categorical features.

    Parameters:
    - dataframes: List of pandas DataFrames to plot.
    - categorical_features: List of column names considered as categorical.
    - df_names: List of names corresponding to each DataFrame.
    - columns_to_plot: List of columns to plot. If None, plot all columns.
    - plot_type: 'kde' or 'hist' for numerical distribution.
    - title: Title of the plot.
    - alpha: Transparency level for curve lines.
    - alpha_shade: Transparency level for shaded area (only for first KDE of each DataFrame).
    - save_path: Path to save the plots.
    - shade: Whether to include shaded area under KDE for the first plot per DataFrame.
    """
    if not isinstance(dataframes, list):
        raise ValueError("dataframes must be a list of pandas DataFrames.")
    if df_names and len(df_names) != len(dataframes):
        raise ValueError("Length of df_names must match the number of dataframes.")

    if df_names is None:
        df_names = [f'DF {idx + 1}' for idx in range(len(dataframes))]

    all_columns = dataframes[0].columns
    columns_to_plot = columns_to_plot if columns_to_plot else all_columns

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
                df_name = df_names[idx]
                value_counts = df[feature].value_counts(normalize=True).reindex(categories, fill_value=0)
                bar_position = positions + (idx * bar_width) - (bar_width * len(dataframes) / 2)
                plt.bar(bar_position, value_counts.values, width=bar_width, alpha=alpha, label=df_name,
                        color=colour_palette[idx])

            plt.xticks(positions, categories)

        elif feature in num_features:
            for idx, df in enumerate(dataframes):
                data = df[feature].dropna()
                df_name = df_names[idx]
                colour = colour_palette[idx]

                if plot_type == 'kde':
                    kde = gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 500)
                    y_vals = kde(x_range)

                    if shade and idx == 0:
                        plt.fill_between(x_range, y_vals, color=colour, alpha=alpha_shade)

                    plt.plot(x_range, y_vals, color=colour, alpha=alpha, label=df_name)
                elif plot_type == 'hist':
                    plt.hist(data, bins=30, alpha=alpha, label=df_name,
                             density=True, color=colour)
                else:
                    raise ValueError("plot_type must be either 'kde' or 'hist'")

        plt.title(f'{feature} - {title}')
        plt.xlabel(feature)
        plt.ylabel('Density' if feature in num_features else 'Frequency')
        plt.legend()
        plt.tight_layout()

        if save_path:
            file_name = os.path.join(save_path, f'{feature}.png')
            plt.savefig(file_name)

        plt.show()



data_sets = ['real.csv', 
             'tabsyn.csv', 
             'copulaDiff.csv',
             'smote.csv', 
             'TabKDE.csv', 
             'simple_KDE.csv',
             'coreset.csv',
             'tabddpm.csv'
]

# Load all specified files
loaded_data = load_files(data_path, *data_sets)

# Extract datasets individually
real = loaded_data.get("real")
test = loaded_data.get("test")
Tabsyn = loaded_data.get("tabsyn")
CopulaDiff = loaded_data.get("copulaDiff")
TabKDE = loaded_data.get("TabKDE")
Simple_KDE = loaded_data.get("simple_KDE")
Smote = loaded_data.get("smote")
Coreset = loaded_data.get("coreset")
Tabddpm = loaded_data.get("tabddpm")

# Confirm successful loading
if loaded_data is not None:
    print("✅ Files loaded successfully.\n")
    print(f"real shape: {real.shape}")
    print(f"test shape: {test.shape}")
    print(f"tabsyn shape: {Tabsyn.shape}")
    print(f"CopulaDiff shape: {CopulaDiff.shape}")
    print(f"TabKDE shape: {TabKDE.shape}")
    print(f"Simple_KDE shape: {Simple_KDE.shape}")
    print(f"Smote shape: {Smote.shape}")
    print(f"Coreset shape: {Coreset.shape}")
    print(f"Tabddpm shape: {Tabddpm.shape}")
else:
    print("❌ Failed to load files.")


base_data_path = '/tabkde-main/synthetic/'
path_to_save = '/tabkde-main/KDE/All_Mardinals/'

Datasets = ['adult', 'magic', 'news', 'shoppers', 'beijing', 'default', 'ibm_func']
Datanames_to_show = {'adult':'Adult', 
                     'magic': 'Magic', 
                     'news':'News', 
                     'shoppers':'Shoppers', 
                     'beijing':'Beijing', 
                     'default':'Default', 
                     'ibm_func': 'IBM'
                    }

dataframes = [real,  Tabsyn, TabKDE ]
df_names= ["Real", "Tabsyn", "TabKDE"]



for data_name in Datasets:
    print(f'Generating Marginal plots for {Datanames_to_show[data_name]}')
    model_path = model_path = f'/tabkde-main/tabsyn/copula_encoding/ckpt/{dataname}/'
    initial_encoder_model = load_model(name='ordinal_encoder_model', path=model_path)
    columns, cat_columns, num_columns = ordinal_encoder_model.get_columns()
    plot_marginals(dataframes, categorical_features = cat_columns, 
                   plot_type='kde', 
                   title=f'{Datanames_to_show[data_name]}', 
                   columns_to_plot= num_columns + Cat_columns_to_plot ,
                   df_names= df_names, alpha=1, shade = True, 
                    save_path =f'Eval Density/{data_name}/marginals')






