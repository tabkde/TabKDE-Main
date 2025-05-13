#!/usr/bin/env python3

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
from utils import load_files


def load_model(name, path):
    full_path = os.path.join(path, f"{name}.joblib")
    return joblib.load(full_path)

def load_all_models(dataname, parent_dir):
    model_path = f'{parent_dir}/tabsyn/copula_encoding/ckpt/{dataname}/'
    cprint(f'Downloading the models from {model_path}', color='green')
    copula_model = load_model(name='copula_model', path=model_path)
    initial_encoder_model = load_model(name='data_aware_model', path=model_path)
    mix_gaussian_model = load_model(name='gauss_model', path=model_path)
    return initial_encoder_model, copula_model, mix_gaussian_model

def main(args):
    # Set precision
    np.random.seed(42)
    if args.precision == 'float64':
        torch.set_default_dtype(torch.float64)
        dtype = np.float64
        torch_dtype = torch.float64
    else:
        torch.set_default_dtype(torch.float32)
        dtype = np.float32
        torch_dtype = torch.float32

    cprint(f'Using precision: {args.precision}', color='cyan')

    cprint(f'Current directory is {os.getcwd()}', color='green')
    parent_dir = os.path.abspath("..")
    cprint(f'Parent directory is {parent_dir}', color='red')

    json_path = os.path.join(parent_dir, f'data/{args.data_name}/info.json')
    data_path = os.path.join(parent_dir, f'data/{args.data_name}/')
    data_sets = ['train.csv', 'test.csv']

    loaded_data = load_files(data_path, *data_sets)
    real = loaded_data['train']
    test = loaded_data['test']

    initial_encoder_model, copula_model, mix_gaussian_model = load_all_models(args.data_name, parent_dir)
    real_copula_encoded = torch.tensor(copula_model.df_ranks.to_numpy(), dtype=torch_dtype, device='cpu')
    

    indices = np.random.choice(real_copula_encoded.shape[0], size=args.coreset_size, replace=True)
    X_  = real_copula_encoded[indices]
    real_encoded = pd.DataFrame(real_copula_encoded)
    X_df = pd.DataFrame(X_, columns=real_encoded.columns)
    print(len(X_df))

    X_for_sampling = X_df.sample(n=real_encoded.shape[0], random_state=42, replace=True).values
    X_prime = perturb_vector_with_partial_resampling_with_DCP(
        X_for_sampling,
        sigma='global',
        model=mix_gaussian_model,
        allowed_outside_boundary=False
    )

    synth_data_encoded_sample_KDE = copula_model.convert(X_prime)
    synth = initial_encoder_model.decode(synth_data_encoded_sample_KDE)

    save_path = os.path.join(parent_dir, f'synthetic/{args.data_name}/')
    os.makedirs(save_path, exist_ok=True)
    synth.to_csv(os.path.join(save_path, 'coreset_random.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KDE model with coreset initialisation")
    parser.add_argument('--data_name', type=str, default='adult')
    parser.add_argument('--coreset_size', type=int, default=5000)
    parser.add_argument('--precision', type=str, default='float32', choices=['float32', 'float64'], help='Choose float precision')
    args = parser.parse_args()
    main(args)
