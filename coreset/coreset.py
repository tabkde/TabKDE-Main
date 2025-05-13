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

from tabkde.tools_1 import *
from tabkde.tools import *
from utils import (
    KDEFunction,
    compute_loss,
    create_dataloader,
    load_files,
    train_kde,
    generate_supervised_data_sigma_scheduler,
    get_sigma_t
)

def load_model(name, path):
    full_path = os.path.join(path, f"{name}.joblib")
    return joblib.load(full_path)

def load_all_models(dataname, parent_dir):
    model_path = f'{parent_dir}/tabkde/copula_encoding/ckpt/{dataname}/'
    cprint(f'Downloading the models from {model_path}', color='green')
    copula_model = load_model(name='copula_model', path=model_path)
    initial_encoder_model = load_model(name='data_aware_model', path=model_path)
    mix_gaussian_model = load_model(name='gauss_model', path=model_path)
    return initial_encoder_model, copula_model, mix_gaussian_model

def main(args):
    # Set precision
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

    json_path = os.path.join(parent_dir, f'data/{args.dataname}/info.json')
    data_path = os.path.join(parent_dir, f'data/{args.dataname}/')
    data_sets = ['train.csv', 'test.csv']

    loaded_data = load_files(data_path, *data_sets)
    real = loaded_data['train']
    test = loaded_data['test']

    initial_encoder_model, copula_model, mix_gaussian_model = load_all_models(args.dataname, parent_dir)
    real_copula_encoded = torch.tensor(copula_model.df_ranks.to_numpy(), dtype=torch_dtype, device='cpu')

    sigma = lambda t, T: get_sigma_t(t, T, scheduler=args.scheduler)
    print(f'bandwith = {args.bandwidth}')

    X, y = generate_supervised_data_sigma_scheduler(
        real_copula_encoded,
        N=args.n_samples,
        bandwidth=args.bandwidth,
        sampled_from=None,
        sigma_schedule=sigma,
        steps=args.steps,
        T=args.T
    )

    

    indices = np.random.choice(real_copula_encoded.shape[0], size=args.coreset_size, replace=True)
    coreset_points_init = torch.tensor(real_copula_encoded[indices], dtype=torch_dtype, device='cuda')


    train_loader = create_dataloader(X, y, batch_size=args.batch_size, dtype=torch_dtype)

    model = KDEFunction(
        n=args.coreset_size,
        d=X.shape[1],
        bandwidth=args.bandwidth,
        coreset_points_init=coreset_points_init
    ).to('cuda')

    scheduler_parameters = {
        'mode': 'min',
        'factor': 0.5,
        'patience': 5,
        'min_lr': 1e-5,
    }

    t_start = time.time()
    model, loss_history, _ = train_kde(
        model,
        train_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device='cuda',
        clamp=True,
        chunk_len=10,
        lambda_reg=args.lambda_reg,
        log_weights=False,
        track_loss=True,
        dtype=torch_dtype,
        **scheduler_parameters
    )

    cprint(f'Time to train: {(time.time() - t_start) / 60:.2f} min', bold=True, color='red')

    loss_dir = os.path.join('loss_history')
    os.makedirs(loss_dir, exist_ok=True)
    loss_path = os.path.join(loss_dir, f"{args.dataname}_{args.n_samples}_{args.coreset_size}.csv")
    pd.DataFrame({'loss': loss_history}).to_csv(loss_path, index=False)
    cprint(f"Loss history saved to {loss_path}", color='blue')

    X_ = model.X.detach().cpu().numpy()
    real_encoded = pd.DataFrame(real_copula_encoded)
    X_df = pd.DataFrame(X_, columns=real_encoded.columns)

    X_for_sampling = X_df.sample(n=real_encoded.shape[0], random_state=42, replace=True).values
    X_prime = perturb_vector_with_partial_resampling_with_DCP(
        X_for_sampling,
        sigma='global',
        model=mix_gaussian_model,
        allowed_outside_boundary=False
    )

    synth_data_encoded_sample_KDE = copula_model.convert(X_prime)
    synth = initial_encoder_model.decode(synth_data_encoded_sample_KDE)

    save_path = os.path.join(parent_dir, f'synthetic/{args.dataname}/')
    os.makedirs(save_path, exist_ok=True)
    synth.to_csv(os.path.join(save_path, 'coreset.csv'), index=False)
    cprint(f'synthetic data was saved on {save_path}', color = 'red', bold= True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KDE model with coreset initialisation")
    parser.add_argument('--dataname', type=str, default='adult')
    parser.add_argument('--n_samples', type=int, default=300000)
    parser.add_argument('--bandwidth', type=float, default=0.2)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--coreset_size', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--lambda_reg', type=float, default=1e-2)
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--precision', type=str, default='float32', choices=['float32', 'float64'], help='Choose float precision')
    args = parser.parse_args()
    main(args)
