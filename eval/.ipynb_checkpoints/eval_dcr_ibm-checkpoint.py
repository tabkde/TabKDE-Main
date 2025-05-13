import numpy as np
import torch 
import pandas as pd
import json
import os
import sys
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
from sklearn.neighbors import KDTree
import argparse
import csv
from sklearn import __version__ as sklearn_version
from packaging import version

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_train import preprocess, TabularDataset

pd.options.mode.chained_assignment = None

def save_dcr_score(file_path, score, dcrs_real, dcrs_test, csv_file):
    score_str = f"{float(score):.2f}"
    with open(file_path, 'w') as f:
        f.write(score_str)
    print(f"Score '{score_str}' written to '{file_path}'")

    df = pd.DataFrame({'syn_to_real': dcrs_real, 'syn_to_test': dcrs_test})
    df.to_csv(csv_file, index=False)
    print(f'DCR distributions saved at {csv_file}')

def compute_dcr_scores_kdtree(real_np, test_np, syn_np, batch_size=100):
    tree_real = KDTree(real_np, metric='manhattan')
    tree_test = KDTree(test_np, metric='manhattan')
    print('KDTree structure has been created')

    dcrs_real = []
    dcrs_test = []
    t = time.time()

    for start in range(0, syn_np.shape[0], batch_size):
        syn_batch = syn_np[start:start + batch_size]
        dist_real, _ = tree_real.query(syn_batch, k=1)
        dist_test, _ = tree_test.query(syn_batch, k=1)

        dcrs_real.append(dist_real.flatten())
        dcrs_test.append(dist_test.flatten())
        print(f'One round query has been done in {time.time()-t_0} seconds')

    dcrs_real = torch.tensor(np.concatenate(dcrs_real))
    dcrs_test = torch.tensor(np.concatenate(dcrs_test))

    return dcrs_real, dcrs_test

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='adult')
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--path', type=str, default=None, help='The file path of the synthetic data')
args = parser.parse_args()

if __name__ == '__main__':
    dataname = args.dataname
    model = args.model
    syn_path = args.path if args.path else f'synthetic/{dataname}/{model}.csv'
    real_path = f'synthetic/{dataname}/real.csv'
    test_path = f'synthetic/{dataname}/test.csv'
    data_dir = f'data/{dataname}'

    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    syn_data = pd.read_csv(syn_path)
    real_data = pd.read_csv(real_path)
    test_data = pd.read_csv(test_path)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']
    task_type = info['task_type']

    if task_type == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    real_data.columns = list(np.arange(len(real_data.columns)))
    syn_data.columns = list(np.arange(len(real_data.columns)))
    test_data.columns = list(np.arange(len(real_data.columns)))

    num_real_data_np = real_data[num_col_idx].to_numpy()
    num_syn_data_np = syn_data[num_col_idx].to_numpy()
    num_test_data_np = test_data[num_col_idx].to_numpy()

    cat_real_data_np = real_data[cat_col_idx].to_numpy().astype(str)
    cat_syn_data_np = syn_data[cat_col_idx].to_numpy().astype(str)
    cat_test_data_np = test_data[cat_col_idx].to_numpy().astype(str)

    if version.parse(sklearn_version) >= version.parse("1.2"):
        encoder = OneHotEncoder(sparse_output=True)
    else:
        encoder = OneHotEncoder(sparse=True)

    encoder.fit(cat_real_data_np)

    cat_real_data_oh = encoder.transform(cat_real_data_np)
    cat_syn_data_oh = encoder.transform(cat_syn_data_np)
    cat_test_data_oh = encoder.transform(cat_test_data_np)

    num_ranges = np.array([real_data[i].max() - real_data[i].min() for i in num_col_idx])
    num_real_data_norm = num_real_data_np / num_ranges
    num_syn_data_norm = num_syn_data_np / num_ranges
    num_test_data_norm = num_test_data_np / num_ranges

    num_real_sparse = sparse.csr_matrix(num_real_data_norm)
    num_syn_sparse = sparse.csr_matrix(num_syn_data_norm)
    num_test_sparse = sparse.csr_matrix(num_test_data_norm)

    real_sparse = sparse.hstack([num_real_sparse, cat_real_data_oh], format='csr')
    syn_sparse = sparse.hstack([num_syn_sparse, cat_syn_data_oh], format='csr')
    test_sparse = sparse.hstack([num_test_sparse, cat_test_data_oh], format='csr')

    dcrs_real, dcrs_test = compute_dcr_scores_kdtree(
        real_sparse.toarray(), test_sparse.toarray(), syn_sparse.toarray()
    )

    score = (dcrs_real < dcrs_test).sum().item() / len(dcrs_real)
    print('DCR Score, a value closer to 0.5 is better')
    print(f'{dataname}-{model}, DCR Score = {score}')

    current_path = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(current_path, f'DCR_scores/{dataname}')
    csv_path = os.path.join(dir_path, 'distributions')
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(csv_path, exist_ok=True)

    file_path = os.path.join(dir_path, f'{model}_DCR_score.txt')
    csv_file = os.path.join(csv_path, f'{model}.csv')
    save_dcr_score(file_path, score, dcrs_real.tolist(), dcrs_test.tolist(), csv_file)
