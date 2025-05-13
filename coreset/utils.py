import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

def load_files(data_path, *file_names):
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

def KDE_custom(x, centers, bandwidth):
    B, d = x.shape
    n = centers.shape[0]
    x_exp = x.unsqueeze(1)
    X_exp = centers.unsqueeze(0)
    sq_dist = ((x_exp - X_exp) ** 2).sum(dim=2)
    log_kernel = -0.5 * sq_dist / bandwidth**2
    log_gauss_norm = -(d / 2) * torch.log(
        torch.tensor(2 * np.pi * bandwidth**2, dtype=x.dtype, device=x.device)
    )
    log_kernel = log_kernel + log_gauss_norm
    log_terms = log_kernel
    log_density = torch.logsumexp(log_terms, dim=1) - np.log(n)
    return log_density



def generate_supervised_data_sigma_scheduler(train_data, N=1000, bandwidth=0.1, sampled_from=None,
                                             sigma_schedule=None, steps=0, T=1000, kde_batch_size=1024,
                                             device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # Ensure inputs are on the correct device
    train_data = train_data.to(device)
    if sampled_from is not None:
        sampled_from = sampled_from.to(device)

    dtype = train_data.dtype
    d = train_data.shape[1]

    if sigma_schedule is None or steps < 1:
        if sampled_from is not None:
            indices = torch.randint(0, sampled_from.shape[0], (N,), device=device)
            X_sampled = sampled_from[indices]
        else:
            X_sampled = torch.rand(N, d, device=device, dtype=dtype)
        return X_sampled, batchwise_kde(X_sampled, train_data, bandwidth, kde_batch_size)

    t_vals = torch.linspace(0, T, steps + 1, device=device).floor().long()
    ns_per_step = [N // (2 ** (i + 1)) for i in range(steps)]
    ns_per_step.append(N - sum(ns_per_step))

    # print(f'the std values for different time steps are {[sigma_schedule(t.item(), T) for t in t_vals]}')
    # print(f'number of samples for each step =: {ns_per_step}')

    X_parts = []
    for i, t in enumerate(t_vals):
        sigma = sigma_schedule(t.item(), T)
        size = (ns_per_step[i], d)
        if sampled_from is not None:
            indices = torch.randint(0, sampled_from.shape[0], (ns_per_step[i],), device=device)
            base = sampled_from[indices]
        else:
            base = torch.rand(size, device=device, dtype=dtype)
        noise = torch.normal(mean=0.0, std=sigma, size=size, device=device, dtype=dtype)
        X_parts.append(base + noise)

    X = torch.cat(X_parts, dim=0)
    log_density = batchwise_kde(X, train_data, bandwidth, kde_batch_size)
    return X, log_density


def batchwise_kde(X, centers, bandwidth, batch_size=1024):
    centers = centers.to(X.device)  # Ensure both are on same device
    results = []
    with torch.no_grad():  # KDE is inference only
        for i in range(0, X.size(0), batch_size):
            xb = X[i:i+batch_size]
            log_dens = KDE_custom(xb, centers, bandwidth)
            results.append(log_dens)
    return torch.cat(results, dim=0)




def get_sigma_t(t, T=1000, scheduler='linear', beta_min=1e-4, beta_max=0.02, s=0.008):
    if t == 0:
        return 0.0
    if scheduler == 'linear':
        beta_t = torch.linspace(beta_min, beta_max, T)[t]
        beta_schedule = torch.linspace(beta_min, beta_max, t + 1)
        alpha_bar_t = torch.prod(1.0 - beta_schedule)
    elif scheduler == 'cosine':
        t_hat = (t + 1) / T
        f = lambda t_: torch.cos((t_ / (1 + s)) * torch.pi / 2) ** 2
        alpha_bar_t = f(torch.tensor(t_hat)) / f(torch.tensor(0.0))
    else:
        raise ValueError("Unsupported scheduler: choose 'linear' or 'cosine'")
    return torch.sqrt(1.0 - alpha_bar_t).item()

    

class KDEFunction(nn.Module):
    def __init__(self, n, d, bandwidth=0.1, use_log_softmax=True, coreset_points_init=None):
        super(KDEFunction, self).__init__()
        self.bandwidth = bandwidth
        self.use_log_softmax = use_log_softmax

        if coreset_points_init is not None:
            self.X = nn.Parameter(coreset_points_init.clone().detach())
        else:
            self.X = nn.Parameter(torch.rand(n, d))
        self.W = nn.Parameter(0.01 * torch.randn(n))

    def forward(self, x):
        B, d = x.shape
        n = self.X.shape[0]
        x_exp = x.unsqueeze(1)
        X_exp = self.X.unsqueeze(0)
        sq_dist = ((x_exp - X_exp) ** 2).sum(dim=2)
        log_kernel = -0.5 * sq_dist / self.bandwidth**2
        log_gauss_norm = -(d / 2) * torch.log(
            torch.tensor(2 * np.pi * self.bandwidth**2, dtype=x.dtype, device=x.device)
        )
        log_kernel = log_kernel + log_gauss_norm

        if self.use_log_softmax:
            log_weights = F.log_softmax(self.W, dim=0).unsqueeze(0)
        else:
            weights = F.softmax(self.W, dim=0).unsqueeze(0)
            log_weights = torch.log(weights + 1e-12)

        log_terms = log_kernel + log_weights
        log_density = torch.logsumexp(log_terms, dim=1)
        return log_density

    def density(self, x):
        return torch.exp(self.forward(x))

    def get_weighted_kernels(self, x):
        kernel = self._compute_kernel(x)
        weights = self.weights.unsqueeze(0)
        return kernel * weights

    def _compute_kernel(self, x):
        x_exp = x.unsqueeze(1)
        X_exp = self.X.unsqueeze(0)
        sq_dist = ((x_exp - X_exp) ** 2).sum(dim=2)
        return torch.exp(-0.5 * sq_dist / self.bandwidth**2)

    @property
    def weights(self):
        return F.softmax(self.W, dim=0)

    @property
    def centres(self):
        return self.X

def compute_loss(model, X, target_log_density, lambda_reg=1e-2, log_weights=False):
    log_pred = model(X)
    mse = F.mse_loss(log_pred, target_log_density)
    l2 = lambda_reg * model.W.pow(2).sum()
    total_loss = mse + l2

    if log_weights:
        with torch.no_grad():
            print("Sample softmax weights:", model.weights[:10].cpu().numpy())
            print("Sample log-predictions:", log_pred[:5].cpu().numpy())

    return total_loss

def create_dataloader(X, y, batch_size=64, shuffle=True, dtype=torch.float64):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=dtype)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=dtype)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_kde(model, train_loader, num_epochs=100, lr=1e-2, device='cpu',
              clamp=True, chunk_len=10, lambda_reg=1e-2, log_weights=False,
              track_loss=False, trak_eval_loss=False, dtype=torch.float64, **scheduler_parameters):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_parameters.get('mode', 'min'),
        factor=scheduler_parameters.get('factor', 0.5),
        patience=scheduler_parameters.get('patience', 5),
        min_lr=scheduler_parameters.get('min_lr', 1e-5)
    )

    loss_history = []
    eval_loss_history = []

    for chunk_start in range(0, num_epochs, chunk_len):
        chunk_end = min(chunk_start + chunk_len, num_epochs)
        chunk_range = range(chunk_start, chunk_end)

        with tqdm(total=len(chunk_range), desc=f"Epochs {chunk_start+1}-{chunk_end}", unit="epoch", ncols=100) as progress:
            for epoch in chunk_range:
                epoch_loss = 0.0
                model.train()
                X_all, y_all = [], []

                for i, (X_batch, y_batch) in enumerate(train_loader):
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device).to(dtype)

                    X_all.append(X_batch.detach().cpu())
                    y_all.append(y_batch.detach().cpu())

                    optimizer.zero_grad()
                    loss = compute_loss(model, X_batch, y_batch, lambda_reg=lambda_reg, log_weights=log_weights)
                    loss.backward()
                    optimizer.step()

                    if clamp:
                        with torch.no_grad():
                            model.X.data.clamp_(0.0, 1.0)

                    with torch.no_grad():
                        epoch_loss += loss.item() * X_batch.size(0)

                with torch.no_grad():
                    avg_loss = epoch_loss / len(train_loader.dataset)
                    scheduler.step(avg_loss)
                    current_lr = optimizer.param_groups[0]['lr']
                    if track_loss:
                        loss_history.append(avg_loss)
                    if trak_eval_loss:
                        full_X = torch.cat(X_all, dim=0).to(device)
                        full_y = torch.cat(y_all, dim=0).to(device)
                        eval_loss = evaluate_kde_loss(model, full_X, full_y, device=device)
                        eval_loss_history.append(eval_loss)
                        progress.set_postfix(loss=f"{avg_loss:.2f}", eval_loss=f"{eval_loss:.2f}", lr=f"{current_lr:.2e}")
                    else:
                        progress.set_postfix(loss=f"{avg_loss:.2f}", lr=f"{current_lr:.2e}")
                    progress.update(1)

    return model, loss_history, eval_loss_history