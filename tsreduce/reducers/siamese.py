"""Siamese distance-regression encoder"""
import numpy as np

from ..base import BaseReducer


def _build_model(target_len, dropout):
    import torch.nn as nn

    class SiameseEncoder(nn.Module):

        def __init__(self):
            super().__init__()
            self.enc_conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
            self.enc_conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
            self.enc_conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(target_len)
            self.latent_conv = nn.Conv1d(64, 1, kernel_size=1)
            self.elu = nn.ELU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            h1 = self.dropout(self.elu(self.enc_conv1(x)))
            h2 = self.dropout(self.elu(self.enc_conv2(h1)))
            h3 = self.elu(self.enc_conv3(h2))
            return self.latent_conv(self.pool(h3))

    return SiameseEncoder()


def _pairwise_distance(z):
    """Full (B, B) Euclidean distance matrix between flattened embeddings/series."""
    import torch
    flat = z.reshape(z.shape[0], -1)
    return torch.cdist(flat, flat)


def _normalize_distance(d):
    """Min-max scale a 1-D distance tensor to [0, 1]."""
    if d.numel() <= 1:
        return d / d
    lo, hi = d.min(), d.max()
    return (d - lo) / (hi - lo)


class Siamese(BaseReducer):

    def __init__(self, *, target_length=None, retention_rate=None,
                 epochs=50, lr=1e-3, batch_size=32, dropout=0.1,
                 verbose=0, random_state=None):
        super().__init__(target_length=target_length, retention_rate=retention_rate)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.verbose = verbose
        self.random_state = random_state

    def _fit(self, X: np.ndarray) -> None:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
        from tqdm.auto import tqdm

        N = X.shape[2]
        w = self.n_timepoints_out_

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = _build_model(w, self.dropout).to(self.device_)

        X_tensor = torch.from_numpy(X.reshape(-1, N).astype(np.float32)).to(self.device_).unsqueeze(1)
        loader = DataLoader(TensorDataset(X_tensor), batch_size=self.batch_size,
                            shuffle=True, drop_last=True)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        self.model_.train()
        for _ in tqdm(range(self.epochs), desc="Siamese fit", disable=not self.verbose):
            for (batch,) in loader:
                if batch.shape[0] < 2:
                    continue
                optimizer.zero_grad()
                z = self.model_(batch)
                with torch.no_grad():
                    dist_raw = _pairwise_distance(batch)
                dist_emb = _pairwise_distance(z)
                mask = torch.tril(torch.ones_like(dist_emb), diagonal=-1).bool()
                loss = F.smooth_l1_loss(
                    _normalize_distance(torch.masked_select(dist_emb, mask)),
                    _normalize_distance(torch.masked_select(dist_raw, mask)),
                )
                loss.backward()
                optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _transform(self, X: np.ndarray) -> np.ndarray:
        import torch

        n_samples, n_channels, N = X.shape
        w = self.n_timepoints_out_

        tensor = torch.from_numpy(X.reshape(-1, N).astype(np.float32)).to(self.device_).unsqueeze(1)
        self.model_.eval()
        with torch.no_grad():
            latent = self.model_(tensor)

        return latent.squeeze(1).cpu().numpy().reshape(n_samples, n_channels, w)
