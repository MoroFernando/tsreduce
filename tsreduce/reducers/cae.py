"""Convolutional Autoencoder (CAE)"""
import numpy as np

from ..base import BaseReducer


def _build_model(input_dim, target_len, dropout):
    import torch.nn as nn

    class ConvAE(nn.Module):

        def __init__(self):
            super().__init__()
            self.enc_conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
            self.enc_conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
            self.enc_conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(target_len)
            self.latent_conv = nn.Conv1d(64, 1, kernel_size=1)
            self.dec_expand = nn.Conv1d(1, 64, kernel_size=1)
            self.dec_upsample = nn.Upsample(size=input_dim, mode="linear", align_corners=False)
            self.dec_conv1 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
            self.dec_conv2 = nn.Conv1d(64, 32, kernel_size=5, padding=2)
            self.dec_conv3 = nn.Conv1d(32, 1, kernel_size=7, padding=3)
            self.elu = nn.ELU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            h1 = self.dropout(self.elu(self.enc_conv1(x)))
            h2 = self.dropout(self.elu(self.enc_conv2(h1)))
            h3 = self.elu(self.enc_conv3(h2))
            pooled = self.pool(h3)
            latent = self.latent_conv(pooled)
            expanded = self.elu(self.dec_expand(latent))
            upsampled = self.dec_upsample(expanded)
            d1 = self.dropout(self.elu(self.dec_conv1(upsampled)))
            d2 = self.dropout(self.elu(self.dec_conv2(d1)))
            recon = self.dec_conv3(d2)
            return recon, latent

    return ConvAE()


class CAE(BaseReducer):

    def __init__(self, *, target_len=None, retention_rate=None,
                 epochs=50, lr=1e-3, batch_size=32, dropout=0.1,
                 verbose: int = 0, random_state=None):
        super().__init__(target_len=target_len, retention_rate=retention_rate)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.verbose = verbose
        self.random_state = random_state

    def _fit(self, X: np.ndarray, y=None) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from tqdm import tqdm

        N = X.shape[2]
        w = self.n_timepoints_out_

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = _build_model(N, w, self.dropout).to(self.device_)

        X_tensor = torch.from_numpy(X.reshape(-1, N).astype(np.float32)).to(self.device_).unsqueeze(1)
        loader = DataLoader(TensorDataset(X_tensor), batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model_.train()
        with tqdm(range(self.epochs), desc="CAE", disable=not self.verbose) as pbar:
            for _ in pbar:
                epoch_loss = 0.0
                for (batch,) in loader:
                    optimizer.zero_grad()
                    recon, _ = self.model_(batch)
                    loss = criterion(recon, batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{epoch_loss / len(loader):.4f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _transform(self, X: np.ndarray) -> np.ndarray:
        import torch

        n_samples, n_channels, N = X.shape
        w = self.n_timepoints_out_

        tensor = torch.from_numpy(X.reshape(-1, N).astype(np.float32)).to(self.device_).unsqueeze(1)
        self.model_.eval()
        with torch.no_grad():
            _, latent = self.model_(tensor)

        return latent.cpu().numpy().reshape(n_samples, n_channels, w)
