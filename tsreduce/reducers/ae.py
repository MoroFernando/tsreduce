"""Autoencoder (AE)"""
import numpy as np

from ..base import BaseReducer


def _build_model(input_dim, latent_dim, dropout):
    # The model is defined here, not at module level, so importing this module
    # does not import torch — the heavy import happens only when the reducer runs.
    import torch.nn as nn

    def clamp(value, lo, hi):
        return max(lo, min(value, hi))

    class DenseAE(nn.Module):

        def __init__(self):
            super().__init__()
            h1 = clamp(input_dim // 2, latent_dim + 1, 256)
            h2 = clamp(input_dim // 4, latent_dim + 1, 128)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, h1), nn.ELU(), nn.Dropout(dropout),
                nn.Linear(h1, h2), nn.ELU(), nn.Dropout(dropout),
                nn.Linear(h2, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, h2), nn.ELU(), nn.Dropout(dropout),
                nn.Linear(h2, h1), nn.ELU(), nn.Dropout(dropout),
                nn.Linear(h2, h1), nn.ELU(), nn.Dropout(dropout),
                nn.Linear(h1, input_dim),
            )

        def forward(self, x):
            latent = self.encoder(x)
            return self.decoder(latent), latent

    return DenseAE()


class AE(BaseReducer):
    """Dense autoencoder with a fully-connected encoder–decoder.

    The bottleneck layer has *n_timepoints_out_* units and produces the latent
    representation. Channels are flattened into the batch axis before training
    and restored at transform time.
    """

    def __init__(self, *, target_len=None, retention_rate=None,
                 epochs=50, lr=1e-3, batch_size=32, dropout=0.1,
                 verbose: bool = False, random_state=None):
        super().__init__(target_len=target_len, retention_rate=retention_rate, verbose=verbose)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.random_state = random_state

    def _fit(self, X: np.ndarray, y=None) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from tqdm.auto import tqdm

        N = X.shape[2]
        w = self.n_timepoints_out_
        if w >= N:
            raise ValueError(
                f"AE needs a bottleneck smaller than the series: w={w}, N={N}."
            )

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = _build_model(N, w, self.dropout).to(self.device_)

        # Dense input is (batch, N) — flatten samples and channels into one batch.
        X_tensor = torch.from_numpy(X.reshape(-1, N).astype(np.float32)).to(self.device_)
        loader = DataLoader(TensorDataset(X_tensor), batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        self.model_.train()
        with tqdm(range(self.epochs), desc="AE", disable=not self.verbose) as pbar:
            for _ in pbar:
                epoch_loss = 0.0
                for (batch,) in loader:
                    optimizer.zero_grad()
                    recon, _ = self.model_(batch)
                    loss = criterion(recon, batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                scheduler.step()
                pbar.set_postfix(loss=f"{epoch_loss / len(loader):.4f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _transform(self, X: np.ndarray) -> np.ndarray:
        import torch

        n_samples, n_channels, N = X.shape
        w = self.n_timepoints_out_
        X_flat = X.reshape(-1, N).astype(np.float32)

        tensor = torch.from_numpy(X_flat).to(self.device_)
        self.model_.eval()
        with torch.no_grad():
            _, latent = self.model_(tensor)
        reduced = latent.cpu().numpy().reshape(-1, w)

        return reduced.reshape(n_samples, n_channels, w)
