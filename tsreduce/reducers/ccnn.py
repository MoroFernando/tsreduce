"""Contrastive CNN (CCNN)"""
import numpy as np

from ..base import BaseReducer


def _build_model(input_dim, target_len, dropout, queue_size=5000, temperature=0.1):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Augmenter(nn.Module):

        def __init__(self, noise_factor=0.1, dropout_rate=0.1):
            super().__init__()
            self.noise_factor = noise_factor
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x, training=True):
            if training:
                return self.dropout(x + torch.randn_like(x) * self.noise_factor)
            return x

    class CCNN(nn.Module):

        def __init__(self):
            super().__init__()
            self.enc_conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
            self.enc_conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
            self.enc_conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(target_len)
            self.latent_conv = nn.Conv1d(64, 1, kernel_size=1)
            self.augmenter = Augmenter(noise_factor=0.1)
            self.temperature = temperature
            self.projection_head = nn.Sequential(
                nn.Linear(target_len, target_len), nn.ReLU(),
                nn.Linear(target_len, target_len),
            )
            self.register_buffer("feature_queue", F.normalize(torch.randn(queue_size, target_len), dim=1))
            self.elu = nn.ELU()
            self.dropout = nn.Dropout(dropout)

        def encode(self, x):
            h1 = self.dropout(self.elu(self.enc_conv1(x)))
            h2 = self.dropout(self.elu(self.enc_conv2(h1)))
            h3 = self.elu(self.enc_conv3(h2))
            return self.latent_conv(self.pool(h3)).squeeze(1)

        def nearest_neighbor(self, projections):
            import torch.nn.functional as F
            projections = F.normalize(projections, dim=1)
            _, nn_indices = torch.matmul(projections, self.feature_queue.t()).max(dim=1)
            nn_proj = self.feature_queue[nn_indices]
            return projections + (nn_proj - projections).detach()

        def _update_queue(self, features):
            import torch.nn.functional as F
            batch_size = features.size(0)
            self.feature_queue = torch.cat([F.normalize(features.detach(), dim=1), self.feature_queue[:-batch_size]], dim=0)

        def contrastive_loss(self, z1, z2):
            import torch.nn.functional as F
            batch_size = z1.size(0)
            z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
            nn_z1, nn_z2 = self.nearest_neighbor(z1), self.nearest_neighbor(z2)
            labels = torch.arange(batch_size).to(z1.device)
            loss = (
                F.cross_entropy(torch.matmul(nn_z1, z2.t()) / self.temperature, labels)
                + F.cross_entropy(torch.matmul(z2, nn_z1.t()) / self.temperature, labels)
                + F.cross_entropy(torch.matmul(nn_z2, z1.t()) / self.temperature, labels)
                + F.cross_entropy(torch.matmul(z1, nn_z2.t()) / self.temperature, labels)
            ) / 4
            self._update_queue(z1)
            return loss

        def forward(self, x):
            if not self.training:
                return self.encode(x), 0.0
            h1 = self.encode(x)
            h2 = self.encode(self.augmenter(x, training=True))
            loss = self.contrastive_loss(self.projection_head(h1), self.projection_head(h2))
            return h1, loss

    return CCNN()


class CCNN(BaseReducer):
    """Contrastive CNN (CCNN) with a nearest-neighbour feature queue.

    A three-layer Conv1d encoder is trained contrastively: each series is
    augmented twice (Gaussian noise + dropout), and both views are aligned with
    their nearest neighbour retrieved from a momentum feature queue via
    symmetric cross-entropy loss. The queue accumulates normalised projected
    features across batches, providing a diverse set of negatives without
    requiring large batch sizes.

    Parameters
    ----------
    target_len : int, optional
        Latent sequence length. Mutually exclusive with ``retention_rate``.
    retention_rate : float, optional
        Fraction of timepoints to retain. Mutually exclusive with
        ``target_len``.
    epochs : int, default=50
        Number of training epochs.
    lr : float, default=1e-3
        Learning rate for the Adam optimiser.
    batch_size : int, default=32
        Mini-batch size.
    dropout : float, default=0.1
        Dropout probability in the encoder.
    verbose : bool, default=False
        Whether to display a tqdm progress bar during training.
    random_state : int or None, default=None
        Random seed for reproducibility.


    Examples
    --------
    >>> from tsreduce import CCNN
    >>> X = np.random.randn(50, 200)
    >>> CCNN(target_len=20, epochs=5).fit_transform(X).shape
    (50, 20)
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
        from torch.utils.data import DataLoader, TensorDataset
        from tqdm.auto import tqdm

        N = X.shape[2]
        w = self.n_timepoints_out_

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = _build_model(N, w, self.dropout).to(self.device_)

        X_tensor = torch.from_numpy(X.reshape(-1, N).astype(np.float32)).to(self.device_).unsqueeze(1)
        loader = DataLoader(TensorDataset(X_tensor), batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        self.model_.train()
        with tqdm(range(self.epochs), desc="CCNN", disable=not self.verbose) as pbar:
            for _ in pbar:
                epoch_loss = 0.0
                for (batch,) in loader:
                    optimizer.zero_grad()
                    _, loss = self.model_(batch)
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
            latent, _ = self.model_(tensor)

        return latent.unsqueeze(1).cpu().numpy().reshape(n_samples, n_channels, w)
