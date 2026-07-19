"""Temporal Convolutional Network (TCN)"""
import numpy as np

from ..base import BaseReducer


def _build_model(input_dim, target_len, n_filters, kernel_size, n_levels, latent_channels, dropout):
    import torch.nn as nn
    from torch.nn.utils import weight_norm

    class TCNBlock(nn.Module):

        def __init__(self, in_channels, n_filters, kernel_size, dilation, dropout):
            super().__init__()
            self.conv1 = weight_norm(nn.Conv1d(in_channels, n_filters, kernel_size, dilation=dilation))
            self.conv2 = weight_norm(nn.Conv1d(n_filters, n_filters, kernel_size, dilation=dilation))
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()
            self.downsample = nn.Conv1d(in_channels, n_filters, 1) if in_channels != n_filters else None
            self.kernel_size = kernel_size
            self.dilation = dilation

        def _causal_pad(self, x):
            return nn.functional.pad(x, ((self.kernel_size - 1) * self.dilation, 0))

        def forward(self, x):
            out = self.dropout(self.relu(self.conv1(self._causal_pad(x))))
            out = self.dropout(self.relu(self.conv2(self._causal_pad(out))))
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)

    class _TCNStack(nn.Module):

        def __init__(self, in_channels, n_filters, kernel_size, n_levels, dropout):
            super().__init__()
            layers = [
                TCNBlock(in_channels if i == 0 else n_filters, n_filters, kernel_size, 2 ** i, dropout)
                for i in range(n_levels)
            ]
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    class TCNAE(nn.Module):

        def __init__(self):
            super().__init__()
            self.encoder_tcn = _TCNStack(1, n_filters, kernel_size, n_levels, dropout)
            self.to_latent_space = nn.Conv1d(n_filters, latent_channels, 1)
            self.pool = nn.AdaptiveAvgPool1d(target_len)
            self.to_latent_channel = nn.Conv1d(latent_channels, 1, 1)
            self.decoder_upsample = nn.Upsample(size=input_dim, mode="nearest")
            self.decoder_tcn = _TCNStack(latent_channels, n_filters, kernel_size, n_levels, dropout)
            self.to_recon = nn.Conv1d(n_filters, 1, 1)

        def forward(self, x):
            encoded = self.encoder_tcn(x)
            pooled = self.pool(self.to_latent_space(encoded))
            latent = self.to_latent_channel(pooled)
            decoded = self.decoder_tcn(self.decoder_upsample(pooled))
            recon = self.to_recon(decoded)
            return recon, latent

    return TCNAE()


class TCN(BaseReducer):
    """Temporal Convolutional Network autoencoder (TCN).

    A stack of causal dilated residual TCN blocks encodes the series with
    exponentially increasing receptive fields (dilation = 2^i per level).
    Adaptive average pooling compresses the latent to ``n_timepoints_out_``
    steps; a symmetric TCN decoder reconstructs the original for the MSE loss.
    The latent sign is corrected at transform time to stay aligned with a
    uniform subsample of the input, preventing orientation flips across runs.

    Parameters
    ----------
    target_len : int, optional
        Latent sequence length. Mutually exclusive with ``retention_rate``.
    retention_rate : float, optional
        Fraction of timepoints to retain. Mutually exclusive with
        ``target_len``.
    n_filters : int, default=20
        Number of filters in each TCN block.
    kernel_size : int, default=20
        Convolutional kernel size in each TCN block.
    n_levels : int, default=5
        Number of TCN blocks in the encoder (and decoder).
    latent_channels : int, default=8
        Number of channels in the latent feature map.
    dropout : float, default=0.2
        Dropout probability in TCN blocks.
    epochs : int, default=50
        Number of training epochs.
    lr : float, default=1e-3
        Learning rate for the Adam optimiser.
    batch_size : int, default=32
        Mini-batch size.
    verbose : bool, default=False
        Whether to display a tqdm progress bar during training.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    model_ : torch.nn.Module
        Fitted TCN autoencoder.
    device_ : torch.device
        Device used for training (CPU or CUDA).

    Examples
    --------
    >>> import numpy as np
    >>> from tsreduce import TCN
    >>> X = np.random.randn(50, 200)
    >>> TCN(target_len=20, epochs=5).fit_transform(X).shape
    (50, 20)
    """

    def __init__(self, *, target_len=None, retention_rate=None,
                 n_filters=20, kernel_size=20, n_levels=5, latent_channels=8,
                 dropout=0.2, epochs=50, lr=1e-3, batch_size=32,
                 verbose: bool = False, random_state=None):
        super().__init__(target_len=target_len, retention_rate=retention_rate, verbose=verbose)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_levels = n_levels
        self.latent_channels = latent_channels
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state

    def _fit(self, X: np.ndarray, y=None) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from tqdm.auto import tqdm

        N = X.shape[2]
        w = self.n_timepoints_out_
        if w >= N:
            raise ValueError(f"TCN needs a bottleneck smaller than the series: w={w}, N={N}.")

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = _build_model(
            N, w, self.n_filters, self.kernel_size, self.n_levels, self.latent_channels, self.dropout
        ).to(self.device_)

        X_tensor = torch.from_numpy(X.reshape(-1, N).astype(np.float32)).to(self.device_).unsqueeze(1)
        loader = DataLoader(TensorDataset(X_tensor), batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model_.train()
        with tqdm(range(self.epochs), desc="TCN", disable=not self.verbose) as pbar:
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
        X_flat = X.reshape(-1, N).astype(np.float32)

        tensor = torch.from_numpy(X_flat).to(self.device_).unsqueeze(1)
        self.model_.eval()
        with torch.no_grad():
            _, latent = self.model_(tensor)
        reduced = latent.cpu().numpy().reshape(-1, w)

        return self._sign_correct(reduced, X_flat, w).reshape(n_samples, n_channels, w)

    def _sign_correct(self, reduced, originals, w):
        """Flip a reduced series' sign if it anti-correlates with the subsampled original.

        The pooled latent has an arbitrary sign; aligning each row with a uniform
        subsample of its original series keeps the representation stable across runs.
        """
        idx = np.linspace(0, originals.shape[1] - 1, w, dtype=int)
        result = reduced.copy()
        for i, (r, s) in enumerate(zip(reduced, originals)):
            sub = s[idx]
            if np.std(r) > 1e-9 and np.std(sub) > 1e-9:
                try:
                    if np.corrcoef(r, sub)[0, 1] < 0:
                        result[i] = -r
                except ValueError:
                    pass
        return result
