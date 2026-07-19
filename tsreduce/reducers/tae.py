"""Transformer Autoencoder (TAE)"""
import numpy as np

from ..base import BaseReducer


def _build_model(input_dim, target_len, dropout, d_model=32, n_heads=4, num_layers=2):
    import math
    import torch
    import torch.nn as nn

    class TransformerAE(nn.Module):

        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(1, d_model)
            pe = self._create_pe(input_dim, d_model)
            self.register_buffer("pe_enc", pe)
            self.register_buffer("pe_dec", pe)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
                dropout=dropout, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.pool = nn.AdaptiveAvgPool1d(target_len)
            self.latent_compress = nn.Linear(d_model, 1)
            self.latent_expand = nn.Linear(1, d_model)
            self.upsample = nn.Upsample(size=input_dim, mode="linear", align_corners=False)
            dec_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
                dropout=dropout, batch_first=True,
            )
            self.decoder = nn.TransformerEncoder(dec_layer, num_layers=num_layers)
            self.output_proj = nn.Linear(d_model, 1)

        def _create_pe(self, max_len, dim):
            pe = torch.zeros(max_len, dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return pe.unsqueeze(0)

        def forward(self, x):
            x = x.transpose(1, 2)
            x_emb = self.input_proj(x) + self.pe_enc
            enc_out = self.encoder(x_emb)
            pooled = self.pool(enc_out.transpose(1, 2)).transpose(1, 2)
            latent_tf = self.latent_compress(pooled)
            latent = latent_tf.transpose(1, 2)
            dec_in = self.latent_expand(latent_tf)
            dec_in = self.upsample(dec_in.transpose(1, 2)).transpose(1, 2)
            dec_out = self.decoder(dec_in + self.pe_dec)
            recon = self.output_proj(dec_out).transpose(1, 2)
            return recon, latent

    return TransformerAE()


class TAE(BaseReducer):
    """Transformer autoencoder with sinusoidal positional encoding.

    A Transformer encoder with adaptive average pooling compresses the sequence
    to *n_timepoints_out_* steps; a symmetric Transformer decoder reconstructs
    the original length for the reconstruction loss.
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

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = _build_model(N, w, self.dropout).to(self.device_)

        X_tensor = torch.from_numpy(X.reshape(-1, N).astype(np.float32)).to(self.device_).unsqueeze(1)
        loader = DataLoader(TensorDataset(X_tensor), batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model_.train()
        with tqdm(range(self.epochs), desc="TAE", disable=not self.verbose) as pbar:
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
