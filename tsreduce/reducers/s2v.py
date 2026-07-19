"""Series2Vec (S2V)"""
import numpy as np

from ..base import BaseReducer


def _fft_magnitudes(series_batch):
    """FFT magnitude spectrum of each series in a (M, N) batch -> (M, N) real array."""
    return np.abs(np.fft.fft(series_batch, axis=1)).astype(np.float32)


def _build_model(latent_dim, emb_size, d_model, n_heads):
    import torch.nn as nn

    class S2VEncoder(nn.Module):

        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(1, emb_size, kernel_size=8, padding="same"),
                nn.BatchNorm1d(emb_size), nn.GELU(),
                nn.Conv1d(emb_size, emb_size, kernel_size=5, padding="same"),
                nn.BatchNorm1d(emb_size), nn.GELU(),
            )
            self.pool = nn.AdaptiveAvgPool1d(latent_dim)
            self.collapse = nn.Conv1d(emb_size, 1, kernel_size=1)
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.xavier_uniform_(m.weight)

        def forward(self, x):
            return self.collapse(self.pool(self.features(x))).squeeze(1)

    class S2VModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.time_enc = S2VEncoder()
            self.freq_enc = S2VEncoder()
            self.proj_t = nn.Linear(latent_dim, d_model)
            self.attn_t = nn.MultiheadAttention(d_model, n_heads, batch_first=False)
            self.norm1_t = nn.LayerNorm(d_model)
            self.ff_t = nn.Linear(d_model, d_model)
            self.norm2_t = nn.LayerNorm(d_model)
            self.proj_f = nn.Linear(latent_dim, d_model)
            self.attn_f = nn.MultiheadAttention(d_model, n_heads, batch_first=False)
            self.norm1_f = nn.LayerNorm(d_model)
            self.ff_f = nn.Linear(d_model, d_model)
            self.norm2_f = nn.LayerNorm(d_model)

        def _branch(self, enc, proj, attn, norm1, ff, norm2, x):
            import torch
            z = enc(x)
            z_proj = proj(z).unsqueeze(0)
            attn_out, _ = attn(z_proj, z_proj, z_proj)
            z = norm1(z_proj + attn_out)
            return norm2(z + ff(z)).squeeze(0)

        def forward(self, x_t, x_f):
            import torch
            z_t = self._branch(self.time_enc, self.proj_t, self.attn_t, self.norm1_t, self.ff_t, self.norm2_t, x_t)
            z_f = self._branch(self.freq_enc, self.proj_f, self.attn_f, self.norm1_f, self.ff_f, self.norm2_f, x_f)
            return torch.cdist(z_t, z_t), torch.cdist(z_f, z_f)

    return S2VModel()


def _minmax(d):
    """Min-max normalise a distance matrix to [0, 1]; zeros if degenerate."""
    import torch
    lo, hi = d.min(), d.max()
    if (hi - lo).abs() < 1e-8:
        return torch.zeros_like(d)
    return (d - lo) / (hi - lo)


class S2V(BaseReducer):

    D_MODEL = 64
    N_HEADS = 8

    def __init__(self, *, target_len=None, retention_rate=None,
                 emb_size=16, epochs=100, lr=1e-3, batch_size=64,
                 verbose: bool = False, random_state=None):
        super().__init__(target_len=target_len, retention_rate=retention_rate, verbose=verbose)
        self.emb_size = emb_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state

    def _fit(self, X: np.ndarray, y=None) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from tqdm import tqdm

        N = X.shape[2]
        w = self.n_timepoints_out_
        if w >= N:
            raise ValueError(f"S2V needs a bottleneck smaller than the series: w={w}, N={N}.")

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = _build_model(w, self.emb_size, self.D_MODEL, self.N_HEADS).to(self.device_)

        X_flat = X.reshape(-1, N).astype(np.float32)
        X_t = torch.from_numpy(X_flat).unsqueeze(1).to(self.device_)
        X_f = torch.from_numpy(_fft_magnitudes(X_flat)).unsqueeze(1).to(self.device_)

        batch_size = min(self.batch_size, X_t.shape[0])
        loader = DataLoader(
            TensorDataset(X_t, X_f), batch_size=batch_size, shuffle=True,
            drop_last=(X_t.shape[0] > batch_size),
        )
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.SmoothL1Loss()

        self.model_.train()
        with tqdm(range(self.epochs), desc="S2V", disable=not self.verbose) as pbar:
            for _ in pbar:
                epoch_loss = 0.0
                for x_t_batch, x_f_batch in loader:
                    B = x_t_batch.size(0)
                    if B < 2:
                        continue
                    optimizer.zero_grad()
                    d_rep_t, d_rep_f = self.model_(x_t_batch, x_f_batch)
                    d_in_t = torch.cdist(x_t_batch.view(B, -1), x_t_batch.view(B, -1))
                    d_in_f = torch.cdist(x_f_batch.view(B, -1), x_f_batch.view(B, -1))
                    tril = torch.tril_indices(B, B, offset=-1)
                    loss = (
                        criterion(_minmax(d_rep_t[tril[0], tril[1]]), _minmax(d_in_t[tril[0], tril[1]]))
                        + criterion(_minmax(d_rep_f[tril[0], tril[1]]), _minmax(d_in_f[tril[0], tril[1]]))
                    )
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=4.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                if epoch_loss > 0:
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
            reduced = self.model_.time_enc(tensor).cpu().numpy()

        return reduced.reshape(n_samples, n_channels, w)
