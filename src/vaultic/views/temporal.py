"""Phase 5 temporal view: a GRU over the uid's last N transactions (features/sequences.py).

Model: the history steps are read by a GRU cell with an exact mask (a padded step leaves the
hidden state unchanged, so padding can never influence the score); the final state is joined
with the scored transaction's own step features and a linear head gives the fraud logit.
ProductCD codes (shared training-period encoder, D37) go through an embedding with code 0 =
missing as padding. Continuous channels are standardised with statistics of the training
period's real steps.

Training: weighted binary cross-entropy (positive class weight = negatives / positives on the
training rows, "balanced"; its square root, "sqrt"; or a given number), Adam with optional
per-epoch learning-rate decay (`lr_decay`) and gradient-norm clipping (`grad_clip`), early
stopping on the early-stopping rows' PR-AUC (patience in epochs; the best epoch's weights are
kept). `overfit_check` trains on one small batch as a sanity check that the model can learn.

Missing view (rule 11): a transaction whose uid has no history gets NaN, never a score; such
rows are also left out of training and of the validation metric.

Requires PyTorch (CPU build in the separate .venv-torch, research/decisions.md D40).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from vaultic.eval.metrics import pr_auc

CLIP = 5.0  # robust_inputs: standardised channels clipped to +-CLIP (D85)


@dataclass
class TemporalData:
    values: np.ndarray  # (n, N, F) history steps (Sequences.values)
    mask: np.ndarray  # (n, N) bool
    current: np.ndarray  # (n, F) the scored transaction (Sequences.current)
    y: np.ndarray | None = None

    @classmethod
    def from_sequences(cls, seq, y=None) -> TemporalData:
        return cls(seq.values, seq.mask, seq.current, None if y is None else np.asarray(y))

    def subset(self, rows: np.ndarray) -> TemporalData:
        y = None if self.y is None else self.y[rows]
        return TemporalData(self.values[rows], self.mask[rows], self.current[rows], y)


class MaskedGRU(nn.Module):
    def __init__(self, n_features: int, n_products: int, product_channel: int = 2,
                 emb_dim: int = 4, hidden: int = 32):  # fmt: skip
        super().__init__()
        self.product_channel = product_channel
        self.continuous = [i for i in range(n_features) if i != product_channel]
        self.embedding = nn.Embedding(n_products, emb_dim, padding_idx=0)
        step_dim = len(self.continuous) + emb_dim
        self.cell = nn.GRUCell(step_dim, hidden)
        self.head = nn.Linear(hidden + step_dim, 1)
        self.hidden = hidden

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        codes = x[..., self.product_channel].long()
        return torch.cat([x[..., self.continuous], self.embedding(codes)], dim=-1)

    def forward(self, values, mask, current) -> torch.Tensor:
        steps = self._embed(values)
        h = values.new_zeros(values.shape[0], self.hidden)
        for t in range(values.shape[1]):
            m = mask[:, t : t + 1].to(values.dtype)
            h = m * self.cell(steps[:, t], h) + (1 - m) * h  # padded step: state unchanged
        return self.head(torch.cat([h, self._embed(current)], dim=-1)).squeeze(-1)


class GRUTemporalView:
    def __init__(
        self,
        n_products: int,
        product_channel: int = 2,
        hidden: int = 32,
        emb_dim: int = 4,
        learning_rate: float = 1e-3,
        batch_size: int = 512,
        max_epochs: int = 50,
        patience: int = 5,
        pos_weight: float | str = "balanced",
        seed: int = 0,
        lr_decay: float | None = None,
        grad_clip: float | None = None,
        robust_inputs: bool = False,
    ):
        self.n_products = n_products
        self.product_channel = product_channel
        self.hidden = hidden
        self.emb_dim = emb_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.pos_weight = pos_weight
        self.seed = seed
        self.lr_decay = lr_decay
        self.grad_clip = grad_clip
        # D85: heavy-tailed counters (C, D columns) dominate a plain standardisation; with
        # robust_inputs the continuous channels get a signed log1p first and are clipped to
        # +-CLIP standard deviations after it (statistics from the training steps only)
        self.robust_inputs = robust_inputs

    # ---- inputs -------------------------------------------------------------------------
    def _transform(self, x: np.ndarray) -> np.ndarray:
        if not self.robust_inputs:
            return x
        out = np.sign(x) * np.log1p(np.abs(x))
        out[..., self.product_channel] = x[..., self.product_channel]
        return out

    def _fit_scaler(self, data: TemporalData) -> None:
        steps = self._transform(data.values[data.mask])  # real history steps of the training period
        self.mean_ = steps.mean(axis=0)
        self.std_ = steps.std(axis=0) + 1e-6
        self.mean_[self.product_channel], self.std_[self.product_channel] = 0.0, 1.0

    def _tensors(self, data: TemporalData):
        values = (self._transform(data.values) - self.mean_) / self.std_
        if self.robust_inputs:
            values = np.clip(values, -CLIP, CLIP)
        values = np.where(data.mask[..., None], values, 0.0)
        # padded steps get code 0 (the embedding's padding row), whatever they contain
        codes = np.where(data.mask, data.values[..., self.product_channel], 0)
        values[..., self.product_channel] = codes
        top = max(codes.max(initial=0), data.current[:, self.product_channel].max(initial=0))
        if top >= self.n_products:
            raise ValueError(f"product code {top:.0f} >= n_products {self.n_products}: use the "
                             "training-period CategoryEncoder the view was built for")  # fmt: skip
        current = (self._transform(data.current) - self.mean_) / self.std_
        if self.robust_inputs:
            current = np.clip(current, -CLIP, CLIP)
        current[:, self.product_channel] = data.current[:, self.product_channel]
        current[:, 1] = 0.0  # the gap channel is undefined for the scored transaction
        return (
            torch.as_tensor(values, dtype=torch.float32),
            torch.as_tensor(data.mask, dtype=torch.bool),
            torch.as_tensor(current, dtype=torch.float32),
        )

    @staticmethod
    def _with_history(data: TemporalData) -> np.ndarray:
        return np.flatnonzero(data.mask.any(axis=1))

    # ---- training -----------------------------------------------------------------------
    def fit(self, train: TemporalData, val: TemporalData) -> GRUTemporalView:
        if train.y is None or val.y is None:
            raise ValueError("train and val need labels")
        train = train.subset(self._with_history(train))
        val = val.subset(self._with_history(val))
        if len(np.unique(train.y)) < 2:
            raise ValueError("training rows with history need both classes")
        if not (val.y == 1).any():
            raise ValueError("early-stopping rows with history need at least one fraud")
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        self._fit_scaler(train)
        self.model_ = MaskedGRU(train.values.shape[2], self.n_products, self.product_channel,
                                self.emb_dim, self.hidden)  # fmt: skip
        y = torch.as_tensor(train.y, dtype=torch.float32)
        pos = float(y.sum())
        ratio = (len(y) - pos) / pos
        if self.pos_weight == "balanced":
            weight = ratio
        elif self.pos_weight == "sqrt":
            weight = float(np.sqrt(ratio))
        else:
            weight = float(self.pos_weight)
        self.pos_weight_ = float(weight)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        sched = (torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.lr_decay)
                 if self.lr_decay is not None else None)  # fmt: skip
        values, mask, current = self._tensors(train)

        best, best_state, waited = -np.inf, None, 0
        self.history_ = []
        for epoch in range(self.max_epochs):
            self.model_.train()
            order = torch.randperm(len(y), generator=gen)
            for start in range(0, len(y), self.batch_size):
                idx = order[start : start + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.model_(values[idx], mask[idx], current[idx]), y[idx])
                loss.backward()
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model_.parameters(), self.grad_clip)
                opt.step()
            if sched is not None:
                sched.step()
            score = pr_auc(val.y, self._predict(val))
            self.history_.append({"epoch": epoch, "val_pr_auc": float(score)})
            if score > best:
                best, best_state, waited = score, copy.deepcopy(self.model_.state_dict()), 0
            else:
                waited += 1
                if waited >= self.patience:
                    break
        self.model_.load_state_dict(best_state)
        self.best_val_pr_auc_ = float(best)
        self.best_epoch_ = int(np.argmax([h["val_pr_auc"] for h in self.history_]))
        return self

    # ---- scoring ------------------------------------------------------------------------
    def _predict(self, data: TemporalData) -> np.ndarray:
        self.model_.eval()
        out = []
        with torch.no_grad():
            values, mask, current = self._tensors(data)
            for start in range(0, len(values), 4096):
                sl = slice(start, start + 4096)
                out.append(torch.sigmoid(self.model_(values[sl], mask[sl], current[sl])).numpy())
        return np.concatenate(out) if out else np.array([])

    def predict_proba(self, data: TemporalData) -> np.ndarray:
        """Fraud probability per row; NaN where the uid has no history (masked view)."""
        out = np.full(len(data.mask), np.nan)
        rows = self._with_history(data)
        if len(rows):
            out[rows] = self._predict(data.subset(rows))
        return out

    def overfit_check(self, data: TemporalData, n: int = 256, epochs: int = 300) -> dict:
        """Sanity check: train on n rows with history (both classes) and report how well the
        model fits THOSE rows. A model that cannot overfit a small batch has a bug or an input
        problem. Uses this view's settings; does not change the fitted model."""
        rows = self._with_history(data)
        rng = np.random.default_rng(self.seed)
        pos = rows[data.y[rows] == 1]
        neg = rows[data.y[rows] == 0]
        k = min(len(pos), n // 2)
        pick = np.concatenate([rng.choice(pos, k, replace=False),
                               rng.choice(neg, n - k, replace=False)])  # fmt: skip
        small = data.subset(pick)
        probe = copy.copy(self)
        probe.max_epochs, probe.patience, probe.batch_size = epochs, epochs, n
        probe.lr_decay = None  # a decayed learning rate would stop it before it can fit
        probe.fit(small, small)
        p = probe._predict(small)
        return {"rows": int(n), "frauds": int(k), "train_pr_auc": pr_auc(small.y, p),
                "epochs": len(probe.history_)}  # fmt: skip

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.model_.parameters()))
