"""Missing-View-Aware Adaptive Fusion (MVAF), roadmap Phase 7.1-7.3.

A small MLP gate looks at the available view scores, which views are available, how confident
and how much in disagreement they are, and the transaction context. It outputs one raw score
g_v(x) per view and a context bias b(x). Missing views get weight exactly 0 through a masked
softmax, and the fused score is a weighted sum in logit space:

    w_v = m_v exp(g_v(x)) / sum_u m_u exp(g_u(x)),    p = sigmoid(sum_v w_v logit(p_v) + b(x))

Training (7.3): the gate is fitted on out-of-sample view predictions (validation period), with
view dropout (each available view hidden with probability 0.2-0.3, never all of them) and a
weighted binary cross-entropy (class weights; optionally times per-row weights such as the
amount, for the cost-weighted variant).

Implemented in NumPy with hand-written gradients (checked against finite differences in
tests/test_fusion.py), so it needs no deep-learning framework.

Inputs everywhere: `views` is an (n, V) array of calibrated view probabilities with NaN where a
view is missing; `context` is an optional (n, C) numeric array.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(s: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(s, -500, 500)))


def view_inputs(views: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mask, logits (0 where missing), confidence |2p-1| (0 where missing), disagreement
    (std of available probabilities; 0 with fewer than two)."""
    views = np.asarray(views, dtype=np.float64)
    mask = ~np.isnan(views)
    p = np.where(mask, views, 0.5)
    logits = np.where(mask, _logit(p), 0.0)
    confidence = np.where(mask, np.abs(2 * p - 1), 0.0)
    count = mask.sum(axis=1)
    mean = np.where(count > 0, np.where(mask, p, 0).sum(axis=1) / np.maximum(count, 1), 0)
    var = np.where(mask, (p - mean[:, None]) ** 2, 0).sum(axis=1) / np.maximum(count, 1)
    disagreement = np.where(count >= 2, np.sqrt(var), 0.0)
    return mask, logits, confidence, disagreement


def masked_softmax(g: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Softmax over the available entries of each row; rows with nothing available are all 0."""
    if mask is None:
        mask = np.ones_like(g, dtype=bool)
    shifted = np.where(mask, g, -np.inf)
    row_max = np.max(shifted, axis=1, keepdims=True)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)
    e = np.exp(np.where(mask, g - row_max, -np.inf))  # masked entries: exp(-inf) = 0, no overflow
    total = e.sum(axis=1, keepdims=True)
    return np.divide(e, total, out=np.zeros_like(e), where=total > 0)


def view_dropout(views: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Hide each available view with probability `rate`, never all of a row's views."""
    views = np.array(views, dtype=np.float64, copy=True)
    mask = ~np.isnan(views)
    drop = mask & (rng.random(views.shape) < rate)
    emptied = mask.any(axis=1) & ~(mask & ~drop).any(axis=1)
    for i in np.flatnonzero(emptied):  # put one originally available view back
        keep = rng.choice(np.flatnonzero(mask[i]))
        drop[i, keep] = False
    views[drop] = np.nan
    return views


class MVAF:
    """The MVAF gate (and, through its options, the gate baselines F2, F5 and F6).

    use_mask=False      F5: no availability mask in the softmax or in the gate's inputs
    dropout=0.0         F6: no view dropout
    gate_inputs="constant"  F2: the gate sees only a constant, so weights are fixed per view
    """

    def __init__(
        self,
        hidden: tuple[int, ...] = (32, 32),
        dropout: float = 0.25,
        use_mask: bool = True,
        gate_inputs: str = "full",
        epochs: int = 30,
        batch_size: int = 1024,
        learning_rate: float = 1e-3,
        l2: float = 1e-4,
        class_weight: str | None = "balanced",
        seed: int = 0,
    ):
        if gate_inputs not in ("full", "constant"):
            raise ValueError("gate_inputs must be 'full' or 'constant'")
        self.hidden = hidden
        self.dropout = dropout
        self.use_mask = use_mask
        self.gate_inputs = gate_inputs
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2 = l2
        self.class_weight = class_weight
        self.seed = seed

    # ---- inputs -------------------------------------------------------------------------
    def _raw_inputs(self, views: np.ndarray, context: np.ndarray | None) -> tuple:
        mask, logits, confidence, disagreement = view_inputs(views)
        if self.gate_inputs == "constant":
            z = np.ones((len(logits), 1))
        else:
            parts = [logits, confidence, disagreement[:, None]]
            if self.use_mask:
                parts.insert(1, mask.astype(float))
            if context is not None:
                parts.append(np.asarray(context, dtype=np.float64))
            z = np.hstack(parts)
        return z, logits, mask

    def _standardize(self, z: np.ndarray) -> np.ndarray:
        return (z - self.z_mean_) / self.z_std_

    # ---- network ------------------------------------------------------------------------
    def _init_params(self, n_in: int, n_views: int, rng: np.random.Generator) -> None:
        sizes = [n_in, *self.hidden, n_views + 1]
        self.params_ = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
            last = i == len(sizes) - 2
            # last layer starts at zero: equal weights over available views, zero bias
            W = np.zeros((a, b)) if last else rng.normal(0, np.sqrt(2.0 / a), (a, b))
            self.params_ += [W, np.zeros(b)]

    def _forward(self, params, z, logits, mask):
        acts = [z]
        h = z
        n_layers = len(params) // 2
        for k in range(n_layers):
            h = h @ params[2 * k] + params[2 * k + 1]
            if k < n_layers - 1:
                h = np.maximum(h, 0)
            acts.append(h)
        out = acts[-1]
        g, bias = out[:, :-1], out[:, -1]
        w = masked_softmax(g, mask if self.use_mask else None)
        lin = (w * logits).sum(axis=1)
        p = _sigmoid(lin + bias)
        return p, w, lin, acts

    def _loss_and_grads(self, params, z, logits, mask, y, c, norm=None, include_l2=True):
        """Weighted BCE (normalised by `norm`, default sum of c) and its gradients."""
        p, w, lin, acts = self._forward(params, z, logits, mask)
        total = c.sum() if norm is None else norm
        loss = -(c * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))).sum() / total
        weights_l2 = sum((W**2).sum() for W in params[0::2])
        l2 = self.l2 if include_l2 else 0.0
        loss += 0.5 * l2 * weights_l2
        ds = c * (p - y) / total  # dL/ds, s = lin + bias
        dg = ds[:, None] * w * (logits - lin[:, None])
        dout = np.hstack([dg, ds[:, None]])
        grads = [None] * len(params)
        n_layers = len(params) // 2
        delta = dout
        for k in reversed(range(n_layers)):
            grads[2 * k] = acts[k].T @ delta + l2 * params[2 * k]
            grads[2 * k + 1] = delta.sum(axis=0)
            if k > 0:
                delta = (delta @ params[2 * k].T) * (acts[k] > 0)
        return loss, grads

    # ---- public API -----------------------------------------------------------------------
    def _row_weights(self, y: np.ndarray, sample_weight: np.ndarray | None) -> np.ndarray:
        c = np.ones(len(y))
        if self.class_weight == "balanced":
            pos = max(y.mean(), 1e-12)
            c = np.where(y == 1, 0.5 / pos, 0.5 / max(1 - pos, 1e-12))
        if sample_weight is not None:
            c = c * np.asarray(sample_weight, dtype=np.float64)
        return c

    def fit(self, views, y, context=None, sample_weight=None) -> MVAF:
        rng = np.random.default_rng(self.seed)
        views = np.asarray(views, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z0, _, _ = self._raw_inputs(views, context)
        self.z_mean_ = z0.mean(axis=0)
        self.z_std_ = z0.std(axis=0)
        self.z_std_[self.z_std_ == 0] = 1.0
        if self.gate_inputs == "constant":
            self.z_mean_[:], self.z_std_[:] = 0.0, 1.0
        self.n_views_ = views.shape[1]
        self._init_params(z0.shape[1], self.n_views_, rng)
        c_all = self._row_weights(y, sample_weight)
        m1 = [np.zeros_like(P) for P in self.params_]
        m2 = [np.zeros_like(P) for P in self.params_]
        beta1, beta2, step = 0.9, 0.999, 0
        context = None if context is None else np.asarray(context, dtype=np.float64)
        for _ in range(self.epochs):
            order = rng.permutation(len(y))
            for start in range(0, len(y), self.batch_size):
                idx = order[start : start + self.batch_size]
                ctx = None if context is None else context[idx]
                grads = self._batch_grads(views[idx], ctx, y[idx], c_all[idx], rng)
                step += 1
                for i, g in enumerate(grads):
                    m1[i] = beta1 * m1[i] + (1 - beta1) * g
                    m2[i] = beta2 * m2[i] + (1 - beta2) * g**2
                    m_hat = m1[i] / (1 - beta1**step)
                    v_hat = m2[i] / (1 - beta2**step)
                    self.params_[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        return self

    def _batch_grads(self, views, context, y, c, rng):
        """Gradients for one mini-batch: view dropout, then the weighted BCE."""
        if self.dropout > 0:
            views = view_dropout(views, self.dropout, rng)
        z, logits, mask = self._raw_inputs(views, context)
        _, grads = self._loss_and_grads(self.params_, self._standardize(z), logits, mask, y, c)
        return grads

    def _predict(self, views, context):
        z, logits, mask = self._raw_inputs(np.asarray(views, dtype=np.float64), context)
        return self._forward(self.params_, self._standardize(z), logits, mask)

    def predict_proba(self, views, context=None) -> np.ndarray:
        """Fused fraud probability per row."""
        return self._predict(views, context)[0]

    def view_weights(self, views, context=None) -> np.ndarray:
        """Gate weight per view (0 for missing views when the mask is used)."""
        return self._predict(views, context)[1]
