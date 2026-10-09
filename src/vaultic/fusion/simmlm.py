"""F7: a SimMLM-style gated fusion baseline with the MoFe ranking loss.

SimMLM (Li, Chen and Han, "SimMLM: A Simple Framework for Multi-modal Learning with Missing
Modality", ICCV 2025, arXiv 2507.19264) mixes modality experts with a learnable gate and adds a
"More vs. Fewer" (MoFe) ranking loss: removing modalities should not make the prediction
better. Here the experts are Vaultic's views; the gate, its inputs, the masked softmax and the
logit-level weighting are MVAF's, so F7 differs from MVAF only in its training objective.

Each mini-batch is scored twice: with every available view ("more") and after modality dropout
("fewer", at least one view kept). With per-row weighted BCE losses l_more and l_fewer:

    loss = BCE(more) + BCE(fewer) + lambda * mean(max(0, l_more - l_fewer))

The MoFe term is zero when the extra views lower the loss and penalises rows where they raise
it. Its gradient is (1 + lambda h) dl_more + (1 - lambda h) dl_fewer with h = 1[l_more > l_fewer],
so it reuses the gate's backward pass with adjusted row weights.

The formula follows the MoFe idea as described in the paper's abstract and Aziz's spec
(max(0, loss_more - loss_fewer)); the full paper's exact formulation was not checked, hence
"SimMLM-style" (research/decisions.md D33).
"""

from __future__ import annotations

import numpy as np

from vaultic.fusion.mvaf import MVAF, view_dropout


def row_bce(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))


def mofe_term(loss_more: np.ndarray, loss_fewer: np.ndarray) -> float:
    """Mean ranking hinge max(0, loss_more - loss_fewer): zero when more views help."""
    return float(np.mean(np.maximum(0.0, np.asarray(loss_more) - np.asarray(loss_fewer))))


class SimMLMGate(MVAF):
    def __init__(self, mofe_weight: float = 1.0, dropout: float = 0.25, **kwargs):
        if dropout <= 0:
            raise ValueError("F7 needs modality dropout > 0 to form the 'fewer' views")
        super().__init__(dropout=dropout, **kwargs)
        self.mofe_weight = mofe_weight

    def _mofe_loss_and_grads(self, params, more, fewer, y, c):
        """Total F7 loss and gradients for fixed 'more' and 'fewer' inputs (z, logits, mask)."""
        p_more = self._forward(params, *more)[0]
        p_fewer = self._forward(params, *fewer)[0]
        l_more, l_fewer = row_bce(p_more, y), row_bce(p_fewer, y)
        h = (l_more > l_fewer).astype(float)
        norm = c.sum()
        loss_m, g_m = self._loss_and_grads(params, *more, y, c * (1 + self.mofe_weight * h), norm)
        loss_f, g_f = self._loss_and_grads(
            params, *fewer, y, c * (1 - self.mofe_weight * h), norm, include_l2=False
        )
        return loss_m + loss_f, [a + b for a, b in zip(g_m, g_f, strict=True)]

    def _inputs(self, views, context):
        z, logits, mask = self._raw_inputs(views, context)
        return self._standardize(z), logits, mask

    def _batch_grads(self, views, context, y, c, rng):
        more = self._inputs(views, context)
        fewer = self._inputs(view_dropout(views, self.dropout, rng), context)
        _, grads = self._mofe_loss_and_grads(self.params_, more, fewer, y, c)
        return grads
