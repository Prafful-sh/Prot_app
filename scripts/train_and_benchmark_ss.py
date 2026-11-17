#!/usr/bin/env python3
"""
Strict Zero-Leakage Training & >=10,500-Protein Multi-Domain Benchmark for ProPurify SS
=======================================================================================
Trains the Cascaded Pure-NumPy Secondary Structure Engine (`modules/secondary_structure.py`)
STRICTLY on `splits == "train"` from `data/ss_10k_corpus.npz` (zero contact with `splits == "test"`
during GOR-IV estimation, feature scaling, neural training, ensemble weighting, or Viterbi calibration),
exports float32 weights with folded BatchNorm to `data/ss_model_weights.npz`, and evaluates:
  1. ProPurify Cascaded SS (3-Model Diverse Stage-1 Ensemble + Stage-2 Context Net + Constrained Viterbi)
  2. Classic GOR IV (Garnier-Gibrat-Robson 1996 17-residue pairwise information baseline)
  3. Classic Chou-Fasman (1978 empirical nucleation/propagation baseline)
across all >=11,000 held-out `test` proteins (>4.0M residues) spanning all domains & kingdoms of life.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modules.secondary_structure import (
    PAD_IDX,
    STATE_TO_IDX,
    _GOR_M_IDX,
    _GOR_N_IDX,
    _ONEHOT_21,
    build_stage2_inputs,
    classify_fold_class,
    compute_q3,
    compute_sov99,
    decode_viterbi_min_length,
    encode_sequence_indices,
    extract_residue_and_global_features,
    load_ss_model_weights,
    predict_chou_fasman,
    predict_gor4,
    predict_secondary_structure,
    _forward_stage1_ensemble,
    _forward_stage2,
)
from modules.sequence_parser import CLASSROOM_PRESET_PROTEINS


# ==============================================================================
# 1. GOR-IV PARAMETER ESTIMATION (WITH OUT-OF-FOLD SUPPORT INSIDE TRAIN)
# ==============================================================================

def accumulate_gor4_counts(
    sequences: List[str],
    ss3_strings: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Accumulates exact 17-residue window single (17, 21, 3) and 136-pair (136, 441, 3)
    state-conditional occurrence counts over a list of training sequences.
    """
    state_totals = np.zeros(3, dtype=np.float64)
    single_counts = np.zeros((17, 21, 3), dtype=np.float64)
    pair_counts = np.zeros((136, 441, 3), dtype=np.float64)

    pos_idx = np.arange(17, dtype=np.int32)

    for seq, ss in zip(sequences, ss3_strings):
        L = len(seq)
        s_idx = encode_sequence_indices(seq)
        s_idx = np.where(s_idx > 20, PAD_IDX, s_idx)
        padded = np.empty(L + 16, dtype=np.int32)
        padded[:8] = PAD_IDX
        padded[8:8 + L] = s_idx
        padded[8 + L:] = PAD_IDX

        w_seq = sliding_window_view(padded, 17)  # (L, 17)
        pair_ab = w_seq[:, _GOR_M_IDX] * 21 + w_seq[:, _GOR_N_IDX]  # (L, 136)

        y_arr = np.fromiter((STATE_TO_IDX[ch] for ch in ss), dtype=np.int32, count=L)
        for st in range(3):
            mask = (y_arr == st)
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            state_totals[st] += cnt
            w_sub = w_seq[mask]       # (cnt, 17)
            p_sub = pair_ab[mask]     # (cnt, 136)
            for m in pos_idx:
                single_counts[m, :, st] += np.bincount(w_sub[:, m], minlength=21)
            for p_i in range(136):
                pair_counts[p_i, :, st] += np.bincount(p_sub[:, p_i], minlength=441)

    return state_totals, single_counts, pair_counts


def counts_to_gor4_tables(
    state_totals: np.ndarray,
    single_counts: np.ndarray,
    pair_counts: np.ndarray,
    alpha_single: float = 8.0,
    alpha_pair: float = 28.0,
) -> Dict[str, np.ndarray]:
    """
    Converts raw GOR-IV count tensors into empirical-Bayes smoothed log-odds tables:
      - dI_single(S; a, m) = ln( P(a_m | S) / P(a_m | ~S) )
      - dI_pair(S; a, b, m, n) = ln( P(a_m, b_n | S) / P(a_m, b_n | ~S) )
        with Dirichlet shrinkage toward independent marginal product P(a_m | S)*P(b_n | S).
    """
    total_res = float(state_totals.sum())
    p_state = state_totals / max(1.0, total_res)
    prior_log_odds = np.log(np.clip(p_state, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - p_state, 1e-6, 1.0)).astype(np.float32)

    # Marginal AA background across central position m=8
    bg_aa = single_counts[8].sum(axis=1)
    bg_aa = np.maximum(bg_aa, 1.0)
    bg_aa = bg_aa / bg_aa.sum()  # (21,)

    single_table = np.zeros((17, 21, 3), dtype=np.float32)
    p_single_given_s = np.zeros((17, 21, 3), dtype=np.float64)
    p_single_given_not_s = np.zeros((17, 21, 3), dtype=np.float64)

    all_single = single_counts.sum(axis=2)  # (17, 21)
    for st in range(3):
        n_s = state_totals[st]
        n_ns = total_res - n_s
        c_s = single_counts[:, :, st]
        c_ns = all_single - c_s

        p_s = (c_s + alpha_single * bg_aa[None, :]) / (n_s + alpha_single)
        p_ns = (c_ns + alpha_single * bg_aa[None, :]) / (n_ns + alpha_single)
        p_single_given_s[:, :, st] = p_s
        p_single_given_not_s[:, :, st] = p_ns
        single_table[:, :, st] = np.log(p_s / p_ns).astype(np.float32)

    # Pairwise tables with shrinkage toward independent single-site product P(a_m)*P(b_n)
    pair_flat_table = np.zeros((136, 441, 3), dtype=np.float32)
    all_pair = pair_counts.sum(axis=2)  # (136, 441)

    for p_i in range(136):
        m = int(_GOR_M_IDX[p_i])
        n = int(_GOR_N_IDX[p_i])
        for st in range(3):
            n_s = state_totals[st]
            n_ns = total_res - n_s
            prior_s = np.outer(p_single_given_s[m, :, st], p_single_given_s[n, :, st]).reshape(441)
            prior_ns = np.outer(p_single_given_not_s[m, :, st], p_single_given_not_s[n, :, st]).reshape(441)

            c_s = pair_counts[p_i, :, st]
            c_ns = all_pair[p_i] - c_s

            p_ab_s = (c_s + alpha_pair * prior_s) / (n_s + alpha_pair)
            p_ab_ns = (c_ns + alpha_pair * prior_ns) / (n_ns + alpha_pair)
            pair_flat_table[p_i, :, st] = np.log(p_ab_s / p_ab_ns).astype(np.float32)

    return {
        "gor_single_table": single_table,
        "gor_pair_flat_table": pair_flat_table,
        "gor_pair_2d": np.ascontiguousarray(pair_flat_table.reshape(136 * 441, 3)),
        "gor_prior_log_odds": prior_log_odds,
        "gor_temperature": np.array(1.0, dtype=np.float32),
    }


# ==============================================================================
# 2. PURE-NUMPY DIFFERENTIABLE PRIMITIVES (ADAMW + BATCHNORM + ACTIVATIONS)
# ==============================================================================

class BatchNorm1D:
    """Differentiable BatchNorm1D with exact folding into preceding Linear(W, b) at export."""

    def __init__(self, dim: int, momentum: float = 0.05, eps: float = 1e-5):
        self.dim = dim
        self.momentum = np.float32(momentum)
        self.eps = np.float32(eps)
        self.gamma = np.ones(dim, dtype=np.float32)
        self.beta = np.zeros(dim, dtype=np.float32)
        self.running_mean = np.zeros(dim, dtype=np.float32)
        self.running_var = np.ones(dim, dtype=np.float32)
        self._cache: Tuple[np.ndarray, np.ndarray] | None = None

    def forward(self, u: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            mu = u.mean(axis=0)
            diff = u - mu
            var = (diff * diff).mean(axis=0)
            inv_std = np.float32(1.0) / np.sqrt(var + self.eps)
            u_hat = diff * inv_std
            self.running_mean = (np.float32(1.0) - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var = (np.float32(1.0) - self.momentum) * self.running_var + self.momentum * var
            self._cache = (u_hat, inv_std)
            return u_hat * self.gamma + self.beta
        inv_std = np.float32(1.0) / np.sqrt(self.running_var + self.eps)
        return (u - self.running_mean) * (self.gamma * inv_std) + self.beta

    def backward(self, dz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._cache is not None
        u_hat, inv_std = self._cache
        n = np.float32(dz.shape[0])
        d_gamma = (dz * u_hat).sum(axis=0)
        d_beta = dz.sum(axis=0)
        du_hat = dz * self.gamma
        du = (inv_std / n) * (n * du_hat - du_hat.sum(axis=0) - u_hat * (du_hat * u_hat).sum(axis=0))
        return du, d_gamma, d_beta

    def fold_linear(self, W: np.ndarray, b: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        scale = self.gamma / np.sqrt(self.running_var + self.eps)
        W_f = (W * scale[None, :]).astype(np.float32)
        if b is None:
            b_f = (-self.running_mean * scale + self.beta).astype(np.float32)
        else:
            b_f = ((b - self.running_mean) * scale + self.beta).astype(np.float32)
        return W_f, b_f


class AdamW:
    """Vectorized pure-NumPy AdamW optimizer supporting decoupled weight decay."""

    def __init__(
        self,
        params: Dict[str, np.ndarray],
        lr: float = 2e-3,
        weight_decay: float = 2e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads: Dict[str, np.ndarray], lr_override: float | None = None) -> None:
        self.t += 1
        lr = self.lr if lr_override is None else lr_override
        b1_corr = 1.0 - (self.beta1 ** self.t)
        b2_corr = 1.0 - (self.beta2 ** self.t)
        step_size = np.float32(lr / b1_corr)
        inv_b2 = np.float32(1.0 / math.sqrt(b2_corr))

        for k, p in self.params.items():
            g = grads[k]
            # Apply decoupled weight decay to 2D weight matrices (not biases or BN gamma/beta)
            if p.ndim >= 2 and self.weight_decay > 0.0:
                p *= np.float32(1.0 - lr * self.weight_decay)
            m_k = self.m[k]
            v_k = self.v[k]
            m_k *= np.float32(self.beta1)
            m_k += np.float32(1.0 - self.beta1) * g
            v_k *= np.float32(self.beta2)
            v_k += np.float32(1.0 - self.beta2) * (g * g)
            p -= step_size * m_k / (np.sqrt(v_k) * inv_b2 + np.float32(self.eps))


def _hs_fwd(x: np.ndarray) -> np.ndarray:
    return x * np.clip(x * np.float32(1.0 / 6.0) + np.float32(0.5), np.float32(0.0), np.float32(1.0))


def _hs_bwd(x: np.ndarray, dy: np.ndarray) -> np.ndarray:
    deriv = np.where(x < -3.0, 0.0, np.where(x > 3.0, 1.0, x * (1.0 / 3.0) + 0.5)).astype(np.float32)
    return dy * deriv


def _lrelu_fwd(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, x * np.float32(0.1))


def _lrelu_bwd(x: np.ndarray, dy: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, dy, dy * np.float32(0.1))


def _softmax_ce_grad(
    logits: np.ndarray,
    targets: np.ndarray,
    label_smoothing: float = 0.02,
    class_weights: np.ndarray | None = None,
    sample_weights: np.ndarray | None = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Computes stabilized softmax cross-entropy loss and gradient wrt logits."""
    n = logits.shape[0]
    z = logits - logits.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    probs = exp_z / exp_z.sum(axis=1, keepdims=True)

    y_dist = np.full((n, 3), label_smoothing / 3.0, dtype=np.float32)
    y_dist[np.arange(n), targets] += np.float32(1.0 - label_smoothing)

    d_logits = probs - y_dist
    log_p = np.log(np.clip(probs, 1e-8, 1.0))
    per_row_loss = -(y_dist * log_p).sum(axis=1)

    if class_weights is not None:
        cw = class_weights[targets]
        per_row_loss = per_row_loss * cw
        d_logits = d_logits * cw[:, None]

    if sample_weights is not None:
        sw = sample_weights / max(1e-6, float(sample_weights.mean()))
        per_row_loss = per_row_loss * sw
        d_logits = d_logits * sw[:, None]

    loss = float(per_row_loss.mean())
    d_logits *= np.float32(1.0 / n)
    return loss, probs, d_logits


def _dropout(x: np.ndarray, rate: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if rate <= 0.0:
        return x, np.ones(1, dtype=np.float32)
    keep = np.float32(1.0 - rate)
    mask = (rng.random(x.shape, dtype=np.float32) < keep).astype(np.float32) * (np.float32(1.0) / keep)
    return x * mask, mask


# ==============================================================================
# 3. PACKED CONTIGUOUS DATASET BUILDER FOR FAST WINDOW SLICING
# ==============================================================================

def pack_protein_features(
    sequences: List[str],
    ss3_strings: List[str],
    gor_tables_per_protein: List[Dict[str, np.ndarray]] | Dict[str, np.ndarray],
    loc_mean: np.ndarray | None = None,
    loc_inv_std: np.ndarray | None = None,
    glob_mean: np.ndarray | None = None,
    glob_inv_std: np.ndarray | None = None,
) -> Dict[str, Any]:
    """
    Extracts features for a list of proteins and packs them into a single padded
    2D buffer separated by 8 boundary rows so 17-mer windows and dilated 1D convolutions
    can slice across thousands of residues in one contiguous NumPy operation.
    """
    n_prots = len(sequences)
    loc_list: List[np.ndarray] = []
    glob_list: List[np.ndarray] = []
    gor_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    lengths: List[int] = []

    for idx, (seq, ss) in enumerate(zip(sequences, ss3_strings)):
        gtab = gor_tables_per_protein[idx] if isinstance(gor_tables_per_protein, list) else gor_tables_per_protein
        xl, xg, pg, _ = extract_residue_and_global_features(seq, gtab)
        loc_list.append(xl)
        glob_list.append(xg)
        gor_list.append(pg)
        y_list.append(np.fromiter((STATE_TO_IDX[ch] for ch in ss), dtype=np.int32, count=len(ss)))
        lengths.append(len(ss))

    if loc_mean is None:
        all_loc = np.vstack(loc_list)
        all_glob = np.vstack(glob_list)
        loc_mean = all_loc.mean(axis=0).astype(np.float32)
        loc_std = np.sqrt(((all_loc - loc_mean) ** 2).mean(axis=0)).astype(np.float32)
        # Keep binary indicator scales stable while normalizing continuous scales
        loc_inv_std = (1.0 / np.maximum(loc_std, 0.05)).astype(np.float32)

        glob_mean = all_glob.mean(axis=0).astype(np.float32)
        glob_std = np.sqrt(((all_glob - glob_mean) ** 2).mean(axis=0)).astype(np.float32)
        glob_inv_std = (1.0 / np.maximum(glob_std, 0.02)).astype(np.float32)

    assert loc_inv_std is not None and glob_mean is not None and glob_inv_std is not None

    raw_pad = np.zeros(81, dtype=np.float32)
    raw_pad[20] = 1.0  # Boundary indicator channel in _ONEHOT_21
    loc_pad_norm = ((raw_pad - loc_mean) * loc_inv_std).astype(np.float32)

    # Pack into buffer with 8 pad rows before each protein and 8 pad rows at the very end
    total_res = sum(lengths)
    buf_rows = total_res + 8 * (n_prots + 1)
    loc_buf = np.broadcast_to(loc_pad_norm, (buf_rows, 81)).copy()
    glob_packed = np.empty((total_res, 31), dtype=np.float32)
    gor_packed = np.empty((total_res, 3), dtype=np.float32)
    raw_loc_packed = np.empty((total_res, 81), dtype=np.float32)
    y_packed = np.empty(total_res, dtype=np.int32)
    buf_centers = np.empty(total_res, dtype=np.int32)
    prot_slices: List[Tuple[int, int, int]] = []  # (res_start, res_end, buf_start)

    cursor_buf = 8
    cursor_res = 0
    for xl, xg, pg, yp in zip(loc_list, glob_list, gor_list, y_list):
        L = len(yp)
        loc_buf[cursor_buf:cursor_buf + L] = (xl - loc_mean) * loc_inv_std
        glob_packed[cursor_res:cursor_res + L] = (xg - glob_mean) * glob_inv_std
        gor_packed[cursor_res:cursor_res + L] = pg
        raw_loc_packed[cursor_res:cursor_res + L] = xl
        y_packed[cursor_res:cursor_res + L] = yp
        buf_centers[cursor_res:cursor_res + L] = np.arange(cursor_buf, cursor_buf + L, dtype=np.int32)
        prot_slices.append((cursor_res, cursor_res + L, cursor_buf))
        cursor_buf += L + 8
        cursor_res += L

    return {
        "loc_buf": loc_buf,
        "glob_packed": glob_packed,
        "gor_packed": gor_packed,
        "raw_loc_packed": raw_loc_packed,
        "raw_glob_list": glob_list,
        "raw_loc_list": loc_list,
        "raw_gor_list": gor_list,
        "y_packed": y_packed,
        "buf_centers": buf_centers,
        "prot_slices": prot_slices,
        "lengths": lengths,
        "loc_mean": loc_mean,
        "loc_inv_std": loc_inv_std,
        "glob_mean": glob_mean,
        "glob_inv_std": glob_inv_std,
        "loc_pad_norm": loc_pad_norm,
    }


def _slice_w17_batch(
    loc_buf: np.ndarray,
    glob_packed: np.ndarray,
    buf_centers: np.ndarray,
    row_idx: np.ndarray,
) -> np.ndarray:
    """Extracts (B, 1408) window-17 + global features for an arbitrary batch of residue indices."""
    centers = buf_centers[row_idx]  # (B,)
    offsets = np.arange(-8, 9, dtype=np.int32)[None, :]  # (1, 17)
    w_loc = loc_buf[centers[:, None] + offsets].reshape(len(row_idx), 17 * 81)
    return np.hstack([w_loc, glob_packed[row_idx]])


# ==============================================================================
# 4. STAGE-1 TRAINERS (MODELS 1, 2, AND 3) & STAGE-2 TRAINER
# ==============================================================================

def train_stage1_model1_wide_mlp(
    tr_pack: Dict[str, Any],
    val_pack: Dict[str, Any],
    seed: int = 101,
    epochs: int = 16,
    batch_size: int = 8192,
) -> Dict[str, np.ndarray]:
    """Trains Model 1: Window-17 Wide HardSwish MLP (1408 -> 256 -> 128 -> 3) with BatchNorm & AdamW."""
    rng = np.random.default_rng(seed)
    d_in, h1, h2 = 1408, 256, 128

    bn1 = BatchNorm1D(h1)
    bn2 = BatchNorm1D(h2)
    params = {
        "W1": (rng.standard_normal((d_in, h1)) * math.sqrt(2.0 / d_in)).astype(np.float32),
        "W2": (rng.standard_normal((h1, h2)) * math.sqrt(2.0 / h1)).astype(np.float32),
        "W3": (rng.standard_normal((h2, 3)) * math.sqrt(1.0 / h2)).astype(np.float32),
        "b3": np.zeros(3, dtype=np.float32),
        "bn1_g": bn1.gamma, "bn1_b": bn1.beta,
        "bn2_g": bn2.gamma, "bn2_b": bn2.beta,
    }
    opt = AdamW(params, lr=2.4e-3, weight_decay=3e-4)

    n_tr = len(tr_pack["y_packed"])
    n_val = len(val_pack["y_packed"])
    best_val_acc = -1.0
    best_snapshot: Dict[str, np.ndarray] = {}

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        # Cosine LR schedule
        lr_ep = 1.2e-4 + 0.5 * (2.4e-3 - 1.2e-4) * (1.0 + math.cos(math.pi * (ep - 1) / epochs))
        perm = rng.permutation(n_tr)
        ep_loss = 0.0
        n_batches = 0

        for start in range(0, n_tr, batch_size):
            b_idx = perm[start:start + batch_size]
            xb = _slice_w17_batch(tr_pack["loc_buf"], tr_pack["glob_packed"], tr_pack["buf_centers"], b_idx)
            yb = tr_pack["y_packed"][b_idx]

            u1 = xb @ params["W1"]
            z1 = bn1.forward(u1, training=True)
            a1 = _hs_fwd(z1)
            a1_d, m1 = _dropout(a1, 0.22, rng)

            u2 = a1_d @ params["W2"]
            z2 = bn2.forward(u2, training=True)
            a2 = _hs_fwd(z2)
            a2_d, m2 = _dropout(a2, 0.12, rng)

            logits = a2_d @ params["W3"] + params["b3"]
            loss, _, d_logits = _softmax_ce_grad(logits, yb, label_smoothing=0.025)
            ep_loss += loss
            n_batches += 1

            gW3 = a2_d.T @ d_logits
            gb3 = d_logits.sum(axis=0)
            da2 = (d_logits @ params["W3"].T) * m2
            dz2 = _hs_bwd(z2, da2)
            du2, g_bn2_g, g_bn2_b = bn2.backward(dz2)
            gW2 = a1_d.T @ du2

            da1 = (du2 @ params["W2"].T) * m1
            dz1 = _hs_bwd(z1, da1)
            du1, g_bn1_g, g_bn1_b = bn1.backward(dz1)
            gW1 = xb.T @ du1

            opt.step(
                {
                    "W1": gW1, "W2": gW2, "W3": gW3, "b3": gb3,
                    "bn1_g": g_bn1_g, "bn1_b": g_bn1_b,
                    "bn2_g": g_bn2_g, "bn2_b": g_bn2_b,
                },
                lr_override=lr_ep,
            )

        # Evaluate on unseen s2_train validation pack
        correct = 0
        for start in range(0, n_val, 16384):
            v_idx = np.arange(start, min(n_val, start + 16384), dtype=np.int32)
            xv = _slice_w17_batch(val_pack["loc_buf"], val_pack["glob_packed"], val_pack["buf_centers"], v_idx)
            a1 = _hs_fwd(bn1.forward(xv @ params["W1"], training=False))
            a2 = _hs_fwd(bn2.forward(a1 @ params["W2"], training=False))
            lv = a2 @ params["W3"] + params["b3"]
            correct += int((lv.argmax(axis=1) == val_pack["y_packed"][v_idx]).sum())

        val_acc = 100.0 * correct / n_val
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            W1_f, b1_f = bn1.fold_linear(params["W1"])
            W2_f, b2_f = bn2.fold_linear(params["W2"])
            best_snapshot = {
                "m1_W1": W1_f.copy(), "m1_b1": b1_f.copy(),
                "m1_W2": W2_f.copy(), "m1_b2": b2_f.copy(),
                "m1_W3": params["W3"].copy(), "m1_b3": params["b3"].copy(),
            }
        if ep in (1, 4, 8, 12, epochs):
            print(f"  [Stage-1 M1 WideMLP] Ep {ep:2d}/{epochs} | loss={ep_loss/n_batches:.4f} | val_Q3={val_acc:.2f}% (best {best_val_acc:.2f}%) [{time.perf_counter()-t0:.1f}s]")

    return best_snapshot


def train_stage1_model2_res_highway(
    tr_pack: Dict[str, Any],
    val_pack: Dict[str, Any],
    seed: int = 202,
    epochs: int = 16,
    batch_size: int = 8192,
) -> Dict[str, np.ndarray]:
    """Trains Model 2: Window-17 Residual Highway LeakyReLU MLP (1408 -> 192 -> [Res 192] -> 96 -> 3)."""
    rng = np.random.default_rng(seed)
    d_in, h1, h2 = 1408, 192, 96

    bn1 = BatchNorm1D(h1)
    bn_res = BatchNorm1D(h1)
    bn2 = BatchNorm1D(h2)
    params = {
        "W1": (rng.standard_normal((d_in, h1)) * math.sqrt(2.0 / d_in)).astype(np.float32),
        "W_res": (rng.standard_normal((h1, h1)) * math.sqrt(1.0 / h1)).astype(np.float32),
        "W2": (rng.standard_normal((h1, h2)) * math.sqrt(2.0 / h1)).astype(np.float32),
        "W3": (rng.standard_normal((h2, 3)) * math.sqrt(1.0 / h2)).astype(np.float32),
        "b3": np.zeros(3, dtype=np.float32),
        "bn1_g": bn1.gamma, "bn1_b": bn1.beta,
        "bnr_g": bn_res.gamma, "bnr_b": bn_res.beta,
        "bn2_g": bn2.gamma, "bn2_b": bn2.beta,
    }
    opt = AdamW(params, lr=2.4e-3, weight_decay=3e-4)
    # Mild structural precision-recall balance boosting sheet & helix sensitivity slightly
    cls_weights = np.array([1.03, 1.12, 0.95], dtype=np.float32)

    n_tr = len(tr_pack["y_packed"])
    n_val = len(val_pack["y_packed"])
    best_val_acc = -1.0
    best_snapshot: Dict[str, np.ndarray] = {}

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        lr_ep = 1.2e-4 + 0.5 * (2.4e-3 - 1.2e-4) * (1.0 + math.cos(math.pi * (ep - 1) / epochs))
        perm = rng.permutation(n_tr)
        ep_loss = 0.0
        n_batches = 0

        for start in range(0, n_tr, batch_size):
            b_idx = perm[start:start + batch_size]
            xb = _slice_w17_batch(tr_pack["loc_buf"], tr_pack["glob_packed"], tr_pack["buf_centers"], b_idx)
            yb = tr_pack["y_packed"][b_idx]

            z1 = bn1.forward(xb @ params["W1"], training=True)
            a1 = _lrelu_fwd(z1)
            a1_d, m1 = _dropout(a1, 0.20, rng)

            zr = bn_res.forward(a1_d @ params["W_res"], training=True)
            ar = _lrelu_fwd(zr)
            h_sum = a1_d + ar

            z2 = bn2.forward(h_sum @ params["W2"], training=True)
            a2 = _lrelu_fwd(z2)
            a2_d, m2 = _dropout(a2, 0.12, rng)

            logits = a2_d @ params["W3"] + params["b3"]
            loss, _, d_logits = _softmax_ce_grad(logits, yb, label_smoothing=0.02, class_weights=cls_weights)
            ep_loss += loss
            n_batches += 1

            gW3 = a2_d.T @ d_logits
            gb3 = d_logits.sum(axis=0)
            da2 = (d_logits @ params["W3"].T) * m2
            dz2 = _lrelu_bwd(z2, da2)
            du2, g_bn2_g, g_bn2_b = bn2.backward(dz2)
            gW2 = h_sum.T @ du2

            dh_sum = du2 @ params["W2"].T
            dzr = _lrelu_bwd(zr, dh_sum)
            dur, g_bnr_g, g_bnr_b = bn_res.backward(dzr)
            gW_res = a1_d.T @ dur

            da1_d = dh_sum + (dur @ params["W_res"].T)
            da1 = da1_d * m1
            dz1 = _lrelu_bwd(z1, da1)
            du1, g_bn1_g, g_bn1_b = bn1.backward(dz1)
            gW1 = xb.T @ du1

            opt.step(
                {
                    "W1": gW1, "W_res": gW_res, "W2": gW2, "W3": gW3, "b3": gb3,
                    "bn1_g": g_bn1_g, "bn1_b": g_bn1_b,
                    "bnr_g": g_bnr_g, "bnr_b": g_bnr_b,
                    "bn2_g": g_bn2_g, "bn2_b": g_bn2_b,
                },
                lr_override=lr_ep,
            )

        correct = 0
        for start in range(0, n_val, 16384):
            v_idx = np.arange(start, min(n_val, start + 16384), dtype=np.int32)
            xv = _slice_w17_batch(val_pack["loc_buf"], val_pack["glob_packed"], val_pack["buf_centers"], v_idx)
            a1 = _lrelu_fwd(bn1.forward(xv @ params["W1"], training=False))
            ar = _lrelu_fwd(bn_res.forward(a1 @ params["W_res"], training=False))
            a2 = _lrelu_fwd(bn2.forward((a1 + ar) @ params["W2"], training=False))
            lv = a2 @ params["W3"] + params["b3"]
            correct += int((lv.argmax(axis=1) == val_pack["y_packed"][v_idx]).sum())

        val_acc = 100.0 * correct / n_val
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            W1_f, b1_f = bn1.fold_linear(params["W1"])
            Wr_f, br_f = bn_res.fold_linear(params["W_res"])
            W2_f, b2_f = bn2.fold_linear(params["W2"])
            best_snapshot = {
                "m2_W1": W1_f.copy(), "m2_b1": b1_f.copy(),
                "m2_W_res": Wr_f.copy(), "m2_b_res": br_f.copy(),
                "m2_W2": W2_f.copy(), "m2_b2": b2_f.copy(),
                "m2_W3": params["W3"].copy(), "m2_b3": params["b3"].copy(),
            }
        if ep in (1, 4, 8, 12, epochs):
            print(f"  [Stage-1 M2 ResHighway] Ep {ep:2d}/{epochs} | loss={ep_loss/n_batches:.4f} | val_Q3={val_acc:.2f}% (best {best_val_acc:.2f}%) [{time.perf_counter()-t0:.1f}s]")

    return best_snapshot


def _conv1d_col_fwd(h: np.ndarray, k: int, d: int) -> np.ndarray:
    L, C = h.shape
    r = (k // 2) * d
    p = np.zeros((L + 2 * r, C), dtype=np.float32)
    p[r:r + L] = h
    return np.hstack([p[s:s + L] for s in range(0, k * d, d)])


def _conv1d_col_bwd(dx_col: np.ndarray, L: int, C: int, k: int, d: int) -> np.ndarray:
    r = (k // 2) * d
    dp = np.zeros((L + 2 * r, C), dtype=np.float32)
    for j, s in enumerate(range(0, k * d, d)):
        dp[s:s + L] += dx_col[:, j * C:(j + 1) * C]
    return dp[r:r + L]


def train_stage1_model3_dilated_conv(
    tr_pack: Dict[str, Any],
    val_pack: Dict[str, Any],
    seed: int = 303,
    epochs: int = 18,
    prots_per_batch: int = 28,
) -> Dict[str, np.ndarray]:
    """
    Trains Model 3: 4-Block Multi-Scale Dilated 1D Residual ConvNet + Squeeze-Excitation
    (receptive field = 31 residues) using 8-zero-row protein boundary separators so each
    protein inside a batch has 100% identical zero-padded boundary behavior to single-protein inference!
    """
    rng = np.random.default_rng(seed)
    d_in, ch, se_ch = 81 + 31, 96, 24

    bn_s = BatchNorm1D(ch)
    bn1 = BatchNorm1D(ch)
    bn2 = BatchNorm1D(ch)
    bn3 = BatchNorm1D(ch)
    bn4 = BatchNorm1D(ch)

    params = {
        "W_stem": (rng.standard_normal((d_in, ch)) * math.sqrt(2.0 / d_in)).astype(np.float32),
        "Wc1": (rng.standard_normal((5 * ch, ch)) * math.sqrt(1.2 / (5 * ch))).astype(np.float32),
        "Wc2": (rng.standard_normal((5 * ch, ch)) * math.sqrt(1.2 / (5 * ch))).astype(np.float32),
        "Wc3": (rng.standard_normal((5 * ch, ch)) * math.sqrt(1.2 / (5 * ch))).astype(np.float32),
        "W_se1": (rng.standard_normal((ch, se_ch)) * math.sqrt(2.0 / ch)).astype(np.float32),
        "b_se1": np.zeros(se_ch, dtype=np.float32),
        "W_se2": (rng.standard_normal((se_ch, ch)) * math.sqrt(1.0 / se_ch)).astype(np.float32),
        "b_se2": np.zeros(ch, dtype=np.float32),
        "Wc4": (rng.standard_normal((3 * ch, ch)) * math.sqrt(1.2 / (3 * ch))).astype(np.float32),
        "W_out": (rng.standard_normal((ch, 3)) * math.sqrt(1.0 / ch)).astype(np.float32),
        "b_out": np.zeros(3, dtype=np.float32),
        "bns_g": bn_s.gamma, "bns_b": bn_s.beta,
        "bn1_g": bn1.gamma, "bn1_b": bn1.beta,
        "bn2_g": bn2.gamma, "bn2_b": bn2.beta,
        "bn3_g": bn3.gamma, "bn3_b": bn3.beta,
        "bn4_g": bn4.gamma, "bn4_b": bn4.beta,
    }
    opt = AdamW(params, lr=2.5e-3, weight_decay=2.5e-4)

    n_prots = len(tr_pack["prot_slices"])
    best_val_acc = -1.0
    best_snapshot: Dict[str, np.ndarray] = {}

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        lr_ep = 1.2e-4 + 0.5 * (2.5e-3 - 1.2e-4) * (1.0 + math.cos(math.pi * (ep - 1) / epochs))
        p_perm = rng.permutation(n_prots)
        ep_loss = 0.0
        n_batches = 0

        for b_start in range(0, n_prots, prots_per_batch):
            p_ids = p_perm[b_start:b_start + prots_per_batch]
            P = len(p_ids)

            # Pack P proteins with 8 zero-separator rows between proteins (max dilation radius = 8)
            segs = [tr_pack["prot_slices"][pid] for pid in p_ids]
            lens = [re - rs for rs, re, _ in segs]
            M = sum(lens) + 8 * max(0, P - 1)

            x_stem = np.zeros((M, d_in), dtype=np.float32)
            valid_mask = np.zeros(M, dtype=bool)
            prot_owner = np.full(M, -1, dtype=np.int32)
            yb_list: List[np.ndarray] = []

            cur = 0
            for local_p, (rs, re, bs) in enumerate(segs):
                L_p = re - rs
                x_stem[cur:cur + L_p, :81] = tr_pack["loc_buf"][bs:bs + L_p]
                x_stem[cur:cur + L_p, 81:] = tr_pack["glob_packed"][rs:re]
                valid_mask[cur:cur + L_p] = True
                prot_owner[cur:cur + L_p] = local_p
                yb_list.append(tr_pack["y_packed"][rs:re])
                cur += L_p + (8 if local_p + 1 < P else 0)

            valid_idx = np.flatnonzero(valid_mask)
            yb = np.concatenate(yb_list)
            owners_valid = prot_owner[valid_idx]
            lens_arr = np.array(lens, dtype=np.float32)[:, None]

            # Forward on valid residues for stem BN, then scatter to M-row sequence buffer
            z_s_val = bn_s.forward(x_stem[valid_idx] @ params["W_stem"], training=True)
            h0_val = _hs_fwd(z_s_val)
            h0_d_val, m0 = _dropout(h0_val, 0.12, rng)
            h0 = np.zeros((M, ch), dtype=np.float32)
            h0[valid_idx] = h0_d_val

            # Block 1 (k=5, d=1)
            col1 = _conv1d_col_fwd(h0, 5, 1)
            z1_val = bn1.forward(col1[valid_idx] @ params["Wc1"], training=True)
            b1_val = _hs_fwd(z1_val)
            h1 = np.zeros((M, ch), dtype=np.float32)
            h1[valid_idx] = h0[valid_idx] + b1_val

            # Block 2 (k=5, d=2)
            col2 = _conv1d_col_fwd(h1, 5, 2)
            z2_val = bn2.forward(col2[valid_idx] @ params["Wc2"], training=True)
            b2_val = _hs_fwd(z2_val)
            h2 = np.zeros((M, ch), dtype=np.float32)
            h2[valid_idx] = h1[valid_idx] + b2_val

            # Block 3 (k=5, d=4) + Squeeze-and-Excitation per-protein gate
            col3 = _conv1d_col_fwd(h2, 5, 4)
            z3_val = bn3.forward(col3[valid_idx] @ params["Wc3"], training=True)
            b3_val = _hs_fwd(z3_val)  # (N_val, ch)

            se_means = np.zeros((P, ch), dtype=np.float32)
            np.add.at(se_means, owners_valid, b3_val)
            se_means /= lens_arr  # (P, ch)

            se_h1 = _hs_fwd(se_means @ params["W_se1"] + params["b_se1"])  # (P, se_ch)
            se_z2 = se_h1 @ params["W_se2"] + params["b_se2"]              # (P, ch)
            se_gate = np.clip(se_z2 * np.float32(0.2) + np.float32(0.5), np.float32(0.0), np.float32(1.0))
            gate_per_res = np.float32(2.0) * se_gate[owners_valid]         # (N_val, ch)

            h3 = np.zeros((M, ch), dtype=np.float32)
            h3[valid_idx] = h2[valid_idx] + b3_val * gate_per_res

            # Block 4 (k=3, d=1)
            col4 = _conv1d_col_fwd(h3, 3, 1)
            z4_val = bn4.forward(col4[valid_idx] @ params["Wc4"], training=True)
            b4_val = _hs_fwd(z4_val)
            h4_val = h3[valid_idx] + b4_val
            h4_d_val, m4 = _dropout(h4_val, 0.12, rng)

            logits = h4_d_val @ params["W_out"] + params["b_out"]
            loss, _, d_logits = _softmax_ce_grad(logits, yb, label_smoothing=0.025)
            ep_loss += loss
            n_batches += 1

            # Backward pass
            gW_out = h4_d_val.T @ d_logits
            gb_out = d_logits.sum(axis=0)
            dh4_val = (d_logits @ params["W_out"].T) * m4

            # Block 4 backward
            dz4_val = _hs_bwd(z4_val, dh4_val)
            du4_val, g_bn4_g, g_bn4_b = bn4.backward(dz4_val)
            gWc4 = col4[valid_idx].T @ du4_val
            dcol4 = np.zeros((M, 3 * ch), dtype=np.float32)
            dcol4[valid_idx] = du4_val @ params["Wc4"].T
            dh3 = _conv1d_col_bwd(dcol4, M, ch, 3, 1)
            dh3[valid_idx] += dh4_val

            # Block 3 + SE backward
            dh3_val = dh3[valid_idx]
            db3_val = dh3_val * gate_per_res
            d_gate_per_res = dh3_val * b3_val * np.float32(2.0)

            d_se_gate = np.zeros((P, ch), dtype=np.float32)
            np.add.at(d_se_gate, owners_valid, d_gate_per_res)
            se_deriv = np.where((se_z2 * 0.2 + 0.5 > 0.0) & (se_z2 * 0.2 + 0.5 < 1.0), np.float32(0.2), np.float32(0.0))
            d_se_z2 = d_se_gate * se_deriv
            gW_se2 = se_h1.T @ d_se_z2
            gb_se2 = d_se_z2.sum(axis=0)
            d_se_h1 = d_se_z2 @ params["W_se2"].T
            d_se_z1 = _hs_bwd(se_means @ params["W_se1"] + params["b_se1"], d_se_h1)
            gW_se1 = se_means.T @ d_se_z1
            gb_se1 = d_se_z1.sum(axis=0)
            d_se_means = (d_se_z1 @ params["W_se1"].T) / lens_arr
            db3_val += d_se_means[owners_valid]

            dz3_val = _hs_bwd(z3_val, db3_val)
            du3_val, g_bn3_g, g_bn3_b = bn3.backward(dz3_val)
            gWc3 = col3[valid_idx].T @ du3_val
            dcol3 = np.zeros((M, 5 * ch), dtype=np.float32)
            dcol3[valid_idx] = du3_val @ params["Wc3"].T
            dh2 = _conv1d_col_bwd(dcol3, M, ch, 5, 4)
            dh2[valid_idx] += dh3_val

            # Block 2 backward
            dh2_val = dh2[valid_idx]
            dz2_val = _hs_bwd(z2_val, dh2_val)
            du2_val, g_bn2_g, g_bn2_b = bn2.backward(dz2_val)
            gWc2 = col2[valid_idx].T @ du2_val
            dcol2 = np.zeros((M, 5 * ch), dtype=np.float32)
            dcol2[valid_idx] = du2_val @ params["Wc2"].T
            dh1 = _conv1d_col_bwd(dcol2, M, ch, 5, 2)
            dh1[valid_idx] += dh2_val

            # Block 1 backward
            dh1_val = dh1[valid_idx]
            dz1_val = _hs_bwd(z1_val, dh1_val)
            du1_val, g_bn1_g, g_bn1_b = bn1.backward(dz1_val)
            gWc1 = col1[valid_idx].T @ du1_val
            dcol1 = np.zeros((M, 5 * ch), dtype=np.float32)
            dcol1[valid_idx] = du1_val @ params["Wc1"].T
            dh0 = _conv1d_col_bwd(dcol1, M, ch, 5, 1)
            dh0_val = (dh0[valid_idx] + dh1_val) * m0

            # Stem backward
            dz_s_val = _hs_bwd(z_s_val, dh0_val)
            du_s_val, g_bns_g, g_bns_b = bn_s.backward(dz_s_val)
            gW_stem = x_stem[valid_idx].T @ du_s_val

            opt.step(
                {
                    "W_stem": gW_stem, "Wc1": gWc1, "Wc2": gWc2, "Wc3": gWc3,
                    "W_se1": gW_se1, "b_se1": gb_se1, "W_se2": gW_se2, "b_se2": gb_se2,
                    "Wc4": gWc4, "W_out": gW_out, "b_out": gb_out,
                    "bns_g": g_bns_g, "bns_b": g_bns_b,
                    "bn1_g": g_bn1_g, "bn1_b": g_bn1_b,
                    "bn2_g": g_bn2_g, "bn2_b": g_bn2_b,
                    "bn3_g": g_bn3_g, "bn3_b": g_bn3_b,
                    "bn4_g": g_bn4_g, "bn4_b": g_bn4_b,
                },
                lr_override=lr_ep,
            )

        # Fold current BN into temporary inference dict and evaluate exact per-protein forward on val_pack
        Ws_f, bs_f = bn_s.fold_linear(params["W_stem"])
        Wc1_f, bc1_f = bn1.fold_linear(params["Wc1"])
        Wc2_f, bc2_f = bn2.fold_linear(params["Wc2"])
        Wc3_f, bc3_f = bn3.fold_linear(params["Wc3"])
        Wc4_f, bc4_f = bn4.fold_linear(params["Wc4"])

        correct = 0
        total_v = 0
        for pid, (rs, re, bs) in enumerate(val_pack["prot_slices"]):
            L_p = re - rs
            st_in = np.hstack([val_pack["loc_buf"][bs:bs + L_p], val_pack["glob_packed"][rs:re]])
            hc = _hs_fwd(st_in @ Ws_f + bs_f)
            hc = hc + _hs_fwd(_conv1d_col_fwd(hc, 5, 1) @ Wc1_f + bc1_f)
            hc = hc + _hs_fwd(_conv1d_col_fwd(hc, 5, 2) @ Wc2_f + bc2_f)
            hb3 = _hs_fwd(_conv1d_col_fwd(hc, 5, 4) @ Wc3_f + bc3_f)
            sem = hb3.mean(axis=0, keepdims=True)
            sez = _hs_fwd(sem @ params["W_se1"] + params["b_se1"]) @ params["W_se2"] + params["b_se2"]
            seg_g = np.clip(sez * np.float32(0.2) + np.float32(0.5), np.float32(0.0), np.float32(1.0))
            hc = hc + hb3 * (np.float32(2.0) * seg_g)
            hc = hc + _hs_fwd(_conv1d_col_fwd(hc, 3, 1) @ Wc4_f + bc4_f)
            lv = hc @ params["W_out"] + params["b_out"]
            correct += int((lv.argmax(axis=1) == val_pack["y_packed"][rs:re]).sum())
            total_v += L_p

        val_acc = 100.0 * correct / total_v
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_snapshot = {
                "m3_W_stem": Ws_f.copy(), "m3_b_stem": bs_f.copy(),
                "m3_Wc1": Wc1_f.copy(), "m3_bc1": bc1_f.copy(),
                "m3_Wc2": Wc2_f.copy(), "m3_bc2": bc2_f.copy(),
                "m3_Wc3": Wc3_f.copy(), "m3_bc3": bc3_f.copy(),
                "m3_W_se1": params["W_se1"].copy(), "m3_b_se1": params["b_se1"].copy(),
                "m3_W_se2": params["W_se2"].copy(), "m3_b_se2": params["b_se2"].copy(),
                "m3_Wc4": Wc4_f.copy(), "m3_bc4": bc4_f.copy(),
                "m3_W_out": params["W_out"].copy(), "m3_b_out": params["b_out"].copy(),
            }
        if ep in (1, 5, 9, 14, epochs):
            print(f"  [Stage-1 M3 DilatedConv1D] Ep {ep:2d}/{epochs} | loss={ep_loss/n_batches:.4f} | val_Q3={val_acc:.2f}% (best {best_val_acc:.2f}%) [{time.perf_counter()-t0:.1f}s]")

    return best_snapshot


def calibrate_temperature_1d(logits_list: List[np.ndarray], y_true: np.ndarray) -> float:
    """Finds optimal inverse-temperature scalar `inv_temp` minimizing NLL on validation residues."""
    cat_logits = np.vstack(logits_list)
    best_it = 1.0
    best_nll = 1e18
    for it in np.linspace(0.55, 1.45, 37):
        z = cat_logits * np.float32(it)
        z = z - z.max(axis=1, keepdims=True)
        ez = np.exp(z)
        pr = ez / ez.sum(axis=1, keepdims=True)
        nll = -float(np.log(np.clip(pr[np.arange(len(y_true)), y_true], 1e-8, 1.0)).mean())
        if nll < best_nll:
            best_nll = nll
            best_it = float(it)
    return best_it


def train_stage2_context_network(
    s2_tr_X: np.ndarray,
    s2_tr_y: np.ndarray,
    s2_tr_w: np.ndarray,
    s2_val_X: np.ndarray,
    s2_val_y: np.ndarray,
    seed: int = 404,
    epochs: int = 18,
    batch_size: int = 4096,
) -> Dict[str, np.ndarray]:
    """
    Trains the Stage-2 Structure-to-Structure Context Network (355 -> 128 -> 64 -> 3)
    taking 15-residue windows of Stage-1 probabilities + boundary breakers + global composition.
    """
    rng = np.random.default_rng(seed)
    s2_mean = s2_tr_X.mean(axis=0).astype(np.float32)
    s2_std = np.sqrt(((s2_tr_X - s2_mean) ** 2).mean(axis=0)).astype(np.float32)
    s2_inv_std = (1.0 / np.maximum(s2_std, 0.03)).astype(np.float32)

    X_tr = ((s2_tr_X - s2_mean) * s2_inv_std).astype(np.float32)
    X_val = ((s2_val_X - s2_mean) * s2_inv_std).astype(np.float32)

    d_in, h1, h2 = X_tr.shape[1], 128, 64
    bn1 = BatchNorm1D(h1)
    bn2 = BatchNorm1D(h2)
    params = {
        "W1": (rng.standard_normal((d_in, h1)) * math.sqrt(2.0 / d_in)).astype(np.float32),
        "W2": (rng.standard_normal((h1, h2)) * math.sqrt(2.0 / h1)).astype(np.float32),
        "W3": (rng.standard_normal((h2, 3)) * math.sqrt(1.0 / h2)).astype(np.float32),
        "b3": np.zeros(3, dtype=np.float32),
        "bn1_g": bn1.gamma, "bn1_b": bn1.beta,
        "bn2_g": bn2.gamma, "bn2_b": bn2.beta,
    }
    opt = AdamW(params, lr=2.0e-3, weight_decay=4e-4)

    n_tr = len(s2_tr_y)
    best_val_acc = -1.0
    best_snapshot: Dict[str, np.ndarray] = {}

    for ep in range(1, epochs + 1):
        lr_ep = 1.0e-4 + 0.5 * (2.0e-3 - 1.0e-4) * (1.0 + math.cos(math.pi * (ep - 1) / epochs))
        perm = rng.permutation(n_tr)
        for start in range(0, n_tr, batch_size):
            b_idx = perm[start:start + batch_size]
            xb = X_tr[b_idx]
            yb = s2_tr_y[b_idx]
            wb = s2_tr_w[b_idx]

            z1 = bn1.forward(xb @ params["W1"], training=True)
            a1 = _hs_fwd(z1)
            a1_d, m1 = _dropout(a1, 0.20, rng)

            z2 = bn2.forward(a1_d @ params["W2"], training=True)
            a2 = _hs_fwd(z2)
            a2_d, m2 = _dropout(a2, 0.10, rng)

            logits = a2_d @ params["W3"] + params["b3"]
            _, _, d_logits = _softmax_ce_grad(logits, yb, label_smoothing=0.015, sample_weights=wb)

            gW3 = a2_d.T @ d_logits
            gb3 = d_logits.sum(axis=0)
            da2 = (d_logits @ params["W3"].T) * m2
            dz2 = _hs_bwd(z2, da2)
            du2, g_bn2_g, g_bn2_b = bn2.backward(dz2)
            gW2 = a1_d.T @ du2

            da1 = (du2 @ params["W2"].T) * m1
            dz1 = _hs_bwd(z1, da1)
            du1, g_bn1_g, g_bn1_b = bn1.backward(dz1)
            gW1 = xb.T @ du1

            opt.step(
                {
                    "W1": gW1, "W2": gW2, "W3": gW3, "b3": gb3,
                    "bn1_g": g_bn1_g, "bn1_b": g_bn1_b,
                    "bn2_g": g_bn2_g, "bn2_b": g_bn2_b,
                },
                lr_override=lr_ep,
            )

        a1_v = _hs_fwd(bn1.forward(X_val @ params["W1"], training=False))
        a2_v = _hs_fwd(bn2.forward(a1_v @ params["W2"], training=False))
        lv = a2_v @ params["W3"] + params["b3"]
        val_acc = 100.0 * float((lv.argmax(axis=1) == s2_val_y).mean())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            W1_f, b1_f = bn1.fold_linear(params["W1"])
            W2_f, b2_f = bn2.fold_linear(params["W2"])
            best_snapshot = {
                "s2_mean": s2_mean,
                "s2_inv_std": s2_inv_std,
                "s2_W1": W1_f.copy(), "s2_b1": b1_f.copy(),
                "s2_W2": W2_f.copy(), "s2_b2": b2_f.copy(),
                "s2_W3": params["W3"].copy(), "s2_b3": params["b3"].copy(),
            }
        if ep in (1, 6, 12, epochs):
            print(f"  [Stage-2 ContextNet] Ep {ep:2d}/{epochs} | cal_val_Q3={val_acc:.2f}% (best {best_val_acc:.2f}%)")

    # Calibrate Stage-2 inverse temperature on cal_val
    a1_v = _hs_fwd(X_val @ best_snapshot["s2_W1"] + best_snapshot["s2_b1"])
    a2_v = _hs_fwd(a1_v @ best_snapshot["s2_W2"] + best_snapshot["s2_b2"])
    lv = a2_v @ best_snapshot["s2_W3"] + best_snapshot["s2_b3"]
    s2_it = calibrate_temperature_1d([lv], s2_val_y)
    best_snapshot["s2_inv_temp"] = np.array(s2_it, dtype=np.float32)
    return best_snapshot


# ==============================================================================
# 5. FULL ISOLATED ZERO-LEAKAGE TRAINING PIPELINE (`splits == "train"` ONLY)
# ==============================================================================

def train_cascaded_ss_engine(
    train_sequences: List[str],
    train_ss3: List[str],
    train_strata: List[str],
    existing_stage1_bundle: Dict[str, np.ndarray] | None = None,
) -> Dict[str, np.ndarray]:
    """
    Trains the entire cascaded secondary structure prediction system using ONLY
    `splits == "train"` proteins (`test` split is never passed or accessed here!).
    """
    print(f"\n=== [TRAIN] Starting Zero-Leakage Training on {len(train_sequences):,} `train` Proteins ===")
    rng = np.random.default_rng(20261116)

    # Stratified split of `train` into s1_train (76%), s2_train (14%), cal_val (10%)
    buckets: Dict[Tuple[str, str], List[int]] = {}
    for idx, (ss, st) in enumerate(zip(train_ss3, train_strata)):
        L = len(ss)
        fc, _ = classify_fold_class(100.0 * ss.count("H") / L, 100.0 * ss.count("E") / L)
        buckets.setdefault((st, fc), []).append(idx)

    s1_idx: List[int] = []
    s2_idx: List[int] = []
    cal_idx: List[int] = []
    for key in sorted(buckets.keys()):
        arr = np.array(buckets[key], dtype=np.int32)
        rng.shuffle(arr)
        n = len(arr)
        n_cal = max(1, int(round(0.10 * n))) if n >= 5 else 0
        n_s2 = max(1, int(round(0.14 * n))) if n >= 4 else 0
        cal_idx.extend(arr[:n_cal].tolist())
        s2_idx.extend(arr[n_cal:n_cal + n_s2].tolist())
        s1_idx.extend(arr[n_cal + n_s2:].tolist())

    print(f"  Internal `train`-only split: s1_train={len(s1_idx)} | s2_train={len(s2_idx)} | cal_val={len(cal_idx)}")

    s1_seqs = [train_sequences[i] for i in s1_idx]
    s1_ss = [train_ss3[i] for i in s1_idx]
    s2_seqs = [train_sequences[i] for i in s2_idx]
    s2_ss = [train_ss3[i] for i in s2_idx]
    cal_seqs = [train_sequences[i] for i in cal_idx]
    cal_ss = [train_ss3[i] for i in cal_idx]

    # 1. 5-Fold Out-of-Fold GOR-IV tables inside s1_train so Stage-1 sees unbiased unseen-protein GOR features
    print("  [1/6] Computing 5-fold out-of-fold GOR-IV directional pairwise tables on `s1_train`...")
    t_g0 = time.perf_counter()
    n_s1 = len(s1_seqs)
    fold_ids = np.arange(n_s1) % 5
    rng.shuffle(fold_ids)

    fold_counts = []
    tot_state = np.zeros(3, dtype=np.float64)
    tot_single = np.zeros((17, 21, 3), dtype=np.float64)
    tot_pair = np.zeros((136, 441, 3), dtype=np.float64)

    for f in range(5):
        f_seqs = [s1_seqs[i] for i in range(n_s1) if fold_ids[i] == f]
        f_ss = [s1_ss[i] for i in range(n_s1) if fold_ids[i] == f]
        st_c, si_c, pa_c = accumulate_gor4_counts(f_seqs, f_ss)
        fold_counts.append((st_c, si_c, pa_c))
        tot_state += st_c
        tot_single += si_c
        tot_pair += pa_c

    oof_gor_tables = [
        counts_to_gor4_tables(tot_state - fold_counts[f][0], tot_single - fold_counts[f][1], tot_pair - fold_counts[f][2])
        for f in range(5)
    ]
    s1_per_prot_gor = [oof_gor_tables[int(fold_ids[i])] for i in range(n_s1)]
    # Exported GOR-IV table keeps gor_temperature = 1.0 so feature scaling matches loc_mean/glob_mean identically!
    s1_full_gor = counts_to_gor4_tables(tot_state, tot_single, tot_pair)
    s1_full_gor["gor_temperature"] = np.array(1.0, dtype=np.float32)
    export_gor = s1_full_gor
    print(f"        GOR-IV tables built in {time.perf_counter() - t_g0:.1f}s.")

    # 2. Extract & pack features
    print("  [2/6] Extracting & normalizing biophysical + evolutionary + amphipathic + GOR-IV features...")
    t_f0 = time.perf_counter()
    if existing_stage1_bundle is not None:
        loc_mean_in = existing_stage1_bundle["loc_mean"]
        loc_inv_std_in = existing_stage1_bundle["loc_inv_std"]
        glob_mean_in = existing_stage1_bundle["glob_mean"]
        glob_inv_std_in = existing_stage1_bundle["glob_inv_std"]
    else:
        loc_mean_in = loc_inv_std_in = glob_mean_in = glob_inv_std_in = None

    tr_pack = pack_protein_features(
        s1_seqs, s1_ss, s1_per_prot_gor,
        loc_mean=loc_mean_in, loc_inv_std=loc_inv_std_in,
        glob_mean=glob_mean_in, glob_inv_std=glob_inv_std_in,
    )
    s2_pack = pack_protein_features(
        s2_seqs, s2_ss, export_gor,
        loc_mean=tr_pack["loc_mean"], loc_inv_std=tr_pack["loc_inv_std"],
        glob_mean=tr_pack["glob_mean"], glob_inv_std=tr_pack["glob_inv_std"],
    )
    cal_pack = pack_protein_features(
        cal_seqs, cal_ss, export_gor,
        loc_mean=tr_pack["loc_mean"], loc_inv_std=tr_pack["loc_inv_std"],
        glob_mean=tr_pack["glob_mean"], glob_inv_std=tr_pack["glob_inv_std"],
    )
    print(f"        Packed {len(tr_pack['y_packed']):,} + {len(s2_pack['y_packed']):,} + {len(cal_pack['y_packed']):,} residues in {time.perf_counter() - t_f0:.1f}s.")

    # 3. Train (or reuse train-only checkpoint of) the 3 diverse Stage-1 models
    if existing_stage1_bundle is not None:
        print("  [3/6] Reusing strictly `train`-fitted Stage-1 M1, M2, M3 weights...")
        m1_weights = {k: existing_stage1_bundle[k] for k in ("m1_W1", "m1_b1", "m1_W2", "m1_b2", "m1_W3", "m1_b3")}
        m2_weights = {k: existing_stage1_bundle[k] for k in ("m2_W1", "m2_b1", "m2_W_res", "m2_b_res", "m2_W2", "m2_b2", "m2_W3", "m2_b3")}
        m3_weights = {
            k: existing_stage1_bundle[k]
            for k in (
                "m3_W_stem", "m3_b_stem", "m3_Wc1", "m3_bc1", "m3_Wc2", "m3_bc2",
                "m3_Wc3", "m3_bc3", "m3_W_se1", "m3_b_se1", "m3_W_se2", "m3_b_se2",
                "m3_Wc4", "m3_bc4", "m3_W_out", "m3_b_out",
            )
        }
    else:
        print("  [3/6] Training 3 Diverse Stage-1 Neural Architectures...")
        m1_weights = train_stage1_model1_wide_mlp(tr_pack, s2_pack, seed=101, epochs=16)
        m2_weights = train_stage1_model2_res_highway(tr_pack, s2_pack, seed=202, epochs=16)
        m3_weights = train_stage1_model3_dilated_conv(tr_pack, s2_pack, seed=303, epochs=18)

    bundle: Dict[str, np.ndarray] = {}
    bundle.update(export_gor)
    bundle["loc_mean"] = tr_pack["loc_mean"]
    bundle["loc_inv_std"] = tr_pack["loc_inv_std"]
    bundle["glob_mean"] = tr_pack["glob_mean"]
    bundle["glob_inv_std"] = tr_pack["glob_inv_std"]
    bundle["loc_pad_norm"] = tr_pack["loc_pad_norm"]
    bundle.update(m1_weights)
    bundle.update(m2_weights)
    bundle.update(m3_weights)
    bundle["m1_inv_temp"] = np.array(1.0, dtype=np.float32)
    bundle["m2_inv_temp"] = np.array(1.0, dtype=np.float32)
    bundle["m3_inv_temp"] = np.array(1.0, dtype=np.float32)
    bundle["s1_ens_weights"] = np.array([0.34, 0.33, 0.33], dtype=np.float32)
    bundle["m12_W1_fused"] = np.ascontiguousarray(np.hstack([bundle["m1_W1"], bundle["m2_W1"]]))
    bundle["m12_b1_fused"] = np.ascontiguousarray(np.concatenate([bundle["m1_b1"], bundle["m2_b1"]]))

    # 4. Calibrate Stage-1 model temperatures & ensemble weights on unseen s2_train
    print("  [4/6] Calibrating Stage-1 Temperatures & Convex Ensemble Weights on `s2_train`...")
    s2_p1_list, s2_p2_list, s2_p3_list = [], [], []
    for xl, xg in zip(s2_pack["raw_loc_list"], s2_pack["raw_glob_list"]):
        p1, p2, p3, _ = _forward_stage1_ensemble(xl, xg, bundle)
        s2_p1_list.append(np.log(np.clip(p1, 1e-7, 1.0)))
        s2_p2_list.append(np.log(np.clip(p2, 1e-7, 1.0)))
        s2_p3_list.append(np.log(np.clip(p3, 1e-7, 1.0)))

    it1 = calibrate_temperature_1d(s2_p1_list, s2_pack["y_packed"])
    it2 = calibrate_temperature_1d(s2_p2_list, s2_pack["y_packed"])
    it3 = calibrate_temperature_1d(s2_p3_list, s2_pack["y_packed"])
    bundle["m1_inv_temp"] = np.array(it1, dtype=np.float32)
    bundle["m2_inv_temp"] = np.array(it2, dtype=np.float32)
    bundle["m3_inv_temp"] = np.array(it3, dtype=np.float32)

    # Grid-search optimal convex weights (w1, w2, w3) on s2_train
    s2_p1_cal, s2_p2_cal, s2_p3_cal = [], [], []
    for xl, xg in zip(s2_pack["raw_loc_list"], s2_pack["raw_glob_list"]):
        p1, p2, p3, _ = _forward_stage1_ensemble(xl, xg, bundle)
        s2_p1_cal.append(p1)
        s2_p2_cal.append(p2)
        s2_p3_cal.append(p3)

    P1_cat = np.vstack(s2_p1_cal)
    P2_cat = np.vstack(s2_p2_cal)
    P3_cat = np.vstack(s2_p3_cal)

    best_ens_q3 = -1.0
    best_w = np.array([0.34, 0.33, 0.33], dtype=np.float32)
    for w1 in np.linspace(0.15, 0.55, 9):
        for w2 in np.linspace(0.15, 0.55, 9):
            w3 = 1.0 - w1 - w2
            if w3 < 0.10 or w3 > 0.65:
                continue
            p_mix = w1 * P1_cat + w2 * P2_cat + w3 * P3_cat
            q3_mix = 100.0 * float((p_mix.argmax(axis=1) == s2_pack["y_packed"]).mean())
            if q3_mix > best_ens_q3:
                best_ens_q3 = q3_mix
                best_w = np.array([w1, w2, w3], dtype=np.float32)

    bundle["s1_ens_weights"] = best_w
    print(f"        Stage-1 Ensemble weights={best_w.round(3).tolist()} -> unseen s2_train Q3={best_ens_q3:.2f}% "
          f"(M1={100.0*float((P1_cat.argmax(1)==s2_pack['y_packed']).mean()):.2f}%, "
          f"M2={100.0*float((P2_cat.argmax(1)==s2_pack['y_packed']).mean()):.2f}%, "
          f"M3={100.0*float((P3_cat.argmax(1)==s2_pack['y_packed']).mean()):.2f}%)")

    # 5. Build Stage-2 Structure-to-Structure training set (s2_train weighted 2.5x + subsample of s1_train weighted 1.0x)
    print("  [5/6] Building Stage-2 Structure-to-Structure Context Dataset & Training Stage-2 Network...")
    s2_X_parts: List[np.ndarray] = []
    s2_y_parts: List[np.ndarray] = []
    s2_w_parts: List[np.ndarray] = []

    for i, (xl, xg, pg) in enumerate(zip(s2_pack["raw_loc_list"], s2_pack["raw_glob_list"], s2_pack["raw_gor_list"])):
        p1, p2, p3, pens = _forward_stage1_ensemble(xl, xg, bundle)
        s2_in = build_stage2_inputs(p1, p2, p3, pens, pg, xl, xg)
        rs, re, _ = s2_pack["prot_slices"][i]
        s2_X_parts.append(s2_in)
        s2_y_parts.append(s2_pack["y_packed"][rs:re])
        s2_w_parts.append(np.full(re - rs, 2.5, dtype=np.float32))

    # Add 550 proteins from s1_train with mild Dirichlet probability jitter so Stage-2 sees extra topological motifs without overconfidence
    sub_s1_ids = rng.choice(len(s1_seqs), size=min(550, len(s1_seqs)), replace=False)
    for idx_p in sub_s1_ids:
        xl = tr_pack["raw_loc_list"][idx_p]
        xg = tr_pack["raw_glob_list"][idx_p]
        pg = tr_pack["raw_gor_list"][idx_p]
        p1, p2, p3, pens = _forward_stage1_ensemble(xl, xg, bundle)
        # Mild temperature softening (0.85) so in-sample probabilities match out-of-sample sharpness
        def _soften(pr: np.ndarray) -> np.ndarray:
            lp = np.log(np.clip(pr, 1e-7, 1.0)) * np.float32(0.85)
            lp -= lp.max(axis=1, keepdims=True)
            ep_ = np.exp(lp)
            return ep_ / ep_.sum(axis=1, keepdims=True)
        p1_s, p2_s, p3_s = _soften(p1), _soften(p2), _soften(p3)
        pens_s = best_w[0] * p1_s + best_w[1] * p2_s + best_w[2] * p3_s
        s2_in = build_stage2_inputs(p1_s, p2_s, p3_s, pens_s, pg, xl, xg)
        rs, re, _ = tr_pack["prot_slices"][idx_p]
        s2_X_parts.append(s2_in)
        s2_y_parts.append(tr_pack["y_packed"][rs:re])
        s2_w_parts.append(np.ones(re - rs, dtype=np.float32))

    # Build untouched cal_val Stage-2 inputs
    cal_X_list: List[np.ndarray] = []
    cal_pens_list: List[np.ndarray] = []
    for xl, xg, pg in zip(cal_pack["raw_loc_list"], cal_pack["raw_glob_list"], cal_pack["raw_gor_list"]):
        p1, p2, p3, pens = _forward_stage1_ensemble(xl, xg, bundle)
        cal_X_list.append(build_stage2_inputs(p1, p2, p3, pens, pg, xl, xg))
        cal_pens_list.append(pens)

    s2_weights = train_stage2_context_network(
        np.vstack(s2_X_parts),
        np.concatenate(s2_y_parts),
        np.concatenate(s2_w_parts),
        np.vstack(cal_X_list),
        cal_pack["y_packed"],
        seed=404,
        epochs=18,
    )
    bundle.update(s2_weights)

    # 6. Calibrate Stage-2 blend alpha & 7-state Constrained Viterbi parameters strictly on untouched `cal_val`
    print("  [6/6] Tuning Biophysical Minimum-Length Viterbi Transitions & Prior Calibration on `cal_val`...")
    cal_s2_probs = [_forward_stage2(cx, bundle) for cx in cal_X_list]

    best_alpha = 0.70
    best_cal_q3 = -1.0
    for alpha in np.linspace(0.45, 0.90, 10):
        corr = 0
        for p_s2, p_s1, (rs, re, _) in zip(cal_s2_probs, cal_pens_list, cal_pack["prot_slices"]):
            pb = alpha * p_s2 + (1.0 - alpha) * p_s1
            corr += int((pb.argmax(axis=1) == cal_pack["y_packed"][rs:re]).sum())
        q3_a = 100.0 * corr / len(cal_pack["y_packed"])
        if q3_a > best_cal_q3:
            best_cal_q3 = q3_a
            best_alpha = float(alpha)

    bundle["s2_blend_alpha"] = np.array(best_alpha, dtype=np.float32)
    cal_blended = [
        (best_alpha * p_s2 + (1.0 - best_alpha) * p_s1) / (best_alpha * p_s2 + (1.0 - best_alpha) * p_s1).sum(axis=1, keepdims=True)
        for p_s2, p_s1 in zip(cal_s2_probs, cal_pens_list)
    ]

    # Coordinate search over state_bias [b_H, b_E, 0.0] and transition penalties on `cal_val`
    def _eval_viterbi_on_cal(bias_h: float, bias_e: float, entry_pen: float, self_bonus: float) -> Tuple[float, float, float]:
        tm = np.array([
            [self_bonus, -0.65, -0.22],
            [-0.65, self_bonus * 0.85, -0.18],
            [-entry_pen, -entry_pen * 0.80, 0.0],
        ], dtype=np.float32)
        sb = np.array([bias_h, bias_e, 0.0], dtype=np.float32)
        tot_c = 0
        tot_n = 0
        sov_s = 0.0
        comp_err = 0.0
        for pb, true_ss in zip(cal_blended, cal_ss):
            pred_ss = decode_viterbi_min_length(pb, trans_matrix=tm, state_bias=sb)
            L_p = len(true_ss)
            tot_c += sum(1 for a, b in zip(true_ss, pred_ss) if a == b)
            tot_n += L_p
            sov_s += compute_sov99(true_ss, pred_ss)
            comp_err += abs(pred_ss.count("H") - true_ss.count("H")) / L_p + abs(pred_ss.count("E") - true_ss.count("E")) / L_p
        q3_v = 100.0 * tot_c / tot_n
        sov_v = sov_s / len(cal_ss)
        mae_v = 50.0 * comp_err / len(cal_ss)
        # Joint objective maximizing Q3 and SOV_99 while keeping composition well-calibrated
        score = q3_v + 0.35 * sov_v - 0.25 * mae_v
        return score, q3_v, sov_v

    best_v_params = (0.04, 0.08, 0.32, 0.08)
    best_v_score, best_vq3, best_vsov = _eval_viterbi_on_cal(*best_v_params)

    for bh in (-0.02, 0.02, 0.06, 0.10):
        for be in (0.02, 0.07, 0.12, 0.17):
            for ep_pen in (0.24, 0.34, 0.44):
                for sb_bon in (0.04, 0.09):
                    sc, vq3, vsov = _eval_viterbi_on_cal(bh, be, ep_pen, sb_bon)
                    if sc > best_v_score:
                        best_v_score = sc
                        best_vq3 = vq3
                        best_vsov = vsov
                        best_v_params = (bh, be, ep_pen, sb_bon)

    bh, be, ep_pen, sb_bon = best_v_params
    bundle["viterbi_trans"] = np.array([
        [sb_bon, -0.65, -0.22],
        [-0.65, sb_bon * 0.85, -0.18],
        [-ep_pen, -ep_pen * 0.80, 0.0],
    ], dtype=np.float32)
    bundle["viterbi_bias"] = np.array([bh, be, 0.0], dtype=np.float32)
    print(f"        Calibrated Viterbi on untouched `cal_val`: Q3={best_vq3:.2f}%, SOV_99={best_vsov:.2f}% "
          f"(bias_H={bh:+.2f}, bias_E={be:+.2f}, entry_pen={ep_pen:.2f})")

    # Strip runtime-only convenience cache keys before saving clean float32 .npz
    save_dict = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in bundle.items()
        if k not in ("m12_W1_fused", "m12_b1_fused", "gor_pair_2d")
    }
    return save_dict


# ==============================================================================
# 6. COMPREHENSIVE MULTI-DOMAIN EVALUATION METRICS
# ==============================================================================

def compute_confusion_and_mcc(conf: np.ndarray) -> Dict[str, Any]:
    """
    Computes pooled Q3, per-state Precision, Recall, F1, and Matthews Correlation
    Coefficient (MCC_H, MCC_E, MCC_C) from a 3x3 integer confusion matrix [obs, pred].
    """
    total = float(conf.sum())
    q3 = 100.0 * float(np.trace(conf)) / max(1.0, total)
    per_state: Dict[str, Dict[str, float]] = {}

    for idx, st in enumerate(("H", "E", "C")):
        tp = float(conf[idx, idx])
        fp = float(conf[:, idx].sum() - tp)
        fn = float(conf[idx, :].sum() - tp)
        tn = total - tp - fp - fn

        prec = 100.0 * tp / max(1.0, tp + fp)
        rec = 100.0 * tp / max(1.0, tp + fn)
        f1 = (2.0 * prec * rec / max(1e-9, prec + rec)) if (prec + rec) > 0 else 0.0
        denom = math.sqrt(max(1.0, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

        per_state[st] = {
            "precision_pct": round(prec, 2),
            "recall_pct": round(rec, 2),
            "f1_pct": round(f1, 2),
            "mcc": round(mcc, 4),
            "support_residues": int(conf[idx, :].sum()),
        }

    return {
        "overall_q3_pct": round(q3, 2),
        "MCC_H": per_state["H"]["mcc"],
        "MCC_E": per_state["E"]["mcc"],
        "MCC_C": per_state["C"]["mcc"],
        "mean_mcc": round((per_state["H"]["mcc"] + per_state["E"]["mcc"] + per_state["C"]["mcc"]) / 3.0, 4),
        "per_state": per_state,
        "confusion_matrix_HEC": conf.tolist(),
    }


def _update_conf_matrix(conf: np.ndarray, obs_ss: str, pred_ss: str) -> None:
    obs_arr = np.fromiter((STATE_TO_IDX[ch] for ch in obs_ss), dtype=np.int32, count=len(obs_ss))
    pred_arr = np.fromiter((STATE_TO_IDX[ch] for ch in pred_ss), dtype=np.int32, count=len(pred_ss))
    pair_idx = obs_arr * 3 + pred_arr
    conf += np.bincount(pair_idx, minlength=9).reshape(3, 3)


def evaluate_method_on_test(
    method_name: str,
    test_records: List[Dict[str, Any]],
    pred_fn,
) -> Dict[str, Any]:
    """Evaluates a secondary structure predictor across all >=11,000 held-out test proteins."""
    print(f"\n=== [BENCHMARK] Evaluating `{method_name}` on {len(test_records):,} Held-Out Test Proteins ===")
    conf_all = np.zeros((3, 3), dtype=np.int64)
    conf_by_domain: Dict[str, np.ndarray] = {}
    conf_by_stratum: Dict[str, np.ndarray] = {}

    q3_list: List[float] = []
    sov_list: List[float] = []
    sov_weighted_num = 0.0
    sov_weighted_den = 0.0

    dom_q3: Dict[str, List[float]] = {}
    dom_sov: Dict[str, List[float]] = {}
    str_q3: Dict[str, List[float]] = {}
    str_sov: Dict[str, List[float]] = {}

    err_h: List[float] = []
    err_e: List[float] = []
    err_c: List[float] = []
    fold_matches = 0
    latencies_ms: List[float] = []

    preset_results: Dict[str, Dict[str, Any]] = {}
    preset_accs = {"P00698", "P42212", "P00918", "P02185", "P37957", "P02769"}

    t_start = time.perf_counter()
    for idx, rec in enumerate(test_records):
        seq = rec["sequence"]
        obs_ss = rec["ss3"]
        L = len(seq)
        dom = rec["domain"]
        strat = rec["stratum"]

        t0 = time.perf_counter()
        out = pred_fn(seq)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)

        if isinstance(out, dict):
            pred_ss = out["ss_string"]
            pred_h_pct = out["helix_pct"]
            pred_e_pct = out["strand_pct"]
            pred_c_pct = out["coil_pct"]
            pred_fold = out["fold_class"]
            h_cnt = out["helix_count"]
            e_cnt = out["strand_count"]
        else:
            pred_ss = out
            pred_h_pct = round(100.0 * pred_ss.count("H") / L, 2)
            pred_e_pct = round(100.0 * pred_ss.count("E") / L, 2)
            pred_c_pct = round(100.0 * pred_ss.count("C") / L, 2)
            pred_fold, _ = classify_fold_class(pred_h_pct, pred_e_pct, pred_c_pct)
            h_cnt = 0
            e_cnt = 0

        _update_conf_matrix(conf_all, obs_ss, pred_ss)
        conf_by_domain.setdefault(dom, np.zeros((3, 3), dtype=np.int64))
        _update_conf_matrix(conf_by_domain[dom], obs_ss, pred_ss)
        conf_by_stratum.setdefault(strat, np.zeros((3, 3), dtype=np.int64))
        _update_conf_matrix(conf_by_stratum[strat], obs_ss, pred_ss)

        q3_p = compute_q3(obs_ss, pred_ss)
        sov_p = compute_sov99(obs_ss, pred_ss)
        q3_list.append(q3_p)
        sov_list.append(sov_p)
        sov_weighted_num += sov_p * L
        sov_weighted_den += L

        dom_q3.setdefault(dom, []).append(q3_p)
        dom_sov.setdefault(dom, []).append(sov_p)
        str_q3.setdefault(strat, []).append(q3_p)
        str_sov.setdefault(strat, []).append(sov_p)

        obs_h_pct = 100.0 * obs_ss.count("H") / L
        obs_e_pct = 100.0 * obs_ss.count("E") / L
        obs_c_pct = 100.0 * obs_ss.count("C") / L
        obs_fold, _ = classify_fold_class(obs_h_pct, obs_e_pct, obs_c_pct)

        err_h.append(abs(pred_h_pct - obs_h_pct))
        err_e.append(abs(pred_e_pct - obs_e_pct))
        err_c.append(abs(pred_c_pct - obs_c_pct))
        if pred_fold == obs_fold:
            fold_matches += 1

        if rec["accession"] in preset_accs:
            preset_results[rec["accession"]] = {
                "accession": rec["accession"],
                "protein_name": rec["protein_name"],
                "organism": rec["organism"],
                "stratum": strat,
                "length": L,
                "q3_pct": round(q3_p, 2),
                "sov99_pct": round(sov_p, 2),
                "obs_helix_pct": round(obs_h_pct, 2),
                "obs_strand_pct": round(obs_e_pct, 2),
                "obs_coil_pct": round(obs_c_pct, 2),
                "obs_fold_class": obs_fold,
                "pred_helix_pct": round(pred_h_pct, 2),
                "pred_strand_pct": round(pred_e_pct, 2),
                "pred_coil_pct": round(pred_c_pct, 2),
                "pred_fold_class": pred_fold,
                "pred_helix_count": h_cnt,
                "pred_strand_count": e_cnt,
            }

        if (idx + 1) % 3000 == 0:
            print(f"  Processed {idx + 1:,}/{len(test_records):,} proteins... (mean latency {np.mean(latencies_ms):.2f} ms)")

    wall_s = time.perf_counter() - t_start
    summary = compute_confusion_and_mcc(conf_all)
    summary["method"] = method_name
    summary["n_test_proteins"] = len(test_records)
    summary["n_test_residues"] = int(conf_all.sum())
    summary["mean_protein_q3_pct"] = round(float(np.mean(q3_list)), 2)
    summary["std_protein_q3_pct"] = round(float(np.std(q3_list)), 2)
    summary["median_protein_q3_pct"] = round(float(np.median(q3_list)), 2)
    summary["mean_protein_sov99_pct"] = round(float(np.mean(sov_list)), 2)
    summary["std_protein_sov99_pct"] = round(float(np.std(sov_list)), 2)
    summary["residue_weighted_sov99_pct"] = round(sov_weighted_num / max(1.0, sov_weighted_den), 2)
    summary["composition_mae"] = {
        "helix_pct_mae": round(float(np.mean(err_h)), 2),
        "strand_pct_mae": round(float(np.mean(err_e)), 2),
        "coil_pct_mae": round(float(np.mean(err_c)), 2),
        "overall_mean_mae": round(float(np.mean(err_h) + np.mean(err_e) + np.mean(err_c)) / 3.0, 2),
    }
    summary["fold_class_accuracy_pct"] = round(100.0 * fold_matches / len(test_records), 2)
    summary["mean_inference_ms"] = round(float(np.mean(latencies_ms)), 3)
    summary["median_inference_ms"] = round(float(np.median(latencies_ms)), 3)
    summary["total_eval_seconds"] = round(wall_s, 2)

    by_domain: Dict[str, Any] = {}
    for dom in ("Bacteria", "Archaea", "Eukaryota", "Viruses"):
        if dom in conf_by_domain:
            d_stat = compute_confusion_and_mcc(conf_by_domain[dom])
            d_stat["n_proteins"] = len(dom_q3[dom])
            d_stat["n_residues"] = int(conf_by_domain[dom].sum())
            d_stat["mean_protein_q3_pct"] = round(float(np.mean(dom_q3[dom])), 2)
            d_stat["mean_protein_sov99_pct"] = round(float(np.mean(dom_sov[dom])), 2)
            by_domain[dom] = d_stat

    by_stratum: Dict[str, Any] = {}
    for strat in ("Bacteria", "Archaea", "Animals", "Plants", "Fungi", "Protists", "Viruses"):
        if strat in conf_by_stratum:
            s_stat = compute_confusion_and_mcc(conf_by_stratum[strat])
            s_stat["n_proteins"] = len(str_q3[strat])
            s_stat["n_residues"] = int(conf_by_stratum[strat].sum())
            s_stat["mean_protein_q3_pct"] = round(float(np.mean(str_q3[strat])), 2)
            s_stat["mean_protein_sov99_pct"] = round(float(np.mean(str_sov[strat])), 2)
            by_stratum[strat] = s_stat

    summary["by_domain"] = by_domain
    summary["by_stratum"] = by_stratum
    summary["uniprot_test_presets"] = preset_results

    print(
        f"  -> {method_name}: Pooled Q3={summary['overall_q3_pct']:.2f}% | "
        f"Mean Prot Q3={summary['mean_protein_q3_pct']:.2f}% | "
        f"SOV_99={summary['mean_protein_sov99_pct']:.2f}% | "
        f"MCC(H/E/C)=({summary['MCC_H']:.3f}, {summary['MCC_E']:.3f}, {summary['MCC_C']:.3f}) | "
        f"Mean Latency={summary['mean_inference_ms']:.2f} ms"
    )
    return summary


def main() -> int:
    reuse_stage1 = ("--reuse-stage1" in sys.argv)
    corpus_path = os.path.join(ROOT_DIR, "data", "ss_10k_corpus.npz")
    weights_path = os.path.join(ROOT_DIR, "data", "ss_model_weights.npz")
    results_path = os.path.join(ROOT_DIR, "data", "ss_10k_benchmark_results.json")

    print(f"Loading corpus from {corpus_path}...")
    with np.load(corpus_path, allow_pickle=False) as d:
        accessions = d["accessions"]
        protein_names = d["protein_names"]
        organisms = d["organisms"]
        domains = d["domains"]
        strata = d["strata"]
        sequences = d["sequences"]
        ss3_strings = d["ss3_strings"]
        splits = d["splits"]
        is_preset = d["is_classroom_preset"]

    train_mask = (splits == "train")
    test_mask = (splits == "test")

    train_accs = set(accessions[train_mask].tolist())
    test_accs = set(accessions[test_mask].tolist())
    train_seqs_list = [str(s) for s in sequences[train_mask]]
    test_seqs_list = [str(s) for s in sequences[test_mask]]

    # Strict Zero-Leakage Assertions
    assert len(train_accs & test_accs) == 0, "FATAL: Accession overlap between train and test!"
    assert len(set(train_seqs_list) & set(test_seqs_list)) == 0, "FATAL: Sequence overlap between train and test!"
    assert int(is_preset[train_mask].sum()) == 0, "FATAL: Classroom preset found in train split!"
    assert int(test_mask.sum()) >= 10500, f"Expected >=10500 test proteins, got {int(test_mask.sum())}"

    train_ss3_list = [str(s) for s in ss3_strings[train_mask]]
    train_strata_list = [str(s) for s in strata[train_mask]]

    # Train ONLY on `splits == "train"`
    existing_s1 = load_ss_model_weights(weights_path, force_reload=True) if (reuse_stage1 and os.path.exists(weights_path)) else None
    t_train_start = time.perf_counter()
    weights_dict = train_cascaded_ss_engine(train_seqs_list, train_ss3_list, train_strata_list, existing_stage1_bundle=existing_s1)
    train_seconds = time.perf_counter() - t_train_start

    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    np.savez_compressed(weights_path, **weights_dict)
    weights_mb = os.path.getsize(weights_path) / (1024.0 * 1024.0)
    print(f"\nSaved folded pure-NumPy weights to {weights_path} ({weights_mb:.2f} MB) in {train_seconds:.1f}s.")

    # Reload from disk via production API so benchmark tests exact runtime code path
    loaded_bundle = load_ss_model_weights(weights_path, force_reload=True)

    # Assemble held-out test records
    test_indices = np.flatnonzero(test_mask)
    test_records: List[Dict[str, Any]] = []
    for idx in test_indices:
        test_records.append({
            "accession": str(accessions[idx]),
            "protein_name": str(protein_names[idx]),
            "organism": str(organisms[idx]),
            "domain": str(domains[idx]),
            "stratum": str(strata[idx]),
            "sequence": str(sequences[idx]),
            "ss3": str(ss3_strings[idx]),
        })

    # Warmup production predictor once
    _ = predict_secondary_structure(test_records[0]["sequence"])

    # Evaluate ProPurify Cascaded SS + GOR IV + Chou-Fasman on identical 11,118 held-out test proteins
    propurify_eval = evaluate_method_on_test(
        "ProPurify Cascaded SS (2-Stage Neural Ensemble + Viterbi)",
        test_records,
        predict_secondary_structure,
    )
    gor4_eval = evaluate_method_on_test(
        "GOR IV Baseline (Garnier-Gibrat-Robson 1996)",
        test_records,
        lambda sq: predict_gor4(sq, loaded_bundle)[0],
    )
    cf_eval = evaluate_method_on_test(
        "Chou-Fasman Baseline (1978 Empirical Nucleation)",
        test_records,
        predict_chou_fasman,
    )

    # Evaluate exact ProPurify Studio UI Classroom Preset mature chains (`CLASSROOM_PRESET_PROTEINS`)
    print("\n=== [CLASSROOM PRESETS] Verifying Textbook Folds on Mature Preset Chains ===")
    classroom_mature_checks: Dict[str, Any] = {}
    for key, info in CLASSROOM_PRESET_PROTEINS.items():
        res_p = predict_secondary_structure(info["sequence"])
        classroom_mature_checks[key] = {
            "preset_key": key,
            "name": info["name"],
            "uniprot_id": info["uniprot_id"],
            "length": res_p["length"],
            "helix_pct": res_p["helix_pct"],
            "strand_pct": res_p["strand_pct"],
            "coil_pct": res_p["coil_pct"],
            "helix_count": res_p["helix_count"],
            "strand_count": res_p["strand_count"],
            "fold_class": res_p["fold_class"],
            "mean_confidence": res_p["mean_confidence"],
        }
        print(
            f"  {key:20s} ({info['uniprot_id']}, {res_p['length']:3d} aa): "
            f"H={res_p['helix_pct']:5.1f}% ({res_p['helix_count']:2d} α) | "
            f"E={res_p['strand_pct']:5.1f}% ({res_p['strand_count']:2d} β) | "
            f"C={res_p['coil_pct']:5.1f}% -> {res_p['fold_class']}"
        )

    final_report = {
        "benchmark_name": "ProPurify Studio >=10,500-Protein All-Domains-of-Life Secondary Structure Benchmark",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_file": "data/ss_10k_corpus.npz",
        "weights_file": "data/ss_model_weights.npz",
        "weights_size_mb": round(weights_mb, 3),
        "train_count": len(train_seqs_list),
        "train_residues": int(sum(len(s) for s in train_seqs_list)),
        "test_count": len(test_records),
        "total_test_residues": propurify_eval["n_test_residues"],
        # Convenient top-level primary metrics
        "Q3": propurify_eval["overall_q3_pct"],
        "overall_q3_pct": propurify_eval["overall_q3_pct"],
        "mean_protein_q3_pct": propurify_eval["mean_protein_q3_pct"],
        "SOV_99": propurify_eval["mean_protein_sov99_pct"],
        "mean_protein_sov99_pct": propurify_eval["mean_protein_sov99_pct"],
        "residue_weighted_sov99_pct": propurify_eval["residue_weighted_sov99_pct"],
        "MCC_H": propurify_eval["MCC_H"],
        "MCC_E": propurify_eval["MCC_E"],
        "MCC_C": propurify_eval["MCC_C"],
        "mean_mcc": propurify_eval["mean_mcc"],
        "per_state_metrics": propurify_eval["per_state"],
        "composition_mae": propurify_eval["composition_mae"],
        "fold_class_accuracy_pct": propurify_eval["fold_class_accuracy_pct"],
        "mean_inference_ms": propurify_eval["mean_inference_ms"],
        "median_inference_ms": propurify_eval["median_inference_ms"],
        "accuracy_gains_vs_baselines": {
            "q3_gain_over_chou_fasman_pct": round(propurify_eval["overall_q3_pct"] - cf_eval["overall_q3_pct"], 2),
            "q3_gain_over_gor4_pct": round(propurify_eval["overall_q3_pct"] - gor4_eval["overall_q3_pct"], 2),
            "sov99_gain_over_chou_fasman_pct": round(propurify_eval["mean_protein_sov99_pct"] - cf_eval["mean_protein_sov99_pct"], 2),
            "sov99_gain_over_gor4_pct": round(propurify_eval["mean_protein_sov99_pct"] - gor4_eval["mean_protein_sov99_pct"], 2),
        },
        "by_domain": propurify_eval["by_domain"],
        "by_stratum": propurify_eval["by_stratum"],
        "classroom_presets_uniprot_test": propurify_eval["uniprot_test_presets"],
        "classroom_presets_mature_ui": classroom_mature_checks,
        "models": {
            "propurify_ss": propurify_eval,
            "gor4_baseline": gor4_eval,
            "chou_fasman_baseline": cf_eval,
        },
        "zero_leakage_verification": {
            "strict_train_only_fitting": True,
            "train_test_accession_overlap": 0,
            "train_test_sequence_overlap": 0,
            "classroom_presets_in_train": 0,
            "internal_train_splits": {
                "s1_train_fraction": 0.76,
                "s2_train_fraction": 0.14,
                "cal_val_fraction": 0.10,
                "oof_gor4_folds": 5,
            },
        },
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    print(f"\nSaved complete benchmark results to {results_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
