"""
Cascaded Pure-NumPy Protein Secondary Structure Prediction & Topology Diagram Engine
====================================================================================
Provides fast (< 20 ms/protein), deterministic 3-state (H = alpha-helix, E = beta-strand,
C = coil/turn/loop) secondary structure prediction, structural segment annotation,
classic Chou-Fasman (1978) & GOR IV (1996) baselines, Zemla et al. (1999) SOV_99 segment
overlap evaluation, and interactive 2-panel PDBsum-style Plotly topology + propensity diagrams.

Architecture (Cascaded Sequence-to-Structure + Structure-to-Structure + Constrained Viterbi):
  1. Rich Residue & Context Feature Extractor:
     - 20-AA one-hot + terminal boundary indicator (21 dims)
     - Unit-normalized NCBI BLOSUM62 20-D evolutionary substitution profile (20 dims)
     - 12 Physicochemical & thermodynamic conformational scales: Chou-Fasman (P_alpha, P_beta,
       P_turn), Pace-Scholtz thermodynamic helix propensity, Aurora-Rose N-cap preference,
       Kyte-Doolittle hydropathy, Eisenberg consensus hydrophobicity, net charge (pH 7),
       Zimmerman polarity, normalized van der Waals volume, Pro/Gly & beta-branched indicators
     - 7 Vectorized Eisenberg amphipathic moments: alpha-helix (100 deg, w=7 & w=11),
       3_10-helix (120 deg, w=7), beta-strand alternating moment (180 deg, w=5, 7, 9 on
       hydrophobicity + w=7 on polarity)
     - 5 Local H-bond / ridge / salt-bridge / Proline-obstruction autocorrelations (i,i+2 / i,i+3 / i,i+4)
     - 12 GOR-IV directional information-theoretic features (single, local-pair, total 136-pair
       log-odds & calibrated GOR-IV probabilities learned strictly on the training corpus)
     - 4 Exponential N- and C-terminal cap proximity decays
     - 31 Global fold-context features (20-AA composition, log length, relative position,
       protein-wide GOR-IV propensity mean/dispersion, and mean amphipathic moments)
  2. Stage-1 Diverse Neural Ensemble (Pure float32 NumPy SGEMM with Folded BatchNorm):
     - Model 1: Window-17 Wide HardSwish MLP (w=17 sliding window + global fold context)
     - Model 2: Window-17 Residual Highway LeakyReLU MLP (class-balanced focal objective)
     - Model 3: 4-Block Multi-Scale Dilated 1D Residual ConvNet (dilations 1, 2, 4, 1;
       31-residue effective receptive field + global context gating)
  3. Stage-2 Structure-to-Structure Context Network (PSIPRED / PHD paradigm):
     - 15-residue sliding window over Stage-1 ensemble & per-model probabilities, Shannon
       entropy, GOR-IV probabilities, and Pro/Gly/turn boundary cues + global AA composition
       -> calibrated per-residue probabilities (P_H, P_E, P_C).
  4. Exact 7-State Minimum-Length Constrained Viterbi Decoder:
     - Enforces biophysical hydrogen-bond geometry constraints globally (alpha-helix >= 4 aa,
       beta-strand >= 2 aa) via exact dynamic programming.

References:
  - Chou, P. Y. & Fasman, G. D. (1978) Adv. Enzymol. Relat. Areas Mol. Biol. 47:45-148
  - Garnier, J., Gibrat, J.-F. & Robson, B. (1996) Methods Enzymol. 266:540-553 (GOR IV)
  - Eisenberg, D., Weiss, R. M. & Terwilliger, T. C. (1984) PNAS 81:140-144 (Hydrophobic moment)
  - Rost, B. & Sander, C. (1993) J. Mol. Biol. 232:584-599 (PHD cascaded neural architecture)
  - Jones, D. T. (1999) J. Mol. Biol. 292:195-202 (PSIPRED two-stage neural network)
  - Zemla, A., Venclovas, C., Fidelis, K. & Rost, B. (1999) Proteins 34:220-223 (SOV_99)
"""

from concurrent.futures import ThreadPoolExecutor
import ctypes
import glob
import math
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Domain-scale overlap-tile constants for giant multi-domain proteins (L > 1,500)
MAX_SINGLE_DOMAIN_LENGTH = 1500
MEGA_TILE_WINDOW = 800
MEGA_TILE_STRIDE = 600
MEGA_EFFECTIVE_DOMAIN_LEN = 300.0
_MEGA_POOL_LOCK = threading.Lock()
_MEGA_TILE_POOL: Optional[ThreadPoolExecutor] = None
_OB_INIT_DONE = False
_OB_SET_THREADS: Optional[Any] = None
_OB_GET_THREADS: Optional[Any] = None


def _get_mega_tile_pool(n_workers: int = 6) -> ThreadPoolExecutor:
    """Returns a lazy shared ThreadPoolExecutor for parallel L2-resident domain-tile inference."""
    global _MEGA_TILE_POOL, _OB_INIT_DONE, _OB_SET_THREADS, _OB_GET_THREADS
    with _MEGA_POOL_LOCK:
        if not _OB_INIT_DONE:
            _OB_INIT_DONE = True
            try:
                np_dir = os.path.dirname(np.__file__)
                for cand_dir in (os.path.join(os.path.dirname(np_dir), "numpy.libs"), np_dir):
                    if os.path.isdir(cand_dir):
                        for dll_path in glob.glob(os.path.join(cand_dir, "*openblas*.dll")):
                            lib = ctypes.CDLL(dll_path)
                            _OB_SET_THREADS = getattr(lib, "openblas_set_num_threads", getattr(lib, "scipy_openblas_set_num_threads", None))
                            _OB_GET_THREADS = getattr(lib, "openblas_get_num_threads", getattr(lib, "scipy_openblas_get_num_threads", None))
                            if _OB_SET_THREADS is not None:
                                break
            except Exception:
                _OB_SET_THREADS = None
                _OB_GET_THREADS = None
        if _MEGA_TILE_POOL is None:
            _MEGA_TILE_POOL = ThreadPoolExecutor(max_workers=n_workers)
        return _MEGA_TILE_POOL

# ==============================================================================
# 1. AMINO ACID ALPHABET, BLOSUM62 & BIOPHYSICAL SCALES
# ==============================================================================

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX: Dict[str, int] = {aa: idx for idx, aa in enumerate(AA_ORDER)}
# Map rare/ambiguous IUPAC letters to closest biochemical standard residue (X -> 21 neutral)
_RARE_AA_MAP: Dict[str, int] = {
    "U": AA_TO_IDX["C"],  # Selenocysteine -> Cys
    "O": AA_TO_IDX["K"],  # Pyrrolysine -> Lys
    "B": AA_TO_IDX["D"],  # Asx -> Asp
    "Z": AA_TO_IDX["E"],  # Glx -> Glu
    "J": AA_TO_IDX["L"],  # Xle -> Leu
    "X": 21,              # Unknown -> neutral average
}
PAD_IDX = 20  # Terminal sequence boundary token in GOR-IV & window features
UNK_IDX = 21

# Fast ASCII lookup table (256 entries) mapping uppercase ASCII bytes -> 0..21 (-1 for invalid)
_ASCII_TO_IDX = np.full(256, -1, dtype=np.int16)
for _aa, _idx in AA_TO_IDX.items():
    _ASCII_TO_IDX[ord(_aa)] = _idx
    _ASCII_TO_IDX[ord(_aa.lower())] = _idx
for _aa, _idx in _RARE_AA_MAP.items():
    _ASCII_TO_IDX[ord(_aa)] = _idx
    _ASCII_TO_IDX[ord(_aa.lower())] = _idx

SS_STATES = ("H", "E", "C")
STATE_TO_IDX = {"H": 0, "E": 1, "C": 2}

# Design-system typography & WCAG-verified palette constants matching ProPurify Studio
PLOTLY_FONT_FAMILY = "IBM Plex Sans, Helvetica Neue, Arial, sans-serif"
PLOTLY_MONO_FAMILY = "IBM Plex Mono, monospace"
PLOTLY_TITLE_FONT = dict(family=PLOTLY_FONT_FAMILY, size=14, color="#111827")
PLOTLY_AXIS_TITLE_FONT = dict(family=PLOTLY_FONT_FAMILY, size=12.5, color="#2C3A47")
PLOTLY_TICK_FONT = dict(family=PLOTLY_FONT_FAMILY, size=11.5, color="#52606D")
PLOTLY_GRID_COLOR = "#E6ECEF"

# Standard NCBI BLOSUM62 substitution matrix ordered alphabetically: A C D E F G H I K L M N P Q R S T V W Y
_BLOSUM62_RAW = np.array([
    #  A   C   D   E   F   G   H   I   K   L   M   N   P   Q   R   S   T   V   W   Y
    [  4,  0, -2, -1, -2,  0, -2, -1, -1, -1, -1, -2, -1, -1, -1,  1,  0,  0, -3, -2],  # A
    [  0,  9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2],  # C
    [ -2, -3,  6,  2, -3, -1, -1, -3, -1, -4, -3,  1, -1,  0, -2,  0, -1, -3, -4, -3],  # D
    [ -1, -4,  2,  5, -3, -2,  0, -3,  1, -3, -2,  0, -1,  2,  0,  0, -1, -2, -3, -2],  # E
    [ -2, -2, -3, -3,  6, -3, -1,  0, -3,  0,  0, -3, -4, -3, -3, -2, -2, -1,  1,  3],  # F
    [  0, -3, -1, -2, -3,  6, -2, -4, -2, -4, -3,  0, -2, -2, -2,  0, -2, -3, -2, -3],  # G
    [ -2, -3, -1,  0, -1, -2,  8, -3, -1, -3, -2,  1, -2,  0,  0, -1, -2, -3, -2,  2],  # H
    [ -1, -1, -3, -3,  0, -4, -3,  4, -3,  2,  1, -3, -3, -3, -3, -2, -1,  3, -3, -1],  # I
    [ -1, -3, -1,  1, -3, -2, -1, -3,  5, -2, -1,  0, -1,  1,  2,  0, -1, -2, -3, -2],  # K
    [ -1, -1, -4, -3,  0, -4, -3,  2, -2,  4,  2, -3, -3, -2, -2, -2, -1,  1, -2, -1],  # L
    [ -1, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5, -2, -2,  0, -1, -1, -1,  1, -1, -1],  # M
    [ -2, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6, -2,  0,  0,  1,  0, -3, -4, -2],  # N
    [ -1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7, -1, -2, -1, -1, -2, -4, -3],  # P
    [ -1, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,  1,  0, -1, -2, -2, -1],  # Q
    [ -1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5, -1, -1, -3, -3, -2],  # R
    [  1, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,  1, -2, -3, -2],  # S
    [  0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,  0, -2, -2],  # T
    [  0, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4, -3, -1],  # V
    [ -3, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11,  2],  # W
    [ -2, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2,  7],  # Y
], dtype=np.float32)

# Unit L2-normalized BLOSUM62 rows (22 x 20: 0..19 AAs, 20=PAD zeros, 21=UNK mean)
_BLOSUM62_NORM = np.zeros((22, 20), dtype=np.float32)
_BLOSUM62_NORM[:20] = _BLOSUM62_RAW / np.linalg.norm(_BLOSUM62_RAW, axis=1, keepdims=True)
_BLOSUM62_NORM[UNK_IDX] = _BLOSUM62_NORM[:20].mean(axis=0)

# Chou & Fasman (1978) conformational parameters (ordered A C D E F G H I K L M N P Q R S T V W Y)
CF_P_ALPHA = np.array([
    1.42, 0.70, 1.01, 1.51, 1.13, 0.57, 1.00, 1.08, 1.16, 1.21,
    1.45, 0.67, 0.57, 1.11, 0.98, 0.77, 0.83, 1.06, 1.08, 0.69
], dtype=np.float32)

CF_P_BETA = np.array([
    0.83, 1.19, 0.54, 0.37, 1.38, 0.75, 0.87, 1.60, 0.74, 1.30,
    1.05, 0.89, 0.55, 1.10, 0.93, 0.75, 1.19, 1.70, 1.37, 1.47
], dtype=np.float32)

CF_P_TURN = np.array([
    0.66, 1.19, 1.46, 0.74, 0.60, 1.56, 0.95, 0.47, 1.01, 0.59,
    0.60, 1.56, 1.52, 0.98, 0.95, 1.43, 0.96, 0.50, 0.96, 1.14
], dtype=np.float32)

# Chou & Fasman (1978) positional beta-turn frequencies (f_i, f_{i+1}, f_{i+2}, f_{i+3})
CF_TURN_F = np.array([
    # f_i    f_i+1  f_i+2  f_i+3
    [0.060, 0.076, 0.035, 0.058],  # A
    [0.149, 0.050, 0.117, 0.128],  # C
    [0.147, 0.110, 0.179, 0.081],  # D
    [0.056, 0.060, 0.077, 0.064],  # E
    [0.059, 0.041, 0.065, 0.065],  # F
    [0.102, 0.085, 0.190, 0.152],  # G
    [0.140, 0.047, 0.093, 0.054],  # H
    [0.043, 0.034, 0.013, 0.056],  # I
    [0.055, 0.115, 0.072, 0.095],  # K
    [0.061, 0.025, 0.036, 0.070],  # L
    [0.068, 0.082, 0.014, 0.055],  # M
    [0.161, 0.083, 0.191, 0.091],  # N
    [0.102, 0.301, 0.034, 0.068],  # P
    [0.074, 0.098, 0.037, 0.098],  # Q
    [0.070, 0.106, 0.099, 0.085],  # R
    [0.120, 0.139, 0.125, 0.106],  # S
    [0.086, 0.108, 0.065, 0.079],  # T
    [0.062, 0.048, 0.028, 0.053],  # V
    [0.077, 0.013, 0.064, 0.167],  # W
    [0.082, 0.065, 0.114, 0.125],  # Y
], dtype=np.float32)

# Pace & Scholtz (1998) experimental thermodynamic helix propensity (higher = stronger helix former)
PACE_HELIX = np.array([
    1.00, 0.32, 0.31, 0.60, 0.46, 0.00, 0.39, 0.59, 0.74, 0.79,
    0.76, 0.35, -2.16, 0.61, 0.79, 0.50, 0.34, 0.39, 0.51, 0.47
], dtype=np.float32)

# Aurora & Rose / Richardson N-cap helix-initiation propensity
NCAP_PROP = np.array([
    0.78, 0.95, 1.62, 0.88, 0.65, 1.15, 1.02, 0.54, 0.75, 0.60,
    0.64, 1.65, 1.10, 0.82, 0.74, 1.48, 1.35, 0.56, 0.68, 0.72
], dtype=np.float32)

# Kyte & Doolittle (1982) hydropathy scaled to [-1, +1]
KD_HYDRO = np.array([
    1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
    1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3
], dtype=np.float32) / 4.5

# Eisenberg et al. (1984) consensus hydrophobicity for amphipathic moment calculation
EISENBERG_H = np.array([
    0.62, 0.29, -0.90, -0.74, 1.19, 0.48, -0.40, 1.38, -1.50, 1.06,
    0.64, -0.78, 0.12, -0.85, -2.53, -0.18, -0.05, 1.08, 0.81, 0.26
], dtype=np.float32)

# Net side-chain charge at pH 7.0
CHARGE_PH7 = np.array([
    0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
], dtype=np.float32)

# Grantham / Zimmerman polarity normalized to [0, 1]
POLARITY_NORM = np.array([
    0.16, 0.05, 1.00, 0.96, 0.03, 0.24, 0.65, 0.03, 0.85, 0.01,
    0.04, 0.80, 0.16, 0.68, 0.86, 0.34, 0.30, 0.05, 0.04, 0.12
], dtype=np.float32)

# Normalized amino acid side-chain van der Waals volume [0, 1] (Gly=0.0 -> Trp=1.0)
VOLUME_NORM = np.array([
    0.17, 0.29, 0.31, 0.47, 0.77, 0.00, 0.56, 0.63, 0.64, 0.63,
    0.61, 0.34, 0.32, 0.50, 0.68, 0.18, 0.34, 0.47, 1.00, 0.80
], dtype=np.float32)

# Explicit structural indicator flags (Proline breaker/N1-cap, Glycine flexibility, beta-branched V/I/T)
IS_PRO = np.array([1.0 if aa == "P" else 0.0 for aa in AA_ORDER], dtype=np.float32)
IS_GLY = np.array([1.0 if aa == "G" else 0.0 for aa in AA_ORDER], dtype=np.float32)
IS_BETA_BRANCHED = np.array([1.0 if aa in ("V", "I", "T") else 0.0 for aa in AA_ORDER], dtype=np.float32)

# Combined 22 x 12 static biophysical lookup table (rows 0..19 AAs, row 20 PAD=0, row 21 UNK=mean)
_PHYSICO_12 = np.zeros((22, 12), dtype=np.float32)
_PHYSICO_12[:20] = np.column_stack([
    CF_P_ALPHA,
    CF_P_BETA,
    CF_P_TURN,
    PACE_HELIX,
    NCAP_PROP,
    KD_HYDRO,
    CHARGE_PH7,
    POLARITY_NORM,
    VOLUME_NORM,
    IS_PRO,
    IS_GLY,
    IS_BETA_BRANCHED,
])
_PHYSICO_12[UNK_IDX] = _PHYSICO_12[:20].mean(axis=0)

_EISENBERG_LOOKUP = np.zeros(22, dtype=np.float32)
_EISENBERG_LOOKUP[:20] = EISENBERG_H
_EISENBERG_LOOKUP[UNK_IDX] = float(EISENBERG_H.mean())

_POLARITY_LOOKUP = np.zeros(22, dtype=np.float32)
_POLARITY_LOOKUP[:20] = POLARITY_NORM
_POLARITY_LOOKUP[UNK_IDX] = float(POLARITY_NORM.mean())

_CHARGE_LOOKUP = np.zeros(22, dtype=np.float32)
_CHARGE_LOOKUP[:20] = CHARGE_PH7

_ONEHOT_21 = np.zeros((22, 21), dtype=np.float32)
_ONEHOT_21[:20, :20] = np.eye(20, dtype=np.float32)
_ONEHOT_21[PAD_IDX, 20] = 1.0  # Explicit boundary channel
_ONEHOT_21[UNK_IDX, :20] = 1.0 / 20.0

# Precompute pair indices for 17-residue GOR-IV window: 136 pairs (m < n in 0..16)
_GOR_M_IDX, _GOR_N_IDX = np.triu_indices(17, k=1)
_GOR_PAIR_ARANGE = np.arange(136, dtype=np.int32)[None, :]
_GOR_FLAT_OFFSETS = (np.arange(136, dtype=np.int32) * 441)[None, :]
_GOR_LOCAL_PAIR_MASK = (_GOR_N_IDX - _GOR_M_IDX) <= 4  # Short-range H-bond & beta-pleat pairs

# Lazy cached runtime model bundle
_DEFAULT_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "ss_model_weights.npz",
)
_CACHED_MODEL_BUNDLE: Optional[Dict[str, np.ndarray]] = None
_CACHED_MODEL_PATH: Optional[str] = None


# ==============================================================================
# 2. SEQUENCE SANITIZATION & ENCODING
# ==============================================================================

def clean_protein_sequence_for_ss(sequence: str) -> str:
    """
    Cleans raw or FASTA-formatted amino acid sequence, stripping headers, digits,
    and whitespace while preserving valid IUPAC amino acid characters in uppercase.
    """
    if not isinstance(sequence, str):
        raise ValueError("Sequence must be a string.")
    lines = []
    for raw_line in sequence.splitlines():
        s = raw_line.strip()
        if not s or s.startswith(">") or s.startswith(";"):
            continue
        lines.append(s)
    joined = "".join(lines).upper()
    cleaned_chars = [ch for ch in joined if ord(ch) < 256 and _ASCII_TO_IDX[ord(ch)] >= 0]
    cleaned = "".join(cleaned_chars)
    if not cleaned:
        raise ValueError("Protein sequence must contain at least 1 valid amino acid residue.")
    return cleaned


def encode_sequence_indices(sequence: str) -> np.ndarray:
    """Maps a cleaned uppercase sequence of length L to integer indices in 0..21."""
    ascii_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    return _ASCII_TO_IDX[ascii_bytes].astype(np.int32)


# ==============================================================================
# 3. VECTORIZED AMPHIPATHIC MOMENTS & GOR-IV DIRECTIONAL INFORMATION
# ==============================================================================

def _compute_all_amphipathic_moments(eis_h: np.ndarray, pol_v: np.ndarray) -> np.ndarray:
    """
    Computes all 7 DC-subtracted Eisenberg amphipathic moments in vectorized O(L) passes:
      0: alpha-helix (100 deg, w=7)
      1: alpha-helix (100 deg, w=11)
      2: 3_10-helix  (120 deg, w=7)
      3: beta-strand (180 deg, w=5)
      4: beta-strand (180 deg, w=7)
      5: beta-strand (180 deg, w=9)
      6: beta-strand polarity (180 deg, w=7)
    Uses float64 phase angles and cumulative sums to prevent float32 phase drift and
    catastrophic cancellation on giant multi-domain proteins (e.g. 34,350-aa Titin).
    """
    L = len(eis_h)
    idx = np.arange(L, dtype=np.float64)
    eis64 = eis_h.astype(np.float64)
    pol64 = pol_v.astype(np.float64)

    th100 = idx * math.radians(100.0)
    c100, s100 = np.cos(th100), np.sin(th100)

    th120 = idx * math.radians(120.0)
    c120, s120 = np.cos(th120), np.sin(th120)

    alt = np.where((np.arange(L) & 1) == 0, 1.0, -1.0)

    # Precompute cumulative sums in float64 with 5-zero padding on each side (supports up to w=11, r=5)
    padded = np.zeros((L + 11, 8), dtype=np.float64)
    padded[6:6 + L, 0] = eis64              # 0: raw h
    padded[6:6 + L, 1] = eis64 * c100       # 1: h cos 100
    padded[6:6 + L, 2] = eis64 * s100       # 2: h sin 100
    padded[6:6 + L, 3] = eis64 * c120       # 3: h cos 120
    padded[6:6 + L, 4] = eis64 * s120       # 4: h sin 120
    padded[6:6 + L, 5] = eis64 * alt        # 5: h * (-1)^j
    padded[6:6 + L, 6] = pol64              # 6: raw pol
    padded[6:6 + L, 7] = pol64 * alt        # 7: pol * (-1)^j

    csum = np.cumsum(padded, axis=0, dtype=np.float64)  # (L + 11, 8)
    # Sum over [i - r .. i + r] for i in 0..L-1 is csum[6 + i + r] - csum[5 + i - r]
    sw5 = csum[8:8 + L] - csum[3:3 + L]
    sw7 = csum[9:9 + L] - csum[2:2 + L]
    sw9 = csum[10:10 + L] - csum[1:1 + L]
    sw11 = csum[11:11 + L] - csum[0:L]

    # Constant DC cosine sums: sum_{k=-r}^{r} cos(k * theta)
    c0_100_7 = -0.532088886237956    # sum_{k=-3..3} cos(k*100 deg)
    c0_100_11 = -0.37860270793112695 # sum_{k=-5..5} cos(k*100 deg)
    c0_120_7 = 1.0                   # sum_{k=-3..3} cos(k*120 deg) == 1.0

    # 100 deg w=7
    hbar7 = sw7[:, 0] * (1.0 / 7.0)
    re7 = (sw7[:, 1] * c100 + sw7[:, 2] * s100) - hbar7 * c0_100_7
    im7 = (sw7[:, 2] * c100 - sw7[:, 1] * s100)
    mu_a7 = np.sqrt(re7 * re7 + im7 * im7) * (1.0 / 7.0)

    # 100 deg w=11
    hbar11 = sw11[:, 0] * (1.0 / 11.0)
    re11 = (sw11[:, 1] * c100 + sw11[:, 2] * s100) - hbar11 * c0_100_11
    im11 = (sw11[:, 2] * c100 - sw11[:, 1] * s100)
    mu_a11 = np.sqrt(re11 * re11 + im11 * im11) * (1.0 / 11.0)

    # 120 deg w=7
    re120 = (sw7[:, 3] * c120 + sw7[:, 4] * s120) - hbar7 * c0_120_7
    im120 = (sw7[:, 4] * c120 - sw7[:, 3] * s120)
    mu_310 = np.sqrt(re120 * re120 + im120 * im120) * (1.0 / 7.0)

    # 180 deg alternating moments: sum_{k=-r..r} (-1)^k = +1 for r even (w=5,9), -1 for r odd (w=7)
    hbar5 = sw5[:, 0] * (1.0 / 5.0)
    hbar9 = sw9[:, 0] * (1.0 / 9.0)
    pbar7 = sw7[:, 6] * (1.0 / 7.0)

    mu_b5 = np.abs(alt * sw5[:, 5] - hbar5) * (1.0 / 5.0)
    mu_b7 = np.abs(alt * sw7[:, 5] + hbar7) * (1.0 / 7.0)
    mu_b9 = np.abs(alt * sw9[:, 5] - hbar9) * (1.0 / 9.0)
    mu_bp7 = np.abs(alt * sw7[:, 7] + pbar7) * (1.0 / 7.0)

    return np.column_stack([mu_a7, mu_a11, mu_310, mu_b5, mu_b7, mu_b9, mu_bp7]).astype(np.float32)


def compute_gor4_features(seq_idx: np.ndarray, gor_params: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes exact Garnier-Gibrat-Robson (1996) GOR-IV 17-residue directional single + 136-pair
    information features for a sequence index array of length L.
     Note: Unknown token 21 is mapped to neutral padding index 20 for GOR table lookup.
    Returns:
      - gor_feats: (L, 12) float32 [I_single(3), I_pair_local(3), L_gor4(3), P_gor4(3)]
      - p_gor4:    (L, 3) float32 calibrated GOR-IV probabilities
    """
    L = len(seq_idx)
    safe_idx = np.where(seq_idx > 20, PAD_IDX, seq_idx)
    padded = np.empty(L + 16, dtype=np.int32)
    padded[:8] = PAD_IDX
    padded[8:8 + L] = safe_idx
    padded[8 + L:] = PAD_IDX

    w_seq = sliding_window_view(padded, 17)  # (L, 17)
    pos_arange = np.arange(17, dtype=np.int32)[None, :]

    # Single-residue directional log-odds: single_table has shape (17, 21, 3)
    single_tab = gor_params["gor_single_table"]
    i_single = single_tab[pos_arange, w_seq].sum(axis=1)  # (L, 3)

    # Pairwise directional log-odds: 1D-flattened (136*441, 3) lookup for maximum cache speed
    if "gor_pair_2d" in gor_params:
        pair_2d = gor_params["gor_pair_2d"]
    else:
        pair_2d = gor_params["gor_pair_flat_table"].reshape(136 * 441, 3)
    pair_ab = w_seq[:, _GOR_M_IDX] * 21 + w_seq[:, _GOR_N_IDX] + _GOR_FLAT_OFFSETS  # (L, 136)
    pair_vals = pair_2d[pair_ab]  # (L, 136, 3)

    i_pair_all = pair_vals.sum(axis=1)  # (L, 3)
    i_pair_local = pair_vals[:, _GOR_LOCAL_PAIR_MASK, :].sum(axis=1)  # (L, 3)

    # Exact Garnier-Gibrat-Robson (1996) Eq. 8-10 combination:
    # (2/17) * sum_{m<n} dI_pair - (15/17) * sum_m dI_single + prior_log_odds
    prior_lo = gor_params["gor_prior_log_odds"]  # (3,)
    l_gor4 = (2.0 / 17.0) * i_pair_all - (15.0 / 17.0) * i_single + prior_lo[None, :]

    # Calibrated probabilities via temperature-scaled softmax
    temp = float(gor_params.get("gor_temperature", np.array(1.0, dtype=np.float32)))
    scaled_logits = l_gor4 * (1.0 / max(0.1, temp))
    scaled_logits = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
    exp_l = np.exp(scaled_logits)
    p_gor4 = (exp_l / exp_l.sum(axis=1, keepdims=True)).astype(np.float32)

    gor_feats = np.hstack([
        (i_single * 0.15).astype(np.float32),
        (i_pair_local * 0.04).astype(np.float32),
        (l_gor4 * 0.25).astype(np.float32),
        p_gor4,
    ])
    return gor_feats, p_gor4


def extract_residue_and_global_features(
    sequence: str,
    gor_params: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts rich per-residue biophysical + evolutionary + amphipathic + GOR-IV features
    and global protein context features for a cleaned protein sequence of length L.
    Returns:
      - x_local:  (L, 81) float32 per-residue feature matrix
      - x_global: (L, 31) float32 per-residue broadcast global context + relative coordinate
      - p_gor4:   (L, 3)  float32 GOR-IV baseline probabilities
      - seq_idx:  (L,)    int32 encoded sequence indices
    """
    seq_idx = encode_sequence_indices(sequence)
    L = len(seq_idx)

    # 1. One-hot (21 dims) + Normalized BLOSUM62 (20 dims) + Static Physicochemical (12 dims) = 53 dims
    oh = _ONEHOT_21[seq_idx]           # (L, 21)
    blosum = _BLOSUM62_NORM[seq_idx]   # (L, 20)
    phys = _PHYSICO_12[seq_idx]        # (L, 12)

    # 2. Vectorized Local Amphipathic Moments (7 dims)
    eis_h = _EISENBERG_LOOKUP[seq_idx]
    pol_v = _POLARITY_LOOKUP[seq_idx]
    moments = _compute_all_amphipathic_moments(eis_h, pol_v)  # (L, 7)
    mu_alpha_11 = moments[:, 1]
    mu_beta_7 = moments[:, 4]

    # 3. Local Pairwise H-bond / Ridge / Salt-Bridge / Proline Obstruction Autocorrelations (5 dims)
    # Pad hydrophobicity, charge, and Proline indicator to compute i,i+k interactions cleanly
    h_pad = np.pad(np.maximum(0.0, eis_h), (4, 4), mode="constant")
    q_pad = np.pad(_CHARGE_LOOKUP[seq_idx], (4, 4), mode="constant")
    pro_pad = np.pad(phys[:, 9], (4, 4), mode="constant")

    # Helical ridge hydrophobic packing (i with i+-3, i+-4)
    hh_34 = 0.25 * (
        h_pad[4:4 + L] * h_pad[7:7 + L] +
        h_pad[4:4 + L] * h_pad[8:8 + L] +
        h_pad[4:4 + L] * h_pad[1:1 + L] +
        h_pad[4:4 + L] * h_pad[0:L]
    )
    # Helical i,i+3 & i,i+4 opposite-charge salt-bridge stabilization (-q_i * q_{i+3,4})
    salt_34 = 0.25 * (
        np.maximum(0.0, -q_pad[4:4 + L] * q_pad[7:7 + L]) +
        np.maximum(0.0, -q_pad[4:4 + L] * q_pad[8:8 + L]) +
        np.maximum(0.0, -q_pad[4:4 + L] * q_pad[1:1 + L]) +
        np.maximum(0.0, -q_pad[4:4 + L] * q_pad[0:L])
    )
    # Beta-sheet same-face i,i+-2 hydrophobic correlation
    eis_pad = np.pad(eis_h, (2, 2), mode="constant")
    strand_hh2 = 0.5 * (eis_pad[2:2 + L] * eis_pad[4:4 + L] + eis_pad[2:2 + L] * eis_pad[0:L])
    # Beta-barrel / surface sheet alternating contrast (-h_i * h_{i+-1})
    strand_alt1 = -0.5 * (eis_pad[2:2 + L] * eis_pad[3:3 + L] + eis_pad[2:2 + L] * eis_pad[1:1 + L])
    # C-ward Proline helix-breaker indicator (Pro at i+1, i+2, or i+3 disrupts i+4->i H-bond)
    pro_cward = (pro_pad[5:5 + L] + pro_pad[6:6 + L] + pro_pad[7:7 + L]) * (1.0 / 3.0)

    autocorr = np.column_stack([hh_34, salt_34, strand_hh2, strand_alt1, pro_cward]).astype(np.float32)

    # 4. GOR-IV Directional Single + Pairwise Log-Odds & Calibrated Probabilities (12 dims)
    gor_feats, p_gor4 = compute_gor4_features(seq_idx, gor_params)

    # 5. Terminal Proximity Features (4 dims)
    pos = np.arange(L, dtype=np.float32)
    rev_pos = (L - 1) - pos
    term_feats = np.column_stack([
        np.exp(-pos / 5.0),
        np.exp(-pos / 15.0),
        np.exp(-rev_pos / 5.0),
        np.exp(-rev_pos / 15.0),
    ]).astype(np.float32)

    # Assemble (L, 81) local per-residue matrix
    x_local = np.hstack([oh, blosum, phys, moments, autocorr, gor_feats, term_feats]).astype(np.float32)

    # 6. Global Protein Fold Context (31 dims: 30 protein-level constants + 1 relative coordinate)
    aa_comp = oh[:, :20].mean(axis=0)  # (20,)
    log_len = math.log(max(1.0, float(L)) / 200.0)
    gor_mean = p_gor4.mean(axis=0)     # (3,)
    gor_std = p_gor4.std(axis=0)       # (3,)
    mean_alpha_mom = float(mu_alpha_11.mean())
    mean_beta_mom = float(mu_beta_7.mean())
    mean_p_turn = float(phys[:, 2].mean())

    glob_vec = np.concatenate([
        aa_comp,
        np.array([log_len], dtype=np.float32),
        gor_mean,
        gor_std,
        np.array([mean_alpha_mom, mean_beta_mom, mean_p_turn], dtype=np.float32),
    ]).astype(np.float32)  # (30,)

    rel_pos = (pos / max(1.0, float(L - 1))).astype(np.float32)[:, None]  # (L, 1)
    x_global = np.hstack([np.broadcast_to(glob_vec[None, :], (L, 30)), rel_pos]).astype(np.float32)

    return x_local, x_global, p_gor4, seq_idx


# ==============================================================================
# 4. PURE-NUMPY NEURAL LAYERS, MULTI-STAGE INFERENCE & VITERBI DECODING
# ==============================================================================

def _hard_swish(x: np.ndarray) -> np.ndarray:
    """Fast SIMD piecewise-polynomial Swish/SiLU: x * clip(x/6 + 0.5, 0, 1)."""
    return x * np.clip(x * np.float32(1.0 / 6.0) + np.float32(0.5), np.float32(0.0), np.float32(1.0))


def _leaky_relu(x: np.ndarray) -> np.ndarray:
    """Fast SIMD LeakyReLU(0.1) without transcendental overhead."""
    return np.maximum(x, x * np.float32(0.1))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)


def _extract_window_17(x_local_norm: np.ndarray, pad_vec: np.ndarray) -> np.ndarray:
    """Builds contiguous (L, 17 * F_local) sliding window matrix using pad_vec for out-of-bounds."""
    L, F = x_local_norm.shape
    padded = np.empty((L + 16, F), dtype=np.float32)
    padded[:8] = pad_vec
    padded[8:8 + L] = x_local_norm
    padded[8 + L:] = pad_vec
    return np.hstack([padded[offset:offset + L] for offset in range(17)])


def _dilated_conv1d_input(h: np.ndarray, kernel_size: int, dilation: int) -> np.ndarray:
    """Builds (L, kernel_size * C) dilated window matrix via fast contiguous slice views."""
    L, C = h.shape
    radius = (kernel_size // 2) * dilation
    padded = np.zeros((L + 2 * radius, C), dtype=np.float32)
    padded[radius:radius + L] = h
    return np.hstack([padded[s:s + L] for s in range(0, kernel_size * dilation, dilation)])


def _forward_stage1_ensemble(
    x_local: np.ndarray,
    x_global: np.ndarray,
    bundle: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Executes all 3 diverse Stage-1 neural architectures in pure float32 NumPy
    (BatchNorm folded into linear weights W, b at export time for < 10 ms latency!):
      - Model 1: Window-17 Wide HardSwish MLP
      - Model 2: Window-17 Residual Highway LeakyReLU MLP
      - Model 3: 4-Block Multi-Scale Dilated 1D Residual ConvNet + Global Squeeze-Excitation
    Returns (p1, p2, p3, p_ens), each of shape (L, 3).
    """
    x_loc_n = (x_local - bundle["loc_mean"]) * bundle["loc_inv_std"]
    x_glob_n = (x_global - bundle["glob_mean"]) * bundle["glob_inv_std"]
    pad_vec = bundle["loc_pad_norm"]

    # Optional learned dense residue bottleneck (81 -> P) before 17-mer windowing for L1/L2 cache speed
    if "s1_W_proj" in bundle:
        L = x_loc_n.shape[0]
        padded_raw = np.empty((L + 16, x_loc_n.shape[1]), dtype=np.float32)
        padded_raw[:8] = pad_vec
        padded_raw[8:8 + L] = x_loc_n
        padded_raw[8 + L:] = pad_vec
        proj_pad = _hard_swish(padded_raw @ bundle["s1_W_proj"] + bundle["s1_b_proj"])
        w17 = np.hstack([proj_pad[offset:offset + L] for offset in range(17)])
    else:
        w17 = _extract_window_17(x_loc_n, pad_vec)
    win_in = np.hstack([w17, x_glob_n])

    # Fused first linear projection for Model 1 + Model 2 in ONE SGEMM call
    if "m12_W1_fused" in bundle:
        z1_fused = win_in @ bundle["m12_W1_fused"] + bundle["m12_b1_fused"]
        d1 = int(bundle["m1_W2"].shape[0])
        z1_m1 = z1_fused[:, :d1]
        z1_m2 = z1_fused[:, d1:]
    else:
        z1_m1 = win_in @ bundle["m1_W1"] + bundle["m1_b1"]
        z1_m2 = win_in @ bundle["m2_W1"] + bundle["m2_b1"]

    # Model 1: Wide HardSwish MLP
    h1_m1 = _hard_swish(z1_m1)
    h2_m1 = _hard_swish(h1_m1 @ bundle["m1_W2"] + bundle["m1_b2"])
    logits_m1 = (h2_m1 @ bundle["m1_W3"] + bundle["m1_b3"]) * np.float32(bundle["m1_inv_temp"])
    p1 = _softmax(logits_m1)

    # Model 2: Residual Highway LeakyReLU MLP
    h1_m2 = _leaky_relu(z1_m2)
    h1_m2 = h1_m2 + _leaky_relu(h1_m2 @ bundle["m2_W_res"] + bundle["m2_b_res"])
    h2_m2 = _leaky_relu(h1_m2 @ bundle["m2_W2"] + bundle["m2_b2"])
    logits_m2 = (h2_m2 @ bundle["m2_W3"] + bundle["m2_b3"]) * np.float32(bundle["m2_inv_temp"])
    p2 = _softmax(logits_m2)

    # Model 3: 4-Block Multi-Scale Dilated 1D Residual ConvNet (dilations 1, 2, 4, 1)
    stem_in = np.hstack([x_loc_n, x_glob_n])
    hc = _hard_swish(stem_in @ bundle["m3_W_stem"] + bundle["m3_b_stem"])

    hc = hc + _hard_swish(_dilated_conv1d_input(hc, 5, 1) @ bundle["m3_Wc1"] + bundle["m3_bc1"])
    hc = hc + _hard_swish(_dilated_conv1d_input(hc, 5, 2) @ bundle["m3_Wc2"] + bundle["m3_bc2"])

    hc_b3 = _hard_swish(_dilated_conv1d_input(hc, 5, 4) @ bundle["m3_Wc3"] + bundle["m3_bc3"])
    se_mean = hc_b3.mean(axis=0, keepdims=True)
    se_z = _hard_swish(se_mean @ bundle["m3_W_se1"] + bundle["m3_b_se1"]) @ bundle["m3_W_se2"] + bundle["m3_b_se2"]
    se_gate = np.clip(se_z * np.float32(0.2) + np.float32(0.5), np.float32(0.0), np.float32(1.0))
    hc = hc + hc_b3 * (np.float32(2.0) * se_gate)

    hc = hc + _hard_swish(_dilated_conv1d_input(hc, 3, 1) @ bundle["m3_Wc4"] + bundle["m3_bc4"])
    logits_m3 = (hc @ bundle["m3_W_out"] + bundle["m3_b_out"]) * np.float32(bundle["m3_inv_temp"])
    p3 = _softmax(logits_m3)

    w_ens = bundle["s1_ens_weights"]
    p_ens = (w_ens[0] * p1 + w_ens[1] * p2 + w_ens[2] * p3).astype(np.float32)
    return p1, p2, p3, p_ens


def build_stage2_inputs(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p_ens: np.ndarray,
    p_gor4: np.ndarray,
    x_local: np.ndarray,
    x_global: np.ndarray,
) -> np.ndarray:
    """
    Constructs the Stage-2 Structure-to-Structure input matrix (L, 15*22 + 25 = 355):
      - 15-residue window over 22 channels:
          p_ens (3), p1 (3), p2 (3), p3 (3), entropy (1), p_gor4 (3),
          is_boundary (1), is_pro (1), is_gly (1), P_turn (1), mu_alpha_11 (1), mu_beta_7 (1)
      - 25 global context features:
          20-AA composition, log_length, rel_pos, global Stage-1 predicted mean (P_H, P_E, P_C)
    """
    L = p_ens.shape[0]
    eps = 1e-7
    p_safe = np.clip(p_ens, eps, 1.0)
    entropy = -(p_safe * np.log(p_safe)).sum(axis=1, keepdims=True)  # (L, 1)

    is_bound = np.zeros((L, 1), dtype=np.float32)
    is_pro = x_local[:, 50:51]        # IS_PRO channel in _PHYSICO_12 (21 + 20 + 9 = 50)
    is_gly = x_local[:, 51:52]        # IS_GLY channel (21 + 20 + 10 = 51)
    p_turn = x_local[:, 43:44]        # CF_P_TURN channel (21 + 20 + 2 = 43)
    mu_a11 = x_local[:, 54:55]        # mu_alpha_11 (53 + 1 = 54)
    mu_b7 = x_local[:, 57:58]         # mu_beta_7 (53 + 4 = 57)

    s2_local = np.hstack([
        p_ens, p1, p2, p3, entropy, p_gor4,
        is_bound, is_pro, is_gly, p_turn, mu_a11, mu_b7
    ]).astype(np.float32)  # (L, 22)

    # Pad 7 positions on each end with boundary token = 1.0 and neutral coil prior
    pad_row = np.zeros(22, dtype=np.float32)
    pad_row[2] = 1.0     # p_ens C=1 outside chain
    pad_row[5] = 1.0     # p1 C=1
    pad_row[8] = 1.0     # p2 C=1
    pad_row[11] = 1.0    # p3 C=1
    pad_row[15] = 1.0    # p_gor4 C=1
    pad_row[16] = 1.0    # is_boundary = 1

    padded = np.empty((L + 14, 22), dtype=np.float32)
    padded[:7] = pad_row
    padded[7:7 + L] = s2_local
    padded[7 + L:] = pad_row

    w15 = np.hstack([padded[offset:offset + L] for offset in range(15)])  # (L, 330)

    s1_global_comp = np.broadcast_to(p_ens.mean(axis=0, keepdims=True), (L, 3))
    glob_ctx = np.hstack([
        x_global[:, :21],    # 20-AA composition + log_len (21)
        x_global[:, 30:31],  # rel_pos (1)
        s1_global_comp,      # Stage-1 protein-wide predicted H/E/C share (3)
    ]).astype(np.float32)    # (L, 25)

    return np.hstack([w15, glob_ctx])  # (L, 355)


def _forward_stage2(s2_in: np.ndarray, bundle: Dict[str, np.ndarray]) -> np.ndarray:
    """Runs the Stage-2 Structure-to-Structure Context Network -> calibrated (L, 3) probabilities."""
    s2_n = (s2_in - bundle["s2_mean"]) * bundle["s2_inv_std"]
    h1 = _hard_swish(s2_n @ bundle["s2_W1"] + bundle["s2_b1"])
    h2 = _hard_swish(h1 @ bundle["s2_W2"] + bundle["s2_b2"])
    logits = (h2 @ bundle["s2_W3"] + bundle["s2_b3"]) * np.float32(bundle["s2_inv_temp"])
    return _softmax(logits).astype(np.float32)


def decode_viterbi_min_length(
    probs: np.ndarray,
    trans_matrix: Optional[np.ndarray] = None,
    state_bias: Optional[np.ndarray] = None,
) -> str:
    """
    Exact 7-State Minimum-Length Constrained Viterbi Decoder.
    Substates:
      0: C   (Coil/Turn, min length >= 1)
      1: H1  (Helix pos 1, entered from C or E2+)
      2: H2  (Helix pos 2, forced from H1)
      3: H3  (Helix pos 3, forced from H2)
      4: H4+ (Helix pos >= 4, entered from H3 or self-loop H4+)
      5: E1  (Strand pos 1, entered from C or H4+)
      6: E2+ (Strand pos >= 2, entered from E1 or self-loop E2+)
    Guarantees every predicted 'H' segment has length >= 4 aa and every 'E' segment >= 2 aa
    (unless total sequence length L < 4, handled cleanly).
    """
    L = probs.shape[0]
    if L == 0:
        return ""
    if L == 1:
        return "C"

    log_p = np.log(np.clip(probs, 1e-9, 1.0)).astype(np.float64)
    if state_bias is not None:
        log_p = log_p + np.asarray(state_bias, dtype=np.float64)[None, :]

    if L < 4:
        # Only E (length >= 2) or C possible when 2 <= L <= 3
        c_score = float(log_p[:, 2].sum())
        e_score = float(log_p[:, 1].sum()) - 0.15
        return "E" * L if e_score > c_score else "C" * L

    # 3x3 macro-state log-transition matrix [H, E, C] -> [H, E, C]
    if trans_matrix is None:
        t_cc = 0.0
        t_ch = -0.35
        t_ce = -0.25
        t_hh = 0.08
        t_hc = -0.25
        t_he = -0.65
        t_ee = 0.06
        t_ec = -0.20
        t_eh = -0.65
    else:
        tm = np.asarray(trans_matrix, dtype=np.float64)
        t_hh, t_he, t_hc = float(tm[0, 0]), float(tm[0, 1]), float(tm[0, 2])
        t_eh, t_ee, t_ec = float(tm[1, 0]), float(tm[1, 1]), float(tm[1, 2])
        t_ch, t_ce, t_cc = float(tm[2, 0]), float(tm[2, 1]), float(tm[2, 2])

    neg_inf = -1e18
    bp = bytearray(L * 7)

    lh_vec = log_p[:, 0].tolist()
    le_vec = log_p[:, 1].tolist()
    lc_vec = log_p[:, 2].tolist()

    # Initialize t = 0: chain can start in C (0), H1 (1), or E1 (5)
    v0 = lc_vec[0]
    v1 = lh_vec[0] + t_ch * 0.5
    v2 = neg_inf
    v3 = neg_inf
    v4 = neg_inf
    v5 = le_vec[0] + t_ce * 0.5
    v6 = neg_inf

    off = 7
    for t in range(1, L):
        lh = lh_vec[t]
        le = le_vec[t]
        lc = lc_vec[t]

        # Next 0 (C): from 0 (C->C), 4 (H4+->C), or 6 (E2+->C)
        s0_from_0 = v0 + t_cc
        s0_from_4 = v4 + t_hc
        s0_from_6 = v6 + t_ec
        if s0_from_0 >= s0_from_4 and s0_from_0 >= s0_from_6:
            nv0 = s0_from_0 + lc
            bp0 = 0
        elif s0_from_4 >= s0_from_6:
            nv0 = s0_from_4 + lc
            bp0 = 4
        else:
            nv0 = s0_from_6 + lc
            bp0 = 6

        # Next 1 (H1): from 0 (C->H1) or 6 (E2+->H1)
        s1_from_0 = v0 + t_ch
        s1_from_6 = v6 + t_eh
        if s1_from_0 >= s1_from_6:
            nv1 = s1_from_0 + lh
            bp1 = 0
        else:
            nv1 = s1_from_6 + lh
            bp1 = 6

        # Deterministic duration chain H1 -> H2 -> H3
        nv2 = v1 + t_hh + lh
        nv3 = v2 + t_hh + lh

        # Next 4 (H4+): from 3 (H3->H4+) or 4 (H4+->H4+)
        if v3 >= v4:
            nv4 = v3 + t_hh + lh
            bp4 = 3
        else:
            nv4 = v4 + t_hh + lh
            bp4 = 4

        # Next 5 (E1): from 0 (C->E1) or 4 (H4+->E1)
        s5_from_0 = v0 + t_ce
        s5_from_4 = v4 + t_he
        if s5_from_0 >= s5_from_4:
            nv5 = s5_from_0 + le
            bp5 = 0
        else:
            nv5 = s5_from_4 + le
            bp5 = 4

        # Next 6 (E2+): from 5 (E1->E2+) or 6 (E2+->E2+)
        if v5 >= v6:
            nv6 = v5 + t_ee + le
            bp6 = 5
        else:
            nv6 = v6 + t_ee + le
            bp6 = 6

        bp[off] = bp0
        bp[off + 1] = bp1
        bp[off + 2] = 1
        bp[off + 3] = 2
        bp[off + 4] = bp4
        bp[off + 5] = bp5
        bp[off + 6] = bp6

        v0, v1, v2, v3, v4, v5, v6 = nv0, nv1, nv2, nv3, nv4, nv5, nv6
        off += 7

    # Valid terminal states at t = L - 1: C (0), completed Helix H4+ (4), or completed Strand E2+ (6)
    if v0 >= v4 and v0 >= v6:
        curr = 0
    elif v4 >= v6:
        curr = 4
    else:
        curr = 6

    sub_to_char = ("C", "H", "H", "H", "H", "E", "E")
    chars = [""] * L
    off = (L - 1) * 7
    for t in range(L - 1, -1, -1):
        chars[t] = sub_to_char[curr]
        curr = bp[off + curr]
        off -= 7
    return "".join(chars)


# ==============================================================================
# 5. MODEL WEIGHTS LOADER & MAIN PUBLIC PREDICTION API
# ==============================================================================

def load_ss_model_weights(weights_path: Optional[str] = None, force_reload: bool = False) -> Dict[str, np.ndarray]:
    """Loads and caches pure-NumPy float32 weights from data/ss_model_weights.npz."""
    global _CACHED_MODEL_BUNDLE, _CACHED_MODEL_PATH
    target_path = weights_path or _DEFAULT_WEIGHTS_PATH
    if not force_reload and _CACHED_MODEL_BUNDLE is not None and _CACHED_MODEL_PATH == target_path:
        return _CACHED_MODEL_BUNDLE

    if not os.path.exists(target_path):
        raise FileNotFoundError(
            f"Secondary structure model weights file not found at {target_path}. "
            "Run scripts/train_and_benchmark_ss.py to generate it."
        )

    with np.load(target_path, allow_pickle=False) as npz:
        bundle = {k: npz[k] for k in npz.files}

    # Pre-fuse Model 1 & Model 2 first-layer matrices and flatten GOR-IV table in memory for speed
    if "m1_W1" in bundle and "m2_W1" in bundle:
        bundle["m12_W1_fused"] = np.ascontiguousarray(np.hstack([bundle["m1_W1"], bundle["m2_W1"]]))
        bundle["m12_b1_fused"] = np.ascontiguousarray(np.concatenate([bundle["m1_b1"], bundle["m2_b1"]]))
    if "gor_pair_flat_table" in bundle:
        bundle["gor_pair_2d"] = np.ascontiguousarray(bundle["gor_pair_flat_table"].reshape(136 * 441, 3))

    _CACHED_MODEL_BUNDLE = bundle
    _CACHED_MODEL_PATH = target_path
    return bundle


def classify_fold_class(helix_pct: float, strand_pct: float, coil_pct: Optional[float] = None) -> Tuple[str, str]:
    """
    Classifies overall protein structural topology into one of the 4 canonical textbook
    secondary structure fold classes along with a student-friendly 1-sentence explanation.
    """
    structured = helix_pct + strand_pct
    if structured < 20.0:
        return (
            "Coil / Intrinsically Disordered-Rich",
            "Predominantly flexible loops and intrinsically disordered segments (<20% regular α-helix or β-sheet), typical of signaling tails and IDPs.",
        )

    # All-alpha: predominantly helical with minimal beta-sheet content (<5% E, or >=55% H with <8% E)
    if (strand_pct < 5.0 and helix_pct >= 15.0) or (helix_pct >= 55.0 and strand_pct < 8.0):
        return (
            "All-α (Helix-Rich)",
            f"Dominated by α-helical bundles ({helix_pct:.1f}% α-helix, {strand_pct:.1f}% β-strand), characteristic of globins (Myoglobin), serum albumins, and helical scaffolds.",
        )

    # All-beta: predominantly extended beta-sheets/barrels with low alpha-helix content
    # (allows short 3_10 cap / central chromophore helices as in GFP's 11-stranded beta-barrel)
    if (helix_pct < 8.0 and strand_pct >= 20.0) or (helix_pct < 5.0 and strand_pct >= 15.0) or (
        strand_pct >= 38.0 and helix_pct <= 18.5 and strand_pct >= 3.1 * max(0.1, helix_pct)
    ):
        return (
            "All-β (Beta-Sheet Rich)",
            f"Dominated by extended β-strands ({strand_pct:.1f}% β-strand, {helix_pct:.1f}% α-helix) that assemble into β-barrels (like GFP) or antiparallel β-sandwiches.",
        )

    return (
        "Mixed α/β",
        f"Combines both α-helices ({helix_pct:.1f}%) and β-strands ({strand_pct:.1f}%) into an α/β or α+β architecture, typical of catalytic enzymes (Lysozyme, Lipase A, Carbonic Anhydrase).",
    )


def extract_ss_segments(
    sequence: str,
    ss_string: str,
    prob_h: np.ndarray,
    prob_e: np.ndarray,
    prob_c: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Extracts contiguous secondary structure elements with 1-based inclusive coordinates,
    sequential element labels (α1..αN, β1..βM, L1..LK), mean state confidence, and sequence.
    """
    L = len(ss_string)
    if L == 0:
        return []

    segments: List[Dict[str, Any]] = []
    counts = {"H": 0, "E": 0, "C": 0}
    prefixes = {"H": "α", "E": "β", "C": "L"}
    type_names = {"H": "α-Helix", "E": "β-Strand", "C": "Coil / Loop"}
    prob_map = {"H": prob_h, "E": prob_e, "C": prob_c}

    seg_start = 0
    curr_state = ss_string[0]

    for idx in range(1, L + 1):
        if idx == L or ss_string[idx] != curr_state:
            counts[curr_state] += 1
            seg_end = idx - 1  # 0-based inclusive
            mean_c = float(prob_map[curr_state][seg_start:idx].mean())
            segments.append({
                "type": curr_state,
                "type_name": type_names[curr_state],
                "label": f"{prefixes[curr_state]}{counts[curr_state]}",
                "start": seg_start + 1,   # 1-based inclusive
                "end": seg_end + 1,       # 1-based inclusive
                "length": idx - seg_start,
                "mean_conf": round(mean_c, 4),
                "sequence": sequence[seg_start:idx],
            })
            if idx < L:
                seg_start = idx
                curr_state = ss_string[idx]

    return segments


def _predict_probs_overlap_tiles(
    clean_seq: str,
    bundle: Dict[str, np.ndarray],
    tile_window: int = MEGA_TILE_WINDOW,
    tile_stride: int = MEGA_TILE_STRIDE,
    eff_domain_len: float = MEGA_EFFECTIVE_DOMAIN_LEN,
) -> np.ndarray:
    """
    Domain-scale overlapping tile inference for giant multi-domain proteins (L > 1,500, e.g.
    3,685-aa Dystrophin or 34,350-aa Titin):
      1. Extracts all local per-residue biophysical, float64-amphipathic, H-bond autocorrelation,
         L1-lag GOR-IV, and true chain-end terminal proximity features continuously across 1..L
         (zero artificial chain breaks inside the polypeptide).
      2. Evaluates Stage-1 + Stage-2 neural ensembles over overlapping W_tile = 800 aa domain
         windows with stride S_tile = 600 aa (200-aa overlap zone = 100-aa margin on both ends),
         computing local domain composition/propensity context and domain-scale log_length so
         giant sequences never extrapolate out-of-distribution or dilute folded repeats against
         disordered regions (such as Titin's PEVK spring vs ~300 Ig/Fn3 beta-sandwich domains).
      3. Cross-fades tile probabilities seamlessly across each 200-aa overlap using a 24-residue
         receptive-field guard + C^1-smooth cosine ramp summing identically to 1.0.
    """
    L = len(clean_seq)
    seq_idx = encode_sequence_indices(clean_seq)
    n_workers = 6 if L >= 4000 else 2
    pool = _get_mega_tile_pool(6)

    eis_h = _EISENBERG_LOOKUP[seq_idx]
    pol_v = _POLARITY_LOOKUP[seq_idx]
    moments = _compute_all_amphipathic_moments(eis_h, pol_v)

    # L1-cache-resident lag-grouped GOR-IV features across strips
    safe_idx = np.where(seq_idx > 20, PAD_IDX, seq_idx)
    padded_g = np.empty(L + 16, dtype=np.int32)
    padded_g[:8] = PAD_IDX
    padded_g[8:8 + L] = safe_idx
    padded_g[8 + L:] = PAD_IDX

    single_tab = bundle["gor_single_table"]
    pair_3d = bundle["gor_pair_flat_table"]
    i_single = np.zeros((L, 3), dtype=np.float32)
    i_pair_all = np.zeros((L, 3), dtype=np.float32)
    i_pair_local = np.zeros((L, 3), dtype=np.float32)

    def _gor_strip(bounds: Tuple[int, int]) -> None:
        gs, ge = bounds
        gl = ge - gs
        if gl <= 0:
            return
        t3 = np.empty((gl, 3), dtype=np.float32)
        is_sub = i_single[gs:ge]
        ipa_sub = i_pair_all[gs:ge]
        ipl_sub = i_pair_local[gs:ge]
        for m in range(17):
            np.take(single_tab[m], padded_g[gs + m:ge + m], axis=0, out=t3)
            is_sub += t3
        l_codes: List[Optional[np.ndarray]] = [None] + [
            padded_g[gs:ge + 16 - d] * 21 + padded_g[gs + d:ge + 16] for d in range(1, 17)
        ]
        pi = 0
        for m in range(16):
            for n in range(m + 1, 17):
                d = n - m
                lc_arr = l_codes[d]
                assert lc_arr is not None
                np.take(pair_3d[pi], lc_arr[m:m + gl], axis=0, out=t3)
                ipa_sub += t3
                if d <= 4:
                    ipl_sub += t3
                pi += 1

    strip_step = (L + n_workers - 1) // n_workers
    strips = [(s, min(L, s + strip_step)) for s in range(0, L, strip_step)]
    if n_workers > 1 and len(strips) > 1:
        list(pool.map(_gor_strip, strips))
    else:
        for b_tup in strips:
            _gor_strip(b_tup)

    l_gor4 = (2.0 / 17.0) * i_pair_all - (15.0 / 17.0) * i_single + bundle["gor_prior_log_odds"][None, :]
    temp = float(bundle.get("gor_temperature", np.array(1.0, dtype=np.float32)))
    scaled_logits = l_gor4 * (1.0 / max(0.1, temp))
    scaled_logits -= scaled_logits.max(axis=1, keepdims=True)
    np.exp(scaled_logits, out=scaled_logits)
    p_gor4 = (scaled_logits / scaled_logits.sum(axis=1, keepdims=True)).astype(np.float32)

    # Assemble full-chain local feature matrix x_local (L, 81) directly into one buffer
    x_local = np.empty((L, 81), dtype=np.float32)
    x_local[:, :21] = _ONEHOT_21[seq_idx]
    x_local[:, 21:41] = _BLOSUM62_NORM[seq_idx]
    phys = _PHYSICO_12[seq_idx]
    x_local[:, 41:53] = phys
    x_local[:, 53:60] = moments

    h_pad = np.pad(np.maximum(0.0, eis_h), (4, 4), mode="constant")
    q_pad = np.pad(_CHARGE_LOOKUP[seq_idx], (4, 4), mode="constant")
    pro_pad = np.pad(phys[:, 9], (4, 4), mode="constant")
    x_local[:, 60] = 0.25 * (
        h_pad[4:4 + L] * h_pad[7:7 + L]
        + h_pad[4:4 + L] * h_pad[8:8 + L]
        + h_pad[4:4 + L] * h_pad[1:1 + L]
        + h_pad[4:4 + L] * h_pad[0:L]
    )
    x_local[:, 61] = 0.25 * (
        np.maximum(0.0, -q_pad[4:4 + L] * q_pad[7:7 + L])
        + np.maximum(0.0, -q_pad[4:4 + L] * q_pad[8:8 + L])
        + np.maximum(0.0, -q_pad[4:4 + L] * q_pad[1:1 + L])
        + np.maximum(0.0, -q_pad[4:4 + L] * q_pad[0:L])
    )
    eis_pad = np.pad(eis_h, (2, 2), mode="constant")
    x_local[:, 62] = 0.5 * (eis_pad[2:2 + L] * eis_pad[4:4 + L] + eis_pad[2:2 + L] * eis_pad[0:L])
    x_local[:, 63] = -0.5 * (eis_pad[2:2 + L] * eis_pad[3:3 + L] + eis_pad[2:2 + L] * eis_pad[1:1 + L])
    x_local[:, 64] = (pro_pad[5:5 + L] + pro_pad[6:6 + L] + pro_pad[7:7 + L]) * (1.0 / 3.0)
    x_local[:, 65:68] = i_single * 0.15
    x_local[:, 68:71] = i_pair_local * 0.04
    x_local[:, 71:74] = l_gor4 * 0.25
    x_local[:, 74:77] = p_gor4

    pos = np.arange(L, dtype=np.float32)
    rev_pos = (L - 1) - pos
    x_local[:, 77] = np.exp(-pos / 5.0)
    x_local[:, 78] = np.exp(-pos / 15.0)
    x_local[:, 79] = np.exp(-rev_pos / 5.0)
    x_local[:, 80] = np.exp(-rev_pos / 15.0)

    x_loc_n = (x_local - bundle["loc_mean"]) * bundle["loc_inv_std"]

    # Build overlapping domain tiles [ts, te) of exact length tile_window (800 aa), stride 600 aa
    tw = min(L, int(tile_window))
    ts_step = max(1, int(tile_stride))
    tiles: List[Tuple[int, int]] = []
    s_cur = 0
    while s_cur + tw < L:
        tiles.append((s_cur, s_cur + tw))
        s_cur += ts_step
    tiles.append((max(0, L - tw), L))
    K = len(tiles)
    probs_3d = np.empty((K, tw, 3), dtype=np.float32)

    rel_pos_full = (pos / max(1.0, float(L - 1))).astype(np.float32)[:, None]
    log_len_t = math.log(max(1.0, float(eff_domain_len)) / 200.0)
    has_blend = "s2_blend_alpha" in bundle
    alpha = float(bundle["s2_blend_alpha"]) if has_blend else 1.0
    use_fast_l2 = ("s1_W_proj" not in bundle) and ("m12_W1_fused" in bundle)

    def _run_tile_batch(t_indices: List[int]) -> None:
        if not t_indices:
            return
        if not use_fast_l2:
            for idx_t in t_indices:
                ts, te = tiles[idx_t]
                tl = te - ts
                xl_t = x_local[ts:te]
                pg_t = p_gor4[ts:te]
                glob_vec_t = np.concatenate([
                    xl_t[:, :20].mean(axis=0),
                    np.array([log_len_t], dtype=np.float32),
                    pg_t.mean(axis=0),
                    pg_t.std(axis=0),
                    np.array([float(xl_t[:, 54].mean()), float(xl_t[:, 57].mean()), float(xl_t[:, 43].mean())], dtype=np.float32),
                ]).astype(np.float32)
                xg_t = np.hstack([np.broadcast_to(glob_vec_t[None, :], (tl, 30)), rel_pos_full[ts:te]]).astype(np.float32)
                p1_t, p2_t, p3_t, pe_t = _forward_stage1_ensemble(xl_t, xg_t, bundle)
                s2_t = build_stage2_inputs(p1_t, p2_t, p3_t, pe_t, pg_t, xl_t, xg_t)
                pr_t = _forward_stage2(s2_t, bundle)
                if has_blend:
                    pr_t = alpha * pr_t + (1.0 - alpha) * pe_t
                    pr_t = pr_t / pr_t.sum(axis=1, keepdims=True)
                probs_3d[idx_t] = pr_t
            return

        # Zero-allocation per-worker L2-cache scratch buffers
        d1 = int(bundle["m1_W2"].shape[0])
        d1_2 = int(bundle["m1_W2"].shape[1])
        d2 = int(bundle["m2_W2"].shape[0])
        d2_2 = int(bundle["m2_W2"].shape[1])
        C_m3 = int(bundle["m3_W_stem"].shape[1])
        ds2_1 = int(bundle["s2_W1"].shape[1])
        ds2_2 = int(bundle["s2_W2"].shape[1])

        win_buf = np.empty((tw, 17 * 81 + 31), dtype=np.float32)
        pad17_buf = np.empty((tw + 16, 81), dtype=np.float32)
        z1_buf = np.empty((tw, d1 + d2), dtype=np.float32)
        h1_m1 = np.empty((tw, d1), dtype=np.float32)
        scr_d1 = np.empty((tw, d1), dtype=np.float32)
        h2_m1 = np.empty((tw, d1_2), dtype=np.float32)
        scr_d1_2 = np.empty((tw, d1_2), dtype=np.float32)
        h1_m2 = np.empty((tw, d2), dtype=np.float32)
        rm2 = np.empty((tw, d2), dtype=np.float32)
        scr_d2 = np.empty((tw, d2), dtype=np.float32)
        h2_m2 = np.empty((tw, d2_2), dtype=np.float32)
        scr_d2_2 = np.empty((tw, d2_2), dtype=np.float32)

        stem_buf = np.empty((tw, 81 + 31), dtype=np.float32)
        hc = np.empty((tw, C_m3), dtype=np.float32)
        h_conv = np.empty((tw, C_m3), dtype=np.float32)
        sc_c = np.empty((tw, C_m3), dtype=np.float32)
        conv5_buf = np.empty((tw, 5 * C_m3), dtype=np.float32)
        conv3_buf = np.empty((tw, 3 * C_m3), dtype=np.float32)
        hp_max = np.zeros((tw + 16, C_m3), dtype=np.float32)

        s2_pad_row = np.zeros(22, dtype=np.float32)
        for c_i in (2, 5, 8, 11, 15, 16):
            s2_pad_row[c_i] = 1.0
        s2p = np.empty((tw + 14, 22), dtype=np.float32)
        s2b = np.empty((tw, 355), dtype=np.float32)
        s2n = np.empty((tw, 355), dtype=np.float32)
        sh1 = np.empty((tw, ds2_1), dtype=np.float32)
        scr_s2_1 = np.empty((tw, ds2_1), dtype=np.float32)
        sh2 = np.empty((tw, ds2_2), dtype=np.float32)
        scr_s2_2 = np.empty((tw, ds2_2), dtype=np.float32)

        pad_vec = bundle["loc_pad_norm"]
        w_ens = bundle["s1_ens_weights"]
        inv_6 = np.float32(1.0 / 6.0)
        half_f = np.float32(0.5)
        zero_f = np.float32(0.0)
        one_f = np.float32(1.0)
        pt1_f = np.float32(0.1)

        def _hs_ip(x: np.ndarray, scr: np.ndarray) -> None:
            np.multiply(x, inv_6, out=scr)
            scr += half_f
            np.clip(scr, zero_f, one_f, out=scr)
            x *= scr

        def _lr_ip(x: np.ndarray, scr: np.ndarray) -> None:
            np.multiply(x, pt1_f, out=scr)
            np.maximum(x, scr, out=x)

        for idx_t in t_indices:
            ts, te = tiles[idx_t]
            p_gor_t = p_gor4[ts:te]
            xl_t = x_local[ts:te]
            glob_vec_t = np.empty(30, dtype=np.float32)
            glob_vec_t[:20] = xl_t[:, :20].mean(axis=0)
            glob_vec_t[20] = log_len_t
            glob_vec_t[21:24] = p_gor_t.mean(axis=0)
            glob_vec_t[24:27] = p_gor_t.std(axis=0)
            glob_vec_t[27] = float(xl_t[:, 54].mean())
            glob_vec_t[28] = float(xl_t[:, 57].mean())
            glob_vec_t[29] = float(xl_t[:, 43].mean())

            xg_view = win_buf[:, 1377:]
            xg_view[:, :30] = (glob_vec_t - bundle["glob_mean"][:30]) * bundle["glob_inv_std"][:30]
            xg_view[:, 30:31] = (rel_pos_full[ts:te] - bundle["glob_mean"][30]) * bundle["glob_inv_std"][30]

            if ts >= 8 and te + 8 <= L:
                p17_src = x_loc_n[ts - 8:te + 8]
            else:
                pad17_buf[:8] = pad_vec if ts == 0 else x_loc_n[ts - 8:ts]
                pad17_buf[8:8 + tw] = x_loc_n[ts:te]
                pad17_buf[8 + tw:] = pad_vec if te == L else x_loc_n[te:te + 8]
                p17_src = pad17_buf

            for k in range(17):
                win_buf[:, k * 81:(k + 1) * 81] = p17_src[k:k + tw]

            np.matmul(win_buf, bundle["m12_W1_fused"], out=z1_buf)
            z1_buf += bundle["m12_b1_fused"]

            h1_m1[:] = z1_buf[:, :d1]
            _hs_ip(h1_m1, scr_d1)
            np.matmul(h1_m1, bundle["m1_W2"], out=h2_m1)
            h2_m1 += bundle["m1_b2"]
            _hs_ip(h2_m1, scr_d1_2)
            p1_t = _softmax((h2_m1 @ bundle["m1_W3"] + bundle["m1_b3"]) * np.float32(bundle["m1_inv_temp"]))

            h1_m2[:] = z1_buf[:, d1:]
            _lr_ip(h1_m2, scr_d2)
            np.matmul(h1_m2, bundle["m2_W_res"], out=rm2)
            rm2 += bundle["m2_b_res"]
            _lr_ip(rm2, scr_d2)
            h1_m2 += rm2
            np.matmul(h1_m2, bundle["m2_W2"], out=h2_m2)
            h2_m2 += bundle["m2_b2"]
            _lr_ip(h2_m2, scr_d2_2)
            p2_t = _softmax((h2_m2 @ bundle["m2_W3"] + bundle["m2_b3"]) * np.float32(bundle["m2_inv_temp"]))

            stem_buf[:, :81] = x_loc_n[ts:te]
            stem_buf[:, 81:] = xg_view
            np.matmul(stem_buf, bundle["m3_W_stem"], out=hc)
            hc += bundle["m3_b_stem"]
            _hs_ip(hc, sc_c)

            for Wc, bc, ksz, dil, is_se in (
                (bundle["m3_Wc1"], bundle["m3_bc1"], 5, 1, False),
                (bundle["m3_Wc2"], bundle["m3_bc2"], 5, 2, False),
                (bundle["m3_Wc3"], bundle["m3_bc3"], 5, 4, True),
                (bundle["m3_Wc4"], bundle["m3_bc4"], 3, 1, False),
            ):
                rad = (ksz // 2) * dil
                hp = hp_max[:tw + 2 * rad]
                hp[:rad] = 0.0
                hp[rad:rad + tw] = hc
                hp[rad + tw:] = 0.0
                cb = conv5_buf if ksz == 5 else conv3_buf
                for j in range(ksz):
                    s_off = j * dil
                    cb[:, j * C_m3:(j + 1) * C_m3] = hp[s_off:s_off + tw]
                np.matmul(cb, Wc, out=h_conv)
                h_conv += bc
                _hs_ip(h_conv, sc_c)
                if is_se:
                    se_mean = h_conv.mean(axis=0, keepdims=True)
                    se_z = _hard_swish(se_mean @ bundle["m3_W_se1"] + bundle["m3_b_se1"]) @ bundle["m3_W_se2"] + bundle["m3_b_se2"]
                    se_gate = np.clip(se_z * np.float32(0.2) + np.float32(0.5), zero_f, one_f)
                    np.multiply(h_conv, np.float32(2.0) * se_gate, out=h_conv)
                    hc += h_conv
                else:
                    hc += h_conv

            p3_t = _softmax((hc @ bundle["m3_W_out"] + bundle["m3_b_out"]) * np.float32(bundle["m3_inv_temp"]))
            p_ens_t = (w_ens[0] * p1_t + w_ens[1] * p2_t + w_ens[2] * p3_t).astype(np.float32)

            p_safe = np.clip(p_ens_t, 1e-7, 1.0)
            ent = -(p_safe * np.log(p_safe)).sum(axis=1)
            s2p[:7] = s2_pad_row
            s2p[7 + tw:] = s2_pad_row
            mid = s2p[7:7 + tw]
            mid[:, 0:3] = p_ens_t
            mid[:, 3:6] = p1_t
            mid[:, 6:9] = p2_t
            mid[:, 9:12] = p3_t
            mid[:, 12] = ent
            mid[:, 13:16] = p_gor_t
            mid[:, 16] = 0.0
            mid[:, 17] = xl_t[:, 50]
            mid[:, 18] = xl_t[:, 51]
            mid[:, 19] = xl_t[:, 43]
            mid[:, 20] = xl_t[:, 54]
            mid[:, 21] = xl_t[:, 57]

            for k in range(15):
                s2b[:, k * 22:(k + 1) * 22] = s2p[k:k + tw]
            s2b[:, 330:350] = glob_vec_t[:20]
            s2b[:, 350] = log_len_t
            s2b[:, 351:352] = rel_pos_full[ts:te]
            s2b[:, 352:355] = p_ens_t.mean(axis=0)

            np.subtract(s2b, bundle["s2_mean"], out=s2n)
            s2n *= bundle["s2_inv_std"]
            np.matmul(s2n, bundle["s2_W1"], out=sh1)
            sh1 += bundle["s2_b1"]
            _hs_ip(sh1, scr_s2_1)
            np.matmul(sh1, bundle["s2_W2"], out=sh2)
            sh2 += bundle["s2_b2"]
            _hs_ip(sh2, scr_s2_2)
            probs_t = _softmax((sh2 @ bundle["s2_W3"] + bundle["s2_b3"]) * np.float32(bundle["s2_inv_temp"])).astype(np.float32)
            if has_blend:
                probs_t = alpha * probs_t + (1.0 - alpha) * p_ens_t
                probs_t /= probs_t.sum(axis=1, keepdims=True)
            probs_3d[idx_t] = probs_t

    prev_ob = _OB_GET_THREADS() if _OB_GET_THREADS is not None else None
    if _OB_SET_THREADS is not None and K >= 4:
        _OB_SET_THREADS(ctypes.c_int(1))
    try:
        if K >= 4 and n_workers > 1:
            batches = [list(range(w_i, K, n_workers)) for w_i in range(n_workers)]
            list(pool.map(_run_tile_batch, batches))
        else:
            _run_tile_batch(list(range(K)))
    finally:
        if _OB_SET_THREADS is not None and prev_ob is not None and K >= 4:
            _OB_SET_THREADS(ctypes.c_int(prev_ob))

    # Smooth C^1 cosine cross-fade across 200-aa overlap zones (24-aa receptive-field guard at internal boundaries)
    probs_accum = np.zeros((L, 3), dtype=np.float64)
    weights_accum = np.zeros((L, 1), dtype=np.float64)
    guard = min(24, max(0, (tw - ts_step) // 4))
    for idx_t, (ts, te) in enumerate(tiles):
        w = np.ones(tw, dtype=np.float64)
        if ts > 0:
            ov_left = min(tw, tiles[idx_t - 1][1] - ts)
            g_l = min(guard, max(0, (ov_left - 2) // 2))
            ramp_len = max(1, ov_left - 2 * g_l)
            if g_l > 0:
                w[:g_l] = 0.0
            u = (np.arange(ramp_len, dtype=np.float64) + 0.5) / float(ramp_len)
            w[g_l:g_l + ramp_len] = 0.5 * (1.0 - np.cos(math.pi * u))
        if te < L:
            ov_right = min(tw, te - tiles[idx_t + 1][0])
            g_r = min(guard, max(0, (ov_right - 2) // 2))
            ramp_len = max(1, ov_right - 2 * g_r)
            if g_r > 0:
                w[tw - g_r:] = 0.0
            u = (np.arange(ramp_len, dtype=np.float64) + 0.5) / float(ramp_len)
            w[tw - g_r - ramp_len:tw - g_r] = 0.5 * (1.0 + np.cos(math.pi * u))
        probs_accum[ts:te] += probs_3d[idx_t].astype(np.float64) * w[:, None]
        weights_accum[ts:te] += w[:, None]

    probs = (probs_accum / np.maximum(1e-12, weights_accum)).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def predict_secondary_structure(
    sequence: str,
    model_bundle: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Predicts 3-state protein secondary structure (H = α-Helix, E = β-Strand, C = Coil/Turn)
    using the pure-NumPy two-stage neural ensemble + constrained minimum-length Viterbi decoder.
    For giant multi-domain proteins (L > 1,500, such as 3,685-aa Dystrophin or 34,350-aa Titin),
    automatically applies seamless 800-aa / 600-stride domain-scale overlap-tile probability
    inference followed by global constrained Viterbi decoding, while keeping L <= 1,500
    predictions 100% identical to the 11,118-protein PDB/DSSP benchmark.

    Returns a dictionary containing:
      - sequence: str (cleaned uppercase protein sequence)
      - length: int
      - ss_string: str ('H', 'E', 'C' string of length L)
      - prob_h, prob_e, prob_c: List[float] per-residue calibrated probabilities
      - helix_pct, strand_pct, coil_pct: float percentages (summing to ~100.0)
      - helix_count, strand_count, coil_count: int counts of structural segments
      - mean_confidence: float mean winning-state probability across residues
      - fold_class: str ('All-α (Helix-Rich)', 'All-β (Beta-Sheet Rich)', 'Mixed α/β', or
                         'Coil / Intrinsically Disordered-Rich')
      - fold_description: str student-friendly summary of the predicted architecture
      - segments: List[Dict[str, Any]] 1-based inclusive segment annotations
    """
    clean_seq = clean_protein_sequence_for_ss(sequence)
    L = len(clean_seq)
    bundle = model_bundle if model_bundle is not None else load_ss_model_weights()

    if L <= MAX_SINGLE_DOMAIN_LENGTH:
        x_local, x_global, p_gor4, _ = extract_residue_and_global_features(clean_seq, bundle)
        p1, p2, p3, p_ens = _forward_stage1_ensemble(x_local, x_global, bundle)
        s2_in = build_stage2_inputs(p1, p2, p3, p_ens, p_gor4, x_local, x_global)
        probs = _forward_stage2(s2_in, bundle)  # (L, 3)

        # Apply optional calibrated post-Stage-2 probability blend with Stage-1 ensemble if stored
        if "s2_blend_alpha" in bundle:
            alpha = float(bundle["s2_blend_alpha"])
            probs = alpha * probs + (1.0 - alpha) * p_ens
            probs = probs / probs.sum(axis=1, keepdims=True)
    else:
        probs = _predict_probs_overlap_tiles(clean_seq, bundle)

    ss_string = decode_viterbi_min_length(
        probs,
        trans_matrix=bundle.get("viterbi_trans"),
        state_bias=bundle.get("viterbi_bias"),
    )

    prob_h_arr = probs[:, 0]
    prob_e_arr = probs[:, 1]
    prob_c_arr = probs[:, 2]

    n_h = ss_string.count("H")
    n_e = ss_string.count("E")
    n_c = L - n_h - n_e

    helix_pct = round(100.0 * n_h / L, 2)
    strand_pct = round(100.0 * n_e / L, 2)
    coil_pct = round(100.0 * n_c / L, 2)

    fold_class, fold_description = classify_fold_class(helix_pct, strand_pct, coil_pct)
    segments = extract_ss_segments(clean_seq, ss_string, prob_h_arr, prob_e_arr, prob_c_arr)

    helix_count = sum(1 for seg in segments if seg["type"] == "H")
    strand_count = sum(1 for seg in segments if seg["type"] == "E")
    coil_count = sum(1 for seg in segments if seg["type"] == "C")

    if L <= MAX_SINGLE_DOMAIN_LENGTH:
        state_indices = np.fromiter((STATE_TO_IDX[ch] for ch in ss_string), dtype=np.int32, count=L)
        prob_h_list = [round(float(v), 4) for v in prob_h_arr]
        prob_e_list = [round(float(v), 4) for v in prob_e_arr]
        prob_c_list = [round(float(v), 4) for v in prob_c_arr]
    else:
        ss_bytes = np.frombuffer(ss_string.encode("ascii"), dtype=np.uint8)
        state_indices = np.where(ss_bytes == ord("H"), 0, np.where(ss_bytes == ord("E"), 1, 2))
        prob_h_list = np.round(prob_h_arr.astype(np.float64), 4).tolist()
        prob_e_list = np.round(prob_e_arr.astype(np.float64), 4).tolist()
        prob_c_list = np.round(prob_c_arr.astype(np.float64), 4).tolist()

    chosen_probs = probs[np.arange(L), state_indices]
    mean_conf = round(float(chosen_probs.mean()), 4)

    return {
        "sequence": clean_seq,
        "length": L,
        "ss_string": ss_string,
        "prob_h": prob_h_list,
        "prob_e": prob_e_list,
        "prob_c": prob_c_list,
        "helix_pct": helix_pct,
        "strand_pct": strand_pct,
        "coil_pct": coil_pct,
        "helix_count": helix_count,
        "strand_count": strand_count,
        "coil_count": coil_count,
        "mean_confidence": mean_conf,
        "fold_class": fold_class,
        "fold_description": fold_description,
        "segments": segments,
    }


# ==============================================================================
# 6. CLASSIC BASELINES (CHOU-FASMAN 1978 & GOR IV 1996) & SOV_99 METRIC
# ==============================================================================

def predict_chou_fasman(sequence: str) -> str:
    """
    Classic Chou & Fasman (1978) empirical rule-based secondary structure baseline:
      1. Identify beta-turns via positional tetrapeptide product p_t > 0.75e-4 and <P_t>_4 > 1.00.
      2. Nucleate alpha-helices on 6-residue windows (>=4 helix formers P_alpha >= 1.00, <2 breakers
         P_alpha < 0.70) and propagate while 4-mer <P_alpha>_4 >= 1.00 (halted by Proline).
      3. Nucleate beta-strands on 5-residue windows (>=3 beta formers P_beta >= 1.00) and propagate
         while 4-mer <P_beta>_4 >= 1.00.
      4. Resolve H/E overlaps by comparing segment mean <P_alpha> vs <P_beta>.
    """
    clean_seq = clean_protein_sequence_for_ss(sequence)
    seq_idx = np.minimum(encode_sequence_indices(clean_seq), 19)
    L = len(seq_idx)
    if L < 5:
        return "C" * L

    pa = CF_P_ALPHA[seq_idx]
    pb = CF_P_BETA[seq_idx]
    pt = CF_P_TURN[seq_idx]

    # Precompute rolling sums via cumulative sums for fast O(L) Chou-Fasman scanning
    cs_pa = np.empty(L + 1, dtype=np.float32); cs_pa[0] = 0.0; np.cumsum(pa, out=cs_pa[1:])
    cs_pb = np.empty(L + 1, dtype=np.float32); cs_pb[0] = 0.0; np.cumsum(pb, out=cs_pb[1:])
    cs_pt = np.empty(L + 1, dtype=np.float32); cs_pt[0] = 0.0; np.cumsum(pt, out=cs_pt[1:])

    pa4 = (cs_pa[4:] - cs_pa[:-4]) * np.float32(0.25)  # length L - 3
    pb4 = (cs_pb[4:] - cs_pb[:-4]) * np.float32(0.25)
    pt4 = (cs_pt[4:] - cs_pt[:-4]) * np.float32(0.25)

    # Beta-turn mask
    is_turn = np.zeros(L, dtype=bool)
    f0 = CF_TURN_F[seq_idx[:-3], 0]
    f1 = CF_TURN_F[seq_idx[1:-2], 1]
    f2 = CF_TURN_F[seq_idx[2:-1], 2]
    f3 = CF_TURN_F[seq_idx[3:], 3]
    turn_hits = np.flatnonzero((f0 * f1 * f2 * f3 > 0.75e-4) & (pt4 > 1.00) & (pt4 > pa4) & (pt4 > pb4))
    for i in turn_hits.tolist():
        is_turn[i:i + 4] = True

    # Alpha-helix nucleation & propagation
    h_mask = np.zeros(L, dtype=bool)
    is_pro = (seq_idx == AA_TO_IDX["P"])
    if L >= 6:
        w6_pa = sliding_window_view(pa, 6)
        w5_pro = sliding_window_view(is_pro[1:], 5)
        h_seeds = np.flatnonzero(((w6_pa >= 1.00).sum(axis=1) >= 4) & ((w6_pa < 0.70).sum(axis=1) < 2) & (~w5_pro.any(axis=1)))
        pa4_l = pa4.tolist()
        pro_l = is_pro.tolist()
        for i in h_seeds.tolist():
            left = i
            while left > 0 and not pro_l[left - 1] and pa4_l[left - 1] >= 1.00:
                left -= 1
            right = i + 5
            while right + 1 < L and not pro_l[right + 1] and pa4_l[right - 2] >= 1.00:
                right += 1
            seg_len = right - left + 1
            if seg_len >= 6 and (cs_pa[right + 1] - cs_pa[left]) > 1.03 * seg_len:
                h_mask[left:right + 1] = True

    # Beta-sheet nucleation & propagation
    e_mask = np.zeros(L, dtype=bool)
    w5_pb = sliding_window_view(pb, 5)
    e_seeds = np.flatnonzero(((w5_pb >= 1.00).sum(axis=1) >= 3) & ((w5_pb < 0.70).sum(axis=1) < 2))
    pb4_l = pb4.tolist()
    for i in e_seeds.tolist():
        left = i
        while left > 0 and pb4_l[left - 1] >= 1.00:
            left -= 1
        right = i + 4
        while right + 1 < L and pb4_l[right - 2] >= 1.00:
            right += 1
        seg_len = right - left + 1
        if seg_len >= 5 and (cs_pb[right + 1] - cs_pb[left]) > 1.05 * seg_len:
            e_mask[left:right + 1] = True

    # Turn regions override interior breaks; resolve remaining H vs E overlaps by mean propensity
    h_mask[is_turn] = False
    e_mask[is_turn] = False

    out = ["C"] * L
    idx = 0
    while idx < L:
        if h_mask[idx] and e_mask[idx]:
            end = idx
            while end + 1 < L and h_mask[end + 1] and e_mask[end + 1]:
                end += 1
            winner = "H" if float(pa[idx:end + 1].mean()) >= float(pb[idx:end + 1].mean()) else "E"
            for k in range(idx, end + 1):
                out[k] = winner
            idx = end + 1
        elif h_mask[idx]:
            out[idx] = "H"
            idx += 1
        elif e_mask[idx]:
            out[idx] = "E"
            idx += 1
        else:
            idx += 1

    # Enforce physical minimum length (H >= 4, E >= 2)
    res_chars = out[:]
    s = 0
    while s < L:
        e = s
        while e + 1 < L and res_chars[e + 1] == res_chars[s]:
            e += 1
        seg_len = e - s + 1
        if (res_chars[s] == "H" and seg_len < 4) or (res_chars[s] == "E" and seg_len < 2):
            for k in range(s, e + 1):
                res_chars[k] = "C"
        s = e + 1

    return "".join(res_chars)


def predict_gor4(sequence: str, gor_params: Optional[Dict[str, np.ndarray]] = None) -> Tuple[str, np.ndarray]:
    """
    Classic Garnier-Gibrat-Robson (1996) GOR-IV 17-residue pairwise information baseline,
    using parameters estimated strictly from the training split and standard GOR-IV
    minimum-length smoothing (Helix >= 4 aa, Strand >= 2 aa).
    """
    clean_seq = clean_protein_sequence_for_ss(sequence)
    params = gor_params if gor_params is not None else load_ss_model_weights()
    seq_idx = encode_sequence_indices(clean_seq)
    _, p_gor4 = compute_gor4_features(seq_idx, params)
    ss_pred = decode_viterbi_min_length(p_gor4, trans_matrix=None, state_bias=None)
    return ss_pred, p_gor4


def compute_q3(obs_ss: str, pred_ss: str) -> float:
    """Computes per-protein 3-state accuracy Q3 (%) in [0.0, 100.0]."""
    n = len(obs_ss)
    if n == 0 or len(pred_ss) != n:
        raise ValueError(f"Mismatched or empty secondary structure strings: {len(obs_ss)} vs {len(pred_ss)}")
    matches = sum(1 for a, b in zip(obs_ss, pred_ss) if a == b)
    return 100.0 * matches / float(n)


def _contiguous_state_intervals(ss: str, state: str) -> List[Tuple[int, int]]:
    """Returns 0-based inclusive (start, end) intervals for maximal contiguous segments of `state`."""
    intervals: List[Tuple[int, int]] = []
    n = len(ss)
    i = 0
    while i < n:
        if ss[i] == state:
            j = i
            while j + 1 < n and ss[j + 1] == state:
                j += 1
            intervals.append((i, j))
            i = j + 1
        else:
            i += 1
    return intervals


def compute_sov99(obs_ss: str, pred_ss: str, states: Tuple[str, ...] = ("H", "E", "C")) -> float:
    """
    Computes exact Segment Overlap SOV_99 (%) according to Zemla, Venclovas, Fidelis & Rost
    (Proteins, 1999, 34:220-223, Equations 2-4).
    """
    if len(obs_ss) != len(pred_ss):
        raise ValueError(f"Length mismatch in SOV_99: {len(obs_ss)} vs {len(pred_ss)}")
    if not obs_ss:
        return 100.0

    total_norm = 0.0
    total_sum = 0.0

    for st in states:
        s1_list = _contiguous_state_intervals(obs_ss, st)
        if not s1_list:
            continue
        s2_list = _contiguous_state_intervals(pred_ss, st)

        for a1, b1 in s1_list:
            len1 = b1 - a1 + 1
            has_overlap = False
            for a2, b2 in s2_list:
                # Overlap test: max(a1, a2) <= min(b1, b2)
                ov_start = a1 if a1 > a2 else a2
                ov_end = b1 if b1 < b2 else b2
                if ov_start <= ov_end:
                    has_overlap = True
                    len2 = b2 - a2 + 1
                    minov = ov_end - ov_start + 1
                    un_start = a1 if a1 < a2 else a2
                    un_end = b1 if b1 > b2 else b2
                    maxov = un_end - un_start + 1

                    # Zemla et al. (1999) Eq. 3 allowance delta(s1, s2)
                    delta = min(
                        maxov - minov,
                        minov,
                        len1 // 2,
                        len2 // 2,
                    )
                    total_norm += len1
                    total_sum += ((minov + delta) / float(maxov)) * len1

            if not has_overlap:
                total_norm += len1

    if total_norm <= 0.0:
        return 100.0
    return 100.0 * total_sum / total_norm


# ==============================================================================
# 7. MONOSPACE ALIGNMENT FORMATTER & 2-PANEL PLOTLY TOPOLOGY DIAGRAM
# ==============================================================================

def format_ss_sequence_blocks(
    sequence: str,
    ss_string: str,
    block_width: int = 60,
    start_pos: int = 1,
) -> str:
    """
    Formats aligned Pos / AA / SS monospace blocks for student & researcher inspection:
      H = α-Helix, E = β-Strand, · = Coil / Loop / Turn
    Supports optional 1-based `start_pos` offset so sub-domain windows of giant proteins
    display true full-protein residue coordinates.
    """
    if len(sequence) != len(ss_string):
        raise ValueError(f"Sequence length ({len(sequence)}) != SS length ({len(ss_string)}).")
    width = max(10, int(block_width))
    base_pos = max(1, int(start_pos))
    L = len(sequence)
    max_coord = max(base_pos, base_pos + L - 1)
    coord_w = max(4, len(str(max_coord)))
    pad_prefix = " " * (coord_w + 6)
    pos_prefix = "Pos" + " " * (coord_w + 3)
    ss_prefix = "SS" + " " * (coord_w + 4)
    ss_display = ss_string.replace("C", "·")

    blocks: List[str] = ["Legend:  H = α-Helix   |   E = β-Strand   |   · = Coil / Loop / Turn", ""]

    for start in range(0, L, width):
        end = min(L, start + width)
        chunk_len = end - start
        seq_chunk = sequence[start:end]
        ss_chunk = ss_display[start:end]
        coord_start = base_pos + start
        coord_end = base_pos + end - 1

        # Build clean ruler marking first residue of the block and right-aligning every 10th position
        ruler = [" "] * chunk_len
        for offset in range(chunk_len):
            pos_1 = base_pos + start + offset
            if offset == 0:
                lbl = str(pos_1)
                if len(lbl) <= chunk_len:
                    for k, ch in enumerate(lbl):
                        ruler[k] = ch
            elif pos_1 % 10 == 0:
                lbl = str(pos_1)
                left_idx = offset - len(lbl) + 1
                if left_idx >= 1 and all(ch == " " for ch in ruler[left_idx - 1:offset + 1]):
                    for k, ch in enumerate(lbl):
                        ruler[left_idx + k] = ch

        blocks.append(f"{pos_prefix}{''.join(ruler)}")
        blocks.append(f"AA  {coord_start:>{coord_w}d}  {seq_chunk}  {coord_end:<{coord_w}d}")
        blocks.append(f"{ss_prefix}{ss_chunk}")
        if end < L:
            blocks.append("")

    return "\n".join(blocks)


def generate_secondary_structure_figure(
    ss_result: Dict[str, Any],
    protein_name: str = "",
    region_start: int = 1,
    region_end: Optional[int] = None,
) -> go.Figure:
    """
    Builds an interactive, classroom-friendly 2-panel linked-axis Plotly figure:
      - Top Track (PDBsum-style Linear Topology Cartoon):
          * Smooth slate coil/loop backbone (#64748B)
          * Teal-Emerald cylinder backing + sinusoidal coiled α-helix ribbon wave (#0D9488)
          * Warm Amber directional block arrows (➔, #D97706 / #B45309) for β-strands
          * Anti-collision WCAG AA/AAA segment callout badges (α1, α2..., β1, β2...)
      - Bottom Track (Per-Residue Propensity / Confidence Profile):
          * Filled probability curves for α-Helix P(H) (#0D9488) & β-Strand P(E) (#D97706),
            plus subtle dashed Coil/Turn P(C) (#94A3B8).
      - Supports optional 1-based `[region_start, region_end]` domain window slicing AND
        automatic batched-polygon + bin-averaged macro overview rendering whenever the
        displayed span exceeds 1,500 residues (rendering 34,350-aa Titin in < 50 ms & < 350 KB).
    """
    seq = ss_result["sequence"]
    L = int(ss_result["length"])
    ss_str = ss_result["ss_string"]
    prob_h = ss_result["prob_h"]
    prob_e = ss_result["prob_e"]
    prob_c = ss_result["prob_c"]
    segments = ss_result["segments"]

    r_start = max(1, min(L, int(region_start)))
    r_end = L if region_end is None else max(r_start, min(L, int(region_end)))
    span_L = r_end - r_start + 1
    is_subwindow = (r_start > 1) or (r_end < L)
    is_macro_span = span_L > MAX_SINGLE_DOMAIN_LENGTH

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.43, 0.57],
    )

    x_pad = max(2.2, span_L * 0.028)

    # --- TOP TRACK: 1. Continuous Slate Coil Backbone ---
    fig.add_trace(
        go.Scatter(
            x=[r_start, r_end],
            y=[0.0, 0.0],
            mode="lines",
            name="Coil / Loop Backbone",
            line=dict(color="#64748B", width=3.4 if not is_macro_span else 2.6),
            hoverinfo="skip",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # N- and C-terminal / window boundary badges
    left_badge_txt = "<b>N</b>" if r_start == 1 else f"<b>{r_start:,}</b>"
    right_badge_txt = "<b>C</b>" if r_end == L else f"<b>{r_end:,}</b>"
    fig.add_annotation(
        x=r_start - x_pad * 0.42,
        y=0.0,
        text=left_badge_txt,
        showarrow=False,
        font=dict(family=PLOTLY_MONO_FAMILY, size=10.5, color="#1E352F"),
        bgcolor="#F1F5F9",
        bordercolor="#CBD5E1",
        borderwidth=1,
        borderpad=2,
        row=1,
        col=1,
    )
    fig.add_annotation(
        x=r_end + x_pad * 0.42,
        y=0.0,
        text=right_badge_txt,
        showarrow=False,
        font=dict(family=PLOTLY_MONO_FAMILY, size=10.5, color="#1E352F"),
        bgcolor="#F1F5F9",
        bordercolor="#CBD5E1",
        borderwidth=1,
        borderpad=2,
        row=1,
        col=1,
    )

    # Filter segments overlapping [r_start, r_end]
    vis_segs = [s for s in segments if s["end"] >= r_start and s["start"] <= r_end]
    helix_segs = [s for s in vis_segs if s["type"] == "H"]
    strand_segs = [s for s in vis_segs if s["type"] == "E"]

    if not is_macro_span:
        # --- HIGH-RESOLUTION MODE (span_L <= 1,500): Individual Cylinders, Coiled Waves & Directional Arrows ---
        helix_wave_x: List[Optional[float]] = []
        helix_wave_y: List[Optional[float]] = []
        helix_hover_x: List[float] = []
        helix_hover_y: List[float] = []
        helix_hover_txt: List[str] = []

        for seg in helix_segs:
            v0 = max(r_start, int(seg["start"]))
            v1 = min(r_end, int(seg["end"]))
            x0 = float(v0) - 0.36
            x1 = float(v1) + 0.36
            span = max(1.0, x1 - x0)

            fig.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=-0.34,
                y1=0.34,
                fillcolor="rgba(13, 148, 136, 0.16)",
                line=dict(color="rgba(15, 118, 110, 0.42)", width=1.1),
                layer="above",
                row=1,
                col=1,
            )

            raw_turns = span / 3.6
            eff_turns = min(raw_turns, max(3.5, 42.0 * (span / max(60.0, float(span_L)))))
            n_pts = max(28, min(180, int(eff_turns * 14)))
            ts = np.linspace(0.0, 1.0, n_pts)
            xs = x0 + ts * span
            env = np.sin(np.pi * np.clip(ts, 0.0, 1.0)) ** 0.28
            ys = 0.36 * env * np.sin(2.0 * np.pi * eff_turns * ts)

            helix_wave_x.extend(xs.tolist())
            helix_wave_x.append(None)
            helix_wave_y.extend(ys.tolist())
            helix_wave_y.append(None)

            seq_snip = seg["sequence"] if len(seg["sequence"]) <= 28 else (seg["sequence"][:25] + "...")
            h_txt = (
                f"<b>{seg['label']} • α-Helix</b><br>"
                f"Residues: {seg['start']}–{seg['end']} ({seg['length']} aa)<br>"
                f"Mean Confidence: {seg['mean_conf'] * 100:.1f}%<br>"
                f"Sequence: <code>{seq_snip}</code>"
            )
            for r_pos in range(v0, v1 + 1):
                helix_hover_x.append(float(r_pos))
                helix_hover_y.append(0.0)
                helix_hover_txt.append(h_txt)

        if helix_wave_x:
            fig.add_trace(
                go.Scatter(
                    x=helix_wave_x,
                    y=helix_wave_y,
                    mode="lines",
                    name="α-Helix (Coiled Ribbon)",
                    line=dict(color="#0D9488", width=3.0),
                    hoverinfo="skip",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=helix_hover_x,
                    y=helix_hover_y,
                    mode="markers",
                    marker=dict(size=12, color="rgba(13, 148, 136, 0.001)"),
                    hoverinfo="text",
                    hovertext=helix_hover_txt,
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        strand_poly_x: List[Optional[float]] = []
        strand_poly_y: List[Optional[float]] = []
        strand_hover_x: List[float] = []
        strand_hover_y: List[float] = []
        strand_hover_txt: List[str] = []

        for seg in strand_segs:
            v0 = max(r_start, int(seg["start"]))
            v1 = min(r_end, int(seg["end"]))
            x0 = float(v0) - 0.36
            x1 = float(v1) + 0.42
            span = max(1.1, x1 - x0)
            head_len = min(span * 0.52, max(1.15, min(3.4, span_L * 0.018)))
            x_neck = x1 - head_len

            y_shaft = 0.23
            y_head = 0.42

            poly_x = [x0, x_neck, x_neck, x1, x_neck, x_neck, x0, x0, None]
            poly_y = [-y_shaft, -y_shaft, -y_head, 0.0, y_head, y_shaft, y_shaft, -y_shaft, None]
            strand_poly_x.extend(poly_x)
            strand_poly_y.extend(poly_y)

            seq_snip = seg["sequence"] if len(seg["sequence"]) <= 28 else (seg["sequence"][:25] + "...")
            s_txt = (
                f"<b>{seg['label']} • β-Strand (➔)</b><br>"
                f"Residues: {seg['start']}–{seg['end']} ({seg['length']} aa)<br>"
                f"Mean Confidence: {seg['mean_conf'] * 100:.1f}%<br>"
                f"Sequence: <code>{seq_snip}</code>"
            )
            for r_pos in range(v0, v1 + 1):
                strand_hover_x.append(float(r_pos))
                strand_hover_y.append(0.0)
                strand_hover_txt.append(s_txt)

        if strand_poly_x:
            fig.add_trace(
                go.Scatter(
                    x=strand_poly_x,
                    y=strand_poly_y,
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(217, 119, 6, 0.88)",
                    line=dict(color="#92400E", width=1.5),
                    name="β-Strand (Directional Sheet ➔)",
                    hoverinfo="skip",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=strand_hover_x,
                    y=strand_hover_y,
                    mode="markers",
                    marker=dict(size=12, color="rgba(217, 119, 6, 0.001)"),
                    hoverinfo="text",
                    hovertext=strand_hover_txt,
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
    else:
        # --- MACRO OVERVIEW MODE (span_L > 1,500): Single Batched Polygon Traces & Compact Hover ---
        helix_poly_x: List[Optional[float]] = []
        helix_poly_y: List[Optional[float]] = []
        for seg in helix_segs:
            x0 = float(max(r_start, int(seg["start"]))) - 0.4
            x1 = float(min(r_end, int(seg["end"]))) + 0.4
            helix_poly_x.extend([x0, x1, x1, x0, x0, None])
            helix_poly_y.extend([-0.36, -0.36, 0.36, 0.36, -0.36, None])

        if helix_poly_x:
            fig.add_trace(
                go.Scatter(
                    x=helix_poly_x,
                    y=helix_poly_y,
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(13, 148, 136, 0.86)",
                    line=dict(color="#0F766E", width=0.9),
                    name="α-Helix (Coiled Ribbon)",
                    hoverinfo="skip",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            top_h = (
                sorted(helix_segs, key=lambda item: (item["length"], item["mean_conf"]), reverse=True)[:280]
                if len(helix_segs) > 280
                else helix_segs
            )
            h_hx = [0.5 * (max(r_start, s["start"]) + min(r_end, s["end"])) for s in top_h]
            h_hy = [0.0] * len(top_h)
            h_htxt = [
                (
                    f"<b>{s['label']} • α-Helix</b><br>"
                    f"Residues: {s['start']:,}–{s['end']:,} ({s['length']} aa) • Conf: {s['mean_conf'] * 100:.1f}%"
                )
                for s in top_h
            ]
            fig.add_trace(
                go.Scatter(
                    x=h_hx,
                    y=h_hy,
                    mode="markers",
                    marker=dict(size=9, color="rgba(13, 148, 136, 0.001)"),
                    hoverinfo="text",
                    hovertext=h_htxt,
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        strand_macro_x: List[Optional[float]] = []
        strand_macro_y: List[Optional[float]] = []
        # When span_L > 8,000 and >500 strands (e.g. 34,350-aa Titin with ~1,890 strands where 1 px ~ 35 aa),
        # coalesce sub-pixel beta-hairpin gaps (<= span_L // 7000 aa) in the macro polygon so payload stays < 350 KB
        merge_gap = span_L // 7000 if (span_L > 8000 and len(strand_segs) > 500) else 0
        merged_e_intervals: List[Tuple[float, float]] = []
        for seg in strand_segs:
            sx0 = float(max(r_start, int(seg["start"]))) - 0.4
            sx1 = float(min(r_end, int(seg["end"]))) + 0.4
            if merge_gap > 0 and merged_e_intervals and (sx0 - merged_e_intervals[-1][1]) <= merge_gap:
                merged_e_intervals[-1] = (merged_e_intervals[-1][0], sx1)
            else:
                merged_e_intervals.append((sx0, sx1))

        for x0, x1 in merged_e_intervals:
            strand_macro_x.extend([x0, x1, x1, x0, x0, None])
            strand_macro_y.extend([-0.30, -0.30, 0.30, 0.30, -0.30, None])

        if strand_macro_x:
            fig.add_trace(
                go.Scatter(
                    x=strand_macro_x,
                    y=strand_macro_y,
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(217, 119, 6, 0.85)",
                    line=dict(color="#B45309", width=0.8),
                    name="β-Strand (Directional Sheet ➔)",
                    hoverinfo="skip",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            top_e = (
                sorted(strand_segs, key=lambda item: (item["length"], item["mean_conf"]), reverse=True)[:180]
                if len(strand_segs) > 180
                else strand_segs
            )
            e_hx = [0.5 * (max(r_start, s["start"]) + min(r_end, s["end"])) for s in top_e]
            e_hy = [0.0] * len(top_e)
            e_htxt = [
                (
                    f"<b>{s['label']} • β-Strand (➔)</b><br>"
                    f"Residues: {s['start']:,}–{s['end']:,} ({s['length']} aa) • Conf: {s['mean_conf'] * 100:.1f}%"
                )
                for s in top_e
            ]
            fig.add_trace(
                go.Scatter(
                    x=e_hx,
                    y=e_hy,
                    mode="markers",
                    marker=dict(size=9, color="rgba(217, 119, 6, 0.001)"),
                    hoverinfo="text",
                    hovertext=e_htxt,
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

    # --- TOP TRACK: 4. Anti-Collision Element Labels (α1..αN, β1..βM) ---
    structured_segs = [s for s in vis_segs if s["type"] in ("H", "E")]
    max_badges = 22 if is_macro_span else 28
    if len(structured_segs) > max_badges:
        keep_ids = {
            id(s) for s in sorted(structured_segs, key=lambda item: (item["length"], item["mean_conf"]), reverse=True)[:max_badges]
        }
        labeled_segs = [s for s in structured_segs if id(s) in keep_ids]
    else:
        labeled_segs = structured_segs

    min_dx = max(3.5, float(span_L) * 0.042)
    tier_ys = [0.66, 0.94, -0.66]
    tier_last_x = [-1e9, -1e9, -1e9]

    for seg in labeled_segs:
        x_mid = 0.5 * (float(max(r_start, seg["start"])) + float(min(r_end, seg["end"])))
        chosen_tier = 0
        for t_idx in range(len(tier_ys)):
            if x_mid - tier_last_x[t_idx] >= min_dx:
                chosen_tier = t_idx
                break
        else:
            chosen_tier = int(np.argmax([x_mid - lx for lx in tier_last_x]))

        tier_last_x[chosen_tier] = x_mid
        y_lbl = tier_ys[chosen_tier]

        if seg["type"] == "H":
            txt_color = "#065F46"   # 7.29:1 WCAG AAA on #ECFDF5
            bg_color = "rgba(236, 253, 245, 0.95)"
            bd_color = "#99F6E4"
        else:
            txt_color = "#92400E"   # 6.84:1 WCAG AA/AAA on #FFFBEB
            bg_color = "rgba(255, 251, 235, 0.95)"
            bd_color = "#FCD34D"

        fig.add_annotation(
            x=x_mid,
            y=y_lbl,
            text=f"<b>{seg['label']}</b>",
            showarrow=False,
            font=dict(family=PLOTLY_FONT_FAMILY, size=10.5, color=txt_color),
            bgcolor=bg_color,
            bordercolor=bd_color,
            borderwidth=1,
            borderpad=2,
            row=1,
            col=1,
        )

    # --- BOTTOM TRACK: Calibrated Probability Curves (Exact Per-Residue <=1,500 aa; Bin-Averaged >1,500 aa) ---
    idx0 = r_start - 1
    idx1 = r_end
    if not is_macro_span:
        x_curve = list(range(r_start, r_end + 1))
        yh_curve = prob_h[idx0:idx1]
        ye_curve = prob_e[idx0:idx1]
        yc_curve = prob_c[idx0:idx1]
        state_labels = {"H": "α-Helix (H)", "E": "β-Strand (E)", "C": "Coil / Turn (C)"}
        custom_hover = [
            (
                f"<b>Residue {pos}: {aa}</b> • Predicted: <b>{state_labels[st]}</b><br>"
                f"P(α-Helix): {ph * 100:.1f}%  |  P(β-Strand): {pe * 100:.1f}%  |  P(Coil): {pc * 100:.1f}%"
            )
            for pos, aa, st, ph, pe, pc in zip(
                x_curve, seq[idx0:idx1], ss_str[idx0:idx1], yh_curve, ye_curve, yc_curve
            )
        ]
    else:
        n_bins = min(span_L, 600)
        edges = np.linspace(0, span_L, n_bins + 1, dtype=np.int64)
        counts = np.maximum(1, np.diff(edges)).astype(np.float64)
        ph_sub = np.asarray(prob_h[idx0:idx1], dtype=np.float64)
        pe_sub = np.asarray(prob_e[idx0:idx1], dtype=np.float64)
        pc_sub = np.asarray(prob_c[idx0:idx1], dtype=np.float64)

        yh_arr = np.round(np.add.reduceat(ph_sub, edges[:-1]) / counts, 4)
        ye_arr = np.round(np.add.reduceat(pe_sub, edges[:-1]) / counts, 4)
        yc_arr = np.round(np.add.reduceat(pc_sub, edges[:-1]) / counts, 4)

        b_starts = (r_start + edges[:-1]).tolist()
        b_ends = (r_start + edges[1:] - 1).tolist()
        x_curve = [0.5 * (bs + be) for bs, be in zip(b_starts, b_ends)]
        yh_curve = yh_arr.tolist()
        ye_curve = ye_arr.tolist()
        yc_curve = yc_arr.tolist()
        custom_hover = [
            (
                f"<b>Residues {bs:,}–{be:,}</b><br>"
                f"Mean P(α-Helix): {ph * 100:.1f}%  |  P(β-Strand): {pe * 100:.1f}%  |  P(Coil): {pc * 100:.1f}%"
            )
            for bs, be, ph, pe, pc in zip(b_starts, b_ends, yh_curve, ye_curve, yc_curve)
        ]

    fig.add_trace(
        go.Scatter(
            x=x_curve,
            y=yh_curve,
            mode="lines",
            name="α-Helix Probability P(H)",
            line=dict(color="#0D9488", width=2.3 if not is_macro_span else 1.8),
            fill="tozeroy",
            fillcolor="rgba(13, 148, 136, 0.16)",
            hoverinfo="text",
            hovertext=custom_hover,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x_curve,
            y=ye_curve,
            mode="lines",
            name="β-Strand Probability P(E)",
            line=dict(color="#D97706", width=2.3 if not is_macro_span else 1.8),
            fill="tozeroy",
            fillcolor="rgba(217, 119, 6, 0.16)",
            hoverinfo="text" if not is_macro_span else "y+name",
            hovertext=custom_hover if not is_macro_span else None,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x_curve,
            y=yc_curve,
            mode="lines",
            name="Coil / Turn Probability P(C)",
            line=dict(color="#94A3B8", width=1.6 if not is_macro_span else 1.3, dash="dash"),
            hoverinfo="text" if not is_macro_span else "y+name",
            hovertext=custom_hover if not is_macro_span else None,
        ),
        row=2,
        col=1,
    )

    # Title & Clean Botanical-Editorial Layout
    clean_title_name = protein_name.strip() if protein_name else "Query Protein"
    if is_subwindow:
        loc_prefix = f"Window {r_start:,}–{r_end:,} ({span_L:,} aa of {L:,} aa)  •  "
    elif is_macro_span:
        loc_prefix = f"{L:,} aa (Macro Overview)  •  "
    else:
        loc_prefix = f"{L} aa  •  "

    subtitle_summary = (
        f"{loc_prefix}{ss_result['fold_class']}  •  "
        f"{ss_result['helix_pct']:.1f}% α-Helix ({ss_result['helix_count']})  |  "
        f"{ss_result['strand_pct']:.1f}% β-Strand ({ss_result['strand_count']})  |  "
        f"{ss_result['coil_pct']:.1f}% Coil"
    )

    fig.update_layout(
        font=dict(family=PLOTLY_FONT_FAMILY, color="#1E352F", size=12),
        title=dict(
            text=f"<b>Secondary Structure Topology & Propensity — {clean_title_name}</b>",
            subtitle=dict(
                text=subtitle_summary,
                font=dict(family=PLOTLY_FONT_FAMILY, size=12, color="#4A635C"),
            ),
            font=PLOTLY_TITLE_FONT,
            x=0.0,
            xanchor="left",
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        template="plotly_white",
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.26,
            xanchor="center",
            x=0.5,
            font=dict(family=PLOTLY_FONT_FAMILY, size=11, color="#1E352F"),
        ),
        margin=dict(l=52, r=28, t=72, b=78),
        height=445,
    )

    # Top Cartoon Track Axes
    fig.update_xaxes(
        range=[r_start - x_pad, r_end + x_pad],
        showgrid=True,
        gridcolor=PLOTLY_GRID_COLOR,
        linecolor="#CBDAD4",
        tickfont=PLOTLY_TICK_FONT,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title=dict(text="Topology", font=PLOTLY_AXIS_TITLE_FONT),
        range=[-0.92, 1.16],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        linecolor="#CBDAD4",
        row=1,
        col=1,
    )

    # Bottom Probability Track Axes (linked x-axis zooms synchronously with Top Track!)
    x_axis_lbl = (
        f"Residue Position in {clean_title_name} (Showing {r_start:,}–{r_end:,} of {L:,} aa)"
        if is_subwindow
        else "Residue Position (N-terminus → C-terminus)"
    )
    fig.update_xaxes(
        title=dict(text=x_axis_lbl, font=PLOTLY_AXIS_TITLE_FONT),
        range=[r_start - x_pad, r_end + x_pad],
        showgrid=True,
        gridcolor=PLOTLY_GRID_COLOR,
        linecolor="#CBDAD4",
        tickfont=PLOTLY_TICK_FONT,
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title=dict(text="State Propensity", font=PLOTLY_AXIS_TITLE_FONT),
        range=[0.0, 1.05],
        tickvals=[0.0, 0.25, 0.50, 0.75, 1.0],
        ticktext=["0.00", "0.25", "0.50", "0.75", "1.00"],
        tickfont=PLOTLY_TICK_FONT,
        showgrid=True,
        gridcolor=PLOTLY_GRID_COLOR,
        zeroline=False,
        linecolor="#CBDAD4",
        row=2,
        col=1,
    )

    return fig
