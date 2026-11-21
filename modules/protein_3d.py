"""
3D Protein Structure & Biophysical Visualization Module
=======================================================
Provides seamless 3D structural acquisition, custom/AI-designed structure file parsing,
interactive WebGL 3Dmol.js molecular visualization, and a 100% offline interactive 3D
Plotly biophysical trajectory explorer — with zero extra pip dependencies beyond
ProPurify Studio's standard stack (numpy, plotly, requests).

Capabilities:
  1. Experimental / AlphaFold 3D Structure Acquisition:
     - `fetch_3d_structure_for_uniprot(uniprot_id, timeout=10.0)`
       Queries AlphaFold DB API (v4/v6) first and gracefully falls back to RCSB PDB
       Search API v2 (sorted by best experimental resolution, capped safely <= 4 MB).
     - `fetch_rcsb_structure_for_uniprot(uniprot_id, timeout=10.0)`
     - `fetch_structure_by_pdb_id(pdb_id, timeout=10.0)`
  2. Custom / AI-Designed PDB & mmCIF Upload Parser:
     - `parse_uploaded_structure_file(file_bytes, filename)`
       Validates `.pdb`, `.ent`, `.cif`, `.mmcif` (plus gzip-compressed `.pdb.gz`/`.cif.gz`)
       outputs from AlphaFold3, AlphaFold2/ColabFold, Boltz-1, RFdiffusion, ESMFold, or
       experimental crystallography/cryo-EM. Detects format, pLDDT confidence scale
       (0-100 and ESMFold 0-1), chains, residues, atoms, bound metals/cofactors,
       disulfide bonds, and C-alpha coordinates.
  3. Interactive WebGL 3Dmol.js HTML Component:
     - `build_3dmol_viewer_html(...)`
       Self-contained HTML+JS using 3Dmol.js (primary + cdnjs fallback), ProPurify Studio
       botanical emerald-teal palette, styles (`cartoon`, `surface`, `stick`, `sphere`),
       color schemes (`ss`, `plddt`, `spectrum`, `hydrophobicity`, `charge`), glowing
       bound metal ions & cofactor sticks, hover residue tooltips, Reset/Spin/PNG controls,
       and safe JSON/script escaping.
  4. 100% Offline Interactive 3D Plotly Explorer:
     - `build_3d_biophysical_backbone_figure(sequence, ss_res, metrics, color_by="ss")`
       Instant sequence-derived 3D C-alpha schematic ribbon/backbone trajectory with true
       right-handed alpha-helix geometry (3.6 res/turn, 1.5 A rise, 2.3 A radius), pleated
       beta-strand geometry (3.3 A/res), compact 3D loop packing, graceful macro-downsampling
       for giant chains (> 1,500 aa such as 34,350-aa Titin), rich per-residue hover
       tooltips, and clear educational labeling distinguishing schematic trajectories from
       experimental atomic coordinates.
"""

import gzip
import html
import json
import math
import os
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import requests

# Eagerly import Scatter3d & Scene schemas so the very first 3D figure call stays < 0.1s
try:
    from plotly.graph_objs import Scatter3d as _PlotlyScatter3d  # noqa: F401
    from plotly.graph_objs.layout import Scene as _PlotlyScene  # noqa: F401
except Exception:
    pass

try:
    from modules.sequence_parser import (
        _http_get as _sp_http_get,
        _normalize_uniprot_input as _sp_normalize_uniprot_input,
    )
except ImportError:
    _sp_http_get = None
    _sp_normalize_uniprot_input = None

try:
    from modules.purification_metrics import KYTE_DOOLITTLE
except ImportError:
    KYTE_DOOLITTLE = {
        "A": 1.8,  "R": -4.5, "N": -3.5, "D": -3.5,
        "C": 2.5,  "E": -3.5, "Q": -3.5, "G": -0.4,
        "H": -3.2, "I": 4.5,  "L": 3.8,  "K": -3.9,
        "M": 1.9,  "F": 2.8,  "P": -1.6, "S": -0.8,
        "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
    }


# ==============================================================================
# 1. CONSTANTS, RESIDUE TABLES & BOTANICAL DESIGN TOKENS
# ==============================================================================

MAX_REMOTE_STRUCTURE_BYTES = 4 * 1024 * 1024    # 4 MB safe cap for remote WebGL fetch
MAX_UPLOAD_STRUCTURE_BYTES = 10 * 1024 * 1024   # 10 MB hard cap for user uploads
MAX_FULL_3D_RESIDUES = 1500                     # Full-res offline 3D threshold; >1500 macro-downsampled

SUPPORTED_3DMOL_STYLES = ("cartoon", "surface", "stick", "sphere")
SUPPORTED_3DMOL_COLOR_SCHEMES = ("ss", "plddt", "spectrum", "hydrophobicity", "charge")
SUPPORTED_OFFLINE_3D_COLOR_MODES = ("ss", "hydrophobicity", "charge")

ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
RCSB_SEARCH_V2_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DOWNLOAD_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

THREEDMOL_PRIMARY_CDN = "https://3Dmol.org/build/3Dmol-min.js"
THREEDMOL_FALLBACK_CDN = "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.1.0/3Dmol-min.js"

# Official UniProtKB primary accession pattern (6- or 10-character canonical accessions)
_UNIPROT_BASE_ACC_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)
_PDB_ID_RE = re.compile(r"^[0-9][A-Z0-9]{3}$")
_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{3}(?:[0-9A-Fa-f]{3})?(?:[0-9A-Fa-f]{2})?$")

# ProPurify Studio Scientific Archival Palette & Publication Typography Tokens
PLOTLY_FONT_FAMILY = "IBM Plex Sans, Helvetica Neue, Arial, sans-serif"
PLOTLY_MONO_FAMILY = "IBM Plex Mono, monospace"
HELIX_COLOR = "#0F4C5C"
STRAND_COLOR = "#9A5B13"
COIL_COLOR = "#5C6B78"
BASIC_COLOR = "#2B4C7E"
ACIDIC_COLOR = "#B42318"
HIS_COLOR = "#6B4C9A"
NEUTRAL_COLOR = "#94A3B8"

SCHEMATIC_3D_DISCLAIMER = (
    "Schematic 3D Secondary-Structure & Property Trajectory "
    "(Sequence-Derived — NOT an experimental atomic fold; load AlphaFold/PDB or upload a .pdb/.cif for true 3D coordinates)"
)

THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Common modified & non-canonical residues found in crystal structures
    "MSE": "M", "SEC": "C", "PYL": "K", "SEP": "S", "TPO": "T",
    "PTR": "Y", "CSO": "C", "CSD": "C", "CME": "C", "KCX": "K",
    "MLY": "K", "HYP": "P", "PCA": "E", "ASX": "B", "GLX": "Z",
    "UNK": "X",
}

ONE_TO_THREE: Dict[str, str] = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
    "Q": "Gln", "E": "Glu", "G": "Gly", "H": "His", "I": "Ile",
    "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
    "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
    "U": "Sec", "O": "Pyl", "B": "Asx", "Z": "Glx", "X": "Unk",
}

WATER_RESNAMES = frozenset({"HOH", "WAT", "DOD", "H2O", "TIP", "TIP3", "SOL"})

METAL_ELEMENTS = frozenset({
    "ZN", "FE", "NI", "MG", "CA", "MN", "CU", "CO", "MO", "CD",
    "HG", "W", "V", "K", "NA", "LI", "BA", "SR", "PT", "AU", "AG", "CR",
})

METAL_RESNAMES = frozenset({
    "ZN", "FE", "FE2", "FE3", "NI", "MG", "CA", "MN", "MN3",
    "CU", "CU1", "CO", "3CO", "MO", "CD", "HG", "K", "NA", "W",
})

COFACTOR_RESNAMES = frozenset({
    "HEM", "HEC", "HEA", "HEB", "HDD", "SRM", "FAD", "FMN",
    "NAD", "NAI", "NAP", "NDP", "PLP", "PMP", "SF4", "FES",
    "F3S", "F4S", "B12", "CNC", "COB", "ADP", "ATP", "AMP",
    "GTP", "GDP", "GMP", "SAM", "SAH", "COA", "ACO", "GSH",
    "GSS", "MGD", "MCN", "F43", "BTN", "TPP", "TDP", "THM",
    "UQ", "UBQ", "CLA", "BCL", "PQQ", "CRO", "GYS", "SYG",
})

BUFFER_ARTIFACT_RESNAMES = frozenset({
    "SO4", "PO4", "GOL", "EDO", "PEG", "PGE", "1PE", "P6G",
    "ACT", "ACY", "TRS", "CL", "BR", "IOD", "DMS", "DMSO",
    "MPD", "CIT", "MES", "EPE", "BME", "DTT", "ACE", "NH2",
    "FMT", "NO3", "SCN", "AZI", "UNX", "UNL", "PG4", "PE4",
})

_AI_MODEL_HINT_RE = re.compile(
    r"(?:alphafold|colabfold|esmfold|boltz|chai[-_ ]?1|omegafold|rosettafold|"
    r"rfdiffusion|plddt|_ma_qa_metric|modelarchive|predicted model|computed structure model)",
    flags=re.IGNORECASE,
)

_FILENAME_AI_HINT_RE = re.compile(
    r"(?:alphafold|colabfold|esmfold|boltz|chai|omegafold|rosettafold|"
    r"plddt|model_v\d|af2|af3|\baf[-_])",
    flags=re.IGNORECASE,
)

_EXPTL_METHOD_RE = re.compile(
    r"(?:X-RAY|DIFFRACTION|ELECTRON MICROSCOPY|CRYO-?EM|SOLUTION NMR|SOLID-STATE NMR|NEUTRON|FIBER)",
    flags=re.IGNORECASE,
)


# ==============================================================================
# 2. HTTP & IDENTIFIER HELPERS
# ==============================================================================

def _http_get(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 10.0,
) -> requests.Response:
    """
    Executes HTTP GET via ProPurify Studio's shared pooled/retrying session when available,
    while seamlessly honoring unittest.mock patches on `requests.get` or `protein_3d._http_get`.
    """
    if _sp_http_get is not None:
        return _sp_http_get(url, headers=headers, timeout=timeout)
    return requests.get(url, headers=headers, timeout=timeout)


def normalize_uniprot_accession(uniprot_id: str) -> str:
    """
    Cleans a raw UniProt accession string (handling FASTA prefixes like 'sp|P00698|LYSC_CHICK',
    URLs, version suffixes like 'P00698.1', and isoform suffixes like 'P04637-2' -> 'P04637').
    Returns empty string if the input is not a valid canonical UniProtKB accession.
    """
    raw = (uniprot_id or "").strip()
    if not raw:
        return ""

    base = ""
    if _sp_normalize_uniprot_input is not None:
        try:
            norm = _sp_normalize_uniprot_input(raw)
            base = str(norm.get("base_accession") or "").strip().upper()
        except Exception:
            base = ""

    if not base:
        token = raw.splitlines()[0].strip().lstrip(">").strip()
        m_url = re.search(
            r"uniprot\.org/(?:uniprotkb|uniprot)/([A-Za-z0-9_.\-]+)",
            token,
            flags=re.IGNORECASE,
        )
        if m_url:
            token = m_url.group(1)
        token = re.sub(r"^(?:uniprotkb|uniprot|id|acc)\s*:\s*", "", token, flags=re.IGNORECASE)
        m_pipe = re.match(r"^(?:sp|tr)\|([A-Za-z0-9\-]+)\|", token, flags=re.IGNORECASE)
        if m_pipe:
            token = m_pipe.group(1)
        else:
            token = token.split()[0]
        token = re.sub(r"\.\d+$", "", token.strip().strip("|,;")).upper()
        base = re.sub(r"-\d+$", "", token)
    else:
        base = re.sub(r"-\d+$", "", base)

    if not _UNIPROT_BASE_ACC_RE.fullmatch(base):
        return ""
    return base


def _extract_response_text_and_size(
    resp: Any,
    max_bytes: int,
) -> Tuple[Optional[str], int]:
    """
    Safely extracts UTF-8/latin-1 text and byte size from a `requests.Response` (or mock),
    enforcing `max_bytes` before loading oversized payloads.
    """
    if resp is None:
        return None, 0

    headers = getattr(resp, "headers", None)
    if isinstance(headers, dict) or hasattr(headers, "get"):
        cl_raw = headers.get("Content-Length") or headers.get("content-length")
        if cl_raw is not None:
            try:
                cl_int = int(str(cl_raw).strip())
                if cl_int > max_bytes:
                    return None, cl_int
            except (TypeError, ValueError):
                pass

    raw_bytes = getattr(resp, "content", None)
    if isinstance(raw_bytes, (bytes, bytearray)) and len(raw_bytes) > 0:
        size = len(raw_bytes)
        if size > max_bytes:
            return None, size
        try:
            text = bytes(raw_bytes).decode("utf-8")
        except UnicodeDecodeError:
            text = bytes(raw_bytes).decode("latin-1", errors="replace")
        return text, size

    raw_text = getattr(resp, "text", None)
    if isinstance(raw_text, str) and raw_text:
        size = len(raw_text.encode("utf-8", errors="ignore"))
        if size > max_bytes:
            return None, size
        return raw_text, size

    return None, 0


# ==============================================================================
# 3. PDB & mmCIF PARSER + pLDDT DETECTOR
# ==============================================================================

def _tokenize_cif_line(line: str) -> List[str]:
    """
    Lightweight, zero-dependency mmCIF data row tokenizer that handles single-quoted ('...')
    and double-quoted ("...") values while preserving unquoted prime characters (e.g. O5').
    """
    tokens: List[str] = []
    n = len(line)
    i = 0
    while i < n:
        while i < n and line[i] in (" ", "\t", "\r", "\n"):
            i += 1
        if i >= n:
            break
        ch = line[i]
        if ch == "#":
            break
        if ch in ("'", '"'):
            quote = ch
            i += 1
            start = i
            while i < n:
                if line[i] == quote and (i + 1 == n or line[i + 1] in (" ", "\t", "\r", "\n")):
                    break
                i += 1
            tokens.append(line[start:i])
            if i < n and line[i] == quote:
                i += 1
        else:
            start = i
            while i < n and line[i] not in (" ", "\t", "\r", "\n"):
                i += 1
            tokens.append(line[start:i])
    return tokens


def _detect_plddt_profile(
    b_values: List[float],
    header_ai_hint: bool,
    experimental_method: Optional[str],
    filename_hint: str = "",
    global_qa_metric: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Classifies whether structure B-factors represent AlphaFold / ESMFold / Boltz-1 pLDDT
    confidence scores (on either 0-100 or 0-1 scale) vs crystallographic temperature factors.
    """
    has_exptl = bool(experimental_method and _EXPTL_METHOD_RE.search(experimental_method))
    fname_ai_hint = bool(filename_hint and _FILENAME_AI_HINT_RE.search(filename_hint))

    if has_exptl and not header_ai_hint:
        mean_b = round(float(np.mean(b_values)), 2) if b_values else None
        return {
            "looks_like_plddt": False,
            "plddt_scale": None,
            "mean_plddt": None,
            "mean_bfactor": mean_b,
            "bfactor_min": round(float(min(b_values)), 2) if b_values else None,
            "bfactor_max": round(float(max(b_values)), 2) if b_values else None,
            "plddt_bands": None,
        }

    if not b_values:
        if global_qa_metric is not None and 0.0 < global_qa_metric <= 100.01:
            scale = "0-1" if global_qa_metric <= 1.001 else "0-100"
            m_plddt = round(global_qa_metric * 100.0 if scale == "0-1" else global_qa_metric, 2)
            return {
                "looks_like_plddt": True,
                "plddt_scale": scale,
                "mean_plddt": m_plddt,
                "mean_bfactor": None,
                "bfactor_min": None,
                "bfactor_max": None,
                "plddt_bands": None,
            }
        return {
            "looks_like_plddt": False,
            "plddt_scale": None,
            "mean_plddt": None,
            "mean_bfactor": None,
            "bfactor_min": None,
            "bfactor_max": None,
            "plddt_bands": None,
        }

    arr = np.asarray(b_values, dtype=np.float64)
    b_min = float(arr.min())
    b_max = float(arr.max())
    b_mean = float(arr.mean())
    is_constant = (b_max - b_min) < 1e-4

    looks_like_plddt = False
    plddt_scale: Optional[str] = None

    if header_ai_hint:
        if 0.0 <= b_min and b_max <= 1.001 and b_max > 0.0:
            looks_like_plddt = True
            plddt_scale = "0-1"
        elif 0.0 <= b_min and b_max <= 100.01 and b_max > 1.0:
            looks_like_plddt = True
            plddt_scale = "0-100"
    else:
        # Stripped/headerless files (common for ColabFold, ESMFold, Boltz-1, RFdiffusion)
        if 0.0 <= b_min and b_max <= 1.001 and b_max > 0.05:
            # Distinguish genuine ESMFold 0-1 pLDDT from constant RFdiffusion placeholders (0.00 / 1.00)
            if (not is_constant) or (fname_ai_hint and b_max > 0.05) or (0.20 < b_mean < 0.999):
                looks_like_plddt = True
                plddt_scale = "0-1"
        elif 0.0 <= b_min and b_max <= 100.01 and b_max > 1.0:
            if fname_ai_hint and b_mean >= 20.0:
                looks_like_plddt = True
                plddt_scale = "0-100"
            elif (not is_constant) and b_max >= 55.0 and b_mean >= 45.0:
                looks_like_plddt = True
                plddt_scale = "0-100"

    mean_plddt: Optional[float] = None
    plddt_bands: Optional[Dict[str, float]] = None
    if looks_like_plddt and plddt_scale is not None:
        vals100 = arr * 100.0 if plddt_scale == "0-1" else arr
        mean_plddt = round(float(vals100.mean()), 2)
        n_v = float(len(vals100))
        plddt_bands = {
            "very_high": round(100.0 * float(np.sum(vals100 >= 90.0)) / n_v, 1),
            "confident": round(100.0 * float(np.sum((vals100 >= 70.0) & (vals100 < 90.0))) / n_v, 1),
            "low": round(100.0 * float(np.sum((vals100 >= 50.0) & (vals100 < 70.0))) / n_v, 1),
            "very_low": round(100.0 * float(np.sum(vals100 < 50.0)) / n_v, 1),
        }

    return {
        "looks_like_plddt": looks_like_plddt,
        "plddt_scale": plddt_scale,
        "mean_plddt": mean_plddt,
        "mean_bfactor": round(b_mean, 2),
        "bfactor_min": round(b_min, 2),
        "bfactor_max": round(b_max, 2),
        "plddt_bands": plddt_bands,
    }


def _parse_pdb_structure(text: str, filename_hint: str = "") -> Dict[str, Any]:
    """Parses PDB format text into structural metadata, HETATM ligands, and CA coordinates."""
    title_parts: List[str] = []
    experimental_method: Optional[str] = None
    resolution_angstrom: Optional[float] = None
    header_ai_hint = False
    disulfide_bonds = 0

    chains_order: List[str] = []
    chains_seen = set()
    polymer_res_seen = set()
    ca_res_seen = set()
    ca_coords: List[List[float]] = []
    ca_bfactors: List[float] = []
    ca_residues: List[Dict[str, Any]] = []
    all_atom_bfactors: List[float] = []
    seq_by_chain: Dict[str, List[str]] = {}

    atom_count = 0
    hetatm_count = 0
    hetatm_ligands_set = set()
    metal_ions_set = set()
    cofactors_set = set()

    model_count = 0
    in_first_model = True
    saw_model_record = False

    for line in text.splitlines():
        rec = line[:6].strip()
        if not rec:
            continue

        if rec == "TITLE":
            t_piece = line[10:].strip()
            if t_piece:
                title_parts.append(t_piece)
            if _AI_MODEL_HINT_RE.search(line):
                header_ai_hint = True
            continue

        if rec in ("HEADER", "COMPND", "SOURCE", "AUTHOR", "REMARK"):
            if _AI_MODEL_HINT_RE.search(line):
                header_ai_hint = True
            if rec == "REMARK" and resolution_angstrom is None and line[6:10].strip() == "2":
                m_res = re.search(r"RESOLUTION\.\s+(\d+(?:\.\d+)?)\s+ANGSTROM", line, flags=re.IGNORECASE)
                if m_res:
                    try:
                        resolution_angstrom = round(float(m_res.group(1)), 2)
                    except ValueError:
                        pass
            continue

        if rec == "EXPDTA":
            val = line[10:].strip()
            if val:
                experimental_method = (experimental_method + " " + val).strip() if experimental_method else val
            if _AI_MODEL_HINT_RE.search(val):
                header_ai_hint = True
            continue

        if rec == "SSBOND":
            disulfide_bonds += 1
            continue

        if rec == "MODEL":
            saw_model_record = True
            model_count += 1
            in_first_model = (model_count == 1)
            continue

        if rec == "ENDMDL":
            if saw_model_record and model_count >= 1:
                in_first_model = False
            continue

        if not in_first_model:
            continue

        if rec == "ATOM":
            if len(line) < 54:
                continue
            alt_loc = line[16:17].strip()
            if alt_loc not in ("", "A", "1"):
                continue

            atom_name = line[12:16].strip()
            res_name = line[17:20].strip().upper()
            chain_id = (line[21:22].strip() or "A")
            res_seq_str = line[22:26].strip()
            ins_code = line[26:27].strip()

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            bfactor: Optional[float] = None
            if len(line) >= 66:
                b_str = line[60:66].strip()
                if b_str:
                    try:
                        bfactor = float(b_str)
                        all_atom_bfactors.append(bfactor)
                    except ValueError:
                        bfactor = None

            atom_count += 1
            if chain_id not in chains_seen:
                chains_seen.add(chain_id)
                chains_order.append(chain_id)

            res_key = (chain_id, res_seq_str, ins_code)
            polymer_res_seen.add(res_key)

            if atom_name == "CA" and res_key not in ca_res_seen:
                ca_res_seen.add(res_key)
                aa1 = THREE_TO_ONE.get(res_name, "X")
                ca_coords.append([round(x, 3), round(y, 3), round(z, 3)])
                if bfactor is not None:
                    ca_bfactors.append(bfactor)
                ca_residues.append({
                    "chain": chain_id,
                    "resseq": res_seq_str,
                    "resname": res_name,
                    "aa": aa1,
                    "bfactor": round(bfactor, 3) if bfactor is not None else None,
                })
                seq_by_chain.setdefault(chain_id, []).append(aa1)

        elif rec == "HETATM":
            if len(line) < 54:
                continue
            alt_loc = line[16:17].strip()
            if alt_loc not in ("", "A", "1"):
                continue
            res_name = line[17:20].strip().upper()
            if res_name in WATER_RESNAMES or not res_name:
                continue
            hetatm_count += 1
            atom_name = line[12:16].strip().upper()
            elem = line[76:78].strip().upper() if len(line) >= 78 else ""
            if not elem:
                elem = re.sub(r"[^A-Z]", "", atom_name)[:2]

            if res_name not in BUFFER_ARTIFACT_RESNAMES:
                hetatm_ligands_set.add(res_name)
            if (elem in METAL_ELEMENTS and res_name in METAL_RESNAMES) or res_name in METAL_RESNAMES:
                metal_ions_set.add(res_name)
            elif elem in METAL_ELEMENTS and res_name in COFACTOR_RESNAMES:
                # e.g., central FE inside HEM
                metal_ions_set.add(elem)
            if res_name in COFACTOR_RESNAMES:
                cofactors_set.add(res_name)

    b_source = ca_bfactors if ca_bfactors else all_atom_bfactors
    plddt_info = _detect_plddt_profile(
        b_values=b_source,
        header_ai_hint=header_ai_hint,
        experimental_method=experimental_method,
        filename_hint=filename_hint,
    )

    title_str = re.sub(r"\s{2,}", " ", " ".join(title_parts)).strip()
    return {
        "format": "pdb",
        "chains": chains_order,
        "chain_count": len(chains_order),
        "residue_count": len(polymer_res_seen),
        "ca_count": len(ca_coords),
        "atom_count": atom_count,
        "hetatm_count": hetatm_count,
        "model_count": max(1, model_count) if (atom_count > 0 or hetatm_count > 0) else 0,
        "ca_coords": ca_coords,
        "ca_bfactors": [round(v, 3) for v in ca_bfactors],
        "ca_residues": ca_residues,
        "sequence_by_chain": {ch: "".join(aas) for ch, aas in seq_by_chain.items()},
        "hetatm_ligands": sorted(hetatm_ligands_set),
        "metal_ions": sorted(metal_ions_set),
        "cofactors": sorted(cofactors_set),
        "disulfide_bonds": disulfide_bonds,
        "resolution_angstrom": resolution_angstrom,
        "experimental_method": experimental_method,
        "title": title_str,
        "is_alphafold": plddt_info["looks_like_plddt"],
        **plddt_info,
    }


def _parse_cif_structure(text: str, filename_hint: str = "") -> Dict[str, Any]:
    """Parses mmCIF (.cif / .mmcif) text into structural metadata, ligands, and CA coordinates."""
    lines = text.splitlines()
    n_lines = len(lines)

    title_str = ""
    experimental_method: Optional[str] = None
    resolution_angstrom: Optional[float] = None
    global_qa_metric: Optional[float] = None
    header_ai_hint = bool(_AI_MODEL_HINT_RE.search(text[:25000]))
    disulfide_bonds = 0

    chains_order: List[str] = []
    chains_seen = set()
    polymer_res_seen = set()
    ca_res_seen = set()
    ca_coords: List[List[float]] = []
    ca_bfactors: List[float] = []
    ca_residues: List[Dict[str, Any]] = []
    all_atom_bfactors: List[float] = []
    seq_by_chain: Dict[str, List[str]] = {}

    atom_count = 0
    hetatm_count = 0
    hetatm_ligands_set = set()
    metal_ions_set = set()
    cofactors_set = set()
    models_seen = set()
    first_model_id: Optional[str] = None

    i = 0
    while i < n_lines:
        raw_line = lines[i]
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        # Single-value key-value metadata lines
        if stripped.startswith("_struct.title"):
            toks = _tokenize_cif_line(stripped)
            if len(toks) >= 2 and toks[1] not in (".", "?"):
                title_str = toks[1].strip()
            elif i + 1 < n_lines and lines[i + 1].startswith(";"):
                # Multiline semicolon-delimited text field
                i += 1
                buf = [lines[i][1:].strip()]
                i += 1
                while i < n_lines and not lines[i].startswith(";"):
                    buf.append(lines[i].strip())
                    i += 1
                title_str = re.sub(r"\s{2,}", " ", " ".join(buf)).strip()
            i += 1
            continue

        if stripped.startswith("_exptl.method"):
            toks = _tokenize_cif_line(stripped)
            if len(toks) >= 2 and toks[1] not in (".", "?"):
                experimental_method = toks[1].strip()
            i += 1
            continue

        if stripped.startswith("_refine.ls_d_res_high") or stripped.startswith("_em_3d_reconstruction.resolution"):
            toks = _tokenize_cif_line(stripped)
            if len(toks) >= 2 and resolution_angstrom is None:
                try:
                    resolution_angstrom = round(float(toks[1]), 2)
                except ValueError:
                    pass
            i += 1
            continue

        if stripped.startswith("_ma_qa_metric_global.metric_value"):
            header_ai_hint = True
            toks = _tokenize_cif_line(stripped)
            if len(toks) >= 2:
                try:
                    global_qa_metric = float(toks[1])
                except ValueError:
                    pass
            i += 1
            continue

        # Parse mmCIF loop_ blocks
        if stripped == "loop_":
            i += 1
            cols: List[str] = []
            while i < n_lines:
                c_line = lines[i].strip()
                if c_line.startswith("_"):
                    cols.append(c_line.split()[0])
                    i += 1
                elif not c_line or c_line.startswith("#"):
                    i += 1
                else:
                    break

            if not cols:
                continue

            col_map = {name: idx for idx, name in enumerate(cols)}
            n_cols = len(cols)

            # 1) _atom_site loop (primary atomic coordinates)
            if cols[0].startswith("_atom_site."):
                idx_group = col_map.get("_atom_site.group_PDB")
                idx_atom = col_map.get("_atom_site.auth_atom_id", col_map.get("_atom_site.label_atom_id"))
                idx_alt = col_map.get("_atom_site.label_alt_id")
                idx_comp = col_map.get("_atom_site.auth_comp_id", col_map.get("_atom_site.label_comp_id"))
                idx_chain = col_map.get("_atom_site.auth_asym_id", col_map.get("_atom_site.label_asym_id"))
                idx_seq = col_map.get("_atom_site.auth_seq_id", col_map.get("_atom_site.label_seq_id"))
                idx_ins = col_map.get("_atom_site.pdbx_PDB_ins_code")
                idx_x = col_map.get("_atom_site.Cartn_x")
                idx_y = col_map.get("_atom_site.Cartn_y")
                idx_z = col_map.get("_atom_site.Cartn_z")
                idx_b = col_map.get("_atom_site.B_iso_or_equiv")
                idx_elem = col_map.get("_atom_site.type_symbol")
                idx_model = col_map.get("_atom_site.pdbx_PDB_model_num")

                row_buf: List[str] = []
                while i < n_lines:
                    row_line = lines[i].strip()
                    if not row_line or row_line.startswith("#"):
                        i += 1
                        continue
                    if row_line == "loop_" or row_line.startswith("_") or row_line.startswith("data_"):
                        break
                    if row_line.startswith(";"):
                        # Skip rare multiline text inside loop
                        i += 1
                        while i < n_lines and not lines[i].startswith(";"):
                            i += 1
                        i += 1
                        continue

                    toks = _tokenize_cif_line(row_line)
                    i += 1
                    if not toks:
                        continue
                    row_buf.extend(toks)
                    while len(row_buf) >= n_cols:
                        row = row_buf[:n_cols]
                        row_buf = row_buf[n_cols:]

                        model_id = row[idx_model] if idx_model is not None else "1"
                        models_seen.add(model_id)
                        if first_model_id is None:
                            first_model_id = model_id
                        if model_id != first_model_id:
                            continue

                        if idx_alt is not None and row[idx_alt] not in (".", "?", "", "A", "1"):
                            continue

                        group_pdb = row[idx_group].upper() if idx_group is not None else "ATOM"
                        res_name = row[idx_comp].upper() if idx_comp is not None else "UNK"
                        chain_id = row[idx_chain] if (idx_chain is not None and row[idx_chain] not in (".", "?")) else "A"
                        atom_name = row[idx_atom] if idx_atom is not None else ""
                        res_seq_str = row[idx_seq] if idx_seq is not None else "0"
                        ins_code = ""
                        if idx_ins is not None and row[idx_ins] not in (".", "?"):
                            ins_code = row[idx_ins]

                        if group_pdb == "ATOM":
                            if idx_x is None or idx_y is None or idx_z is None:
                                continue
                            try:
                                x = float(row[idx_x])
                                y = float(row[idx_y])
                                z = float(row[idx_z])
                            except ValueError:
                                continue

                            bfactor: Optional[float] = None
                            if idx_b is not None and row[idx_b] not in (".", "?"):
                                try:
                                    bfactor = float(row[idx_b])
                                    all_atom_bfactors.append(bfactor)
                                except ValueError:
                                    bfactor = None

                            atom_count += 1
                            if chain_id not in chains_seen:
                                chains_seen.add(chain_id)
                                chains_order.append(chain_id)

                            res_key = (chain_id, res_seq_str, ins_code)
                            polymer_res_seen.add(res_key)

                            if atom_name == "CA" and res_key not in ca_res_seen:
                                ca_res_seen.add(res_key)
                                aa1 = THREE_TO_ONE.get(res_name, "X")
                                ca_coords.append([round(x, 3), round(y, 3), round(z, 3)])
                                if bfactor is not None:
                                    ca_bfactors.append(bfactor)
                                ca_residues.append({
                                    "chain": chain_id,
                                    "resseq": res_seq_str,
                                    "resname": res_name,
                                    "aa": aa1,
                                    "bfactor": round(bfactor, 3) if bfactor is not None else None,
                                })
                                seq_by_chain.setdefault(chain_id, []).append(aa1)

                        elif group_pdb == "HETATM":
                            if res_name in WATER_RESNAMES or res_name in (".", "?", ""):
                                continue
                            hetatm_count += 1
                            elem = row[idx_elem].upper() if idx_elem is not None else ""
                            if res_name not in BUFFER_ARTIFACT_RESNAMES:
                                hetatm_ligands_set.add(res_name)
                            if (elem in METAL_ELEMENTS and res_name in METAL_RESNAMES) or res_name in METAL_RESNAMES:
                                metal_ions_set.add(res_name)
                            elif elem in METAL_ELEMENTS and res_name in COFACTOR_RESNAMES:
                                metal_ions_set.add(elem)
                            if res_name in COFACTOR_RESNAMES:
                                cofactors_set.add(res_name)
                continue

            # 2) _struct_conn loop (disulfide bonds)
            if cols[0].startswith("_struct_conn.") and "_struct_conn.conn_type_id" in col_map:
                idx_ct = col_map["_struct_conn.conn_type_id"]
                while i < n_lines:
                    row_line = lines[i].strip()
                    if not row_line or row_line.startswith("#"):
                        i += 1
                        continue
                    if row_line == "loop_" or row_line.startswith("_") or row_line.startswith("data_"):
                        break
                    toks = _tokenize_cif_line(row_line)
                    if idx_ct < len(toks) and toks[idx_ct].lower() == "disulf":
                        disulfide_bonds += 1
                    i += 1
                continue

            # 3) _ma_qa_metric_global loop (AlphaFold3 / ModelArchive global pLDDT)
            if cols[0].startswith("_ma_qa_metric_global."):
                header_ai_hint = True
                idx_mv = col_map.get("_ma_qa_metric_global.metric_value")
                while i < n_lines:
                    row_line = lines[i].strip()
                    if not row_line or row_line.startswith("#"):
                        i += 1
                        continue
                    if row_line == "loop_" or row_line.startswith("_") or row_line.startswith("data_"):
                        break
                    toks = _tokenize_cif_line(row_line)
                    if idx_mv is not None and idx_mv < len(toks) and global_qa_metric is None:
                        try:
                            global_qa_metric = float(toks[idx_mv])
                        except ValueError:
                            pass
                    i += 1
                continue

            # 4) _exptl loop
            if cols[0].startswith("_exptl.") and "_exptl.method" in col_map:
                idx_em = col_map["_exptl.method"]
                while i < n_lines:
                    row_line = lines[i].strip()
                    if not row_line or row_line.startswith("#"):
                        i += 1
                        continue
                    if row_line == "loop_" or row_line.startswith("_") or row_line.startswith("data_"):
                        break
                    toks = _tokenize_cif_line(row_line)
                    if idx_em < len(toks) and experimental_method is None:
                        experimental_method = toks[idx_em].strip()
                    i += 1
                continue

            continue

        i += 1

    b_source = ca_bfactors if ca_bfactors else all_atom_bfactors
    plddt_info = _detect_plddt_profile(
        b_values=b_source,
        header_ai_hint=header_ai_hint,
        experimental_method=experimental_method,
        filename_hint=filename_hint,
        global_qa_metric=global_qa_metric,
    )

    return {
        "format": "cif",
        "chains": chains_order,
        "chain_count": len(chains_order),
        "residue_count": len(polymer_res_seen),
        "ca_count": len(ca_coords),
        "atom_count": atom_count,
        "hetatm_count": hetatm_count,
        "model_count": len(models_seen) if models_seen else (1 if (atom_count > 0 or hetatm_count > 0) else 0),
        "ca_coords": ca_coords,
        "ca_bfactors": [round(v, 3) for v in ca_bfactors],
        "ca_residues": ca_residues,
        "sequence_by_chain": {ch: "".join(aas) for ch, aas in seq_by_chain.items()},
        "hetatm_ligands": sorted(hetatm_ligands_set),
        "metal_ions": sorted(metal_ions_set),
        "cofactors": sorted(cofactors_set),
        "disulfide_bonds": disulfide_bonds,
        "resolution_angstrom": resolution_angstrom,
        "experimental_method": experimental_method,
        "title": title_str,
        "is_alphafold": plddt_info["looks_like_plddt"],
        **plddt_info,
    }


def summarize_structure_text(
    structure_text: str,
    fmt: str = "pdb",
    filename_hint: str = "",
) -> Dict[str, Any]:
    """
    Summarizes a PDB or mmCIF structure text string: chains, residues, atoms, HETATM ligands,
    bound metal ions/cofactors, disulfide bonds, pLDDT confidence detection, and C-alpha coordinates.
    """
    fmt_norm = "cif" if str(fmt or "").strip().lower() in ("cif", "mmcif") else "pdb"
    if fmt_norm == "cif":
        return _parse_cif_structure(structure_text or "", filename_hint=filename_hint)
    return _parse_pdb_structure(structure_text or "", filename_hint=filename_hint)


def _detect_structure_format(text: str, filename: str) -> Optional[str]:
    """Detects whether uploaded text is `'pdb'` or `'cif'` from extension and content."""
    lower_name = (filename or "").strip().lower()
    if lower_name.endswith(".gz"):
        lower_name = lower_name[:-3]
    ext = os.path.splitext(lower_name)[1]

    sample = text[:20000]
    has_cif_markers = ("_atom_site." in sample) or bool(re.search(r"(?m)^data_\S+", sample))
    has_pdb_markers = bool(re.search(r"(?m)^(?:ATOM  |HETATM)", sample))

    if ext in (".cif", ".mmcif"):
        if has_cif_markers:
            return "cif"
        if has_pdb_markers:
            return "pdb"
        return "cif"
    if ext in (".pdb", ".ent"):
        if has_pdb_markers:
            return "pdb"
        if has_cif_markers:
            return "cif"
        return "pdb"

    # Fallback: content-only sniff if no extension was supplied
    if not ext:
        if has_cif_markers:
            return "cif"
        if has_pdb_markers:
            return "pdb"
    return None


def parse_uploaded_structure_file(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Validates and parses a user-uploaded `.pdb`, `.ent`, `.cif`, or `.mmcif` (optionally `.gz`)
    3D structure file — including experimental structures and AI-designed models from
    AlphaFold3, AlphaFold2/ColabFold, Boltz-1, RFdiffusion, or ESMFold.

    Returns a clean dictionary with `valid: bool`, `error: Optional[str]`, `format` (`"pdb"` or `"cif"`),
    `data` (decoded structure text), `looks_like_plddt`, `plddt_scale` (`"0-100"` | `"0-1"` | None),
    `mean_plddt`, `chains`, `chain_count`, `residue_count`, `atom_count`, `hetatm_ligands`,
    `metal_ions`, `cofactors`, `ca_coords`, `ca_residues`, `sequence_by_chain`, and `warnings`.
    Never raises unhandled exceptions on malformed uploads.
    """
    clean_name = os.path.basename((filename or "").strip()) or "uploaded_structure"

    def _invalid(msg: str) -> Dict[str, Any]:
        return {
            "valid": False,
            "error": msg,
            "warnings": [],
            "filename": clean_name,
            "source": f"Uploaded File ({clean_name})",
            "format": None,
            "data": "",
            "size_bytes": len(file_bytes) if isinstance(file_bytes, (bytes, bytearray)) else 0,
            "chains": [],
            "chain_count": 0,
            "residue_count": 0,
            "ca_count": 0,
            "atom_count": 0,
            "hetatm_count": 0,
            "model_count": 0,
            "ca_coords": [],
            "ca_bfactors": [],
            "ca_residues": [],
            "sequence_by_chain": {},
            "hetatm_ligands": [],
            "metal_ions": [],
            "cofactors": [],
            "disulfide_bonds": 0,
            "resolution_angstrom": None,
            "experimental_method": None,
            "title": "",
            "looks_like_plddt": False,
            "is_alphafold": False,
            "plddt_scale": None,
            "mean_plddt": None,
            "mean_bfactor": None,
            "bfactor_min": None,
            "bfactor_max": None,
            "plddt_bands": None,
        }

    if not isinstance(file_bytes, (bytes, bytearray)) or len(file_bytes) == 0:
        return _invalid("Uploaded file is empty. Please upload a valid .pdb, .cif, or .mmcif file.")

    raw_bytes = bytes(file_bytes)
    lower_name = clean_name.lower()

    # Validate file extension before heavy decoding
    ext_check_name = lower_name[:-3] if lower_name.endswith(".gz") else lower_name
    ext = os.path.splitext(ext_check_name)[1]
    if ext not in (".pdb", ".ent", ".cif", ".mmcif"):
        return _invalid(
            f"Unsupported structure file extension '{ext or '(none)'}'. "
            "Please upload a .pdb, .cif, or .mmcif file (AlphaFold3, Boltz-1, RFdiffusion, ESMFold, or RCSB PDB)."
        )

    # Transparently decompress gzip-compressed PDB/mmCIF uploads
    if raw_bytes[:2] == b"\x1f\x8b" or lower_name.endswith(".gz"):
        try:
            raw_bytes = gzip.decompress(raw_bytes)
        except Exception:
            return _invalid("Could not decompress gzipped structure file. Please verify the file is intact.")

    size_bytes = len(raw_bytes)
    if size_bytes == 0:
        return _invalid("Decompressed structure file is empty.")
    if size_bytes > MAX_UPLOAD_STRUCTURE_BYTES:
        mb = size_bytes / (1024 * 1024)
        cap_mb = MAX_UPLOAD_STRUCTURE_BYTES / (1024 * 1024)
        return _invalid(
            f"Structure file is too large ({mb:.1f} MB; limit is {cap_mb:.0f} MB) for smooth browser WebGL rendering."
        )

    # Reject binary non-text files (e.g., renamed PNG/ZIP/BCIF)
    if b"\x00" in raw_bytes[:4096]:
        return _invalid("Binary file detected. Please upload a plain-text .pdb, .cif, or .mmcif coordinate file.")

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1", errors="replace")

    fmt = _detect_structure_format(text, clean_name)
    if fmt not in ("pdb", "cif"):
        return _invalid("Could not recognize valid PDB or mmCIF atomic coordinate records in this file.")

    summary = summarize_structure_text(text, fmt=fmt, filename_hint=clean_name)
    if summary["atom_count"] == 0 and summary["hetatm_count"] == 0:
        return _invalid(
            f"No valid ATOM or HETATM coordinate records were found in '{clean_name}'. "
            "Verify the file is a complete .pdb or .cif/.mmcif coordinate file."
        )

    warnings: List[str] = []
    if size_bytes > MAX_REMOTE_STRUCTURE_BYTES:
        warnings.append(
            f"Large structure ({size_bytes / (1024 * 1024):.1f} MB): browser WebGL rotation or surface rendering may be slower."
        )
    if summary["model_count"] > 1:
        warnings.append(
            f"Multi-model ensemble detected ({summary['model_count']} models); summary statistics reflect Model 1."
        )
    if summary["plddt_scale"] == "0-1":
        warnings.append(
            "Detected 0–1 pLDDT confidence scale (typical of ESMFold); scores were normalized to 0–100 for display and coloring."
        )

    return {
        "valid": True,
        "error": None,
        "warnings": warnings,
        "filename": clean_name,
        "source": f"Uploaded {fmt.upper()} ({clean_name})",
        "data": text,
        "size_bytes": size_bytes,
        **summary,
    }


# ==============================================================================
# 4. REMOTE 3D STRUCTURE ACQUISITION (ALPHAFOLD DB + RCSB PDB FALLBACK)
# ==============================================================================

def _select_best_alphafold_entry(entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Selects primary F1 model (or longest valid fragment with pdbUrl/cifUrl) from AlphaFold DB API response."""
    valid_entries = [
        e for e in entries
        if isinstance(e, dict) and (e.get("pdbUrl") or e.get("cifUrl"))
    ]
    if not valid_entries:
        return None
    for e in valid_entries:
        eid = str(e.get("entryId") or "")
        if eid.endswith("-F1"):
            return e

    def _span(entry: Dict[str, Any]) -> int:
        try:
            s = int(entry.get("sequenceStart") or 1)
            en = int(entry.get("sequenceEnd") or 1)
            return max(0, en - s)
        except (TypeError, ValueError):
            return 0

    return max(valid_entries, key=_span)


def _fetch_alphafold_structure(
    base_acc: str,
    raw_acc: str,
    timeout: float = 10.0,
    max_bytes: int = MAX_REMOTE_STRUCTURE_BYTES,
) -> Optional[Dict[str, Any]]:
    """Fetches a predicted 3D model for a canonical UniProt accession from EBI AlphaFold DB."""
    try:
        meta_url = ALPHAFOLD_API_URL.format(acc=urllib.parse.quote(base_acc))
        resp = _http_get(meta_url, headers={"Accept": "application/json"}, timeout=timeout)
        if resp is None or getattr(resp, "status_code", 0) != 200:
            return None

        payload = resp.json()
        if isinstance(payload, dict):
            entries = [payload]
        elif isinstance(payload, list):
            entries = payload
        else:
            return None

        entry = _select_best_alphafold_entry(entries)
        if entry is None:
            return None

        pdb_url = entry.get("pdbUrl")
        cif_url = entry.get("cifUrl")
        if pdb_url:
            download_url = str(pdb_url).strip()
            fmt = "pdb"
        elif cif_url:
            download_url = str(cif_url).strip()
            fmt = "cif"
        else:
            return None

        dl_resp = _http_get(download_url, headers={"Accept": "text/plain, */*"}, timeout=timeout)
        if dl_resp is None or getattr(dl_resp, "status_code", 0) != 200:
            return None

        struct_text, size_bytes = _extract_response_text_and_size(dl_resp, max_bytes=max_bytes)
        if not struct_text:
            return None

        entry_id = str(entry.get("entryId") or f"AF-{base_acc}-F1").strip()
        summary = summarize_structure_text(struct_text, fmt=fmt, filename_hint=f"{entry_id}.{fmt}")
        if summary["atom_count"] == 0 and summary["hetatm_count"] == 0:
            return None

        # Prefer authoritative globalMetricValue from AlphaFold DB API if numeric; else CA B-factor mean
        mean_plddt = summary["mean_plddt"]
        gmv = entry.get("globalMetricValue")
        if gmv is not None:
            try:
                gmv_val = float(gmv)
                if 0.0 < gmv_val <= 1.0:
                    mean_plddt = round(gmv_val * 100.0, 2)
                elif 1.0 < gmv_val <= 100.0:
                    mean_plddt = round(gmv_val, 2)
            except (TypeError, ValueError):
                pass

        latest_ver = entry.get("latestVersion")
        method_label = f"AlphaFold2 (AlphaFold DB v{latest_ver})" if latest_ver else "AlphaFold DB Prediction"
        title = str(entry.get("uniprotDescription") or summary["title"] or f"AlphaFold Model {entry_id}").strip()
        seq_start = entry.get("sequenceStart")
        seq_end = entry.get("sequenceEnd")
        frag_range = (int(seq_start), int(seq_end)) if (seq_start and seq_end) else None

        return {
            "source": "AlphaFold DB",
            "format": fmt,
            "data": struct_text,
            "accession": base_acc,
            "raw_accession": raw_acc,
            "entry_id": entry_id,
            "pdb_id": None,
            "mean_plddt": mean_plddt,
            "plddt_scale": summary["plddt_scale"] or "0-100",
            "plddt_bands": summary["plddt_bands"],
            "looks_like_plddt": True,
            "is_alphafold": True,
            "url": download_url,
            "entry_url": f"https://alphafold.ebi.ac.uk/entry/{base_acc}",
            "resolution_angstrom": None,
            "experimental_method": method_label,
            "title": title,
            "chains": summary["chains"],
            "chain_count": summary["chain_count"],
            "residue_count": summary["residue_count"],
            "atom_count": summary["atom_count"],
            "hetatm_count": summary["hetatm_count"],
            "hetatm_ligands": summary["hetatm_ligands"],
            "metal_ions": summary["metal_ions"],
            "cofactors": summary["cofactors"],
            "disulfide_bonds": summary["disulfide_bonds"],
            "size_bytes": size_bytes,
            "is_fragment": len(entries) > 1,
            "fragment_range": frag_range,
        }
    except Exception:
        return None


def fetch_structure_by_pdb_id(
    pdb_id: str,
    timeout: float = 10.0,
    max_bytes: int = MAX_REMOTE_STRUCTURE_BYTES,
    accession: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Downloads an experimental `.pdb` structure directly from RCSB PDB (`files.rcsb.org`)
    given a 4-character PDB ID (e.g. '2VB1', '1MBN', '6D0T'), enforcing `max_bytes` <= 4 MB.
    Returns `None` cleanly on invalid ID, 404, oversize file, or network error.
    """
    clean_pdb = (pdb_id or "").strip().upper()
    if not _PDB_ID_RE.fullmatch(clean_pdb):
        return None

    try:
        pdb_url = RCSB_DOWNLOAD_PDB_URL.format(pdb_id=clean_pdb)
        resp = _http_get(pdb_url, headers={"Accept": "text/plain, */*"}, timeout=timeout)
        if resp is None or getattr(resp, "status_code", 0) != 200:
            return None

        pdb_text, size_bytes = _extract_response_text_and_size(resp, max_bytes=max_bytes)
        if not pdb_text:
            return None

        summary = summarize_structure_text(pdb_text, fmt="pdb", filename_hint=f"{clean_pdb}.pdb")
        if summary["atom_count"] == 0 and summary["hetatm_count"] == 0:
            return None

        return {
            "source": "RCSB PDB",
            "format": "pdb",
            "data": pdb_text,
            "accession": accession or clean_pdb,
            "raw_accession": accession or clean_pdb,
            "entry_id": clean_pdb,
            "pdb_id": clean_pdb,
            "mean_plddt": None,
            "plddt_scale": None,
            "plddt_bands": None,
            "looks_like_plddt": False,
            "is_alphafold": False,
            "url": pdb_url,
            "entry_url": f"https://www.rcsb.org/structure/{clean_pdb}",
            "resolution_angstrom": summary["resolution_angstrom"],
            "experimental_method": summary["experimental_method"] or "Experimental PDB",
            "title": summary["title"] or f"RCSB PDB Entry {clean_pdb}",
            "chains": summary["chains"],
            "chain_count": summary["chain_count"],
            "residue_count": summary["residue_count"],
            "atom_count": summary["atom_count"],
            "hetatm_count": summary["hetatm_count"],
            "hetatm_ligands": summary["hetatm_ligands"],
            "metal_ions": summary["metal_ions"],
            "cofactors": summary["cofactors"],
            "disulfide_bonds": summary["disulfide_bonds"],
            "size_bytes": size_bytes,
            "is_fragment": False,
            "fragment_range": None,
        }
    except Exception:
        return None


def fetch_rcsb_structure_for_uniprot(
    uniprot_id: str,
    timeout: float = 10.0,
    max_bytes: int = MAX_REMOTE_STRUCTURE_BYTES,
) -> Optional[Dict[str, Any]]:
    """
    Searches RCSB PDB Search API v2 for experimental structures mapped to a UniProt accession
    (sorted by best experimental resolution) and downloads the best-resolution `.pdb` <= `max_bytes`.
    Returns `None` cleanly on 204/404, oversize files, or network failure.
    """
    base_acc = normalize_uniprot_accession(uniprot_id)
    if not base_acc:
        return None

    query_obj = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                        "operator": "exact_match",
                        "value": base_acc,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
                        "operator": "exact_match",
                        "value": "UniProt",
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "sort": [
                {
                    "sort_by": "rcsb_entry_info.resolution_combined",
                    "direction": "asc",
                }
            ],
            "paginate": {"start": 0, "rows": 5},
        },
    }

    try:
        encoded = urllib.parse.quote(json.dumps(query_obj, separators=(",", ":")))
        search_url = f"{RCSB_SEARCH_V2_URL}?json={encoded}"
        resp = _http_get(search_url, headers={"Accept": "application/json"}, timeout=timeout)
        if resp is None or getattr(resp, "status_code", 0) != 200:
            return None
        # RCSB returns HTTP 204 or empty text when there are 0 hits
        raw_txt = getattr(resp, "text", "")
        if isinstance(raw_txt, str) and not raw_txt.strip():
            # Allow mocks that only implement .json() without setting .text
            if not callable(getattr(resp, "json", None)):
                return None

        payload = resp.json()
        result_set = payload.get("result_set") if isinstance(payload, dict) else None
        if not isinstance(result_set, list) or not result_set:
            return None

        for hit in result_set[:5]:
            pdb_id = str((hit or {}).get("identifier") or "").strip().upper()
            if not _PDB_ID_RE.fullmatch(pdb_id):
                continue
            res = fetch_structure_by_pdb_id(
                pdb_id=pdb_id,
                timeout=timeout,
                max_bytes=max_bytes,
                accession=base_acc,
            )
            if res is not None:
                res["raw_accession"] = (uniprot_id or "").strip()
                return res
        return None
    except Exception:
        return None


def fetch_3d_structure_for_uniprot(
    uniprot_id: str,
    timeout: float = 10.0,
    prefer_source: str = "alphafold",
) -> Optional[Dict[str, Any]]:
    """
    Acquires a 3D atomic structure for a UniProt accession (automatically stripping isoform
    suffixes such as '-1' / '-2' for canonical base lookup):
      - Default (`prefer_source="alphafold"`): tries AlphaFold DB API first, then gracefully
        falls back to RCSB PDB Search API v2 (best-resolution `.pdb`, capped <= 4 MB).
      - Optional (`prefer_source="rcsb"`): tries experimental RCSB PDB first (great for
        visualizing bound Heme / Zn2+ / Ni2+ cofactors), then falls back to AlphaFold DB.

    Never raises on timeouts, 404s, 204s, malformed payloads, or offline environments;
    returns `None` cleanly so the UI smoothly switches to the 100% offline 3D Explorer.
    """
    raw_acc = (uniprot_id or "").strip()
    base_acc = normalize_uniprot_accession(raw_acc)
    if not base_acc:
        return None

    pref = (prefer_source or "alphafold").strip().lower()
    if pref in ("rcsb", "pdb", "experimental"):
        res = fetch_rcsb_structure_for_uniprot(base_acc, timeout=timeout, max_bytes=MAX_REMOTE_STRUCTURE_BYTES)
        if res is not None:
            res["raw_accession"] = raw_acc
            return res
        return _fetch_alphafold_structure(base_acc, raw_acc, timeout=timeout, max_bytes=MAX_REMOTE_STRUCTURE_BYTES)

    res = _fetch_alphafold_structure(base_acc, raw_acc, timeout=timeout, max_bytes=MAX_REMOTE_STRUCTURE_BYTES)
    if res is not None:
        return res
    res_rcsb = fetch_rcsb_structure_for_uniprot(base_acc, timeout=timeout, max_bytes=MAX_REMOTE_STRUCTURE_BYTES)
    if res_rcsb is not None:
        res_rcsb["raw_accession"] = raw_acc
    return res_rcsb


# ==============================================================================
# 5. INTERACTIVE WEBGL 3DMOL.JS HTML COMPONENT
# ==============================================================================

def _safe_js_json(obj: Any) -> str:
    """
    Serializes a Python object into a JSON literal safe for inline embedding inside an
    HTML `<script>` block (prevents `</script>` breakout, HTML comments, and JS line-separator bugs).
    """
    return (
        json.dumps(obj, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("<!--", "<\\!--")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _build_3dmol_legend_html(color_scheme: str, show_hetatms: bool) -> str:
    """Builds a compact WCAG-readable botanical overlay legend matching the active color scheme."""
    items: List[Tuple[str, str]] = []
    if color_scheme == "ss":
        items = [
            ("#0D9488", "α-Helix"),
            ("#D97706", "β-Sheet"),
            ("#64748B", "Coil / Loop"),
        ]
    elif color_scheme == "plddt":
        items = [
            ("#0053D6", "pLDDT > 90"),
            ("#65CBF3", "70–90"),
            ("#FFDB13", "50–70"),
            ("#FF7D45", "< 50"),
        ]
    elif color_scheme == "spectrum":
        items = [
            ("#2563EB", "N-term"),
            ("#0D9488", "Core"),
            ("#D97706", "→"),
            ("#DC2626", "C-term"),
        ]
    elif color_scheme == "hydrophobicity":
        items = [
            ("#0D9488", "Polar (-4.5)"),
            ("#CBD5E1", "Neutral (0)"),
            ("#EA580C", "Hydrophobic (+4.5)"),
        ]
    elif color_scheme == "charge":
        items = [
            ("#2563EB", "Basic (+ K/R)"),
            ("#7C3AED", "His (pKa~6)"),
            ("#DC2626", "Acidic (− D/E)"),
            ("#94A3B8", "Neutral"),
        ]

    chips = "".join(
        f'<span class="pp3d-leg-item"><span class="pp3d-dot" style="background:{color};"></span>{html.escape(label)}</span>'
        for color, label in items
    )
    if show_hetatms:
        chips += (
            '<span class="pp3d-leg-sep"></span>'
            '<span class="pp3d-leg-item">'
            '<span class="pp3d-dot pp3d-dot-glow" style="background:#EC4899;"></span>'
            'Cofactors / Metals</span>'
        )
    return chips


def build_3dmol_viewer_html(
    structure_text: str,
    fmt: str = "pdb",
    style: str = "cartoon",
    color_scheme: str = "ss",
    bg_color: str = "#F8FBF9",
    spin: bool = False,
    show_hetatms: bool = True,
    height_px: int = 440,
) -> str:
    """
    Returns a clean, self-contained HTML+JS WebGL 3Dmol.js molecular viewer component styled
    in ProPurify Studio's botanical emerald-teal palette.

    Parameters
    ----------
    structure_text : str
        Full PDB or mmCIF text payload.
    fmt : str
        `"pdb"` or `"cif"` (`"mmcif"` accepted and normalized to `"cif"`).
    style : str
        One of `"cartoon"`, `"surface"`, `"stick"`, `"sphere"`.
    color_scheme : str
        One of `"ss"`, `"plddt"`, `"spectrum"`, `"hydrophobicity"`, `"charge"`.
    bg_color : str
        Hex background color (default `"#F8FBF9"`).
    spin : bool
        Enable smooth Y-axis autorotation on load.
    show_hetatms : bool
        Highlight bound non-water cofactors (Heme, FAD, NAD, etc.) and glowing metal ions (Zn2+, Fe2+, Ni2+).
    height_px : int
        Viewport height in pixels (clamped to [240, 1200]).
    """
    safe_height = max(240, min(1200, int(height_px)))
    safe_bg = bg_color.strip() if (isinstance(bg_color, str) and _HEX_COLOR_RE.fullmatch(bg_color.strip())) else "#F8FBF9"

    fmt_clean = "cif" if str(fmt or "").strip().lower() in ("cif", "mmcif") else "pdb"

    style_clean = str(style or "cartoon").strip().lower()
    if style_clean not in SUPPORTED_3DMOL_STYLES:
        style_clean = "cartoon"

    scheme_clean = str(color_scheme or "ss").strip().lower()
    if scheme_clean not in SUPPORTED_3DMOL_COLOR_SCHEMES:
        scheme_clean = "ss"

    spin_bool = bool(spin)
    het_bool = bool(show_hetatms)

    if not isinstance(structure_text, str) or not structure_text.strip():
        return (
            f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
            f'<style>body{{margin:0;padding:0;font-family:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;background:transparent;}}'
            f'.pp3d-empty{{height:{safe_height}px;display:flex;flex-direction:column;align-items:center;justify-content:center;'
            f'background:{safe_bg};border:1px solid #DCE2E6;border-radius:8px;color:#2C3A47;text-align:center;padding:24px;box-sizing:border-box;}}'
            f'.pp3d-empty h4{{margin:0 0 8px 0;color:#111827;font-size:15px;}}'
            f'.pp3d-empty p{{margin:0;color:#52606D;font-size:13px;max-width:420px;line-height:1.5;}}</style></head>'
            f'<body><div class="pp3d-empty"><h4>No 3D Atomic Structure Loaded</h4>'
            f'<p>Load an AlphaFold DB / RCSB PDB structure, upload a .pdb/.cif file, or explore the 100% Offline 3D Trajectory.</p>'
            f'</div></body></html>'
        )

    config_payload = {
        "structure": structure_text,
        "format": fmt_clean,
        "style": style_clean,
        "colorScheme": scheme_clean,
        "bgColor": safe_bg,
        "spin": spin_bool,
        "showHetatms": het_bool,
        "primaryCdn": THREEDMOL_PRIMARY_CDN,
        "fallbackCdn": THREEDMOL_FALLBACK_CDN,
    }
    config_js = _safe_js_json(config_payload)
    legend_html = _build_3dmol_legend_html(scheme_clean, het_bool)
    spin_label = "Pause Spin" if spin_bool else "Spin"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  html, body {{
    margin: 0;
    padding: 0;
    width: 100%;
    height: {safe_height}px;
    overflow: hidden;
    background: transparent;
    font-family: 'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif;
  }}
  .pp3d-shell {{
    position: relative;
    width: 100%;
    height: {safe_height}px;
    box-sizing: border-box;
    border: 1px solid #DCE2E6;
    border-radius: 8px;
    background: {safe_bg};
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(17, 24, 39, 0.04);
  }}
  #pp3d-viewport {{
    position: relative;
    width: 100%;
    height: 100%;
  }}
  .pp3d-toolbar {{
    position: absolute;
    top: 10px;
    right: 10px;
    z-index: 20;
    display: flex;
    gap: 6px;
    align-items: center;
  }}
  .pp3d-btn {{
    appearance: none;
    border: 1px solid #C6DDD5;
    background: rgba(255, 255, 255, 0.92);
    color: #0F2D25;
    font-family: inherit;
    font-size: 11.5px;
    font-weight: 600;
    padding: 5px 10px;
    border-radius: 8px;
    cursor: pointer;
    backdrop-filter: blur(4px);
    transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
  }}
  .pp3d-btn:hover {{
    background: #ECFDF5;
    border-color: #0D9488;
    color: #065F46;
  }}
  .pp3d-btn.active {{
    background: #0D9488;
    border-color: #0F766E;
    color: #FFFFFF;
  }}
  .pp3d-legend {{
    position: absolute;
    left: 10px;
    bottom: 10px;
    z-index: 20;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    max-width: calc(100% - 24px);
    padding: 6px 10px;
    border-radius: 9px;
    background: rgba(255, 255, 255, 0.90);
    border: 1px solid #D5E5DF;
    backdrop-filter: blur(4px);
    font-size: 11px;
    font-weight: 600;
    color: #1E352F;
  }}
  .pp3d-leg-item {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    white-space: nowrap;
  }}
  .pp3d-dot {{
    width: 9px;
    height: 9px;
    border-radius: 999px;
    display: inline-block;
    border: 1px solid rgba(11, 34, 28, 0.16);
  }}
  .pp3d-dot-glow {{
    box-shadow: 0 0 6px rgba(236, 72, 153, 0.65);
  }}
  .pp3d-leg-sep {{
    width: 1px;
    height: 12px;
    background: #CBDAD4;
    display: inline-block;
  }}
  .pp3d-hover-pill {{
    position: absolute;
    top: 10px;
    left: 10px;
    z-index: 20;
    padding: 5px 10px;
    border-radius: 8px;
    background: rgba(11, 34, 28, 0.88);
    color: #F8FAFC;
    font-size: 11.5px;
    font-weight: 500;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.12s ease;
    max-width: 62%;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .pp3d-hover-pill.visible {{
    opacity: 1;
  }}
  .pp3d-status {{
    position: absolute;
    inset: 0;
    z-index: 30;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 24px;
    text-align: center;
    background: {safe_bg};
    color: #1E352F;
  }}
  .pp3d-status.hidden {{
    display: none;
  }}
  .pp3d-status-card {{
    max-width: 460px;
    background: #FFFFFF;
    border: 1px solid #D5E5DF;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 4px 16px rgba(11, 34, 28, 0.06);
  }}
  .pp3d-status-title {{
    margin: 0 0 6px 0;
    font-size: 14.5px;
    font-weight: 700;
    color: #0B221C;
  }}
  .pp3d-status-msg {{
    margin: 0;
    font-size: 12.5px;
    line-height: 1.55;
    color: #47625B;
  }}
</style>
</head>
<body>
<div class="pp3d-shell">
  <div id="pp3d-viewport"></div>
  <div id="pp3d-hover" class="pp3d-hover-pill"></div>
  <div class="pp3d-toolbar">
    <button type="button" id="pp3d-reset" class="pp3d-btn" title="Re-center 3D structure">Reset View</button>
    <button type="button" id="pp3d-spin" class="pp3d-btn{' active' if spin_bool else ''}" title="Toggle gentle rotation">{spin_label}</button>
    <button type="button" id="pp3d-snap" class="pp3d-btn" title="Save PNG image">PNG</button>
  </div>
  <div class="pp3d-legend">{legend_html}</div>
  <div id="pp3d-status" class="pp3d-status">
    <div class="pp3d-status-card">
      <h4 id="pp3d-status-title" class="pp3d-status-title">Initializing Interactive 3D Viewer…</h4>
      <p id="pp3d-status-msg" class="pp3d-status-msg">Loading 3Dmol.js WebGL molecular renderer…</p>
    </div>
  </div>
</div>
<script>
(function() {{
  var CFG = {config_js};
  var statusEl = document.getElementById("pp3d-status");
  var statusTitle = document.getElementById("pp3d-status-title");
  var statusMsg = document.getElementById("pp3d-status-msg");
  var hoverEl = document.getElementById("pp3d-hover");
  var spinBtn = document.getElementById("pp3d-spin");
  var resetBtn = document.getElementById("pp3d-reset");
  var snapBtn = document.getElementById("pp3d-snap");

  var KD_MAP = {{
    ILE: 4.5, VAL: 4.2, LEU: 3.8, PHE: 2.8, CYS: 2.5,
    MET: 1.9, MSE: 1.9, ALA: 1.8, GLY: -0.4, THR: -0.7,
    SER: -0.8, TRP: -0.9, TYR: -1.3, PRO: -1.6, HIS: -3.2,
    GLU: -3.5, GLN: -3.5, ASP: -3.5, ASN: -3.5, LYS: -3.9, ARG: -4.5
  }};

  var WATER_SET = {{ HOH: 1, WAT: 1, DOD: 1, H2O: 1, TIP: 1, TIP3: 1, SOL: 1 }};
  var METAL_COLORS = {{
    ZN: 0x7C3AED, FE: 0xEA580C, NI: 0x10B981, MG: 0x22C55E,
    CA: 0x14B8A6, MN: 0xA855F7, CU: 0x0284C7, CO: 0xEC4899,
    MO: 0x0EA5E9, CD: 0x8B5CF6, W: 0x6366F1, V: 0x06B6D4,
    K: 0x9333EA, NA: 0x6366F1
  }};
  var HEME_SET = {{ HEM: 1, HEC: 1, HEA: 1, HEB: 1, HDD: 1, SRM: 1 }};

  function showError(title, msg) {{
    statusTitle.textContent = title;
    statusMsg.textContent = msg;
    statusEl.classList.remove("hidden");
  }}

  function hideStatus() {{
    statusEl.classList.add("hidden");
  }}

  function lerpRgbHex(c1, c2, t) {{
    var u = Math.max(0, Math.min(1, t));
    var r1 = (c1 >> 16) & 255, g1 = (c1 >> 8) & 255, b1 = c1 & 255;
    var r2 = (c2 >> 16) & 255, g2 = (c2 >> 8) & 255, b2 = c2 & 255;
    var r = Math.round(r1 + (r2 - r1) * u);
    var g = Math.round(g1 + (g2 - g1) * u);
    var b = Math.round(b1 + (b2 - b1) * u);
    return (r << 16) | (g << 8) | b;
  }}

  function spectrumColor(t) {{
    var u = Math.max(0, Math.min(1, t));
    var stops = [0x2563EB, 0x0D9488, 0x10B981, 0xF59E0B, 0xDC2626];
    var scaled = u * (stops.length - 1);
    var idx = Math.min(stops.length - 2, Math.floor(scaled));
    return lerpRgbHex(stops[idx], stops[idx + 1], scaled - idx);
  }}

  function buildColorFunc(scheme, meta) {{
    if (scheme === "ss") {{
      return function(atom) {{
        var s = (atom && atom.ss ? String(atom.ss).toLowerCase() : "c");
        if (s === "h" || s === "g" || s === "i") return 0x0D9488;
        if (s === "s" || s === "e" || s === "b") return 0xD97706;
        return 0x64748B;
      }};
    }}
    if (scheme === "plddt") {{
      var scale = meta.bIsUnit ? 100.0 : 1.0;
      return function(atom) {{
        var b = (atom && typeof atom.b === "number") ? atom.b * scale : 0;
        if (b >= 90) return 0x0053D6;
        if (b >= 70) return 0x65CBF3;
        if (b >= 50) return 0xFFDB13;
        return 0xFF7D45;
      }};
    }}
    if (scheme === "spectrum") {{
      var minR = meta.minResi;
      var span = Math.max(1, meta.maxResi - meta.minResi);
      return function(atom) {{
        var r = (atom && typeof atom.resi === "number") ? atom.resi : minR;
        return spectrumColor((r - minR) / span);
      }};
    }}
    if (scheme === "hydrophobicity") {{
      return function(atom) {{
        var rn = (atom && atom.resn) ? String(atom.resn).toUpperCase() : "";
        var kd = (rn in KD_MAP) ? KD_MAP[rn] : 0.0;
        if (kd < 0) {{
          return lerpRgbHex(0x0D9488, 0xE2ECE8, (kd + 4.5) / 4.5);
        }}
        return lerpRgbHex(0xE2ECE8, 0xEA580C, kd / 4.5);
      }};
    }}
    if (scheme === "charge") {{
      return function(atom) {{
        var rn = (atom && atom.resn) ? String(atom.resn).toUpperCase() : "";
        if (rn === "ARG" || rn === "LYS") return 0x2563EB;
        if (rn === "ASP" || rn === "GLU") return 0xDC2626;
        if (rn === "HIS") return 0x7C3AED;
        return 0x94A3B8;
      }};
    }}
    return function() {{ return 0x0D9488; }};
  }}

  function initViewer() {{
    if (typeof window.$3Dmol === "undefined") {{
      showError(
        "3Dmol.js WebGL Library Unavailable (Offline / CDN Blocked)",
        "Could not load 3Dmol.js from primary or fallback CDN. Use the 100% Offline 3D Trajectory Explorer tab to inspect this protein without internet access."
      );
      return;
    }}
    try {{
      var testCanvas = document.createElement("canvas");
      var gl = testCanvas.getContext("webgl") || testCanvas.getContext("experimental-webgl");
      if (!gl) {{
        showError(
          "WebGL Unavailable in Browser",
          "Your browser or GPU settings disabled WebGL acceleration. Switch to the Offline 3D Trajectory view to explore the structure interactively."
        );
        return;
      }}

      var vp = document.getElementById("pp3d-viewport");
      var viewer = window.$3Dmol.createViewer(vp, {{
        backgroundColor: CFG.bgColor,
        antialias: true
      }});

      var model = viewer.addModel(CFG.structure, CFG.format);
      var atoms = model.selectedAtoms({{ hetflag: false }});
      if (!atoms || atoms.length === 0) {{
        atoms = model.selectedAtoms({{}});
      }}

      var minResi = Infinity, maxResi = -Infinity, maxB = 0;
      for (var i = 0; i < atoms.length; i++) {{
        var a = atoms[i];
        if (typeof a.resi === "number") {{
          if (a.resi < minResi) minResi = a.resi;
          if (a.resi > maxResi) maxResi = a.resi;
        }}
        if (typeof a.b === "number" && a.b > maxB) {{
          maxB = a.b;
        }}
      }}
      if (!isFinite(minResi)) {{ minResi = 1; maxResi = 100; }}

      var meta = {{
        minResi: minResi,
        maxResi: maxResi,
        bIsUnit: (maxB > 0 && maxB <= 1.01)
      }};
      var cfunc = buildColorFunc(CFG.colorScheme, meta);

      viewer.setStyle({{}}, {{}});
      var polySel = {{ hetflag: false }};

      if (CFG.style === "cartoon") {{
        viewer.setStyle(polySel, {{
          cartoon: {{ colorfunc: cfunc, arrows: true, thickness: 0.42 }}
        }});
      }} else if (CFG.style === "surface") {{
        viewer.setStyle(polySel, {{
          cartoon: {{ colorfunc: cfunc, arrows: true, thickness: 0.36 }}
        }});
        viewer.addSurface(
          window.$3Dmol.SurfaceType.VDW,
          {{ opacity: 0.68, colorfunc: cfunc }},
          polySel
        );
      }} else if (CFG.style === "stick") {{
        viewer.setStyle(polySel, {{
          cartoon: {{ colorfunc: cfunc, opacity: 0.72, arrows: true }},
          stick: {{ colorfunc: cfunc, radius: 0.16 }}
        }});
      }} else if (CFG.style === "sphere") {{
        viewer.setStyle(polySel, {{
          sphere: {{ colorfunc: cfunc, scale: 0.85 }}
        }});
      }}

      if (CFG.showHetatms) {{
        var hetAtoms = model.selectedAtoms({{ hetflag: true }});
        var metalList = [];
        var hemePresent = false;
        var otherHetPresent = false;

        for (var j = 0; j < hetAtoms.length; j++) {{
          var ha = hetAtoms[j];
          var rn = (ha.resn || "").toUpperCase();
          if (WATER_SET[rn]) continue;
          var el = (ha.elem || "").toUpperCase();
          if (METAL_COLORS[el]) {{
            metalList.push(ha);
          }} else if (HEME_SET[rn]) {{
            hemePresent = true;
          }} else {{
            otherHetPresent = true;
          }}
        }}

        if (otherHetPresent) {{
          viewer.setStyle(
            {{ hetflag: true, not: {{ resn: ["HOH", "WAT", "DOD", "H2O", "TIP", "TIP3", "SOL", "HEM", "HEC", "HEA", "HEB", "HDD", "SRM"] }} }},
            {{
              stick: {{ radius: 0.22, colorscheme: "magentaCarbon" }},
              sphere: {{ scale: 0.28, colorscheme: "magentaCarbon" }}
            }}
          );
        }}

        if (hemePresent) {{
          viewer.setStyle(
            {{ hetflag: true, resn: ["HEM", "HEC", "HEA", "HEB", "HDD", "SRM"] }},
            {{
              stick: {{ radius: 0.25, colorscheme: "orangeCarbon" }},
              sphere: {{ scale: 0.30, colorscheme: "orangeCarbon" }}
            }}
          );
        }}

        for (var m = 0; m < metalList.length; m++) {{
          var ma = metalList[m];
          var mel = (ma.elem || "").toUpperCase();
          var mcol = METAL_COLORS[mel] || 0xF59E0B;
          viewer.setStyle(
            {{ serial: ma.serial }},
            {{ sphere: {{ radius: 1.05, color: mcol }} }}
          );
          viewer.addSphere({{
            center: {{ x: ma.x, y: ma.y, z: ma.z }},
            radius: 1.65,
            color: mcol,
            alpha: 0.26
          }});
        }}
      }}

      viewer.setHoverable(
        {{}},
        true,
        function(atom) {{
          if (!atom) return;
          var rn = (atom.resn || "").toUpperCase();
          if (atom.hetflag) {{
            if (WATER_SET[rn]) return;
            hoverEl.textContent = "Ligand / Cofactor: " + rn + " (" + (atom.elem || atom.atom) + ") • Chain " + (atom.chain || "A") + " #" + atom.resi;
          }} else {{
            var ssMap = {{ h: "α-Helix", s: "β-Sheet", c: "Coil/Loop" }};
            var ssLabel = ssMap[(atom.ss || "c").toLowerCase()] || "Coil/Loop";
            var bStr = "";
            if (typeof atom.b === "number") {{
              var bVal = meta.bIsUnit ? (atom.b * 100.0) : atom.b;
              bStr = (CFG.colorScheme === "plddt" ? " • pLDDT " : " • B ") + bVal.toFixed(1);
            }}
            hoverEl.textContent = rn + " " + atom.resi + " (Chain " + (atom.chain || "A") + ") • " + ssLabel + bStr;
          }}
          hoverEl.classList.add("visible");
        }},
        function() {{
          hoverEl.classList.remove("visible");
        }}
      );

      var spinning = !!CFG.spin;
      viewer.zoomTo({{ hetflag: false }});
      if (atoms.length === 0) viewer.zoomTo();
      viewer.render();
      if (spinning) viewer.spin("y", 0.45);
      hideStatus();

      resetBtn.addEventListener("click", function() {{
        viewer.zoomTo({{ hetflag: false }});
        viewer.render();
      }});

      spinBtn.addEventListener("click", function() {{
        spinning = !spinning;
        if (spinning) {{
          viewer.spin("y", 0.45);
          spinBtn.textContent = "Pause Spin";
          spinBtn.classList.add("active");
        }} else {{
          viewer.spin(false);
          spinBtn.textContent = "Spin";
          spinBtn.classList.remove("active");
        }}
      }});

      snapBtn.addEventListener("click", function() {{
        try {{
          var uri = viewer.pngURI();
          var link = document.createElement("a");
          link.href = uri;
          link.download = "propurify_3d_structure.png";
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
        }} catch (e) {{}}
      }});
    }} catch (err) {{
      showError(
        "3D Structure Rendering Error",
        "Unable to render WebGL structure (" + (err && err.message ? err.message : "unknown error") + "). Switch to the Offline 3D Trajectory view."
      );
    }}
  }}

  function loadScript(src, onLoad, onError) {{
    var s = document.createElement("script");
    s.src = src;
    s.async = true;
    var done = false;
    var timer = setTimeout(function() {{
      if (!done) {{
        done = true;
        onError();
      }}
    }}, 9000);
    s.onload = function() {{
      if (!done) {{
        done = true;
        clearTimeout(timer);
        onLoad();
      }}
    }};
    s.onerror = function() {{
      if (!done) {{
        done = true;
        clearTimeout(timer);
        onError();
      }}
    }};
    document.head.appendChild(s);
  }}

  if (typeof window.$3Dmol !== "undefined") {{
    initViewer();
  }} else {{
    loadScript(
      CFG.primaryCdn,
      function() {{
        if (typeof window.$3Dmol !== "undefined") {{
          initViewer();
        }} else {{
          loadScript(CFG.fallbackCdn, initViewer, initViewer);
        }}
      }},
      function() {{
        loadScript(CFG.fallbackCdn, initViewer, initViewer);
      }}
    );
  }}
}})();
</script>
</body>
</html>"""


# ==============================================================================
# 6. 100% OFFLINE INTERACTIVE 3D PLOTLY BIOPHYSICAL EXPLORER
# ==============================================================================

def _orthonormalize_frame(t_vec: np.ndarray, n_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds a right-handed orthonormal frame (T, N, B) via Gram-Schmidt."""
    t_norm = float(np.linalg.norm(t_vec))
    T = (t_vec / t_norm) if t_norm > 1e-8 else np.array([1.0, 0.0, 0.0], dtype=np.float64)

    n_proj = n_vec - float(np.dot(n_vec, T)) * T
    n_norm = float(np.linalg.norm(n_proj))
    if n_norm < 1e-6:
        # Pick cardinal axis least aligned with T
        axes = np.eye(3, dtype=np.float64)
        idx = int(np.argmin(np.abs(axes @ T)))
        n_proj = axes[idx] - float(np.dot(axes[idx], T)) * T
        n_norm = float(np.linalg.norm(n_proj))
    N = n_proj / n_norm
    B = np.cross(T, N)
    return T, N, B


def _generate_schematic_3d_backbone(
    states: List[str],
    helix_substeps: int = 3,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, int, int]]]:
    """
    Constructs a deterministic, compact 3D C-alpha-inspired schematic backbone path directly
    from predicted 3-state secondary structure (`H`, `E`, `C`):
      - `H` (alpha-helix): true right-handed Pauling-Corey geometry (3.6 res/turn, 1.5 A rise,
        2.3 A radius; exact ~3.83 A virtual Ca-Ca step).
      - `E` (beta-strand): pleated zig-zag geometry (3.3 A axial advance, +/-0.95 A pleat,
        12 deg/res right-handed twist; exact ~3.81 A virtual Ca-Ca step).
      - `C` (coil/turn/loop): smooth curved transitions (beta-hairpins between adjacent strands,
        golden-angle 3D turns + soft centripetal packing so globular proteins remain compact).

    Returns:
      - `node_coords`: (M, 3) float64 array of node positions centered at the origin.
      - `curve_coords`: (K, 3) float64 array of smooth ribbon vertices (including helix substeps).
      - `curve_segments`: list of `(state, curve_start_idx, curve_end_idx_exclusive)` spans.
    """
    M = len(states)
    if M == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64), []

    # Group contiguous SS runs
    runs: List[Tuple[str, int, int]] = []
    idx = 0
    while idx < M:
        st = states[idx]
        end = idx + 1
        while end < M and states[end] == st:
            end += 1
        runs.append((st, idx, end))
        idx = end

    node_coords = np.zeros((M, 3), dtype=np.float64)
    curve_chunks: List[np.ndarray] = []
    curve_segments: List[Tuple[str, int, int]] = []
    curve_cursor = 0

    P = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    T = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    N = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    B = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    coord_sum = np.zeros(3, dtype=np.float64)
    coord_count = 0
    r_glob = max(14.0, 4.8 * (float(M) ** (1.0 / 3.0)))
    golden_angle = 2.399963229728653  # 137.507764 deg
    turn_idx = 0

    dtheta_helix = 2.0 * math.pi / 3.6  # 100 deg/res
    helix_rise = 1.5                    # Angstrom/res
    helix_radius = 2.3                  # Angstrom Ca cylinder radius
    strand_step = 3.3                   # Angstrom/res
    strand_pleat = 0.95                 # Angstrom alternating pleat
    strand_twist = math.radians(12.0)   # Natural right-handed beta twist
    coil_step = 3.5                     # Angstrom/res along smooth turn

    sub = max(1, int(helix_substeps))

    for r_i, (st, s0, s1) in enumerate(runs):
        k = s1 - s0
        prev_st = runs[r_i - 1][0] if r_i > 0 else None
        next_st = runs[r_i + 1][0] if (r_i + 1) < len(runs) else None

        if st == "H":
            # Right-handed alpha-helix starting smoothly at current point P on the cylinder
            c0 = P - helix_radius * N
            j = np.arange(1, k + 1, dtype=np.float64)
            theta = j * dtheta_helix
            pts = (
                c0[None, :]
                + (j[:, None] * helix_rise) * T[None, :]
                + helix_radius * (np.cos(theta)[:, None] * N[None, :] + np.sin(theta)[:, None] * B[None, :])
            )
            node_coords[s0:s1] = pts
            coord_sum += pts.sum(axis=0)
            coord_count += k

            if sub > 1:
                t_sub = np.linspace(0.0, float(k), k * sub + 1, dtype=np.float64)
                th_sub = t_sub * dtheta_helix
                c_pts = (
                    c0[None, :]
                    + (t_sub[:, None] * helix_rise) * T[None, :]
                    + helix_radius * (np.cos(th_sub)[:, None] * N[None, :] + np.sin(th_sub)[:, None] * B[None, :])
                )
            else:
                c_pts = np.vstack([P[None, :], pts]) if r_i > 0 else pts

            curve_chunks.append(c_pts)
            curve_segments.append(("H", curve_cursor, curve_cursor + len(c_pts)))
            curve_cursor += len(c_pts)

            P = pts[-1].copy()
            th_end = float(k) * dtheta_helix
            n_new = math.cos(th_end) * N + math.sin(th_end) * B
            T, N, B = _orthonormalize_frame(T, n_new)

        elif st == "E":
            # Extended pleated beta-strand with alternating C-alpha displacement and gentle right-handed twist
            j = np.arange(k, dtype=np.float64)
            tw = (j + 1.0) * strand_twist
            signs = np.where((np.arange(k) % 2) == 0, 1.0, -1.0)
            pleat_vecs = (
                np.cos(tw)[:, None] * N[None, :] + np.sin(tw)[:, None] * B[None, :]
            ) * (signs[:, None] * strand_pleat)
            pts = P[None, :] + ((j + 1.0)[:, None] * strand_step) * T[None, :] + pleat_vecs

            node_coords[s0:s1] = pts
            coord_sum += pts.sum(axis=0)
            coord_count += k

            c_pts = np.vstack([P[None, :], pts]) if r_i > 0 else pts
            curve_chunks.append(c_pts)
            curve_segments.append(("E", curve_cursor, curve_cursor + len(c_pts)))
            curve_cursor += len(c_pts)

            P = pts[-1].copy()
            tw_end = float(k) * strand_twist
            n_new = math.cos(tw_end) * N + math.sin(tw_end) * B
            T, N, B = _orthonormalize_frame(T, n_new)

        else:
            # Smooth curved coil/turn/loop; break long disordered coils into gentle arcs
            c_pts_list: List[np.ndarray] = [P.copy()] if r_i > 0 else []
            local_pos = s0
            remaining = k
            is_beta_hairpin = (prev_st == "E" and next_st == "E" and k <= 8)

            while remaining > 0:
                ck = remaining if remaining <= 10 else 8
                turn_idx += 1
                az = turn_idx * golden_angle
                u_orth = math.cos(az) * N + math.sin(az) * B

                centroid = (coord_sum / float(coord_count)) if coord_count > 0 else np.zeros(3, dtype=np.float64)
                r_vec = P - centroid
                r_dist = float(np.linalg.norm(r_vec))

                if is_beta_hairpin:
                    t_target = -0.82 * T + 0.48 * u_orth + 0.30 * B
                else:
                    pull = min(0.86, (r_dist / r_glob) ** 1.35) if r_dist > 1e-4 else 0.0
                    inward = (-r_vec / r_dist) if r_dist > 1e-4 else u_orth
                    explore = 0.25 * T + 0.75 * u_orth
                    t_target = (1.0 - pull) * explore + pull * (0.72 * inward + 0.28 * u_orth)

                tt_norm = float(np.linalg.norm(t_target))
                t_target = (t_target / tt_norm) if tt_norm > 1e-8 else u_orth

                # Avoid exact 180-deg anti-parallel interpolation singularity
                if float(np.dot(T, t_target)) < -0.96:
                    t_target = t_target + 0.35 * u_orth
                    t_target = t_target / float(np.linalg.norm(t_target))

                t_start = T.copy()
                for step_i in range(ck):
                    frac = float(step_i + 1) / float(ck)
                    # Smoothstep easing for natural loop curvature
                    ease = frac * frac * (3.0 - 2.0 * frac)
                    t_cur = (1.0 - ease) * t_start + ease * t_target
                    T, N, B = _orthonormalize_frame(t_cur, N)
                    P = P + coil_step * T
                    node_coords[local_pos] = P
                    coord_sum += P
                    coord_count += 1
                    c_pts_list.append(P.copy())
                    local_pos += 1

                remaining -= ck
                is_beta_hairpin = False

            c_pts = np.asarray(c_pts_list, dtype=np.float64)
            curve_chunks.append(c_pts)
            curve_segments.append(("C", curve_cursor, curve_cursor + len(c_pts)))
            curve_cursor += len(c_pts)

    # Center geometry at the origin for balanced 3D orbit rotation
    center = node_coords.mean(axis=0)
    node_coords = node_coords - center
    curve_coords = np.vstack(curve_chunks) - center if curve_chunks else np.zeros((0, 3), dtype=np.float64)
    return node_coords, curve_coords, curve_segments


def _normalize_color_by(color_by: str) -> str:
    """Normalizes offline 3D color mode aliases into `'ss'`, `'hydrophobicity'`, or `'charge'`."""
    val = str(color_by or "ss").strip().lower()
    if val in ("ss", "secondary", "secondary_structure", "structure"):
        return "ss"
    if val in ("hydrophobicity", "hydropathy", "hydro", "kd", "kyte_doolittle", "gravy"):
        return "hydrophobicity"
    if val in ("charge", "electrostatic", "electrostatics", "ionic"):
        return "charge"
    return "ss"


def build_3d_biophysical_backbone_figure(
    sequence: str,
    ss_res: Dict[str, Any],
    metrics: Dict[str, Any],
    color_by: str = "ss",
    protein_name: str = "",
    height_px: int = 480,
) -> go.Figure:
    """
    Generates a 100% offline, interactive 3D Plotly biophysical backbone visualization directly
    from the amino acid sequence, secondary structure prediction (`ss_res`), and physicochemical
    profile (`metrics`). Requires ZERO external CDNs or atomic coordinate files and runs in
    < 50 ms on typical proteins (and < 150 ms on 34,350-aa Titin via automatic vectorized
    macro-downsampling).

    Color modes (`color_by`):
      - `"ss"`: Secondary Structure (`H` Emerald-Teal α-Helix, `E` Warm Amber β-Strand, `C` Slate Coil)
      - `"hydrophobicity"`: Continuous Kyte-Doolittle scale (-4.5 Polar Teal -> +4.5 Hydrophobic Coral)
      - `"charge"`: Side-chain charge at physiological pH (`+` Arg/Lys Blue, `−` Asp/Glu Red, His Violet, Neutral Slate)

    Clearly labels the plot as a sequence-derived schematic trajectory (NOT an experimental atomic fold).
    """
    clean_seq = re.sub(r"[^A-Z]", "", (sequence or "").strip().upper())
    L = len(clean_seq)
    mode = _normalize_color_by(color_by)
    safe_height = max(320, min(900, int(height_px)))

    if L == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No amino acid sequence available for 3D biophysical visualization.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(family=PLOTLY_FONT_FAMILY, size=13, color="#47625B"),
        )
        fig.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FBF9",
            height=safe_height,
            margin=dict(l=16, r=16, t=60, b=32),
        )
        return fig

    ss_dict = ss_res if isinstance(ss_res, dict) else {}
    met_dict = metrics if isinstance(metrics, dict) else {}

    raw_ss = str(ss_dict.get("ss_string") or "").strip().upper()
    if len(raw_ss) != L:
        if len(raw_ss) > L:
            raw_ss = raw_ss[:L]
        else:
            raw_ss = raw_ss + ("C" * (L - len(raw_ss)))

    # Vectorized per-residue biophysical arrays (O(L) pure NumPy for instant Titin speed)
    seq_bytes = np.frombuffer(clean_seq.encode("ascii", errors="ignore"), dtype=np.uint8)
    ss_bytes = np.frombuffer(raw_ss.encode("ascii", errors="ignore"), dtype=np.uint8)

    kd_lut = np.zeros(256, dtype=np.float64)
    for aa_ch, kd_val in KYTE_DOOLITTLE.items():
        kd_lut[ord(aa_ch)] = float(kd_val)
    kd_arr = kd_lut[seq_bytes]

    # Charge class: 1=Basic(K,R), -1=Acidic(D,E), 2=His(H), 0=Neutral
    chg_lut = np.zeros(256, dtype=np.int8)
    chg_lut[ord("K")] = 1
    chg_lut[ord("R")] = 1
    chg_lut[ord("D")] = -1
    chg_lut[ord("E")] = -1
    chg_lut[ord("H")] = 2
    chg_arr = chg_lut[seq_bytes]

    # Per-residue predicted state confidence
    ph_raw = ss_dict.get("prob_h")
    pe_raw = ss_dict.get("prob_e")
    pc_raw = ss_dict.get("prob_c")
    has_probs = (
        isinstance(ph_raw, (list, np.ndarray)) and len(ph_raw) == L
        and isinstance(pe_raw, (list, np.ndarray)) and len(pe_raw) == L
        and isinstance(pc_raw, (list, np.ndarray)) and len(pc_raw) == L
    )
    if has_probs:
        ph_arr = np.asarray(ph_raw, dtype=np.float64)
        pe_arr = np.asarray(pe_raw, dtype=np.float64)
        pc_arr = np.asarray(pc_raw, dtype=np.float64)
        conf_arr = np.where(ss_bytes == ord("H"), ph_arr, np.where(ss_bytes == ord("E"), pe_arr, pc_arr))
    else:
        conf_arr = np.full(L, float(ss_dict.get("mean_confidence") or 0.70), dtype=np.float64)

    # Full-resolution (<= 1,500 aa) vs vectorized macro-downsampling (> 1,500 aa e.g. Titin)
    is_downsampled = L > MAX_FULL_3D_RESIDUES
    state_names = {"H": "α-Helix (H)", "E": "β-Strand (E)", "C": "Coil / Loop (C)"}

    if not is_downsampled:
        node_states = list(raw_ss)
        node_kd = kd_arr
        node_chg = chg_arr
        helix_sub = 3 if L <= 550 else 2

        chg_labels = {
            1: "+1 (Basic — binds CEX)",
            -1: "−1 (Acidic — binds AEX)",
            2: "Partial + (His, pKa ~6.0)",
            0: "0 (Neutral)",
        }
        hover_texts = [
            (
                f"<b>Res {pos}: {ONE_TO_THREE.get(aa, aa)} ({aa})</b><br>"
                f"Predicted SS: <b>{state_names.get(st, 'Coil / Loop (C)')}</b> ({cf * 100.0:.1f}% conf)<br>"
                f"KD Hydropathy: <b>{kd:+.2f}</b> ({'Hydrophobic' if kd > 0.5 else ('Polar/Hydrophilic' if kd < -0.5 else 'Amphipathic/Neutral')})<br>"
                f"Side-Chain Charge (pH 7): {chg_labels[int(cg)]}"
            )
            for pos, aa, st, cf, kd, cg in zip(
                range(1, L + 1), clean_seq, node_states, conf_arr.tolist(), kd_arr.tolist(), chg_arr.tolist()
            )
        ]
        downsample_step = 1
    else:
        # Macro-downsampling into ~950 evenly spaced bins via pure NumPy reduceat (< 5 ms on 34,350 aa!)
        n_bins = 950
        edges = np.linspace(0, L, n_bins + 1, dtype=np.int64)
        counts = np.maximum(1, np.diff(edges)).astype(np.float64)
        starts_idx = edges[:-1]
        ends_idx = edges[1:]

        is_h = (ss_bytes == ord("H")).astype(np.int32)
        is_e = (ss_bytes == ord("E")).astype(np.int32)
        is_c = (ss_bytes == ord("C")).astype(np.int32)

        sum_h = np.add.reduceat(is_h, starts_idx)
        sum_e = np.add.reduceat(is_e, starts_idx)
        sum_c = np.add.reduceat(is_c, starts_idx)

        dom_idx = np.argmax(np.vstack([sum_h, sum_e, sum_c]), axis=0)
        lut_st = ("H", "E", "C")
        node_states = [lut_st[int(d)] for d in dom_idx]

        node_kd = np.add.reduceat(kd_arr, starts_idx) / counts
        node_conf = np.add.reduceat(conf_arr, starts_idx) / counts

        n_pos = np.add.reduceat((chg_arr == 1).astype(np.int32), starts_idx)
        n_neg = np.add.reduceat((chg_arr == -1).astype(np.int32), starts_idx)
        n_his = np.add.reduceat((chg_arr == 2).astype(np.int32), starts_idx)
        net_q = n_pos - n_neg
        # Dominant window charge class
        node_chg = np.where(
            net_q >= 2, 1,
            np.where(net_q <= -2, -1, np.where((n_his >= 2) & (np.abs(net_q) < 2), 2, 0))
        ).astype(np.int8)

        b_s = (starts_idx + 1).tolist()
        b_e = ends_idx.tolist()
        h_pcts = (100.0 * sum_h / counts).tolist()
        e_pcts = (100.0 * sum_e / counts).tolist()
        hover_texts = [
            (
                f"<b>Residues {bs:,}–{be:,}</b> ({be - bs + 1} aa window)<br>"
                f"Dominant SS: <b>{state_names[st]}</b> (H {hp:.0f}% | E {ep:.0f}%) • Conf {cf * 100.0:.1f}%<br>"
                f"Mean KD Hydropathy: <b>{kd:+.2f}</b><br>"
                f"Window Charge: +{p_c} Basic / −{n_c} Acidic / {h_c} His (Net {nq:+d})"
            )
            for bs, be, st, hp, ep, cf, kd, p_c, n_c, h_c, nq in zip(
                b_s, b_e, node_states, h_pcts, e_pcts,
                node_conf.tolist(), node_kd.tolist(),
                n_pos.tolist(), n_neg.tolist(), n_his.tolist(), net_q.tolist()
            )
        ]
        helix_sub = 1
        downsample_step = max(2, int(round(float(L) / float(n_bins))))

    # Build 3D geometry
    node_coords, curve_coords, curve_segments = _generate_schematic_3d_backbone(
        node_states, helix_substeps=helix_sub
    )
    nx = np.round(node_coords[:, 0], 2)
    ny = np.round(node_coords[:, 1], 2)
    nz = np.round(node_coords[:, 2], 2)

    cx = np.round(curve_coords[:, 0], 2)
    cy = np.round(curve_coords[:, 1], 2)
    cz = np.round(curve_coords[:, 2], 2)

    traces: List[Dict[str, Any]] = []

    helix_pct = float(ss_dict.get("helix_pct", 100.0 * raw_ss.count("H") / L))
    strand_pct = float(ss_dict.get("strand_pct", 100.0 * raw_ss.count("E") / L))
    coil_pct = float(ss_dict.get("coil_pct", 100.0 * raw_ss.count("C") / L))

    if mode == "ss":
        # Continuous faint spine underneath so transitions remain seamless when toggling legend traces
        traces.append(
            dict(
                type="scatter3d",
                x=cx, y=cy, z=cz,
                mode="lines",
                name="Backbone Spine",
                line=dict(color="rgba(148, 163, 184, 0.35)", width=2.5),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # Build batched ribbon line + residue marker traces per SS state (H, E, C)
        ss_specs = [
            ("H", f"α-Helix (H) — {helix_pct:.1f}%", HELIX_COLOR, 7.5, 4.4 if not is_downsampled else 3.4),
            ("E", f"β-Strand (E) — {strand_pct:.1f}%", STRAND_COLOR, 7.5, 4.4 if not is_downsampled else 3.4),
            ("C", f"Coil / Loop (C) — {coil_pct:.1f}%", COIL_COLOR, 4.2, 2.9 if not is_downsampled else 2.5),
        ]
        node_states_arr = np.asarray(node_states)
        nan_sep = np.array([np.nan], dtype=np.float64)

        for st_code, st_label, st_col, l_width, m_size in ss_specs:
            # Batched NumPy ribbon path separated by NaN (zero Python-list validator overhead)
            segs_for_st = [(c0, c1) for seg_st, c0, c1 in curve_segments if seg_st == st_code]
            has_ribbon = bool(segs_for_st)
            if has_ribbon:
                parts_x: List[np.ndarray] = []
                parts_y: List[np.ndarray] = []
                parts_z: List[np.ndarray] = []
                for c0, c1 in segs_for_st:
                    parts_x.append(cx[c0:c1])
                    parts_x.append(nan_sep)
                    parts_y.append(cy[c0:c1])
                    parts_y.append(nan_sep)
                    parts_z.append(cz[c0:c1])
                    parts_z.append(nan_sep)
                traces.append(
                    dict(
                        type="scatter3d",
                        x=np.concatenate(parts_x),
                        y=np.concatenate(parts_y),
                        z=np.concatenate(parts_z),
                        mode="lines",
                        name=st_label,
                        legendgroup=f"ss_{st_code}",
                        line=dict(color=st_col, width=l_width),
                        hoverinfo="skip",
                        showlegend=True,
                    )
                )

            mask = np.flatnonzero(node_states_arr == st_code)
            if len(mask) > 0:
                traces.append(
                    dict(
                        type="scatter3d",
                        x=nx[mask],
                        y=ny[mask],
                        z=nz[mask],
                        mode="markers",
                        name=st_label,
                        legendgroup=f"ss_{st_code}",
                        marker=dict(
                            size=m_size,
                            color=st_col,
                            opacity=0.92,
                            line=dict(color="#FFFFFF", width=0.4),
                        ),
                        hoverinfo="text",
                        hovertext=[hover_texts[int(i)] for i in mask],
                        showlegend=(not has_ribbon),
                    )
                )

    elif mode == "hydrophobicity":
        kd_colorscale = [
            [0.00, "#0F766E"],  # -4.5 Deep Polar Teal
            [0.25, "#2DD4BF"],  # -2.25 Mint Teal
            [0.50, "#E2ECE8"],  #  0.00 Neutral Sage-Slate
            [0.75, "#F59E0B"],  # +2.25 Warm Amber
            [1.00, "#DC2626"],  # +4.50 Hydrophobic Coral-Crimson
        ]
        # Smooth continuous backbone line
        traces.append(
            dict(
                type="scatter3d",
                x=cx, y=cy, z=cz,
                mode="lines",
                name="Backbone Path",
                line=dict(color="rgba(100, 116, 139, 0.45)", width=4.5),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        # Residue spheres colored by Kyte-Doolittle hydropathy
        traces.append(
            dict(
                type="scatter3d",
                x=nx, y=ny, z=nz,
                mode="markers",
                name="KD Hydrophobicity",
                marker=dict(
                    size=5.4 if not is_downsampled else 4.0,
                    color=np.round(node_kd, 3),
                    cmin=-4.5,
                    cmax=4.5,
                    colorscale=kd_colorscale,
                    opacity=0.94,
                    line=dict(color="rgba(11, 34, 28, 0.18)", width=0.4),
                    colorbar=dict(
                        title=dict(
                            text="Kyte–Doolittle<br>Hydropathy",
                            font=dict(family=PLOTLY_FONT_FAMILY, size=11, color="#1E352F"),
                        ),
                        tickvals=[-4.5, -2.0, 0.0, 2.0, 4.5],
                        ticktext=["-4.5 Polar", "-2.0", "0.0", "+2.0", "+4.5 Hydrophobic"],
                        tickfont=dict(family=PLOTLY_FONT_FAMILY, size=10, color="#4A635C"),
                        thickness=14,
                        len=0.72,
                        x=0.98,
                    ),
                ),
                hoverinfo="text",
                hovertext=hover_texts,
                showlegend=False,
            )
        )

    else:  # mode == "charge"
        # Continuous subtle backbone ribbon
        traces.append(
            dict(
                type="scatter3d",
                x=cx, y=cy, z=cz,
                mode="lines",
                name="Backbone Path",
                line=dict(color="rgba(148, 163, 184, 0.55)", width=4.4),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        charge_groups = [
            (1, "Basic (+) Arg / Lys", BASIC_COLOR, 6.2 if not is_downsampled else 4.8, 0.96),
            (-1, "Acidic (−) Asp / Glu", ACIDIC_COLOR, 6.2 if not is_downsampled else 4.8, 0.96),
            (2, "Histidine (His, pKa ~6.0)", HIS_COLOR, 5.6 if not is_downsampled else 4.4, 0.94),
            (0, "Neutral Residues", NEUTRAL_COLOR, 3.2 if not is_downsampled else 2.6, 0.58),
        ]
        for cg_val, cg_label, cg_col, cg_sz, cg_op in charge_groups:
            mask = np.flatnonzero(node_chg == cg_val)
            if len(mask) == 0:
                continue
            traces.append(
                dict(
                    type="scatter3d",
                    x=nx[mask],
                    y=ny[mask],
                    z=nz[mask],
                    mode="markers",
                    name=cg_label,
                    marker=dict(
                        size=cg_sz,
                        color=cg_col,
                        opacity=cg_op,
                        line=dict(color="#FFFFFF", width=0.5 if cg_val != 0 else 0.0),
                    ),
                    hoverinfo="text",
                    hovertext=[hover_texts[int(i)] for i in mask],
                    showlegend=True,
                )
            )

    # Prominent N-terminus and C-terminus directional anchor markers
    traces.append(
        dict(
            type="scatter3d",
            x=np.array([nx[0], nx[-1]], dtype=np.float64),
            y=np.array([ny[0], ny[-1]], dtype=np.float64),
            z=np.array([nz[0], nz[-1]], dtype=np.float64),
            mode="markers+text",
            name="N / C Termini",
            text=["N", "C"],
            textposition="top center",
            textfont=dict(family=PLOTLY_MONO_FAMILY, size=11, color="#0B221C"),
            marker=dict(
                size=7.5,
                symbol="diamond",
                color=["#065F46", "#92400E"],
                line=dict(color="#FFFFFF", width=1.5),
            ),
            hoverinfo="text",
            hovertext=[
                f"<b>N-Terminus • Res 1 ({ONE_TO_THREE.get(clean_seq[0], clean_seq[0])}-{clean_seq[0]})</b>",
                f"<b>C-Terminus • Res {L:,} ({ONE_TO_THREE.get(clean_seq[-1], clean_seq[-1])}-{clean_seq[-1]})</b>",
            ],
            showlegend=True,
        )
    )

    # Assemble informative header + honest schematic disclaimer
    clean_pname = protein_name.strip() if isinstance(protein_name, str) and protein_name.strip() else "Query Protein"
    mode_titles = {
        "ss": "Secondary Structure (H / E / C)",
        "hydrophobicity": "Kyte–Doolittle Hydrophobicity",
        "charge": "Side-Chain Charge (pH ~7)",
    }

    stat_parts = [f"{L:,} aa"]
    if met_dict.get("mw_kda") is not None:
        stat_parts.append(f"MW {float(met_dict['mw_kda']):.2f} kDa")
    if met_dict.get("pi") is not None:
        stat_parts.append(f"pI {float(met_dict['pi']):.2f}")
    if met_dict.get("gravy") is not None:
        stat_parts.append(f"GRAVY {float(met_dict['gravy']):+.3f}")
    if is_downsampled:
        stat_parts.append(f"Macro view (~{downsample_step} aa/node)")
    stat_parts.append(f"Color: {mode_titles[mode]}")
    stats_footer = "  •  ".join(stat_parts)

    axis_title_font = dict(family=PLOTLY_FONT_FAMILY, size=11, color="#1E352F")
    axis_style = dict(
        showbackground=True,
        backgroundcolor="#F1F7F5",
        gridcolor="#DCEAE4",
        zeroline=False,
        showspikes=False,
        tickfont=dict(family=PLOTLY_FONT_FAMILY, size=10, color="#4A635C"),
    )

    layout = dict(
        font=dict(family=PLOTLY_FONT_FAMILY, color="#1E352F", size=12),
        title=dict(
            text=f"<b>Offline 3D Biophysical Explorer — {clean_pname}</b>",
            subtitle=dict(
                text=SCHEMATIC_3D_DISCLAIMER,
                font=dict(family=PLOTLY_FONT_FAMILY, size=11, color="#47625B"),
            ),
            font=dict(family=PLOTLY_FONT_FAMILY, size=14, color="#0B221C"),
            x=0.0,
            xanchor="left",
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FBF9",
        scene=dict(
            bgcolor="#F8FBF9",
            aspectmode="data",
            xaxis=dict(title=dict(text="X (Å)", font=axis_title_font), **axis_style),
            yaxis=dict(title=dict(text="Y (Å)", font=axis_title_font), **axis_style),
            zaxis=dict(title=dict(text="Z (Å)", font=axis_title_font), **axis_style),
            camera=dict(
                eye=dict(x=1.45, y=1.35, z=1.05),
                up=dict(x=0.0, y=0.0, z=1.0),
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.04,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.90)",
            bordercolor="#D5E5DF",
            borderwidth=1,
            font=dict(family=PLOTLY_FONT_FAMILY, size=11, color="#1E352F"),
        ),
        annotations=[
            dict(
                text=stats_footer,
                xref="paper",
                yref="paper",
                x=0.0,
                y=-0.11,
                xanchor="left",
                yanchor="top",
                showarrow=False,
                font=dict(family=PLOTLY_FONT_FAMILY, size=10.5, color="#47625B"),
            )
        ],
        margin=dict(l=12, r=12, t=72, b=58),
        height=safe_height,
        hoverlabel=dict(
            bgcolor="#0B221C",
            bordercolor="#0D9488",
            font=dict(family=PLOTLY_FONT_FAMILY, size=11.5, color="#F8FAFC"),
        ),
    )

    return go.Figure(data=traces, layout=layout)


__all__ = [
    "MAX_REMOTE_STRUCTURE_BYTES",
    "MAX_UPLOAD_STRUCTURE_BYTES",
    "MAX_FULL_3D_RESIDUES",
    "SUPPORTED_3DMOL_STYLES",
    "SUPPORTED_3DMOL_COLOR_SCHEMES",
    "SUPPORTED_OFFLINE_3D_COLOR_MODES",
    "SCHEMATIC_3D_DISCLAIMER",
    "normalize_uniprot_accession",
    "summarize_structure_text",
    "parse_uploaded_structure_file",
    "fetch_structure_by_pdb_id",
    "fetch_rcsb_structure_for_uniprot",
    "fetch_3d_structure_for_uniprot",
    "build_3dmol_viewer_html",
    "build_3d_biophysical_backbone_figure",
]
