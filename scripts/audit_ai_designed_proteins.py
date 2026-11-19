#!/usr/bin/env python3
"""
ProPurify Studio — AI-Designed / De Novo Protein Discrepancy Audit & Scaling Study
==================================================================================
Audits EVERY ProPurify module (`purification_metrics`, `buffer_advisor`, `sec_simulator`,
`sds_page_simulator`, `secondary_structure`) on a curated, publication-verified panel of
32 experimentally solved PDB 3D structures of AI-designed & de novo proteins spanning:
  - Landmark Rosetta / De Novo folds (Top7 1QYS, sTIM11 5BVL, DeNovoTIM15 6WVS,
    mFAP2b beta-barrel 6D0T, 17_bp_sh3 7UR7, DSBH 6E5C, Pizza6 beta-propeller 6F0Q,
    Symfoil-4P 3Q7X, Ferredoxin-fold Di-I_5 2KL8, Rossmann 2LV8, NTF2 5TPJ, Foldit1/4
    6MRR/6WI5, minimal thioredoxin 7LDF)
  - De novo helical bundles, coiled-coils & alpha-solenoids (4TQL, 5J73, 6Z35, 7UDJ,
    9HNH, 8FJE)
  - De novo integral membrane proteins (TMH4C4 alpha-pore 6M6Z, TMB2.3 beta-barrel 6X1K)
  - De novo self-assembling icosahedral nanocage (I53-40A 5IM5)
  - Deep-learning hallucination, ProteinMPNN & RFdiffusion designs (trRosetta 7K3H/7M0Q,
    HALC1_878 8CYK, HALC2_065 8D05, LCB1 7JZL, RFdiffusion HA_20 8SK7, Bim_fulldiff 8T5E,
    RFdiffusion+MPNN Retroaldolase RAD32 9FW7, ProteinMPNN Ubiquitin R4 9LQM)

Includes:
  - Live RCSB PDB REST + mmCIF experimental coordinate & DSSP secondary structure extraction
    (with offline fallback if cached panel exists in `data/ai_designed_proteins_audit.json`)
  - Exact 3D Radius of Gyration (Rg), inertia-tensor axial ratio, Perrin frictional shape
    factor (f/f0), and structure-derived hydrodynamic Stokes radius (Rh_3D) vs SEC predictions
  - Strict zero-leakage 5-mer similarity audit against `data/ss_10k_corpus.npz` training split
  - Complete Question B parameter audit & optional controlled SS learning-curve integration.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modules.buffer_advisor import (
    build_compatibility_matrix,
    infer_protein_biochemical_profile,
    recommend_full_purification_workflow,
)
from modules.purification_metrics import (
    analyze_all_parameters,
    calculate_net_charge_at_ph,
    DIWV,
)
from modules.sds_page_simulator import calculate_sds_rf
from modules.sec_simulator import (
    SEC_COLUMNS,
    calculate_dual_uv_ratio,
    calculate_retention_volume,
    predict_stokes_radius_rh,
    recommend_optimal_columns,
    simulate_chromatogram_peaks,
)
from modules.secondary_structure import (
    AA_ORDER,
    STATE_TO_IDX,
    classify_fold_class,
    compute_q3,
    compute_sov99,
    load_ss_model_weights,
    predict_chou_fasman,
    predict_gor4,
    predict_secondary_structure,
)
from modules.sequence_parser import detect_sequence_terminal_tag


# ==============================================================================
# 1. CURATED 32-PROTEIN REAL EXPERIMENTAL PDB AI-DESIGNED / DE NOVO PANEL
# ==============================================================================

AI_DESIGNED_PANEL_SPEC: List[Dict[str, Any]] = [
    # --- Mixed alpha/beta landmark de novo folds ---
    {
        "pdb_id": "1QYS", "entity_id": 1, "asym_id": "A",
        "short_name": "Top7",
        "realistic_name": "Top7 computationally designed novel fold protein",
        "method_group": "Rosetta / Classical De Novo",
        "citation": "Kuhlman et al., Science 2003",
        "expected_fold": "Mixed α/β",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "Compact globular α/β",
    },
    {
        "pdb_id": "5BVL", "entity_id": 1, "asym_id": "A",
        "short_name": "sTIM11",
        "realistic_name": "ProteinMPNN / Rosetta designed TIM barrel sTIM11",
        "method_group": "Rosetta / Classical De Novo",
        "citation": "Huang et al., Nat Chem Biol 2016",
        "expected_fold": "Mixed α/β",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "Compact globular 8-fold TIM barrel",
    },
    {
        "pdb_id": "6WVS", "entity_id": 1, "asym_id": "A",
        "short_name": "DeNovoTIM15",
        "realistic_name": "Hyperstable de novo TIM barrel DeNovoTIM15",
        "method_group": "Rosetta / Classical De Novo",
        "citation": "Wiese et al., 2020",
        "expected_fold": "Mixed α/β",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "Compact globular TIM barrel (Tm > 95°C)",
    },
    {
        "pdb_id": "2KL8", "entity_id": 1, "asym_id": "A",
        "short_name": "Di-I_5 (Ferredoxin fold)",
        "realistic_name": "De novo ferredoxin fold protein Di-I_5",
        "method_group": "Rosetta / Classical De Novo",
        "citation": "Koga et al., Nature 2012",
        "expected_fold": "Mixed α/β",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "Compact globular ferredoxin-like scaffold (NO Fe-S cluster!)",
    },
    {
        "pdb_id": "2LV8", "entity_id": 1, "asym_id": "A",
        "short_name": "Rossmann 2x2",
        "realistic_name": "De novo Rossmann 2x2 fold protein",
        "method_group": "Rosetta / Classical De Novo",
        "citation": "Koga et al., Nature 2012",
        "expected_fold": "Mixed α/β",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "Compact globular Rossmann fold",
    },
    {
        "pdb_id": "5TPJ", "entity_id": 1, "asym_id": "A",
        "short_name": "DeNovo-NTF2",
        "realistic_name": "De novo luciferase LuxSit NTF2-like curved beta-sheet scaffold",
        "method_group": "Rosetta / Classical De Novo",
        "citation": "Marcos et al., Science 2017",
        "expected_fold": "Mixed α/β",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "Compact cone/pocket NTF2 scaffold (LuxSit fold family)",
    },
    {
        "pdb_id": "6MRR", "entity_id": 1, "asym_id": "A",
        "short_name": "Foldit1",
        "realistic_name": "De novo designed protein Foldit1",
        "method_group": "Foldit / De Novo",
        "citation": "Koepnick et al., Nature 2019",
        "expected_fold": "Mixed α/β",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "Compact small α/β fold (7.9 kDa)",
    },
    {
        "pdb_id": "6WI5", "entity_id": 1, "asym_id": "A",
        "short_name": "Foldit4",
        "realistic_name": "De novo designed protein Foldit4 (zero Trp/Tyr)",
        "method_group": "Foldit / De Novo",
        "citation": "Koepnick et al., 2020",
        "expected_fold": "Mixed α/β",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "Compact α/β, ZERO Trp & ZERO Tyr (A280 = 0)",
    },
    {
        "pdb_id": "7LDF", "entity_id": 1, "asym_id": "A",
        "short_name": "ems_thioM_802",
        "realistic_name": "De novo minimal thioredoxin fold protein",
        "method_group": "Rosetta / Classical De Novo",
        "citation": "Baker/Marcos Lab, 2022",
        "expected_fold": "Mixed α/β",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "Compact minimal thioredoxin-like α/β (8.5 kDa)",
    },
    {
        "pdb_id": "5IM5", "entity_id": 1, "asym_id": "A",
        "short_name": "I53-40A Nanocage",
        "realistic_name": "De novo icosahedral nanocage subunit I53-40A",
        "method_group": "De Novo Nanocage",
        "citation": "Bale et al., Science 2016",
        "expected_fold": "Mixed α/β",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 60, "shape_note": "Monomer subunit (15.5 kDa) of 120-subunit ~2.4 MDa icosahedral nanocage (I53-40A60+B60, Rh ~120 Å)",
    },
    # --- All-beta de novo folds ---
    {
        "pdb_id": "6D0T", "entity_id": 1, "asym_id": "A",
        "short_name": "mFAP2b (BB1)",
        "realistic_name": "esmGFP / De novo fluorescence-activating beta-barrel mFAP2b",
        "method_group": "Rosetta / Classical De Novo",
        "citation": "Dou et al., Nature 2018",
        "expected_fold": "All-β (Beta-Sheet Rich)",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "Compact 8-stranded de novo β-barrel",
    },
    {
        "pdb_id": "7UR7", "entity_id": 1, "asym_id": "A",
        "short_name": "17_bp_sh3",
        "realistic_name": "De novo small SH3-like beta-barrel 17_bp_sh3",
        "method_group": "Rosetta / Classical De Novo",
        "citation": "Marcos Lab, 2023",
        "expected_fold": "All-β (Beta-Sheet Rich)",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "Ultra-small 63-aa β-barrel (7.2 kDa)",
    },
    {
        "pdb_id": "6E5C", "entity_id": 1, "asym_id": "A",
        "short_name": "DSBH-DeNovo",
        "realistic_name": "De novo double-stranded beta-helix protein",
        "method_group": "Rosetta / Classical De Novo",
        "citation": "Marcos et al., Nature Struct Mol Biol 2018",
        "expected_fold": "All-β (Beta-Sheet Rich)",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "Pure β-sheet (0% α-helix)",
    },
    {
        "pdb_id": "6F0Q", "entity_id": 1, "asym_id": "A",
        "short_name": "Pizza6-AYW",
        "realistic_name": "De novo symmetric 6-bladed beta-propeller Pizza6",
        "method_group": "Symmetric De Novo",
        "citation": "Voet et al., 2018",
        "expected_fold": "All-β (Beta-Sheet Rich)",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "251-aa symmetric β-propeller torus (26.5 kDa)",
    },
    {
        "pdb_id": "3Q7X", "entity_id": 1, "asym_id": "A",
        "short_name": "Symfoil-4P",
        "realistic_name": "De novo symmetric beta-trefoil Symfoil-4P",
        "method_group": "Symmetric De Novo",
        "citation": "Lee & Blaber, PNAS 2011",
        "expected_fold": "All-β (Beta-Sheet Rich)",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "Compact β-trefoil, ZERO Trp & ZERO Tyr (A280 = 0)",
    },
    # --- All-alpha bundles, coiled-coils & alpha-solenoids ---
    {
        "pdb_id": "4TQL", "entity_id": 1, "asym_id": "A",
        "short_name": "3-Helix Bundle",
        "realistic_name": "De novo elongated three-helix bundle",
        "method_group": "Rosetta / Classical De Novo",
        "citation": "Huang et al., Science 2014",
        "expected_fold": "All-α (Helix-Rich)",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "Strongly elongated 3-helix rod (~110 Å long!)",
    },
    {
        "pdb_id": "5J73", "entity_id": 1, "asym_id": "A",
        "short_name": "2L4HC2_9",
        "realistic_name": "De novo H-bond network coiled-coil homo-oligomer 2L4HC2_9",
        "method_group": "Rosetta / Classical De Novo",
        "citation": "Boyken et al., Science 2016",
        "expected_fold": "All-α (Helix-Rich)",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 2, "shape_note": "Elongated helical coiled-coil dimer",
    },
    {
        "pdb_id": "6Z35", "entity_id": 1, "asym_id": "A",
        "short_name": "Maquette 2-1ip",
        "realistic_name": "De novo 4-helix bundle Maquette 2",
        "method_group": "De Novo Helical Bundle",
        "citation": "Dutton/Anderson Lab, 2021",
        "expected_fold": "All-α (Helix-Rich)",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "Hyper-charged soluble 4-helix bundle",
    },
    {
        "pdb_id": "7UDJ", "entity_id": 2, "asym_id": "B",
        "short_name": "RPB_PEW3_R4",
        "realistic_name": "De novo designed helical repeat protein RPB_PEW3_R4",
        "method_group": "De Novo Repeat / Solenoid",
        "citation": "Baker Lab, 2023",
        "expected_fold": "All-α (Helix-Rich)",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "Elongated helical repeat solenoid, ZERO Trp & ZERO Tyr (A280 = 0)",
    },
    {
        "pdb_id": "9HNH", "entity_id": 1, "asym_id": "A",
        "short_name": "A7 Alpha-Solenoid",
        "realistic_name": "A7 Synthetic alpha-solenoid repeat protein",
        "method_group": "De Novo Repeat / Solenoid",
        "citation": "2025",
        "expected_fold": "All-α (Helix-Rich)",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "Elongated synthetic α-solenoid",
    },
    {
        "pdb_id": "8FJE", "entity_id": 1, "asym_id": "A",
        "short_name": "E8 5-Repeat",
        "realistic_name": "De novo five-repeat helical solenoid E8",
        "method_group": "De Novo Repeat / Solenoid",
        "citation": "2023",
        "expected_fold": "All-α (Helix-Rich)",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "Elongated 5-repeat solenoid",
    },
    # --- De novo integral membrane proteins ---
    {
        "pdb_id": "6M6Z", "entity_id": 1, "asym_id": "A",
        "short_name": "TMH4C4 (TM Pore)",
        "realistic_name": "De novo transmembrane nanopore TMH4C4",
        "method_group": "De Novo Membrane",
        "citation": "Xu et al., Nature 2020",
        "expected_fold": "All-α (Helix-Rich)",
        "expected_membrane": True, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 8, "shape_note": "Integral membrane transmembrane α-helical nanopore (needs DDM micelle!)",
    },
    {
        "pdb_id": "6X1K", "entity_id": 1, "asym_id": "A",
        "short_name": "TMB2.3 (TM Beta-Barrel)",
        "realistic_name": "De novo transmembrane beta-barrel TMB2.3",
        "method_group": "De Novo Membrane",
        "citation": "Vorobieva et al., Science 2021",
        "expected_fold": "All-β (Beta-Sheet Rich)",
        "expected_membrane": True, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "Integral membrane 8-stranded β-barrel (alternating amphipathic strands; needs detergent!)",
    },
    # --- Deep Learning Hallucination (trRosetta) ---
    {
        "pdb_id": "7K3H", "entity_id": 1, "asym_id": "A",
        "short_name": "Hallucinated 0217",
        "realistic_name": "Deep network hallucinated protein 0217",
        "method_group": "Deep-Learning Hallucination",
        "citation": "Anishchenko et al., Nature 2021",
        "expected_fold": "All-α (Helix-Rich)",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "Compact trRosetta hallucinated α-bundle",
    },
    {
        "pdb_id": "7M0Q", "entity_id": 1, "asym_id": "A",
        "short_name": "Hallucinated 0738",
        "realistic_name": "Deep network hallucinated protein 0738_mod",
        "method_group": "Deep-Learning Hallucination",
        "citation": "Anishchenko et al., Nature 2021",
        "expected_fold": "Mixed α/β",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "Compact trRosetta hallucinated α/β",
    },
    # --- ProteinMPNN & Hallucination Assemblies ---
    {
        "pdb_id": "8CYK", "entity_id": 1, "asym_id": "A",
        "short_name": "HALC1_878",
        "realistic_name": "ProteinMPNN hallucinated protein HALC1_878",
        "method_group": "ProteinMPNN / Hallucination",
        "citation": "Wicky et al., Science 2022",
        "expected_fold": "Mixed α/β",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "Globular ProteinMPNN-designed α/β",
    },
    {
        "pdb_id": "8D05", "entity_id": 1, "asym_id": "A",
        "short_name": "HALC2_065",
        "realistic_name": "ProteinMPNN hallucinated C2 assembly HALC2_065",
        "method_group": "ProteinMPNN / Hallucination",
        "citation": "Wicky et al., Science 2022",
        "expected_fold": "Mixed α/β",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 2, "shape_note": "Small 60-aa subunit of C2 homodimer",
    },
    {
        "pdb_id": "9LQM", "entity_id": 1, "asym_id": "A",
        "short_name": "MPNN-Ub R4",
        "realistic_name": "ProteinMPNN synthetic ubiquitin variant R4",
        "method_group": "ProteinMPNN",
        "citation": "2025",
        "expected_fold": "Mixed α/β",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "Compact 81-aa ProteinMPNN ubiquitin fold (9.1 kDa)",
    },
    # --- De novo & RFdiffusion + ProteinMPNN Minibinders & Enzymes ---
    {
        "pdb_id": "7JZL", "entity_id": 2, "asym_id": "B",
        "short_name": "LCB1 Minibinder",
        "realistic_name": "De novo SARS-CoV-2 RBD minibinder LCB1",
        "method_group": "De Novo Minibinder",
        "citation": "Cao et al., Science 2020",
        "expected_fold": "All-α (Helix-Rich)",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "Ultra-small 55-aa 3-helix minibinder (6.6 kDa)",
    },
    {
        "pdb_id": "8SK7", "entity_id": 3, "asym_id": "C",
        "short_name": "RFdiff HA_20",
        "realistic_name": "RFdiffusion binder HA_20 against Influenza Hemagglutinin",
        "method_group": "RFdiffusion + ProteinMPNN",
        "citation": "Watson et al., Nature 2023",
        "expected_fold": "Mixed α/β",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "Small 65-aa RFdiffusion minibinder (7.6 kDa)",
    },
    {
        "pdb_id": "8T5E", "entity_id": 1, "asym_id": "A",
        "short_name": "RFdiff Bim_fulldiff",
        "realistic_name": "RFdiffusion binder Bim_fulldiff to bioactive helical peptide",
        "method_group": "RFdiffusion + ProteinMPNN",
        "citation": "Vázquez Torres et al., Nature 2024",
        "expected_fold": "All-α (Helix-Rich)",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "RFdiffusion + ProteinMPNN helical binder (15.4 kDa)",
    },
    {
        "pdb_id": "9FW7", "entity_id": 1, "asym_id": "A",
        "short_name": "RFdiff RAD32",
        "realistic_name": "RFdiffusion ProteinMPNN de novo retroaldolase RAD32",
        "method_group": "RFdiffusion + ProteinMPNN",
        "citation": "2025",
        "expected_fold": "All-α (Helix-Rich)",
        "expected_membrane": False, "expected_metal": False, "expected_disulfide": False, "expected_idp": False,
        "oligomer_n": 1, "shape_note": "RFdiffusion + ProteinMPNN de novo catalytic enzyme (22.6 kDa)",
    },
]


# ==============================================================================
# 2. RCSB PDB REST + mmCIF 3D GEOMETRY & DSSP FETCHERS
# ==============================================================================

def _parse_cif_ca_coords(cif_text: str, target_asym_id: str) -> np.ndarray:
    """
    Extracts Model-1 C-alpha 3D coordinates (N, 3) for `label_asym_id == target_asym_id`
    directly from an RCSB mmCIF file (zero external parser dependencies).
    """
    lines = cif_text.splitlines()
    in_atom_loop = False
    col_names: List[str] = []
    coords: List[Tuple[float, float, float]] = []
    seen_seq_ids = set()

    i = 0
    n_lines = len(lines)
    while i < n_lines:
        ln = lines[i].strip()
        if ln == "loop_":
            cols = []
            j = i + 1
            while j < n_lines and lines[j].lstrip().startswith("_atom_site."):
                cols.append(lines[j].strip())
                j += 1
            if cols:
                col_names = cols
                col_map = {c: idx for idx, c in enumerate(col_names)}
                idx_group = col_map.get("_atom_site.group_PDB", 0)
                idx_atom = col_map.get("_atom_site.label_atom_id", -1)
                idx_asym = col_map.get("_atom_site.label_asym_id", -1)
                idx_seq = col_map.get("_atom_site.label_seq_id", -1)
                idx_alt = col_map.get("_atom_site.label_alt_id", -1)
                idx_x = col_map.get("_atom_site.Cartn_x", -1)
                idx_y = col_map.get("_atom_site.Cartn_y", -1)
                idx_z = col_map.get("_atom_site.Cartn_z", -1)
                idx_mod = col_map.get("_atom_site.pdbx_PDB_model_num", -1)

                i = j
                while i < n_lines:
                    row = lines[i].strip()
                    if not row or row.startswith("#") or row.startswith("loop_") or row.startswith("_"):
                        break
                    parts = row.split()
                    if len(parts) >= len(col_names):
                        if parts[idx_group] == "ATOM":
                            if idx_mod >= 0 and parts[idx_mod] != "1":
                                i += 1
                                continue
                            if parts[idx_asym] == target_asym_id and parts[idx_atom] == "CA":
                                alt = parts[idx_alt] if idx_alt >= 0 else "."
                                seq_id = parts[idx_seq] if idx_seq >= 0 else str(len(coords))
                                if alt in (".", "?", "A") and seq_id not in seen_seq_ids:
                                    seen_seq_ids.add(seq_id)
                                    coords.append((float(parts[idx_x]), float(parts[idx_y]), float(parts[idx_z])))
                    i += 1
                break
        i += 1

    return np.array(coords, dtype=np.float64) if coords else np.zeros((0, 3), dtype=np.float64)


def _compute_3d_hydrodynamic_geometry(coords: np.ndarray, mw_kda: float) -> Dict[str, Any]:
    """
    Computes exact 3D structural hydrodynamics from crystallographic C-alpha coordinates:
      - Radius of gyration Rg (Å)
      - Principal semi-axes (a >= b >= c) of equivalent uniform-density ellipsoid:
        eigenvalues of gyration tensor S = (1/N) X^T X satisfy lambda_k = axis_k^2 / 5
      - Prolate aspect ratio p = a / sqrt(b * c)
      - Exact Perrin (1936) prolate ellipsoid frictional shape ratio f / f0(p)
      - Structure-derived hydrodynamic Stokes radius Rh_3D (Å):
          Rh_sphere_hyd = R_min_vdw(MW) + 2.8 Å hydration shell (Erickson 2009 / La Verde 2017)
          where R_min_vdw = 0.66 * (MW_Da)^(1/3) Å
          Rh_3D_perrin  = Rh_sphere_hyd * (f / f0)_perrin
        And direct Rg-based estimate Rh_from_Rg = Rg * sqrt(5/3) + 2.8 Å (for globular check).
    """
    if coords.shape[0] < 5:
        return {}
    centered = coords - coords.mean(axis=0, keepdims=True)
    gyr = (centered.T @ centered) / float(coords.shape[0])
    eigvals = np.sort(np.maximum(1e-6, np.linalg.eigvalsh(gyr)))[::-1]  # l1 >= l2 >= l3
    rg = float(math.sqrt(float(eigvals.sum())))
    # Equivalent ellipsoid semi-axes
    a, b, c = [math.sqrt(5.0 * float(ev)) for ev in eigvals]
    # Prolate aspect ratio p = a / sqrt(b*c)
    p = max(1.0, a / max(1e-6, math.sqrt(b * c)))
    # Exact Perrin (1936) translational frictional ratio for a prolate ellipsoid of revolution
    if p <= 1.005:
        f_perrin = 1.0
    else:
        s = math.sqrt(p * p - 1.0)
        f_perrin = s / ((p ** (1.0 / 3.0)) * math.log(p + s))

    # Physics-consistent shape-adjusted hydrodynamic radius:
    # `predict_stokes_radius_rh(mw_kda, "globular")` is calibrated directly against experimental Cytiva
    # markers (R^2 = 0.9950 across 1.35 kDa - 3.0 MDa); scaling by Perrin's normalized prolate shape factor
    # (f/f0)_Perrin / 1.01 isolates pure 3D elongation shift cleanly across all MW regimes.
    rh_glob_ref = predict_stokes_radius_rh(mw_kda, "globular")
    rh_3d_perrin = rh_glob_ref * max(1.0, f_perrin / 1.01)
    # Direct crystallographic Rg-based estimate (Nygaard/Erickson: Rg / Rh ≈ 0.775 for spheres)
    rh_from_rg = rg / 0.775

    if p >= 2.6 or f_perrin >= 1.13:
        geom_class = "Elongated / Rod-like"
        recommended_sec_conf = "elongated" if f_perrin >= 1.18 else "moderately_elongated"
    elif p >= 1.8 or f_perrin >= 1.06:
        geom_class = "Moderately Elongated"
        recommended_sec_conf = "moderately_elongated"
    else:
        geom_class = "Compact Globular"
        recommended_sec_conf = "globular"

    return {
        "n_ca_atoms": int(coords.shape[0]),
        "rg_angstrom": round(rg, 2),
        "semi_axes_a_b_c": [round(a, 1), round(b, 1), round(c, 1)],
        "max_end_to_end_span_a": round(float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))), 1),
        "prolate_aspect_ratio_p": round(p, 2),
        "perrin_f_over_f0": round(f_perrin, 3),
        "rh_3d_perrin_a": round(rh_3d_perrin, 1),
        "rh_from_rg_a": round(rh_from_rg, 1),
        "geometry_class": geom_class,
        "recommended_sec_conformation": recommended_sec_conf,
    }


def fetch_pdb_record(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Fetches sequence, experimental DSSP SS3, unobserved mask, and C-alpha 3D geometry from RCSB PDB."""
    pid = spec["pdb_id"]
    eid = spec["entity_id"]
    asym = spec["asym_id"]

    entry = json.loads(urllib.request.urlopen(f"https://data.rcsb.org/rest/v1/core/entry/{pid}", timeout=15).read().decode())
    pe = json.loads(urllib.request.urlopen(f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pid}/{eid}", timeout=15).read().decode())
    inst = json.loads(urllib.request.urlopen(f"https://data.rcsb.org/rest/v1/core/polymer_entity_instance/{pid}/{asym}", timeout=15).read().decode())
    cif_txt = urllib.request.urlopen(f"https://files.rcsb.org/download/{pid}.cif", timeout=20).read().decode("utf-8", errors="ignore")

    full_seq = pe["entity_poly"]["pdbx_seq_one_letter_code_can"].replace("\n", "").strip().upper()
    L = len(full_seq)
    ss = ["C"] * L
    obs = [True] * L

    for f in inst.get("rcsb_polymer_instance_feature", []):
        ft = f.get("type")
        if ft == "HELIX_P":
            for p in f.get("feature_positions", []):
                for i in range(max(0, p["beg_seq_id"] - 1), min(L, p["end_seq_id"])):
                    ss[i] = "H"
        elif ft == "SHEET":
            for p in f.get("feature_positions", []):
                for i in range(max(0, p["beg_seq_id"] - 1), min(L, p["end_seq_id"])):
                    ss[i] = "E"
        elif ft == "UNOBSERVED_RESIDUE_XYZ":
            for p in f.get("feature_positions", []):
                for i in range(max(0, p["beg_seq_id"] - 1), min(L, p["end_seq_id"])):
                    obs[i] = False

    obs_idx = [i for i, o in enumerate(obs) if o]
    start, end = obs_idx[0], obs_idx[-1] + 1
    core_seq = full_seq[start:end]
    core_ss = "".join(ss[start:end])
    core_obs_mask = obs[start:end]

    coords = _parse_cif_ca_coords(cif_txt, asym)
    core_metrics = analyze_all_parameters(core_seq)
    geom = _compute_3d_hydrodynamic_geometry(coords, core_metrics["mw_kda"])

    return {
        **spec,
        "pdb_title": (entry.get("struct") or {}).get("title", ""),
        "exp_method": ((entry.get("exptl") or [{}])[0]).get("method", ""),
        "release_year": ((entry.get("rcsb_accession_info") or {}).get("initial_release_date") or "")[:4],
        "full_pdb_sequence": full_seq,
        "core_sequence": core_seq,
        "core_ss3": core_ss,
        "core_obs_mask": core_obs_mask,
        "trimmed_n_tag": full_seq[:start],
        "trimmed_c_tag": full_seq[end:],
        "geometry_3d": geom,
    }


# ==============================================================================
# 3. ZERO-LEAKAGE CHECK AGAINST NATURAL TRAINING CORPUS
# ==============================================================================

def check_zero_leakage_vs_train_corpus(panel: List[Dict[str, Any]]) -> Dict[str, Any]:
    corpus_path = os.path.join(ROOT_DIR, "data", "ss_10k_corpus.npz")
    with np.load(corpus_path, allow_pickle=False) as d:
        tr_mask = d["splits"] == "train"
        tr_seqs = [str(s) for s in d["sequences"][tr_mask]]
        tr_accs = [str(a) for a in d["accessions"][tr_mask]]
        tr_names = [str(n) for n in d["protein_names"][tr_mask]]

    # Build 5-mer inverted index over all 3,751 train sequences
    kmer_to_idx: Dict[str, List[int]] = {}
    tr_kmer_cnt = np.zeros(len(tr_seqs), dtype=np.int32)
    for idx, seq in enumerate(tr_seqs):
        ks = {seq[i:i + 5] for i in range(len(seq) - 4)}
        tr_kmer_cnt[idx] = len(ks)
        for km in ks:
            kmer_to_idx.setdefault(km, []).append(idx)

    per_prot = {}
    max_cont_all = 0.0
    max_jacc_all = 0.0
    exact_hits = 0

    for rec in panel:
        sq = rec["core_sequence"]
        ks = {sq[i:i + 5] for i in range(len(sq) - 4)}
        hit_counts: Dict[int, int] = {}
        for km in ks:
            for tidx in kmer_to_idx.get(km, ()):
                hit_counts[tidx] = hit_counts.get(tidx, 0) + 1
        best_c = 0.0
        best_j = 0.0
        best_match = None
        for tidx, inter in hit_counts.items():
            if sq == tr_seqs[tidx]:
                exact_hits += 1
            cont = inter / max(1, min(len(ks), int(tr_kmer_cnt[tidx])))
            jacc = inter / max(1, len(ks) + int(tr_kmer_cnt[tidx]) - inter)
            if cont > best_c:
                best_c = cont
                best_j = jacc
                best_match = f"{tr_accs[tidx]} ({tr_names[tidx][:40]})"
        max_cont_all = max(max_cont_all, best_c)
        max_jacc_all = max(max_jacc_all, best_j)
        per_prot[rec["pdb_id"]] = {
            "max_5mer_containment_vs_train": round(best_c, 4),
            "max_5mer_jaccard_vs_train": round(best_j, 4),
            "closest_train_protein": best_match,
        }

    # Also compute natural train AA composition baseline for comparison
    all_tr_str = "".join(tr_seqs)
    tot_tr = len(all_tr_str)
    nat_aa_pct = {aa: round(100.0 * all_tr_str.count(aa) / tot_tr, 2) for aa in AA_ORDER}

    return {
        "n_train_corpus_checked": len(tr_seqs),
        "exact_sequence_overlaps": exact_hits,
        "max_5mer_containment": round(max_cont_all, 4),
        "max_5mer_jaccard": round(max_jacc_all, 4),
        "any_pair_ge_0_30": bool(max_cont_all >= 0.30 or max_jacc_all >= 0.30),
        "per_protein": per_prot,
        "natural_train_aa_pct": nat_aa_pct,
    }


# ==============================================================================
# 4. COMPLETE MODULE-BY-MODULE AUDIT ON ALL 32 AI-DESIGNED PROTEINS
# ==============================================================================

def _compute_mcc_and_stats(conf: np.ndarray) -> Dict[str, Any]:
    total = float(conf.sum())
    q3 = 100.0 * float(np.trace(conf)) / max(1.0, total)
    per_state = {}
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
            "support": int(conf[idx, :].sum()),
        }
    return {
        "pooled_q3_pct": round(q3, 2),
        "MCC_H": per_state["H"]["mcc"],
        "MCC_E": per_state["E"]["mcc"],
        "MCC_C": per_state["C"]["mcc"],
        "mean_mcc": round((per_state["H"]["mcc"] + per_state["E"]["mcc"] + per_state["C"]["mcc"]) / 3.0, 4),
        "per_state": per_state,
        "confusion_HEC": conf.tolist(),
    }


def _dipeptide_instability_drivers(seq: str) -> List[Tuple[str, int, float]]:
    """Returns top dipeptides contributing to Guruprasad Instability Index > 40."""
    counts: Dict[str, int] = {}
    for i in range(len(seq) - 1):
        dp = seq[i:i + 2]
        counts[dp] = counts.get(dp, 0) + 1
    contrib = []
    L = len(seq)
    for dp, c in counts.items():
        w = DIWV.get(dp, 1.0)
        if w > 1.0:
            contrib.append((dp, c, round((10.0 / L) * c * w, 2)))
    contrib.sort(key=lambda x: x[2], reverse=True)
    return contrib[:5]


def run_full_audit(panel: List[Dict[str, Any]], leakage_info: Dict[str, Any]) -> Dict[str, Any]:
    bundle = load_ss_model_weights()
    # Warmup once
    _ = predict_secondary_structure(panel[0]["core_sequence"], bundle)

    per_protein_results: List[Dict[str, Any]] = []
    conf_all = np.zeros((3, 3), dtype=np.int64)
    conf_gor4 = np.zeros((3, 3), dtype=np.int64)
    conf_cf = np.zeros((3, 3), dtype=np.int64)
    conf_by_fold: Dict[str, np.ndarray] = {}
    conf_by_method: Dict[str, np.ndarray] = {}

    q3_list: List[float] = []
    sov_list: List[float] = []
    gor4_q3_list: List[float] = []
    cf_q3_list: List[float] = []
    latencies_ms: List[float] = []
    err_h, err_e, err_c = [], [], []
    bias_h, bias_e, bias_c = [], [], []
    fold_matches = 0

    # Counters for discrepancy diagnostics
    zero_a280_ids = []
    low_a280_ids = []
    false_unstable_ids = []
    wh_insoluble_ids = []
    wh_borderline_ids = []
    wh_soluble_ids = []
    buffer_discrepancies: List[Dict[str, Any]] = []
    sec_discrepancies: List[Dict[str, Any]] = []

    ai_all_seq = ""

    for rec in panel:
        pid = rec["pdb_id"]
        seq = rec["core_sequence"]
        obs_ss = rec["core_ss3"]
        obs_mask = rec["core_obs_mask"]
        L = len(seq)
        ai_all_seq += seq

        # --- 1. PURIFICATION METRICS ---
        metrics = analyze_all_parameters(seq)
        q_74 = round(calculate_net_charge_at_ph(seq, 7.4), 2)
        eps_red = metrics["extinction"]["molar_extinction_reduced"]
        abs01 = metrics["extinction"]["abs_01_reduced"]
        if eps_red == 0:
            zero_a280_ids.append(f"{pid} ({rec['short_name']})")
        elif abs01 < 0.30:
            low_a280_ids.append(f"{pid} ({rec['short_name']}, Abs0.1%={abs01})")

        ii = metrics["instability_index"]
        ii_drivers = _dipeptide_instability_drivers(seq)
        if ii >= 40.0:
            false_unstable_ids.append({
                "pdb_id": pid, "name": rec["short_name"], "ii": ii,
                "top_dipeptides": ii_drivers[:4],
                "E_pct": metrics["aa_percentages"]["E"],
                "K_pct": metrics["aa_percentages"]["K"],
                "R_pct": metrics["aa_percentages"]["R"],
            })

        sol = metrics["solubility"]
        if "Inclusion-Body" in sol["solubility_class"]:
            wh_insoluble_ids.append({
                "pdb_id": pid, "name": rec["short_name"], "cv": sol["canonical_cv"],
                "cv_diff": sol["cv_discriminant"], "ngps_pct": sol["turn_residue_fraction"],
                "net_q_per_res": sol["net_charge_per_residue"],
            })
        elif "Borderline" in sol["solubility_class"]:
            wh_borderline_ids.append({
                "pdb_id": pid, "name": rec["short_name"], "cv": sol["canonical_cv"],
                "cv_diff": sol["cv_discriminant"],
            })
        else:
            wh_soluble_ids.append(pid)

        # --- 2. BUFFER ADVISOR (3 MODES: RAW UNNAMED, REALISTIC AI NAME, FULL PDB TAGGED) ---
        prof_raw = infer_protein_biochemical_profile(sequence=seq, protein_name="", metrics=metrics)
        wf_raw = recommend_full_purification_workflow(metrics=metrics, working_ph=7.4, net_charge=q_74, sequence=seq, protein_name="", protein_profile=prof_raw)

        prof_named = infer_protein_biochemical_profile(sequence=seq, protein_name=rec["realistic_name"], metrics=metrics)
        wf_named = recommend_full_purification_workflow(metrics=metrics, working_ph=7.4, net_charge=q_74, sequence=seq, protein_name=rec["realistic_name"], protein_profile=prof_named)
        compat_named = build_compatibility_matrix(metrics, 7.4, q_74, wf_named)

        full_seq = rec["full_pdb_sequence"]
        tag_info_full = detect_sequence_terminal_tag(full_seq)
        metrics_full = analyze_all_parameters(full_seq)
        q_74_full = round(calculate_net_charge_at_ph(full_seq, 7.4), 2)
        prof_full = infer_protein_biochemical_profile(sequence=full_seq, protein_name=rec["realistic_name"], metrics=metrics_full)

        # Check false positives & false negatives in buffer_advisor!
        for mode_lbl, prof_obj in [("raw_unnamed", prof_raw), ("named_ai", prof_named), ("full_tagged_named", prof_full)]:
            cats = set(prof_obj["categories"])
            if prof_obj["is_metalloprotein"] and not rec["expected_metal"]:
                buffer_discrepancies.append({
                    "pdb_id": pid, "name": rec["short_name"], "mode": mode_lbl,
                    "type": "FALSE_POSITIVE_METALLOPROTEIN",
                    "categories": sorted(cats), "metal_labels": prof_obj["metal_labels"],
                    "tag_conflict": bool(prof_obj.get("tag_conflict")),
                    "input_name": rec["realistic_name"] if mode_lbl != "raw_unnamed" else "",
                })
            if prof_obj["requires_detergent_micelles"] and not rec["expected_membrane"]:
                buffer_discrepancies.append({
                    "pdb_id": pid, "name": rec["short_name"], "mode": mode_lbl,
                    "type": "FALSE_POSITIVE_MEMBRANE",
                    "gravy": metrics["gravy"], "categories": sorted(cats),
                })
            if (not prof_obj["requires_detergent_micelles"]) and rec["expected_membrane"]:
                buffer_discrepancies.append({
                    "pdb_id": pid, "name": rec["short_name"], "mode": mode_lbl,
                    "type": "FALSE_NEGATIVE_MEMBRANE",
                    "gravy": metrics["gravy"], "categories": sorted(cats),
                    "input_name": rec["realistic_name"] if mode_lbl != "raw_unnamed" else "",
                })
            if ("idp_disordered" in cats) and not rec["expected_idp"]:
                buffer_discrepancies.append({
                    "pdb_id": pid, "name": rec["short_name"], "mode": mode_lbl,
                    "type": "FALSE_POSITIVE_IDP",
                    "gravy": metrics["gravy"], "aliphatic_index": metrics["aliphatic_index"],
                    "length": L,
                })
            if prof_obj["avoid_reducing_agents"] and not rec["expected_disulfide"]:
                buffer_discrepancies.append({
                    "pdb_id": pid, "name": rec["short_name"], "mode": mode_lbl,
                    "type": "FALSE_POSITIVE_DISULFIDE",
                    "cys_count": metrics["counts"]["cysteines"],
                })
            if prof_obj["protease_class"] is not None:
                buffer_discrepancies.append({
                    "pdb_id": pid, "name": rec["short_name"], "mode": mode_lbl,
                    "type": "FALSE_POSITIVE_PROTEASE",
                    "protease_class": prof_obj["protease_class"],
                })

        # --- 3. SEC SIMULATOR & SDS-PAGE ---
        mw_kda = metrics["mw_kda"]
        prep_col_id, anal_col_id = recommend_optimal_columns(mw_kda)
        col_cfg = SEC_COLUMNS[anal_col_id]
        rh_glob = predict_stokes_radius_rh(mw_kda, "globular")
        ve_glob, kav_glob, note_glob = calculate_retention_volume(mw_kda, col_cfg, "globular")

        geom = rec.get("geometry_3d") or {}
        rec_conf = geom.get("recommended_sec_conformation", "globular")
        rh_shape = predict_stokes_radius_rh(mw_kda, rec_conf)
        ve_shape, kav_shape, _ = calculate_retention_volume(mw_kda, col_cfg, rec_conf)
        rh_3d = geom.get("rh_3d_perrin_a")
        ve_3d = None
        if rh_3d:
            ve_3d, _, _ = calculate_retention_volume(mw_kda, col_cfg, rh_override_a=rh_3d)
            if abs(ve_glob - ve_3d) >= 0.45:
                sec_discrepancies.append({
                    "pdb_id": pid, "name": rec["short_name"], "mw_kda": mw_kda,
                    "column": col_cfg["name"], "aspect_ratio_p": geom.get("prolate_aspect_ratio_p"),
                    "perrin_f_f0": geom.get("perrin_f_over_f0"),
                    "rh_glob_a": round(rh_glob, 1), "rh_3d_perrin_a": rh_3d,
                    "ve_glob_ml": ve_glob, "ve_3d_ml": ve_3d, "delta_ve_ml": round(ve_glob - ve_3d, 2),
                    "ve_shape_profile_ml": ve_shape, "recommended_profile": rec_conf,
                })

        uv_sec = calculate_dual_uv_ratio(mw_kda, prot_eps01=abs01)
        sim_sec = simulate_chromatogram_peaks(
            [{"label": rec["short_name"], "mw_kda": mw_kda, "prot_eps01": metrics["extinction"]["abs_01_cystines"]}],
            col_cfg,
            resolution_points=100,
        )
        sec_uv280_invisible = bool(sim_sec["peaks"][0].get("uv280_invisible", False))
        wf_has_uv280_alert = any("UV-280 Invisible" in a for a in wf_named["alerts"])
        wf_has_ii_denovo_caveat = any("Guruprasad" in a for a in wf_named["alerts"])
        sds_rf_15 = calculate_sds_rf(mw_kda, 15.0)
        sds_rf_12 = calculate_sds_rf(mw_kda, 12.0)

        # --- 4. SECONDARY STRUCTURE ENGINE ---
        t0 = time.perf_counter()
        ss_res = predict_secondary_structure(seq, bundle)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)

        pred_ss = ss_res["ss_string"]
        gor4_ss, _ = predict_gor4(seq, bundle)
        cf_ss = predict_chou_fasman(seq)

        # Mask out any internal unobserved residues when evaluating exact crystallographic DSSP accuracy
        obs_ss_eval = "".join(ch for ch, ok in zip(obs_ss, obs_mask) if ok)
        pred_ss_eval = "".join(ch for ch, ok in zip(pred_ss, obs_mask) if ok)
        gor4_ss_eval = "".join(ch for ch, ok in zip(gor4_ss, obs_mask) if ok)
        cf_ss_eval = "".join(ch for ch, ok in zip(cf_ss, obs_mask) if ok)

        q3_p = compute_q3(obs_ss_eval, pred_ss_eval)
        sov_p = compute_sov99(obs_ss_eval, pred_ss_eval)
        q3_gor = compute_q3(obs_ss_eval, gor4_ss_eval)
        q3_cf = compute_q3(obs_ss_eval, cf_ss_eval)

        q3_list.append(q3_p)
        sov_list.append(sov_p)
        gor4_q3_list.append(q3_gor)
        cf_q3_list.append(q3_cf)

        # Confusion matrices
        def _upd(cm, o_s, p_s):
            oa = np.fromiter((STATE_TO_IDX[c] for c in o_s), dtype=np.int32, count=len(o_s))
            pa = np.fromiter((STATE_TO_IDX[c] for c in p_s), dtype=np.int32, count=len(p_s))
            cm += np.bincount(oa * 3 + pa, minlength=9).reshape(3, 3)

        _upd(conf_all, obs_ss_eval, pred_ss_eval)
        _upd(conf_gor4, obs_ss_eval, gor4_ss_eval)
        _upd(conf_cf, obs_ss_eval, cf_ss_eval)

        L_ev = len(obs_ss_eval)
        obs_h_pct = 100.0 * obs_ss_eval.count("H") / L_ev
        obs_e_pct = 100.0 * obs_ss_eval.count("E") / L_ev
        obs_c_pct = 100.0 * obs_ss_eval.count("C") / L_ev
        obs_fold, _ = classify_fold_class(obs_h_pct, obs_e_pct, obs_c_pct)

        pred_h_pct = 100.0 * pred_ss_eval.count("H") / L_ev
        pred_e_pct = 100.0 * pred_ss_eval.count("E") / L_ev
        pred_c_pct = 100.0 * pred_ss_eval.count("C") / L_ev
        pred_fold, _ = classify_fold_class(pred_h_pct, pred_e_pct, pred_c_pct)

        err_h.append(abs(pred_h_pct - obs_h_pct))
        err_e.append(abs(pred_e_pct - obs_e_pct))
        err_c.append(abs(pred_c_pct - obs_c_pct))
        bias_h.append(pred_h_pct - obs_h_pct)
        bias_e.append(pred_e_pct - obs_e_pct)
        bias_c.append(pred_c_pct - obs_c_pct)

        if pred_fold == obs_fold:
            fold_matches += 1

        fold_bucket = "Transmembrane De Novo" if rec["expected_membrane"] else obs_fold
        conf_by_fold.setdefault(fold_bucket, np.zeros((3, 3), dtype=np.int64))
        _upd(conf_by_fold[fold_bucket], obs_ss_eval, pred_ss_eval)

        mg = rec["method_group"]
        conf_by_method.setdefault(mg, np.zeros((3, 3), dtype=np.int64))
        _upd(conf_by_method[mg], obs_ss_eval, pred_ss_eval)

        per_protein_results.append({
            "pdb_id": pid,
            "entity_id": rec["entity_id"],
            "asym_id": rec["asym_id"],
            "short_name": rec["short_name"],
            "realistic_name": rec["realistic_name"],
            "method_group": mg,
            "citation": rec["citation"],
            "exp_method": rec["exp_method"],
            "release_year": rec["release_year"],
            "length_core": L,
            "length_eval": L_ev,
            "core_sequence": seq,
            "core_ss3": obs_ss,
            "pred_ss3": pred_ss,
            "metrics": {
                "mw_kda": mw_kda,
                "pi": metrics["pi"],
                "net_charge_ph74": q_74,
                "num_trp": metrics["extinction"]["num_Trp"],
                "num_tyr": metrics["extinction"]["num_Tyr"],
                "num_cys": metrics["extinction"]["num_Cys"],
                "eps_280_reduced": eps_red,
                "abs_01_reduced": abs01,
                "gravy": metrics["gravy"],
                "aliphatic_index": metrics["aliphatic_index"],
                "instability_index": ii,
                "instability_class": metrics["instability_class"],
                "ii_top_dipeptides": ii_drivers[:3],
                "wh_cv": sol["canonical_cv"],
                "wh_cv_diff": sol["cv_discriminant"],
                "wh_class": sol["solubility_class"],
                "wh_sol_pct": sol["solubility_probability_pct"],
                "wh_ngps_pct": sol["turn_residue_fraction"],
                "wh_net_q_per_res": sol["net_charge_per_residue"],
                "charged_pct": round(100.0 * (seq.count("D") + seq.count("E") + seq.count("K") + seq.count("R")) / L, 1),
            },
            "buffer_advisor": {
                "raw_unnamed_categories": prof_raw["categories"],
                "named_ai_categories": prof_named["categories"],
                "full_tagged_detected_tag": tag_info_full["tag_detected"],
                "full_tagged_categories": prof_full["categories"],
                "named_tag_conflict": prof_named["tag_conflict"]["headline"] if prof_named.get("tag_conflict") else None,
                "full_tag_conflict": prof_full["tag_conflict"]["headline"] if prof_full.get("tag_conflict") else None,
                "named_step1": wf_named["roadmap"][0]["title"],
                "named_alerts_count": len(wf_named["alerts"]),
                "uv280_invisible_alert_present": wf_has_uv280_alert,
                "ii_denovo_caveat_present": wf_has_ii_denovo_caveat,
            },
            "sec_and_sds": {
                "optimal_analytical_col": col_cfg["name"],
                "rh_globular_a": round(rh_glob, 1),
                "ve_globular_ml": ve_glob,
                "kav_globular": kav_glob,
                "rh_shape_profile_a": round(rh_shape, 1),
                "ve_shape_profile_ml": ve_shape,
                "ve_from_3d_perrin_ml": ve_3d,
                "geometry_3d": geom,
                "uv_a260_a280_reported": uv_sec["a260_a280_ratio"],
                "sec_uv280_invisible_flag": sec_uv280_invisible,
                "sds_rf_15pct": sds_rf_15,
                "sds_rf_12pct": sds_rf_12,
                "oligomer_n": rec["oligomer_n"],
            },
            "secondary_structure": {
                "q3_pct": round(q3_p, 2),
                "sov99_pct": round(sov_p, 2),
                "gor4_q3_pct": round(q3_gor, 2),
                "chou_fasman_q3_pct": round(q3_cf, 2),
                "obs_helix_pct": round(obs_h_pct, 2),
                "obs_strand_pct": round(obs_e_pct, 2),
                "obs_coil_pct": round(obs_c_pct, 2),
                "pred_helix_pct": round(pred_h_pct, 2),
                "pred_strand_pct": round(pred_e_pct, 2),
                "pred_coil_pct": round(pred_c_pct, 2),
                "obs_fold_class": obs_fold,
                "pred_fold_class": pred_fold,
                "fold_match": bool(pred_fold == obs_fold),
                "mean_confidence": ss_res["mean_confidence"],
                "latency_ms": round(dt_ms, 2),
            },
            "zero_leakage": leakage_info["per_protein"][pid],
        })

    # Oligomer / Nanocage special SEC checks (e.g., I53-40 nanocage 2.4 MDa, TMH4C4 8-mer + DDM micelle)
    special_sec_assemblies = []
    for label, mw_ass, conf, exp_rh, col_id, note in [
        ("I53-40 Icosahedral Nanocage (60A + 60B, ~2.38 MDa)", 2380.0, "globular", 120.0, "superose_6_increase_10_300",
         "Bale et al. Science 2016: hollow 24 nm outer-diameter icosahedral shell (Rh ~120 Å; hollow cage has ~20% larger Rh than solid sphere of equal MW!)"),
        ("5J73 2L4HC2_9 Coiled-Coil Homodimer (17.8 kDa)", 17.8, "elongated", None, "s75_increase_10_300",
         "Elongated helical coiled-coil dimer; elutes earlier than 17.8 kDa globular sphere"),
        ("8D05 HALC2_065 C2 Homodimer (13.6 kDa)", 13.6, "globular", None, "s75_increase_10_300",
         "Compact C2 homodimer"),
        ("6M6Z TMH4C4 Octameric Nanopore + DDM Micelle (~260 kDa)", 260.0, "globular", None, "s200_increase_10_300",
         "8 x 23.8 kDa pore (190 kDa) + ~70 kDa DDM detergent belt (~260 kDa)"),
    ]:
        col_c = SEC_COLUMNS[col_id]
        rh_p = predict_stokes_radius_rh(mw_ass, conf)
        ve_p, kav_p, st_n = calculate_retention_volume(mw_ass, col_c, conf)
        ve_exp = calculate_retention_volume(mw_ass, col_c, rh_override_a=exp_rh)[0] if exp_rh else None
        special_sec_assemblies.append({
            "assembly": label, "mw_kda": mw_ass, "conformation": conf, "column": col_c["name"],
            "pred_rh_a": round(rh_p, 1), "exp_rh_a": exp_rh,
            "pred_ve_ml": ve_p, "exp_ve_ml": ve_exp, "kav": kav_p, "note": note,
        })

    # AI-designed vs natural amino acid composition shift
    tot_ai = len(ai_all_seq)
    ai_aa_pct = {aa: round(100.0 * ai_all_seq.count(aa) / tot_ai, 2) for aa in AA_ORDER}
    nat_aa_pct = leakage_info["natural_train_aa_pct"]
    aa_shift = {aa: round(ai_aa_pct[aa] - nat_aa_pct[aa], 2) for aa in AA_ORDER}

    ss_overall = _compute_mcc_and_stats(conf_all)
    ss_gor4 = _compute_mcc_and_stats(conf_gor4)
    ss_cf = _compute_mcc_and_stats(conf_cf)

    by_fold_summary = {}
    for fb, cm in conf_by_fold.items():
        st = _compute_mcc_and_stats(cm)
        sub = [p for p in per_protein_results if (("Transmembrane De Novo" if p["pdb_id"] in ("6M6Z", "6X1K") else p["secondary_structure"]["obs_fold_class"]) == fb)]
        st["n_proteins"] = len(sub)
        st["mean_q3_pct"] = round(float(np.mean([p["secondary_structure"]["q3_pct"] for p in sub])), 2)
        st["mean_sov99_pct"] = round(float(np.mean([p["secondary_structure"]["sov99_pct"] for p in sub])), 2)
        by_fold_summary[fb] = st

    by_method_summary = {}
    for mg, cm in conf_by_method.items():
        st = _compute_mcc_and_stats(cm)
        sub = [p for p in per_protein_results if p["method_group"] == mg]
        st["n_proteins"] = len(sub)
        st["mean_q3_pct"] = round(float(np.mean([p["secondary_structure"]["q3_pct"] for p in sub])), 2)
        st["mean_sov99_pct"] = round(float(np.mean([p["secondary_structure"]["sov99_pct"] for p in sub])), 2)
        by_method_summary[mg] = st

    return {
        "panel_size": len(panel),
        "total_eval_residues": int(conf_all.sum()),
        "zero_leakage_vs_natural_train": {
            "exact_sequence_overlaps": leakage_info["exact_sequence_overlaps"],
            "max_5mer_containment": leakage_info["max_5mer_containment"],
            "max_5mer_jaccard": leakage_info["max_5mer_jaccard"],
            "any_pair_ge_0_30": leakage_info["any_pair_ge_0_30"],
        },
        "aa_composition_comparison": {
            "ai_designed_pct": ai_aa_pct,
            "natural_train_pct": nat_aa_pct,
            "shift_ai_minus_natural_pct": aa_shift,
        },
        "secondary_structure_summary": {
            "propurify_ss": {
                **ss_overall,
                "mean_protein_q3_pct": round(float(np.mean(q3_list)), 2),
                "median_protein_q3_pct": round(float(np.median(q3_list)), 2),
                "std_protein_q3_pct": round(float(np.std(q3_list)), 2),
                "mean_sov99_pct": round(float(np.mean(sov_list)), 2),
                "fold_class_accuracy_pct": round(100.0 * fold_matches / len(panel), 2),
                "composition_mae": {
                    "helix_mae": round(float(np.mean(err_h)), 2),
                    "strand_mae": round(float(np.mean(err_e)), 2),
                    "coil_mae": round(float(np.mean(err_c)), 2),
                },
                "composition_signed_bias_pred_minus_obs": {
                    "helix_bias": round(float(np.mean(bias_h)), 2),
                    "strand_bias": round(float(np.mean(bias_e)), 2),
                    "coil_bias": round(float(np.mean(bias_c)), 2),
                },
                "mean_latency_ms": round(float(np.mean(latencies_ms)), 2),
            },
            "gor4_baseline": {
                **ss_gor4,
                "mean_protein_q3_pct": round(float(np.mean(gor4_q3_list)), 2),
            },
            "chou_fasman_baseline": {
                **ss_cf,
                "mean_protein_q3_pct": round(float(np.mean(cf_q3_list)), 2),
            },
            "natural_11k_reference": {
                "pooled_q3_pct": 69.31,
                "mean_protein_q3_pct": 69.50,
                "mean_sov99_pct": 59.78,
                "fold_class_accuracy_pct": 81.48,
                "helix_mae": 7.84, "strand_mae": 6.58, "coil_mae": 8.50,
            },
            "by_fold_class": by_fold_summary,
            "by_design_method": by_method_summary,
        },
        "purification_metrics_audit": {
            "zero_a280_proteins": zero_a280_ids,
            "low_a280_proteins": low_a280_ids,
            "instability_index_false_unstable_count": len(false_unstable_ids),
            "instability_index_false_unstable_pct": round(100.0 * len(false_unstable_ids) / len(panel), 1),
            "instability_index_false_unstable_details": false_unstable_ids,
            "wilkinson_harrison_breakdown": {
                "high_solubility_count": len(wh_soluble_ids),
                "borderline_count": len(wh_borderline_ids),
                "elevated_inclusion_body_risk_count": len(wh_insoluble_ids),
                "inclusion_body_false_alarm_details": wh_insoluble_ids,
                "borderline_details": wh_borderline_ids,
            },
        },
        "buffer_advisor_audit": {
            "total_discrepancies_found": len(buffer_discrepancies),
            "discrepancies": buffer_discrepancies,
        },
        "sec_simulator_audit": {
            "elongated_shape_shifts_ge_0_45ml": sec_discrepancies,
            "special_oligomer_and_nanocage_checks": special_sec_assemblies,
            "zero_a280_uv280_invisible_flagged_count": sum(1 for p in per_protein_results if p["sec_and_sds"]["sec_uv280_invisible_flag"]),
            "zero_a280_workflow_alert_count": sum(1 for p in per_protein_results if p["buffer_advisor"]["uv280_invisible_alert_present"]),
            "ii_denovo_caveat_count_named": sum(1 for p in per_protein_results if p["buffer_advisor"]["ii_denovo_caveat_present"]),
        },
        "per_protein_results": per_protein_results,
    }


def build_full_app_parameter_taxonomy() -> List[Dict[str, Any]]:
    """
    Exhaustive Question B parameter-by-parameter audit across every ProPurify Studio module,
    distinguishing exact closed-form physics/chemistry formulas from low-DOF physical calibrations,
    deterministic expert rules, fixed published literature heuristics, and the single locally
    machine-learned model (`modules/secondary_structure.py`).
    """
    return [
        {
            "module": "modules/purification_metrics.py",
            "parameters": ["mw_da", "mw_kda"],
            "category": "Exact Closed-Form Physics/Chemistry",
            "trainable_10x_applicable": False,
            "rationale": "Exact IUPAC average isotopic residue mass sum + 18.01528 Da (H2O). Zero free parameters; exact to 0.01 Da for both natural and AI-designed polypeptides.",
        },
        {
            "module": "modules/purification_metrics.py",
            "parameters": ["pi", "pi_oxidized", "net_charge_at_ph", "charge_ph_profile"],
            "category": "Exact Closed-Form Physics/Chemistry",
            "trainable_10x_applicable": False,
            "rationale": "Exact Henderson-Hasselbalch bisection with Bjellqvist et al. (1993, 1994) pKa constants matching ExPASy ProtParam. Not trained on a corpus.",
        },
        {
            "module": "modules/purification_metrics.py",
            "parameters": ["molar_extinction_reduced", "molar_extinction_cystines", "abs_01_reduced", "abs_01_cystines"],
            "category": "Exact Closed-Form Spectroscopy",
            "trainable_10x_applicable": False,
            "rationale": "Exact Pace et al. (1995) Beer-Lambert sum (5500*nW + 1490*nY + 125*nSS). Zero-Trp/Tyr AI designs are handled via explicit UV-280 blindness warnings (214/205 nm), not retraining.",
        },
        {
            "module": "modules/purification_metrics.py",
            "parameters": ["gravy", "aliphatic_index"],
            "category": "Fixed Published Biophysical Scales",
            "trainable_10x_applicable": False,
            "rationale": "Kyte & Doolittle (1982) hydropathy average and Ikai (1980) aliphatic index; standardized definitions with zero local training.",
        },
        {
            "module": "modules/purification_metrics.py",
            "parameters": ["instability_index", "solubility_heuristic_wilkinson_harrison"],
            "category": "Fixed Published Literature Heuristics",
            "trainable_10x_applicable": False,
            "rationale": "Guruprasad (1990) 400-dipeptide DIWV table (44 proteins) and Wilkinson-Harrison/Davis (1991/1999) 2-term discriminant (81 E. coli proteins). Modifying weights would break ExPASy ProtParam parity; instead, Glu/Lys-rich de novo caveats are surfaced directly in workflow alerts.",
        },
        {
            "module": "modules/buffer_advisor.py",
            "parameters": ["custom_buffer_stoichiometry", "pka_buffering_windows", "affinity_recipes", "iex_hic_sec_roadmap", "biochemical_profile_guards"],
            "category": "Exact Stoichiometry & Deterministic Biochemical Rules",
            "trainable_10x_applicable": False,
            "rationale": "Exact m = C*V*MW and V_liq = m/rho stoichiometry, IUPAC formula weights, Cytiva CIPP chromatography rules, and UniProt ChEBI/keyword guards. AI-designed edge cases (2KL8, 6D0T, 6X1K, zero-A280, Glu/Lys II) are solved deterministically by biophysical rules.",
        },
        {
            "module": "modules/sec_simulator.py",
            "parameters": ["stokes_radius_rh", "ackers_laurent_killander_ve_kav", "plate_theory_sigma", "dual_uv_a260_a280"],
            "category": "Low-DOF Calibrated Physical Hydrodynamics & Pore Partitioning",
            "trainable_10x_applicable": False,
            "rationale": "3-coeff quadratic log-MW Rh law (R^2 = 0.9950 across 1,254 PDB assemblies, 55 Cytiva calibration standards, MAE 0.17-0.22 mL) + 2-parameter Ackers erfc pore integral per column + Uversky (2002) coil + Tirado-Garcia de la Torre DNA rod equations. Already constrained below column-to-column variance (~0.2 mL); 10x more points cannot reduce physical shape/hydration variance.",
        },
        {
            "module": "modules/sds_page_simulator.py",
            "parameters": ["calculate_sds_rf", "precision_plus_ladder"],
            "category": "Low-DOF Physical Sieving Law",
            "trainable_10x_applicable": False,
            "rationale": "3-parameter Ferguson / Ogston-Rodbard-Chrambach sigmoidal sieving curve calibrated to Bio-Rad Precision Plus Dual Color markers across 4%-22% polyacrylamide.",
        },
        {
            "module": "modules/gel_recipe.py",
            "parameters": ["laemmli_tricine_gel_recipes", "linear_separation_ranges"],
            "category": "Exact Analytical Volumetric Stoichiometry",
            "trainable_10x_applicable": False,
            "rationale": "Exact C1*V1 = C2*V2 volumetric dilution with volume conservation (Sambrook/Bio-Rad Laemmli & Schagger 2006 Tricine). Zero learned parameters.",
        },
        {
            "module": "modules/protein_3d.py",
            "parameters": ["offline_3d_backbone_schematic", "pdb_mmcif_parser", "3dmol_viewer"],
            "category": "Deterministic 3D Coordinate Geometry & Parser",
            "trainable_10x_applicable": False,
            "rationale": "Deterministic helix/strand/coil parametric geometry and exact PDB/mmCIF parser (AlphaFold2/ESMFold/RFdiffusion/X-ray). Zero learned parameters.",
        },
        {
            "module": "modules/sequence_parser.py",
            "parameters": ["genetic_code_translation", "orf_finder", "terminal_tag_detection", "nl_uniprot_ncbi_parser"],
            "category": "Exact Genetic Code & Deterministic Parser",
            "trainable_10x_applicable": False,
            "rationale": "Exact NCBI Standard Genetic Code (transl_table=1, 64 codons) and deterministic API/regex resolution.",
        },
        {
            "module": "modules/secondary_structure.py",
            "parameters": ["gor4_17mer_tables", "stage1_ensemble_m1_m2_m3", "stage2_context_net", "viterbi_transitions"],
            "category": "Locally Machine-Learned Single-Sequence Neural Model (1,138,586 params)",
            "trainable_10x_applicable": True,
            "rationale": "THE ONLY machine-learned model in ProPurify Studio (currently trained on 3,751 PDB/DSSP chains out of 14,869 in data/ss_10k_corpus.npz). Evaluated empirically via controlled zero-leakage learning curve (N = 500 -> 1,000 -> 3,751 -> 7,500 -> 11,230) + 2x-width capacity ablation + power-law extrapolation to 10x (37,510 chains).",
        },
    ]


def main() -> int:
    out_path = os.path.join(ROOT_DIR, "data", "ai_designed_proteins_audit.json")
    print(f"Fetching {len(AI_DESIGNED_PANEL_SPEC)} real experimental AI-designed PDB structures from RCSB...")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as ex:
        panel = list(ex.map(fetch_pdb_record, AI_DESIGNED_PANEL_SPEC))
    print(f"Fetched all {len(panel)} experimental PDB structures + 3D mmCIF geometries in {time.perf_counter() - t0:.1f}s.")

    print("Running zero-leakage 5-mer check against natural train corpus (3,751 chains)...")
    leakage_info = check_zero_leakage_vs_train_corpus(panel)
    print(f"  Zero-leakage check: exact={leakage_info['exact_sequence_overlaps']}, "
          f"max_5mer_cont={leakage_info['max_5mer_containment']:.4f}, "
          f"max_5mer_jacc={leakage_info['max_5mer_jaccard']:.4f}")

    print("Running complete ProPurify module audit across all 32 AI-designed proteins...")
    qa_report = run_full_audit(panel, leakage_info)

    # Preserve existing question_b learning-curve block if already computed
    full_json: Dict[str, Any] = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                full_json = json.load(f)
        except Exception:
            full_json = {}

    full_json["generated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    full_json["question_a_ai_designed_audit"] = qa_report
    qb_existing = full_json.get("question_b_scaling_and_parameter_audit") or {}
    qb_existing["full_app_parameter_taxonomy"] = build_full_app_parameter_taxonomy()
    full_json["question_b_scaling_and_parameter_audit"] = qb_existing

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_json, f, indent=2)

    # Print crisp executive summary to stdout
    ss = qa_report["secondary_structure_summary"]["propurify_ss"]
    pm = qa_report["purification_metrics_audit"]
    ba = qa_report["buffer_advisor_audit"]
    sec = qa_report["sec_simulator_audit"]

    print("\n" + "=" * 82)
    print("QUESTION A SUMMARY — 32 REAL EXPERIMENTAL AI-DESIGNED / DE NOVO PDB PROTEINS")
    print("=" * 82)
    print(f"Secondary Structure: Pooled Q3={ss['pooled_q3_pct']:.2f}% | Mean Prot Q3={ss['mean_protein_q3_pct']:.2f}% "
          f"(vs Natural 11k: 69.31% / 69.50%)")
    print(f"  SOV_99={ss['mean_sov99_pct']:.2f}% (vs Nat 59.78%) | Fold-Class Acc={ss['fold_class_accuracy_pct']:.1f}% (vs Nat 81.48%)")
    print(f"  MCC(H/E/C)=({ss['MCC_H']:.3f}, {ss['MCC_E']:.3f}, {ss['MCC_C']:.3f}) | "
          f"GOR-IV Q3={qa_report['secondary_structure_summary']['gor4_baseline']['pooled_q3_pct']:.2f}% | "
          f"Chou-Fasman Q3={qa_report['secondary_structure_summary']['chou_fasman_baseline']['pooled_q3_pct']:.2f}%")
    print(f"  Comp MAE (H/E/C): {ss['composition_mae']} | Signed Bias: {ss['composition_signed_bias_pred_minus_obs']}")
    print("  By Fold Class:")
    for fb, st in qa_report["secondary_structure_summary"]["by_fold_class"].items():
        print(f"    {fb:30s} (n={st['n_proteins']:2d}): Pooled Q3={st['pooled_q3_pct']:5.2f}%, Mean Q3={st['mean_q3_pct']:5.2f}%, SOV_99={st['mean_sov99_pct']:5.2f}%")
    print("  By Design Method:")
    for mg, st in qa_report["secondary_structure_summary"]["by_design_method"].items():
        print(f"    {mg:30s} (n={st['n_proteins']:2d}): Pooled Q3={st['pooled_q3_pct']:5.2f}%, Mean Q3={st['mean_q3_pct']:5.2f}%, SOV_99={st['mean_sov99_pct']:5.2f}%")

    print("\nPurification Metrics Audit:")
    print(f"  Zero-A280 (W=0, Y=0) designs: {pm['zero_a280_proteins']}")
    print(f"  Low-A280 (<0.30) designs:     {pm['low_a280_proteins']}")
    print(f"  Instability Index >=40 ('Unstable' false alarm on crystallized hyperstable de novo): "
          f"{pm['instability_index_false_unstable_count']}/{qa_report['panel_size']} ({pm['instability_index_false_unstable_pct']}%)")
    wh = pm["wilkinson_harrison_breakdown"]
    print(f"  Wilkinson-Harrison Solubility: High={wh['high_solubility_count']}, Borderline={wh['borderline_count']}, "
          f"Inclusion-Body Risk={wh['elevated_inclusion_body_risk_count']}")

    print(f"\nBuffer Advisor Audit: {ba['total_discrepancies_found']} discrepancies found:")
    for d in ba["discrepancies"]:
        print(" ", d)

    print(f"\nSEC Simulator Audit: {len(sec['elongated_shape_shifts_ge_0_45ml'])} elongated proteins shift >=0.45 mL vs globular:")
    for s in sec["elongated_shape_shifts_ge_0_45ml"]:
        print(f"  {s['pdb_id']} {s['name']:18s} ({s['mw_kda']:5.1f} kDa, p={s['aspect_ratio_p']}, f/f0={s['perrin_f_f0']}): "
              f"Rh_glob={s['rh_glob_a']} vs Rh_3D={s['rh_3d_perrin_a']} Å | Ve_glob={s['ve_glob_ml']} vs Ve_3D={s['ve_3d_ml']} mL (dVe={s['delta_ve_ml']:+.2f} mL, prof={s['recommended_profile']}->{s['ve_shape_profile_ml']} mL)")

    print("\nPer-Protein Secondary Structure Table:")
    for p in qa_report["per_protein_results"]:
        s = p["secondary_structure"]
        m = p["metrics"]
        print(f"  {p['pdb_id']} {p['short_name']:22s} L={p['length_core']:3d} | Q3={s['q3_pct']:5.1f}% SOV={s['sov99_pct']:5.1f}% | "
              f"Obs(H/E)={s['obs_helix_pct']:4.1f}/{s['obs_strand_pct']:4.1f} Pred={s['pred_helix_pct']:4.1f}/{s['pred_strand_pct']:4.1f} "
              f"FoldOK={str(s['fold_match']):5s} | II={m['instability_index']:5.1f} GRAVY={m['gravy']:+5.2f} WH={m['wh_cv']:+5.2f}")

    print(f"\nSaved full Question A report to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
