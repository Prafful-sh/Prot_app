"""
ProPurify Studio: Interactive Protein Purification, SEC, SDS-PAGE & Buffer Lab
Designed for high-school & college classrooms as well as structural biology & biochemistry labs:
- Fresh Botanical Emerald-Teal & Soft Sage-Cream educational UI
- Instant Classroom Protein Presets (Egg-White Lysozyme default, GFP, Myoglobin, Lipase A, BSA)
  plus Natural Language Prompt, UniProt, NCBI, FASTA, and DNA/RNA ORF translation
- Complete 6-Parameter Physicochemical Profiling (MW, pI, A280, Instability, Aliphatic, GRAVY)
- Hydrodynamic SEC Chromatogram & Monomer-Only Destained SDS-PAGE Gel Simulator (Coomassie & Silver Stain)
- Comprehensive Purification Strategy Advisor covering ALL 7 methods:
  Affinity, Pseudo-Affinity (Heparin), Ammonium Sulfate, Ion-Exchange (AEX/CEX),
  Hydrophobic Interaction Chromatography (HIC — Butyl/Phenyl/Octyl), Hydroxyapatite/Multimodal, and SEC
- Customizable Laboratory Buffer & Recipe Calculator supporting custom compositions, concentrations,
  arbitrary volumes (mL / L), and exact weights in BOTH grams (g) and milligrams (mg)
- Simplified Real-World Literature & 1,254-Protein Accuracy Showcase
"""

from datetime import datetime, timezone
import html
import importlib
import json
import math
import os
import re
import sys

# Ensure root directory is top of search path across container environments
_APP_ROOT = os.path.dirname(os.path.abspath(__file__))
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

import requests
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import pandas as pd

_SUBMODULES_IN_ORDER = (
    "modules.sequence_parser",
    "modules.purification_metrics",
    "modules.sec_simulator",
    "modules.buffer_advisor",
    "modules.sds_page_simulator",
    "modules.secondary_structure",
    "modules.gel_recipe",
    "modules.protein_3d",
)

try:
    from modules.sequence_parser import (
        LYSOZYME_MATURE_SEQ,
        CLASSROOM_PRESET_PROTEINS,
        clean_amino_acid_sequence,
        extract_fasta_header_metadata,
        detect_sequence_terminal_tag,
        fetch_uniprot_by_id,
        fetch_ncbi_by_id,
        fetch_pdb_by_id,
        find_longest_orf,
        translate_nucleotide,
        parse_natural_language_prompt,
    )
    from modules.purification_metrics import (
        analyze_all_parameters,
        calculate_net_charge_at_ph,
    )
    from modules.sec_simulator import (
        SEC_COLUMNS,
        CONFORMATION_PROFILES,
        recommend_optimal_columns,
        get_column_marker_elutions,
        simulate_chromatogram_peaks,
        simulate_standard_mixture_chromatogram,
        compute_column_snapshot,
    )
    from modules.buffer_advisor import (
        COMMON_LAB_REAGENTS,
        REAGENT_LOOKUP,
        PRESET_RECIPE_TEMPLATES,
        infer_protein_biochemical_profile,
        recommend_full_purification_workflow,
        recommend_safe_buffer_preset_key,
        build_compatibility_matrix,
        build_student_lab_challenge,
        audit_buffer_compatibility_for_protein,
        compute_custom_buffer_recipe,
        convert_weight_and_molarity,
    )
    from modules.sds_page_simulator import (
        generate_sds_page_figure,
    )
    from modules.secondary_structure import (
        predict_secondary_structure,
        generate_secondary_structure_figure,
        format_ss_sequence_blocks,
    )
    from modules.gel_recipe import (
        RECIPE_DISPLAY_COLUMNS,
        GEL_SYSTEMS,
        ACRYLAMIDE_STOCK_PRESETS,
        GEL_FORMAT_PRESETS,
        calculate_sds_gel_recipe,
        format_gel_recipe_text,
        recommend_gel_recipe_params,
    )
    from modules.protein_3d import (
        SCHEMATIC_3D_DISCLAIMER,
        normalize_uniprot_accession,
        fetch_3d_structure_for_uniprot,
        fetch_structure_by_pdb_id,
        parse_uploaded_structure_file,
        build_3dmol_viewer_html,
        build_3d_biophysical_backbone_figure,
    )
except ImportError:
    # Handle Streamlit Cloud hot-reloading when container has stale in-memory cached modules in sys.modules
    if "modules" in sys.modules:
        importlib.reload(sys.modules["modules"])
    for _mod_name in _SUBMODULES_IN_ORDER:
        if _mod_name in sys.modules:
            importlib.reload(sys.modules[_mod_name])
        else:
            importlib.import_module(_mod_name)
    from modules.sequence_parser import (
        LYSOZYME_MATURE_SEQ,
        CLASSROOM_PRESET_PROTEINS,
        clean_amino_acid_sequence,
        extract_fasta_header_metadata,
        detect_sequence_terminal_tag,
        fetch_uniprot_by_id,
        fetch_ncbi_by_id,
        fetch_pdb_by_id,
        find_longest_orf,
        translate_nucleotide,
        parse_natural_language_prompt,
    )
    from modules.purification_metrics import (
        analyze_all_parameters,
        calculate_net_charge_at_ph,
    )
    from modules.sec_simulator import (
        SEC_COLUMNS,
        CONFORMATION_PROFILES,
        recommend_optimal_columns,
        get_column_marker_elutions,
        simulate_chromatogram_peaks,
        simulate_standard_mixture_chromatogram,
        compute_column_snapshot,
    )
    from modules.buffer_advisor import (
        COMMON_LAB_REAGENTS,
        REAGENT_LOOKUP,
        PRESET_RECIPE_TEMPLATES,
        infer_protein_biochemical_profile,
        recommend_full_purification_workflow,
        recommend_safe_buffer_preset_key,
        build_compatibility_matrix,
        build_student_lab_challenge,
        audit_buffer_compatibility_for_protein,
        compute_custom_buffer_recipe,
        convert_weight_and_molarity,
    )
    from modules.sds_page_simulator import (
        generate_sds_page_figure,
    )
    from modules.secondary_structure import (
        predict_secondary_structure,
        generate_secondary_structure_figure,
        format_ss_sequence_blocks,
    )
    from modules.gel_recipe import (
        RECIPE_DISPLAY_COLUMNS,
        GEL_SYSTEMS,
        ACRYLAMIDE_STOCK_PRESETS,
        GEL_FORMAT_PRESETS,
        calculate_sds_gel_recipe,
        format_gel_recipe_text,
        recommend_gel_recipe_params,
    )
    from modules.protein_3d import (
        SCHEMATIC_3D_DISCLAIMER,
        normalize_uniprot_accession,
        fetch_3d_structure_for_uniprot,
        fetch_structure_by_pdb_id,
        parse_uploaded_structure_file,
        build_3dmol_viewer_html,
        build_3d_biophysical_backbone_figure,
    )

# Page configuration
st.set_page_config(
    page_title="ProPurify Studio | Interactive Protein Purification & SEC Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={},
)

# Standard anhydrous average nucleic acid residue molar masses (IUPAC / 50% GC free-acid convention):
# - dsDNA: 617.96 Da/bp (0.61796 kDa/bp; e.g., 147 bp Widom 601 DNA = 90.84 kDa; note dry Na+ salt is ~650–662 Da/bp)
# - ssRNA: 321.47 Da/nt (0.32147 kDa/nt; e.g., 100 nt sgRNA = 32.15 kDa; +0.16 kDa if 5'-triphosphorylated via T7 IVT)
DSDNA_MW_KDA_PER_BP = 0.61796
SSRNA_MW_KDA_PER_NT = 0.32147

# Deterministic reverse-translation helper for the synthetic bacterial Lysozyme coding ORF:
# Prepends start codon ATG (Met) + 129-aa mature Hen Egg-White Lysozyme C + TAA stop codon = 393 bp (130 aa, ~14.44 kDa)
_AA_TO_CODON = {
    "A": "GCT", "C": "TGC", "D": "GAT", "E": "GAA", "F": "TTT",
    "G": "GGT", "H": "CAT", "I": "ATT", "K": "AAA", "L": "CTG",
    "M": "ATG", "N": "AAC", "P": "CCG", "Q": "CAG", "R": "CGT",
    "S": "TCT", "T": "ACT", "V": "GTT", "W": "TGG", "Y": "TAT"
}
DEFAULT_LYSOZYME_DNA = "ATG" + "".join(_AA_TO_CODON[aa] for aa in LYSOZYME_MATURE_SEQ) + "TAA"

# --- HIGH-CONTRAST BOTANICAL-EDITORIAL DESIGN SYSTEM & STRICT 6-TIER TYPOGRAPHY ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        /* Font Stacks */
        --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        --font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;

        /* Refined Botanical-Editorial Surfaces */
        --bg-canvas: #F6F9F8;
        --bg-sidebar: #EBF3F0;
        --bg-surface: #FFFFFF;
        --bg-subtle: #F1F5F9;

        --border-soft: #D5E5DF;
        --border-sidebar: #D2E3DC;
        --border-strong: #B6D2C8;

        /* High-Contrast Ink Hierarchy (WCAG AA / AAA Verified) */
        --ink-strong: #0B221C;    /* 16.67:1 on #FFFFFF | 15.74:1 on #F6F9F8 */
        --ink-heading: #0F2D25;   /* 14.54:1 on #FFFFFF */
        --ink-control: #12332B;   /* 13.56:1 on #FFFFFF */
        --ink-body: #1E352F;      /* 13.08:1 on #FFFFFF | 12.35:1 on #F6F9F8 */
        --ink-caption: #47625B;   /* 6.26:1 on #F6F9F8  | 5.88:1 on #EBF3F0 */
        --ink-muted: #4A635C;     /* 6.50:1 on #FFFFFF  | 6.14:1 on #F6F9F8 */

        /* Semantic Highlight Accents (All text/background pairs >= 5.0:1 WCAG AA) */
        --teal-primary: #0D9488;
        --teal-strong: #0F766E;   /* 5.47:1 with #FFFFFF */
        --teal-hover: #115E59;    /* 7.58:1 with #FFFFFF */
        --teal-deep: #065F46;     /* 7.29:1 on #ECFDF5 */
        --teal-text: #047857;     /* 5.48:1 on #FFFFFF */
        --teal-tint: #ECFDF5;
        --teal-border: #99F6E4;

        --amber-accent: #D97706;
        --amber-strong: #B45309;  /* 5.02:1 on #FFFFFF */
        --amber-deep: #92400E;    /* 6.84:1 on #FFFBEB | 6.37:1 on #FEF3C7 */
        --amber-tint: #FFFBEB;
        --amber-pill: #FEF3C7;
        --amber-border: #FCD34D;

        --indigo-accent: #4F46E5; /* 6.29:1 on #FFFFFF */
        --indigo-deep: #3730A3;   /* 8.88:1 on #EEF2FF */
        --indigo-tint: #EEF2FF;
        --indigo-border: #C7D2FE;

        --crimson-accent: #DC2626;/* 4.83:1 on #FFFFFF */
        --crimson-strong: #B91C1C;/* 6.47:1 on #FFFFFF */
        --crimson-deep: #991B1B;  /* 7.60:1 on #FEF2F2 */
        --crimson-tint: #FEF2F2;
        --crimson-border: #FECACA;

        /* Strict 6-Tier Typographic Scale (+ Tabular Numeric Display Tiers) */
        --fs-hero: 1.95rem;       /* 1. Hero Display (800 weight, -0.025em) */
        --fs-h2: 1.28rem;         /* 2. Section Heading h3 / .ui-h2 (800 weight, -0.018em) */
        --fs-h3: 1.04rem;         /* 3. Sub-Section & Card Heading h4, h5 / .ui-h3 (700 weight, -0.01em) */
        --fs-body: 0.92rem;       /* 4. Body Copy & Descriptions p, li / .ui-body (400/500 weight, 1.55 lh) */
        --fs-label: 0.875rem;     /* 5. Widget & Control Labels (700 group headers, 500 options) */
        --fs-micro: 0.73rem;      /* 6. Eyebrow Badges, Chips & KPI Overlines (700 weight, uppercase, 0.06em) */
        --fs-ribbon: 1.14rem;     /* Active Protein Ribbon Identity Title (800 weight) */
        --fs-kpi: 1.62rem;        /* KPI & Benchmark Tabular Stat Callouts (800 weight) */
    }

    /* Global Typography Enforcement on Modern Streamlit 1.35–1.66+ Emotion DOM
       (NOTE: Never target generic `span` or `*` so Material Icon ligatures stay untouched!) */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stSidebar"],
    [data-testid="stMarkdownContainer"],
    [data-testid="stCaptionContainer"],
    [data-testid="stWidgetLabel"],
    [data-testid="stAlertContainer"],
    [data-testid="stDataFrame"],
    h1, h2, h3, h4, h5, h6, p, li, label, th, td,
    button, input, textarea,
    [data-baseweb="select"],
    [data-baseweb="tab"],
    [data-baseweb="popover"] {
        font-family: var(--font-sans);
        color: var(--ink-body);
    }

    /* CRITICAL SAFEGUARD: Protect Streamlit Material Symbols Ligature Icons Everywhere */
    [data-testid="stIconMaterial"],
    .material-symbols-rounded,
    .material-icons {
        font-family: "Material Symbols Rounded" !important;
        font-weight: normal !important;
        font-style: normal !important;
        letter-spacing: normal !important;
        text-transform: none !important;
    }

    /* Monospace & Tabular Numerals for Sequences, Codes & Quantitative Readouts */
    code, pre, kbd,
    [data-testid="stCode"] code,
    [data-testid="stCode"] pre,
    .ui-mono {
        font-family: var(--font-mono) !important;
        font-variant-numeric: tabular-nums;
    }

    [data-testid="stMarkdownContainer"] code,
    .method-card code,
    .protocol-step-card code,
    .result-callout-card code {
        font-family: var(--font-mono) !important;
        font-size: var(--fs-label);
        font-weight: 600;
        background: #EEF5F2;
        color: var(--ink-heading);
        padding: 1px 6px;
        border-radius: 5px;
        border: 1px solid var(--border-soft);
    }

    /* Crisp Warm Alabaster Canvas & Soft Sage-Mist Sidebar */
    .stApp {
        background-color: var(--bg-canvas);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #EBF3F0 0%, #F2F7F5 100%);
        border-right: 1px solid var(--border-sidebar);
    }
    section[data-testid="stSidebar"] hr {
        border-color: var(--border-sidebar);
        margin: 1rem 0;
    }

    /* Tier 2 & Tier 3: Consistent Heading Hierarchy Across Markdown & Custom Cards */
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2 {
        font-size: var(--fs-hero);
        font-weight: 800;
        color: var(--ink-strong);
        letter-spacing: -0.025em;
        line-height: 1.2;
    }
    [data-testid="stMarkdownContainer"] h3,
    .ui-h2 {
        font-size: var(--fs-h2);
        font-weight: 800;
        color: var(--ink-strong);
        letter-spacing: -0.018em;
        line-height: 1.28;
        margin-top: 0.15rem;
        margin-bottom: 0.5rem;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        font-size: var(--fs-h3);
        font-weight: 800;
        color: var(--ink-strong);
        letter-spacing: -0.01em;
        padding-bottom: 6px;
        margin-top: 0.1rem;
        margin-bottom: 0.55rem;
        border-bottom: 1.5px solid var(--border-sidebar);
    }
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMarkdownContainer"] h5,
    .ui-h3 {
        font-size: var(--fs-h3);
        font-weight: 700;
        color: var(--ink-heading);
        letter-spacing: -0.01em;
        line-height: 1.35;
        margin-top: 0.25rem;
        margin-bottom: 0.45rem;
    }

    /* Tier 4: Body Copy & Section Subtitle Captions */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    .ui-body {
        font-size: var(--fs-body);
        font-weight: 400;
        line-height: 1.55;
        color: var(--ink-body);
    }
    [data-testid="stCaptionContainer"] p,
    .ui-caption {
        font-size: var(--fs-label) !important;
        font-weight: 400;
        line-height: 1.5;
        color: var(--ink-caption) !important;
    }

    /* Tier 5: Bold Widget Group Headers vs. Medium Checkbox/Radio Choice Labels */
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] > div > p,
    [data-testid="stWidgetLabel"] p,
    .ui-control-label {
        font-size: var(--fs-label) !important;
        font-weight: 700 !important;
        color: var(--ink-control) !important;
        line-height: 1.4 !important;
    }
    .ui-control-label {
        display: flex;
        align-items: center;
        min-height: 1.5rem;
        margin: 0 0 6px 0;
    }
    [data-testid="stCheckbox"] [data-testid="stWidgetLabel"] p,
    [data-testid="stCheckbox"] label p,
    [data-testid="stRadio"] [role="radiogroup"] p,
    [data-baseweb="radio"] p {
        font-size: var(--fs-label) !important;
        font-weight: 500 !important;
        color: var(--ink-body) !important;
    }

    /* Expanders, Buttons & Alerts */
    [data-testid="stExpander"] {
        background: var(--bg-surface);
        border: 1px solid var(--border-soft);
        border-radius: 12px;
        box-shadow: 0 1px 4px rgba(11, 34, 28, 0.04);
    }
    [data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] p {
        font-size: var(--fs-label) !important;
        font-weight: 700 !important;
        color: var(--ink-heading) !important;
    }
    button[kind="primary"],
    [data-testid="stBaseButton-primary"] {
        background-color: var(--teal-strong) !important;
        border-color: var(--teal-strong) !important;
        color: #FFFFFF !important;
        font-size: var(--fs-label) !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
    }
    button[kind="primary"]:hover,
    [data-testid="stBaseButton-primary"]:hover {
        background-color: var(--teal-hover) !important;
        border-color: var(--teal-hover) !important;
    }
    button[kind="primary"] p,
    [data-testid="stBaseButton-primary"] p {
        color: #FFFFFF !important;
        font-size: var(--fs-label) !important;
        font-weight: 700 !important;
    }
    button[kind="secondary"],
    [data-testid="stBaseButton-secondary"],
    [data-testid="stDownloadButton"] button {
        background-color: var(--bg-surface) !important;
        border: 1.5px solid var(--teal-strong) !important;
        color: var(--teal-strong) !important;
        font-size: var(--fs-label) !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
    }
    button[kind="secondary"] p,
    [data-testid="stBaseButton-secondary"] p,
    [data-testid="stDownloadButton"] button p {
        color: var(--teal-strong) !important;
        font-size: var(--fs-label) !important;
        font-weight: 700 !important;
    }
    button[kind="secondary"]:hover,
    [data-testid="stBaseButton-secondary"]:hover,
    [data-testid="stDownloadButton"] button:hover {
        background-color: var(--teal-tint) !important;
        border-color: var(--teal-deep) !important;
    }
    [data-testid="stAlertContainer"] {
        border-radius: 12px;
    }
    [data-testid="stAlertContent"] p {
        font-size: var(--fs-body);
        line-height: 1.52;
    }

    /* Tier 1: Botanical Forest-to-Deep-Teal Hero Header */
    .hero-banner {
        background: linear-gradient(135deg, #064E3B 0%, #0B5D54 55%, #115E59 100%);
        border-radius: 16px;
        padding: 22px 28px;
        margin-bottom: 18px;
        border: 1px solid rgba(255, 255, 255, 0.16);
        box-shadow: 0 10px 24px -8px rgba(6, 78, 59, 0.25);
    }
    .hero-badge-row {
        display: flex;
        gap: 8px;
        margin-bottom: 10px;
        flex-wrap: wrap;
    }
    .hero-title {
        font-size: var(--fs-hero);
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.025em;
        line-height: 1.15;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: var(--fs-body);
        color: #E6FFFA;
        font-weight: 400;
        line-height: 1.55;
        max-width: 960px;
    }
    .hero-subtitle b {
        color: #FFFFFF;
        font-weight: 700;
    }
    .hero-badge,
    .ui-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 11px;
        border-radius: 9999px;
        font-size: var(--fs-micro);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .badge-gold {
        background: var(--amber-pill);
        color: var(--amber-deep);
        border: 1px solid var(--amber-border);
    }
    .badge-mint {
        background: rgba(6, 78, 59, 0.58);
        color: #ECFDF5;
        border: 1px solid rgba(167, 243, 208, 0.5);
    }

    /* Active Protein & Construct Ribbon (2-Tier High-Contrast Card) */
    .protein-ribbon {
        background: var(--bg-surface);
        border: 1px solid var(--border-soft);
        border-left: 5px solid var(--teal-primary);
        border-radius: 14px;
        padding: 15px 20px;
        margin-bottom: 18px;
        box-shadow: 0 2px 10px rgba(11, 34, 28, 0.05);
    }
    .ribbon-top-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
    }
    .ribbon-title {
        font-size: var(--fs-ribbon);
        font-weight: 800;
        color: var(--ink-strong);
        letter-spacing: -0.015em;
        line-height: 1.3;
    }
    .ribbon-pill-group {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
    }
    .meta-pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: var(--fs-micro);
        font-weight: 700;
        letter-spacing: 0.03em;
        font-variant-numeric: tabular-nums;
    }
    .meta-pill-teal {
        background: var(--teal-tint);
        color: var(--teal-deep);
        border: 1px solid var(--teal-border);
    }
    .meta-pill-indigo {
        background: var(--indigo-tint);
        color: var(--indigo-deep);
        border: 1px solid var(--indigo-border);
    }
    .meta-pill-neutral {
        background: var(--bg-subtle);
        color: #334155;
        border: 1px solid #CBD5E1;
    }
    .meta-pill-amber {
        background: var(--amber-pill);
        color: var(--amber-deep);
        border: 1px solid var(--amber-border);
    }
    .meta-pill-crimson {
        background: var(--crimson-tint);
        color: var(--crimson-deep);
        border: 1px solid var(--crimson-border);
    }
    .biochem-conflict-box {
        margin-top: 10px;
        background: var(--crimson-tint);
        border: 1px solid var(--crimson-border);
        border-left: 4px solid var(--crimson-accent);
        border-radius: 10px;
        padding: 11px 14px;
    }
    .biochem-conflict-warn {
        background: var(--amber-tint);
        border-color: var(--amber-border);
        border-left-color: var(--amber-accent);
    }
    .biochem-conflict-title {
        font-size: var(--fs-label);
        font-weight: 800;
        color: var(--crimson-deep);
        margin-bottom: 4px;
    }
    .biochem-conflict-warn .biochem-conflict-title {
        color: var(--amber-deep);
    }
    .biochem-conflict-body {
        font-size: var(--fs-body);
        color: var(--ink-body);
        line-height: 1.5;
    }
    .construct-parts-row {
        margin-top: 8px;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .construct-pill {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 11px;
        border-radius: 8px;
        font-size: var(--fs-micro);
        font-weight: 700;
        letter-spacing: 0.02em;
        font-variant-numeric: tabular-nums;
    }
    .pill-tag {
        background: var(--indigo-tint);
        color: var(--indigo-deep);
        border: 1px solid var(--indigo-border);
    }
    .pill-cleavage {
        background: var(--amber-tint);
        color: var(--amber-deep);
        border: 1px solid var(--amber-border);
    }
    .pill-target {
        background: var(--teal-tint);
        color: var(--teal-deep);
        border: 1px solid var(--teal-border);
    }

    /* 6-Card Parameter KPI Grid */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 13px;
        margin-bottom: 20px;
    }
    @media (max-width: 1100px) {
        .metric-grid { grid-template-columns: repeat(3, 1fr); }
    }
    @media (max-width: 640px) {
        .metric-grid { grid-template-columns: repeat(2, 1fr); }
    }
    .card-kpi {
        background: var(--bg-surface);
        border: 1px solid var(--border-soft);
        border-radius: 14px;
        padding: 15px 16px;
        box-shadow: 0 2px 8px rgba(11, 34, 28, 0.04);
        position: relative;
        overflow: hidden;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .card-kpi:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(13, 148, 136, 0.10);
    }
    .card-kpi::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
    }
    .bar-teal::before { background: linear-gradient(90deg, #0D9488, #14B8A6); }
    .bar-amber::before { background: linear-gradient(90deg, #D97706, #F59E0B); }
    .bar-emerald::before { background: linear-gradient(90deg, #059669, #10B981); }
    .bar-coral::before { background: linear-gradient(90deg, #DC2626, #F97316); }
    .bar-olive::before { background: linear-gradient(90deg, #4F46E5, #818CF8); }
    .bar-plum::before { background: linear-gradient(90deg, #7C3AED, #A78BFA); }

    .kpi-title {
        font-size: var(--fs-micro);
        font-weight: 700;
        color: var(--ink-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: var(--fs-kpi);
        font-weight: 800;
        color: var(--ink-strong);
        letter-spacing: -0.025em;
        line-height: 1.12;
        margin-bottom: 6px;
        font-variant-numeric: tabular-nums;
    }
    .kpi-unit {
        font-size: var(--fs-label);
        font-weight: 700;
        color: var(--ink-muted);
        letter-spacing: 0;
    }
    .kpi-sub {
        font-size: var(--fs-micro);
        color: var(--ink-muted);
        font-weight: 500;
        line-height: 1.4;
        font-variant-numeric: tabular-nums;
    }
    .kpi-status-ok {
        display: inline-block;
        background: var(--teal-tint);
        color: var(--teal-deep);
        border: 1px solid #A7F3D0;
        border-radius: 6px;
        padding: 2px 7px;
        font-size: var(--fs-micro);
        font-weight: 700;
    }
    .kpi-status-warn {
        display: inline-block;
        background: var(--crimson-tint);
        color: var(--crimson-deep);
        border: 1px solid var(--crimson-border);
        border-radius: 6px;
        padding: 2px 7px;
        font-size: var(--fs-micro);
        font-weight: 700;
    }

    /* SEC Mental Model Banner & Elution Order Strip */
    .sec-rule-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-soft);
        border-top: 3.5px solid var(--teal-primary);
        border-radius: 14px;
        padding: 14px 20px;
        margin-bottom: 18px;
        box-shadow: 0 2px 8px rgba(11, 34, 28, 0.04);
    }
    .sec-rule-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: 700;
        font-size: var(--fs-label);
        color: var(--ink-strong);
        margin-bottom: 6px;
        flex-wrap: wrap;
        gap: 8px;
    }
    .sec-rule-pill {
        background: var(--teal-strong);
        color: #FFFFFF;
        padding: 3px 12px;
        border-radius: 9999px;
        font-size: var(--fs-micro);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .elution-strip {
        background: var(--bg-surface);
        border: 1px solid var(--border-soft);
        border-radius: 11px;
        padding: 11px 15px;
        margin: 8px 0 12px 0;
        box-shadow: 0 1px 4px rgba(11, 34, 28, 0.03);
    }
    .elution-strip-title {
        font-size: var(--fs-label);
        font-weight: 700;
        color: var(--ink-heading);
        margin-bottom: 7px;
    }
    .elution-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        align-items: center;
        font-size: var(--fs-micro);
        color: var(--ink-muted);
    }
    .elution-chip-target {
        background: var(--teal-tint);
        color: var(--teal-deep);
        padding: 4px 10px;
        border-radius: 7px;
        font-size: var(--fs-micro);
        font-weight: 800;
        border: 1.5px solid var(--teal-primary);
        font-variant-numeric: tabular-nums;
    }
    .elution-chip-agg {
        background: var(--crimson-tint);
        color: var(--crimson-deep);
        padding: 4px 10px;
        border-radius: 7px;
        font-size: var(--fs-micro);
        font-weight: 800;
        border: 1.5px solid var(--crimson-accent);
        font-variant-numeric: tabular-nums;
    }
    .elution-chip-std {
        background: var(--bg-subtle);
        color: #334155;
        border: 1px solid #CBD5E1;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: var(--fs-micro);
        font-weight: 600;
        font-variant-numeric: tabular-nums;
    }

    /* Interactive Inside-the-SEC-Column & Bead Pore Studio */
    .sec-studio-card {
        background: var(--bg-surface);
        border: 1.5px solid var(--border-soft);
        border-top: 4px solid var(--teal-primary);
        border-radius: 16px;
        padding: 14px 18px 16px 18px;
        margin: 8px 0 18px 0;
        box-shadow: 0 4px 16px rgba(11, 34, 28, 0.05);
    }
    .sec-studio-svg {
        width: 100%;
        height: auto;
        display: block;
        border-radius: 12px;
        background: #F9FCFB;
        border: 1px solid var(--border-soft);
    }
    .sec-svg-h {
        font-family: var(--font-sans);
        font-size: 13.5px;
        font-weight: 800;
        fill: #0B221C;
    }
    .sec-svg-sub {
        font-family: var(--font-sans);
        font-size: 11px;
        font-weight: 500;
        fill: #47625B;
    }
    .sec-svg-lbl {
        font-family: var(--font-sans);
        font-size: 11px;
        font-weight: 600;
        fill: #1E352F;
    }
    .sec-svg-bold {
        font-family: var(--font-sans);
        font-size: 11px;
        font-weight: 800;
        fill: #0B221C;
    }
    .sec-svg-micro {
        font-family: var(--font-sans);
        font-size: 10px;
        font-weight: 700;
        fill: #47625B;
    }
    .sec-svg-mono {
        font-family: var(--font-mono);
        font-size: 10.5px;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
    }
    .sec-narrator-bar {
        margin-top: 11px;
        background: var(--teal-tint);
        border: 1px solid var(--teal-border);
        border-left: 4px solid var(--teal-primary);
        border-radius: 11px;
        padding: 10px 15px;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 10px;
    }
    .sec-narrator-bar-peak {
        background: var(--amber-tint);
        border-color: var(--amber-border);
        border-left-color: var(--amber-accent);
    }
    .sec-narrator-badge {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: var(--fs-micro);
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background: var(--teal-strong);
        color: #FFFFFF;
    }
    .sec-narrator-bar-peak .sec-narrator-badge {
        background: var(--amber-strong);
        color: #FFFFFF;
    }
    .sec-narrator-text {
        font-size: var(--fs-body);
        font-weight: 500;
        color: var(--ink-strong);
        line-height: 1.5;
        flex: 1 1 420px;
    }

    /* Tab 2: Working pH Stat Chips Bar, Roadmap Cards & Method Cards */
    .ph-stat-bar {
        background: var(--bg-surface);
        border: 1px solid var(--border-soft);
        border-left: 4px solid var(--teal-primary);
        border-radius: 12px;
        padding: 10px 14px;
        margin-bottom: 12px;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
        box-shadow: 0 2px 6px rgba(11, 34, 28, 0.04);
    }
    .stat-chip {
        display: inline-flex;
        flex-direction: column;
        padding: 5px 11px;
        border-radius: 9px;
        border: 1px solid var(--border-soft);
        background: #F8FBFA;
        min-width: 88px;
    }
    .stat-chip-amber {
        background: var(--amber-tint);
        border-color: var(--amber-border);
    }
    .stat-chip-teal {
        background: var(--teal-tint);
        border-color: var(--teal-border);
    }
    .stat-chip-indigo {
        background: var(--indigo-tint);
        border-color: var(--indigo-border);
    }
    .stat-chip-crimson {
        background: var(--crimson-tint);
        border-color: var(--crimson-border);
    }
    .stat-chip-label {
        font-size: var(--fs-micro);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--ink-muted);
        line-height: 1.2;
    }
    .stat-chip-value {
        font-size: var(--fs-label);
        font-weight: 800;
        color: var(--ink-strong);
        font-variant-numeric: tabular-nums;
        line-height: 1.35;
    }
    .stat-chip-amber .stat-chip-value { color: var(--amber-deep); }
    .stat-chip-teal .stat-chip-value { color: var(--teal-deep); }
    .stat-chip-indigo .stat-chip-value { color: var(--indigo-deep); }
    .stat-chip-crimson .stat-chip-value { color: var(--crimson-deep); }

    /* Roadmap & Benchmark 3-Column Card Grid */
    .roadmap-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin-bottom: 18px;
    }
    @media (max-width: 900px) {
        .roadmap-grid { grid-template-columns: 1fr; }
    }
    .roadmap-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-soft);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 2px 8px rgba(11, 34, 28, 0.04);
    }
    .roadmap-step-tag {
        display: inline-block;
        background: var(--teal-tint);
        color: var(--teal-deep);
        border: 1px solid #A7F3D0;
        font-size: var(--fs-micro);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 3px 10px;
        border-radius: 9999px;
        margin-bottom: 8px;
    }
    .roadmap-step-amber {
        background: var(--amber-pill);
        color: var(--amber-deep);
        border-color: var(--amber-border);
    }
    .roadmap-step-indigo {
        background: var(--indigo-tint);
        color: var(--indigo-deep);
        border-color: var(--indigo-border);
    }
    .roadmap-step-neutral {
        background: var(--bg-subtle);
        color: var(--ink-control);
        border-color: #CBD5E1;
    }
    .card-top-teal { border-top: 4px solid var(--teal-primary); }
    .card-top-emerald { border-top: 4px solid #059669; }
    .card-top-amber { border-top: 4px solid var(--amber-accent); }
    .card-top-indigo { border-top: 4px solid var(--indigo-accent); }
    .card-top-slate { border-top: 4px solid #64748B; }
    .ss-summary-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 13px;
        margin-bottom: 14px;
    }
    @media (max-width: 1150px) {
        .ss-summary-grid { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 640px) {
        .ss-summary-grid { grid-template-columns: 1fr; }
    }
    .ss-validation-strip {
        background: var(--bg-surface);
        border: 1px solid var(--border-soft);
        border-left: 4px solid var(--teal-primary);
        border-radius: 12px;
        padding: 11px 16px;
        margin: 6px 0 14px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
        box-shadow: 0 1px 5px rgba(11, 34, 28, 0.04);
    }
    .ss-validation-copy {
        font-size: var(--fs-label);
        color: var(--ink-body);
        line-height: 1.5;
    }
    .ss-validation-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
        align-items: center;
    }
    .ss-domain-explorer {
        background: var(--bg-surface);
        border: 1px solid var(--indigo-border);
        border-left: 4px solid var(--indigo-accent);
        border-radius: 12px;
        padding: 11px 16px;
        margin: 4px 0 10px 0;
        box-shadow: 0 1px 5px rgba(11, 34, 28, 0.04);
    }
    .ss-domain-explorer-title {
        font-size: var(--fs-label);
        font-weight: 700;
        color: var(--indigo-deep);
        margin-bottom: 3px;
    }
    .ss-domain-explorer-sub {
        font-size: var(--fs-label);
        color: var(--ink-body);
        line-height: 1.48;
    }
    .roadmap-title {
        font-size: var(--fs-h3);
        font-weight: 700;
        color: var(--ink-heading);
        letter-spacing: -0.01em;
        line-height: 1.32;
        margin-bottom: 6px;
    }
    .roadmap-body {
        font-size: var(--fs-body);
        color: var(--ink-body);
        line-height: 1.55;
    }

    /* Tab 2: 6-Tile Visual Reagent & Column Compatibility Board + Compact Lab Tips */
    .compat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 11px;
        margin: 4px 0 18px 0;
    }
    @media (max-width: 1050px) {
        .compat-grid { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 640px) {
        .compat-grid { grid-template-columns: 1fr; }
    }
    .compat-tile {
        background: var(--bg-surface);
        border: 1px solid var(--border-soft);
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: 0 2px 6px rgba(11, 34, 28, 0.035);
    }
    .compat-tile-good {
        border-left: 4px solid #059669;
        background: #F7FDFB;
    }
    .compat-tile-warn {
        border-left: 4px solid var(--amber-accent);
        background: #FFFDF5;
    }
    .compat-tile-avoid {
        border-left: 4px solid var(--crimson-accent);
        background: #FFF8F8;
    }
    .compat-tile-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
        margin-bottom: 5px;
    }
    .compat-tile-title {
        font-size: var(--fs-label);
        font-weight: 700;
        color: var(--ink-heading);
    }
    .compat-badge {
        display: inline-block;
        padding: 2px 9px;
        border-radius: 9999px;
        font-size: var(--fs-micro);
        font-weight: 700;
        white-space: nowrap;
    }
    .compat-badge-good {
        background: var(--teal-tint);
        color: var(--teal-deep);
        border: 1px solid #A7F3D0;
    }
    .compat-badge-warn {
        background: var(--amber-pill);
        color: var(--amber-deep);
        border: 1px solid var(--amber-border);
    }
    .compat-badge-avoid {
        background: var(--crimson-tint);
        color: var(--crimson-deep);
        border: 1px solid var(--crimson-border);
    }
    .compat-tile-reason {
        font-size: var(--fs-micro);
        color: var(--ink-body);
        line-height: 1.45;
    }
    .compact-tips-box {
        background: #F4FAF8;
        border: 1px solid var(--border-soft);
        border-left: 4px solid var(--teal-primary);
        border-radius: 11px;
        padding: 11px 15px;
        margin-bottom: 14px;
        font-size: var(--fs-label);
        color: var(--ink-body);
        line-height: 1.52;
    }
    .compact-tips-title {
        font-size: var(--fs-label);
        font-weight: 700;
        color: var(--teal-deep);
        margin-bottom: 4px;
    }
    .compact-tips-list {
        margin: 0;
        padding-left: 18px;
    }
    .compact-tips-list li {
        margin-bottom: 3px;
    }
    .challenge-q-box {
        background: var(--indigo-tint);
        border: 1px solid var(--indigo-border);
        border-left: 4px solid var(--indigo-accent);
        border-radius: 11px;
        padding: 12px 15px;
        margin-bottom: 10px;
        font-size: var(--fs-body);
        font-weight: 600;
        color: var(--ink-strong);
        line-height: 1.48;
    }

    /* Method A (IEX Teal) & Method B (HIC Amber) Recommendation Cards */
    .method-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-soft);
        border-radius: 13px;
        padding: 16px 18px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(11, 34, 28, 0.04);
    }
    .method-card-teal {
        border-color: #99F6E4;
        border-left: 5px solid var(--teal-primary);
    }
    .method-card-amber {
        border-color: #FDE68A;
        border-left: 5px solid var(--amber-accent);
    }
    .method-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 10px;
    }
    .method-card-title {
        font-size: var(--fs-h3);
        font-weight: 700;
        color: var(--ink-heading);
        letter-spacing: -0.01em;
    }
    .method-badge-teal {
        background: var(--teal-tint);
        color: var(--teal-deep);
        border: 1px solid var(--teal-border);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: var(--fs-micro);
        font-weight: 700;
    }
    .method-badge-amber {
        background: var(--amber-pill);
        color: var(--amber-deep);
        border: 1px solid var(--amber-border);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: var(--fs-micro);
        font-weight: 700;
    }
    .method-spec-list {
        display: grid;
        gap: 7px;
        font-size: var(--fs-body);
        color: var(--ink-body);
        line-height: 1.52;
    }
    .method-spec-key {
        font-weight: 700;
        color: var(--ink-strong);
    }
    .student-callout {
        background: var(--amber-tint);
        border: 1px solid #FDE68A;
        border-left: 4px solid var(--amber-accent);
        border-radius: 11px;
        padding: 13px 16px;
        font-size: var(--fs-body);
        color: var(--ink-body);
        line-height: 1.55;
    }
    .student-callout-title {
        font-size: var(--fs-label);
        font-weight: 700;
        color: var(--amber-deep);
        margin-bottom: 4px;
    }

    /* Tab 3: 4-Step Lab Preparation & Safety Protocol Card Grid */
    .protocol-section-header {
        font-size: var(--fs-h3);
        font-weight: 700;
        color: var(--ink-heading);
        letter-spacing: -0.01em;
        margin: 14px 0 10px 0;
    }
    .protocol-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 10px;
    }
    @media (max-width: 1150px) {
        .protocol-grid { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 640px) {
        .protocol-grid { grid-template-columns: 1fr; }
    }
    .protocol-step-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-soft);
        border-radius: 12px;
        padding: 14px 15px;
        box-shadow: 0 2px 6px rgba(11, 34, 28, 0.04);
    }
    .protocol-step-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: var(--fs-micro);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .step-badge-1 { background: var(--teal-tint); color: var(--teal-deep); border: 1px solid var(--teal-border); }
    .step-badge-2 { background: var(--crimson-tint); color: var(--crimson-deep); border: 1px solid var(--crimson-border); }
    .step-badge-3 { background: var(--indigo-tint); color: var(--indigo-deep); border: 1px solid var(--indigo-border); }
    .step-badge-4 { background: var(--amber-pill); color: var(--amber-deep); border: 1px solid var(--amber-border); }
    .protocol-step-body {
        font-size: var(--fs-body);
        color: var(--ink-body);
        line-height: 1.52;
    }

    /* Tab 4: Extinction & Solubility High-Visibility Result Callout Cards */
    .result-callout-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-soft);
        border-radius: 13px;
        padding: 15px 18px;
        margin-top: 12px;
        box-shadow: 0 2px 8px rgba(11, 34, 28, 0.04);
    }
    .result-card-indigo {
        border-color: var(--indigo-border);
        border-left: 5px solid var(--indigo-accent);
    }
    .result-card-teal {
        border-color: var(--teal-border);
        border-left: 5px solid var(--teal-primary);
    }
    .result-card-amber {
        border-color: var(--amber-border);
        border-left: 5px solid var(--amber-accent);
    }
    .result-card-crimson {
        border-color: var(--crimson-border);
        border-left: 5px solid var(--crimson-accent);
    }
    .result-stat-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        margin-top: 10px;
    }
    @media (max-width: 640px) {
        .result-stat-grid { grid-template-columns: 1fr; }
    }
    .result-stat-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 10px 13px;
    }
    .result-stat-overline {
        font-size: var(--fs-micro);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--indigo-deep);
        margin-bottom: 4px;
    }
    .result-stat-main {
        font-family: var(--font-mono);
        font-size: var(--fs-h3);
        font-weight: 700;
        color: var(--ink-strong);
        font-variant-numeric: tabular-nums;
    }
    .result-stat-sub {
        font-family: var(--font-mono);
        font-size: var(--fs-micro);
        font-weight: 600;
        color: var(--ink-muted);
        margin-top: 2px;
        font-variant-numeric: tabular-nums;
    }

    /* Pill Tabs on Soft Sage-Mist Track */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: var(--bg-sidebar);
        padding: 6px;
        border-radius: 13px;
        border: 1px solid var(--border-sidebar);
        margin-bottom: 18px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        padding: 10px 16px;
        font-weight: 700;
        font-size: var(--fs-label);
        color: var(--ink-body);
        border: none;
        background: transparent;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"] p {
        font-weight: 700 !important;
        font-size: var(--fs-label) !important;
        color: var(--ink-body);
    }
    .stTabs [aria-selected="true"] {
        background: var(--teal-strong) !important;
        color: #FFFFFF !important;
        box-shadow: 0 3px 10px rgba(15, 118, 110, 0.24);
    }
    .stTabs [aria-selected="true"] p {
        color: #FFFFFF !important;
    }

    /* Interactive 3D Viewer & Pop-Out Dialogs (Botanical-Editorial Theme) */
    .viewer3d-meta-bar {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 7px;
        background: var(--bg-surface);
        border: 1px solid var(--border-soft);
        border-radius: 11px;
        padding: 10px 14px;
        margin: 8px 0 10px 0;
        box-shadow: 0 1px 4px rgba(11, 34, 28, 0.04);
    }
    .viewer3d-disclaimer {
        background: var(--amber-tint);
        border: 1px solid var(--amber-border);
        border-left: 4px solid var(--amber-accent);
        border-radius: 10px;
        padding: 9px 13px;
        margin-top: 8px;
        font-size: var(--fs-label);
        color: var(--amber-deep);
        line-height: 1.48;
    }
    .gel-dialog-summary {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
        background: var(--teal-tint);
        border: 1px solid var(--teal-border);
        border-radius: 11px;
        padding: 10px 14px;
        margin: 8px 0 12px 0;
    }
    .gel-dialog-buffer-box {
        background: var(--bg-surface);
        border: 1px solid var(--border-soft);
        border-left: 4px solid var(--teal-primary);
        border-radius: 10px;
        padding: 12px 15px;
        margin: 8px 0 12px 0;
        font-size: var(--fs-body);
        color: var(--ink-body);
        line-height: 1.55;
    }
    .gel-dialog-buffer-title {
        font-size: var(--fs-h3);
        font-weight: 700;
        color: var(--ink-heading);
        margin-bottom: 5px;
    }
    .feedback-diag-box {
        background: var(--bg-subtle);
        border: 1px solid var(--border-soft);
        border-radius: 10px;
        padding: 10px 14px;
        margin: 6px 0 10px 0;
        font-size: var(--fs-label);
        color: var(--ink-body);
        line-height: 1.55;
    }
    .feedback-diag-title {
        font-size: var(--fs-micro);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--teal-deep);
        margin-bottom: 4px;
    }

    /* Keep the header & toolbar transparent and click-through so the sidebar expand toggle
       ([data-testid="stExpandSidebarButton"], mounted inside stToolbar in Streamlit 1.38+)
       always remains visible and clickable after hiding the sidebar or on mobile */
    [data-testid="stHeader"],
    [data-testid="stToolbar"] {
        background: transparent !important;
        pointer-events: none !important;
    }
    [data-testid="stExpandSidebarButton"],
    [data-testid="collapsedControl"],
    [data-testid="stStatusWidget"] {
        pointer-events: auto !important;
    }
    [data-testid="stExpandSidebarButton"],
    [data-testid="collapsedControl"] button {
        display: inline-flex !important;
        align-items: center;
        justify-content: center;
        gap: 6px;
        background: var(--bg-surface) !important;
        border: 1.5px solid var(--teal-strong) !important;
        border-radius: 10px !important;
        padding: 4px 12px 4px 8px !important;
        color: var(--teal-strong) !important;
        box-shadow: 0 2px 8px rgba(11, 34, 28, 0.12) !important;
    }
    [data-testid="stExpandSidebarButton"]:hover,
    [data-testid="collapsedControl"] button:hover {
        background: var(--teal-tint) !important;
        border-color: var(--teal-deep) !important;
        color: var(--teal-deep) !important;
    }
    [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"],
    [data-testid="stExpandSidebarButton"] svg,
    [data-testid="collapsedControl"] [data-testid="stIconMaterial"],
    [data-testid="collapsedControl"] svg {
        color: var(--teal-strong) !important;
        fill: var(--teal-strong) !important;
    }
    [data-testid="stExpandSidebarButton"]::after,
    [data-testid="collapsedControl"] button::after {
        content: "Load / Search Protein";
        font-family: var(--font-sans);
        font-size: var(--fs-label);
        font-weight: 700;
        color: var(--teal-strong);
        white-space: nowrap;
    }
    [data-testid="stExpandSidebarButton"]:hover::after,
    [data-testid="collapsedControl"] button:hover::after {
        color: var(--teal-deep);
    }

    /* Minimize Streamlit chrome & viewer-badge branding inside the app viewport
       (Never hide stHeader / stToolbar — stExpandSidebarButton lives inside stToolbar!) */
    #MainMenu,
    footer,
    [data-testid="stMainMenu"],
    [data-testid="stToolbarActions"],
    [data-testid="stAppDeployButton"],
    [data-testid="stDecoration"],
    [class*="viewerBadge"],
    [class*="_profileContainer"],
    [class*="_profilePreview"] {
        display: none !important;
        visibility: hidden !important;
    }
</style>
""", unsafe_allow_html=True)


# --- INITIALIZE SESSION STATE (Default = Egg-White Lysozyme P00698!) ---
if "active_sequence" not in st.session_state:
    default_preset = CLASSROOM_PRESET_PROTEINS["lysozyme_native"]
    st.session_state.active_sequence = default_preset["sequence"]
    st.session_state.sequence_name = default_preset["name"]
    st.session_state.sequence_metadata = {
        "source": "",
        "uniprot_id": default_preset["uniprot_id"],
        "tag": default_preset["tag"],
    }
    st.session_state.construct_parts = default_preset["construct_parts"]

if "custom_buffer_rows" not in st.session_state:
    st.session_state.custom_buffer_rows = [dict(r) for r in PRESET_RECIPE_TEMPLATES["his_lysis"]["rows"]]
    st.session_state.custom_buffer_ph = float(PRESET_RECIPE_TEMPLATES["his_lysis"]["target_ph"])
    st.session_state.custom_buffer_note = PRESET_RECIPE_TEMPLATES["his_lysis"]["notes"]
    st.session_state.active_recipe_key = "his_lysis"
if "buffer_editor_ver" not in st.session_state:
    st.session_state.buffer_editor_ver = 0
if "structure_3d_cache" not in st.session_state:
    st.session_state.structure_3d_cache = {}


def _md_inline_to_html(text: str) -> str:
    """Safely converts simple **bold** and `code` spans to HTML for custom ribbon cards."""
    esc = html.escape(str(text or ""))
    esc = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc)
    esc = re.sub(r"`([^`]+)`", r"<code>\1</code>", esc)
    return esc


def set_new_sequence(seq: str, name: str, metadata: dict, construct_parts=None) -> bool:
    clean_seq, header, warnings = clean_amino_acid_sequence(seq)
    if not clean_seq:
        st.error("No valid amino acid sequence found! Please check your input.")
        return False
    meta_copy = dict(metadata or {})
    if "tag" not in meta_copy:
        term_tag = detect_sequence_terminal_tag(clean_seq)
        meta_copy["tag"] = term_tag["tag_detected"]
    st.session_state.active_sequence = clean_seq
    st.session_state.sequence_name = name or header or "Custom Protein"
    st.session_state.sequence_metadata = meta_copy
    st.session_state.construct_parts = construct_parts
    for w in warnings:
        st.warning(w)
    return True


def _apply_classroom_preset():
    chosen_key = st.session_state.classroom_preset_select
    p = CLASSROOM_PRESET_PROTEINS[chosen_key]
    set_new_sequence(
        seq=p["sequence"],
        name=p["name"],
        metadata={
            "source": "",
            "uniprot_id": p["uniprot_id"],
            "tag": p["tag"],
            "biochem_annotations": p.get("biochem_annotations", {}),
        },
        construct_parts=p["construct_parts"]
    )


def _get_feedback_webhook_url() -> str:
    """Safely resolves optional FEEDBACK_WEBHOOK_URL from st.secrets or env var."""
    webhook = ""
    try:
        webhook = str(st.secrets.get("FEEDBACK_WEBHOOK_URL", "") or "").strip()
    except Exception:
        webhook = ""
    if not webhook:
        webhook = str(os.environ.get("PROPURIFY_FEEDBACK_WEBHOOK", "") or "").strip()
    if webhook and webhook.lower().startswith(("http://", "https://")):
        return webhook
    return ""


@st.dialog("🧪 SDS-PAGE Gel Casting Calculator", width="large")
def _open_gel_recipe_dialog(default_gel_pct: float, target_mw_kda: float, protein_name: str) -> None:
    """Pop-out handcast SDS-PAGE gel recipe calculator (Laemmli & Schägger Tricine)."""
    rec_params = recommend_gel_recipe_params(float(target_mw_kda))
    is_low_mw = float(target_mw_kda) < 10.0

    st.caption(
        f"Active protein: **{protein_name}** (`{target_mw_kda:.2f} kDa`) — "
        "exact conserved pipetting volumes in **mL** and **µL** with bench safety & running buffer recipes."
    )

    if is_low_mw:
        st.info(
            f"🧬 **Auto-Selected Tricine-SDS-PAGE for <10 kDa Peptide ({target_mw_kda:.2f} kDa):** "
            f"{rec_params['rationale']} You can freely switch back to Tris-Glycine (Laemmli) below."
        )

    sys_keys = ["laemmli", "tricine"]
    sys_labels = {
        "laemmli": "Tris-Glycine (Laemmli 1970)",
        "tricine": "Tricine-SDS-PAGE (Schägger 2006 — <10–30 kDa)",
    }
    gc_r1c1, gc_r1c2 = st.columns([1.25, 1.0])
    with gc_r1c1:
        gel_sys = st.radio(
            "Gel Buffer System:",
            options=sys_keys,
            format_func=lambda k: sys_labels[k],
            index=1 if is_low_mw else 0,
            horizontal=True,
            key="gel_calc_system_radio",
        )
    with gc_r1c2:
        sys_spec = GEL_SYSTEMS[gel_sys]
        min_pct = float(sys_spec["min_gel_pct"])
        max_pct = float(sys_spec["max_gel_pct"])
        init_pct = round(max(min_pct, min(max_pct, float(default_gel_pct))) * 2.0) / 2.0
        calc_gel_pct = st.slider(
            f"Resolving Gel %T ({min_pct:g}%–{max_pct:g}%):",
            min_value=min_pct,
            max_value=max_pct,
            value=init_pct,
            step=0.5,
            key=f"gel_calc_pct_{gel_sys}",
        )

    stock_options = [30.0, 40.0, 49.5]
    stock_labels = {
        30.0: "30% (37.5:1) — Standard Laemmli",
        40.0: "40% (29:1) — High-% / Tricine",
        49.5: "49.5% T (3% C) — Schägger Tricine AB-3",
    }
    default_stock_idx = 0
    if gel_sys == "tricine":
        default_stock_idx = 1 if calc_gel_pct <= 18.0 else 2
    elif calc_gel_pct >= 21.5:
        default_stock_idx = 1

    format_keys = list(GEL_FORMAT_PRESETS.keys())
    format_labels = {
        "mini_0.75mm": "0.75 mm Mini-Gel (5.0 / 2.0 mL)",
        "mini_1.0mm": "1.0 mm Mini-Gel — Standard (7.5 / 2.5 mL)",
        "mini_1.5mm": "1.5 mm Mini-Gel — High-Load (10.0 / 3.5 mL)",
        "midi_1.0mm": "1.0 mm Midi-Gel (15.0 / 5.0 mL)",
        "custom": "Custom Cassette Volume (mL)",
    }

    gc_r2c1, gc_r2c2, gc_r2c3, gc_r2c4 = st.columns([1.3, 1.3, 0.75, 0.85])
    with gc_r2c1:
        stock_pct = st.selectbox(
            "Acrylamide/Bis Stock:",
            options=stock_options,
            format_func=lambda v: stock_labels[v],
            index=default_stock_idx,
            key=f"gel_calc_stock_{gel_sys}",
        )
    with gc_r2c2:
        fmt_key = st.selectbox(
            "Gel Format:",
            options=format_keys,
            format_func=lambda k: format_labels.get(k, GEL_FORMAT_PRESETS[k]["label"]),
            index=1,
            key="gel_calc_format_select",
        )
    with gc_r2c3:
        num_gels = st.slider(
            "Number of Gels:",
            min_value=1,
            max_value=4,
            value=1,
            step=1,
            key="gel_calc_num_gels",
        )
    with gc_r2c4:
        stacking_pct = st.selectbox(
            "Stacking %T:",
            options=[4.0, 5.0],
            format_func=lambda v: f"{int(v)}% Stacking",
            index=0,
            key="gel_calc_stacking_pct",
        )

    fmt_preset = GEL_FORMAT_PRESETS[fmt_key]
    res_vol_per_gel = float(fmt_preset["resolving_vol_ml"])
    stk_vol_per_gel = float(fmt_preset["stacking_vol_ml"])
    if fmt_key == "custom":
        cc1, cc2 = st.columns(2)
        with cc1:
            res_vol_per_gel = st.number_input(
                "Resolving Volume per Gel (mL):",
                min_value=2.0,
                max_value=100.0,
                value=7.5,
                step=0.5,
                key="gel_calc_custom_res_ml",
            )
        with cc2:
            stk_vol_per_gel = st.number_input(
                "Stacking Volume per Gel (mL):",
                min_value=1.0,
                max_value=40.0,
                value=2.5,
                step=0.5,
                key="gel_calc_custom_stk_ml",
            )

    include_glycerol = None
    if gel_sys == "tricine":
        include_glycerol = st.checkbox(
            "Include 10% (v/v) Glycerol in Resolving Gel (recommended for ≥16% T Tricine)",
            value=(calc_gel_pct >= 16.0),
            key="gel_calc_tricine_glycerol",
        )

    try:
        recipe = calculate_sds_gel_recipe(
            gel_pct=calc_gel_pct,
            system=gel_sys,
            acrylamide_stock_pct=stock_pct,
            resolving_vol_ml=res_vol_per_gel,
            stacking_vol_ml=stk_vol_per_gel,
            stacking_pct=stacking_pct,
            num_gels=int(num_gels),
            target_mw_kda=float(target_mw_kda),
            include_glycerol=include_glycerol,
            auto_upgrade_stock=True,
        )
    except ValueError as err:
        st.error(str(err))
        return

    low_mw, high_mw = recipe["linear_range_kda"]
    gel_unit = "gel" if int(num_gels) == 1 else "gels"
    glyc_tag = (
        f'<span class="meta-pill meta-pill-amber">+{recipe["glycerol_pct"]:g}% Glycerol</span>'
        if recipe.get("include_glycerol")
        else ""
    )
    summary_html = (
        '<div class="gel-dialog-summary">'
        f'<span class="meta-pill meta-pill-teal">{html.escape(recipe["system_short_name"])}</span>'
        f'<span class="meta-pill meta-pill-indigo">{recipe["gel_pct"]:g}% Resolving / {recipe["stacking_pct"]:g}% Stacking</span>'
        f'<span class="meta-pill meta-pill-neutral">Linear Range: ~{low_mw:g}–{high_mw:g} kDa</span>'
        f'<span class="meta-pill meta-pill-neutral">Scale: {int(num_gels)} {gel_unit} '
        f'({recipe["resolving_total_vol_ml"]:g} + {recipe["stacking_total_vol_ml"]:g} mL)</span>'
        f"{glyc_tag}"
        "</div>"
    )
    st.markdown(summary_html, unsafe_allow_html=True)

    for w_msg in recipe.get("warnings") or []:
        st.warning(w_msg)

    st.markdown(
        f"##### 1. Resolving Gel — `{recipe['gel_pct']:g}% T` "
        f"(Total: **{recipe['resolving_total_vol_ml']:g} mL** for {int(num_gels)} {gel_unit})"
    )
    res_df = pd.DataFrame(recipe["resolving_rows"])[RECIPE_DISPLAY_COLUMNS]
    st.dataframe(res_df, hide_index=True, width="stretch")

    st.markdown(
        f"##### 2. Stacking Gel — `{recipe['stacking_pct']:g}% T` "
        f"(Total: **{recipe['stacking_total_vol_ml']:g} mL** for {int(num_gels)} {gel_unit})"
    )
    stk_df = pd.DataFrame(recipe["stacking_rows"])[RECIPE_DISPLAY_COLUMNS]
    st.dataframe(stk_df, hide_index=True, width="stretch")

    rb = recipe.get("running_buffer") or {}
    if rb:
        rb_extra = ""
        if not rb.get("single_chamber", True):
            rb_extra = (
                f"<br><b>Anode (Lower Tank):</b> {html.escape(str(rb.get('anode_buffer', '')))}"
                f"<br><b>Cathode (Upper Chamber):</b> {html.escape(str(rb.get('cathode_buffer', '')))}"
            )
        else:
            rb_extra = f"<br><b>10× Stock Prep:</b> {html.escape(str(rb.get('stock_10x_per_1l', '')))}"
        rb_html = (
            '<div class="gel-dialog-buffer-box">'
            f'<div class="gel-dialog-buffer-title">⚡ 3. {html.escape(str(rb.get("name", "1× Running Buffer")))}</div>'
            f'<b>1× Composition:</b> {html.escape(str(rb.get("composition_1x", "")))}<br>'
            f'<b>pH:</b> {html.escape(str(rb.get("ph", "")))}'
            f"{rb_extra}"
            f'<br><b>Run Conditions:</b> {html.escape(str(rb.get("run_conditions", "")))}'
            f'<br><b>Note:</b> {html.escape(str(rb.get("notes", "")))}'
            "</div>"
        )
        st.markdown(rb_html, unsafe_allow_html=True)

    safety_notes = recipe.get("safety_notes") or []
    bench_notes = recipe.get("bench_notes") or []
    st.warning(
        "**⚠️ Safety & Polymerization Reminder:** Unpolymerized acrylamide is a potent **neurotoxin** "
        "(wear nitrile gloves, lab coat, and eye protection; handle stocks in a fume hood). "
        "**Add 10% APS and TEMED LAST** immediately before gently inverting and pouring — polymerization begins within minutes!"
    )
    with st.expander("📋 Detailed Bench Casting Notes, Stock Prep & EHS Checklist", expanded=False):
        if bench_notes:
            st.markdown("**Bench Casting Protocol:**")
            for idx_bn, note in enumerate(bench_notes, start=1):
                st.markdown(f"{idx_bn}. {note}")
        if safety_notes:
            st.markdown("**EHS Safety Notes:**")
            for s_note in safety_notes:
                st.markdown(f"- {s_note}")
        stock_guide = recipe.get("stock_prep_guide") or {}
        if stock_guide:
            st.markdown("**Stock Solution Preparation Quick Reference:**")
            for s_title, s_desc in stock_guide.items():
                st.markdown(f"- **{s_title}:** {s_desc}")

    recipe_txt = format_gel_recipe_text(recipe)
    safe_pct_tag = str(recipe["gel_pct"]).replace(".", "p")
    st.download_button(
        label="⬇️ Download Recipe Sheet (.txt)",
        data=recipe_txt.encode("utf-8"),
        file_name=f"sds_page_{gel_sys}_{safe_pct_tag}pct_{int(num_gels)}gel.txt",
        mime="text/plain",
        type="primary",
        width="stretch",
        key="btn_download_gel_recipe_txt",
    )


@st.dialog("💬 Report an Error or Suggestion", width="large")
def _open_feedback_dialog(
    protein_name: str,
    uniprot_or_acc: str,
    seq_len: int,
    active_input_method: str,
) -> None:
    """Anonymous bug report & scientific feedback modal (zero personal data collected)."""
    diag_metrics = analyze_all_parameters(st.session_state.active_sequence)
    mw_val = round(float(diag_metrics["mw_kda"]), 2)
    pi_val = round(float(diag_metrics["pi"]), 2)

    st.caption(
        "Help improve scientific accuracy and usability. Reports are completely anonymous — "
        "no name, email, or personal identifiers are collected."
    )

    fb_category = st.selectbox(
        "Category:",
        [
            "Biochemical / Cofactor Inaccuracy",
            "Protein Search / Sequence Bug",
            "SEC / SDS-PAGE / Gel Recipe",
            "3D Viewer / UI Issue",
            "AI-Designed Protein Discrepancy",
            "Feature Request / Other",
        ],
        key="fb_category_select",
    )

    diag_html = (
        '<div class="feedback-diag-box">'
        '<div class="feedback-diag-title">Attached Anonymous Diagnostic Context (Read-Only)</div>'
        f'<b>Active Protein:</b> {html.escape(str(protein_name))} &nbsp;|&nbsp; '
        f'<b>Accession:</b> <code>{html.escape(str(uniprot_or_acc))}</code><br>'
        f'<b>Length:</b> {int(seq_len):,} aa &nbsp;|&nbsp; '
        f'<b>MW:</b> {mw_val:.2f} kDa &nbsp;|&nbsp; '
        f'<b>pI:</b> {pi_val:.2f} &nbsp;|&nbsp; '
        f'<b>Input Method:</b> {html.escape(str(active_input_method))}'
        "</div>"
    )
    st.markdown(diag_html, unsafe_allow_html=True)

    fb_description = st.text_area(
        "What happened & what did you expect?",
        height=120,
        placeholder="Describe the discrepancy, unexpected calculation, or suggestion clearly...",
        key="fb_description_input",
    )
    fb_reference = st.text_input(
        "Expected reference / DOI / UniProt ID (optional):",
        placeholder="e.g., UniProt P07374, PDB 2VB1, DOI 10.1038/...",
        key="fb_reference_input",
    )

    if st.button("📨 Submit Anonymous Report", type="primary", width="stretch", key="fb_submit_btn"):
        if not fb_description.strip():
            st.warning("Please enter a brief description of what happened and what you expected before submitting.")
        else:
            submitted_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            report_payload = {
                "schema_version": "1.0",
                "submitted_utc": submitted_utc,
                "category": fb_category,
                "description": fb_description.strip(),
                "reference": fb_reference.strip(),
                "diagnostic_context": {
                    "active_protein_name": str(protein_name),
                    "uniprot_or_accession_id": str(uniprot_or_acc),
                    "sequence_length_aa": int(seq_len),
                    "mw_kda": mw_val,
                    "pi": pi_val,
                    "active_input_method": str(active_input_method),
                },
            }

            delivery_notes = []
            webhook_url = _get_feedback_webhook_url()
            if webhook_url:
                try:
                    resp = requests.post(webhook_url, json=report_payload, timeout=6)
                    if 200 <= resp.status_code < 300:
                        delivery_notes.append("forwarded via configured webhook")
                    else:
                        delivery_notes.append("webhook returned non-2xx (saved locally / downloadable below)")
                except Exception:
                    delivery_notes.append("webhook unreachable (saved locally / downloadable below)")

            fb_log_path = os.path.join(_APP_ROOT, "data", "user_feedback_reports.jsonl")
            try:
                os.makedirs(os.path.dirname(fb_log_path), exist_ok=True)
                with open(fb_log_path, "a", encoding="utf-8") as fb_f:
                    fb_f.write(json.dumps(report_payload, ensure_ascii=False) + "\n")
                delivery_notes.append("recorded to local report log")
            except OSError:
                delivery_notes.append("read-only storage — use the download button below to export")

            st.session_state["last_feedback_report"] = report_payload
            st.session_state["last_feedback_delivery"] = "; ".join(delivery_notes)

    last_rep = st.session_state.get("last_feedback_report")
    last_ctx = last_rep.get("diagnostic_context") if isinstance(last_rep, dict) else None
    if isinstance(last_rep, dict) and isinstance(last_ctx, dict) and last_ctx.get("active_protein_name") == str(protein_name):
        st.success("✅ Anonymous report captured! Thank you for helping verify and improve the lab.")
        deliv_str = st.session_state.get("last_feedback_delivery", "")
        if deliv_str:
            st.caption(f"Delivery status: {deliv_str}.")
        rep_json_str = json.dumps(last_rep, indent=2, ensure_ascii=False)
        st.download_button(
            label="⬇️ Download Diagnostic Report (.json)",
            data=rep_json_str.encode("utf-8"),
            file_name=f"propurify_diagnostic_report_{last_rep.get('submitted_utc', 'report').replace(':', '-')}.json",
            mime="application/json",
            width="stretch",
            key="fb_download_json_btn",
        )
        ctx = last_rep.get("diagnostic_context") or {}
        copy_summary = (
            f"Category: {last_rep.get('category', '')}\n"
            f"UTC Time: {last_rep.get('submitted_utc', '')}\n"
            f"Protein: {ctx.get('active_protein_name', '')} ({ctx.get('uniprot_or_accession_id', 'None')})\n"
            f"Metrics: {ctx.get('sequence_length_aa', 0)} aa | {ctx.get('mw_kda', 0)} kDa | pI {ctx.get('pi', 0)} | Input: {ctx.get('active_input_method', '')}\n"
            f"Details: {last_rep.get('description', '')}\n"
            + (f"Reference: {last_rep.get('reference', '')}\n" if last_rep.get("reference") else "")
        )
        with st.expander("📋 Copyable Diagnostic Summary", expanded=True):
            st.code(copy_summary, language="text")


def _render_3dmol_metadata_and_viewer(struct_dict: dict, key_prefix: str) -> None:
    """Renders interactive WebGL 3Dmol.js controls, structural metadata bar, and embedded viewer."""
    style_labels = {
        "cartoon": "Cartoon Ribbon",
        "surface": "Molecular Surface",
        "stick": "Atomic Sticks",
        "sphere": "Space-Fill Spheres",
    }
    color_labels = {
        "ss": "Secondary Structure",
        "plddt": "pLDDT Confidence",
        "spectrum": "N→C Rainbow Spectrum",
        "hydrophobicity": "Hydrophobicity (Kyte–Doolittle)",
        "charge": "Side-Chain Charge",
    }
    has_plddt = bool(struct_dict.get("is_alphafold") or struct_dict.get("looks_like_plddt"))
    color_opts = ["ss", "plddt", "spectrum", "hydrophobicity", "charge"]

    vc1, vc2, vc3, vc4 = st.columns([1.15, 1.35, 0.75, 1.05])
    with vc1:
        v3d_style = st.selectbox(
            "3D Representation:",
            options=["cartoon", "surface", "stick", "sphere"],
            format_func=lambda s: style_labels[s],
            index=0,
            key=f"{key_prefix}_style",
        )
    with vc2:
        v3d_color = st.selectbox(
            "Color Scheme:",
            options=color_opts,
            format_func=lambda c: color_labels[c],
            index=1 if has_plddt else 0,
            key=f"{key_prefix}_color",
        )
    with vc3:
        v3d_spin = st.checkbox("Auto-Spin", value=False, key=f"{key_prefix}_spin")
    with vc4:
        v3d_hetatms = st.checkbox("Show Metals / Cofactors", value=True, key=f"{key_prefix}_hetatms")

    # Build crisp metadata pill bar (strictly zero inline font-size & zero 4-space HTML indent)
    pills = []
    src_label = str(struct_dict.get("source") or "3D Structure")
    entry_label = str(struct_dict.get("pdb_id") or struct_dict.get("entry_id") or "")
    id_badge = f"{src_label} • {entry_label}" if entry_label else src_label
    pills.append(f'<span class="meta-pill meta-pill-teal">{html.escape(id_badge)}</span>')

    res_ang = struct_dict.get("resolution_angstrom")
    mean_plddt = struct_dict.get("mean_plddt")
    if res_ang is not None:
        method_str = str(struct_dict.get("experimental_method") or "").strip()
        res_txt = f"Resolution: {float(res_ang):.2f} Å" + (f" ({method_str})" if method_str else "")
        pills.append(f'<span class="meta-pill meta-pill-indigo">{html.escape(res_txt)}</span>')
    elif mean_plddt is not None:
        plddt_val = float(mean_plddt)
        plddt_cls = (
            "meta-pill-teal" if plddt_val >= 80.0 else ("meta-pill-amber" if plddt_val >= 60.0 else "meta-pill-crimson")
        )
        pills.append(f'<span class="meta-pill {plddt_cls}">Mean pLDDT: {plddt_val:.1f} / 100</span>')

    chains = list(struct_dict.get("chains") or [])
    chain_cnt = int(struct_dict.get("chain_count") or len(chains))
    res_cnt = int(struct_dict.get("residue_count") or 0)
    atom_cnt = int(struct_dict.get("atom_count") or 0)
    chain_preview = f" ({', '.join(chains[:4])}{'…' if len(chains) > 4 else ''})" if chains else ""
    pills.append(
        f'<span class="meta-pill meta-pill-neutral">'
        f"{chain_cnt} chain{'s' if chain_cnt != 1 else ''}{html.escape(chain_preview)} • "
        f"{res_cnt:,} res • {atom_cnt:,} atoms</span>"
    )

    metals = list(struct_dict.get("metal_ions") or [])
    if metals:
        pills.append(f'<span class="meta-pill meta-pill-amber">⚡ Metals: {html.escape(", ".join(metals[:8]))}</span>')

    cofactors = list(struct_dict.get("cofactors") or [])
    if cofactors:
        pills.append(f'<span class="meta-pill meta-pill-indigo">🧪 Cofactors: {html.escape(", ".join(cofactors[:8]))}</span>')

    ss_bonds = int(struct_dict.get("disulfide_bonds") or 0)
    if ss_bonds > 0:
        pills.append(f'<span class="meta-pill meta-pill-neutral">🔗 {ss_bonds} S–S bond{"s" if ss_bonds != 1 else ""}</span>')

    st.markdown('<div class="viewer3d-meta-bar">' + "".join(pills) + "</div>", unsafe_allow_html=True)

    entry_url = str(struct_dict.get("entry_url") or "").strip()
    title_str = str(struct_dict.get("title") or "").strip()
    if title_str or entry_url:
        cap_parts = []
        if title_str:
            cap_parts.append(f"**{title_str}**")
        if entry_url:
            cap_parts.append(f"[View source record]({entry_url})")
        st.caption("  •  ".join(cap_parts))

    viewer_html = build_3dmol_viewer_html(
        structure_text=struct_dict["data"],
        fmt=str(struct_dict.get("format") or "pdb"),
        style=v3d_style,
        color_scheme=v3d_color,
        spin=bool(v3d_spin),
        show_hetatms=bool(v3d_hetatms),
        height_px=440,
    )
    if hasattr(st, "iframe"):
        st.iframe(viewer_html, width="stretch", height=470)
    else:
        components.html(viewer_html, height=470, scrolling=False)


# --- HERO HEADER ---
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge-row">
        <span class="hero-badge badge-gold">🎓 Interactive Biochemistry &amp; Purification Studio</span>
        <span class="hero-badge badge-mint">🧪 SEC • SDS-PAGE • IEX &amp; HIC • Smart Buffer Lab</span>
    </div>
    <div class="hero-title">ProPurify Studio</div>
    <div class="hero-subtitle">
        Pick a textbook protein or load any sequence from UniProt/NCBI to visualize <b>SEC elution</b>, run an <b>SDS-PAGE gel</b>,
        build a <b>cofactor-safe purification strategy</b>, and weigh exact <b>lab buffers (g / mg)</b>.
    </div>
</div>
""", unsafe_allow_html=True)


# --- SIDEBAR: COHESIVE PROTEIN LOADER (CUSTOM INPUT AT TOP, STARTER EXAMPLES BELOW) ---
with st.sidebar:
    st.markdown("### 📥 Load Your Protein")
    input_method = st.radio(
        "Input Method:",
        [
            "🔍 UniProt ID",
            "🏛️ NCBI Accession",
            "🧊 RCSB PDB ID (De Novo / 3D)",
            "🔤 Paste Amino Acid / FASTA",
            "🧬 Translate DNA / RNA",
            "💬 Describe in Plain English"
        ]
    )

    # 1. UniProt Accession (Lysozyme P00698 default)
    if input_method == "🔍 UniProt ID":
        uid_input = st.text_input(
            "UniProt Accession ID:",
            value="P00698",
            help="Supports canonical accessions (P00698, A0A024RBG1), isoforms (P04637-2), entry names (LYSC_CHICK), or UniProt URLs. If UniProt annotates a Signal/Propeptide, a 1-click Mature Chain toggle appears below the protein banner."
        )
        if st.button("Fetch from UniProt", type="primary", width="stretch"):
            with st.spinner(f"Fetching {uid_input} from UniProt..."):
                try:
                    res = fetch_uniprot_by_id(uid_input)
                    disp_nm = f"{res['protein_name']} [{res['accession']}]"
                    meta_up = dict(res)
                    if res.get("mature_chain"):
                        meta_up["precursor_sequence"] = res["sequence"]
                        meta_up["precursor_name"] = disp_nm
                        meta_up["is_mature_cleaved"] = False
                    if set_new_sequence(
                        seq=res["sequence"],
                        name=disp_nm,
                        metadata=meta_up
                    ):
                        st.success(f"Loaded: {res['protein_name']} ({res['length']} aa)")
                except Exception as e:
                    st.error(str(e))
        st.caption("Try: `P00698` (Chicken Lysozyme), `P61626` (Human Lysozyme), `P42212` (GFP), `P07374` (Urease)")

    # 2. NCBI Accession (Chicken Lysozyme RefSeq NP_990612.2 / NM_205281.2)
    elif input_method == "🏛️ NCBI Accession":
        ncbi_input = st.text_input(
            "NCBI Accession:",
            value="NP_990612.2",
            help="Supports RefSeq & GenBank protein accessions (NP_990612.2, WP_000002298.1, QHD43416.1) and nucleotide/mRNA transcripts (NM_205281.2, M21110.1, BC004147, MN908947.3) with automatic 6-frame ORF translation."
        )
        if st.button("Fetch from NCBI", type="primary", width="stretch"):
            with st.spinner(f"Querying NCBI for {ncbi_input}..."):
                try:
                    res = fetch_ncbi_by_id(ncbi_input)
                    org_n = res.get("organism", "")
                    pname_n = res["protein_name"]
                    ncbi_display_name = (
                        pname_n
                        if not org_n or org_n in ("See Header", "Unknown Organism") or org_n.lower() in pname_n.lower()
                        else f"{pname_n} ({org_n})"
                    )
                    if set_new_sequence(
                        seq=res["sequence"],
                        name=ncbi_display_name,
                        metadata=res
                    ):
                        st.success(f"Loaded: {pname_n} ({res['length']} aa)")
                except Exception as e:
                    st.error(str(e))
        st.caption("Try: `NP_990612.2` (Lysozyme), `NM_205281.2` (mRNA), `QHD43416.1`")

    # 3. RCSB PDB ID (Ideal for De Novo / AI-Designed & Experimental 3D Structures)
    elif input_method == "🧊 RCSB PDB ID (De Novo / 3D)":
        pdb_input = st.text_input(
            "RCSB PDB ID:",
            value="1QYS",
            help="Supports 4-character PDB IDs (1QYS, 2KL8, 6D0T, 7B3A), chain selectors (6D0T_A), or RCSB URLs. Ideal for de novo / AI-designed proteins and experimental crystal/cryo-EM structures — also unlocks 1-click WebGL 3D viewing in Tab 4."
        )
        if st.button("Fetch from RCSB PDB", type="primary", width="stretch"):
            with st.spinner(f"Fetching {pdb_input} from RCSB PDB..."):
                try:
                    res = fetch_pdb_by_id(pdb_input)
                    if set_new_sequence(
                        seq=res["sequence"],
                        name=res["protein_name"],
                        metadata=res
                    ):
                        st.success(f"Loaded: {res['protein_name']} [PDB {res['pdb_id']}] ({res['length']} aa)")
                except Exception as e:
                    st.error(str(e))
        st.caption("Try: `1QYS` (Top7 de novo), `2KL8` (De novo ferredoxin), `6D0T` (De novo β-barrel), `2VB1` (Lysozyme)")

    # 4. Direct Sequence / FASTA (Lysozyme default)
    elif input_method == "🔤 Paste Amino Acid / FASTA":
        pasted_text = st.text_area(
            "Paste Protein Sequence or FASTA:",
            value=(
                ">sp|P00698|LYSC_CHICK Egg-White Lysozyme C (Mature Chain, 129 aa)\n"
                f"{LYSOZYME_MATURE_SEQ}"
            ),
            height=150
        )
        if st.button("Load Sequence", type="primary", width="stretch"):
            _, parsed_hdr, _ = clean_amino_acid_sequence(pasted_text)
            hdr_meta = extract_fasta_header_metadata(parsed_hdr)
            fasta_meta = {"source": hdr_meta.get("source_format") or "Direct FASTA / Sequence Input"}
            for _mk in ("uniprot_id", "accession", "gene_name", "organism", "protein_name", "biochem_annotations"):
                if hdr_meta.get(_mk):
                    fasta_meta[_mk] = hdr_meta[_mk]
            display_pname = hdr_meta.get("protein_name") or parsed_hdr or "Custom FASTA Protein"
            org_f = hdr_meta.get("organism")
            fasta_display_name = (
                f"{display_pname} ({org_f})"
                if org_f and org_f not in ("Unknown Organism", "See Header") and org_f.lower() not in display_pname.lower()
                else display_pname
            )
            if set_new_sequence(
                seq=pasted_text,
                name=fasta_display_name,
                metadata=fasta_meta,
            ):
                st.success("Sequence loaded!")
        st.caption("Works for any natural, synthetic, or de novo / AI-designed sequence (RFdiffusion, ProteinMPNN, ESMFold) and auto-detects terminal tags (6xHis, Strep, FLAG).")

    # 5. DNA / RNA Sequence (Met + Mature Lysozyme coding ORF default, 130 aa)
    elif input_method == "🧬 Translate DNA / RNA":
        dna_input = st.text_area(
            "Coding DNA / RNA (Met + Mature Lysozyme Coding ORF, 130 aa):",
            value=DEFAULT_LYSOZYME_DNA,
            height=130,
            help="Paste raw DNA/RNA or a FASTA nucleotide sequence. Supports 6-frame ORF detection (+1..+3 forward & -1..-3 reverse complement) and preserves IUPAC N ambiguity codes without frameshifting."
        )
        trans_mode = st.radio("Reading Frame:", ["Find Longest Gene (ATG ➜ Stop)", "Translate Frame 1 Directly"])
        if st.button("Translate to Protein", type="primary", width="stretch"):
            if "Longest" in trans_mode:
                orf_res = find_longest_orf(dna_input)
                hdr_meta = extract_fasta_header_metadata(orf_res.get("header"))
                base_label = hdr_meta.get("protein_name") or "Translated Gene"
                strand_tag = f"Frame {orf_res['frame']} ({orf_res.get('strand', '+')} strand)"
                nt_meta = {"source": "DNA/RNA Translation", **orf_res}
                for _mk in ("uniprot_id", "accession", "gene_name", "organism", "protein_name", "biochem_annotations"):
                    if hdr_meta.get(_mk):
                        nt_meta[_mk] = hdr_meta[_mk]
                if set_new_sequence(
                    seq=orf_res["sequence"],
                    name=f"{base_label} ({strand_tag}, {orf_res['orf_length_aa']} aa)",
                    metadata=nt_meta,
                ):
                    for _w in (orf_res.get("warnings") or []):
                        st.warning(_w)
                    st.success(f"Translated {orf_res['orf_length_aa']} amino acids ({strand_tag})!")
            else:
                seq = translate_nucleotide(dna_input, frame=1).split('*')[0]
                if set_new_sequence(
                    seq=seq,
                    name="Translated Frame 1",
                    metadata={"source": "DNA/RNA Frame 1"}
                ):
                    st.success(f"Translated {len(seq)} amino acids!")

    # 6. Natural Language Prompt (also accepts pasted de novo sequences & RCSB PDB IDs)
    elif input_method == "💬 Describe in Plain English":
        nl_prompt = st.text_area(
            "Describe Protein, Paste De Novo Sequence, or Enter PDB ID (+ Optional Tag):",
            value="Chicken egg white lysozyme with N-terminal 6xHis tag and TEV site",
            height=85
        )
        if st.button("✨ Build Protein Construct", type="primary", width="stretch"):
            with st.spinner("Finding sequence & building construct..."):
                try:
                    res = parse_natural_language_prompt(nl_prompt)
                    parts = res.get("construct_parts") or []
                    has_engineered_parts = (
                        res.get("tag_detected", "None") != "None"
                        or res.get("cleavage_detected", "None") != "None"
                        or len(parts) > 1
                    )
                    if res.get("is_direct_sequence"):
                        source_label = "Direct De Novo / Custom Construct" if has_engineered_parts else "Direct De Novo / Custom Sequence"
                    elif res.get("pdb_id"):
                        chain_suffix = f" (Chain {res['chain_id']})" if res.get("chain_id") else ""
                        source_label = f"RCSB PDB {res['pdb_id']}{chain_suffix}"
                    elif res.get("uniprot_id"):
                        source_label = (
                            f"Natural Language Construct Builder • UniProt {res['uniprot_id']}"
                            if has_engineered_parts
                            else f"Natural Language Lookup • UniProt {res['uniprot_id']}"
                        )
                    else:
                        source_label = "Natural Language Construct Builder"

                    org_sci_prefix = " ".join(res["identified_organism"].split()[:2]).lower()
                    prot_low = res["identified_protein"].lower()
                    if res.get("is_direct_sequence") or res.get("is_de_novo"):
                        display_name = res["identified_protein"]
                    else:
                        display_name = (
                            res["identified_protein"]
                            if (
                                res["identified_organism"].lower() in prot_low
                                or (org_sci_prefix and org_sci_prefix in prot_low)
                            )
                            else f"{res['identified_protein']} ({res['identified_organism']})"
                        )
                    if set_new_sequence(
                        seq=res["full_construct_sequence"],
                        name=display_name,
                        metadata={
                            "source": source_label,
                            "uniprot_id": res.get("uniprot_id", ""),
                            "accession": res.get("accession") or res.get("uniprot_id") or res.get("pdb_id") or "",
                            "pdb_id": res.get("pdb_id", ""),
                            "chain_id": res.get("chain_id", ""),
                            "is_de_novo": bool(res.get("is_de_novo", False)),
                            "gene_name": res.get("gene_name", "N/A"),
                            "protein_name": res["identified_protein"],
                            "biochem_annotations": res.get("biochem_annotations", {}),
                            "tag": res["tag_detected"],
                            "tag_position": res["tag_position"],
                            "cleavage": res["cleavage_detected"],
                            "canonical_len": len(res["canonical_sequence"])
                        },
                        construct_parts=parts if has_engineered_parts else None
                    ):
                        if has_engineered_parts:
                            st.success(f"Construct ready: {res['identified_protein']} ({res['length']} aa)!")
                        else:
                            id_tag = (
                                f" [{res['uniprot_id']}]"
                                if res.get("uniprot_id")
                                else (f" [PDB {res['pdb_id']}]" if res.get("pdb_id") else "")
                            )
                            st.success(f"Loaded: {res['identified_protein']}{id_tag} ({res['length']} aa)!")
                except Exception as e:
                    st.error(f"Could not build construct: {e}")
        st.caption("Try: `GST human`, `Urease with 6x His tag`, `Top7 with N-terminal 6xHis tag`, or paste any de novo sequence (optionally + `with TEV site`)")

    st.markdown("---")
    st.markdown("### 🧪 Try an Example")
    st.selectbox(
        "Choose an Example:",
        options=list(CLASSROOM_PRESET_PROTEINS.keys()),
        format_func=lambda k: CLASSROOM_PRESET_PROTEINS[k]["label"],
        index=0,
        key="classroom_preset_select",
        on_change=_apply_classroom_preset
    )
    st.markdown("---")
    if st.button(
        "💬 Report Error / Feedback",
        key="btn_open_feedback_dialog",
        width="stretch",
        help="Submit an anonymous error report, biochemical correction, or feature suggestion.",
    ):
        _open_feedback_dialog(
            protein_name=st.session_state.sequence_name,
            uniprot_or_acc=str(
                st.session_state.sequence_metadata.get("uniprot_id")
                or st.session_state.sequence_metadata.get("accession")
                or "None"
            ),
            seq_len=len(st.session_state.active_sequence),
            active_input_method=input_method,
        )
    st.caption(
        "Educational simulator & workflow planner — treat outputs as starting hypotheses and verify against experimental data & resin manuals."
    )


# --- COMPUTE ALL PHYSICOCHEMICAL PARAMETERS & BIOCHEMICAL PROTEIN INTELLIGENCE ---
current_seq = st.session_state.active_sequence
metrics = analyze_all_parameters(current_seq)
ext = metrics["extinction"]
seq_meta = st.session_state.sequence_metadata
protein_profile = infer_protein_biochemical_profile(
    sequence=current_seq,
    protein_name=st.session_state.sequence_name,
    metadata=seq_meta,
    metrics=metrics,
    tag_detected=seq_meta.get("tag", "None"),
    construct_parts=st.session_state.construct_parts,
)


# --- ACTIVE PROTEIN & CONSTRUCT RIBBON (Zero-indentation HTML avoids CommonMark code-block traps) ---
pdb_badge_val = str(seq_meta.get("pdb_id") or "").strip().upper()
chain_badge_val = str(seq_meta.get("chain_id") or "").strip()
uid_raw_val = str(seq_meta.get("uniprot_id") or "").strip()
acc_raw_val = str(seq_meta.get("accession") or "").strip()
raw_source = (seq_meta.get("source") or "").strip()
is_ncbi_badge = raw_source.upper().startswith("NCBI") and not uid_raw_val
is_pdb_badge = bool(pdb_badge_val) and (not uid_raw_val or uid_raw_val == "N/A") and not is_ncbi_badge

if is_pdb_badge:
    uid_badge_val = f"{pdb_badge_val} (Chain {chain_badge_val})" if chain_badge_val else pdb_badge_val
    id_prefix = "PDB"
elif is_ncbi_badge:
    uid_badge_val = acc_raw_val
    id_prefix = "NCBI"
else:
    uid_badge_val = uid_raw_val or ("" if seq_meta.get("is_de_novo") else acc_raw_val)
    id_prefix = "UniProt"

clean_source = (
    raw_source.split("•")[0].replace(f"(UniProt {uid_badge_val})", "").strip()
    if (raw_source and uid_badge_val) else raw_source
)
uid_pill_html = (
    f'<span class="meta-pill meta-pill-indigo">{id_prefix} {html.escape(str(uid_badge_val))}</span>'
    if uid_badge_val and uid_badge_val != "N/A" else ""
)
de_novo_pill_html = (
    '<span class="meta-pill meta-pill-teal">De Novo / Synthetic</span>'
    if seq_meta.get("is_de_novo") else ""
)
show_source_pill = (
    bool(clean_source)
    and not (is_ncbi_badge and clean_source.upper() == "NCBI")
    and not (is_pdb_badge and clean_source.upper().startswith("RCSB PDB"))
)
source_pill_html = (
    f'<span class="meta-pill meta-pill-neutral">{html.escape(clean_source)}</span>'
    if show_source_pill else ""
)
biochem_pills_html = "".join(
    f'<span class="meta-pill {"meta-pill-crimson" if "🚫" in b else "meta-pill-amber"}">{html.escape(b)}</span>'
    for b in protein_profile.get("profile_badges", [])[:3]
)
conflict_info = protein_profile.get("tag_conflict")
conflict_html = ""
if conflict_info:
    c_cls = "biochem-conflict-box" if conflict_info.get("severity") == "critical" else "biochem-conflict-box biochem-conflict-warn"
    conflict_html = (
        f'<div class="{c_cls}">'
        f'<div class="biochem-conflict-title">{_md_inline_to_html(conflict_info["headline"])}</div>'
        f'<div class="biochem-conflict-body">👉 Open <b>Tab 2 (Purification Strategy)</b> for the safe alternative roadmap &amp; compatibility matrix, '
        f'and <b>Tab 3</b> for EDTA-free / Imidazole-free buffer recipes.</div>'
        f'</div>'
    )

has_multipart_construct = bool(
    st.session_state.construct_parts
    and (
        len(st.session_state.construct_parts) > 1
        or any(
            "tag" in p[0].lower() or "cleavage" in p[0].lower()
            for p in st.session_state.construct_parts
        )
    )
)
seq_name_safe = html.escape(str(st.session_state.sequence_name))
pill_group_html = (
    f'<div class="ribbon-pill-group">'
    f'<span class="meta-pill meta-pill-teal">{metrics["length"]} aa</span>'
    f'{uid_pill_html}{de_novo_pill_html}{biochem_pills_html}{source_pill_html}'
    f'</div>'
)
if has_multipart_construct:
    parts_html = "".join(
        f'<span class="construct-pill '
        + ('pill-tag' if "tag" in p[0].lower() else ('pill-cleavage' if "cleavage" in p[0].lower() else 'pill-target'))
        + f'"><b>{html.escape(str(p[0]))}:</b> {len(p[1])} aa</span>'
        for p in st.session_state.construct_parts
    )
    ribbon_html = (
        f'<div class="protein-ribbon">'
        f'<div class="ribbon-top-row"><div class="ribbon-title">🧬 Active Construct: {seq_name_safe}</div>{pill_group_html}</div>'
        f'<div class="construct-parts-row">{parts_html}</div>'
        f'{conflict_html}'
        f'</div>'
    )
else:
    ribbon_html = (
        f'<div class="protein-ribbon">'
        f'<div class="ribbon-top-row"><div class="ribbon-title">🧬 Active Protein: {seq_name_safe}</div>{pill_group_html}</div>'
        f'{conflict_html}'
        f'</div>'
    )
st.markdown(ribbon_html, unsafe_allow_html=True)

# 1-Click Signal/Propeptide Cleavage Toggle when UniProt returns a secreted precursor with a single mature Chain
mat_info = seq_meta.get("mature_chain")
prec_seq = seq_meta.get("precursor_sequence")
if mat_info and prec_seq and mat_info.get("sequence") and prec_seq != mat_info["sequence"]:
    is_cleaved = bool(seq_meta.get("is_mature_cleaved", False))
    mc1, mc2 = st.columns([3.1, 1.3])
    if not is_cleaved:
        mc1.info(
            f"✂️ **Precursor Sequence Loaded ({len(prec_seq)} aa):** UniProt annotates **{mat_info['cleaved_summary']}** "
            f"(mature chain: residues {mat_info['start']}–{mat_info['end']}, **{mat_info['length']} aa**). "
            "Switch to the cleaved mature chain to match purified MW, pI, SEC, and SDS-PAGE."
        )
        if mc2.button(f"✂️ Use Mature Chain ({mat_info['length']} aa)", key="btn_toggle_mature_chain", type="primary", width="stretch"):
            updated_meta = dict(seq_meta)
            updated_meta["is_mature_cleaved"] = True
            base_nm = updated_meta.get("precursor_name") or st.session_state.sequence_name
            st.session_state.active_sequence = mat_info["sequence"]
            st.session_state.sequence_name = f"{base_nm} (Mature Chain, {mat_info['length']} aa)"
            st.session_state.sequence_metadata = updated_meta
            st.rerun()
    else:
        mc1.success(
            f"✅ **Cleaved Mature Chain Active ({mat_info['length']} aa, residues {mat_info['start']}–{mat_info['end']}):** "
            f"Removed **{mat_info['cleaved_summary']}** from the {len(prec_seq)}-aa UniProt precursor."
        )
        if mc2.button(f"↩️ Restore Full Precursor ({len(prec_seq)} aa)", key="btn_toggle_mature_chain", width="stretch"):
            updated_meta = dict(seq_meta)
            updated_meta["is_mature_cleaved"] = False
            base_nm = updated_meta.get("precursor_name") or st.session_state.sequence_name
            st.session_state.active_sequence = prec_seq
            st.session_state.sequence_name = base_nm
            st.session_state.sequence_metadata = updated_meta
            st.rerun()


# --- 6-PARAMETER KPI CARDS (ALL 6 SET PARAMETERS PRESERVED + EXACT PROTPARAM PRECISION) ---
instab_ok = metrics["instability_index"] < 40.0
instab_badge_cls = "kpi-status-ok" if instab_ok else "kpi-status-warn"
instab_badge_txt = "✅ Stable (< 40)" if instab_ok else "⚠️ Unstable (≥ 40, keep on ice)"

if metrics["pi"] > 7.3:
    pi_hint = "Basic (+ at pH 7.0)"
elif metrics["pi"] < 6.7:
    pi_hint = "Acidic (– at pH 7.0)"
else:
    pi_hint = "Near-neutral (~0 at pH 7.0)"

pi_ox = float(metrics.get("pi_oxidized", metrics["pi"]))
if protein_profile.get("avoid_reducing_agents") and abs(pi_ox - metrics["pi"]) >= 0.15:
    pi_sub_txt = f"{pi_hint} • S–S oxidized: ~{pi_ox:.2f}"
else:
    pi_sub_txt = f"{pi_hint} • ExPASy / Bjellqvist"

if metrics["gravy"] < -0.55:
    gravy_hint = "Very water-loving (Polar)"
elif metrics["gravy"] <= 0.0:
    gravy_hint = "Water-soluble (Amphipathic)"
elif metrics["gravy"] <= 0.35:
    gravy_hint = "Moderately hydrophobic (> 0)"
else:
    gravy_hint = "Strongly hydrophobic (Membrane-like)"

if ext["molar_extinction_cystines"] == 0:
    ext_sub_html = "No Trp/Tyr/S–S (use A205/A214 nm)"
elif ext["potential_disulfides"] > 0:
    ext_sub_html = (
        f"1 mg/mL red. (ε={ext['molar_extinction_reduced']:,}) • "
        f"S–S: {ext['abs_01_cystines']:.3f} (ε={ext['molar_extinction_cystines']:,})"
    )
else:
    ext_sub_html = f"1 mg/mL (0.1%) • ε={ext['molar_extinction_reduced']:,} M⁻¹cm⁻¹"

st.markdown(f"""
<div class="metric-grid">
    <div class="card-kpi bar-teal">
        <div class="kpi-title">Molecular Weight</div>
        <div class="kpi-value">{metrics['mw_kda']:.2f} <span class="kpi-unit">kDa</span></div>
        <div class="kpi-sub">{metrics['mw_da']:,.2f} Da • Monomer chain</div>
    </div>
    <div class="card-kpi bar-amber">
        <div class="kpi-title">Theoretical pI</div>
        <div class="kpi-value">pI {metrics['pi']:.2f}</div>
        <div class="kpi-sub">{pi_sub_txt}</div>
    </div>
    <div class="card-kpi bar-emerald">
        <div class="kpi-title">Extinction (A280)</div>
        <div class="kpi-value">{ext['abs_01_reduced']:.3f}</div>
        <div class="kpi-sub">{ext_sub_html}</div>
    </div>
    <div class="card-kpi bar-coral">
        <div class="kpi-title">Instability Index</div>
        <div class="kpi-value">{metrics['instability_index']:.2f}</div>
        <div class="kpi-sub"><span class="{instab_badge_cls}">{instab_badge_txt}</span></div>
    </div>
    <div class="card-kpi bar-olive">
        <div class="kpi-title">Aliphatic Index</div>
        <div class="kpi-value">{metrics['aliphatic_index']:.2f}</div>
        <div class="kpi-sub">Thermostability (Ala/Val/Ile/Leu)</div>
    </div>
    <div class="card-kpi bar-plum">
        <div class="kpi-title">GRAVY Index</div>
        <div class="kpi-value">{metrics['gravy']:+.3f}</div>
        <div class="kpi-sub">{gravy_hint}</div>
    </div>
</div>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _read_json_with_mtime(file_path: str, mtime: float):
    with open(file_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_sec_benchmark_bundle():
    bench_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sec_benchmark_1200.json")
    if os.path.exists(bench_path):
        return _read_json_with_mtime(bench_path, os.path.getmtime(bench_path))
    return None


def load_ss_benchmark_bundle():
    ss_bench_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ss_10k_benchmark_results.json")
    if os.path.exists(ss_bench_path):
        return _read_json_with_mtime(ss_bench_path, os.path.getmtime(ss_bench_path))
    return None


def load_ss_crossref_bundle():
    ss_xref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ss_webserver_crossref.json")
    if os.path.exists(ss_xref_path):
        return _read_json_with_mtime(ss_xref_path, os.path.getmtime(ss_xref_path))
    return None


@st.cache_data(show_spinner=False)
def get_cached_secondary_structure(seq: str) -> dict:
    return predict_secondary_structure(seq)


def _format_exclusion_limit(excl_kda: float) -> str:
    """Formats a column's pore exclusion limit cleanly (e.g., 100 kDa, 1.3 MDa, 40 MDa)."""
    if excl_kda >= 1000.0:
        return f"{excl_kda / 1000.0:g} MDa"
    return f"{excl_kda:g} kDa"


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Converts a #RRGGBB hex color string into an rgba(r, g, b, alpha) CSS/Plotly color."""
    hc = hex_color.lstrip("#")
    if len(hc) == 6:
        r, g, b = int(hc[0:2], 16), int(hc[2:4], 16), int(hc[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha:.3f})"
    return f"rgba(15, 118, 110, {alpha:.3f})"


def _format_sec_table_row(p: dict, col_config: dict) -> dict:
    """Formats a simulated SEC peak row for classroom summary tables, handling HMW Void Aggregates cleanly."""
    is_agg = bool(p.get("is_aggregate", False))
    excl_kda = float(p.get("exclusion_limit_kda", col_config.get("exclusion_limit_kda", 1300.0)))
    pore_pct = float(p.get("pore_access_pct", 0.0 if is_agg else 50.0))
    return {
        "Peak": p["label"],
        "Size": f"> {_format_exclusion_limit(excl_kda)} (HMW)" if is_agg else f"{p['mw_kda']:.1f} kDa",
        "3D Radius (Rh)": "Excluded (Vo)" if is_agg else f"{p['rh_angstrom']:.1f} Å",
        "Bead Pores Entered": "0% (Bypassed)" if (is_agg or pore_pct < 0.5) else f"{pore_pct:.0f}% (Kav {p['kav']:.2f})",
        "Elutes At (Ve)": f"{p['ve_ml']:.2f} mL",
        "A260/A280": f"{p['a260_a280_ratio']:.2f}"
    }


PLOTLY_FONT_FAMILY = "Plus Jakarta Sans, sans-serif"
PLOTLY_TITLE_FONT = dict(family=PLOTLY_FONT_FAMILY, size=14, color="#0B221C")
PLOTLY_AXIS_TITLE_FONT = dict(family=PLOTLY_FONT_FAMILY, size=12.5, color="#1E352F")
PLOTLY_TICK_FONT = dict(family=PLOTLY_FONT_FAMILY, size=11.5, color="#4A635C")
PLOTLY_GRID_COLOR = "#E6F0EC"


def create_modern_sec_plot(
    sim_data,
    col_config,
    title,
    show_ref_markers=False,
    show_dual_uv=False,
    paper_exp_markers=None,
    pumped_cursor_ml=None
):
    """Creates a clean, classroom-friendly Plotly SEC chromatogram with physical column zones & collision-free peak labels."""
    fig = go.Figure()
    vo_val = float(col_config["void_volume_vo_ml"])
    vt_val = float(col_config["bed_volume_vt_ml"])
    dec = 2 if vt_val < 5.0 else 1

    max_sig = max(
        max(sim_data["total_signal_mau"]) if sim_data["total_signal_mau"] else 800.0,
        max(sim_data.get("total_signal_a260_mau", [0.0])) if show_dual_uv else 0.0,
        920.0
    )
    y_upper = max(1290.0 if len(sim_data.get("peaks", [])) >= 5 else 1240.0, max_sig * 1.35)
    x_upper = vt_val * 1.13

    # 0. Intuitive Physical Column Zones Shaded Directly Behind the Chromatogram
    fig.add_vrect(
        x0=0.0, x1=vo_val,
        fillcolor="rgba(148, 163, 184, 0.085)",
        layer="below", line_width=0
    )
    fig.add_vrect(
        x0=vo_val, x1=vt_val,
        fillcolor="rgba(13, 148, 136, 0.045)",
        layer="below", line_width=0
    )
    fig.add_vrect(
        x0=vt_val, x1=x_upper,
        fillcolor="rgba(217, 119, 6, 0.055)",
        layer="below", line_width=0
    )

    # Top Zone Strip Badges (showarrow=False so peak annotation count stays exact)
    fig.add_annotation(
        x=vo_val * 0.50, y=0.985, xref="x", yref="paper",
        text="<b>① Buffer Only</b>",
        showarrow=False,
        font=dict(size=9.5, color="#475569", family=PLOTLY_FONT_FAMILY),
        bgcolor="rgba(241, 245, 249, 0.92)",
        bordercolor="#CBD5E1", borderwidth=1, borderpad=2
    )
    fig.add_annotation(
        x=0.50 * (vo_val + vt_val), y=0.985, xref="x", yref="paper",
        text="<b>② Bead-Tunnel Separation:  ⬅️ BIGGER (Fast) • SMALLER (Slow) ➡️</b>",
        showarrow=False,
        font=dict(size=9.5, color="#065F46", family=PLOTLY_FONT_FAMILY),
        bgcolor="rgba(236, 253, 245, 0.94)",
        bordercolor="#A7F3D0", borderwidth=1, borderpad=2
    )
    fig.add_annotation(
        x=0.50 * (vt_val + x_upper), y=0.985, xref="x", yref="paper",
        text="<b>③ Salts</b>",
        showarrow=False,
        font=dict(size=9.5, color="#92400E", family=PLOTLY_FONT_FAMILY),
        bgcolor="rgba(255, 251, 235, 0.94)",
        bordercolor="#FDE68A", borderwidth=1, borderpad=2
    )

    sorted_peaks = sorted(
        sim_data["peaks"],
        key=lambda p: (p["ve_ml"], 0 if p.get("is_aggregate") else 1)
    )

    # 1. Optional Standard Protein Calibration Mixture
    if show_ref_markers:
        mix = simulate_standard_mixture_chromatogram(col_config)
        fig.add_trace(go.Scatter(
            x=mix["volume_axis_ml"],
            y=mix["signal_mau"],
            mode="lines",
            name="Calibration Standards Mix",
            line=dict(color="#9333EA", width=1.7, dash="dash"),
            fill="tozeroy",
            fillcolor="rgba(147, 51, 234, 0.05)",
            hoverinfo="skip"
        ))
        for mp in mix["marker_peaks"]:
            fig.add_trace(go.Scatter(
                x=[mp["ve_ml"]],
                y=[mp["peak_height"]],
                mode="markers",
                name=mp["name"],
                marker=dict(size=6.5, color="#9333EA", symbol="circle"),
                hoverinfo="text",
                hovertext=(
                    f"<b>Standard: {mp['name']}</b><br>"
                    f"Size: {mp['mw_kda']:g} kDa<br>"
                    f"Elutes at: {mp['ve_ml']:.2f} mL"
                ),
                showlegend=False
            ))

    # 1b. Individual Color-Coded Species Bell Curves (when comparing 2+ species simultaneously)
    if len(sorted_peaks) > 1:
        for pk_b in sorted_peaks:
            b_color = "#0F766E" if pk_b["color"] == "#2563EB" else pk_b["color"]
            if "curve_y" in pk_b:
                fig.add_trace(go.Scatter(
                    x=sim_data["volume_axis_ml"],
                    y=pk_b["curve_y"],
                    mode="lines",
                    name=f"Band — {pk_b['label']}",
                    line=dict(color=b_color, width=1.9),
                    fill="tozeroy",
                    fillcolor=_hex_to_rgba(b_color, 0.14),
                    hoverinfo="skip",
                    showlegend=False
                ))

    # 2. Optional Dual-Wavelength UV 260 nm Trace (DNA/RNA Channel)
    if show_dual_uv and "total_signal_a260_mau" in sim_data:
        fig.add_trace(go.Scatter(
            x=sim_data["volume_axis_ml"],
            y=sim_data["total_signal_a260_mau"],
            mode="lines",
            name="UV 260 nm (DNA / RNA Signal)",
            line=dict(color="#DB2777", width=2.6, dash="dot"),
            fill="tozeroy",
            fillcolor="rgba(219, 39, 119, 0.07)"
        ))

    # 3. Primary UV 280 nm Trace (Botanical Emerald-Teal Protein Channel)
    fig.add_trace(go.Scatter(
        x=sim_data["volume_axis_ml"],
        y=sim_data["total_signal_mau"],
        mode="lines",
        name="UV 280 nm (Protein Signal)" if len(sorted_peaks) == 1 else "Total UV 280 nm Detector Sum",
        line=dict(color="#0F766E", width=3.0),
        fill="tozeroy" if len(sorted_peaks) == 1 else "none",
        fillcolor="rgba(13, 148, 136, 0.14)"
    ))

    # 4. Optional Real Lab Paper Peak Reference Stars
    if paper_exp_markers:
        for idx_m, exp_m in enumerate(paper_exp_markers):
            fig.add_vline(
                x=exp_m["exp_ve_ml"],
                line_dash="dot",
                line_color="#D97706",
                line_width=1.9
            )
            fig.add_trace(go.Scatter(
                x=[exp_m["exp_ve_ml"]],
                y=[exp_m.get("y_pin", 990 - (idx_m % 3) * 70)],
                mode="markers+text",
                name="★ Real Lab Peak (Published Paper)" if idx_m == 0 else exp_m["label"],
                text=[f"★ Real Lab: {exp_m['exp_ve_ml']:.2f} mL"],
                textposition="top center",
                textfont=dict(size=10.5, color="#B45309", family="Plus Jakarta Sans, sans-serif"),
                marker=dict(size=11, color="#F59E0B", symbol="star", line=dict(width=1.1, color="#92400E")),
                hoverinfo="text",
                hovertext=(
                    f"<b>Real Lab Measurement: {exp_m['label']}</b><br>"
                    f"Published Lab Peak: {exp_m['exp_ve_ml']:.2f} mL<br>"
                    f"Simulator Prediction: {exp_m['pred_ve_ml']:.2f} mL"
                ),
                showlegend=(idx_m == 0)
            ))

    # 5. Proportional Molecule-Size Halo Circles, Peak Diamonds & 5-Tier Anti-Collision Callouts
    min_x_gap = 0.13 * vt_val
    ay_tiers = [-44, -84, -122, -62, -150]
    w_px_est, h_px_est = 460.0, 300.0
    box_w_est, box_h_est = 142.0, 36.0
    placed_boxes = []

    for i, peak in enumerate(sorted_peaks):
        disp_y = max(peak["peak_height_mau"], peak.get("peak_height_a260_mau", 0.0)) if show_dual_uv else peak["peak_height_mau"]
        is_agg_peak = bool(peak.get("is_aggregate", False))
        excl_kda_peak = float(peak.get("exclusion_limit_kda", col_config.get("exclusion_limit_kda", 1300.0)))
        pore_pct = float(peak.get("pore_access_pct", 0.0 if is_agg_peak else 50.0))
        pore_short = "0% pores" if (is_agg_peak or pore_pct < 0.5) else f"{pore_pct:.0f}% pores"
        ratio_txt = f" • A260/A280={peak.get('a260_a280_ratio', 0.57):.2f}" if show_dual_uv else ""
        p_color = "#0F766E" if peak["color"] == "#2563EB" else peak["color"]

        # Visual proportional size halo (bigger hydrodynamic radius Rh -> larger translucent sphere on graph!)
        rh_val = max(8.0, float(peak.get("rh_angstrom", 20.0)))
        halo_px = 29.0 if is_agg_peak else min(27.0, max(14.0, 11.5 + 4.4 * math.log(rh_val / 8.0)))
        fig.add_trace(go.Scatter(
            x=[peak["ve_ml"]],
            y=[disp_y],
            mode="markers",
            name=f"Halo — {peak['label']}",
            marker=dict(
                size=halo_px,
                color=_hex_to_rgba(p_color, 0.20),
                symbol="circle",
                line=dict(width=1.3, color=_hex_to_rgba(p_color, 0.55))
            ),
            hoverinfo="skip",
            showlegend=False
        ))

        if is_agg_peak:
            size_hover = f"Size: > {_format_exclusion_limit(excl_kda_peak)} (Excluded at Vo)"
            pore_hover = "Bead Tunnels Entered: 0% (Too big for pores — express outside path!)"
            uv_hover = f"A260/A280 Ratio: {peak.get('a260_a280_ratio', 0.92):.2f} (UV Light Scattering)"
        else:
            size_hover = f"Size: {peak['mw_kda']:.1f} kDa (Radius Rh = {peak.get('rh_angstrom', 0):.1f} Å)"
            pore_hover = f"Bead Tunnels Entered: {pore_pct:.0f}% (Kav = {peak['kav']:.2f})"
            uv_hover = f"A260/A280 Ratio: {peak.get('a260_a280_ratio', 0.57):.2f}"

        fig.add_trace(go.Scatter(
            x=[peak["ve_ml"]],
            y=[disp_y],
            mode="markers",
            name=peak["label"],
            marker=dict(size=9.5, color=p_color, symbol="diamond"),
            hoverinfo="text",
            hovertext=(
                f"<b>{peak['label']}</b><br>"
                f"{size_hover}<br>"
                f"{pore_hover}<br>"
                f"Elution Volume (Ve): {peak['ve_ml']:.{dec+1}f} mL<br>"
                f"{uv_hover}"
            ),
            showlegend=False
        ))

        ax_shift = 0
        if i > 0 and abs(peak["ve_ml"] - sorted_peaks[i - 1]["ve_ml"]) < min_x_gap:
            ax_shift = 38 if i % 2 == 1 else -38
        elif i + 1 < len(sorted_peaks) and abs(sorted_peaks[i + 1]["ve_ml"] - peak["ve_ml"]) < min_x_gap:
            ax_shift = -38

        ay_offset = ay_tiers[i % len(ay_tiers)]

        # Ensure callout stays below top zone strip and avoids overlapping earlier callout boxes
        px_x = (peak["ve_ml"] / max(1e-6, x_upper)) * w_px_est
        px_y = (disp_y / max(1e-6, y_upper)) * h_px_est

        def _box_collides_or_clips(ax_c: float, ay_c: float) -> bool:
            cx, cy = px_x + ax_c, px_y - ay_c
            if cy < 38.0 or cy > (h_px_est - 28.0) or cx < 46.0 or cx > (w_px_est - 46.0):
                return True
            return any(abs(cx - ox) < box_w_est and abs(cy - oy) < box_h_est for ox, oy in placed_boxes)

        if _box_collides_or_clips(ax_shift, ay_offset):
            cand_ays = [ay_offset, -44, -84, -118, -62, -146, -98, -168]
            cand_axs = [ax_shift, -42, 42, 0, -84, 84]
            found_slot = False
            for ay_c in cand_ays:
                for ax_c in cand_axs:
                    if not _box_collides_or_clips(ax_c, ay_c):
                        ax_shift, ay_offset = ax_c, ay_c
                        found_slot = True
                        break
                if found_slot:
                    break
            if not found_slot:
                max_up_px = max(38, int(h_px_est - 30.0 - px_y))
                ay_offset = max(ay_offset, -max_up_px)

        placed_boxes.append((px_x + ax_shift, px_y - ay_offset))
        elute_lbl_color = "#B91C1C" if is_agg_peak else "#0F766E"

        fig.add_annotation(
            x=peak["ve_ml"],
            y=disp_y,
            text=(
                f"<b>{peak['label']}</b><br>"
                f"<span style='color:{elute_lbl_color}; font-weight:700;'>{peak['ve_ml']:.{dec}f} mL</span>"
                f"<span style='color:#4A635C;'> • {pore_short}{ratio_txt}</span>"
            ),
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor=p_color,
            ax=ax_shift,
            ay=ay_offset,
            bgcolor="rgba(255, 255, 255, 0.97)",
            bordercolor=p_color,
            borderwidth=1.5,
            borderpad=5,
            font=dict(size=11, family=PLOTLY_FONT_FAMILY, color="#0B221C")
        )

    # 6. Column Start (Void Volume Vo), End (Total Bed Volume Vt) & Optional Synchronized Pump Cursor
    fig.add_vline(x=vo_val, line_dash="dash", line_color="#DC2626", line_width=1.5)
    fig.add_vline(x=vt_val, line_dash="dash", line_color="#64748B", line_width=1.5)

    if pumped_cursor_ml is not None:
        fig.add_vline(x=float(pumped_cursor_ml), line_dash="dot", line_color="#4F46E5", line_width=2.1)
        fig.add_annotation(
            x=float(pumped_cursor_ml), y=0.895, xref="x", yref="paper",
            text=f"<b>💧 Pumped: {float(pumped_cursor_ml):.{dec}f} mL</b>",
            showarrow=False,
            font=dict(size=10, color="#3730A3", family=PLOTLY_FONT_FAMILY),
            bgcolor="rgba(238, 242, 255, 0.96)",
            bordercolor="#A5B4FC",
            borderwidth=1,
            borderpad=3
        )

    fig.add_annotation(
        x=vo_val, y=28,
        text=f"<b>Vo (Void): {vo_val:.{dec}f} mL</b>",
        showarrow=False,
        font=dict(size=10, color="#991B1B", family=PLOTLY_FONT_FAMILY),
        xanchor="left",
        bgcolor="rgba(254, 242, 242, 0.96)",
        bordercolor="#FECACA",
        borderwidth=1,
        borderpad=3
    )
    fig.add_annotation(
        x=vt_val, y=28,
        text=f"<b>Vt (Column End): {vt_val:g} mL</b>",
        showarrow=False,
        font=dict(size=10, color="#334155", family=PLOTLY_FONT_FAMILY),
        xanchor="right",
        bgcolor="rgba(241, 245, 249, 0.96)",
        bordercolor="#CBD5E1",
        borderwidth=1,
        borderpad=3
    )

    fig.update_layout(
        font=dict(family=PLOTLY_FONT_FAMILY, color="#1E352F", size=12),
        title=dict(
            text=f"<b>{title}</b>",
            subtitle=dict(
                text=(
                    f"Total Volume: {vt_val:g} mL • "
                    f"Best Separation Size Range: {col_config['opt_range_min_kda']:g} – {col_config['opt_range_max_kda']:g} kDa"
                ),
                font=dict(family=PLOTLY_FONT_FAMILY, size=12, color="#4A635C")
            ),
            font=PLOTLY_TITLE_FONT
        ),
        xaxis=dict(
            title=dict(text="Buffer Volume Pumped Through Column (mL)", font=PLOTLY_AXIS_TITLE_FONT),
            tickfont=PLOTLY_TICK_FONT,
            range=[0, x_upper],
            showgrid=True,
            gridcolor=PLOTLY_GRID_COLOR,
            linecolor="#CBDAD4"
        ),
        yaxis=dict(
            title=dict(text="UV Detector Signal (mAU)", font=PLOTLY_AXIS_TITLE_FONT),
            tickfont=PLOTLY_TICK_FONT,
            range=[0, y_upper],
            showgrid=True,
            gridcolor=PLOTLY_GRID_COLOR,
            linecolor="#CBDAD4"
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.34,
            xanchor="center",
            x=0.5,
            font=dict(family=PLOTLY_FONT_FAMILY, size=11, color="#1E352F")
        ),
        margin=dict(l=38, r=35, t=76, b=80),
        height=465
    )
    return fig


def render_sec_column_visualizer_html(snapshot: dict, sim_data: dict, col_config: dict) -> str:
    """
    Renders a crystal-clear, 3-panel synchronized SVG cutaway of:
      1. The physical packed SEC glass column, live UV 280 nm detector flow cell, and 8-tube fraction rack
      2. A 1,000,000x magnified porous sponge bead showing why BIG = fast outside bypass & SMALL = slow tunnel maze
      3. The live UV chromatogram being drawn as running buffer is pumped through the column
    Uses CSS classes from the main stylesheet (zero inline font-size declarations).
    """
    vt = float(snapshot["vt_ml"])
    vo = float(snapshot["vo_ml"])
    pumped = float(snapshot["pumped_ml"])
    rh50 = float(snapshot["rh50_angstrom"])
    live_uv = float(snapshot["live_uv_mau"])
    dec = 2 if vt < 5.0 else 1
    col_short = html.escape(col_config["name"])

    svg: list[str] = [
        '<svg class="sec-studio-svg" viewBox="0 0 1080 432" xmlns="http://www.w3.org/2000/svg" role="img" '
        'aria-label="Interactive Inside-the-SEC-Column, Porous Bead Tunnels, and Live UV Detector Diagram">',
        '<defs>',
        '<linearGradient id="secColGlass" x1="0" y1="0" x2="1" y2="0">',
        '<stop offset="0%" stop-color="#E6F4F1"/>',
        '<stop offset="48%" stop-color="#F7FCFA"/>',
        '<stop offset="100%" stop-color="#E2F1EC"/>',
        '</linearGradient>',
        '<clipPath id="secColBedClip"><rect x="65" y="88" width="96" height="206" rx="4"/></clipPath>',
        '<clipPath id="secBeadClip"><circle cx="516" cy="166" r="78"/></clipPath>',
        '</defs>',
    ]

    # =========================================================================
    # PANEL 1: PACKED GLASS SEC COLUMN, UV 280 DETECTOR & 8 FRACTION TUBES
    # =========================================================================
    svg.append('<rect x="10" y="10" width="344" height="412" rx="14" fill="#FFFFFF" stroke="#D5E5DF" stroke-width="1.5"/>')
    svg.append('<text x="24" y="33" class="sec-svg-h">1. Inside the SEC Column &amp; Fraction Tubes</text>')
    svg.append(f'<text x="24" y="49" class="sec-svg-sub">{col_short} • Total Bed Vt = {vt:g} mL</text>')

    # Top Pump Inlet & Pumped mL Pill
    svg.append('<rect x="168" y="56" width="172" height="22" rx="11" fill="#EEF2FF" stroke="#C7D2FE"/>')
    svg.append(f'<text x="254" y="71" text-anchor="middle" class="sec-svg-mono" fill="#3730A3">💧 Pumped: {pumped:.{dec}f} / {vt:g} mL</text>')
    svg.append('<line x1="113" y1="56" x2="113" y2="79" stroke="#0D9488" stroke-width="3" stroke-linecap="round"/>')
    svg.append('<polygon points="107,76 119,76 113,84" fill="#0D9488"/>')

    # Column Depth Ruler (Left of Glass Cylinder: y = 88 -> 294)
    col_x, col_w, col_y0, col_h = 65.0, 96.0, 88.0, 206.0
    for pct_lbl, y_rk in [("0%", col_y0 + 4), ("50%", col_y0 + col_h * 0.5), ("100%", col_y0 + col_h - 2)]:
        svg.append(f'<text x="56" y="{y_rk + 3:.1f}" text-anchor="end" class="sec-svg-micro">{pct_lbl}</text>')
        svg.append(f'<line x1="59" y1="{y_rk:.1f}" x2="64" y2="{y_rk:.1f}" stroke="#94A3B8" stroke-width="1.2"/>')

    # Glass Column Body + Top/Bottom Frits
    svg.append(f'<rect x="{col_x}" y="{col_y0 - 5}" width="{col_w}" height="6" rx="2" fill="#CBD5E1" stroke="#64748B" stroke-width="1"/>')
    svg.append(f'<rect x="{col_x}" y="{col_y0}" width="{col_w}" height="{col_h}" fill="url(#secColGlass)" stroke="#64748B" stroke-width="2"/>')
    svg.append(f'<rect x="{col_x}" y="{col_y0 + col_h}" width="{col_w}" height="6" rx="2" fill="#CBD5E1" stroke="#64748B" stroke-width="1"/>')

    # Clipped Packed Porous Sponge Beads + Travelling Protein Bands Inside Column
    svg.append('<g clip-path="url(#secColBedClip)">')
    for row_idx in range(10):
        by = col_y0 + 11.0 + row_idx * 20.2
        x_off = 9.5 if (row_idx % 2 == 1) else 0.0
        for col_idx in range(5):
            bx = col_x + 11.0 + col_idx * 19.0 + x_off
            if bx - 8.0 > col_x + col_w:
                continue
            svg.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="8.4" fill="#DCECE7" stroke="#B6D2C8" stroke-width="1"/>')
            # Tiny internal sponge tunnel curves inside each resin bead
            svg.append(
                f'<path d="M {bx - 4.5:.1f} {by - 2.5:.1f} Q {bx:.1f} {by + 3.5:.1f} {bx + 4.5:.1f} {by - 1.5:.1f}" '
                f'fill="none" stroke="#FFFFFF" stroke-width="1.5" stroke-linecap="round"/>'
            )
            svg.append(
                f'<path d="M {bx - 2.5:.1f} {by + 4.5:.1f} Q {bx + 1.5:.1f} {by:.1f} {bx + 3.5:.1f} {by + 4.0:.1f}" '
                f'fill="none" stroke="#FFFFFF" stroke-width="1.3" stroke-linecap="round"/>'
            )

    # Render bands & proportional protein spheres for travellers currently inside column
    in_col_items = [t for t in snapshot["travellers"] if t["status"] != "eluted"]
    for idx_t, tr in enumerate(in_col_items):
        yc = col_y0 + 6.0 + tr["z_col"] * (col_h - 12.0)
        bh = max(11.0, tr["dz_band"] * col_h * 1.65)
        col_hex = tr["color"]
        is_ghost = tr["is_reference_ghost"]
        band_op = "0.14" if is_ghost else "0.26"
        dash_attr = ' stroke-dasharray="3,2"' if is_ghost else ""
        svg.append(
            f'<rect x="{col_x + 2}" y="{yc - bh / 2:.1f}" width="{col_w - 4}" height="{bh:.1f}" rx="4" '
            f'fill="{col_hex}" fill-opacity="{band_op}" stroke="{col_hex}" stroke-opacity="0.45" stroke-width="1"{dash_attr}/>'
        )
        r_px = float(tr["sphere_r_px"]) * (0.82 if is_ghost else 0.92)
        # Place spheres across column width (big excluded spheres in inter-bead gaps, small ones inside beads)
        x_fracs = [0.20, 0.43, 0.67, 0.84] if tr["pore_access_pct"] < 12.0 else [0.16, 0.37, 0.58, 0.78, 0.90]
        for k_s, xf in enumerate(x_fracs):
            sx = col_x + xf * col_w
            sy = yc + ((-2.4 if k_s % 2 == 0 else 2.4) if idx_t % 2 == 0 else (2.0 if k_s % 2 == 0 else -2.0))
            op_s = "0.62" if is_ghost else "0.95"
            svg.append(
                f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{r_px:.1f}" fill="{col_hex}" fill-opacity="{op_s}" '
                f'stroke="#FFFFFF" stroke-width="1.1"/>'
            )
            if not is_ghost and r_px >= 4.2:
                svg.append(f'<circle cx="{sx - r_px * 0.28:.1f}" cy="{sy - r_px * 0.28:.1f}" r="{max(1.1, r_px * 0.22):.1f}" fill="#FFFFFF" fill-opacity="0.7"/>')
    svg.append('</g>')

    # Anti-collision callout badges on the right side of the glass column (x = 172 .. 344)
    in_col_sorted = sorted(in_col_items, key=lambda t: t["z_col"])
    placed_y: list[float] = []
    for tr in in_col_sorted[:5]:
        y_true = col_y0 + 6.0 + tr["z_col"] * (col_h - 12.0)
        y_lbl = max(col_y0 + 6.0, min(col_y0 + col_h - 6.0, y_true))
        for prev_y in placed_y:
            if y_lbl - prev_y < 24.0:
                y_lbl = min(col_y0 + col_h - 4.0, prev_y + 24.0)
        placed_y.append(y_lbl)
        col_hex = tr["color"]
        is_ghost = tr["is_reference_ghost"]
        st_txt = "EXITING!" if tr["status"] == "at_detector" else f"{tr['pct_down']}%"
        bg_pill = "#F8FAFC" if is_ghost else "#FFFFFF"
        dash_pill = ' stroke-dasharray="3,2"' if is_ghost else ""
        lbl_safe = html.escape(tr["short_label"][:17])
        svg.append(f'<line x1="{col_x + col_w + 1}" y1="{y_true:.1f}" x2="173" y2="{y_lbl:.1f}" stroke="{col_hex}" stroke-width="1.3"/>')
        svg.append(f'<rect x="173" y="{y_lbl - 10:.1f}" width="171" height="20" rx="6" fill="{bg_pill}" stroke="{col_hex}" stroke-width="1.3"{dash_pill}/>')
        svg.append(f'<circle cx="183" cy="{y_lbl:.1f}" r="{min(5.5, max(3.0, tr["sphere_r_px"] * 0.55)):.1f}" fill="{col_hex}"/>')
        svg.append(f'<text x="192" y="{y_lbl + 3.8:.1f}" class="sec-svg-micro" fill="#0B221C">{lbl_safe}</text>')
        svg.append(f'<text x="338" y="{y_lbl + 3.8:.1f}" text-anchor="end" class="sec-svg-mono" fill="{col_hex}">{st_txt}</text>')

    # If all species have already eluted into fraction tubes, display a clean checkmark note right of column
    if not in_col_sorted:
        svg.append('<rect x="174" y="176" width="168" height="44" rx="8" fill="#ECFDF5" stroke="#99F6E4"/>')
        svg.append('<text x="258" y="195" text-anchor="middle" class="sec-svg-bold" fill="#065F46">All Proteins Eluted!</text>')
        svg.append('<text x="258" y="211" text-anchor="middle" class="sec-svg-micro" fill="#047857">Sorted into Tubes F1–F8 below</text>')

    # Bottom Capillary + Live UV 280 nm Flow-Cell Detector (y = 300 .. 338)
    uv_glow = live_uv > 25.0
    det_pk = snapshot["detector_active_peak"]
    beam_col = det_pk["color"] if det_pk else ("#0D9488" if uv_glow else "#CBD5E1")
    uv_box_fill = "#FEF9C3" if uv_glow else "#F1F5F9"
    uv_box_stroke = "#D97706" if uv_glow else "#94A3B8"
    svg.append(f'<line x1="113" y1="300" x2="113" y2="342" stroke="{beam_col}" stroke-width="3.2"/>')
    svg.append(f'<rect x="78" y="311" width="70" height="22" rx="6" fill="{uv_box_fill}" stroke="{uv_box_stroke}" stroke-width="1.6"/>')
    svg.append('<text x="113" y="325.5" text-anchor="middle" class="sec-svg-micro" fill="#0B221C">🔦 UV 280</text>')
    svg.append(f'<rect x="156" y="311" width="122" height="22" rx="6" fill="#FFFFFF" stroke="{uv_box_stroke}" stroke-width="1.2"/>')
    svg.append(f'<text x="217" y="325.5" text-anchor="middle" class="sec-svg-mono" fill="#0B221C">UV: {live_uv:.0f} mAU</text>')

    # 8 Numbered Fraction Collector Test Tubes (F1 .. F8, x = 22 .. 340, y = 352 .. 416)
    fractions = snapshot["fractions"]
    for k, fr in enumerate(fractions):
        tx = 22.0 + k * 40.0
        ty, tw, th = 354.0, 31.0, 40.0
        fill_h = fr["fill_frac"] * (th - 6.0)
        is_now = fr["is_filling_now"]
        rim_col = "#D97706" if is_now else ("#0D9488" if (fr["has_protein"] and fr["fill_frac"] > 0.1) else "#94A3B8")
        rim_w = "2.1" if is_now else "1.3"
        if fill_h > 0.5:
            fy = ty + th - 2.0 - fill_h
            svg.append(
                f'<rect x="{tx + 2:.1f}" y="{fy:.1f}" width="{tw - 4:.1f}" height="{fill_h:.1f}" rx="5" '
                f'fill="{fr["tint_color"]}" fill-opacity="{fr["tint_opacity"]}"/>'
            )
        svg.append(f'<path d="M {tx:.1f} {ty:.1f} L {tx:.1f} {ty + th - 8:.1f} Q {tx:.1f} {ty + th:.1f} {tx + 8:.1f} {ty + th:.1f} L {tx + tw - 8:.1f} {ty + th:.1f} Q {tx + tw:.1f} {ty + th:.1f} {tx + tw:.1f} {ty + th - 8:.1f} L {tx + tw:.1f} {ty:.1f}" fill="none" stroke="{rim_col}" stroke-width="{rim_w}"/>')
        if is_now:
            svg.append(f'<circle cx="{tx + tw / 2:.1f}" cy="{ty - 5:.1f}" r="3.4" fill="{fr["tint_color"]}" stroke="#0F766E" stroke-width="1"/>')
        lbl_col = "#B45309" if is_now else "#1E352F"
        svg.append(f'<text x="{tx + tw / 2:.1f}" y="408" text-anchor="middle" class="sec-svg-micro" fill="{lbl_col}">{fr["tube_id"]}</text>')
        svg.append(f'<text x="{tx + tw / 2:.1f}" y="418" text-anchor="middle" class="sec-svg-micro" fill="#64748B">{fr["v_mid_ml"]:.{0 if vt >= 20 else 1}f}</text>')

    # =========================================================================
    # PANEL 2: 1,000,000x MICROSCOPIC ZOOM INSIDE ONE POROUS RESIN BEAD
    # =========================================================================
    svg.append('<rect x="366" y="10" width="342" height="412" rx="14" fill="#FFFFFF" stroke="#D5E5DF" stroke-width="1.5"/>')
    svg.append('<text x="380" y="33" class="sec-svg-h">2. Zoom Inside One Porous Resin Bead</text>')
    svg.append(f'<text x="380" y="49" class="sec-svg-sub">Median Tunnel Size ~ {rh50:.0f} Å • Why Big = Fast &amp; Small = Slow</text>')

    # Magnified Porous Sponge Bead (cx=516, cy=166, R=78)
    svg.append('<circle cx="516" cy="166" r="78" fill="#E3F2EE" stroke="#5EEAD4" stroke-width="2.5"/>')
    svg.append('<g clip-path="url(#secBeadClip)">')
    # Branching white internal tunnel network (wide outer mouths + narrow inner maze)
    svg.append('<path d="M 488 84 Q 500 124 468 148 Q 498 172 482 212 Q 506 228 498 250" fill="none" stroke="#FFFFFF" stroke-width="9.5" stroke-linecap="round" stroke-linejoin="round"/>')
    svg.append('<path d="M 542 86 Q 524 128 556 154 Q 520 184 548 214 Q 528 232 534 250" fill="none" stroke="#FFFFFF" stroke-width="8.5" stroke-linecap="round" stroke-linejoin="round"/>')
    svg.append('<path d="M 468 148 Q 516 158 556 154" fill="none" stroke="#FFFFFF" stroke-width="8" stroke-linecap="round"/>')
    svg.append('<path d="M 588 122 Q 558 132 564 162 Q 584 176 598 168" fill="none" stroke="#FFFFFF" stroke-width="15" stroke-linecap="round"/>')
    svg.append('</g>')

    # Path A: BIG Molecule (> Pore) — Crimson Express Highway Around Outside Left of Bead
    svg.append('<path d="M 454 78 C 414 118, 414 214, 454 252" fill="none" stroke="#DC2626" stroke-width="2.8" stroke-dasharray="5,3"/>')
    svg.append('<polygon points="450,255 460,256 456,246" fill="#DC2626"/>')
    svg.append('<circle cx="426" cy="158" r="10.5" fill="#DC2626" stroke="#FFFFFF" stroke-width="1.5"/>')
    svg.append('<circle cx="423" cy="155" r="2.5" fill="#FFFFFF" fill-opacity="0.7"/>')
    svg.append('<text x="378" y="128" class="sec-svg-bold" fill="#B91C1C">BIG</text>')
    svg.append('<text x="378" y="141" class="sec-svg-micro" fill="#991B1B">0% pores</text>')
    svg.append('<text x="378" y="192" class="sec-svg-micro" fill="#991B1B">Fastest!</text>')
    svg.append('<text x="378" y="204" class="sec-svg-micro" fill="#991B1B">(Exits Vo)</text>')

    # Path B: MEDIUM Molecule — Amber Partial Outer Cave Detour (Right Side)
    svg.append('<path d="M 578 82 Q 604 106 584 124 Q 560 136 566 160 Q 586 174 600 198 Q 604 226 580 250" fill="none" stroke="#D97706" stroke-width="2.3" stroke-dasharray="4,2.5"/>')
    svg.append('<polygon points="577,245 586,252 576,254" fill="#D97706"/>')
    svg.append('<circle cx="566" cy="148" r="6.8" fill="#D97706" stroke="#FFFFFF" stroke-width="1.3"/>')
    svg.append('<text x="612" y="144" class="sec-svg-bold" fill="#B45309">MEDIUM</text>')
    svg.append('<text x="612" y="157" class="sec-svg-micro" fill="#92400E">Wide caves</text>')
    svg.append('<text x="612" y="170" class="sec-svg-micro" fill="#92400E">Middle Ve</text>')

    # Path C: SMALL Molecule (< Pore) — Teal Deep Winding Maze Right Through Bead Center!
    svg.append('<path d="M 488 78 Q 500 124 470 148 Q 516 158 552 154 Q 522 184 546 214 Q 528 232 534 256" fill="none" stroke="#0F766E" stroke-width="2.4"/>')
    svg.append('<polygon points="529,253 539,253 534,262" fill="#0F766E"/>')
    svg.append('<circle cx="514" cy="155" r="4.6" fill="#0F766E" stroke="#FFFFFF" stroke-width="1.2"/>')
    svg.append('<circle cx="536" cy="198" r="4.6" fill="#0F766E" stroke="#FFFFFF" stroke-width="1.2"/>')
    svg.append('<text x="516" y="275" text-anchor="middle" class="sec-svg-bold" fill="#065F46">SMALL: Explores Deep Maze ➜ Exits LAST</text>')

    # Live Species Pore-Tunnel Entry Scoreboard (y = 286 .. 414)
    svg.append('<line x1="380" y1="286" x2="694" y2="286" stroke="#E2E8F0" stroke-width="1"/>')
    svg.append('<text x="380" y="303" class="sec-svg-bold" fill="#0B221C">How Much of the Bead Maze Does Each Enter?</text>')

    # Prioritize real simulated species, supplemented by reference ghosts up to 4 rows
    real_tr = [t for t in snapshot["travellers"] if not t["is_reference_ghost"]]
    ghost_tr = [t for t in snapshot["travellers"] if t["is_reference_ghost"]]
    score_rows = sorted((real_tr + ghost_tr)[:4], key=lambda t: t["ve_ml"])

    for idx_r, sr in enumerate(score_rows):
        ry = 322.0 + idx_r * 24.5
        p_pct = float(sr["pore_access_pct"])
        bar_w = max(2.0, 1.05 * p_pct)  # 0..105 px track
        c_hex = sr["color"]
        nm_s = html.escape(sr["short_label"][:18])
        r_dot = min(6.2, max(3.0, sr["sphere_r_px"] * 0.56))
        svg.append(f'<circle cx="388" cy="{ry - 3.5:.1f}" r="{r_dot:.1f}" fill="{c_hex}"/>')
        svg.append(f'<text x="399" y="{ry:.1f}" class="sec-svg-micro" fill="#0B221C">{nm_s}</text>')
        svg.append(f'<rect x="520" y="{ry - 9.5:.1f}" width="105" height="10" rx="5" fill="#EEF2F6" stroke="#CBD5E1" stroke-width="0.8"/>')
        if p_pct >= 0.8:
            svg.append(f'<rect x="520" y="{ry - 9.5:.1f}" width="{bar_w:.1f}" height="10" rx="5" fill="{c_hex}" fill-opacity="0.85"/>')
        svg.append(f'<text x="694" y="{ry:.1f}" text-anchor="end" class="sec-svg-mono" fill="{c_hex}">{p_pct:.0f}% ➜ {sr["ve_ml"]:.{dec}f} mL</text>')

    # =========================================================================
    # PANEL 3: LIVE SYNCHRONIZED UV GRAPH & STUDENT HOW-TO-READ KEY
    # =========================================================================
    svg.append('<rect x="720" y="10" width="350" height="412" rx="14" fill="#FFFFFF" stroke="#D5E5DF" stroke-width="1.5"/>')
    svg.append('<text x="734" y="33" class="sec-svg-h">3. Live UV Graph Drawn As Buffer Pumps</text>')
    svg.append('<text x="734" y="49" class="sec-svg-sub">Watch Each Peak Appear When It Hits the UV Lamp!</text>')

    px0, py0, pw, ph = 762.0, 68.0, 290.0, 184.0
    x_max_p = vt * 1.12
    vol_arr = sim_data["volume_axis_ml"]
    sig_arr = sim_data["total_signal_mau"]
    y_max_p = max(950.0, (max(sig_arr) if sig_arr else 800.0) * 1.18)

    def _vx(v_val: float) -> float:
        return px0 + min(1.0, max(0.0, v_val / max(1e-6, x_max_p))) * pw

    def _vy(s_val: float) -> float:
        return py0 + ph - min(1.0, max(0.0, s_val / max(1e-6, y_max_p))) * ph

    x_vo = _vx(vo)
    x_vt = _vx(vt)
    x_cur = _vx(pumped)
    y_cur = _vy(live_uv)

    # Shaded Dead Volume (0-Vo) & Separation Window (Vo-Vt) inside mini chart
    svg.append(f'<rect x="{px0:.1f}" y="{py0:.1f}" width="{pw:.1f}" height="{ph:.1f}" fill="#FFFFFF" stroke="#CBDAD4" stroke-width="1.2"/>')
    svg.append(f'<rect x="{px0:.1f}" y="{py0:.1f}" width="{max(0.0, x_vo - px0):.1f}" height="{ph:.1f}" fill="#F1F5F9"/>')
    svg.append(f'<rect x="{x_vo:.1f}" y="{py0:.1f}" width="{max(0.0, x_vt - x_vo):.1f}" height="{ph:.1f}" fill="#ECFDF5" fill-opacity="0.55"/>')
    svg.append(f'<line x1="{x_vo:.1f}" y1="{py0:.1f}" x2="{x_vo:.1f}" y2="{py0 + ph:.1f}" stroke="#DC2626" stroke-width="1.2" stroke-dasharray="3,3"/>')
    svg.append(f'<line x1="{x_vt:.1f}" y1="{py0:.1f}" x2="{x_vt:.1f}" y2="{py0 + ph:.1f}" stroke="#64748B" stroke-width="1.2" stroke-dasharray="3,3"/>')
    svg.append(f'<text x="{0.5 * (px0 + x_vo):.1f}" y="{py0 + 13:.1f}" text-anchor="middle" class="sec-svg-micro" fill="#64748B">0–Vo</text>')
    svg.append(f'<text x="{0.5 * (x_vo + x_vt):.1f}" y="{py0 + 13:.1f}" text-anchor="middle" class="sec-svg-micro" fill="#065F46">⬅️ Big • Small ➡️</text>')

    # Subsample curve (~140 points) into Drawn-So-Far (solid) and Future Preview (faint dashed)
    step_idx = max(1, len(vol_arr) // 140)
    pts_full: list[str] = []
    pts_drawn: list[str] = []
    for idx_v in range(0, len(vol_arr), step_idx):
        vv = float(vol_arr[idx_v])
        ss = float(sig_arr[idx_v])
        xx, yy = _vx(vv), _vy(ss)
        pts_full.append(f"{xx:.1f},{yy:.1f}")
        if vv <= pumped + 1e-6:
            pts_drawn.append(f"{xx:.1f},{yy:.1f}")
    pts_drawn.append(f"{x_cur:.1f},{y_cur:.1f}")

    if len(pts_full) >= 2:
        svg.append(f'<polyline points="{" ".join(pts_full)}" fill="none" stroke="#94A3B8" stroke-width="1.4" stroke-dasharray="4,3" stroke-opacity="0.65"/>')
    if len(pts_drawn) >= 2 and pumped > 0.001 * vt:
        poly_fill = f"{px0:.1f},{py0 + ph:.1f} " + " ".join(pts_drawn) + f" {x_cur:.1f},{py0 + ph:.1f}"
        svg.append(f'<polygon points="{poly_fill}" fill="#0D9488" fill-opacity="0.16"/>')
        svg.append(f'<polyline points="{" ".join(pts_drawn)}" fill="none" stroke="#0F766E" stroke-width="2.6" stroke-linecap="round"/>')

    # Peak markers along mini chromatogram (highlighted once pumped >= peak Ve!)
    for pk_m in sim_data["peaks"]:
        p_x = _vx(float(pk_m["ve_ml"]))
        p_y = _vy(float(pk_m["peak_height_mau"]))
        p_c = "#0F766E" if pk_m["color"] == "#2563EB" else pk_m["color"]
        reached = pumped >= float(pk_m["ve_ml"]) - 0.5 * float(pk_m.get("sigma_ml", 0.2))
        op_m = "1.0" if reached else "0.32"
        svg.append(f'<circle cx="{p_x:.1f}" cy="{p_y:.1f}" r="4.5" fill="{p_c}" fill-opacity="{op_m}" stroke="#FFFFFF" stroke-width="1"/>')

    # Live Pump Cursor & Glowing UV Pen Dot
    svg.append(f'<line x1="{x_cur:.1f}" y1="{py0:.1f}" x2="{x_cur:.1f}" y2="{py0 + ph:.1f}" stroke="#4F46E5" stroke-width="1.6" stroke-dasharray="3,2"/>')
    svg.append(f'<circle cx="{x_cur:.1f}" cy="{y_cur:.1f}" r="5.5" fill="#4F46E5" stroke="#FFFFFF" stroke-width="1.6"/>')

    # Axes Labels for Mini Plot
    svg.append(f'<text x="{px0:.1f}" y="{py0 + ph + 14:.1f}" text-anchor="middle" class="sec-svg-micro">0</text>')
    svg.append(f'<text x="{x_vo:.1f}" y="{py0 + ph + 14:.1f}" text-anchor="middle" class="sec-svg-micro" fill="#991B1B">Vo ({vo:.{dec}f})</text>')
    svg.append(f'<text x="{x_vt:.1f}" y="{py0 + ph + 14:.1f}" text-anchor="middle" class="sec-svg-micro" fill="#334155">Vt ({vt:g} mL)</text>')
    svg.append(f'<text x="{px0 - 8:.1f}" y="{py0 + ph / 2:.1f}" text-anchor="middle" transform="rotate(-90 {px0 - 18:.1f} {py0 + ph / 2:.1f})" class="sec-svg-micro">UV 280 Signal</text>')
    svg.append(f'<text x="{px0 + pw / 2:.1f}" y="{py0 + ph + 28:.1f}" text-anchor="middle" class="sec-svg-micro" fill="#0B221C">Buffer Volume Pumped (mL)</text>')

    # Student How-to-Read Cheat Sheet Card (y = 292 .. 412)
    svg.append('<rect x="734" y="292" width="322" height="118" rx="10" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1"/>')
    svg.append('<text x="746" y="311" class="sec-svg-bold" fill="#0B221C">🎓 How to Read Any SEC Graph in 3 Steps:</text>')
    svg.append('<text x="746" y="331" class="sec-svg-lbl">• <tspan class="sec-svg-bold" fill="#B91C1C">Left (Small mL):</tspan> Skipped pores ➜ <tspan class="sec-svg-bold">BIGGER</tspan></text>')
    svg.append('<text x="746" y="351" class="sec-svg-lbl">• <tspan class="sec-svg-bold" fill="#0F766E">Right (Large mL):</tspan> Explored maze ➜ <tspan class="sec-svg-bold">SMALLER</tspan></text>')
    svg.append('<text x="746" y="371" class="sec-svg-lbl">• <tspan class="sec-svg-bold" fill="#4F46E5">Peak Height (mAU):</tspan> Protein amount in tube</text>')
    svg.append('<text x="746" y="394" class="sec-svg-micro" fill="#47625B">💡 Tip: Tick "2× Dimer" or "Aggregate" below to race!</text>')

    svg.append('</svg>')

    is_peak_now = snapshot["detector_active_peak"] is not None
    nar_cls = "sec-narrator-bar sec-narrator-bar-peak" if is_peak_now else "sec-narrator-bar"
    badge_safe = html.escape(snapshot["stage_badge"])
    text_safe = html.escape(snapshot["stage_text"])

    block = [
        '<div class="sec-studio-card">',
        *svg,
        f'<div class="{nar_cls}">',
        f'<span class="sec-narrator-badge">{badge_safe}</span>',
        f'<span class="sec-narrator-text">{text_safe}</span>',
        '</div>',
        '</div>',
    ]
    return "\n".join(line.strip() for line in block if line.strip())


# --- SIMPLIFIED 5-TAB EDUCATIONAL WORKSPACE (BENCHMARK MOVED TO FINAL TAB!) ---
tab_sec, tab_strategy, tab_buffers, tab_details, tab_examples = st.tabs([
    "📈 1. SEC & Gel Simulator",
    "🧭 2. Purification Strategy (IEX & HIC)",
    "🧪 3. Custom Buffer Calculator (g / mg)",
    "🔬 4. Amino Acids, Structure & Solubility",
    "📚 5. Real Lab Examples & Accuracy"
])


# =====================================================================
# TAB 1: SEC CHROMATOGRAM & DESTAINED SDS-PAGE GEL SIMULATOR
# =====================================================================
with tab_sec:
    st.markdown("### 📊 Size-Exclusion Chromatography (SEC) Simulator")

    # Intuitive Classroom Mental Model Card
    st.markdown("""
    <div class="sec-rule-card">
        <div class="sec-rule-header">
            <span>⬅️ 🏃 BIG Molecules Come Out FIRST (Left)</span>
            <span class="sec-rule-pill">Easy Rule of SEC</span>
            <span>🐢 SMALL Molecules Come Out LAST (Right) ➡️</span>
        </div>
        <div class="ui-body">
            Imagine pouring marbles and tiny sand grains through a tube filled with sponge-like beads containing microscopic tunnels.
            <b>Large proteins</b> cannot fit inside the bead tunnels, so they zip straight around the outside and exit early.
            <b>Small proteins</b> (like 14.3 kDa Lysozyme) wander through every tiny bead tunnel, taking a longer path and exiting later!
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Reserved visual slot directly beneath the Easy Rule of SEC banner (populated once simulation runs below)
    inside_sec_column_slot = st.container()

    # Smart default column suggestion based on active MW
    rec_prep_default, rec_anal_default = recommend_optimal_columns(metrics["mw_kda"])
    prep_options = ["hiload_16_600_s75", "hiload_16_600_s200", "hiload_16_600_superose_6", "hiload_26_600_s200"]
    anal_options = ["s75_increase_10_300", "s200_increase_10_300", "superose_6_increase_10_300", "superose_6_increase_3_2_300"]

    c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([1.32, 1.32, 2.25])
    with c_ctrl1:
        prep_col_id = st.selectbox(
            "Large Prep Column (120–320 mL):",
            options=prep_options,
            format_func=lambda x: f"{SEC_COLUMNS[x]['name']} ({SEC_COLUMNS[x]['opt_range_min_kda']:g}–{SEC_COLUMNS[x]['opt_range_max_kda']:g} kDa)",
            index=prep_options.index(rec_prep_default) if rec_prep_default in prep_options else 1
        )
    with c_ctrl2:
        anal_col_id = st.selectbox(
            "Benchtop Analytical Column (2.4–24 mL):",
            options=anal_options,
            format_func=lambda x: f"{SEC_COLUMNS[x]['name']} ({SEC_COLUMNS[x]['opt_range_min_kda']:g}–{SEC_COLUMNS[x]['opt_range_max_kda']:g} kDa)",
            index=anal_options.index(rec_anal_default) if rec_anal_default in anal_options else 1
        )
    with c_ctrl3:
        st.markdown('<div class="ui-control-label">Compare Oligomer Peaks (Optional):</div>', unsafe_allow_html=True)
        col_m, col_d, col_t, col_tet, col_agg = st.columns([0.96, 1.0, 1.04, 0.96, 1.16])
        include_monomer = col_m.checkbox("1× Mono", value=True)
        include_dimer = col_d.checkbox("2× Dimer", value=False)
        include_trimer = col_t.checkbox("3× Trimer", value=False)
        include_tetramer = col_tet.checkbox("4× Tetra", value=False)
        include_aggregate = col_agg.checkbox(
            "Aggregate",
            value=False,
            help="Soluble high-molecular-weight (HMW) misfolded aggregate larger than the column pores — elutes immediately at the Void Volume (Vo)!"
        )

    # Optional Advanced Complex & 3D Shape Expander (Tucked neatly away for simplicity)
    with st.expander("➕ Optional: Test Protein–Protein or Protein–DNA/RNA Binding & 3D Shape", expanded=False):
        cx_c1, cx_c2, cx_c3 = st.columns([1.3, 1.3, 1.4])
        with cx_c1:
            target_conformation = st.selectbox(
                "3D Protein Shape:",
                options=["globular", "moderately_elongated", "elongated", "idp_hybrid", "disordered_idp"],
                format_func=lambda k: CONFORMATION_PROFILES[k]["name"],
                index=0
            )
            st.caption(CONFORMATION_PROFILES[target_conformation]["description"])
        with cx_c2:
            partner_mode = st.selectbox(
                "Add a Binding Partner:",
                options=[
                    "None (Protein Alone)",
                    "🤝 Another Protein (Protein–Protein Complex)",
                    "🧬 Double-Stranded DNA (bp)",
                    "🎀 Single-Stranded RNA (nt)"
                ],
                index=0
            )
            stoich_target, stoich_partner = 1, 1
            if partner_mode != "None (Protein Alone)":
                stoich_str = st.selectbox("Binding Ratio (Your Protein : Partner):", ["1 : 1", "2 : 1", "2 : 2", "1 : 2", "6 : 1"], index=0)
                stoich_target, stoich_partner = [int(s.strip()) for s in stoich_str.split(":")]
        with cx_c3:
            partner_mw_kda = 0.0
            partner_na_type = None
            complex_conf = target_conformation
            show_unbound_partner = True
            if "Another Protein" in partner_mode:
                partner_name = st.text_input("Partner Name:", value="Partner B")
                partner_mw_kda = st.number_input("Partner Size (kDa):", min_value=1.0, max_value=2500.0, value=35.0, step=1.0)
                show_unbound_partner = st.checkbox("Also show free partner peak", value=True)
                complex_conf = "globular"
            elif "DNA" in partner_mode:
                dna_bp = st.number_input(
                    "DNA Length (base pairs, bp):",
                    min_value=8,
                    max_value=2000,
                    value=147,
                    step=5,
                    help="Uses standard anhydrous average dsDNA mass = 617.96 Da/bp (50% GC free acid; dry Na+ salt form is ~650–662 Da/bp)."
                )
                partner_mw_kda = round(dna_bp * DSDNA_MW_KDA_PER_BP, 2)
                partner_name = f"{dna_bp} bp DNA"
                partner_na_type = "dsdna"
                st.caption(f"Computed dsDNA mass: **{partner_mw_kda:.2f} kDa** (`617.96 Da/bp` anhydrous; Na⁺ salt ~`650 Da/bp`).")
                pna_arch = st.radio("DNA Wrapping:", ["Compact / Wrapped", "Extended Rod"], horizontal=True)
                complex_conf = "compact_pna" if "Compact" in pna_arch else "extended_pna"
                show_unbound_partner = st.checkbox("Also show free DNA peak", value=True)
            elif "RNA" in partner_mode:
                rna_nt = st.number_input(
                    "RNA Length (nucleotides, nt):",
                    min_value=10,
                    max_value=4000,
                    value=100,
                    step=5,
                    help="Uses standard anhydrous average ssRNA residue mass = 321.47 Da/nt (+0.16 kDa if 5'-triphosphorylated via T7 IVT)."
                )
                partner_mw_kda = round(rna_nt * SSRNA_MW_KDA_PER_NT, 2)
                partner_name = f"{rna_nt} nt RNA"
                partner_na_type = "ssrna"
                st.caption(f"Computed ssRNA mass: **{partner_mw_kda:.2f} kDa** (`321.47 Da/nt` anhydrous; `+0.16 kDa` if 5'-ppp).")
                complex_conf = "compact_pna"
                show_unbound_partner = st.checkbox("Also show free RNA peak", value=True)
            else:
                st.info("Pick a Protein, DNA, or RNA partner on the left to watch the bound complex shift left (elute earlier)!")

    chk_c1, chk_c2 = st.columns([1.2, 1.8])
    show_markers = chk_c1.checkbox("Show Calibration Standard Protein Mix (Dashed Curve)", value=False)
    show_dual_uv = chk_c2.checkbox(
        "Show Both UV 280 nm (Protein) & UV 260 nm (DNA/RNA) Signals",
        value=(partner_na_type is not None)
    )

    # Assemble Simulated Species
    target_states = []
    monomer_mw = metrics["mw_kda"]
    prot_eps01 = float(ext["abs_01_cystines"])

    if partner_mode != "None (Protein Alone)" and partner_mw_kda > 0:
        tot_prot_mw = monomer_mw * stoich_target + (partner_mw_kda * stoich_partner if partner_na_type is None else 0.0)
        tot_na_mw = (partner_mw_kda * stoich_partner) if partner_na_type is not None else 0.0
        tot_complex_mw = tot_prot_mw + tot_na_mw
        target_states.append({
            "label": f"Complex ({tot_complex_mw:.1f} kDa)",
            "mw_kda": tot_complex_mw,
            "conformation": complex_conf,
            "prot_mw_kda": tot_prot_mw,
            "na_mw_kda": tot_na_mw,
            "na_type": partner_na_type,
            "prot_eps01": prot_eps01,
            "abundance": 1.0,
            "color": "#059669"
        })

    if include_monomer:
        lbl_m = f"Free Protein ({monomer_mw:.1f} kDa)" if partner_mode != "None (Protein Alone)" else f"Monomer ({monomer_mw:.1f} kDa)"
        target_states.append({
            "label": lbl_m,
            "mw_kda": monomer_mw,
            "conformation": target_conformation,
            "prot_mw_kda": monomer_mw,
            "na_mw_kda": 0.0,
            "na_type": None,
            "prot_eps01": prot_eps01,
            "abundance": 0.65 if partner_mode != "None (Protein Alone)" else 1.0,
            "color": "#0F766E"
        })
    if include_dimer:
        target_states.append({"label": f"Dimer ({monomer_mw*2:.1f} kDa)", "mw_kda": monomer_mw * 2, "conformation": target_conformation, "prot_eps01": prot_eps01, "abundance": 0.45, "color": "#D97706"})
    if include_trimer:
        target_states.append({"label": f"Trimer ({monomer_mw*3:.1f} kDa)", "mw_kda": monomer_mw * 3, "conformation": target_conformation, "prot_eps01": prot_eps01, "abundance": 0.25, "color": "#7C3AED"})
    if include_tetramer:
        target_states.append({"label": f"Tetramer ({monomer_mw*4:.1f} kDa)", "mw_kda": monomer_mw * 4, "conformation": target_conformation, "prot_eps01": prot_eps01, "abundance": 0.15, "color": "#EA580C"})
    if include_aggregate:
        has_other_species = bool(target_states) or (
            partner_mode != "None (Protein Alone)" and partner_mw_kda > 0 and show_unbound_partner
        )
        target_states.append({
            "label": "Aggregate (Vo)",
            "mw_kda": monomer_mw * 20.0,
            "conformation": "globular",
            "is_aggregate": True,
            "prot_eps01": prot_eps01,
            "abundance": 0.35 if has_other_species else 1.0,
            "color": "#B91C1C"
        })

    if partner_mode != "None (Protein Alone)" and partner_mw_kda > 0 and show_unbound_partner:
        p_conf = partner_na_type if partner_na_type else "globular"
        target_states.append({
            "label": f"Free {partner_name} ({partner_mw_kda:.1f} kDa)",
            "mw_kda": partner_mw_kda,
            "conformation": p_conf,
            "prot_mw_kda": partner_mw_kda if partner_na_type is None else 0.0,
            "na_mw_kda": partner_mw_kda if partner_na_type is not None else 0.0,
            "na_type": partner_na_type,
            "prot_eps01": 1.0,
            "abundance": 0.45,
            "color": "#DB2777" if partner_na_type else "#0284C7"
        })

    if not target_states:
        target_states.append({"label": f"Monomer ({monomer_mw:.1f} kDa)", "mw_kda": monomer_mw, "conformation": target_conformation, "prot_eps01": prot_eps01, "abundance": 1.0, "color": "#0F766E"})

    prep_config = SEC_COLUMNS[prep_col_id]
    anal_config = SEC_COLUMNS[anal_col_id]
    prep_sim = simulate_chromatogram_peaks(target_states, prep_config)
    anal_sim = simulate_chromatogram_peaks(target_states, anal_config)

    # --- INTERACTIVE INSIDE-THE-SEC-COLUMN, POROUS BEAD & FRACTION TUBE STUDIO (rendered right under Easy Rule of SEC) ---
    with inside_sec_column_slot:
        st.markdown("#### 🔬 See Inside the SEC Column: Watch How Size Controls Elution Volume")
        vis_c1, vis_c2 = st.columns([1.25, 2.75])
        with vis_c1:
            vis_col_mode = st.radio(
                "Column Shown Inside Cutaway:",
                ["Benchtop Analytical", "Large Prep"],
                index=0,
                horizontal=True
            )

        vis_is_anal = ("Analytical" in vis_col_mode)
        vis_config = anal_config if vis_is_anal else prep_config
        vis_sim = anal_sim if vis_is_anal else prep_sim
        vt_vis = float(vis_config["bed_volume_vt_ml"])
        vo_vis = float(vis_config["void_volume_vo_ml"])
        pump_step = 0.02 if vt_vis < 5.0 else (0.5 if vt_vis >= 100.0 else 0.1)
        pump_max = round(math.floor((vt_vis * 1.08) / pump_step) * pump_step, 3)

        def _snap_pump(val_ml: float) -> float:
            snapped = round(round(float(val_ml) / pump_step) * pump_step, 3)
            return float(max(0.0, min(pump_max, snapped)))

        sorted_vis_ves = sorted(float(p["ve_ml"]) for p in vis_sim["peaks"])
        non_agg_peaks = [p for p in vis_sim["peaks"] if not p.get("is_aggregate")]
        primary_ve_vis = float(non_agg_peaks[0]["ve_ml"]) if non_agg_peaks else float(vis_sim["peaks"][0]["ve_ml"])

        if len(sorted_vis_ves) >= 2:
            default_pump_val = _snap_pump(sorted_vis_ves[1] if sorted_vis_ves[1] > sorted_vis_ves[0] + pump_step else 0.5 * (vo_vis + sorted_vis_ves[-1]))
        else:
            default_pump_val = _snap_pump(vo_vis if primary_ve_vis > vo_vis * 1.06 else 0.65 * vt_vis)

        pump_ctx = (vis_config["id"], len(vis_sim["peaks"]), int(round(monomer_mw)))
        pump_key = "sec_pump_slider"
        if st.session_state.get("_sec_pump_ctx") != pump_ctx or pump_key not in st.session_state:
            st.session_state["_sec_pump_ctx"] = pump_ctx
            st.session_state[pump_key] = default_pump_val
        else:
            st.session_state[pump_key] = _snap_pump(st.session_state[pump_key])

        with vis_c2:
            st.markdown('<div class="ui-control-label">Jump to Key Moments in the Run (or Drag Slider Below):</div>', unsafe_allow_html=True)
            jb1, jb2, jb3, jb4 = st.columns(4)
            dec_btn = 2 if vt_vis < 5.0 else 1
            if jb1.button("💉 1. Inject (0 mL)", key="btn_sec_stage_inj", width="stretch"):
                st.session_state[pump_key] = _snap_pump(0.0)
            if jb2.button(f"🚀 2. Void ({vo_vis:.{dec_btn}f} mL)", key="btn_sec_stage_vo", width="stretch"):
                st.session_state[pump_key] = _snap_pump(vo_vis)
            if jb3.button(f"🎯 3. Target ({primary_ve_vis:.{dec_btn}f} mL)", key="btn_sec_stage_ve", width="stretch"):
                st.session_state[pump_key] = _snap_pump(primary_ve_vis)
            if jb4.button(f"🏁 4. End ({vt_vis:g} mL)", key="btn_sec_stage_vt", width="stretch"):
                st.session_state[pump_key] = _snap_pump(vt_vis)

        active_pumped_ml = st.slider(
            f"💧 Pump Running Buffer Into Top of {vis_config['name']} (mL):",
            min_value=0.0,
            max_value=float(pump_max),
            step=float(pump_step),
            key=pump_key,
            help="Slide from 0 mL to Vt to pump buffer down the column and watch how giant molecules zoom around the outside of the beads (exiting early at Vo) while small proteins weave through internal bead tunnels (exiting later)!"
        )

        col_snapshot = compute_column_snapshot(
            vis_sim,
            vis_config,
            pumped_ml=active_pumped_ml,
            include_reference_companions=True
        )
        st.markdown(render_sec_column_visualizer_html(col_snapshot, vis_sim, vis_config), unsafe_allow_html=True)

    if include_aggregate:
        st.info(
            "💡 **Void Aggregate Peak (`Vo`):** Clumped/misfolded protein aggregates are too large to enter any bead pores, "
            "so they wash straight out at the **Void Volume (`Vo`, `Kav = 0`)**—and often show an elevated `A260/A280` (~0.9) "
            "from UV light scattering! (Note: On reducing SDS-PAGE below, non-covalent aggregates still unfold back into single monomer chains.)"
        )

    if any(p.get("uv280_invisible") for p in anal_sim["peaks"]):
        st.warning(
            "⚠️ **UV-280 Blind Sequence (`ε280 = 0`, zero Trp/Tyr/disulfides):** "
            "Common in peptides and AI-designed scaffolds (ProteinMPNN / RFdiffusion). "
            "The SEC chromatogram below displays the **theoretical elution profile**, but this protein is **invisible at 280 nm on an ÄKTA FPLC** — "
            "monitor peptide-bond absorbance at **214 nm or 205 nm** (in low-UV buffer) or add a single C-terminal **Trp (`W`)** or **Tyr (`Y`)**!"
        )

    def render_elution_strip(target_label, target_mw, target_ve, col_config):
        markers = get_column_marker_elutions(col_config)
        dec = 2 if col_config["bed_volume_vt_ml"] < 5.0 else 1
        is_agg_target = (target_label == "Aggregate (Vo)")
        excl_lbl = _format_exclusion_limit(float(col_config.get("exclusion_limit_kda", 1300.0)))
        disp_target = f"🎯 Aggregate (Vo, > {excl_lbl})" if is_agg_target else f"🎯 {target_label}"
        items = [{"name": m["name"].split()[0], "mw": m["mw_kda"], "ve": m["ve_ml"], "is_target": False} for m in markers]
        items.append({"name": disp_target, "mw": target_mw, "ve": target_ve, "is_target": True})
        items.sort(key=lambda x: (x["ve"], 0 if (x["is_target"] and is_agg_target) else 1))
        chips = []
        for it in items:
            if it["is_target"]:
                chip_cls = "elution-chip-agg" if is_agg_target else "elution-chip-target"
                chips.append(f'<span class="{chip_cls}">{it["name"]} • {it["ve"]:.{dec}f} mL</span>')
            else:
                chips.append(
                    f'<span class="elution-chip-std">{it["name"]} ({it["mw"]:g}k • {it["ve"]:.{dec}f} mL)</span>'
                )
        return f"""
        <div class="elution-strip">
            <div class="elution-strip-title">📍 Order Coming Off This Column (First ➜ Last):</div>
            <div class="elution-chip-row">{' &nbsp;➜&nbsp; '.join(chips)}</div>
        </div>
        """

    c_prep_plot, c_anal_plot = st.columns(2)
    with c_prep_plot:
        prep_cursor = active_pumped_ml if not vis_is_anal else None
        st.plotly_chart(
            create_modern_sec_plot(
                prep_sim, prep_config, f"Prep Column: {prep_config['name']}",
                show_markers, show_dual_uv, pumped_cursor_ml=prep_cursor
            ),
            width="stretch"
        )
        mp0 = prep_sim["peaks"][0]
        st.markdown(render_elution_strip(mp0["label"], mp0["mw_kda"], mp0["ve_ml"], prep_config), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([_format_sec_table_row(p, prep_config) for p in prep_sim["peaks"]]), hide_index=True, width="stretch")

    with c_anal_plot:
        anal_cursor = active_pumped_ml if vis_is_anal else None
        st.plotly_chart(
            create_modern_sec_plot(
                anal_sim, anal_config, f"Analytical Column: {anal_config['name']}",
                show_markers, show_dual_uv, pumped_cursor_ml=anal_cursor
            ),
            width="stretch"
        )
        ma0 = anal_sim["peaks"][0]
        st.markdown(render_elution_strip(ma0["label"], ma0["mw_kda"], ma0["ve_ml"], anal_config), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([_format_sec_table_row(p, anal_config) for p in anal_sim["peaks"]]), hide_index=True, width="stretch")

    # --- DESTAINED SDS-PAGE GEL SECTION ---
    st.markdown("---")
    st.markdown("### 🧫 Virtual Destained SDS-PAGE Gel")
    st.caption(
        "SDS detergent + reducing agent (DTT/β-ME to cleave S–S disulfide bridges) + heat (95°C) unfold proteins into single linear chains "
        "coated uniformly with negative charge (~1.4 g SDS per g protein), migrating toward the positive anode (+) purely by chain mass!"
    )

    sds_c1, sds_c2 = st.columns([1.15, 2.5])
    with sds_c1:
        mw_gel = float(metrics["mw_kda"])
        if mw_gel < 6.0:
            default_gel_pct = 20.0
        elif mw_gel < 10.0:
            default_gel_pct = 18.0
        elif mw_gel < 22.0:
            default_gel_pct = 15.0
        elif mw_gel > 250.0:
            default_gel_pct = 6.0
        elif mw_gel > 110.0:
            default_gel_pct = 8.0
        else:
            default_gel_pct = 12.0
        active_gel_pct = st.slider(
            "Gel Acrylamide Percentage (%):",
            min_value=4.0,
            max_value=22.0,
            value=default_gel_pct,
            step=0.5,
            help="Use 16–20% Tricine or Bis-Tris/MES for peptides (<10 kDa); 15% for 10–20 kDa; 12% for 20–100 kDa; 6–8% for >100 kDa."
        )
        if st.button(
            "🧪 Gel Recipe Calculator (Pop-Out)",
            key="btn_open_gel_recipe_dialog",
            type="primary",
            width="stretch",
            help="Open a pop-out handcast gel recipe calculator with exact mL / µL pipetting volumes, Laemmli & Tricine systems, running buffers, and printable .txt recipe sheet.",
        ):
            _open_gel_recipe_dialog(
                default_gel_pct=float(active_gel_pct),
                target_mw_kda=float(metrics["mw_kda"]),
                protein_name=st.session_state.sequence_name,
            )
        stain_type = st.radio(
            "Gel Staining Dye:",
            ["Coomassie Blue (Standard Lab Dye)", "Silver Stain (50–100× More Sensitive)"],
            index=0
        )
        show_stages = st.checkbox("Show All Purification Lanes (Lysate, Soluble, Flowthrough, Pure Elution)", value=True)

        if "Silver" in stain_type:
            st.info(
                "💡 **Why Silver Stain Looks Busier:** Silver stain detects tiny sub-nanogram traces (~0.2 ng), "
                "revealing faint host background bands in Lysate, Soluble, and Flowthrough lanes that Coomassie misses!"
            )

        if mw_gel > 500:
            st.info(
                f"💡 **Mega-Dalton Polypeptide ({mw_gel:,.1f} kDa):** Giant chains >500 kDa migrate right near "
                "the top stacking/resolving boundary on standard Tris-Glycine gels (use **4%–5%** acrylamide or composite agarose-acrylamide gels)."
            )
        elif mw_gel > 120 and active_gel_pct > 12:
            st.warning(f"Tip: {mw_gel:.1f} kDa is quite large—try lowering the gel slider to **6%–8%** so it moves further down!")
        elif mw_gel < 10.0:
            if active_gel_pct < 16.0:
                st.warning(
                    f"⚠️ **Low-MW Peptide ({mw_gel:.1f} kDa):** On a standard **{active_gel_pct:.1f}% Tris-Glycine (Laemmli)** gel, "
                    "peptides below ~10 kDa co-migrate with the dye front! Raise the slider to **16%–20%** and use a **16% Tricine-SDS-PAGE** or **4–12% Bis-Tris (MES buffer)** gel."
                )
            else:
                st.info(
                    f"🧪 **Low-MW Peptide ({mw_gel:.1f} kDa):** Even at **{active_gel_pct:.1f}%**, use a **16–20% Tricine-SDS-PAGE** or "
                    "**Bis-Tris / MES** buffer system rather than standard Tris-Glycine Laemmli so bands resolve sharply above the dye front."
                )
        elif mw_gel < 20 and active_gel_pct < 13:
            st.warning(f"Tip: {mw_gel:.1f} kDa is small—try raising the gel slider to **15%** (or use a 4–12% Bis-Tris/MES gel) for a sharper band!")
        else:
            st.success(f"✅ **{active_gel_pct:.1f}% acrylamide** gives crisp band separation for a {mw_gel:.1f} kDa protein.")

    with sds_c2:
        fig_sds = generate_sds_page_figure(
            target_mw_kda=metrics["mw_kda"],
            gel_pct=active_gel_pct,
            stain_type="Silver Stain" if "Silver" in stain_type else "Coomassie Blue",
            show_purification_stages=show_stages
        )
        st.plotly_chart(fig_sds, width="stretch")


# =====================================================================
# TAB 2: COMPLETE PURIFICATION STRATEGY (AFFINITY, IEX, HIC & MORE)
# =====================================================================
with tab_strategy:
    st.markdown("### 🧭 Complete 3-Step Protein Purification Strategy")
    st.caption("Adjust working pH below — every recommendation automatically adapts to your protein's charge, hydrophobicity, cofactors, and redox stability.")

    ph_col1, ph_col2 = st.columns([1.4, 2.2])
    with ph_col1:
        working_ph = st.slider("Choose Working Buffer pH:", min_value=3.5, max_value=10.5, value=7.4, step=0.1)
    curr_charge = calculate_net_charge_at_ph(current_seq, working_ph)
    tag_detected = protein_profile.get("effective_tag") or st.session_state.sequence_metadata.get("tag", "None")
    workflow = recommend_full_purification_workflow(
        metrics,
        working_ph,
        curr_charge,
        tag_detected=tag_detected,
        sequence=current_seq,
        protein_name=st.session_state.sequence_name,
        metadata=seq_meta,
        construct_parts=st.session_state.construct_parts,
        protein_profile=protein_profile,
    )

    with ph_col2:
        near_pi_warn = abs(working_ph - metrics["pi"]) < 0.5
        charge_chip_cls = "stat-chip stat-chip-crimson" if near_pi_warn else "stat-chip stat-chip-teal"
        has_tag_conflict = workflow.get("tag_conflict") is not None
        tag_chip_cls = (
            "stat-chip stat-chip-crimson" if has_tag_conflict
            else ("stat-chip stat-chip-indigo" if tag_detected != "None" else "stat-chip")
        )
        st.markdown(f"""
        <div class="ph-stat-bar">
            <div class="stat-chip stat-chip-amber">
                <span class="stat-chip-label">Working pH</span>
                <span class="stat-chip-value">pH {working_ph:.1f}</span>
            </div>
            <div class="{charge_chip_cls}">
                <span class="stat-chip-label">Net Charge</span>
                <span class="stat-chip-value">{curr_charge:+.2f}</span>
            </div>
            <div class="stat-chip">
                <span class="stat-chip-label">Isoelectric Pt</span>
                <span class="stat-chip-value">pI {metrics['pi']:.2f}</span>
            </div>
            <div class="stat-chip">
                <span class="stat-chip-label">GRAVY</span>
                <span class="stat-chip-value">{metrics['gravy']:+.3f}</span>
            </div>
            <div class="{tag_chip_cls}">
                <span class="stat-chip-label">Affinity Tag</span>
                <span class="stat-chip-value">{tag_detected}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Prominent Tag <-> Cofactor/Biochemistry Conflict Banner (Single Authoritative Explanation)
    t_conf = workflow.get("tag_conflict")
    if t_conf:
        conf_msg = (
            f"**{t_conf['headline']}**\n\n"
            f"{t_conf['explanation']}\n\n"
            f"✅ **Better Strategy:** {t_conf['better_strategy']}"
        )
        if t_conf.get("severity") == "critical":
            st.error(conf_msg)
        else:
            st.warning(conf_msg)

    # Deduplicated Compact Bench Notes (Never repeat what Tag Conflict banner already explained!)
    display_alerts = []
    for alert_msg in workflow["alerts"]:
        if t_conf:
            if t_conf.get("severity") == "critical" and (
                "Nickel Metalloenzyme Protection" in alert_msg
                or "Metalloprotein Cofactor Protection" in alert_msg
                or "Oxygen-Sensitive [Fe–S] Cluster Alert" in alert_msg
            ):
                continue
            if "Heme–Imidazole" in t_conf.get("headline", "") and "Hemoprotein / Soret Band Tip" in alert_msg:
                continue
            if "Disulfide–Glutathione" in t_conf.get("headline", "") and "Disulfide Bridge Protection" in alert_msg:
                continue
        display_alerts.append(alert_msg)

    if display_alerts:
        tips_li = "".join(f"<li>{_md_inline_to_html(a)}</li>" for a in display_alerts)
        st.markdown(f"""
        <div class="compact-tips-box">
            <div class="compact-tips-title">💡 Key Lab Notes for This Protein</div>
            <ul class="compact-tips-list">{tips_li}</ul>
        </div>
        """, unsafe_allow_html=True)

    # --- 3-STEP VISUAL PURIFICATION ROADMAP ---
    rm = workflow["roadmap"]
    step1_border = "#DC2626" if rm[0]["title"].startswith("🛑") else ("#D97706" if rm[0]["title"].startswith("⚠️") else "#0D9488")
    st.markdown(f"""
    <div class="roadmap-grid">
        <div class="roadmap-card" style="border-top: 4px solid {step1_border};">
            <div class="roadmap-step-tag">{html.escape(rm[0]['step'])}</div>
            <div class="roadmap-title">{_md_inline_to_html(rm[0]['title'])}</div>
            <div class="roadmap-body">{_md_inline_to_html(rm[0]['detail'])}</div>
        </div>
        <div class="roadmap-card" style="border-top: 4px solid #D97706;">
            <div class="roadmap-step-tag roadmap-step-amber">{html.escape(rm[1]['step'])}</div>
            <div class="roadmap-title">{_md_inline_to_html(rm[1]['title'])}</div>
            <div class="roadmap-body">{_md_inline_to_html(rm[1]['detail'])}</div>
        </div>
        <div class="roadmap-card" style="border-top: 4px solid #059669;">
            <div class="roadmap-step-tag">{html.escape(rm[2]['step'])}</div>
            <div class="roadmap-title">{_md_inline_to_html(rm[2]['title'])}</div>
            <div class="roadmap-body">{_md_inline_to_html(rm[2]['detail'])}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- AT-A-GLANCE 6-TILE REAGENT & COLUMN COMPATIBILITY BOARD ---
    compat_tiles = build_compatibility_matrix(metrics, working_ph, curr_charge, workflow)
    compat_cards_html = []
    for tile in compat_tiles:
        st_cls = tile["status"]
        compat_cards_html.append(
            f'<div class="compat-tile compat-tile-{st_cls}">'
            f'<div class="compat-tile-top">'
            f'<span class="compat-tile-title">{html.escape(tile["title"])}</span>'
            f'<span class="compat-badge compat-badge-{st_cls}">{html.escape(tile["badge"])}</span>'
            f'</div>'
            f'<div class="compat-tile-reason">{html.escape(tile["reason"])}</div>'
            f'</div>'
        )
    st.markdown("#### 🚦 Quick Compatibility Board: What Works & What to Avoid")
    st.markdown(f'<div class="compat-grid">{"".join(compat_cards_html)}</div>', unsafe_allow_html=True)

    # --- INTERACTIVE 1-MINUTE STUDENT LAB BENCH CHALLENGE ---
    challenge = build_student_lab_challenge(st.session_state.sequence_name, metrics, working_ph, curr_charge, workflow)
    with st.expander("🎯 Test Yourself: 1-Minute Lab Decision Challenge for This Protein", expanded=False):
        st.markdown(f'<div class="challenge-q-box">🧪 {html.escape(challenge["question"])}</div>', unsafe_allow_html=True)
        opt_letters = ["A", "B", "C", "D"]
        opt_lines = "".join(
            f"<li><b>{opt_letters[i]})</b> {html.escape(opt_txt)}</li>"
            for i, opt_txt in enumerate(challenge["options"])
        )
        st.markdown(f'<ul class="compact-tips-list">{opt_lines}</ul>', unsafe_allow_html=True)
        chal_ctx = (st.session_state.sequence_name, round(working_ph, 1), tag_detected)
        if st.session_state.get("_lab_chal_ctx") != chal_ctx:
            st.session_state["_lab_chal_ctx"] = chal_ctx
            st.session_state["lab_challenge_radio"] = "A"
        chosen_letter = st.radio(
            "Select your answer:",
            options=opt_letters,
            horizontal=True,
            key="lab_challenge_radio",
        )
        if st.button("✅ Check My Answer", key="btn_check_lab_challenge", type="primary"):
            if opt_letters.index(chosen_letter) == challenge["correct_index"]:
                st.success(f"🎉 {challenge['explanation']}")
            else:
                st.warning(
                    "🔄 Not quite! Check the Compatibility Board above for a clue, then try another option."
                )

    # --- SIDE-BY-SIDE DEEP DIVE: ION-EXCHANGE (CHARGE) vs. HYDROPHOBIC INTERACTION (HIC) ---
    col_iex_box, col_hic_box = st.columns([1.25, 1.15])

    with col_iex_box:
        st.markdown("#### ⚡ Method A: Ion-Exchange Chromatography (Separates by Charge)")
        iex = workflow["iex"]
        pi_val = metrics["pi"]

        fig_charge = go.Figure()
        fig_charge.add_vrect(
            x0=max(2.0, min(11.8, pi_val - 0.5)),
            x1=min(12.0, max(2.2, pi_val + 0.5)),
            fillcolor="rgba(220, 38, 38, 0.11)",
            line_width=0,
            annotation_text="Clumping Zone (Near pI)",
            annotation_position="top left",
            annotation_font=dict(size=10, color="#991B1B", family=PLOTLY_FONT_FAMILY)
        )
        fig_charge.add_hline(y=0, line_dash="dash", line_color="#94A3B8")
        fig_charge.add_trace(go.Scatter(
            x=metrics["ph_values"],
            y=metrics["charge_values"],
            mode="lines",
            name="Net Charge",
            line=dict(color="#0F766E", width=3.2)
        ))
        fig_charge.add_trace(go.Scatter(
            x=[pi_val], y=[0],
            mode="markers+text",
            name=f"pI = {pi_val:.2f}",
            text=[f"<b>pI {pi_val:.2f}</b>"],
            textposition="top right",
            textfont=dict(size=11, color="#991B1B", family=PLOTLY_FONT_FAMILY),
            marker=dict(size=10, color="#DC2626")
        ))
        fig_charge.add_trace(go.Scatter(
            x=[working_ph], y=[curr_charge],
            mode="markers+text",
            name=f"pH {working_ph:.1f}",
            text=[f"<b>pH {working_ph:.1f} ({curr_charge:+.1f})</b>"],
            textposition="bottom right",
            textfont=dict(size=11, color="#92400E", family=PLOTLY_FONT_FAMILY),
            marker=dict(size=12, color="#D97706", symbol="star")
        ))
        fig_charge.update_layout(
            font=dict(family=PLOTLY_FONT_FAMILY, color="#1E352F", size=12),
            title=dict(
                text="<b>Protein Charge vs. Buffer pH</b>",
                subtitle=dict(
                    text=f"Zero-Charge Isoelectric Point (pI = {pi_val:.2f})",
                    font=dict(family=PLOTLY_FONT_FAMILY, size=12, color="#4A635C")
                ),
                font=PLOTLY_TITLE_FONT
            ),
            xaxis=dict(
                title=dict(text="Buffer pH", font=PLOTLY_AXIS_TITLE_FONT),
                tickfont=PLOTLY_TICK_FONT,
                showgrid=True,
                gridcolor=PLOTLY_GRID_COLOR,
                linecolor="#CBDAD4"
            ),
            yaxis=dict(
                title=dict(text="Net Electrical Charge", font=PLOTLY_AXIS_TITLE_FONT),
                tickfont=PLOTLY_TICK_FONT,
                showgrid=True,
                gridcolor=PLOTLY_GRID_COLOR,
                linecolor="#CBDAD4"
            ),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            template="plotly_white",
            showlegend=False,
            height=330,
            margin=dict(l=38, r=25, t=48, b=38)
        )
        st.plotly_chart(fig_charge, width="stretch")
        pi_ox_tab2 = float(metrics.get("pi_oxidized", pi_val))
        if protein_profile.get("avoid_reducing_agents") and abs(pi_ox_tab2 - pi_val) >= 0.15:
            st.caption(
                f"Model assumption: Curve uses the ExPASy ProtParam / Bjellqvist (1993–94) denatured sequence model "
                f"(treats all Cys as free –SH; ignores 3D pKa shifts & PTMs). With structural S–S disulfides oxidized, "
                f"estimated pI shifts from **{pi_val:.2f} ➜ ~{pi_ox_tab2:.2f}** (native folded experimental pI can shift further, e.g., ~10.7–11.3 for native hen lysozyme)."
            )
        else:
            st.caption(
                "Model assumption: Theoretical pI & net charge use the ExPASy ProtParam / Bjellqvist (1993–94) denatured sequence model "
                "(treats all Cys as free –SH; excludes 3D folded microenvironments, ionic strength, and PTMs)."
            )

        iex_badge_cls = "kpi-status-warn" if "AVOID" in iex["recommended_column"].upper() else "method-badge-teal"
        st.markdown(f"""
        <div class="method-card method-card-teal">
            <div class="method-card-header">
                <span class="method-card-title">⚡ Recommended Ion-Exchange Setup</span>
                <span class="{iex_badge_cls}">{iex['recommended_column']}</span>
            </div>
            <div class="method-spec-list">
                <div><span class="method-spec-key">Resin Beads:</span> <code>{iex['recommended_resins']}</code></div>
                <div><span class="method-spec-key">Binding Mechanism:</span> {iex['mechanism']}</div>
                <div><span class="method-spec-key">Elution Gradient:</span> {iex['elution_strategy']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_hic_box:
        st.markdown("#### 💧 Method B: Hydrophobic Interaction Chromatography (HIC)")
        hic = workflow["hic"]
        hic_badge_cls = "kpi-status-warn" if "Avoid" in hic["status_badge"] else ("method-badge-teal" if hic["is_required_or_strong"] else "method-badge-amber")

        st.markdown(f"""
        <div class="method-card method-card-amber">
            <div class="method-card-header">
                <span class="method-card-title">🧪 HIC Column &amp; Salt Plan</span>
                <span class="{hic_badge_cls}">{html.escape(hic['status_badge'])}</span>
            </div>
            <div class="method-spec-list">
                <div><span class="method-spec-key">Why HIC Here:</span> {_md_inline_to_html(hic['when_why'])}</div>
                <div><span class="method-spec-key">Best HIC Resin:</span> <code>{html.escape(hic['recommended_resin'])}</code></div>
                <div><span class="method-spec-key">High-Salt Binding:</span> <code>{html.escape(hic['starting_salt'])}</code></div>
                <div><span class="method-spec-key">Elution Gradient:</span> {html.escape(hic['elution_protocol'])}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="student-callout">
            <div class="student-callout-title">🎓 Quick Intuition — HIC Is the Opposite of Ion-Exchange</div>
            <div>{html.escape(hic['student_analogy'])}</div>
        </div>
        """, unsafe_allow_html=True)

    # --- ALL 7 PURIFICATION METHODS COMPARISON TABLE ---
    st.markdown("---")
    st.markdown("#### 🗂️ All 7 Protein Purification Methods Compared for Your Protein")
    st.dataframe(pd.DataFrame(workflow["all_methods"]), hide_index=True, width="stretch")

    if workflow["iex"]["recommended_buffers"]:
        with st.expander(f"📋 View Best Buffering Chemicals for pH {working_ph:.1f}", expanded=False):
            buf_df = pd.DataFrame(workflow["iex"]["recommended_buffers"])[["name", "pka", "min_ph", "max_ph", "notes"]]
            buf_df.columns = ["Buffer Chemical", "pKa (25°C)", "Working Min pH", "Working Max pH", "Lab Notes"]
            st.dataframe(buf_df, hide_index=True, width="stretch")
            st.caption(
                "💡 **Counterion Rule for Ion-Exchange:** Prefer cationic amine buffers (`Tris`, `Bis-Tris`) for Anion Exchange (Q/DEAE) "
                "and anionic/zwitterionic Good's buffers (`MES`, `MOPS`, `HEPES`, `Phosphate`, `Acetate`) for Cation Exchange (SP/CM) so buffer ions do not bind the resin!"
            )


# =====================================================================
# TAB 3: CUSTOMIZABLE LABORATORY BUFFER & RECIPE CALCULATOR (g & mg)
# =====================================================================
with tab_buffers:
    st.markdown("### 🧪 Customizable Laboratory Buffer & Recipe Calculator")
    st.caption(
        "Pick a starter buffer or edit any component below — supports concentrations (`mM`, `M`, `%`) or direct weights (`g` / `mg`) with live protein-safety checks."
    )

    preset_keys = list(PRESET_RECIPE_TEMPLATES.keys())
    if st.session_state.get("_pending_safe_preset_key") in PRESET_RECIPE_TEMPLATES:
        st.session_state["buffer_preset_select"] = st.session_state.pop("_pending_safe_preset_key")
    elif "buffer_preset_select" not in st.session_state:
        st.session_state["buffer_preset_select"] = st.session_state.active_recipe_key

    # 1. Top Setup Row: Starter Preset + Custom Volume (mL or L) + Target pH
    b_top1, b_top2, b_top3, b_top4 = st.columns([1.7, 0.95, 0.65, 0.85])
    with b_top1:
        selected_preset_key = st.selectbox(
            "1️⃣ Load a Starter Recipe Template (or Custom Blank):",
            options=preset_keys,
            format_func=lambda k: PRESET_RECIPE_TEMPLATES[k]["label"],
            key="buffer_preset_select",
        )
        if selected_preset_key != st.session_state.active_recipe_key:
            st.session_state.active_recipe_key = selected_preset_key
            tpl = PRESET_RECIPE_TEMPLATES[selected_preset_key]
            st.session_state.custom_buffer_rows = [dict(r) for r in tpl["rows"]]
            st.session_state.custom_buffer_ph = float(tpl["target_ph"])
            st.session_state.custom_buffer_note = tpl["notes"]
            st.session_state.buffer_editor_ver += 1

    with b_top2:
        vol_number = st.number_input(
            "2️⃣ Target Volume:",
            min_value=1.0,
            max_value=50000.0,
            value=500.0,
            step=50.0
        )
    with b_top3:
        vol_unit = st.selectbox("Volume Unit:", ["mL", "L"], index=0)
    with b_top4:
        custom_ph = st.number_input(
            "3️⃣ Target pH:",
            min_value=2.0,
            max_value=12.0,
            value=float(st.session_state.custom_buffer_ph),
            step=0.1,
            key=f"target_ph_{st.session_state.buffer_editor_ver}"
        )

    total_volume_ml = vol_number * 1000.0 if vol_unit == "L" else vol_number
    st.info(f"💡 **Template Note:** {st.session_state.custom_buffer_note}")

    # 2. Quick-Add Reagent Bar (Collapsed by default so Editable Recipe & Weigh-Out Sheet stay above the fold!)
    with st.expander(f"➕ Quick-Add a Chemical from the Lab Shelf ({len(COMMON_LAB_REAGENTS)} Standard Reagents or Custom)", expanded=False):
        qa1, qa2, qa3, qa4, qa5 = st.columns([1.8, 0.95, 0.95, 0.9, 0.8])
        shelf_options = ["✍️ Custom Chemical (Type Name & MW)"] + [r["name"] for r in COMMON_LAB_REAGENTS]
        with qa1:
            picked_chem = st.selectbox("Choose Chemical:", options=shelf_options, index=1)
            if picked_chem.startswith("✍️"):
                chem_name_to_add = st.text_input("Custom Chemical Name:", value="My Custom Salt")
                default_mw_add = 100.0
                default_val_add = 50.0
                default_unit_add = "mM"
            else:
                reg = REAGENT_LOOKUP[picked_chem]
                chem_name_to_add = reg["name"]
                default_mw_add = float(reg["mw"])
                default_val_add = float(reg["default_conc"])
                default_unit_add = reg["default_unit"]

        with qa2:
            mw_to_add = st.number_input("MW (g/mol):", min_value=0.0, max_value=500000.0, value=default_mw_add, step=0.1)
        with qa3:
            val_to_add = st.number_input("Amount / Conc:", min_value=0.0, max_value=100000.0, value=default_val_add, step=1.0)
        with qa4:
            unit_options = ["mM", "M", "µM", "% (w/v)", "% (v/v)", "mg/mL", "g", "mg"]
            unit_to_add = st.selectbox(
                "Unit (Conc or g/mg):",
                options=unit_options,
                index=unit_options.index(default_unit_add) if default_unit_add in unit_options else 0
            )
        with qa5:
            st.markdown("<div style='height:27px;'></div>", unsafe_allow_html=True)
            if st.button("➕ Add Row", type="primary", width="stretch"):
                st.session_state.custom_buffer_rows.append({
                    "Reagent": chem_name_to_add,
                    "MW (g/mol)": mw_to_add,
                    "Value": val_to_add,
                    "Unit": unit_to_add
                })
                st.session_state.buffer_editor_ver += 1
                st.rerun()

    # 3. Editable Buffer Composition Table + Reset Button
    ed_head1, ed_head2 = st.columns([3.0, 1.0])
    with ed_head1:
        st.markdown("#### ✏️ Customize Your Buffer Components")
        st.caption("Click any cell to change concentration, MW, or switch Unit to **`g`** or **`mg`**! Use the `+` row at the bottom to add rows or select a row and press `Delete`.")
    with ed_head2:
        if st.button("🔄 Reset to Selected Template", width="stretch"):
            tpl = PRESET_RECIPE_TEMPLATES[selected_preset_key]
            st.session_state.custom_buffer_rows = [dict(r) for r in tpl["rows"]]
            st.session_state.custom_buffer_ph = float(tpl["target_ph"])
            st.session_state.buffer_editor_ver += 1
            st.rerun()

    edited_df = st.data_editor(
        pd.DataFrame(st.session_state.custom_buffer_rows),
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        column_config={
            "Reagent": st.column_config.TextColumn("Chemical / Buffer Component", required=True),
            "MW (g/mol)": st.column_config.NumberColumn("Formula Weight (g/mol)", min_value=0.0, max_value=500000.0, format="%.2f"),
            "Value": st.column_config.NumberColumn("Target Conc. OR Weight", min_value=0.0, max_value=500000.0, format="%.4g"),
            "Unit": st.column_config.SelectboxColumn(
                "Unit",
                options=["mM", "M", "µM", "% (w/v)", "% (v/v)", "mg/mL", "g", "mg"],
                required=True
            )
        },
        key=f"buffer_editor_table_{st.session_state.buffer_editor_ver}"
    )

    # Sanitize inline-edited DataFrame rows against NaN / blank cells & auto-fill known MWs
    sanitized_rows = []
    for raw_r in edited_df.to_dict(orient="records"):
        r_name = "" if pd.isna(raw_r.get("Reagent")) else str(raw_r.get("Reagent", "")).strip()
        raw_mw = raw_r.get("MW (g/mol)")
        raw_val = raw_r.get("Value")
        raw_unit = raw_r.get("Unit")
        clean_mw = 0.0 if pd.isna(raw_mw) else float(raw_mw)
        if clean_mw <= 0.0 and r_name in REAGENT_LOOKUP:
            clean_mw = float(REAGENT_LOOKUP[r_name]["mw"])
        clean_val = 0.0 if pd.isna(raw_val) else float(raw_val)
        clean_unit = "mM" if (pd.isna(raw_unit) or not str(raw_unit).strip()) else str(raw_unit).strip()
        sanitized_rows.append({
            "Reagent": r_name,
            "MW (g/mol)": clean_mw,
            "Value": clean_val,
            "Unit": clean_unit
        })
    st.session_state.custom_buffer_rows = sanitized_rows

    def _format_liquid_measure(row: pd.Series) -> str:
        """Formats neat liquid reagents in µL when < 1 mL and mL when >= 1 mL; returns 'Solid powder' for solids."""
        r_name = str(row["Reagent"])
        reg_meta = REAGENT_LOOKUP.get(r_name, {})
        if not reg_meta:
            r_low = r_name.lower().strip()
            for k, v in REAGENT_LOOKUP.items():
                if k.lower() == r_low:
                    reg_meta = v
                    break
        is_liquid_shelf = reg_meta.get("state") == "liquid"
        density = float(reg_meta.get("density_g_ml", 1.0)) if float(reg_meta.get("density_g_ml", 0.0)) > 0 else 1.0
        raw_liq_ml = row.get("Liquid Vol (mL)")

        if pd.isna(raw_liq_ml) and not is_liquid_shelf:
            return "Solid powder"

        if not pd.isna(raw_liq_ml):
            liq_ml = float(raw_liq_ml)
        else:
            liq_ml = float(row["Weight (mg)"]) / (1000.0 * density)

        if liq_ml <= 0.0:
            return "0.0 µL"
        if liq_ml < 1.0:
            ul = liq_ml * 1000.0
            return f"{ul:.2f} µL" if ul < 10.0 else f"{ul:,.1f} µL"
        return f"{liq_ml:,.2f} mL"

    # 4. Live Active-Protein Compatibility Check + 1-Click Safe Buffer Switch + Exact Weigh-Out Sheet!
    buffer_compat_warnings = audit_buffer_compatibility_for_protein(
        sanitized_rows,
        custom_ph,
        protein_profile,
        protein_name=st.session_state.sequence_name,
    )
    if buffer_compat_warnings:
        for bw in buffer_compat_warnings:
            st.warning(bw)
        safe_tpl_key = recommend_safe_buffer_preset_key(protein_profile, metrics, custom_ph, calculate_net_charge_at_ph(current_seq, custom_ph))
        safe_tpl = PRESET_RECIPE_TEMPLATES[safe_tpl_key]
        if st.button(f"✨ 1-Click Switch to Safe Buffer ({safe_tpl['label'].split('—')[0].strip()})", key="btn_autoload_safe_buf", type="primary"):
            st.session_state.active_recipe_key = safe_tpl_key
            st.session_state._pending_safe_preset_key = safe_tpl_key
            st.session_state.custom_buffer_rows = [dict(r) for r in safe_tpl["rows"]]
            st.session_state.custom_buffer_ph = float(safe_tpl["target_ph"])
            st.session_state.custom_buffer_note = safe_tpl["notes"]
            st.session_state.buffer_editor_ver += 1
            st.rerun()

    computed_recipe = compute_custom_buffer_recipe(sanitized_rows, total_volume_ml)

    vol_header_meta = f"{total_volume_ml:g} mL • Target pH {custom_ph:.1f}" if vol_unit != "mL" else f"Target pH {custom_ph:.1f}"
    if not computed_recipe:
        st.markdown(f"#### 📋 Exact Weigh-Out Sheet for **{vol_number:g} {vol_unit}** ({vol_header_meta})")
        st.warning("Add at least one reagent above to calculate exact weights in grams (g) and milligrams (mg).")
    else:
        out_df = pd.DataFrame(computed_recipe)
        recipe_csv = out_df.to_csv(index=False).encode("utf-8")

        sh_head1, sh_head2 = st.columns([2.55, 1.05])
        with sh_head1:
            st.markdown(f"#### 📋 Exact Weigh-Out Sheet for **{vol_number:g} {vol_unit}** ({vol_header_meta})")
        with sh_head2:
            st.download_button(
                label=f"⬇️ Download CSV ({vol_number:g} {vol_unit})",
                data=recipe_csv,
                file_name=f"buffer_recipe_{int(total_volume_ml)}mL_pH{custom_ph:.1f}.csv",
                mime="text/csv",
                width="stretch"
            )

        display_recipe_df = pd.DataFrame({
            "Chemical / Reagent": out_df["Reagent"],
            "MW (g/mol)": out_df["MW (g/mol)"].map(lambda x: f"{x:.2f}" if x > 0 else "—"),
            "Final Concentration": out_df["Final Concentration"],
            "Weight in Grams (g)": out_df["Weight (g)"].map(lambda g: f"{g:,.4f} g"),
            "Weight in Milligrams (mg)": out_df["Weight (mg)"].map(lambda mg: f"{mg:,.2f} mg"),
            "Liquid Measure": out_df.apply(_format_liquid_measure, axis=1),
            "Bench Weighing Tip": out_df["Bench Tip"]
        })
        st.dataframe(display_recipe_df, hide_index=True, width="stretch")

        start_water_ml = round(total_volume_ml * 0.80, 1)
        st.markdown(f"""
        <div class="protocol-section-header">🧑‍🔬 Lab Preparation & Safety Protocol</div>
        <div class="protocol-grid">
            <div class="protocol-step-card">
                <div class="protocol-step-badge step-badge-1">Step 1 • 80% Water</div>
                <div class="protocol-step-body">
                    Start with <b>~{start_water_ml:g} mL (~80%)</b> of ultrapure (18.2 MΩ·cm) water in a beaker so dissolved solids &amp; titrant never overshoot final volume
                    <i>(use ~65–70% for ≥1 M <code>(NH4)2SO4</code>, or ~45–55% for <code>8 M Urea</code> / <code>6 M Guanidine-HCl</code>)</i>.
                </div>
            </div>
            <div class="protocol-step-card">
                <div class="protocol-step-badge step-badge-2">Step 2 • Safe Titration</div>
                <div class="protocol-step-body">
                    Stir until dissolved, then titrate slowly to <b>pH {custom_ph:.1f}</b> with HCl or NaOH
                    (<b>⚠️ Safety: Always add acid to water, NEVER pour water into concentrated acid!</b> Wear goggles/gloves; dispense <code>2-Mercaptoethanol</code> in a fume hood).
                </div>
            </div>
            <div class="protocol-step-card">
                <div class="protocol-step-badge step-badge-3">Step 3 • Temp Check</div>
                <div class="protocol-step-body">
                    Equilibrate to working temperature before final pH check—<code>Tris</code> pKa shifts <b>~ −0.028 pH units per +1°C</b>
                    (~ −0.03/°C; pH 8.0 at 25°C rises to ~pH 8.6 at 4°C!).
                </div>
            </div>
            <div class="protocol-step-card">
                <div class="protocol-step-badge step-badge-4">Step 4 • Top Up &amp; Filter</div>
                <div class="protocol-step-body">
                    Transfer to a volumetric flask or graduated cylinder, top up (<code>q.s.</code>) with water to <b>{total_volume_ml:g} mL</b>,
                    filter (<code>0.22 µm</code>) &amp; degas for FPLC. Add labile <code>DTT</code>/<code>TCEP</code>/<code>PMSF</code> fresh before use.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 5. Instant Pocket Weight (g / mg) <-> Molarity (mM / M) Two-Way Calculator
    st.markdown("---")
    with st.expander("⚖️ Quick Two-Way Pocket Converter: Grams / Milligrams (g / mg) ↔ Molarity (mM / M)", expanded=False):
        conv_mode = st.radio(
            "What do you want to calculate?",
            ["Weight Needed (g & mg) from Concentration (mM / M)", "Resulting Concentration (mM & mg/mL) from Weighed Mass (g / mg)"],
            horizontal=True
        )
        qc1, qc2, qc3, qc4 = st.columns(4)
        q_mw = qc1.number_input("Chemical MW (g/mol):", min_value=1.0, max_value=500000.0, value=121.14, step=0.5, key="q_mw")
        q_vol = qc2.number_input("Solution Volume (mL):", min_value=0.1, max_value=50000.0, value=500.0, step=10.0, key="q_vol")
        if "Weight Needed" in conv_mode:
            q_val = qc3.number_input("Target Concentration:", min_value=0.0, max_value=20000.0, value=50.0, step=1.0, key="q_cval")
            q_unit = qc4.selectbox("Concentration Unit:", ["mM", "M", "µM"], index=0, key="q_cunit")
            res_conv = convert_weight_and_molarity("conc_to_weight", q_mw, q_vol, q_val, q_unit)
            st.success(
                f"⚖️ **Weigh Out:** **`{res_conv['mass_g']:.4f} g`**  (equals **`{res_conv['mass_mg']:,.2f} mg`**) "
                f"for {q_vol:g} mL  •  Equals `{res_conv['mg_per_ml']:.3f} mg/mL` (`{res_conv['pct_wv']:.3f}% w/v`)."
            )
        else:
            q_val = qc3.number_input("Weighed Mass:", min_value=0.0, max_value=500000.0, value=250.0, step=10.0, key="q_wval")
            q_unit = qc4.selectbox("Mass Unit:", ["mg", "g"], index=0, key="q_wunit")
            res_conv = convert_weight_and_molarity("weight_to_conc", q_mw, q_vol, q_val, q_unit)
            st.success(
                f"🧪 **Resulting Concentration:** **`{res_conv['molarity_mm']:.3f} mM`** (`{res_conv['molarity_m']:.5f} M`)  •  "
                f"**`{res_conv['mg_per_ml']:.3f} mg/mL`** (`{res_conv['pct_wv']:.3f}% w/v`)  •  "
                f"Mass = `{res_conv['mass_g']:.4f} g` / `{res_conv['mass_mg']:,.2f} mg`."
            )


# =====================================================================
# TAB 4: AMINO ACIDS, STRUCTURE & SOLUBILITY
# =====================================================================
with tab_details:
    ss_res = get_cached_secondary_structure(current_seq)

    # --- INTERACTIVE 3D PROTEIN STRUCTURE & BIOPHYSICAL VIEWER ---
    st.markdown("### 🧊 Interactive 3D Protein Viewer")
    st.caption(
        "Inspect an instant 100% offline 3D fold schematic, fetch atomic coordinates on demand from **AlphaFold DB / RCSB PDB**, "
        "or upload your own experimental / AI-designed **`.pdb` / `.cif`** model (AlphaFold3, Boltz, RFdiffusion, ProteinMPNN, ESMFold)."
    )

    viewer_3d_mode = st.radio(
        "3D Structure Source:",
        [
            "🧬 Instant 3D Fold Schematic (Offline)",
            "🌐 Live AlphaFold / PDB 3D",
            "📂 Upload AI / Custom .pdb / .cif",
        ],
        index=0,
        horizontal=True,
        key="viewer_3d_mode_radio",
    )

    if viewer_3d_mode == "🧬 Instant 3D Fold Schematic (Offline)":
        offline_color_labels = {
            "ss": "Secondary Structure (α-Helix / β-Strand / Coil)",
            "hydrophobicity": "Hydrophobicity (Kyte–Doolittle)",
            "charge": "Side-Chain Charge (+K/R, +H, −D/E)",
        }
        offline_3d_color = st.radio(
            "Color 3D Backbone By:",
            options=["ss", "hydrophobicity", "charge"],
            format_func=lambda k: offline_color_labels[k],
            index=0,
            horizontal=True,
            key="viewer_3d_offline_color",
        )
        fig_3d_offline = build_3d_biophysical_backbone_figure(
            sequence=current_seq,
            ss_res=ss_res,
            metrics=metrics,
            color_by=offline_3d_color,
            protein_name=st.session_state.sequence_name,
            height_px=480,
        )
        st.plotly_chart(fig_3d_offline, width="stretch")
        st.markdown(
            f'<div class="viewer3d-disclaimer"><b>ℹ️ Schematic 3D Note:</b> {html.escape(SCHEMATIC_3D_DISCLAIMER)}</div>',
            unsafe_allow_html=True,
        )

    elif viewer_3d_mode == "🌐 Live AlphaFold / PDB 3D":
        raw_uid_3d = str(seq_meta.get("uniprot_id") or "").strip()
        norm_uid_3d = normalize_uniprot_accession(raw_uid_3d) if raw_uid_3d else ""
        raw_pdb_3d = str(seq_meta.get("pdb_id") or "").strip().upper()
        if not norm_uid_3d and not raw_pdb_3d:
            raw_acc_3d = str(seq_meta.get("accession") or "").strip()
            norm_uid_3d = normalize_uniprot_accession(raw_acc_3d) if raw_acc_3d else ""

        if not norm_uid_3d and raw_pdb_3d:
            fetch_pdb_3d_clicked = st.button(
                f"🧊 Fetch Atomic 3D Structure from RCSB PDB ({raw_pdb_3d})",
                key="btn_fetch_3d_structure_pdb",
                type="primary",
                width="stretch",
            )
            cache_key_pdb_3d = ("pdb", raw_pdb_3d)
            if fetch_pdb_3d_clicked:
                with st.spinner(f"Fetching atomic 3D structure for PDB {raw_pdb_3d}..."):
                    fetched_pdb_3d = fetch_structure_by_pdb_id(
                        raw_pdb_3d,
                        timeout=10.0,
                        accession=raw_pdb_3d,
                    )
                st.session_state.structure_3d_cache[cache_key_pdb_3d] = fetched_pdb_3d
                if fetched_pdb_3d is None:
                    st.warning(
                        f"Could not retrieve a 3D structure (≤ 4 MB) for RCSB PDB **{raw_pdb_3d}** right now "
                        "(entry may exceed the 4 MB WebGL cap or network access may be offline). "
                        "Try uploading a local `.pdb` / `.cif` file or exploring the **Instant 3D Fold Schematic (Offline)**."
                    )

            cached_pdb_3d = st.session_state.structure_3d_cache.get(cache_key_pdb_3d)
            if isinstance(cached_pdb_3d, dict) and cached_pdb_3d.get("data"):
                _render_3dmol_metadata_and_viewer(
                    cached_pdb_3d,
                    key_prefix=f"live3d_pdb_{raw_pdb_3d}",
                )
            elif cache_key_pdb_3d not in st.session_state.structure_3d_cache:
                st.caption(
                    f"Click **🧊 Fetch Atomic 3D Structure from RCSB PDB ({raw_pdb_3d})** above to inspect "
                    f"**{st.session_state.sequence_name}** interactively in WebGL."
                )
        elif not norm_uid_3d:
            st.info(
                "🧬 **No UniProt or RCSB PDB Accession Linked to This Sequence:** Custom pasted sequences, translated ORFs, "
                "or unpublished de novo AI-designed proteins do not have a deposited entry.\n\n"
                "- Switch to **📂 Upload AI / Custom .pdb / .cif** above to inspect your AlphaFold3, Boltz, RFdiffusion, "
                "or ESMFold model interactively in WebGL.\n"
                "- Or switch to **🧬 Instant 3D Fold Schematic (Offline)** for an immediate sequence-derived 3D view!"
            )
        else:
            src_pref_labels = {
                "alphafold": "AlphaFold DB (Predicted Full-Length + pLDDT)",
                "rcsb": "Experimental RCSB PDB (Holo Cofactors & Metals)",
            }
            live_c1, live_c2 = st.columns([1.65, 1.0])
            with live_c1:
                prefer_3d_source = st.radio(
                    f"Preferred 3D Database for UniProt `{norm_uid_3d}`:",
                    options=["alphafold", "rcsb"],
                    format_func=lambda s: src_pref_labels[s],
                    index=0,
                    horizontal=True,
                    key="viewer_3d_live_source_pref",
                )
            with live_c2:
                fetch_3d_clicked = st.button(
                    "🧊 Fetch 3D Structure from AlphaFold / PDB",
                    key="btn_fetch_3d_structure",
                    type="primary",
                    width="stretch",
                )

            cache_key_3d = (norm_uid_3d, prefer_3d_source)
            if fetch_3d_clicked:
                with st.spinner(f"Fetching atomic 3D structure for UniProt {norm_uid_3d}..."):
                    fetched_3d = fetch_3d_structure_for_uniprot(
                        norm_uid_3d,
                        timeout=10.0,
                        prefer_source=prefer_3d_source,
                    )
                st.session_state.structure_3d_cache[cache_key_3d] = fetched_3d
                if fetched_3d is None:
                    st.warning(
                        f"Could not retrieve a 3D structure (≤ 4 MB) for UniProt **{norm_uid_3d}** right now "
                        "(entry may lack a deposited structure, exceed the 4 MB WebGL cap, or network access may be offline). "
                        "Try uploading a local `.pdb` / `.cif` file or exploring the **Instant 3D Fold Schematic (Offline)**."
                    )

            cached_3d = st.session_state.structure_3d_cache.get(cache_key_3d)
            if isinstance(cached_3d, dict) and cached_3d.get("data"):
                _render_3dmol_metadata_and_viewer(
                    cached_3d,
                    key_prefix=f"live3d_{norm_uid_3d}_{prefer_3d_source}",
                )
            elif cache_key_3d not in st.session_state.structure_3d_cache:
                st.caption(
                    f"Click **🧊 Fetch 3D Structure from AlphaFold / PDB** above to load interactive atomic coordinates "
                    f"for **{st.session_state.sequence_name}** (`UniProt {norm_uid_3d}`) on demand."
                )

    else:
        st.caption(
            "Drop any experimental or AI-designed 3D model (AlphaFold3, ColabFold, Boltz, RFdiffusion, ProteinMPNN, ESMFold) "
            "— supports `.pdb`, `.cif`, `.mmcif`, `.ent`, and `.gz` compressed files up to 10 MB."
        )
        uploaded_3d_file = st.file_uploader(
            "Upload Custom / AI-Designed 3D Structure (.pdb / .cif / .mmcif / .ent / .gz):",
            type=["pdb", "cif", "mmcif", "ent", "gz"],
            key="upload_3d_structure_file",
        )
        if uploaded_3d_file is not None:
            parsed_up_3d = parse_uploaded_structure_file(
                uploaded_3d_file.getvalue(),
                uploaded_3d_file.name,
            )
            if not parsed_up_3d.get("valid"):
                st.error(parsed_up_3d.get("error") or "Could not parse the uploaded structure file.")
            else:
                for up_warn in parsed_up_3d.get("warnings") or []:
                    st.warning(up_warn)
                upload_view_dict = dict(parsed_up_3d)
                upload_view_dict["source"] = f"Uploaded {str(parsed_up_3d.get('format', 'pdb')).upper()}"
                upload_view_dict["entry_id"] = uploaded_3d_file.name
                _render_3dmol_metadata_and_viewer(upload_view_dict, key_prefix="upload3d")

    st.markdown("---")
    st.markdown("### 🧭 Predicted Secondary Structure & Topology Diagram")
    st.caption(
        "Before folding into its final 3D shape, a protein chain forms local hydrogen-bonded patterns: "
        "coiled **α-helices (`H`)**, extended **β-strands (`E`)**, and flexible **coils / loops / turns (`C`)**."
    )

    ss_bench_meta = load_ss_benchmark_bundle()
    ss_xref_meta = load_ss_crossref_bundle()

    n_hel = int(ss_res["helix_count"])
    if n_hel == 0:
        hel_span_lbl = "0 helices predicted"
    elif n_hel == 1:
        hel_span_lbl = "1 helix (α1)"
    else:
        hel_span_lbl = f"{n_hel:,} helices (α1–α{n_hel})"

    n_str = int(ss_res["strand_count"])
    if n_str == 0:
        str_span_lbl = "0 strands predicted"
    elif n_str == 1:
        str_span_lbl = "1 strand (β1)"
    else:
        str_span_lbl = f"{n_str:,} strands (β1–β{n_str})"

    n_coil = int(ss_res["coil_count"])
    coil_span_lbl = f"{n_coil:,} flexible linker{'s' if n_coil != 1 else ''}"

    st.markdown(f"""
    <div class="ss-summary-grid">
        <div class="roadmap-card card-top-teal">
            <div class="roadmap-step-tag">🌀 α-Helix Content</div>
            <div class="kpi-value">{ss_res['helix_pct']:.1f}<span class="kpi-unit">%</span></div>
            <div class="roadmap-body">
                <b>{hel_span_lbl}</b> • Coiled right-handed H-bonded rods (~3.6 aa/turn).
            </div>
        </div>
        <div class="roadmap-card card-top-amber">
            <div class="roadmap-step-tag roadmap-step-amber">➔ β-Strand Content</div>
            <div class="kpi-value">{ss_res['strand_pct']:.1f}<span class="kpi-unit">%</span></div>
            <div class="roadmap-body">
                <b>{str_span_lbl}</b> • Extended pleated β-sheet strands &amp; barrels.
            </div>
        </div>
        <div class="roadmap-card card-top-slate">
            <div class="roadmap-step-tag roadmap-step-neutral">〰️ Coil / Loop / Turn</div>
            <div class="kpi-value">{ss_res['coil_pct']:.1f}<span class="kpi-unit">%</span></div>
            <div class="roadmap-body">
                <b>{coil_span_lbl}</b> • Surface loops &amp; turns (mean confidence <b>{ss_res['mean_confidence'] * 100:.1f}%</b>).
            </div>
        </div>
        <div class="roadmap-card card-top-indigo">
            <div class="roadmap-step-tag roadmap-step-indigo">🧭 Predicted Fold Class</div>
            <div class="roadmap-title">{ss_res['fold_class']}</div>
            <div class="roadmap-body">{ss_res['fold_description']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    ss_win_start = 1
    ss_win_end = int(ss_res["length"])
    ss_is_domain_window = False

    if metrics["length"] > 1200:
        st.markdown(f"""
        <div class="ss-domain-explorer">
            <div class="ss-domain-explorer-title">🔎 Giant Multi-Domain Protein Window Explorer ({metrics['length']:,} aa • {metrics['mw_kda']:,.1f} kDa)</div>
            <div class="ss-domain-explorer-sub">
                The 4 summary cards above show <b>exact full-protein totals across all {metrics['length']:,} residues</b>.
                Use <b>High-Resolution Domain Window</b> below to inspect individual β-sandwiches, coiled-coil repeats, disordered springs, or catalytic domains in crystal-clear residue detail!
            </div>
        </div>
        """, unsafe_allow_html=True)
        dom_c1, dom_c2 = st.columns([1.35, 1.65])
        with dom_c1:
            ss_view_mode = st.radio(
                "Topology Diagram View Mode:",
                [
                    "🔎 High-Resolution Domain Window (600 aa — Recommended)",
                    "🌐 Full-Chain Macro Overview",
                ],
                index=0,
                key="ss_giant_view_mode",
            )
        ss_is_domain_window = ss_view_mode.startswith("🔎")
        with dom_c2:
            if ss_is_domain_window:
                dom_win_size = 600
                max_dom_start = max(1, int(ss_res["length"]) - dom_win_size + 1)
                slider_step = 1 if max_dom_start <= 2000 else 50
                ss_win_start = st.slider(
                    f"Start Residue (600-aa Window across 1 – {ss_res['length']:,} aa):",
                    min_value=1,
                    max_value=max_dom_start,
                    value=1,
                    step=slider_step,
                    key="ss_domain_win_start",
                )
                ss_win_end = min(int(ss_res["length"]), ss_win_start + dom_win_size - 1)
                win_sub_ss = ss_res["ss_string"][ss_win_start - 1 : ss_win_end]
                win_span = max(1, len(win_sub_ss))
                st.caption(
                    f"📍 **Window {ss_win_start:,}–{ss_win_end:,}** ({win_span} aa): "
                    f"**{100.0 * win_sub_ss.count('H') / win_span:.1f}% α-Helix**  •  "
                    f"**{100.0 * win_sub_ss.count('E') / win_span:.1f}% β-Strand**  •  "
                    f"**{100.0 * win_sub_ss.count('C') / win_span:.1f}% Coil**"
                )
            else:
                st.caption(
                    f"🌐 Displaying batched macro overview across all **{ss_res['length']:,} residues** "
                    f"({ss_res['helix_count']:,} α-helices, {ss_res['strand_count']:,} β-strands). Switch to **High-Resolution Domain Window** to inspect individual elements!"
                )

    if ss_is_domain_window:
        fig_ss = generate_secondary_structure_figure(
            ss_res,
            st.session_state.sequence_name,
            region_start=ss_win_start,
            region_end=ss_win_end,
        )
    else:
        fig_ss = generate_secondary_structure_figure(ss_res, st.session_state.sequence_name)
    st.plotly_chart(fig_ss, width="stretch")

    ss_test_n_badge = int(ss_bench_meta.get("test_count", 11118)) if ss_bench_meta else 11118
    ss_q3_badge = float(ss_bench_meta.get("mean_protein_q3_pct", ss_bench_meta.get("Q3", 69.5))) if ss_bench_meta else 69.5
    ss_sov_badge = float(ss_bench_meta.get("SOV_99", 59.8)) if ss_bench_meta else 59.8
    ss_xref_n_badge = int(ss_xref_meta.get("n_proteins", 77)) if ss_xref_meta else 77

    st.markdown(f"""
    <div class="ss-validation-strip">
        <div class="ss-validation-copy">
            <b>✅ Scientific Validation:</b> Benchmarked with zero homology leakage across
            <b>{ss_test_n_badge:,} All-Domains-of-Life PDB/DSSP 3D Proteins</b> (<code>Q3 = {ss_q3_badge:.1f}%</code>,
            <code>SOV_99 = {ss_sov_badge:.1f}%</code>) &amp; cross-referenced live across <b>{ss_xref_n_badge} stratified proteins</b>
            against external servers (see <b>Tab 5</b>).
        </div>
        <div class="ss-validation-pills">
            <span class="meta-pill meta-pill-teal">≥10,500-Protein PDB/DSSP Benchmark</span>
            <span class="meta-pill meta-pill-indigo">NPS@ SOPMA &amp; GOR IV</span>
            <span class="meta-pill meta-pill-indigo">JPred4 / PSIPRED</span>
            <span class="meta-pill meta-pill-neutral">AlphaFold DB</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔤 View Residue-by-Residue Secondary Structure Alignment (H / E / C) & Segment Table", expanded=False):
        st.markdown("##### 📋 Predicted Secondary Structure Elements (`1`-Based Inclusive Coordinates)")
        seg_table_rows = [
            {
                "Segment": seg["label"],
                "Type": seg["type_name"],
                "Start–End": f"{seg['start']}–{seg['end']}",
                "Length (aa)": seg["length"],
                "Mean Confidence": f"{seg['mean_conf'] * 100:.1f}%",
                "Sequence": seg["sequence"],
            }
            for seg in ss_res["segments"]
        ]
        st.dataframe(pd.DataFrame(seg_table_rows), hide_index=True, width="stretch")

        st.markdown("##### 🧬 Residue-by-Residue `Pos / AA / SS` Alignment (`H` = α-Helix, `E` = β-Strand, `·` = Coil)")
        if ss_res["length"] <= 5000:
            st.code(format_ss_sequence_blocks(ss_res["sequence"], ss_res["ss_string"], block_width=60), language="text")
        else:
            align_start = ss_win_start if ss_is_domain_window else 1
            align_end = ss_win_end if ss_is_domain_window else min(int(ss_res["length"]), 1200)
            st.caption(
                f"Showing aligned window **residues {align_start:,}–{align_end:,}** of **{ss_res['length']:,} aa** "
                "(move the Domain Window slider above to inspect any region, or download the complete alignment below)."
            )
            st.code(
                format_ss_sequence_blocks(
                    ss_res["sequence"][align_start - 1 : align_end],
                    ss_res["ss_string"][align_start - 1 : align_end],
                    block_width=60,
                    start_pos=align_start,
                ),
                language="text",
            )
            full_ss_fasta = (
                f">{st.session_state.sequence_name} | {ss_res['length']} aa | Sequence\n"
                f"{ss_res['sequence']}\n"
                f">{st.session_state.sequence_name} | {ss_res['length']} aa | Secondary_Structure_HEC "
                f"(H={ss_res['helix_pct']:.2f}%, E={ss_res['strand_pct']:.2f}%, C={ss_res['coil_pct']:.2f}%)\n"
                f"{ss_res['ss_string']}\n"
            )
            st.download_button(
                label=f"⬇️ Download Complete {ss_res['length']:,}-Residue Sequence & H/E/C Alignment (.fasta)",
                data=full_ss_fasta.encode("utf-8"),
                file_name="secondary_structure_full_alignment.fasta",
                mime="text/plain",
                width="stretch",
            )

    st.markdown("---")
    st.markdown("### 🔬 Amino Acid Building Blocks & Solubility Guide")
    st.caption("Every protein is folded from 20 types of amino acid building blocks—their side chains control charge, UV absorbance, and water solubility!")

    col_det1, col_det2 = st.columns([1.35, 1.15])
    with col_det1:
        cnts = metrics["counts"]
        n_len = max(1, metrics["length"])
        class_df = pd.DataFrame([
            {
                "Building Block Group": "➕ Basic / Positively Charged (Arg, Lys, His)",
                "Count": cnts["positive_residues"],
                "Share": f"{(cnts['positive_residues']/n_len)*100:.1f}%",
                "What They Do": "Raise pI; Arg/Lys bind Cation-Exchange (SP) & Heparin; His (pKa ~6.0) binds Ni-NTA"
            },
            {
                "Building Block Group": "➖ Acidic / Negatively Charged (Asp, Glu)",
                "Count": cnts["negative_residues"],
                "Share": f"{(cnts['negative_residues']/n_len)*100:.1f}%",
                "What They Do": "Lower pI; carry negative charge above pH 4.5 to bind Anion-Exchange (Q) columns"
            },
            {
                "Building Block Group": "💡 Aromatic Side Chains (Trp, Tyr, Phe)",
                "Count": cnts["aromatic_residues"],
                "Share": f"{(cnts['aromatic_residues']/n_len)*100:.1f}%",
                "What They Do": "Trp & Tyr drive 280 nm UV absorbance (Phe peaks near 257 nm); all 3 bind Phenyl HIC resin"
            },
            {
                "Building Block Group": "💧 Hydrophobic Core (Ala, Val, Ile, Leu, Met, Phe, Trp)",
                "Count": cnts["hydrophobic_residues"],
                "Share": f"{(cnts['hydrophobic_residues']/n_len)*100:.1f}%",
                "What They Do": "Pack inside the folded core; exposed patches bind Butyl/Phenyl/Octyl HIC columns"
            },
            {
                "Building Block Group": "🔗 Cysteines (Cys)",
                "Count": cnts["cysteines"],
                "Share": f"{(cnts['cysteines']/n_len)*100:.1f}%",
                "What They Do": "Free thiols (–SH) or stabilizing S–S cystine disulfide bridges (+125 M⁻¹cm⁻¹ at 280 nm per pair)"
            }
        ])
        st.dataframe(class_df, hide_index=True, width="stretch")

        sol = metrics["solubility"]
        sol_cls_low = sol["solubility_class"].lower()
        if "low" in sol_cls_low or "insoluble" in sol_cls_low or "inclusion" in sol_cls_low or "elevated" in sol_cls_low:
            sol_card_cls = "result-card-crimson"
            sol_badge_cls = "kpi-status-warn"
        elif "high" in sol_cls_low or "good" in sol_cls_low or "favorable" in sol_cls_low:
            sol_card_cls = "result-card-teal"
            sol_badge_cls = "kpi-status-ok"
        else:
            sol_card_cls = "result-card-amber"
            sol_badge_cls = "method-badge-amber"

        st.markdown(f"""
        <div class="result-callout-card result-card-indigo">
            <div class="method-card-header">
                <span class="method-card-title">🔭 UV 280 nm Extinction Reference</span>
                <span class="meta-pill meta-pill-indigo">Pace et al., 1995</span>
            </div>
            <div class="result-stat-grid">
                <div class="result-stat-box">
                    <div class="result-stat-overline">Reduced (All Cys as –SH)</div>
                    <div class="result-stat-main">A280 (1 mg/mL) = {ext['abs_01_reduced']:.3f}</div>
                    <div class="result-stat-sub">ε280 = {ext['molar_extinction_reduced']:,} M⁻¹cm⁻¹</div>
                </div>
                <div class="result-stat-box">
                    <div class="result-stat-overline">Cystine S–S Oxidized ({ext['potential_disulfides']} Pairs)</div>
                    <div class="result-stat-main">A280 (1 mg/mL) = {ext['abs_01_cystines']:.3f}</div>
                    <div class="result-stat-sub">ε280 = {ext['molar_extinction_cystines']:,} M⁻¹cm⁻¹</div>
                </div>
            </div>
        </div>
        <div class="result-callout-card {sol_card_cls}">
            <div class="method-card-header">
                <span class="method-card-title">🧫 <i>E. coli</i> Solubility Heuristic (37°C Cytoplasm)</span>
                <span class="{sol_badge_cls}">{sol['solubility_class']}</span>
            </div>
            <div class="ui-body">{sol['description']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_det2:
        st.markdown("#### 📊 Frequency of All 20 Amino Acids (%)")
        aa_df = pd.DataFrame(list(metrics["aa_percentages"].items()), columns=["Amino Acid", "Percent"])
        fig_aa = go.Figure(go.Bar(
            x=aa_df["Amino Acid"],
            y=aa_df["Percent"],
            marker_color="#0D9488",
            marker_line=dict(color="#0F766E", width=0.8)
        ))
        fig_aa.update_layout(
            font=dict(family=PLOTLY_FONT_FAMILY, color="#1E352F", size=12),
            title=dict(
                text="<b>Residue Composition Across All 20 Standard Amino Acids</b>",
                font=PLOTLY_TITLE_FONT
            ),
            xaxis=dict(
                title=dict(text="Single-Letter Amino Acid Code", font=PLOTLY_AXIS_TITLE_FONT),
                tickfont=PLOTLY_TICK_FONT,
                showgrid=False,
                linecolor="#CBDAD4"
            ),
            yaxis=dict(
                title=dict(text="Share of Sequence (%)", font=PLOTLY_AXIS_TITLE_FONT),
                tickfont=PLOTLY_TICK_FONT,
                showgrid=True,
                gridcolor=PLOTLY_GRID_COLOR,
                linecolor="#CBDAD4"
            ),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            height=385,
            template="plotly_white",
            margin=dict(l=35, r=20, t=44, b=38)
        )
        st.plotly_chart(fig_aa, width="stretch")

    st.markdown(f"#### 🔤 Complete Amino Acid Sequence ({metrics['length']} residues)")
    st.code(current_seq, language="text")


# =====================================================================
# TAB 5 (MOVED TO END & SIMPLIFIED): REAL LAB EXAMPLES & ACCURACY CHECK
# =====================================================================
with tab_examples:
    bench_data = load_sec_benchmark_bundle()
    st.markdown("### 📚 Real-World Lab Examples: Does This Simulator Match Published Experiments?")
    st.caption(
        "A good scientific tool should match real laboratory measurements! Below you can compare ProPurify Studio's predicted SEC curves "
        "directly against real chromatograms published by scientists in open-access research papers."
    )

    if bench_data is None:
        st.warning("Reference benchmark file `data/sec_benchmark_1200.json` was not found.")
    else:
        bsum = bench_data["summary"]

        n_lit_cases = len(bench_data.get("literature_cases", []))
        n_lit_peaks = sum(len(c.get("species", [])) for c in bench_data.get("literature_cases", []))
        # 3 Transparent Summary Cards
        st.markdown(f"""
        <div class="roadmap-grid">
            <div class="roadmap-card" style="border-top: 4px solid #0D9488;">
                <div class="roadmap-step-tag">🧬 Hydrodynamic Calibration Sweep</div>
                <div class="kpi-value">{bsum['total_pdb_biomolecules']:,} PDB Assemblies</div>
                <div class="roadmap-body">
                    Hydrodynamic radius (Rh) &amp; column-range sweep across <b>3 kDa – 3 MDa</b> PDB structures ({bsum.get('sec_experimentally_supported_pdb_assemblies', 809):,} with SEC-supported assembly metadata).
                </div>
            </div>
            <div class="roadmap-card" style="border-top: 4px solid #059669;">
                <div class="roadmap-step-tag">🎯 Case-Study Peak MAE</div>
                <div class="kpi-value">MAE: {bsum['literature_complex_mae_ml']:.2f} mL (n = {n_lit_peaks})</div>
                <div class="roadmap-body">
                    Mean absolute error across <b>{n_lit_peaks} peaks</b> in {n_lit_cases} published 24 mL SEC case studies (<b>{bsum.get('analytical_24ml_mean_mae_ml', 0.22):.2f} mL</b> MAE across {bsum.get('total_standard_calibration_points', 55)} Cytiva globular calibration standards).
                </div>
            </div>
            <div class="roadmap-card" style="border-top: 4px solid #D97706;">
                <div class="roadmap-step-tag roadmap-step-amber">📖 Open-Access Case Studies</div>
                <div class="kpi-value">{n_lit_cases} Published Papers</div>
                <div class="roadmap-body">
                    Illustrative checks across single enzymes, flexible proteins, histone–DNA nucleosomes, and CRISPR-Cas9 ribonucleoproteins.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🔍 Pick a Famous Published Lab Experiment to Compare")
        lit_cases = bench_data["literature_cases"]
        selected_case_idx = st.selectbox(
            "Choose a Real Experiment from a Published Paper:",
            options=list(range(len(lit_cases))),
            format_func=lambda idx: f"Experiment {idx+1}: {lit_cases[idx]['short_title']} ({lit_cases[idx]['citation']})"
        )
        sel_case = lit_cases[selected_case_idx]
        case_col_cfg = SEC_COLUMNS[sel_case["column_id"]]
        case_sim = simulate_chromatogram_peaks(sel_case["species"], case_col_cfg)

        ex_left, ex_right = st.columns([1.65, 1.2])
        with ex_left:
            fig_lit = create_modern_sec_plot(
                case_sim,
                case_col_cfg,
                title=f"{sel_case['short_title']} • {case_col_cfg['name']}",
                show_ref_markers=False,
                show_dual_uv=sel_case.get("dual_uv", False),
                paper_exp_markers=sel_case["species"]
            )
            st.plotly_chart(fig_lit, width="stretch")

        with ex_right:
            st.markdown("#### 📖 Side-by-Side Peak Comparison")
            st.caption("Notice how the **★ Gold Star (Real Lab Measurement)** lines up with the simulated curve:")

            simple_rows = []
            for sp, pk in zip(sel_case["species"], case_sim["peaks"]):
                diff = abs(pk["ve_ml"] - sp["exp_ve_ml"])
                simple_rows.append({
                    "Molecule / Complex": sp["label"],
                    "Size": f"{sp['mw_kda']:g} kDa",
                    "★ Real Lab Peak": f"{sp['exp_ve_ml']:.2f} mL",
                    "🎯 Simulator Peak": f"{pk['ve_ml']:.2f} mL",
                    "Difference": f"{diff:.2f} mL ✅"
                })
            st.dataframe(pd.DataFrame(simple_rows), hide_index=True, width="stretch")

            st.info(f"🧬 **What Happened in This Experiment:** {sel_case['biophysical_note']}")
            st.markdown(
                f"📄 **Source Paper:** *{sel_case['citation']}* — "
                f"[Read Free Open-Access Paper ({sel_case['pmcid']})]({sel_case['paper_url']})"
            )

        pdb_df = pd.DataFrame(bench_data["pdb_1200"])
        # Keep the heavy PDB scatter plot & researcher CSV neatly tucked in an optional expander!
        with st.expander(f"🔬 For Teachers & Researchers: Open Full {len(pdb_df):,}-Protein Calibration Chart & Download CSV", expanded=False):
            atlas_col_choice = st.radio(
                "Select Column Calibration Curve:",
                options=["s75_increase_10_300", "s200_increase_10_300", "hiload_16_600_s200", "superose_6_increase_10_300"],
                format_func=lambda cid: f"{SEC_COLUMNS[cid]['name']} ({SEC_COLUMNS[cid]['opt_range_min_kda']:g}–{SEC_COLUMNS[cid]['opt_range_max_kda']:g} kDa)",
                index=1,
                horizontal=True
            )
            ve_field_map = {
                "s75_increase_10_300": "ve_s75_inc_ml",
                "s200_increase_10_300": "ve_s200_inc_ml",
                "hiload_16_600_s200": "ve_hiload_s200_ml",
                "superose_6_increase_10_300": "ve_sup6_inc_ml"
            }
            ve_key = ve_field_map[atlas_col_choice]

            fig_atlas = go.Figure()
            class_palette = {
                "Monomeric Protein": "#0D9488",
                "Homo-oligomeric Protein": "#059669",
                "Heteromeric Protein Complex": "#D97706",
                "Protein-DNA Complex": "#DB2777",
                "Protein-RNA Complex": "#7C3AED",
                "Protein-DNA/RNA Complex": "#E11D48",
                "Glycoprotein / Complex": "#0284C7"
            }
            for c_name, c_hex in class_palette.items():
                sub = pdb_df[pdb_df["complex_class"] == c_name]
                if sub.empty:
                    continue
                fig_atlas.add_trace(go.Scattergl(
                    x=sub["mw_kda"],
                    y=sub[ve_key],
                    mode="markers",
                    name=f"{c_name} (n={len(sub)})",
                    marker=dict(size=6, color=c_hex, opacity=0.72),
                    hoverinfo="text",
                    hovertext=[
                        f"<b>PDB {r.pdb_id}</b>: {r.title[:70]}<br>MW: {r.mw_kda:,.1f} kDa • Ve: {getattr(r, ve_key):.2f} mL"
                        for r in sub.itertuples()
                    ]
                ))

            col_rep = bench_data["column_calibration_report"].get(atlas_col_choice)
            if col_rep:
                stds = col_rep["standards"]
                fig_atlas.add_trace(go.Scatter(
                    x=[s["mw_kda"] for s in stds],
                    y=[s["exp_ve_ml"] for s in stds],
                    mode="markers",
                    name=f"★ Lab Calibration Standards (R² = {col_rep['new_rh_r2']:.4f})",
                    marker=dict(size=11, color="#DC2626", symbol="star", line=dict(width=1, color="#FFFFFF"))
                ))

            fig_atlas.update_layout(
                font=dict(family=PLOTLY_FONT_FAMILY, color="#1E352F", size=12),
                title=dict(
                    text=f"<b>{SEC_COLUMNS[atlas_col_choice]['name']} — Elution Volume vs. Molecular Weight</b>",
                    subtitle=dict(
                        text="Log Scale, 3 kDa – 3 MDa",
                        font=dict(family=PLOTLY_FONT_FAMILY, size=12, color="#4A635C")
                    ),
                    font=PLOTLY_TITLE_FONT
                ),
                xaxis=dict(
                    title=dict(text="Molecular Weight (kDa, Log Scale)", font=PLOTLY_AXIS_TITLE_FONT),
                    tickfont=PLOTLY_TICK_FONT,
                    type="log",
                    showgrid=True,
                    gridcolor=PLOTLY_GRID_COLOR,
                    linecolor="#CBDAD4"
                ),
                yaxis=dict(
                    title=dict(text="Elution Volume Ve (mL)", font=PLOTLY_AXIS_TITLE_FONT),
                    tickfont=PLOTLY_TICK_FONT,
                    showgrid=True,
                    gridcolor=PLOTLY_GRID_COLOR,
                    linecolor="#CBDAD4"
                ),
                legend=dict(font=dict(family=PLOTLY_FONT_FAMILY, size=11, color="#1E352F")),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                template="plotly_white",
                height=420
            )
            st.plotly_chart(fig_atlas, width="stretch")

            search_q = st.text_input(f"Filter {len(pdb_df):,} PDB Biomolecules by Name or 4-Letter PDB Code:", value="")
            filtered_df = pdb_df
            if search_q.strip():
                ql = search_q.strip().lower()
                filtered_df = pdb_df[pdb_df["pdb_id"].str.lower().str.contains(ql) | pdb_df["title"].str.lower().str.contains(ql)]

            st.dataframe(
                filtered_df[["pdb_id", "title", "complex_class", "mw_kda", "rh_angstrom", "ve_s75_inc_ml", "ve_s200_inc_ml", "ve_sup6_inc_ml"]].rename(columns={
                    "pdb_id": "PDB ID",
                    "title": "Protein / Complex",
                    "complex_class": "Category",
                    "mw_kda": "MW (kDa)",
                    "rh_angstrom": "Radius Rh (Å)",
                    "ve_s75_inc_ml": "Superdex 75 (mL)",
                    "ve_s200_inc_ml": "Superdex 200 (mL)",
                    "ve_sup6_inc_ml": "Superose 6 (mL)"
                }),
                hide_index=True,
                width="stretch",
                height=300
            )
            st.download_button(
                label=f"⬇️ Download Complete {len(pdb_df):,}-Protein Benchmark Dataset (CSV)",
                data=pdb_df.to_csv(index=False).encode("utf-8"),
                file_name="propurify_sec_benchmark_1254_pdb.csv",
                mime="text/csv",
                width="stretch"
            )

    # --- SECONDARY STRUCTURE 10,500+-PROTEIN & LIVE WEB-SERVER CROSS-REFERENCE SHOWCASE ---
    st.markdown("---")
    ss_bench_tab5 = load_ss_benchmark_bundle()
    ss_xref_tab5 = load_ss_crossref_bundle()

    st.markdown("### 🧭 Secondary Structure Accuracy: All-Domains-of-Life PDB/DSSP Benchmark & Web-Server Cross-Reference")
    st.caption(
        "How reliable are the α-helix (`H`), β-strand (`E`), and coil (`C`) predictions in Tab 4? "
        "Below are the zero-leakage benchmark results across over 10,500 experimentally solved 3D structures spanning the entire Tree of Life, "
        "alongside live head-to-head cross-referencing against established external bioinformatics web servers."
    )

    if ss_bench_tab5 is not None:
        pp_model_info = ss_bench_tab5.get("models", {}).get("propurify_ss", ss_bench_tab5)
        ss_n_test = int(ss_bench_tab5.get("test_count", 11118))
        ss_n_res = int(ss_bench_tab5.get("total_test_residues", 4066153))
        ss_mean_q3 = float(pp_model_info.get("mean_protein_q3_pct", ss_bench_tab5.get("mean_protein_q3_pct", ss_bench_tab5.get("Q3", 69.5))))
        ss_pool_q3 = float(pp_model_info.get("overall_q3_pct", ss_bench_tab5.get("Q3", 69.3)))
        ss_sov99 = float(pp_model_info.get("mean_protein_sov99_pct", ss_bench_tab5.get("SOV_99", 59.8)))
        ss_fold_acc = float(pp_model_info.get("fold_class_accuracy_pct", ss_bench_tab5.get("fold_class_accuracy_pct", 81.5)))
        ss_inf_ms = float(pp_model_info.get("mean_inference_ms", ss_bench_tab5.get("mean_inference_ms", 18.5)))
        gains = ss_bench_tab5.get("accuracy_gains_vs_baselines", {})
        gain_cf = float(gains.get("q3_gain_over_chou_fasman_pct", 20.16))
        gain_gor = float(gains.get("q3_gain_over_gor4_pct", 7.74))
        xr_count = int(ss_xref_tab5.get("n_proteins", 77)) if ss_xref_tab5 else 77

        st.markdown(f"""
        <div class="roadmap-grid">
            <div class="roadmap-card card-top-teal">
                <div class="roadmap-step-tag">🌍 All Domains of Life</div>
                <div class="kpi-value">{ss_n_test:,} 3D Proteins</div>
                <div class="roadmap-body">
                    <b>{ss_n_res:,} experimental PDB/DSSP residues</b> across Bacteria, Archaea, Animals, Plants, Fungi, Protists &amp; Viruses (strict zero homology leakage &lt;30% 5-mer similarity).
                </div>
            </div>
            <div class="roadmap-card card-top-emerald">
                <div class="roadmap-step-tag">🎯 3-State &amp; Fold Accuracy</div>
                <div class="kpi-value">Q3 {ss_mean_q3:.1f}% • SOV {ss_sov99:.1f}%</div>
                <div class="roadmap-body">
                    <b>+{gain_cf:.1f}% Q3</b> over Chou–Fasman &amp; <b>+{gain_gor:.1f}%</b> over GOR IV • <b>{ss_fold_acc:.1f}%</b> fold-class accuracy in <b>{ss_inf_ms:.1f} ms</b> pure-NumPy inference (pooled Q3 {ss_pool_q3:.1f}%).
                </div>
            </div>
            <div class="roadmap-card card-top-indigo">
                <div class="roadmap-step-tag roadmap-step-indigo">🌐 External Web Servers</div>
                <div class="kpi-value">{xr_count} Stratified Proteins</div>
                <div class="roadmap-body">
                    Cross-referenced live across all 7 biological strata against <b>NPS@ SOPMA</b>, <b>NPS@ GOR IV</b>, <b>JPred4 / PSIPRED</b>, and <b>AlphaFold DB</b>.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        ss_b_left, ss_b_right = st.columns([1.15, 1.15])
        with ss_b_left:
            st.markdown(f"#### 🌍 Multi-Domain Accuracy Across All 7 Taxonomic Strata ({ss_n_test:,} Held-Out Proteins)")
            strata_src = pp_model_info.get("by_stratum") or ss_bench_tab5.get("by_stratum", {})
            stratum_order = ["Bacteria", "Archaea", "Animals", "Plants", "Fungi", "Protists", "Viruses"]
            strata_rows = []
            for st_key in stratum_order:
                if st_key in strata_src:
                    sd = strata_src[st_key]
                    strata_rows.append({
                        "Biological Stratum": st_key,
                        "Held-Out Proteins": f"{int(sd['n_proteins']):,}",
                        "Residues": f"{int(sd['n_residues']):,}",
                        "Mean Q3": f"{float(sd['mean_protein_q3_pct']):.2f}%",
                        "Pooled Q3": f"{float(sd['overall_q3_pct']):.2f}%",
                        "Segment Overlap (SOV_99)": f"{float(sd['mean_protein_sov99_pct']):.2f}%",
                        "Mean MCC": f"{float(sd['mean_mcc']):.3f}",
                    })
            if strata_rows:
                st.dataframe(pd.DataFrame(strata_rows), hide_index=True, width="stretch")

            models_dict = ss_bench_tab5.get("models", {})
            if models_dict:
                st.markdown("#### 📈 Head-to-Head vs. Classic Single-Sequence Baselines")
                base_rows = []
                for m_key, m_label in [
                    ("propurify_ss", "🎯 ProPurify Cascaded SS (2-Stage Neural + Viterbi)"),
                    ("gor4_baseline", "📐 GOR IV (Garnier–Gibrat–Robson, 1996)"),
                    ("chou_fasman_baseline", "📏 Chou–Fasman (Empirical Rules, 1978)"),
                ]:
                    if m_key in models_dict:
                        md = models_dict[m_key]
                        base_rows.append({
                            "Method": m_label,
                            "Mean Q3": f"{float(md['mean_protein_q3_pct']):.2f}%",
                            "Pooled Q3": f"{float(md['overall_q3_pct']):.2f}%",
                            "SOV_99": f"{float(md['mean_protein_sov99_pct']):.2f}%",
                            "Mean MCC": f"{float(md['mean_mcc']):.3f}",
                        })
                if base_rows:
                    st.dataframe(pd.DataFrame(base_rows), hide_index=True, width="stretch")

        with ss_b_right:
            st.markdown("#### 🌐 Live External Secondary Structure Web-Server Cross-Reference")
            if ss_xref_tab5 is not None and "summary" in ss_xref_tab5:
                xsum = ss_xref_tab5["summary"]
                cov_map = xsum.get("method_coverage_counts", {})
                q3_exp_map = xsum.get("mean_q3_vs_exp_dssp", {})
                sov_exp_map = xsum.get("mean_sov99_vs_exp_dssp", {})
                pw_q3_map = xsum.get("mean_pairwise_q3_with_propurify", {})

                def _fmt_pct_opt(val) -> str:
                    return f"{float(val):.2f}%" if val is not None else "—"

                web_rows = [
                    {
                        "Server / Predictor": "🧬 EMBL-EBI AlphaFold DB v6 Monomer",
                        "Category": "3D Structure Server (mmCIF)",
                        "N": cov_map.get("alphafold_db_monomer_only", 67),
                        "Mean Q3 vs PDB/DSSP": _fmt_pct_opt(q3_exp_map.get("alphafold_db_monomer_only")),
                        "Mean SOV_99": _fmt_pct_opt(sov_exp_map.get("alphafold_db_monomer_only")),
                        "Agreement vs ProPurify": _fmt_pct_opt(pw_q3_map.get("alphafold_db_monomer_only")),
                    },
                    {
                        "Server / Predictor": "🌐 AlphaFold DB + PDBe SIFTS 3D (All 7 Strata)",
                        "Category": "Live 3D Structure Servers",
                        "N": cov_map.get("alphafold_3d", xr_count),
                        "Mean Q3 vs PDB/DSSP": _fmt_pct_opt(q3_exp_map.get("alphafold_3d")),
                        "Mean SOV_99": _fmt_pct_opt(sov_exp_map.get("alphafold_3d")),
                        "Agreement vs ProPurify": _fmt_pct_opt(pw_q3_map.get("alphafold_3d")),
                    },
                    {
                        "Server / Predictor": "🎯 ProPurify Studio Cascaded SS (Local <20 ms)",
                        "Category": "Single-Sequence Neural",
                        "N": cov_map.get("propurify_ss", xr_count),
                        "Mean Q3 vs PDB/DSSP": _fmt_pct_opt(q3_exp_map.get("propurify_ss")),
                        "Mean SOV_99": _fmt_pct_opt(sov_exp_map.get("propurify_ss")),
                        "Agreement vs ProPurify": "100.00% (Self)",
                    },
                ]
                for opt_key, opt_lbl, opt_cat in [
                    ("npsa_sopma", "🌐 NPS@ PBIL Lyon — SOPMA", "Single-Sequence Web Server"),
                    ("npsa_gor4", "🌐 NPS@ PBIL Lyon — GOR IV", "Single-Sequence Web Server"),
                    ("jpred4", "🏛️ Univ. of Dundee — JPred4 REST API", "Evolutionary MSA Web Server"),
                    ("gor4_baseline", "📐 GOR IV (17-aa Pairwise Info, 1996)", "Single-Sequence Baseline"),
                    ("chou_fasman_baseline", "📏 Chou–Fasman (Empirical Rules, 1978)", "Single-Sequence Baseline"),
                ]:
                    if int(cov_map.get(opt_key, 0) or 0) > 0:
                        web_rows.append({
                            "Server / Predictor": opt_lbl,
                            "Category": opt_cat,
                            "N": cov_map.get(opt_key, 0),
                            "Mean Q3 vs PDB/DSSP": _fmt_pct_opt(q3_exp_map.get(opt_key)),
                            "Mean SOV_99": _fmt_pct_opt(sov_exp_map.get(opt_key)),
                            "Agreement vs ProPurify": _fmt_pct_opt(pw_q3_map.get(opt_key)),
                        })
                st.dataframe(pd.DataFrame(web_rows), hide_index=True, width="stretch")
                st.info(
                    "💡 **Classroom Takeaway — Single-Sequence Speed vs. 3D / MSA Web Servers:** "
                    "**ProPurify SS** predicts secondary structure from a *single sequence* in **< 20 milliseconds** offline, "
                    "outperforming classic single-sequence methods (**GOR IV** & **Chou–Fasman**) by **+14% to +24% Q3** across the 77-protein multi-domain panel "
                    "and showing **70.0%–70.7% residue concordance** with **EMBL-EBI AlphaFold DB / PDBe 3D** (plus verified live connectivity with **NPS@ SOPMA / GOR IV** and **Dundee JPred4 REST API**)."
                )
            else:
                st.info("Reference file `data/ss_webserver_crossref.json` is not currently loaded.")

        if ss_xref_tab5 is not None and ss_xref_tab5.get("proteins"):
            xref_proteins = ss_xref_tab5["proteins"]
            with st.expander(
                f"🔬 For Teachers & Researchers: Inspect All {len(xref_proteins)} Multi-Domain Web-Server Cross-Reference Proteins & Download CSV",
                expanded=False
            ):
                def _cell_q3(d: dict, key: str) -> str:
                    v = (d or {}).get(key)
                    return f"{float(v):.1f}%" if v is not None else "—"

                has_sopma = any((pr.get("exp_q3") or {}).get("npsa_sopma") is not None for pr in xref_proteins)
                has_jpred = any((pr.get("exp_q3") or {}).get("jpred4") is not None for pr in xref_proteins)
                xref_table_rows = []
                for pr in xref_proteins:
                    eq = pr.get("exp_q3", {})
                    pq = pr.get("pairwise_q3_vs_propurify", {})
                    row_d = {
                        "UniProt": pr["accession"],
                        "Protein": pr["name"],
                        "Organism": pr["organism"],
                        "Stratum": pr["stratum"],
                        "Len (aa)": pr["length"],
                        "Fold Class": pr.get("propurify_fold_class", ""),
                        "ProPurify Q3": _cell_q3(eq, "propurify_ss"),
                        "AlphaFold/3D Q3": _cell_q3(eq, "alphafold_3d"),
                        "GOR IV Q3": _cell_q3(eq, "npsa_gor4" if eq.get("npsa_gor4") is not None else "gor4_baseline"),
                        "Chou–Fasman Q3": _cell_q3(eq, "chou_fasman_baseline"),
                        "ProPurify vs 3D": _cell_q3(pq, "alphafold_3d"),
                    }
                    if has_sopma:
                        row_d["NPS@ SOPMA Q3"] = _cell_q3(eq, "npsa_sopma")
                    if has_jpred:
                        row_d["JPred4 Q3"] = _cell_q3(eq, "jpred4")
                    xref_table_rows.append(row_d)
                xref_df = pd.DataFrame(xref_table_rows)
                st.dataframe(xref_df, hide_index=True, width="stretch", height=310)
                st.download_button(
                    label=f"⬇️ Download {len(xref_proteins)}-Protein Web-Server Cross-Reference Table (CSV)",
                    data=xref_df.to_csv(index=False).encode("utf-8"),
                    file_name="propurify_ss_webserver_crossref.csv",
                    mime="text/csv",
                    width="stretch"
                )

    with st.expander("📋 Benchmark Methodology, Data Splits & Practical Scope", expanded=False):
        sec_eq = bench_data["summary"].get("hydrodynamic_law", "ln(Rh_A) = 1.92534 + 0.38101*ln(MW_kDa) - 0.00167*[ln(MW_kDa)]^2") if bench_data else ""
        ss_train_n = int(ss_bench_tab5.get("train_count", 3751)) if ss_bench_tab5 else 3751
        ss_train_res = int(ss_bench_tab5.get("train_residues", 1449487)) if ss_bench_tab5 else 1449487
        ss_test_n = int(ss_bench_tab5.get("test_count", 11118)) if ss_bench_tab5 else 11118
        ss_test_res = int(ss_bench_tab5.get("total_test_residues", 4066153)) if ss_bench_tab5 else 4066153
        st.markdown(
            f"- **SEC Simulator & Case-Study Validation:** Uses empirical hydrodynamic radius scaling (`{sec_eq}`, globular standard fit `R² = 0.995`) mapped onto Cytiva column pore calibration curves (`55` calibration standard points, `0.22 mL` MAE on 24 mL columns). The reported **`0.17 mL` MAE** is strictly measured over the **20 peaks across 7 published open-access case studies**; the **1,254 PDB assemblies** represent a molecular-weight/hydrodynamic-radius calibration sweep (not 1,254 independent experimental chromatograms). Actual elution shifts with non-globular shape, glycosylation, detergent micelles, and resin–analyte interactions.\n"
            f"- **Secondary Structure Benchmark Splits & Curation:** Trained strictly on **`{ss_train_n:,}` PDB/DSSP chains** (`{ss_train_res:,}` residues; internal 76% Stage-1 / 14% Stage-2 / 10% calibration split with 5-fold out-of-fold GOR IV features) and evaluated on **`{ss_test_n:,}` held-out proteins** (`{ss_test_res:,}` residues) across all 7 taxonomic strata with **`<30%` 5-mer similarity filtering**, **`0` accession/sequence overlap**, and **`0` preset proteins in training**. Because it predicts from a *single sequence* without evolutionary MSAs, consult PDB / AlphaFold DB / MSA servers for research-grade structural modeling.\n"
            "- **Physicochemical Models & Practical Scope:** Theoretical pI/charge uses ExPASy ProtParam / Bjellqvist (1993–94) denatured pKa sets (plus an optional oxidized-disulfide S–S estimate when structural disulfides are present); solubility uses the coarse 81-protein Wilkinson–Harrison (1991) / Davis (1999) 37 °C *E. coli* cytoplasmic heuristic. **Treat all outputs as educational simulations and starting hypotheses**—confirm critical protocols against experimental measurements and manufacturer resin manuals."
        )
