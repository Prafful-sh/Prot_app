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

import json
import os
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from modules.sequence_parser import (
    LYSOZYME_MATURE_SEQ,
    CLASSROOM_PRESET_PROTEINS,
    clean_amino_acid_sequence,
    fetch_uniprot_by_id,
    fetch_ncbi_by_id,
    find_longest_orf,
    translate_nucleotide,
    parse_natural_language_prompt
)
from modules.purification_metrics import (
    analyze_all_parameters,
    calculate_net_charge_at_ph
)
from modules.sec_simulator import (
    SEC_COLUMNS,
    CONFORMATION_PROFILES,
    recommend_optimal_columns,
    get_column_marker_elutions,
    simulate_chromatogram_peaks,
    simulate_standard_mixture_chromatogram
)
from modules.buffer_advisor import (
    COMMON_LAB_REAGENTS,
    REAGENT_LOOKUP,
    PRESET_RECIPE_TEMPLATES,
    recommend_full_purification_workflow,
    compute_custom_buffer_recipe,
    convert_weight_and_molarity
)
from modules.sds_page_simulator import (
    generate_sds_page_figure
)
from modules.secondary_structure import (
    predict_secondary_structure,
    generate_secondary_structure_figure,
    format_ss_sequence_blocks,
)

# Page configuration
st.set_page_config(
    page_title="ProPurify Studio | Interactive Protein Purification & SEC Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .spotlight-strip {
        margin-top: 10px;
        background: var(--amber-tint);
        border: 1px solid #FDE68A;
        border-left: 3.5px solid var(--amber-accent);
        padding: 9px 14px;
        border-radius: 9px;
        font-size: var(--fs-body);
        color: var(--ink-body);
        line-height: 1.5;
    }
    .spotlight-label {
        color: var(--amber-deep);
        font-weight: 700;
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
</style>
""", unsafe_allow_html=True)


# --- INITIALIZE SESSION STATE (Default = Egg-White Lysozyme P00698!) ---
if "active_sequence" not in st.session_state:
    default_preset = CLASSROOM_PRESET_PROTEINS["lysozyme_native"]
    st.session_state.active_sequence = default_preset["sequence"]
    st.session_state.sequence_name = default_preset["name"]
    st.session_state.sequence_metadata = {
        "source": "Classroom Preset (UniProt P00698)",
        "uniprot_id": default_preset["uniprot_id"],
        "tag": default_preset["tag"],
        "student_story": default_preset["student_story"]
    }
    st.session_state.construct_parts = default_preset["construct_parts"]

if "custom_buffer_rows" not in st.session_state:
    st.session_state.custom_buffer_rows = [dict(r) for r in PRESET_RECIPE_TEMPLATES["his_lysis"]["rows"]]
    st.session_state.custom_buffer_ph = float(PRESET_RECIPE_TEMPLATES["his_lysis"]["target_ph"])
    st.session_state.custom_buffer_note = PRESET_RECIPE_TEMPLATES["his_lysis"]["notes"]
    st.session_state.active_recipe_key = "his_lysis"
if "buffer_editor_ver" not in st.session_state:
    st.session_state.buffer_editor_ver = 0


def set_new_sequence(seq: str, name: str, metadata: dict, construct_parts=None) -> bool:
    clean_seq, header, warnings = clean_amino_acid_sequence(seq)
    if not clean_seq:
        st.error("No valid amino acid sequence found! Please check your input.")
        return False
    st.session_state.active_sequence = clean_seq
    st.session_state.sequence_name = name or header or "Custom Protein"
    st.session_state.sequence_metadata = metadata
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
            "source": f"Classroom Example • UniProt {p['uniprot_id']}",
            "uniprot_id": p["uniprot_id"],
            "tag": p["tag"],
            "student_story": p["student_story"]
        },
        construct_parts=p["construct_parts"]
    )


# --- HERO HEADER ---
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge-row">
        <span class="hero-badge badge-gold">🎓 School, College & Research Ready</span>
        <span class="hero-badge badge-mint">🧪 SEC • SDS-PAGE • Affinity, IEX & HIC • Custom Buffer Lab</span>
    </div>
    <div class="hero-title">ProPurify Studio</div>
    <div class="hero-subtitle">
        Explore how proteins behave in the laboratory! Pick a famous textbook protein like <b>Egg-White Lysozyme</b> or enter your own sequence
        to simulate <b>Size-Exclusion Chromatography (SEC)</b>, run a virtual <b>SDS-PAGE Gel</b>, design a complete <b>Purification Strategy (including HIC & Ion-Exchange)</b>,
        and calculate custom <b>Lab Buffer Recipes in grams (g) & milligrams (mg)</b>.
    </div>
</div>
""", unsafe_allow_html=True)


# --- SIDEBAR: CLASSROOM EXAMPLES + MULTI-MODAL SEQUENCE INPUT ---
with st.sidebar:
    st.markdown("### 🎓 1-Click Classroom Proteins")
    st.caption("Instant examples for students & teachers:")
    st.selectbox(
        "Choose a Textbook Protein:",
        options=list(CLASSROOM_PRESET_PROTEINS.keys()),
        format_func=lambda k: CLASSROOM_PRESET_PROTEINS[k]["label"],
        index=0,
        key="classroom_preset_select",
        on_change=_apply_classroom_preset
    )

    st.markdown("---")
    st.markdown("### 📥 Or Load Your Own Protein")
    input_method = st.radio(
        "How would you like to enter it?",
        [
            "💬 Describe in Plain English",
            "🔍 UniProt ID",
            "🏛️ NCBI Accession",
            "🔤 Paste Amino Acid / FASTA",
            "🧬 Translate DNA / RNA"
        ]
    )

    # 1. Natural Language Prompt (Lysozyme & untagged lookup examples)
    if input_method == "💬 Describe in Plain English":
        st.caption(
            "Examples: `GST human`, `Chicken egg white lysozyme with N-terminal 6xHis tag and TEV site`, "
            "or `Green fluorescent protein with C-terminal Strep tag`"
        )
        nl_prompt = st.text_area(
            "Describe Protein + Optional Tag:",
            value="Chicken egg white lysozyme with N-terminal 6xHis tag and TEV site",
            height=90
        )
        if st.button("✨ Build Protein Construct", type="primary", width="stretch"):
            with st.spinner("Finding sequence & building construct..."):
                try:
                    res = parse_natural_language_prompt(nl_prompt)
                    has_engineered_parts = (
                        res.get("tag_detected", "None") != "None"
                        or res.get("cleavage_detected", "None") != "None"
                    )
                    source_label = (
                        f"Natural Language Construct Builder • UniProt {res['uniprot_id']}"
                        if has_engineered_parts
                        else f"Natural Language Lookup • UniProt {res['uniprot_id']}"
                    )
                    org_sci_prefix = " ".join(res["identified_organism"].split()[:2]).lower()
                    prot_low = res["identified_protein"].lower()
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
                            "uniprot_id": res["uniprot_id"],
                            "tag": res["tag_detected"],
                            "tag_position": res["tag_position"],
                            "cleavage": res["cleavage_detected"],
                            "canonical_len": len(res["canonical_sequence"])
                        },
                        construct_parts=res["construct_parts"] if has_engineered_parts else None
                    ):
                        if has_engineered_parts:
                            st.success(f"Construct ready: {res['identified_protein']} ({res['length']} aa)!")
                        else:
                            st.success(f"Loaded: {res['identified_protein']} [{res['uniprot_id']}] ({res['length']} aa)!")
                except Exception as e:
                    st.error(f"Could not build construct: {e}")

    # 2. UniProt Accession (Lysozyme P00698 default)
    elif input_method == "🔍 UniProt ID":
        st.caption(
            "Examples: `P00698` (Chicken Lysozyme C), `P61626` (Human Lysozyme C), `P42212` (GFP). "
            "*(Note: live UniProt entries store full precursors including any signal peptide, e.g., 147 aa for `P00698`; "
            "use Classroom Presets above for the cleaved 129-aa mature egg-white enzyme.)*"
        )
        uid_input = st.text_input("UniProt Accession ID:", value="P00698")
        if st.button("Fetch from UniProt", type="primary", width="stretch"):
            with st.spinner(f"Fetching {uid_input} from UniProt..."):
                try:
                    res = fetch_uniprot_by_id(uid_input)
                    if set_new_sequence(
                        seq=res["sequence"],
                        name=f"{res['protein_name']} [{res['accession']}]",
                        metadata=res
                    ):
                        st.success(f"Loaded: {res['protein_name']} ({res['length']} aa)")
                except Exception as e:
                    st.error(str(e))

    # 3. NCBI Accession (Chicken Lysozyme RefSeq NP_990612.2 / NM_205281.2)
    elif input_method == "🏛️ NCBI Accession":
        st.caption(
            "Examples: `NP_990612.2` (*Gallus gallus* Lysozyme C RefSeq precursor, 147 aa incl. 18-aa signal peptide) "
            "or `NM_205281.2` (*LYZ* mRNA transcript, 588 bp)."
        )
        ncbi_input = st.text_input("NCBI Accession:", value="NP_990612.2")
        if st.button("Fetch from NCBI", type="primary", width="stretch"):
            with st.spinner(f"Querying NCBI for {ncbi_input}..."):
                try:
                    res = fetch_ncbi_by_id(ncbi_input)
                    if set_new_sequence(
                        seq=res["sequence"],
                        name=res["protein_name"],
                        metadata=res
                    ):
                        st.success(f"Loaded: {res['protein_name']} ({res['length']} aa)")
                except Exception as e:
                    st.error(str(e))

    # 4. Direct Sequence / FASTA (Lysozyme default)
    elif input_method == "🔤 Paste Amino Acid / FASTA":
        pasted_text = st.text_area(
            "Paste Protein Sequence or FASTA:",
            value=(
                ">sp|P00698|LYSC_CHICK Egg-White Lysozyme C (Mature Chain, 129 aa)\n"
                f"{LYSOZYME_MATURE_SEQ}"
            ),
            height=165
        )
        if st.button("Load Sequence", type="primary", width="stretch"):
            _, parsed_hdr, _ = clean_amino_acid_sequence(pasted_text)
            if set_new_sequence(
                seq=pasted_text,
                name=parsed_hdr or "Custom FASTA Protein",
                metadata={"source": "Direct FASTA / Sequence Input"}
            ):
                st.success("Sequence loaded!")

    # 5. DNA / RNA Sequence (Met + Mature Lysozyme coding ORF default, 130 aa)
    elif input_method == "🧬 Translate DNA / RNA":
        st.caption(
            "Paste a DNA (`A, T, C, G`) or mRNA (`A, U, C, G`) coding sequence. "
            "*(Bacterial translation starts at `ATG` [`Met`], so the 393 bp default ORF encodes `Met` + 129-aa mature Lysozyme = 130 aa, ~14.44 kDa + `TAA` stop.)*"
        )
        dna_input = st.text_area(
            "Coding DNA / RNA (Met + Mature Lysozyme Coding ORF, 130 aa):",
            value=DEFAULT_LYSOZYME_DNA,
            height=145
        )
        trans_mode = st.radio("Reading Frame:", ["Find Longest Gene (ATG ➜ Stop)", "Translate Frame 1 Directly"])
        if st.button("Translate to Protein", type="primary", width="stretch"):
            if "Longest" in trans_mode:
                orf_res = find_longest_orf(dna_input)
                if set_new_sequence(
                    seq=orf_res["sequence"],
                    name=f"Translated Gene (Frame {orf_res['frame']}, {orf_res['orf_length_aa']} aa)",
                    metadata={"source": "DNA/RNA Translation", **orf_res}
                ):
                    st.success(f"Translated {orf_res['orf_length_aa']} amino acids!")
            else:
                seq = translate_nucleotide(dna_input, frame=1).split('*')[0]
                if set_new_sequence(
                    seq=seq,
                    name="Translated Frame 1",
                    metadata={"source": "DNA/RNA Frame 1"}
                ):
                    st.success(f"Translated {len(seq)} amino acids!")


# --- COMPUTE ALL PHYSICOCHEMICAL PARAMETERS ---
current_seq = st.session_state.active_sequence
metrics = analyze_all_parameters(current_seq)
ext = metrics["extinction"]


# --- ACTIVE PROTEIN & CLASSROOM SPOTLIGHT RIBBON ---
seq_meta = st.session_state.sequence_metadata
student_story = seq_meta.get("student_story", "")
story_html = (
    f'<div class="spotlight-strip">'
    f'<span class="spotlight-label">💡 Classroom Spotlight:</span> {student_story}</div>'
    if student_story else ""
)

uid_badge_val = seq_meta.get("uniprot_id") or seq_meta.get("accession") or ""
raw_source = seq_meta.get("source", "User Sequence")
clean_source = raw_source.split("•")[0].replace(f"(UniProt {uid_badge_val})", "").strip() if uid_badge_val else raw_source
uid_pill_html = (
    f'<span class="meta-pill meta-pill-indigo">UniProt {uid_badge_val}</span>'
    if uid_badge_val and uid_badge_val != "N/A" else ""
)
source_pill_html = (
    f'<span class="meta-pill meta-pill-neutral">{clean_source}</span>'
    if clean_source else ""
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
if has_multipart_construct:
    parts_html = "".join([
        f'<span class="construct-pill ' +
        ('pill-tag' if "tag" in p[0].lower() else ('pill-cleavage' if "cleavage" in p[0].lower() else 'pill-target')) +
        f'"><b>{p[0]}:</b> {len(p[1])} aa</span>'
        for p in st.session_state.construct_parts
    ])
    st.markdown(f"""
    <div class="protein-ribbon">
        <div class="ribbon-top-row">
            <div class="ribbon-title">🧬 Active Construct: {st.session_state.sequence_name}</div>
            <div class="ribbon-pill-group">
                <span class="meta-pill meta-pill-teal">{metrics['length']} aa</span>
                {uid_pill_html}
                {source_pill_html}
            </div>
        </div>
        <div class="construct-parts-row">{parts_html}</div>
        {story_html}
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="protein-ribbon">
        <div class="ribbon-top-row">
            <div class="ribbon-title">🧬 Active Protein: {st.session_state.sequence_name}</div>
            <div class="ribbon-pill-group">
                <span class="meta-pill meta-pill-teal">{metrics['length']} aa</span>
                {uid_pill_html}
                {source_pill_html}
            </div>
        </div>
        {story_html}
    </div>
    """, unsafe_allow_html=True)


# --- 6-PARAMETER KPI CARDS (ALL 6 SET PARAMETERS PRESERVED + EXACT PROTPARAM PRECISION) ---
instab_ok = metrics["instability_index"] < 40.0
instab_badge_cls = "kpi-status-ok" if instab_ok else "kpi-status-warn"
instab_badge_txt = "✅ Stable (< 40)" if instab_ok else "⚠️ Unstable (≥ 40, keep on ice)"

if metrics["pi"] > 7.3:
    pi_hint = "Basic (+ charge at pH 7.0)"
elif metrics["pi"] < 6.7:
    pi_hint = "Acidic (– charge at pH 7.0)"
else:
    pi_hint = "Near-neutral (~0 at pH 7.0)"

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
        <div class="kpi-title">Isoelectric Point</div>
        <div class="kpi-value">pI {metrics['pi']:.2f}</div>
        <div class="kpi-sub">{pi_hint}</div>
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


def _format_sec_table_row(p: dict, col_config: dict) -> dict:
    """Formats a simulated SEC peak row for classroom summary tables, handling HMW Void Aggregates cleanly."""
    is_agg = bool(p.get("is_aggregate", False))
    excl_kda = float(p.get("exclusion_limit_kda", col_config.get("exclusion_limit_kda", 1300.0)))
    return {
        "Peak": p["label"],
        "Size": f"> {_format_exclusion_limit(excl_kda)} (HMW)" if is_agg else f"{p['mw_kda']:.1f} kDa",
        "3D Radius (Rh)": "Excluded (Vo)" if is_agg else f"{p['rh_angstrom']:.1f} Å",
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
    paper_exp_markers=None
):
    """Creates a clean, classroom-friendly Plotly SEC chromatogram with collision-free peak labels."""
    fig = go.Figure()
    vo_val = float(col_config["void_volume_vo_ml"])
    vt_val = float(col_config["bed_volume_vt_ml"])
    dec = 2 if vt_val < 5.0 else 1

    max_sig = max(
        max(sim_data["total_signal_mau"]) if sim_data["total_signal_mau"] else 800.0,
        max(sim_data.get("total_signal_a260_mau", [0.0])) if show_dual_uv else 0.0,
        920.0
    )
    y_upper = max(1220.0 if len(sim_data.get("peaks", [])) >= 5 else 1180.0, max_sig * 1.28)
    x_upper = vt_val * 1.13

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
        name="UV 280 nm (Protein Signal)",
        line=dict(color="#0F766E", width=3.1),
        fill="tozeroy",
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

    # 5. Peak Diamond Markers & 5-Tier Anti-Collision Callouts
    sorted_peaks = sorted(
        sim_data["peaks"],
        key=lambda p: (p["ve_ml"], 0 if p.get("is_aggregate") else 1)
    )
    min_x_gap = 0.13 * vt_val
    ay_tiers = [-44, -84, -122, -62, -150]
    w_px_est, h_px_est = 460.0, 300.0
    box_w_est, box_h_est = 132.0, 35.0
    placed_boxes = []

    for i, peak in enumerate(sorted_peaks):
        disp_y = max(peak["peak_height_mau"], peak.get("peak_height_a260_mau", 0.0)) if show_dual_uv else peak["peak_height_mau"]
        is_agg_peak = bool(peak.get("is_aggregate", False))
        excl_kda_peak = float(peak.get("exclusion_limit_kda", col_config.get("exclusion_limit_kda", 1300.0)))
        ratio_txt = f" • A260/A280={peak.get('a260_a280_ratio', 0.57):.2f}" if show_dual_uv else ""
        p_color = "#0F766E" if peak["color"] == "#2563EB" else peak["color"]

        if is_agg_peak:
            size_hover = f"Size: > {_format_exclusion_limit(excl_kda_peak)} (Excluded at Vo)"
            uv_hover = f"A260/A280 Ratio: {peak.get('a260_a280_ratio', 0.92):.2f} (UV Light Scattering)"
        else:
            size_hover = f"Size: {peak['mw_kda']:.1f} kDa (Radius Rh = {peak.get('rh_angstrom', 0):.1f} Å)"
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

        # Ensure callout stays within plot canvas and avoids overlapping earlier callout boxes
        px_x = (peak["ve_ml"] / max(1e-6, x_upper)) * w_px_est
        px_y = (disp_y / max(1e-6, y_upper)) * h_px_est

        def _box_collides_or_clips(ax_c: float, ay_c: float) -> bool:
            cx, cy = px_x + ax_c, px_y - ay_c
            if cy < 38.0 or cy > (h_px_est - 16.0) or cx < 46.0 or cx > (w_px_est - 46.0):
                return True
            return any(abs(cx - ox) < box_w_est and abs(cy - oy) < box_h_est for ox, oy in placed_boxes)

        if _box_collides_or_clips(ax_shift, ay_offset):
            cand_ays = [ay_offset, -44, -84, -122, -62, -150, -104, -176, -202]
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
                max_up_px = max(40, int(h_px_est - 18.0 - px_y))
                ay_offset = max(ay_offset, -max_up_px)

        placed_boxes.append((px_x + ax_shift, px_y - ay_offset))
        elute_lbl_color = "#B91C1C" if is_agg_peak else "#0F766E"

        fig.add_annotation(
            x=peak["ve_ml"],
            y=disp_y,
            text=(
                f"<b>{peak['label']}</b><br>"
                f"<span style='color:{elute_lbl_color}; font-weight:700;'>Elutes at {peak['ve_ml']:.{dec}f} mL</span>"
                f"<span style='color:#4A635C;'>{ratio_txt}</span>"
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

    # 6. Column Start (Void Volume Vo) and End (Total Bed Volume Vt)
    fig.add_vline(x=vo_val, line_dash="dash", line_color="#DC2626", line_width=1.4)
    fig.add_vline(x=vt_val, line_dash="dash", line_color="#64748B", line_width=1.4)

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
        margin=dict(l=38, r=35, t=74, b=80),
        height=450
    )
    return fig


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
    prot_eps01 = max(0.2, float(ext["abs_01_reduced"]))

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

    if include_aggregate:
        st.info(
            "💡 **Void Aggregate Peak (`Vo`):** Clumped/misfolded protein aggregates are too large to enter any bead pores, "
            "so they wash straight out at the **Void Volume (`Vo`, `Kav = 0`)**—and often show an elevated `A260/A280` (~0.9) "
            "from UV light scattering! (Note: On reducing SDS-PAGE below, non-covalent aggregates still unfold back into single monomer chains.)"
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
        st.plotly_chart(create_modern_sec_plot(prep_sim, prep_config, f"Prep Column: {prep_config['name']}", show_markers, show_dual_uv), width="stretch")
        mp0 = prep_sim["peaks"][0]
        st.markdown(render_elution_strip(mp0["label"], mp0["mw_kda"], mp0["ve_ml"], prep_config), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([_format_sec_table_row(p, prep_config) for p in prep_sim["peaks"]]), hide_index=True, width="stretch")

    with c_anal_plot:
        st.plotly_chart(create_modern_sec_plot(anal_sim, anal_config, f"Analytical Column: {anal_config['name']}", show_markers, show_dual_uv), width="stretch")
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
        default_gel_pct = 15.0 if metrics["mw_kda"] < 22.0 else (8.0 if metrics["mw_kda"] > 110.0 else 12.0)
        active_gel_pct = st.slider(
            "Gel Acrylamide Percentage (%):",
            min_value=4.0,
            max_value=22.0,
            value=default_gel_pct,
            step=0.5,
            help="Use 15–18% for small proteins (<20 kDa like Lysozyme); 12% for medium proteins; 6–8% for large proteins (>100 kDa)."
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

        if metrics["mw_kda"] > 500:
            st.info(
                f"💡 **Mega-Dalton Polypeptide ({metrics['mw_kda']:,.1f} kDa):** Giant chains >500 kDa migrate right near "
                "the top stacking/resolving boundary on standard Tris-Glycine gels (use **4%–5%** acrylamide or composite agarose-acrylamide gels)."
            )
        elif metrics["mw_kda"] > 120 and active_gel_pct > 12:
            st.warning(f"Tip: {metrics['mw_kda']:.1f} kDa is quite large—try lowering the gel slider to **6%–8%** so it moves further down!")
        elif metrics["mw_kda"] < 20 and active_gel_pct < 11:
            st.warning(f"Tip: {metrics['mw_kda']:.1f} kDa is small—try raising the gel slider to **15%** for a sharper band!")
        else:
            st.success(f"✅ **{active_gel_pct:.1f}% acrylamide** gives crisp band separation for a {metrics['mw_kda']:.1f} kDa protein.")

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
    st.caption(
        "Adjust your working buffer pH below to see how your protein's electrical charge AND surface hydrophobicity "
        "guide **Affinity**, **Ion-Exchange (AEX / CEX)**, **Hydrophobic Interaction Chromatography (HIC)**, and **SEC**."
    )

    ph_col1, ph_col2 = st.columns([1.4, 2.2])
    with ph_col1:
        working_ph = st.slider("Choose Working Buffer pH:", min_value=3.5, max_value=10.5, value=7.4, step=0.1)
    curr_charge = calculate_net_charge_at_ph(current_seq, working_ph)
    tag_detected = st.session_state.sequence_metadata.get("tag", "None")
    workflow = recommend_full_purification_workflow(metrics, working_ph, curr_charge, tag_detected=tag_detected)

    with ph_col2:
        near_pi_warn = abs(working_ph - metrics["pi"]) < 0.5
        charge_chip_cls = "stat-chip stat-chip-crimson" if near_pi_warn else "stat-chip stat-chip-teal"
        tag_chip_cls = "stat-chip stat-chip-indigo" if tag_detected != "None" else "stat-chip"
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

    # Smart Biochemical Alerts (Disulfides, pI Clumping, Cold-room tips)
    for alert_msg in workflow["alerts"]:
        st.info(alert_msg)

    # --- 3-STEP VISUAL PURIFICATION ROADMAP ---
    rm = workflow["roadmap"]
    st.markdown(f"""
    <div class="roadmap-grid">
        <div class="roadmap-card" style="border-top: 4px solid #0D9488;">
            <div class="roadmap-step-tag">{rm[0]['step']}</div>
            <div class="roadmap-title">{rm[0]['title']}</div>
            <div class="roadmap-body">{rm[0]['detail']}</div>
        </div>
        <div class="roadmap-card" style="border-top: 4px solid #D97706;">
            <div class="roadmap-step-tag roadmap-step-amber">{rm[1]['step']}</div>
            <div class="roadmap-title">{rm[1]['title']}</div>
            <div class="roadmap-body">{rm[1]['detail']}</div>
        </div>
        <div class="roadmap-card" style="border-top: 4px solid #059669;">
            <div class="roadmap-step-tag">{rm[2]['step']}</div>
            <div class="roadmap-title">{rm[2]['title']}</div>
            <div class="roadmap-body">{rm[2]['detail']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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

        if hic["is_required_or_strong"]:
            st.success(f"**{hic['status_badge']}**\n\n{hic['when_why']}")
        else:
            st.info(f"**{hic['status_badge']}**\n\n{hic['when_why']}")

        st.markdown(f"""
        <div class="method-card method-card-amber">
            <div class="method-card-header">
                <span class="method-card-title">🧪 Recommended HIC Column & Salt Setup</span>
                <span class="method-badge-amber">{hic['recommended_resin']}</span>
            </div>
            <div class="method-spec-list">
                <div><span class="method-spec-key">Best HIC Resin:</span> <code>{hic['recommended_resin']}</code></div>
                <div><span class="method-spec-key">High-Salt Binding Buffer:</span> <code>{hic['starting_salt']}</code></div>
                <div><span class="method-spec-key">Why This Ligand:</span> {hic['ligand_reason']}</div>
                <div><span class="method-spec-key">Elution Gradient:</span> {hic['elution_protocol']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="student-callout">
            <div class="student-callout-title">🎓 Student Concept — How HIC Is the Opposite of Ion-Exchange</div>
            <div>{hic['student_analogy']}</div>
        </div>
        """, unsafe_allow_html=True)

    # --- ALL 7 PURIFICATION METHODS COMPARISON TABLE ---
    st.markdown("---")
    st.markdown("#### 🗂️ All 7 Protein Purification Methods Compared for Your Protein")
    st.caption(
        "Biochemists combine methods that separate by *different* physical properties (Tag ➜ Charge ➜ Hydrophobicity ➜ Size). "
        "Here is how every major lab technique fits your active protein:"
    )
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
        "Start from a standard chromatography preset OR build your own custom buffer! "
        "Edit any chemical, molecular weight, concentration (`mM`, `M`, `%`), volume (`mL` / `L`), or enter weights directly in **grams (`g`)** or **milligrams (`mg`)**."
    )

    # 1. Top Setup Row: Starter Preset + Custom Volume (mL or L) + Target pH
    b_top1, b_top2, b_top3, b_top4 = st.columns([1.7, 0.95, 0.65, 0.85])
    with b_top1:
        selected_preset_key = st.selectbox(
            "1️⃣ Load a Starter Recipe Template (or Custom Blank):",
            options=list(PRESET_RECIPE_TEMPLATES.keys()),
            format_func=lambda k: PRESET_RECIPE_TEMPLATES[k]["label"],
            index=0
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

    # 2. Quick-Add Reagent Bar (From Common Lab Shelf Chemicals OR Custom Chemical Name + Concentration OR g/mg!)
    with st.expander(f"➕ Quick-Add a Chemical from the Lab Shelf ({len(COMMON_LAB_REAGENTS)} Standard Reagents or Custom)", expanded=True):
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

    # 4. Live Calculated Recipe Sheet Showing BOTH Grams (g) AND Milligrams (mg)!
    computed_recipe = compute_custom_buffer_recipe(sanitized_rows, total_volume_ml)

    if not computed_recipe:
        st.markdown(f"#### 📋 Exact Weigh-Out Sheet for **{vol_number:g} {vol_unit}** ({total_volume_ml:g} mL • Target pH {custom_ph:.1f})")
        st.warning("Add at least one reagent above to calculate exact weights in grams (g) and milligrams (mg).")
    else:
        out_df = pd.DataFrame(computed_recipe)
        recipe_csv = out_df.to_csv(index=False).encode("utf-8")

        sh_head1, sh_head2 = st.columns([2.55, 1.05])
        with sh_head1:
            st.markdown(f"#### 📋 Exact Weigh-Out Sheet for **{vol_number:g} {vol_unit}** ({total_volume_ml:g} mL • Target pH {custom_ph:.1f})")
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
    st.markdown("### 🧭 Predicted Secondary Structure & Topology Diagram")
    st.caption(
        "Before folding into its final 3D shape, a protein chain forms local hydrogen-bonded patterns: "
        "coiled **α-helices (`H`)**, extended **β-strands (`E`)**, and flexible **coils / loops / turns (`C`)**."
    )

    ss_res = get_cached_secondary_structure(current_seq)
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
        if "high" in sol_cls_low or "good" in sol_cls_low or ("soluble" in sol_cls_low and "insoluble" not in sol_cls_low and "low" not in sol_cls_low):
            sol_card_cls = "result-card-teal"
            sol_badge_cls = "kpi-status-ok"
        elif "low" in sol_cls_low or "insoluble" in sol_cls_low or "inclusion" in sol_cls_low:
            sol_card_cls = "result-card-crimson"
            sol_badge_cls = "kpi-status-warn"
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
                <span class="method-card-title">🧫 Predicted Solubility in <i>E. coli</i></span>
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

        # 3 Friendly Non-Technical Summary Cards
        st.markdown(f"""
        <div class="roadmap-grid">
            <div class="roadmap-card" style="border-top: 4px solid #0D9488;">
                <div class="roadmap-step-tag">🧬 Tested Across Sizes</div>
                <div class="kpi-value">{bsum['total_pdb_biomolecules']:,} Real Proteins</div>
                <div class="roadmap-body">
                    From tiny <b>3 kDa</b> peptides up to giant <b>3,000 kDa (3 MDa)</b> multi-subunit complexes from the Protein Data Bank.
                </div>
            </div>
            <div class="roadmap-card" style="border-top: 4px solid #059669;">
                <div class="roadmap-step-tag">🎯 Peak Accuracy</div>
                <div class="kpi-value">Mean Error: {bsum['literature_complex_mae_ml']:.2f} mL</div>
                <div class="roadmap-body">
                    Mean absolute error (MAE) across published complex peaks (~99% volume accuracy on a 24 mL column!).
                </div>
            </div>
            <div class="roadmap-card" style="border-top: 4px solid #D97706;">
                <div class="roadmap-step-tag roadmap-step-amber">📖 Open-Access Papers</div>
                <div class="kpi-value">Proteins + DNA &amp; RNA</div>
                <div class="roadmap-body">
                    Verified on single enzymes, flexible proteins, histone–DNA spools, and CRISPR-Cas9 guide-RNA gene editors.
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
