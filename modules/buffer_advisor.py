"""
Buffer & Comprehensive Chromatography Strategy Advisor Module
Recommends & Calculates:
- Complete 3-Step Purification Roadmaps (Capture -> Intermediate -> Polishing)
- Evaluation across ALL 7 major protein purification methods:
  1. Affinity Chromatography (IMAC Ni-NTA/TALON, Strep-Tactin / Strep-Tactin XT, GST, MBP, FLAG)
  2. Pseudo-Affinity / Bio-Affinity (Heparin Sepharose, Cibacron Blue) for tagless proteins
  3. Ammonium Sulfate (NH4)2SO4 Salting-Out Precipitation (couples directly into HIC!)
  4. Ion-Exchange Chromatography (AEX [Q/DEAE] vs. CEX [SP/CM])
  5. Hydrophobic Interaction Chromatography (HIC — Butyl-S / Butyl / Phenyl / Octyl Sepharose)
  6. Hydroxyapatite & Multimodal Chromatography (CHT Ca10(PO4)6(OH)2 / Capto MMC)
  7. Size-Exclusion Chromatography (SEC — Superdex 75 / Superdex 200 / Superose 6)
- Fully Customizable Laboratory Buffer & Recipe Calculator supporting custom compositions,
  concentrations (mM, M, µM, % w/v, % v/v, mg/mL), volumes (mL / L), neat liquid dispensing (µL / mL),
  solubility/pH bench warnings, and weights in BOTH grams (g) and milligrams (mg).
"""

import re
from typing import Dict, Any, List, Optional, Tuple

from modules.purification_metrics import KYTE_DOOLITTLE
from modules.sequence_parser import detect_sequence_terminal_tag

# Authoritative Biological & Chromatographic Buffer Systems (25°C reference constants;
# IUPAC/PubChem formula weights with explicit hydration & protonation state).
BUFFER_SYSTEMS: List[Dict[str, Any]] = [
    {
        "name": "Sodium Acetate (Anhydrous)",
        "formula": "CH3COONa",
        "pka": 4.76,
        "min_ph": 3.7,
        "max_ph": 5.6,
        "mw": 82.03,
        "type": "Carboxylate (Anionic)",
        "notes": "Ideal for acidic Cation Exchange (CEX); avoid in AEX. (Note: Trihydrate MW = 136.08 g/mol)."
    },
    {
        "name": "Trisodium Citrate·2H2O",
        "formula": "Na3C6H5O7·2H2O",
        "pka": 6.40,
        "min_ph": 3.0,
        "max_ph": 6.2,
        "mw": 294.10,
        "type": "Tricarboxylate (Anionic)",
        "notes": "Polyprotic (pKa1=3.13, pKa2=4.76, pKa3=6.40; effective pH 3.0–6.2); great for acidic CEX, but chelates Ni2+/Ca2+ (avoid on IMAC & CHT)."
    },
    {
        "name": "MES (Free Acid, Anhydrous)",
        "formula": "C6H13NO4S",
        "pka": 6.15,
        "min_ph": 5.5,
        "max_ph": 6.7,
        "mw": 195.24,
        "type": "Zwitterionic (Good's)",
        "notes": "Ideal for Cation Exchange (CEX) and metalloproteins; negligible metal chelation. (Monohydrate MW = 213.25 g/mol)."
    },
    {
        "name": "Bis-Tris (Free Base, Anhydrous)",
        "formula": "C8H19NO5",
        "pka": 6.46,
        "min_ph": 5.8,
        "max_ph": 7.2,
        "mw": 209.24,
        "type": "Tertiary Amine (Cationic)",
        "notes": "Excellent cationic buffer for Anion Exchange (AEX) at pH 5.8–7.2; avoid on CEX and high-stringency IMAC."
    },
    {
        "name": "PIPES (Free Acid, Anhydrous)",
        "formula": "C8H18N2O6S2",
        "pka": 6.76,
        "min_ph": 6.1,
        "max_ph": 7.5,
        "mw": 302.37,
        "type": "Zwitterionic (Good's)",
        "notes": "Minimal metal-ion binding; great for CEX and SEC. Titrate free acid with NaOH to pH ~6.5 to dissolve."
    },
    {
        "name": "Sodium Phosphate (Na2HPO4, Anhydrous)",
        "formula": "Na2HPO4",
        "pka": 7.20,
        "min_ph": 5.8,
        "max_ph": 8.0,
        "mw": 141.96,
        "type": "Inorganic Orthophosphate",
        "notes": "pKa2 = 7.20 at I=0 (~6.86 at 0.15 M salt). Classic for HIC, CEX & IMAC; avoid in AEX (competes with Q) and precipitates Ca2+/Mg2+."
    },
    {
        "name": "MOPS (Free Acid, Anhydrous)",
        "formula": "C7H15NO4S",
        "pka": 7.20,
        "min_ph": 6.5,
        "max_ph": 7.9,
        "mw": 209.26,
        "type": "Zwitterionic (Good's)",
        "notes": "Negligible metal binding; excellent for CEX, SEC, and IMAC (do not autoclave with glucose)."
    },
    {
        "name": "HEPES (Free Acid, Anhydrous)",
        "formula": "C8H18N2O4S",
        "pka": 7.55,
        "min_ph": 6.8,
        "max_ph": 8.2,
        "mw": 238.30,
        "type": "Zwitterionic (Good's)",
        "notes": "Gold standard for structural biology, SEC, CEX & IMAC (apparent pKa ~7.48 at 25°C, I=0.1 M; Na-salt MW = 260.29 g/mol)."
    },
    {
        "name": "Tris-Base (Anhydrous Free Base)",
        "formula": "C4H11NO3",
        "pka": 8.06,
        "min_ph": 7.0,
        "max_ph": 9.0,
        "mw": 121.14,
        "type": "Primary Amine (Cationic)",
        "notes": "Workhorse for IMAC, AEX & SEC (avoid in CEX & NHS-ester labeling; dpKa/dT ≈ -0.028/°C so pH rises ~+0.6 cooled to 4°C; Tris-HCl MW = 157.60)."
    },
    {
        "name": "Bicine (Anhydrous)",
        "formula": "C6H13NO4",
        "pka": 8.26,
        "min_ph": 7.6,
        "max_ph": 9.0,
        "mw": 163.17,
        "type": "Zwitterionic (Good's)",
        "notes": "Useful alkaline buffer for crystallization and biochemical assays (weakly chelates divalent metals; keep ≤20 mM on Ni-NTA)."
    },
    {
        "name": "CHES (Free Acid, Anhydrous)",
        "formula": "C8H17NO3S",
        "pka": 9.30,
        "min_ph": 8.6,
        "max_ph": 10.0,
        "mw": 207.29,
        "type": "Zwitterionic",
        "notes": "High-pH buffer when purifying strongly basic proteins (high pI) or assaying alkaline enzymes."
    },
    {
        "name": "CAPS (Free Acid, Anhydrous)",
        "formula": "C9H19NO3S",
        "pka": 10.40,
        "min_ph": 9.7,
        "max_ph": 11.1,
        "mw": 221.32,
        "type": "Zwitterionic",
        "notes": "Extreme alkaline buffer (pH 9.7–11.1) for high-pI electroblotting and alkaline chromatography."
    }
]


# Comprehensive Lab Shelf Catalog for Customizable Buffer Calculator (47 verified laboratory reagents)
COMMON_LAB_REAGENTS: List[Dict[str, Any]] = [
    # Buffers (21)
    {"name": "Tris-Base", "formula": "C4H11NO3", "mw": 121.14, "category": "Buffer", "state": "solid", "default_conc": 50.0, "default_unit": "mM"},
    {"name": "Tris-HCl", "formula": "C4H11NO3·HCl", "mw": 157.60, "category": "Buffer", "state": "solid", "default_conc": 50.0, "default_unit": "mM"},
    {"name": "HEPES (Free Acid)", "formula": "C8H18N2O4S", "mw": 238.30, "category": "Buffer", "state": "solid", "default_conc": 25.0, "default_unit": "mM"},
    {"name": "HEPES Sodium Salt", "formula": "C8H17N2NaO4S", "mw": 260.29, "category": "Buffer", "state": "solid", "default_conc": 25.0, "default_unit": "mM"},
    {"name": "MES (Free Acid, Anhydrous)", "formula": "C6H13NO4S", "mw": 195.24, "category": "Buffer", "state": "solid", "default_conc": 50.0, "default_unit": "mM"},
    {"name": "MES (Monohydrate)", "formula": "C6H13NO4S·H2O", "mw": 213.25, "category": "Buffer", "state": "solid", "default_conc": 50.0, "default_unit": "mM"},
    {"name": "MOPS (Free Acid, Anhydrous)", "formula": "C7H15NO4S", "mw": 209.26, "category": "Buffer", "state": "solid", "default_conc": 50.0, "default_unit": "mM"},
    {"name": "PIPES (Free Acid, Anhydrous)", "formula": "C8H18N2O6S2", "mw": 302.37, "category": "Buffer", "state": "solid", "default_conc": 25.0, "default_unit": "mM"},
    {"name": "Bis-Tris (Free Base, Anhydrous)", "formula": "C8H19NO5", "mw": 209.24, "category": "Buffer", "state": "solid", "default_conc": 25.0, "default_unit": "mM"},
    {"name": "Na2HPO4 (Anhydrous)", "formula": "Na2HPO4", "mw": 141.96, "category": "Buffer", "state": "solid", "default_conc": 50.0, "default_unit": "mM"},
    {"name": "Na2HPO4·7H2O", "formula": "Na2HPO4·7H2O", "mw": 268.07, "category": "Buffer", "state": "solid", "default_conc": 50.0, "default_unit": "mM"},
    {"name": "NaH2PO4 (Anhydrous)", "formula": "NaH2PO4", "mw": 119.98, "category": "Buffer", "state": "solid", "default_conc": 50.0, "default_unit": "mM"},
    {"name": "NaH2PO4·H2O", "formula": "NaH2PO4·H2O", "mw": 137.99, "category": "Buffer", "state": "solid", "default_conc": 50.0, "default_unit": "mM"},
    {"name": "KH2PO4 (Anhydrous)", "formula": "KH2PO4", "mw": 136.09, "category": "Buffer", "state": "solid", "default_conc": 20.0, "default_unit": "mM"},
    {"name": "Sodium Acetate (Anhydrous)", "formula": "CH3COONa", "mw": 82.03, "category": "Buffer", "state": "solid", "default_conc": 50.0, "default_unit": "mM"},
    {"name": "Sodium Acetate (Trihydrate)", "formula": "CH3COONa·3H2O", "mw": 136.08, "category": "Buffer", "state": "solid", "default_conc": 50.0, "default_unit": "mM"},
    {"name": "Trisodium Citrate·2H2O", "formula": "Na3C6H5O7·2H2O", "mw": 294.10, "category": "Buffer", "state": "solid", "default_conc": 50.0, "default_unit": "mM"},
    {"name": "Citric Acid (Anhydrous)", "formula": "C6H8O7", "mw": 192.12, "category": "Buffer", "state": "solid", "default_conc": 50.0, "default_unit": "mM"},
    {"name": "Bicine (Anhydrous)", "formula": "C6H13NO4", "mw": 163.17, "category": "Buffer", "state": "solid", "default_conc": 50.0, "default_unit": "mM"},
    {"name": "CHES (Free Acid, Anhydrous)", "formula": "C8H17NO3S", "mw": 207.29, "category": "Buffer", "state": "solid", "default_conc": 50.0, "default_unit": "mM"},
    {"name": "CAPS (Free Acid, Anhydrous)", "formula": "C9H19NO3S", "mw": 221.32, "category": "Buffer", "state": "solid", "default_conc": 50.0, "default_unit": "mM"},
    # Salts & HIC Kosmotropes (7)
    {"name": "NaCl (Sodium Chloride)", "formula": "NaCl", "mw": 58.44, "category": "Salt / HIC", "state": "solid", "default_conc": 150.0, "default_unit": "mM"},
    {"name": "KCl (Potassium Chloride)", "formula": "KCl", "mw": 74.55, "category": "Salt / HIC", "state": "solid", "default_conc": 150.0, "default_unit": "mM"},
    {"name": "Ammonium Sulfate (NH4)2SO4 [HIC Salt]", "formula": "(NH4)2SO4", "mw": 132.14, "category": "Salt / HIC", "state": "solid", "default_conc": 1.3, "default_unit": "M"},
    {"name": "Sodium Sulfate (Na2SO4, Anhydrous) [HIC Salt]", "formula": "Na2SO4", "mw": 142.04, "category": "Salt / HIC", "state": "solid", "default_conc": 1.0, "default_unit": "M"},
    {"name": "MgCl2·6H2O", "formula": "MgCl2·6H2O", "mw": 203.30, "category": "Salt / Cofactor", "state": "solid", "default_conc": 5.0, "default_unit": "mM"},
    {"name": "CaCl2·2H2O", "formula": "CaCl2·2H2O", "mw": 147.01, "category": "Salt / Cofactor", "state": "solid", "default_conc": 2.0, "default_unit": "mM"},
    {"name": "ZnSO4·7H2O", "formula": "ZnSO4·7H2O", "mw": 287.54, "category": "Salt / Cofactor", "state": "solid", "default_conc": 0.1, "default_unit": "mM"},
    # Affinity Eluants (5)
    {"name": "Imidazole (His-Tag Eluant)", "formula": "C3H4N2", "mw": 68.08, "category": "Affinity Eluant", "state": "solid", "default_conc": 250.0, "default_unit": "mM"},
    {"name": "D-Desthiobiotin (Strep Eluant)", "formula": "C10H18N2O3", "mw": 214.26, "category": "Affinity Eluant", "state": "solid", "default_conc": 2.5, "default_unit": "mM"},
    {"name": "D-Biotin (Strep-Tactin XT Eluant)", "formula": "C10H16N2O3S", "mw": 244.31, "category": "Affinity Eluant", "state": "solid", "default_conc": 50.0, "default_unit": "mM"},
    {"name": "Reduced L-Glutathione (GSH • GST Eluant)", "formula": "C10H17N3O6S", "mw": 307.32, "category": "Affinity Eluant", "state": "solid", "default_conc": 15.0, "default_unit": "mM"},
    {"name": "D-Maltose·H2O (MBP Eluant)", "formula": "C12H22O11·H2O", "mw": 360.31, "category": "Affinity Eluant", "state": "solid", "default_conc": 10.0, "default_unit": "mM"},
    # Reducing Agents & Chelators (5)
    {"name": "DTT (Dithiothreitol)", "formula": "C4H10O2S2", "mw": 154.25, "category": "Reducing Agent", "state": "solid", "default_conc": 1.0, "default_unit": "mM"},
    {"name": "TCEP-HCl", "formula": "C9H15O6P·HCl", "mw": 286.65, "category": "Reducing Agent", "state": "solid", "default_conc": 0.5, "default_unit": "mM"},
    {"name": "2-Mercaptoethanol (β-ME)", "formula": "C2H6OS", "mw": 78.13, "category": "Reducing Agent", "state": "liquid", "density_g_ml": 1.114, "default_conc": 5.0, "default_unit": "mM"},
    {"name": "EDTA Disodium·2H2O", "formula": "C10H14N2Na2O8·2H2O", "mw": 372.24, "category": "Chelator", "state": "solid", "default_conc": 1.0, "default_unit": "mM"},
    {"name": "EGTA (Free Acid)", "formula": "C14H24N2O10", "mw": 380.35, "category": "Chelator", "state": "solid", "default_conc": 1.0, "default_unit": "mM"},
    # Stabilizers, Detergents, Denaturants & Inhibitors (9)
    {"name": "Glycerol (Liquid, ρ = 1.261 g/mL)", "formula": "C3H8O3", "mw": 92.09, "category": "Stabilizer", "state": "liquid", "density_g_ml": 1.261, "default_conc": 5.0, "default_unit": "% (v/v)"},
    {"name": "Sucrose", "formula": "C12H22O11", "mw": 342.30, "category": "Stabilizer", "state": "solid", "default_conc": 5.0, "default_unit": "% (w/v)"},
    {"name": "L-Arginine HCl (Anti-Aggregation)", "formula": "C6H14N4O2·HCl", "mw": 210.66, "category": "Stabilizer", "state": "solid", "default_conc": 50.0, "default_unit": "mM"},
    {"name": "Triton X-100", "formula": "C14H22O·(C2H4O)n (n≈9.5 avg; n=10 is 646.86)", "mw": 625.00, "category": "Detergent", "state": "liquid", "density_g_ml": 1.07, "default_conc": 0.1, "default_unit": "% (v/v)"},
    {"name": "Tween-20", "formula": "C58H114O26 (Polysorbate 20 avg.)", "mw": 1227.54, "category": "Detergent", "state": "liquid", "density_g_ml": 1.10, "default_conc": 0.05, "default_unit": "% (v/v)"},
    {"name": "SDS (Sodium Dodecyl Sulfate)", "formula": "C12H25NaO4S", "mw": 288.38, "category": "Detergent", "state": "solid", "default_conc": 0.1, "default_unit": "% (w/v)"},
    {"name": "Urea (Denaturant)", "formula": "CH4N2O", "mw": 60.06, "category": "Denaturant", "state": "solid", "default_conc": 8.0, "default_unit": "M"},
    {"name": "Guanidine-HCl (Denaturant)", "formula": "CH5N3·HCl", "mw": 95.53, "category": "Denaturant", "state": "solid", "default_conc": 6.0, "default_unit": "M"},
    {"name": "PMSF (Protease Inhibitor)", "formula": "C7H7FO2S", "mw": 174.19, "category": "Inhibitor", "state": "solid", "default_conc": 1.0, "default_unit": "mM"}
]

REAGENT_LOOKUP: Dict[str, Dict[str, Any]] = {r["name"]: r for r in COMMON_LAB_REAGENTS}
# Backward-compatible aliases so legacy/short reagent names still resolve exact density & state
_LEGACY_REAGENT_ALIASES = {
    "Glycerol (Liquid, ρ = 1.26 g/mL)": "Glycerol (Liquid, ρ = 1.261 g/mL)",
    "Glycerol": "Glycerol (Liquid, ρ = 1.261 g/mL)",
    "MOPS": "MOPS (Free Acid, Anhydrous)",
    "PIPES": "PIPES (Free Acid, Anhydrous)",
    "Bis-Tris": "Bis-Tris (Free Base, Anhydrous)",
    "KH2PO4": "KH2PO4 (Anhydrous)",
    "Bicine": "Bicine (Anhydrous)",
    "CHES": "CHES (Free Acid, Anhydrous)",
    "CAPS": "CAPS (Free Acid, Anhydrous)",
    "Sodium Sulfate (Na2SO4) [HIC Salt]": "Sodium Sulfate (Na2SO4, Anhydrous) [HIC Salt]",
    "EGTA": "EGTA (Free Acid)",
    "2-Mercaptoethanol": "2-Mercaptoethanol (β-ME)",
    "β-Mercaptoethanol": "2-Mercaptoethanol (β-ME)",
    "beta-Mercaptoethanol": "2-Mercaptoethanol (β-ME)",
}
for _alias, _canonical in _LEGACY_REAGENT_ALIASES.items():
    if _canonical in REAGENT_LOOKUP:
        REAGENT_LOOKUP[_alias] = REAGENT_LOOKUP[_canonical]


def _resolve_reagent_info(name: str) -> Dict[str, Any]:
    """Resolves reagent physical state and density across canonical names, aliases, and user-typed names."""
    if name in REAGENT_LOOKUP:
        return REAGENT_LOOKUP[name]
    low = name.lower().strip()
    for k, v in REAGENT_LOOKUP.items():
        if k.lower() == low:
            return v
    if "mercaptoethanol" in low or low in ("β-me", "b-me", "bme"):
        return REAGENT_LOOKUP["2-Mercaptoethanol (β-ME)"]
    if "glycerol" in low:
        return REAGENT_LOOKUP["Glycerol (Liquid, ρ = 1.261 g/mL)"]
    if "triton x-100" in low or "triton-x100" in low:
        return REAGENT_LOOKUP["Triton X-100"]
    if "tween-20" in low or "tween 20" in low or "polysorbate 20" in low:
        return REAGENT_LOOKUP["Tween-20"]
    return {}


# Editable Starter Templates for the Custom Buffer Calculator
PRESET_RECIPE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "his_lysis": {
        "label": "🏷️ His-Tag (Ni-NTA) — Lysis / Binding Buffer",
        "target_ph": 8.0,
        "notes": "Low imidazole (20 mM) + 300 mM NaCl prevents non-specific host background binding onto Ni2+ beads. Keep EDTA-free (EDTA strips Ni2+ from standard Ni-NTA resin) and omit TCEP if purifying disulfide-bonded proteins.",
        "rows": [
            {"Reagent": "Tris-Base", "MW (g/mol)": 121.14, "Value": 50.0, "Unit": "mM"},
            {"Reagent": "NaCl (Sodium Chloride)", "MW (g/mol)": 58.44, "Value": 300.0, "Unit": "mM"},
            {"Reagent": "Imidazole (His-Tag Eluant)", "MW (g/mol)": 68.08, "Value": 20.0, "Unit": "mM"},
            {"Reagent": "Glycerol (Liquid, ρ = 1.261 g/mL)", "MW (g/mol)": 92.09, "Value": 5.0, "Unit": "% (v/v)"},
            {"Reagent": "TCEP-HCl", "MW (g/mol)": 286.65, "Value": 0.5, "Unit": "mM"}
        ]
    },
    "his_elution": {
        "label": "🏷️ His-Tag (Ni-NTA) — Elution Buffer (300 mM Imidazole)",
        "target_ph": 8.0,
        "notes": "300 mM Imidazole competes with the 6xHis tag for Ni2+ sites. Re-adjust pH down to 8.0 with HCl AFTER dissolving alkaline imidazole!",
        "rows": [
            {"Reagent": "Tris-Base", "MW (g/mol)": 121.14, "Value": 50.0, "Unit": "mM"},
            {"Reagent": "NaCl (Sodium Chloride)", "MW (g/mol)": 58.44, "Value": 300.0, "Unit": "mM"},
            {"Reagent": "Imidazole (His-Tag Eluant)", "MW (g/mol)": 68.08, "Value": 300.0, "Unit": "mM"},
            {"Reagent": "Glycerol (Liquid, ρ = 1.261 g/mL)", "MW (g/mol)": 92.09, "Value": 5.0, "Unit": "% (v/v)"}
        ]
    },
    "iex_low_salt": {
        "label": "⚡ Ion-Exchange (IEX) — Buffer A (Low-Salt Binding)",
        "target_ph": 7.5,
        "notes": "Low ionic strength (20–30 mM NaCl) lets charged proteins bind tightly to SP (CEX) or Q (AEX) resin.",
        "rows": [
            {"Reagent": "HEPES (Free Acid)", "MW (g/mol)": 238.30, "Value": 25.0, "Unit": "mM"},
            {"Reagent": "NaCl (Sodium Chloride)", "MW (g/mol)": 58.44, "Value": 25.0, "Unit": "mM"},
            {"Reagent": "DTT (Dithiothreitol)", "MW (g/mol)": 154.25, "Value": 1.0, "Unit": "mM"}
        ]
    },
    "iex_high_salt": {
        "label": "⚡ Ion-Exchange (IEX) — Buffer B (1.0 M NaCl Elution)",
        "target_ph": 7.5,
        "notes": "Mix linearly from 0% to 100% Buffer B (25 mM ➜ 1.0 M NaCl) to elute proteins in order of increasing net surface charge.",
        "rows": [
            {"Reagent": "HEPES (Free Acid)", "MW (g/mol)": 238.30, "Value": 25.0, "Unit": "mM"},
            {"Reagent": "NaCl (Sodium Chloride)", "MW (g/mol)": 58.44, "Value": 1.0, "Unit": "M"},
            {"Reagent": "DTT (Dithiothreitol)", "MW (g/mol)": 154.25, "Value": 1.0, "Unit": "mM"}
        ]
    },
    "hic_binding": {
        "label": "💧 Hydrophobic Interaction (HIC) — Buffer A (High-Salt Binding)",
        "target_ph": 7.0,
        "notes": "Kosmotropic Ammonium Sulfate (1.3 M) drives hydrophobic patches onto Butyl/Phenyl/Octyl Sepharose. Start with ~65–70% water and adjust pH to 7.0.",
        "rows": [
            {"Reagent": "Na2HPO4 (Anhydrous)", "MW (g/mol)": 141.96, "Value": 50.0, "Unit": "mM"},
            {"Reagent": "Ammonium Sulfate (NH4)2SO4 [HIC Salt]", "MW (g/mol)": 132.14, "Value": 1.3, "Unit": "M"}
        ]
    },
    "hic_elution": {
        "label": "💧 Hydrophobic Interaction (HIC) — Buffer B (Salt-Free Elution)",
        "target_ph": 7.0,
        "notes": "Lowering Ammonium Sulfate linearly down to 0 M restores protein hydration shells and releases bound proteins in order of increasing hydrophobicity!",
        "rows": [
            {"Reagent": "Na2HPO4 (Anhydrous)", "MW (g/mol)": 141.96, "Value": 50.0, "Unit": "mM"}
        ]
    },
    "sec_running": {
        "label": "📏 Size-Exclusion (SEC) — Final Polishing Buffer",
        "target_ph": 7.5,
        "notes": "150 mM NaCl suppresses non-specific ionic interactions with Superdex/Superose agarose-dextran beads. Always 0.22 µm filter & degas!",
        "rows": [
            {"Reagent": "HEPES (Free Acid)", "MW (g/mol)": 238.30, "Value": 25.0, "Unit": "mM"},
            {"Reagent": "NaCl (Sodium Chloride)", "MW (g/mol)": 58.44, "Value": 150.0, "Unit": "mM"},
            {"Reagent": "TCEP-HCl", "MW (g/mol)": 286.65, "Value": 0.5, "Unit": "mM"}
        ]
    },
    "metalloprotein_safe": {
        "label": "🪙 Metalloprotein-Safe — EDTA-Free & Imidazole-Free Buffer (Urease / Zn / Cu / Fe)",
        "target_ph": 7.5,
        "notes": "Non-chelating Good's buffer (HEPES) + 150 mM NaCl + 5% Glycerol — ZERO Imidazole, ZERO EDTA, ZERO Citrate so catalytic Ni²⁺/Zn²⁺/Cu²⁺/Fe²⁺ cofactors remain intact!",
        "rows": [
            {"Reagent": "HEPES (Free Acid)", "MW (g/mol)": 238.30, "Value": 50.0, "Unit": "mM"},
            {"Reagent": "NaCl (Sodium Chloride)", "MW (g/mol)": 58.44, "Value": 150.0, "Unit": "mM"},
            {"Reagent": "Glycerol (Liquid, ρ = 1.261 g/mL)", "MW (g/mol)": 92.09, "Value": 5.0, "Unit": "% (v/v)"}
        ]
    },
    "strep_elution": {
        "label": "🧲 Strep-Tactin — Elution Buffer E (2.5 mM Desthiobiotin)",
        "target_ph": 8.0,
        "notes": "2.5 mM D-Desthiobiotin gently elutes Strep-tag II from classic Strep-Tactin! (Note: For Strep-Tactin XT, use the 50 mM D-Biotin Buffer BXT template below).",
        "rows": [
            {"Reagent": "Tris-Base", "MW (g/mol)": 121.14, "Value": 100.0, "Unit": "mM"},
            {"Reagent": "NaCl (Sodium Chloride)", "MW (g/mol)": 58.44, "Value": 150.0, "Unit": "mM"},
            {"Reagent": "EDTA Disodium·2H2O", "MW (g/mol)": 372.24, "Value": 1.0, "Unit": "mM"},
            {"Reagent": "D-Desthiobiotin (Strep Eluant)", "MW (g/mol)": 214.26, "Value": 2.5, "Unit": "mM"}
        ]
    },
    "strep_xt_elution": {
        "label": "🧲 Strep-Tactin XT — Elution Buffer BXT (50 mM D-Biotin)",
        "target_ph": 8.0,
        "notes": "Strep-Tactin XT binds with pM–nM affinity and requires 50 mM D-Biotin (not Desthiobiotin) to elute! Dissolve D-Biotin with NaOH titration to pH 8.0.",
        "rows": [
            {"Reagent": "Tris-Base", "MW (g/mol)": 121.14, "Value": 100.0, "Unit": "mM"},
            {"Reagent": "NaCl (Sodium Chloride)", "MW (g/mol)": 58.44, "Value": 150.0, "Unit": "mM"},
            {"Reagent": "EDTA Disodium·2H2O", "MW (g/mol)": 372.24, "Value": 1.0, "Unit": "mM"},
            {"Reagent": "D-Biotin (Strep-Tactin XT Eluant)", "MW (g/mol)": 244.31, "Value": 50.0, "Unit": "mM"}
        ]
    },
    "gst_elution": {
        "label": "🧪 GST-Tag — Reduced Glutathione Elution Buffer",
        "target_ph": 8.0,
        "notes": "Re-adjust pH to 8.0 with NaOH AFTER dissolving Reduced L-Glutathione (acidic GSH drops unbuffered/Tris pH sharply!).",
        "rows": [
            {"Reagent": "Tris-Base", "MW (g/mol)": 121.14, "Value": 50.0, "Unit": "mM"},
            {"Reagent": "NaCl (Sodium Chloride)", "MW (g/mol)": 58.44, "Value": 150.0, "Unit": "mM"},
            {"Reagent": "Reduced L-Glutathione (GSH • GST Eluant)", "MW (g/mol)": 307.32, "Value": 15.0, "Unit": "mM"}
        ]
    },
    "mbp_elution": {
        "label": "🍬 MBP-Tag — Amylose Maltose Elution Buffer",
        "target_ph": 7.4,
        "notes": "10 mM D-Maltose gently displaces MBP-fusion proteins from Amylose resin under native conditions.",
        "rows": [
            {"Reagent": "Tris-Base", "MW (g/mol)": 121.14, "Value": 20.0, "Unit": "mM"},
            {"Reagent": "NaCl (Sodium Chloride)", "MW (g/mol)": 58.44, "Value": 200.0, "Unit": "mM"},
            {"Reagent": "EDTA Disodium·2H2O", "MW (g/mol)": 372.24, "Value": 1.0, "Unit": "mM"},
            {"Reagent": "D-Maltose·H2O (MBP Eluant)", "MW (g/mol)": 360.31, "Value": 10.0, "Unit": "mM"}
        ]
    },
    "custom_blank": {
        "label": "✨ Custom Recipe (Build Your Own Buffer From Scratch)",
        "target_ph": 7.4,
        "notes": "Add any chemical below using mM, M, µM, % (w/v), % (v/v), mg/mL, or direct weight in grams (g) / milligrams (mg)!",
        "rows": [
            {"Reagent": "Tris-Base", "MW (g/mol)": 121.14, "Value": 50.0, "Unit": "mM"},
            {"Reagent": "NaCl (Sodium Chloride)", "MW (g/mol)": 58.44, "Value": 150.0, "Unit": "mM"}
        ]
    }
}


def advise_iex_strategy(pi: float, working_ph: float, net_charge: float) -> Dict[str, Any]:
    """
    Evaluates protein behavior for Ion-Exchange Chromatography (IEX) at a chosen working pH
    per Cytiva IEX Handbook guidelines: |working_ph - pI| >= 0.8–1.0 prevents isoelectric precipitation,
    ensures strong binding capacity, and dictates counter-ion buffer selection.
    """
    delta_ph = working_ph - pi

    if abs(delta_ph) < 0.5:
        risk_level = "HIGH RISK OF PRECIPITATION"
        risk_desc = (
            f"Working pH ({working_ph:.1f}) is within 0.5 units of your protein's pI ({pi:.2f}). "
            f"Net charge is nearly zero ({net_charge:+.1f}), so protein molecules lose electrostatic repulsion "
            f"and tend to aggregate/precipitate—especially in low-salt IEX binding buffer! Shift pH by ±1.0 unit OR use Hydrophobic Interaction (HIC) instead."
        )
    elif abs(delta_ph) < 1.0:
        risk_level = "MODERATE CAUTION"
        risk_desc = (
            f"Working pH ({working_ph:.1f}) is fairly close to pI ({pi:.2f}; net charge {net_charge:+.1f}). "
            "Keep protein moderately diluted or shift pH ≥1.0 unit away from pI for stronger dynamic binding capacity."
        )
    else:
        risk_level = "OPTIMAL STABILITY"
        risk_desc = (
            f"Working pH ({working_ph:.1f}) sits safely away from pI ({pi:.2f}), giving a strong net charge ({net_charge:+.1f}) "
            "that maintains electrostatic repulsion in solution."
        )

    if net_charge <= -2.0:
        recommended_column = "Anion Exchange Chromatography (AEX — Positively Charged Beads)"
        resins = "HiTrap Q HP / Capto Q (Strong AEX, pH 2–12) or DEAE Sepharose FF (Weak AEX, pH 2–9)"
        mechanism = (
            f"At pH {working_ph:.1f} (above pI {pi:.2f}), your protein is NEGATIVELY charged ({net_charge:+.1f}) "
            "and binds positively charged quaternary amine (Q) or DEAE ligands (use cationic/zwitterionic buffers like Bis-Tris or Tris; avoid phosphate/citrate!)."
        )
        elution = "Bind in low ionic strength (20–30 mM NaCl); elute with a linear gradient up to 1,000 mM (1.0 M) NaCl over 15–20 CV."
    elif net_charge >= 2.0:
        recommended_column = "Cation Exchange Chromatography (CEX — Negatively Charged Beads)"
        resins = "HiTrap SP HP / Capto S (Strong CEX, pH 2–12) or CM Sepharose FF (Weak CEX, pH 6–10)"
        mechanism = (
            f"At pH {working_ph:.1f} (below pI {pi:.2f}), your protein is POSITIVELY charged ({net_charge:+.1f}) "
            "and binds negatively charged sulfopropyl (SP) or carboxymethyl (CM) ligands (use anionic/Good's sulfonate buffers like Acetate, MES, Phosphate, or HEPES; avoid Tris!)."
        )
        elution = "Bind in low ionic strength (20–30 mM NaCl); elute with a linear gradient up to 1,000 mM (1.0 M) NaCl over 15–20 CV."
    else:
        recommended_column = "Weak Charge — Switch to HIC (Hydrophobic) or Adjust Buffer pH!"
        resins = "Use HiTrap Phenyl HP / Butyl HP (HIC) at this pH, OR shift pH until |Net Charge| ≥ 3."
        mechanism = (
            f"Net charge ({net_charge:+.1f}) is too weak for reliable Ion-Exchange binding at pH {working_ph:.1f}."
        )
        elution = "Either adjust buffer pH ≥ 1 unit away from pI, or use Hydrophobic Interaction Chromatography (HIC)."

    matching_buffers = [
        b for b in BUFFER_SYSTEMS if b["min_ph"] <= working_ph <= b["max_ph"]
    ]

    return {
        "working_ph": working_ph,
        "pi": pi,
        "net_charge": round(net_charge, 2),
        "risk_level": risk_level,
        "risk_desc": risk_desc,
        "recommended_column": recommended_column,
        "recommended_resins": resins,
        "mechanism": mechanism,
        "elution_strategy": elution,
        "recommended_buffers": matching_buffers
    }


def _count_internal_tm_helices(core_seq: str) -> int:
    """
    Scans sequence (skipping N-terminal 25-aa signal-peptide window when length > 60)
    for non-overlapping 19-residue transmembrane helices with mean Kyte-Doolittle >= 1.65.
    """
    s = core_seq[25:] if len(core_seq) > 60 else core_seq
    cnt, idx = 0, 0
    while idx <= len(s) - 19:
        mean_kd = sum(KYTE_DOOLITTLE.get(aa, 0.0) for aa in s[idx:idx + 19]) / 19.0
        if mean_kd >= 1.65:
            cnt += 1
            idx += 19
        else:
            idx += 1
    return cnt


def _strip_construct_tag_phrases_from_name(raw_name: str) -> str:
    """
    Removes engineered tag/protease-site phrases (e.g., '6xHis-TEV-', 'with N-terminal GST tag')
    from a construct display name so linker tags/proteases never misclassify the target protein.
    """
    s = (raw_name or "").lower()
    s = re.sub(r"\bwith\s+(?:an?\s+)?(?:n|c)[-\s]*terminal\b.*$", " ", s)
    s = re.sub(r"\((?:recombinant|synthetic)[^)]*\)", " ", s)
    s = re.sub(r"\b(?:6xhis|8xhis|10xhis|his6|his8|his10|polyhis|twin-?strep|strep-?tag(?:\s*ii)?|flag-?tag)\b", " ", s)
    s = re.sub(r"(?:^|[\s\-_/])(?:his|gst|mbp|sumo)[-\s]+(?:tev|3c|prescission|thrombin|factor\s*xa)[-\s]+", " ", s)
    s = re.sub(r"[-\s]+(?:tev|3c|prescission)[-\s]+(?:6xhis|8xhis|10xhis|his6|his|strep|flag)\b", " ", s)
    s = re.sub(r"\b(?:gst|mbp|his|flag|strep|myc|ha)-tagged\b", " ", s)
    return " ".join(s.split())


def infer_protein_biochemical_profile(
    sequence: str = "",
    protein_name: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    tag_detected: str = "None",
    construct_parts: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Multi-evidence biochemical protein intelligence engine combining:
    1. UniProtKB structured annotations (cofactors, metal binding sites, Fe-S clusters, heme,
       disulfide bonds, transmembrane segments, Swiss-Prot keywords, subcellular locations, EC numbers)
    2. Curated enzyme/protein family & accession knowledge base across metalloproteins (Ni2+, Zn2+,
       Cu+/2+, Mn2+, Co2+, Mo, non-heme Fe), Fe-S clusters, hemoproteins, Ca2+/Mg2+ proteins,
       secreted disulfide proteins, cytosolic thiol proteins, integral membrane proteins,
       nucleic-acid-binding proteins, PLP/flavin holoenzymes, active proteases, and IDPs
    3. High-precision sequence motifs & biophysical properties evaluated on the tag-stripped core sequence.
    """
    meta = metadata or {}
    annot = meta.get("biochem_annotations") or {}

    # 1. Isolate canonical target sequence (excluding engineered affinity tags & protease linkers)
    raw_seq = (sequence or "").strip().upper()
    term_tag_info = detect_sequence_terminal_tag(raw_seq) if raw_seq else {"tag_detected": "None", "tag_position": "N/A", "core_sequence": ""}
    core_seq = term_tag_info["core_sequence"]
    if construct_parts:
        target_segs = [seg for lbl, seg in construct_parts if "target" in lbl.lower() and seg]
        if target_segs:
            core_seq = "".join(target_segs).strip().upper()

    effective_tag = tag_detected if (tag_detected and tag_detected.lower() not in ("none", "n/a", "")) else term_tag_info["tag_detected"]
    tag_low = (effective_tag or "none").lower()
    has_tag = tag_low not in ("none", "n/a", "")

    acc = str(meta.get("uniprot_id") or meta.get("accession") or "").strip().upper()
    gene_str = str(meta.get("gene_name") or "").strip().lower()
    if gene_str in ("n/a", "none", acc.lower()):
        gene_str = ""

    raw_title = " ".join(
        str(x) for x in (protein_name, meta.get("protein_name", ""), meta.get("identified_protein", "")) if x
    )
    clean_name = _strip_construct_tag_phrases_from_name(raw_title)
    search_text = f"{clean_name} {gene_str}".strip()

    # Only trust structured UniProt cofactors, metal-binding ligands, ChEBI IDs, keywords & EC numbers —
    # NEVER regex-scan free-text prose `cofactor_notes` (which routinely say 'not nickel(II)' or 'inhibited by Zn(2+)')!
    cofactor_items = [str(x).strip().lower() for x in (annot.get("cofactors") or []) if str(x).strip()]
    ligand_items = [str(x).strip().lower() for x in (annot.get("metal_ligands") or []) if str(x).strip()]
    cofactor_chebis = {str(x).strip().upper() for x in (annot.get("cofactor_chebi_ids") or []) if str(x).strip()}
    ligand_chebis = {str(x).strip().upper() for x in (annot.get("metal_ligand_chebi_ids") or []) if str(x).strip()}
    all_chebis = cofactor_chebis | ligand_chebis
    cofactors_low = " | ".join(cofactor_items)
    ligands_low = " | ".join(ligand_items)
    keywords_low = {str(k).lower() for k in (annot.get("keywords") or [])}
    locs_low = " | ".join(str(x).lower() for x in (annot.get("subcellular_locations") or []))
    ecs = set(annot.get("ec_numbers") or [])
    metal_blob = f"{cofactors_low} | {ligands_low}"
    has_uniprot_annot = bool(
        keywords_low or cofactor_items or ligand_items or all_chebis or ecs
        or int(annot.get("disulfide_count") or 0) > 0
        or int(annot.get("transmembrane_count") or 0) > 0
    )

    # Serum cargo/transport carriers (e.g., BSA/HSA) bind transient blood ions non-catalytically (COFACTOR is empty)
    # and fold via disulfide bridges—they are NOT catalytic metalloenzymes.
    is_serum_carrier = (
        acc in {"P02769", "P02768"}
        or bool(re.search(r"\b(?:serum\s+albumin|bovine\s+serum\s+albumin)\b|^albumin\b", clean_name))
    )

    core_len = max(1, len(core_seq)) if core_seq else int((metrics or {}).get("length") or 1)
    cys_count = core_seq.count("C") if core_seq else int(((metrics or {}).get("counts") or {}).get("cysteines", 0))
    if core_seq:
        aromatic_pct = ((core_seq.count("W") + core_seq.count("Y") + core_seq.count("F")) / core_len) * 100.0
    elif metrics and (metrics.get("counts") or {}).get("aromatic_residues") is not None:
        aromatic_pct = (float(metrics["counts"]["aromatic_residues"]) / core_len) * 100.0
    else:
        aromatic_pct = 0.0
    gravy = float((metrics or {}).get("gravy", 0.0))
    aliphatic_idx = float((metrics or {}).get("aliphatic_index", 75.0))

    categories: List[str] = []
    metal_labels: List[str] = []

    # --- A. NICKEL METALLOENZYMES & MATURATION CHAPERONES (e.g. Urease, [NiFe]-Hydrogenase, CODH, Ni-SOD) ---
    ni_catalytic_urease_accs = {
        "P07374", "E6Y5X0", "P18314", "P18315", "P18316",
        "P41020", "P41022", "P69996", "P14916", "P9WFF1", "P17086", "P08298", "B5YUX3",
    }
    ni_chaperone_accs = {"P18317", "P50049", "O25560", "P0A0U4", "Q09065", "Q09066"}
    is_urea_channel_or_mod_enz = bool(re.search(
        r"\b(?:urea\s+channel|urea\s+transporter|urea-proton\s+symporter|symporter|urei|methyltransferase|kinase|phosphatase|inhibitor)\b",
        clean_name,
    ))
    is_aerobic_mo_codh = bool(re.search(r"\b(?:cox[lms]|cut[lms])\b", gene_str))
    has_ni_annot = (
        not is_serum_carrier
        and not is_urea_channel_or_mod_enz
        and (
            "nickel" in keywords_low
            or bool({"CHEBI:49786", "CHEBI:25516"} & all_chebis)
            or bool(re.search(r"\bni\s*\(\s*2\+\s*\)|\bni\s+cation\b|\bni2\+|\bnickel\b|\[nife\]|\bf430\b", metal_blob))
            or "3.5.1.5" in ecs
        )
    )
    has_ni_gene = bool(re.search(r"\b(?:ure[a-g]|hyp[ab]|sodn|coos\d*)\b", gene_str))
    has_ni_name = (not is_urea_channel_or_mod_enz and not is_aerobic_mo_codh) and (
        has_ni_gene
        or bool(re.search(
            r"\b(?:urease|urea\s+amidohydrolase|\[?nife\]?[-\s]*hydrogenase|"
            r"nickel\s+superoxide\s+dismutase|superoxide\s+dismutase\s*\[ni\]|"
            r"carbon\s+monoxide\s+dehydrogenase|methyl[-\s]*coenzyme\s+m\s+reductase)\b",
            clean_name,
        ))
    )
    has_urease_motif = bool(core_seq and re.search(r"EH[LM]DMLMVCHHL|[MVLI][VLI]CHHL[DNSK]", core_seq))
    is_urease_accessory = bool(re.search(r"\b(?:accessory|maturation|chaperone|ure[defg]|hyp[ab])\b", search_text)) or (acc in ni_chaperone_accs)
    if acc in ni_catalytic_urease_accs or acc in ni_chaperone_accs or has_ni_annot or has_ni_name or has_urease_motif:
        categories.append("ni_metalloenzyme")
        if is_urease_accessory and not has_urease_motif and acc not in ni_catalytic_urease_accs:
            metal_labels.append("Ni²⁺ Maturation / Chaperone")
        elif ("urease" in clean_name or "3.5.1.5" in ecs or acc in ni_catalytic_urease_accs or has_urease_motif):
            metal_labels.append("Ni²⁺ (Dinuclear Active Site)")
        else:
            metal_labels.append("Ni²⁺ Cofactor")

    # --- B. O2-SENSITIVE IRON-SULFUR ([2Fe-2S], [3Fe-4S], [4Fe-4S]) & RADICAL SAM ENZYMES ---
    fes_chebis = {"CHEBI:49883", "CHEBI:49601", "CHEBI:21137", "CHEBI:30408"}
    has_fes_annot = (
        not is_serum_carrier
        and (
            "iron-sulfur" in keywords_low
            or bool(fes_chebis & all_chebis)
            or bool(re.search(r"\[?[234]fe-[234]s\]?|iron-sulfur|fe-s\s+cluster", metal_blob))
        )
    )
    is_fnr_flavo = bool(re.search(r"ferredoxin\s*[-\u2013]+\s*nadp", search_text))
    # A real [Fe-S] cluster requires >= 3 coordinating Cys residues; metal-free de novo ferredoxin-fold
    # scaffolds (e.g., 2KL8 with 0 Cys) cannot coordinate an [Fe-S] cluster unless UniProt explicitly annotates one.
    lacks_fes_cys_ligands = (len(core_seq) >= 20) and (not has_fes_annot) and (core_seq.count("C") < 3)
    has_fes_name = (not is_fnr_flavo) and (not lacks_fes_cys_ligands) and bool(re.search(
        r"\b(?:ferredoxin|aconitase|rubredoxin|rieske|nitrogenase|\[?fefe\]?[-\s]*hydrogenase|"
        r"radical\s+sam|biotin\s+synthase|lipoyl\s+synthase|lipoate\s+synthase|"
        r"endonuclease\s+iii|sulfhydrogenase|xanthine\s+dehydrogenase)\b",
        search_text,
    ))
    has_fes_motif = bool((not has_uniprot_annot) and core_seq and re.search(r"C.{2}C.{2}C.{3}CP", core_seq))
    if has_fes_annot or has_fes_name or has_fes_motif:
        categories.append("fe_s_cluster")
        metal_labels.append("[Fe–S] Cluster (O₂-Labile)")

    # --- C. HEME PROTEINS / HEMOPROTEINS (Globins, Cytochromes, P450s, Catalases, Heme Peroxidases) ---
    heme_chebis = {"CHEBI:60344", "CHEBI:26355", "CHEBI:61717", "CHEBI:60342", "CHEBI:17627"}
    has_heme_annot = (
        not is_serum_carrier
        and (
            "heme" in keywords_low
            or bool(heme_chebis & all_chebis)
            or bool(re.search(r"\bheme\b|\bprotoheme\b|\bsiroheme\b|\bhemin\b", metal_blob))
        )
    )
    is_thiol_peroxidase = bool(re.search(r"\b(?:glutathione\s+peroxidase|peroxiredoxin|thioredoxin\s+peroxidase)\b", search_text))
    is_p450_reductase = bool(re.search(r"cytochrome\s+p450\s+reductase", search_text))
    has_heme_name = (not is_thiol_peroxidase and not is_p450_reductase) and bool(re.search(
        r"\b(?:myoglobin|hemoglobin|neuroglobin|cytoglobin|leghemoglobin|cytochrome|cyp\d+[a-z]\d*|"
        r"catalase|(?:horseradish|myelo|lacto|ascorbate|lignin|manganese|heme)\s+peroxidase|"
        r"nitric\s+oxide\s+synthase|indoleamine\s+2,3-dioxygenase|tryptophan\s+2,3-dioxygenase)\b",
        search_text,
    ))
    has_cyp_motif = bool(
        (not has_uniprot_annot)
        and core_seq
        and "deaminase" not in clean_name
        and re.search(r"F[SGNH].[GD][^F][RKHPT][^P]C[LIVMFAP][GAD]", core_seq)
    )
    if acc == "P02185" or has_heme_annot or has_heme_name or has_cyp_motif:
        categories.append("heme_protein")
        metal_labels.append("Heme Fe (Porphyrin)")

    # --- D. ZINC METALLOENZYMES & STRUCTURAL ZINC-FINGER PROTEINS ---
    has_zn_primary_annot = (
        not is_serum_carrier
        and (
            "zinc" in keywords_low
            or "zinc-finger" in keywords_low
            or "metalloprotease" in keywords_low
            or int(annot.get("zinc_finger_count") or 0) > 0
            or "CHEBI:29105" in ligand_chebis
            or bool(re.search(r"\bzn\s*\(\s*2\+\s*\)|\bzn\s+cation\b|\bzn2\+|\bzinc\b", ligands_low))
        )
    )
    has_zn_annot = (
        not is_serum_carrier
        and (
            has_zn_primary_annot
            or "CHEBI:29105" in cofactor_chebis
            or bool(re.search(r"\bzn\s*\(\s*2\+\s*\)|\bzn\s+cation\b|\bzn2\+|\bzinc\b", cofactors_low))
        )
    )
    has_zn_name = (not is_serum_carrier) and bool(re.search(
        r"\b(?:carbonic\s+anhydrase|metalloproteinase|metalloprotease|matrix\s+metalloproteinase|mmp-?\d+|"
        r"thermolysin|collagenase|stromelysin|gelatinase|matrilysin|astacin|adamalysin|serralysin|"
        r"carboxypeptidase\s*[ab]|alkaline\s+phosphatase|alcohol\s+dehydrogenase|metallo-?beta-?lactamase|"
        r"ndm-?\d+|aminopeptidase|histone\s+deacetylase|hdac\d*|superoxide\s+dismutase\s*\[?cu-?zn\]?|sod1|"
        r"dihydropyrimidinase|zinc[-\s]+finger|cellular\s+tumor\s+antigen\s+p53|tumor\s+protein\s+p53|tp53)\b",
        search_text,
    ))
    has_zn_motif = bool(
        (not has_uniprot_annot)
        and (not is_serum_carrier)
        and core_seq
        and "ni_metalloenzyme" not in categories
        and (
            re.search(r"HE[LIVMFYW]{2}H[A-Z]{2}[GSTA][LIVMFYW]|HE[A-Z]{2}H[A-Z]{2}G[A-Z]{2}[HD]", core_seq)
            or re.search(r"QFHFHW|HFHWG.{10,35}LHLVHW", core_seq)
            or len(re.findall(r"C[A-Z]{2,4}C[A-Z]{12}H[A-Z]{3,5}H", core_seq)) >= 2
        )
    )
    if (not is_serum_carrier) and (acc in {"P00918", "P04637"} or has_zn_annot or has_zn_name or has_zn_motif):
        categories.append("zn_metalloprotein")
        if "Zn²⁺" not in " ".join(metal_labels):
            metal_labels.append("Zn²⁺ Cofactor")

    # --- E. COPPER, MANGANESE, COBALT, MOLYBDENUM & NON-HEME IRON METALLOENZYMES ---
    # When physiological Zn2+ is already confirmed by Swiss-Prot keywords/ligands (e.g. Carbonic Anhydrase II P00918
    # where Co(2+) is only listed as a lower-efficiency in-vitro substitute), require keyword or ligand confirmation
    # for secondary Co/Cu/Mn/Fe so minor in-vitro substitutes never pollute the native cofactor identity!
    tm_text_for_check = ligands_low if has_zn_primary_annot else metal_blob
    tm_chebis_for_check = ligand_chebis if has_zn_primary_annot else all_chebis
    cu_chebis = {"CHEBI:29036", "CHEBI:49552", "CHEBI:23378"}
    has_cu_sig = (
        "copper" in keywords_low
        or bool(cu_chebis & tm_chebis_for_check)
        or bool(re.search(r"\bcu\s*\(\s*[12]?\+\s*\)|\bcu\s+cation\b|\bcopper\b", tm_text_for_check))
    )
    has_mn_sig = (
        "manganese" in keywords_low
        or "CHEBI:29035" in tm_chebis_for_check
        or bool(re.search(r"\bmn\s*\(\s*2\+\s*\)|\bmn\s+cation\b|\bmanganese\b", tm_text_for_check))
    )
    has_co_sig = (
        "cobalt" in keywords_low
        or bool({"CHEBI:48828", "CHEBI:30411"} & tm_chebis_for_check)
        or bool(re.search(r"\bco\s*\(\s*2\+\s*\)|\bco\s+cation\b|\bcobalt\b|\bcobalamin\b", tm_text_for_check))
    )
    has_mo_sig = (
        bool({"molybdenum", "tungsten", "vanadium"} & keywords_low)
        or bool({"CHEBI:60539", "CHEBI:71302"} & tm_chebis_for_check)
        or bool(re.search(r"\bmolybd|\btungsten\b|\bvanadium\b", tm_text_for_check))
    )
    has_nonheme_fe_sig = (
        (
            "iron" in keywords_low
            or bool({"CHEBI:29033", "CHEBI:29034"} & tm_chebis_for_check)
            or bool(re.search(r"\bfe\s*\(\s*[23]\+\s*\)|\bfe\s+cation\b|\bdiiron\b", tm_text_for_check))
        )
        and "heme_protein" not in categories
        and "fe_s_cluster" not in categories
    )
    has_other_tm_annot = (not is_serum_carrier) and (
        has_cu_sig or has_mn_sig or has_co_sig or has_mo_sig or has_nonheme_fe_sig or is_aerobic_mo_codh
    )
    has_other_tm_name = (not is_serum_carrier) and bool(re.search(
        r"\b(?:laccase|tyrosinase|plastocyanin|azurin|ceruloplasmin|ascorbate\s+oxidase|hemocyanin|"
        r"arginase|superoxide\s+dismutase\s*\[?(?:mn|fe|cu-?zn)\]?|sod[123ab]|ribonucleotide\s+reductase|"
        r"lipoxygenase|phenylalanine-?4-?hydroxylase|tyrosine\s+3-?monooxygenase|nitrile\s+hydratase|"
        r"nitrate\s+reductase|sulfite\s+oxidase|xanthine\s+oxidase|metallothionein)\b",
        search_text,
    ))
    has_generic_metal_word = (
        not is_serum_carrier
        and not any(c in categories for c in ("ni_metalloenzyme", "fe_s_cluster", "zn_metalloprotein", "heme_protein"))
        and bool(re.search(r"\b(?:metalloenzyme|metalloprotein)\b", search_text))
    )
    if has_other_tm_annot or has_other_tm_name or has_generic_metal_word:
        categories.append("cu_mn_co_mo_metalloprotein")
        if has_cu_sig or re.search(r"\b(?:laccase|tyrosinase|plastocyanin|azurin|ceruloplasmin|hemocyanin|sod1|cu-?zn)\b", search_text):
            metal_labels.append("Cu⁺/²⁺ Cofactor")
        elif has_mn_sig or re.search(r"\b(?:arginase|sod2|soda)\b", search_text):
            metal_labels.append("Mn²⁺ Cofactor")
        elif has_co_sig:
            metal_labels.append("Co²⁺ / Cobalamin")
        elif has_mo_sig or is_aerobic_mo_codh or re.search(r"\b(?:nitrate\s+reductase|sulfite\s+oxidase|xanthine\s+oxidase)\b", search_text):
            metal_labels.append("Mo Cofactor")
        else:
            metal_labels.append("Non-Heme Fe / Transition Metal")

    # --- F. CALCIUM / MAGNESIUM COFACTOR PROTEINS (EF-Hand, Gla-Domain, Polymerases, Kinases, Subtilisin/Amylase) ---
    needs_calcium = (not is_serum_carrier) and (
        acc == "P00734"
        or "calcium" in keywords_low
        or "CHEBI:29108" in all_chebis
        or bool(re.search(r"\bca\s*\(\s*2\+\s*\)|\bca\s+cation\b|\bcalcium\b", metal_blob))
        or bool(re.search(
            r"\b(?:calmodulin|troponin\s+c|parvalbumin|annexin|cadherin|prothrombin|subtilisin|"
            r"alpha-?amylase|deoxyribonuclease-?1|dnase\s+i|thermolysin)\b",
            search_text,
        ))
    )
    needs_magnesium = (not is_serum_carrier) and (
        "magnesium" in keywords_low
        or "CHEBI:18420" in all_chebis
        or bool(re.search(r"\bmg\s*\(\s*2\+\s*\)|\bmg\s+cation\b|\bmagnesium\b", metal_blob))
        or bool(re.search(
            r"\b(?:dna\s+polymerase|rna\s+polymerase|reverse\s+transcriptase|hexokinase|enolase|"
            r"pyrophosphatase|atpase|gtpase|cas9|cas12|topoisomerase|gyrase)\b",
            search_text,
        ))
    )
    if needs_calcium or needs_magnesium:
        categories.append("ca_mg_cofactor")
        if needs_calcium:
            metal_labels.append("Ca²⁺ Cofactor")
        if needs_magnesium and not needs_calcium:
            metal_labels.append("Mg²⁺ Cofactor")

    is_metalloprotein = any(
        c in categories for c in ("ni_metalloenzyme", "fe_s_cluster", "zn_metalloprotein", "cu_mn_co_mo_metalloprotein")
    )
    is_heme_protein = "heme_protein" in categories
    is_o2_sensitive = "fe_s_cluster" in categories or (
        not is_urea_channel_or_mod_enz
        and bool(re.search(r"\b(?:nitrogenase|hydrogenase|methyl[-\s]*coenzyme\s+m\s+reductase|carbon\s+monoxide\s+dehydrogenase)\b", search_text))
    )
    is_chelator_sensitive = is_metalloprotein or ("ca_mg_cofactor" in categories)

    # --- G. INTEGRAL MEMBRANE PROTEINS ---
    tm_annot_cnt = int(annot.get("transmembrane_count") or 0)
    tm_seq_cnt = _count_internal_tm_helices(core_seq) if core_seq else 0
    has_membrane_annot = tm_annot_cnt >= 1 or ("transmembrane" in keywords_low) or ("multi-pass membrane" in locs_low)
    has_membrane_name = bool(re.search(
        r"\b(?:rhodopsin|bacteriorhodopsin|aquaporin|(?:potassium|sodium|calcium|chloride|ion|urea)\s+channel|"
        r"kcsa|abc\s+transporter|permease|g-?protein\s+coupled\s+receptor|gpcr|adrenergic\s+receptor|"
        r"(?:transmembrane|outer[-\s]+membrane|integral\s+membrane|membrane|tm)[-\s]+(?:beta|β)[-\s]*barrel|"
        r"transmembrane\s+(?:nanopore|pore|channel|protein)|outer[-\s]+membrane\s+(?:protein|porin)|"
        r"porin|(?!ompr\b)omp[a-z]|omp85|vdac|bama|tolc)\b",
        search_text,
    ))
    has_membrane_seq = (
        not has_uniprot_annot
        and ((tm_seq_cnt >= 2 and (tm_seq_cnt * 19.0 / core_len) >= 0.10 and gravy >= -0.10) or gravy >= 0.45)
    )
    if has_membrane_annot or has_membrane_name or has_membrane_seq:
        categories.append("membrane_integral")

    # --- K (Evaluated Before Redox). ACTIVE PROTEASES (Autolysis & Self-Inhibition Guard) ---
    protease_class: Optional[str] = None
    if "metalloprotease" in keywords_low or bool(re.search(r"\b(?:metalloproteinase|metalloprotease|mmp-?\d+|thermolysin|collagenase|stromelysin|gelatinase|astacin|serralysin|carboxypeptidase\s*[ab])\b", search_text)):
        protease_class = "metallo"
    elif "thiol protease" in keywords_low or acc == "P04517" or bool(re.search(r"\b(?:tev\s+protease|nuclear\s+inclusion\s+protein\s+a|3c\s+protease|caspase|papain|cathepsin\s+[blks]|calpain|main\s+protease|3clpro)\b", search_text)):
        protease_class = "cysteine"
    elif "serine protease" in keywords_low or acc == "P00734" or bool(re.search(r"\b(?:trypsin|chymotrypsin|elastase|thrombin|prothrombin|subtilisin|enteropeptidase|proteinase\s+k|coagulation\s+factor\s+x)\b", search_text)):
        protease_class = "serine"
    elif "aspartyl protease" in keywords_low or bool(re.search(r"\b(?:pepsin|renin|hiv-?1\s+protease|cathepsin\s+d|bace1)\b", search_text)):
        protease_class = "aspartic"
    if protease_class:
        categories.append("active_protease")

    # --- H. REDOX ARCHITECTURE: DISULFIDE-BONDED SECRETED vs. CYTOSOLIC FREE-THIOL PROTEINS ---
    disulf_annot_cnt = int(annot.get("disulfide_count") or 0)
    is_cytosolic_cys_protease = (
        acc == "P04517"
        or (protease_class == "cysteine" and "secreted" not in keywords_low)
    )
    has_disulf_name = bool(re.search(
        r"\b(?:lysozyme|serum\s+albumin|albumin|immunoglobulin|antibody|igg\d*|fab|scfv|nanobody|"
        r"insulin|ribonuclease\s+a|rnase\s+a|trypsin|chymotrypsin|elastase|prothrombin|thrombin|"
        r"interleukin|interferon|erythropoietin|bone\s+morphogenetic\s+protein)\b",
        search_text,
    ))
    has_secreted_cys = (
        ("secreted" in keywords_low or "extracellular" in locs_low or bool(annot.get("has_signal_peptide")))
        and cys_count >= 2
    )
    # High cysteine density (>= 4.5% Cys & >= 6 Cys) on unannotated non-metalloprotein sequences (e.g. raw HEWL/BSA FASTA)
    has_high_cys_density = (
        not has_uniprot_annot
        and not is_metalloprotein
        and not is_cytosolic_cys_protease
        and "membrane_integral" not in categories
        and cys_count >= 6
        and (cys_count / core_len) >= 0.045
    )
    if (not is_cytosolic_cys_protease) and (
        acc in {"P00698", "P61626", "P02769", "P00734"}
        or disulf_annot_cnt >= 1
        or ("disulfide bond" in keywords_low)
        or has_disulf_name
        or has_secreted_cys
        or has_high_cys_density
    ):
        categories.append("disulfide_structural")
    elif (
        is_cytosolic_cys_protease
        or acc in {"P09211", "P08263", "P09488", "P08515", "P04637"}
        or bool(re.search(r"\b(?:deubiquitinase|tyrosine-?protein\s+phosphatase|ptp1b)\b", search_text))
        or (cys_count >= 3 and "secreted" not in keywords_low)
    ):
        categories.append("cytosolic_free_thiol")

    # --- I. NUCLEIC-ACID-BINDING PROTEINS (Host DNA/RNA Co-Purification Risk) ---
    if (
        acc in {"P04637", "P01106", "P0A707"}
        or bool({"dna-binding", "rna-binding", "ribonucleoprotein"} & keywords_low)
        or bool(re.search(
            r"\b(?:cas9|cas12[a-z]?|cas13[a-z]?|crispr|histone|transcription\s+factor|homeobox|"
            r"helicase|topoisomerase|gyrase|dna\s+polymerase|rna\s+polymerase|dna\s+ligase|rna\s+ligase|"
            r"ribosomal\s+protein|initiation\s+factor|reverse\s+transcriptase|integrase|argonaute|"
            r"cellular\s+tumor\s+antigen\s+p53|myc\s+proto-oncogene)\b",
            search_text,
        ))
    ):
        categories.append("nucleic_acid_binding")

    # --- J. ORGANIC COFACTOR HOLOENZYMES (PLP Vitamin B6 & Flavin FAD/FMN) ---
    if (
        "pyridoxal phosphate" in keywords_low
        or "CHEBI:597326" in all_chebis
        or "pyridoxal" in metal_blob
        or bool(re.search(
            r"\b(?:aminotransferase|transaminase|(?:glutamate|ornithine|lysine|histidine|dopa|aromatic-l-amino-acid)\s+decarboxylase|"
            r"serine\s+hydroxymethyltransferase|cystathionine|tryptophan\s+synthase|glycogen\s+phosphorylase)\b",
            search_text,
        ))
    ):
        categories.append("plp_dependent")

    if (
        "flavoprotein" in keywords_low
        or bool({"CHEBI:57692", "CHEBI:58210", "CHEBI:57618"} & all_chebis)
        or bool(re.search(r"\bfad\b|\bfmn\b|\bflavin\b", metal_blob))
        or bool(re.search(
            r"\b(?:glucose\s+oxidase|d-amino-acid\s+oxidase|glutathione\s+reductase|thioredoxin\s+reductase|"
            r"flavodoxin|monoamine\s+oxidase|cryptochrome|ferredoxin\s*[-\u2013]+\s*nadp|cytochrome\s+p450\s+reductase)\b",
            search_text,
        ))
    ):
        categories.append("flavin_dependent")

    # --- L. INTRINSICALLY DISORDERED PROTEINS (IDPs) ---
    # True IDPs are strongly depleted in bulky aromatic fold-core residues (Trp/Tyr/Phe typically < 6–7%),
    # distinguishing them from polar, aromatic-rich folded de novo beta-barrels (e.g., 6D0T with 12.6% aromatics).
    if (
        acc == "P02686"
        or bool(re.search(r"\b(?:synuclein|microtubule-associated\s+protein\s+tau|myelin\s+basic\s+protein)\b", search_text))
        or (core_len >= 80 and gravy < -0.85 and aliphatic_idx < 60.0 and aromatic_pct < 7.0)
    ):
        categories.append("idp_disordered")

    avoid_imidazole_imac = is_metalloprotein or is_o2_sensitive
    caution_imidazole_imac = is_heme_protein and not avoid_imidazole_imac
    avoid_reducing_agents = "disulfide_structural" in categories
    require_reducing_agents = ("cytosolic_free_thiol" in categories or "fe_s_cluster" in categories) and not avoid_reducing_agents
    avoid_tris_buffer = "plp_dependent" in categories
    avoid_phosphate_buffer = needs_calcium or ("Mn²⁺ Cofactor" in metal_labels)
    requires_detergent_micelles = "membrane_integral" in categories

    metal_summary = ", ".join(metal_labels) if metal_labels else "catalytic metal"

    # --- M. DETECT SPECIFIC TAG <-> PROTEIN BIOCHEMICAL CONFLICTS ---
    tag_conflict: Optional[Dict[str, str]] = None
    if has_tag and "his" in tag_low and avoid_imidazole_imac:
        if "ni_metalloenzyme" in categories:
            tag_conflict = {
                "severity": "critical",
                "headline": f"🛑 Metalloprotein–IMAC Conflict: Do NOT Use Ni-NTA & Imidazole for {effective_tag}-Tagged {metal_summary} Enzyme!",
                "explanation": (
                    f"This protein is a **{metal_summary} metalloenzyme** (like Urease). Running it on a **Ni-NTA IMAC** column "
                    "causes two severe biochemical failures: **(1)** apo/surface metal-binding pockets & accessory sites chelate and strip "
                    "Ni²⁺ off the column resin (causing Ni²⁺ leaching & protein precipitation), and **(2)** wash/elution **Imidazole (20–500 mM)** "
                    "directly coordinates and competes with the protein-bound Ni²⁺ center—inhibiting catalysis or stripping the active site into inactive apoprotein!"
                ),
                "better_strategy": (
                    "Bypass Ni-NTA & Imidazole! Purify natively using **EDTA-free Ion-Exchange (HiTrap Q / SP) + HIC + SEC** in a non-chelating "
                    "Good's buffer (`HEPES` or `MOPS`, **0 mM EDTA**, **0 mM Imidazole**), OR switch the affinity tag to **Twin-Strep-tag / Strep-tag II** "
                    "or **MBP-tag** run strictly in **EDTA-free** buffer."
                ),
            }
        elif "fe_s_cluster" in categories:
            tag_conflict = {
                "severity": "critical",
                "headline": f"🛑 [Fe–S] Metalloenzyme Conflict: Avoid Aerobic Ni-NTA & Imidazole for {effective_tag}-Tagged Construct!",
                "explanation": (
                    "Iron–sulfur (`[2Fe-2S]` / `[4Fe-4S]`) clusters are destroyed by dissolved O₂, disrupted by millimolar **Imidazole**, "
                    "and poisoned when leached **Ni²⁺** displaces Fe in Cys-coordination pockets (while DTT reduces Ni-NTA resin brown)."
                ),
                "better_strategy": (
                    "Purify inside an **anaerobic glovebox (< 2 ppm O₂)** using **EDTA-free Strep-Tactin** or **anaerobic Anion-Exchange (HiTrap Q / DEAE) + SEC** "
                    "in N₂-sparged `HEPES`/`Tris` + `1–2 mM DTT` or `Sodium Dithionite`, monitoring cluster integrity at `A410/A280`."
                ),
            }
        else:
            tight_zn_note = (
                " *(Note: ultra-tight buried Zn²⁺ sites such as Carbonic Anhydrase II, Kd ~10⁻¹² M, can occasionally tolerate rapid IMAC with low-imidazole washes and immediate desalting—verify metal occupancy & activity if attempted.)*"
                if "zn_metalloenzyme" in categories and "cu_mn_co_mo_metalloenzyme" not in categories else ""
            )
            tag_conflict = {
                "severity": "critical",
                "headline": f"🛑 Metalloprotein–IMAC Conflict: Avoid Ni-NTA & Imidazole for {effective_tag}-Tagged {metal_summary} Protein!",
                "explanation": (
                    f"This protein coordinates **{metal_summary}**. On **Ni-NTA**, leached Ni²⁺ substitutes into native metal sites via the "
                    "Irving–Williams series (**mismetallation**, permanently killing enzymatic activity) or protein sites strip metal from the column, "
                    f"while **20–500 mM Imidazole** competes with and extracts bound **{metal_summary}** into inactive apoprotein!{tight_zn_note}"
                ),
                "better_strategy": (
                    "Bypass Ni-NTA & Imidazole! Purify via **EDTA-free Ion-Exchange + HIC + SEC** in non-chelating `HEPES`/`MOPS` buffer, "
                    "OR use **Twin-Strep-tag**, **MBP-tag**, or **FLAG-tag** in **EDTA-free** buffer (never add EDTA, EGTA, or Citrate!)."
                ),
            }
    elif has_tag and "his" in tag_low and caution_imidazole_imac:
        tag_conflict = {
            "severity": "warning",
            "headline": f"⚠️ Heme–Imidazole Coordination Caution ({effective_tag} on Hemoprotein)!",
            "explanation": (
                "High **Imidazole (50–500 mM)** in standard Ni-NTA elution binds directly to the **6th axial coordination site of heme iron** "
                "(shifting the Soret band ~410 ➜ ~426 nm and competitively inhibiting P450s, peroxidases, and globins)."
            ),
            "better_strategy": (
                "Prefer **native IEX/HIC** or **EDTA-free Strep/MBP affinity**; if using IMAC (`TALON Co²⁺` / `Ni-NTA`), elute via "
                "**imidazole-free on-column TEV/3C protease cleavage** or immediately desalt out Imidazole on SEC and verify the heme **Soret/A280 (Rz)** ratio!"
            ),
        }
    elif has_tag and "gst" in tag_low and avoid_reducing_agents:
        tag_conflict = {
            "severity": "warning",
            "headline": "⚠️ Disulfide–Glutathione Conflict: GST Elution Risks Reducing Structural S–S Bridges!",
            "explanation": (
                "Standard GST purification uses `1–5 mM DTT` in binding buffer and `10–20 mM Reduced L-Glutathione (GSH)` for elution, "
                "which can reduce and scramble essential structural disulfide (`S–S`) bonds in secreted/extracellular proteins."
            ),
            "better_strategy": (
                "Omit DTT completely, keep GSH elution brief at 4°C followed by immediate desalting, OR prefer a redox-inert tag "
                "such as **Twin-Strep-tag**, **FLAG-tag**, or **reductant-free HisTrap**."
            ),
        }

    # Compact UI pills summarizing detected biochemical traits
    profile_badges: List[str] = []
    if is_metalloprotein:
        profile_badges.append(f"🪙 Metalloprotein ({metal_labels[0]})")
        profile_badges.append("🚫 No Ni-NTA / Imidazole / EDTA")
    elif is_heme_protein:
        profile_badges.append("🩸 Heme Cofactor (Avoid High Imidazole)")
    elif needs_calcium:
        profile_badges.append("🦴 Ca²⁺-Dependent (No EDTA / No Phosphate)")
    elif needs_magnesium:
        profile_badges.append("⚡ Mg²⁺ Cofactor (No EDTA)")

    if is_o2_sensitive:
        profile_badges.append("🫧 O₂-Sensitive (Anaerobic)")
    if requires_detergent_micelles:
        profile_badges.append("🧴 Integral Membrane (Detergent Required)")
    if avoid_reducing_agents:
        profile_badges.append("🔗 Disulfide-Stabilized (Omit DTT/TCEP)")
    if "plp_dependent" in categories:
        profile_badges.append("💊 PLP Cofactor (Avoid Tris!)")
    if "flavin_dependent" in categories:
        profile_badges.append("💛 Flavin FAD/FMN Cofactor")
    if "nucleic_acid_binding" in categories:
        profile_badges.append("🧬 DNA/RNA-Binding (1 M NaCl Wash)")

    return {
        "categories": categories,
        "effective_tag": effective_tag,
        "core_sequence": core_seq,
        "metal_labels": metal_labels,
        "metal_summary": metal_summary,
        "is_metalloprotein": is_metalloprotein,
        "is_heme_protein": is_heme_protein,
        "is_o2_sensitive": is_o2_sensitive,
        "is_chelator_sensitive": is_chelator_sensitive,
        "needs_calcium": needs_calcium,
        "needs_magnesium": needs_magnesium,
        "avoid_imidazole_imac": avoid_imidazole_imac,
        "caution_imidazole_imac": caution_imidazole_imac,
        "avoid_reducing_agents": avoid_reducing_agents,
        "require_reducing_agents": require_reducing_agents,
        "avoid_tris_buffer": avoid_tris_buffer,
        "avoid_phosphate_buffer": avoid_phosphate_buffer,
        "requires_detergent_micelles": requires_detergent_micelles,
        "protease_class": protease_class,
        "tag_conflict": tag_conflict,
        "profile_badges": profile_badges,
    }


def advise_hic_strategy(
    metrics: Dict[str, Any],
    working_ph: float,
    net_charge: float,
    protein_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates protein suitability and exact conditions for Hydrophobic Interaction Chromatography (HIC)
    per the Cytiva HIC Handbook, while honoring membrane-protein detergent rules, Ca2+/phosphate
    precipitation avoidance, and metalloprotein IMAC-bypass guidance.
    """
    prof = protein_profile or {}
    gravy = float(metrics["gravy"])
    aliphatic = float(metrics["aliphatic_index"])
    pi = float(metrics["pi"])
    length = max(1, int(metrics["length"]))
    cnts = metrics["counts"]

    aromatic_pct = (cnts["aromatic_residues"] / length) * 100.0
    hydrophobic_pct = (cnts["hydrophobic_residues"] / length) * 100.0
    near_pi = (abs(working_ph - pi) < 0.9) or (abs(net_charge) < 2.2)

    # 1. Decide Priority / When HIC is Required
    if prof.get("requires_detergent_micelles") or gravy > 0.35:
        status_badge = "🚫 Avoid Aqueous HIC (Membrane / Very Hydrophobic)" if prof.get("requires_detergent_micelles") else "⚠️ Caution: Membrane / Very Hydrophobic"
        is_required = False
        when_why = (
            "Integral membrane proteins require detergent micelles (e.g., DDM / LMNG) to stay soluble; high Ammonium Sulfate causes "
            "detergent phase-separation/aggregation and transmembrane helices bind irreversibly to HIC resins! Use Detergent Affinity + IEX + Detergent SEC instead."
            if prof.get("requires_detergent_micelles")
            else (
                f"GRAVY is positive ({gravy:+.2f}), typical of transmembrane or poorly water-soluble proteins. "
                "Standard HIC resins risk irreversible binding without detergents—prefer Affinity + Detergent SEC, or screen Cytiva's mildest HiTrap Butyl-S FF resin."
            )
        )
    elif near_pi:
        status_badge = "⭐ REQUIRED HERE (Rescues Purification Near pI!)"
        is_required = True
        when_why = (
            f"At pH {working_ph:.1f}, your protein is close to its pI ({pi:.2f}) with weak net charge ({net_charge:+.1f}), "
            "where Ion-Exchange (IEX) fails! HIC separates by surface hydrophobic patches in kosmotropic salt—independent of net charge—making HIC essential here."
        )
    elif "idp_disordered" in (prof.get("categories") or []):
        status_badge = "🔹 Secondary Option (IDP Lacks Compact Hydrophobic Core)"
        is_required = False
        when_why = (
            f"Intrinsically disordered proteins (GRAVY {gravy:+.2f}) have few stable hydrophobic patches and can nucleate fibrils in high Ammonium Sulfate. "
            "Prefer high-resolution Ion-Exchange (HiTrap Q / SP) + SEC first!"
        )
    elif gravy >= -0.35 or (gravy >= -0.58 and (aromatic_pct >= 7.5 or aliphatic >= 78.0 or hydrophobic_pct >= 38.0)):
        status_badge = "✅ Strongly Recommended Orthogonal Method"
        is_required = True
        upstream_note = (
            "especially right after Ammonium Sulfate fractionation or high-salt Ion-Exchange elution with ZERO dialysis needed!"
            if (prof.get("avoid_imidazole_imac") or prof.get("caution_imidazole_imac"))
            else "especially right after Ammonium Sulfate precipitation or high-salt IMAC/IEX elution with ZERO dialysis needed!"
        )
        flavin_caution = (
            " *(Note: monitor yellow FAD/FMN retention at A450, as >1.3 M salt can partially strip non-covalent flavins.)*"
            if "flavin_dependent" in (prof.get("categories") or []) else ""
        )
        when_why = (
            f"Your protein has an ideal amphipathic balance (GRAVY {gravy:+.2f}, {aromatic_pct:.1f}% aromatic Trp/Tyr/Phe, "
            f"{hydrophobic_pct:.1f}% hydrophobic residues). HIC gives crisp orthogonal resolution—{upstream_note}{flavin_caution}"
        )
    else:
        status_badge = "🔹 Optional Secondary Method (Very Polar Protein)"
        is_required = False
        when_why = (
            f"Your protein is highly hydrophilic (GRAVY {gravy:+.2f}, aromatic content {aromatic_pct:.1f}%). "
            "Ion-Exchange (IEX) is usually easier first, though HIC works well using a stronger C8 Octyl ligand with 1.5–1.8 M Ammonium Sulfate."
        )

    # 2. Select Optimal HIC Resin Ligand (Cytiva series: Butyl-S < Butyl HP [C4] < Phenyl HP < Octyl FF [C8])
    alkaline_salt_note = working_ph > 8.0
    if gravy > -0.22 or hydrophobic_pct > 42.0:
        recommended_resin = "HiTrap Butyl HP / Capto Butyl (Mild C4 Alkyl Ligand)"
        start_salt = (
            "0.8 – 1.0 M Sodium Sulfate [Na2SO4] (preferred at pH > 8.0)"
            if alkaline_salt_note else "0.8 – 1.0 M Ammonium Sulfate [(NH4)2SO4]"
        )
        ligand_reason = (
            "In Cytiva's Sepharose HIC series (Butyl-S < Butyl HP < Phenyl HP < Octyl FF), a MILD 4-carbon Butyl ligand "
            "prevents overly tight binding of hydrophobic proteins so your target elutes cleanly in its native folded state."
        )
    elif gravy >= -0.62 or aromatic_pct >= 7.0:
        recommended_resin = "HiTrap Phenyl HP / Capto Phenyl ImpRes (Mixed Aromatic/Hydrophobic Ligand)"
        start_salt = (
            "1.0 – 1.2 M Sodium Sulfate [Na2SO4] (preferred at pH > 8.0)"
            if alkaline_salt_note else "1.2 – 1.5 M Ammonium Sulfate [(NH4)2SO4]"
        )
        ligand_reason = (
            f"In Cytiva's Sepharose series (Butyl HP < Phenyl HP < Octyl FF), Phenyl Sepharose HP combines moderate hydrophobicity "
            f"with π–π aromatic selectivity ({aromatic_pct:.1f}% Phe/Trp/Tyr in your protein)—the gold-standard HIC choice for globular enzymes (like Lysozyme and GFP)."
        )
    else:
        recommended_resin = "HiTrap Octyl FF / Capto Octyl (Strong C8 Alkyl Ligand)"
        start_salt = (
            "1.2 – 1.35 M Sodium Sulfate [Na2SO4] or 3.0 M NaCl (at pH > 8.0)"
            if alkaline_salt_note else "1.5 – 1.8 M Ammonium Sulfate [(NH4)2SO4]"
        )
        ligand_reason = (
            "Because this protein is very polar, a longer 8-carbon Octyl chain (stronger than Butyl C4 and Phenyl HP on Sepharose) "
            "plus higher kosmotropic salt is needed to promote binding."
        )

    if prof.get("avoid_phosphate_buffer"):
        hic_buf_name = f"50 mM HEPES (pH {min(8.0, max(6.8, working_ph)):.1f} — avoid Sodium Phosphate so Ca²⁺/Mn²⁺ cofactors never precipitate!)"
    elif working_ph < 5.8:
        hic_buf_name = f"50 mM Sodium Acetate (pH {working_ph:.1f}; or run HIC at neutral pH 7.0 in 50 mM Sodium Phosphate)"
    elif working_ph <= 8.0:
        hic_buf_name = f"50 mM Sodium Phosphate or HEPES (pH {working_ph:.1f}, EDTA-free)" if prof.get("is_chelator_sensitive") else f"50 mM Sodium Phosphate (pH {working_ph:.1f})"
    elif prof.get("avoid_tris_buffer"):
        hic_buf_name = f"50 mM Bicine or Sodium Borate/HEPES (pH {working_ph:.1f} — avoid Tris with PLP enzymes & use Na2SO4 above pH 8.0)"
    else:
        hic_buf_name = (
            f"50 mM Tris-HCl (pH {working_ph:.1f} — avoid (NH4)2SO4 above pH 8.0 as NH4+ [pKa 9.25] releases NH3; "
            "use Na2SO4 at room temp OR run HIC at pH 7.0 in 50 mM Sodium Phosphate)"
        )

    elution_protocol = (
        f"Bind in {hic_buf_name} + {start_salt}. "
        "Elute with a linear REVERSE (decreasing) salt gradient down to 0 M kosmotropic salt over 15 column volumes."
    )

    student_analogy = (
        "Think of HIC as the exact opposite of Ion-Exchange! High kosmotropic salt (Ammonium Sulfate) 'keeps water busy' hydrating salt ions, "
        "nudging exposed oil-like patches on your protein onto the column ligands. Lowering salt back to zero lets water re-coat the protein so it elutes gently!"
    )

    return {
        "status_badge": status_badge,
        "is_required_or_strong": is_required,
        "gravy": round(gravy, 3),
        "aliphatic_index": round(aliphatic, 1),
        "aromatic_pct": round(aromatic_pct, 1),
        "hydrophobic_pct": round(hydrophobic_pct, 1),
        "when_why": when_why,
        "recommended_resin": recommended_resin,
        "starting_salt": start_salt,
        "ligand_reason": ligand_reason,
        "elution_protocol": elution_protocol,
        "student_analogy": student_analogy
    }


def recommend_full_purification_workflow(
    metrics: Dict[str, Any],
    working_ph: float,
    net_charge: float,
    tag_detected: str = "None",
    sequence: str = "",
    protein_name: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    construct_parts: Optional[List[Tuple[str, str]]] = None,
    protein_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Builds a context-aware, textbook-grade 3-Step Purification Roadmap (Capture -> Intermediate -> Polishing)
    and evaluates ALL 7 major protein purification methods per Cytiva CIPP principles—adapting wisely to
    metalloproteins (Ni2+, Zn2+, Cu, Mn, Co, Mo, Fe-S), hemoproteins, Ca2+/Mg2+ cofactors, disulfide bridges,
    integral membrane topology, nucleic-acid binding, PLP/flavin cofactors, proteases, and IDPs.
    """
    prof = protein_profile or infer_protein_biochemical_profile(
        sequence=sequence,
        protein_name=protein_name,
        metadata=metadata,
        metrics=metrics,
        tag_detected=tag_detected,
        construct_parts=construct_parts,
    )
    cats = set(prof["categories"])

    mw_kda = float(metrics["mw_kda"])
    pi = float(metrics["pi"])
    gravy = float(metrics["gravy"])
    cys_count = int(metrics["counts"]["cysteines"])
    instab = float(metrics["instability_index"])

    effective_tag = prof["effective_tag"] if (not tag_detected or tag_detected.lower() in ("none", "n/a", "")) else tag_detected
    has_tag = bool(effective_tag and effective_tag.lower() not in ("none", "n/a", ""))
    tag_low = (effective_tag or "").lower()

    iex = advise_iex_strategy(pi, working_ph, net_charge)
    # Annotate IEX buffer table notes with active-protein cofactor/chelator warnings
    adapted_buffers: List[Dict[str, Any]] = []
    for b in iex["recommended_buffers"]:
        b_copy = dict(b)
        b_nm = b_copy["name"].lower()
        if prof["is_chelator_sensitive"] and "citrate" in b_nm:
            b_copy["notes"] = f"🚫 AVOID for {prof['metal_summary']} proteins — Citrate chelates & strips divalent metal cofactors! ({b_copy['notes']})"
        elif prof["is_chelator_sensitive"] and "bicine" in b_nm:
            b_copy["notes"] = f"⚠️ Caution: Bicine weakly coordinates divalent metals; prefer HEPES/MOPS/CHES for {prof['metal_summary']} proteins."
        elif prof["avoid_tris_buffer"] and ("tris" in b_nm):
            b_copy["notes"] = "🚫 AVOID for PLP-dependent enzymes — primary/amine buffer forms Schiff-base adduct with PLP! Use HEPES or Phosphate."
        elif prof["avoid_phosphate_buffer"] and "phosphate" in b_nm:
            b_copy["notes"] = "🚫 AVOID when Ca²⁺/Mn²⁺ is required — orthophosphate precipitates insoluble metal phosphate!"
        elif prof["is_chelator_sensitive"] and any(g in b_nm for g in ("hepes", "mops", "mes", "pipes")):
            b_copy["notes"] = f"⭐ Ideal non-chelating Good's buffer for {prof['metal_summary']} metalloproteins! {b_copy['notes']}"
        adapted_buffers.append(b_copy)
    iex["recommended_buffers"] = adapted_buffers

    hic = advise_hic_strategy(metrics, working_ph, net_charge, protein_profile=prof)

    # Helper strings for buffer components tailored to protein biochemistry
    base_buf_chem = "HEPES" if (prof["is_chelator_sensitive"] or prof["avoid_tris_buffer"]) else "Tris-HCl"
    edta_clause = "0 mM EDTA (EDTA-FREE to protect bound metal!)" if prof["is_chelator_sensitive"] else "1 mM EDTA"
    na_wash_clause = " Include a **0.8–1.0 M NaCl high-salt wash** on-column to strip co-purifying host DNA/RNA (`A260/A280 < 0.60`)." if "nucleic_acid_binding" in cats else ""
    det_clause = " Include **0.03% (w/v) DDM** (or `0.005% LMNG + 0.001% CHS`) in all buffers after `1% DDM` membrane solubilization." if prof["requires_detergent_micelles"] else ""
    anaerobic_clause = " Perform **anaerobically (< 2 ppm O₂)** in N₂/Ar-sparged degassed buffer + `1–2 mM DTT / Dithionite`." if prof["is_o2_sensitive"] else ""
    imac_reductant_note = (
        " *(Omit DTT/TCEP/β-ME to preserve native disulfide bridges!)*"
        if prof["avoid_reducing_agents"]
        else (" *(Use 0.5 mM TCEP rather than >1 mM DTT on Ni-NTA.)*" if prof["require_reducing_agents"] else "")
    )

    imac_bypassed = False

    # --- STEP 1: CAPTURE (WISE PROTEIN-AWARE ROUTING) ---
    if prof["requires_detergent_micelles"] and not has_tag:
        step1_title = "Membrane Solubilization (1% DDM / LMNG) + Detergent Ion-Exchange Capture"
        step1_detail = (
            "Solubilize isolated membranes in `1% (w/v) DDM` (or `LMNG + CHS`) at 4°C, clear unsolubilized debris at `100,000 × g`, "
            f"and capture in `0.03% DDM` on {'HiTrap Q (AEX)' if net_charge < 0 else 'HiTrap SP (CEX)'} (never use Ammonium Sulfate salting-out!)."
        )
    elif has_tag:
        if "his" in tag_low and prof["avoid_imidazole_imac"]:
            # CRITICAL FIX: NEVER recommend Ni-NTA + Imidazole for Urease or other metalloproteins / Fe-S enzymes!
            imac_bypassed = True
            if abs(net_charge) >= 2.2:
                native_cap = "HiTrap Q (Anion Exchange)" if net_charge < 0 else "HiTrap SP (Cation Exchange)"
                native_proto = f"capture clarified lysate directly on **{native_cap}** in EDTA-free `25–50 mM {base_buf_chem}` (pH {working_ph:.1f}, `20–30 mM NaCl`)"
            else:
                native_cap = f"{hic['recommended_resin'].split('(')[0].strip()} (HIC)"
                native_proto = f"capture on **{native_cap}** in EDTA-free `{base_buf_chem}` + `{hic['starting_salt']}`"
            step1_title = f"🛑 Bypass Ni-NTA / Imidazole ➜ Metal-Safe {native_cap} (or EDTA-Free Strep/MBP)"
            step1_detail = (
                f"**Avoid Ni-NTA & Imidazole** for this **{prof['metal_summary']}** protein: protein metal sites chelate resin Ni²⁺ "
                "(or suffer Irving–Williams Ni²⁺ mismetallation), and `20–500 mM Imidazole` competes with/strips bound metal! "
                f"Instead, {native_proto}, OR re-clone with **Twin-Strep-tag / MBP-tag** in **EDTA-free** buffer.{anaerobic_clause}{det_clause}{na_wash_clause}"
            )
        elif "his" in tag_low and prof["caution_imidazole_imac"]:
            step1_title = f"⚠️ Heme-Safe Capture: {effective_tag} (On-Column Protease Elution) or Native IEX"
            step1_detail = (
                "High Imidazole (`250–500 mM`) coordinates axial heme Fe (shifting Soret & inhibiting catalysis)! "
                "Prefer **on-column TEV/3C cleavage** (0 mM Imidazole elution), **immediate SEC desalting**, or **EDTA-free Strep/IEX**; monitor `A_Soret/A280`."
                f"{det_clause}{na_wash_clause}"
            )
        elif "his" in tag_low:
            step1_title = f"Affinity Capture: Ni-NTA / HisTrap HP ({effective_tag})"
            step1_detail = (
                f"Bind clarified lysate in `50 mM {base_buf_chem}`, `300–500 mM NaCl`, `20 mM Imidazole`, **`0 mM EDTA`** (pH 7.4–8.0 — omit or remove EDTA before Ni-NTA so resin Ni²⁺ is not stripped); "
                f"wash at `40–50 mM Imidazole`; elute at `250–300 mM Imidazole`.{imac_reductant_note}{det_clause}{na_wash_clause}"
            )
        elif "strep" in tag_low:
            step1_title = f"Affinity Capture: Strep-Tactin / Strep-Tactin XT ({effective_tag})"
            step1_detail = (
                f"Bind in `100 mM {base_buf_chem}`, `150 mM NaCl`, **{edta_clause}** (pH 8.0); elute classic Strep-Tactin with "
                f"`2.5 mM D-Desthiobiotin` OR Strep-Tactin XT with `50 mM D-Biotin`.{anaerobic_clause}{det_clause}{na_wash_clause}"
            )
        elif "gst" in tag_low:
            if prof["avoid_reducing_agents"]:
                step1_title = "⚠️ Caution Affinity Capture: Glutathione Sepharose 4B (GST-Tag — GSH Risk on Disulfides!)"
                step1_detail = (
                    "Omit DTT from binding buffer and keep `10 mM Reduced L-Glutathione (GSH)` elution brief at 4°C followed by immediate desalting "
                    "so structural `S–S` disulfide bonds are not reduced/scrambled (prefer Twin-Strep or FLAG when cloning secreted proteins)."
                )
            else:
                buf_gst = "50 mM HEPES (pH 7.4), 150 mM NaCl" if prof["avoid_phosphate_buffer"] else "PBS (pH 7.4)"
                step1_title = "Affinity Capture: Glutathione Sepharose 4B (GST-Tag)"
                step1_detail = f"Bind in `{buf_gst}` + `1–2 mM DTT`; elute natively in `50 mM {base_buf_chem}`, `150 mM NaCl` + `10–20 mM Reduced L-Glutathione` (re-titrated to pH 8.0).{det_clause}{na_wash_clause}"
        elif "mbp" in tag_low:
            step1_title = "Affinity Capture: Amylose Resin / MBPTrap HP (MBP-Tag)"
            step1_detail = (
                f"Bind in `20 mM {base_buf_chem}` (pH 7.4), `200 mM NaCl`, **{edta_clause}**; "
                f"elute gently under native conditions with `10 mM D-Maltose`.{anaerobic_clause}{det_clause}{na_wash_clause}"
            )
        elif "flag" in tag_low:
            ca_note = " (use Ca²⁺-independent Anti-FLAG M2, NOT Ca²⁺/EDTA-dependent M1!)" if prof["is_chelator_sensitive"] else ""
            step1_title = f"Affinity Capture: Anti-FLAG M2 Resin ({effective_tag})"
            step1_detail = f"Bind in `50 mM {base_buf_chem}` (pH 7.4), `150 mM NaCl`{ca_note}; elute competitively under native conditions with `100–150 µg/mL 3× FLAG peptide`.{det_clause}{na_wash_clause}"
        else:
            step1_title = f"Affinity Capture ({effective_tag})"
            step1_detail = f"Capture recombinant tagged protein from clarified lysate in `{base_buf_chem}` ({edta_clause}).{det_clause}{na_wash_clause}"
    else:
        # Tagless Native Capture Logic (respecting metalloprotein/anaerobic/nucleic-acid/chelator constraints)
        edta_free_tag = " in EDTA-free buffer" if prof["is_chelator_sensitive"] else ""
        if "nucleic_acid_binding" in cats and net_charge >= 1.5:
            step1_title = "Tagless Capture: HiTrap Heparin HP (Biomimetic DNA-Mimic) or Cation Exchange"
            step1_detail = (
                f"Clear host nucleic acids (`0.8–1.0 M NaCl` / PEI cut), then capture on **HiTrap Heparin HP**{edta_free_tag} "
                "and elute with a `0.15 ➜ 1.5 M NaCl` gradient."
            )
        elif pi >= 8.4 and net_charge >= 4.0:
            step1_title = "Tagless Capture: Cation Exchange (HiTrap SP) or Heparin Sepharose"
            step1_detail = (
                f"Because your protein is strongly basic (pI {pi:.2f}, charge {net_charge:+.1f}), most host proteins are anionic and flow straight through! "
                f"Capture directly on SP Sepharose (CEX) or HiTrap Heparin HP{edta_free_tag}.{anaerobic_clause}"
            )
        elif prof["is_o2_sensitive"]:
            col_short = "HiTrap Q / DEAE (Anion Exchange)" if net_charge <= 0 else "HiTrap SP (Cation Exchange)"
            step1_title = f"Anaerobic Tagless Capture: {col_short} (< 2 ppm O₂)"
            step1_detail = (
                f"Load clarified anaerobic lysate onto **{col_short}** in N₂-sparged, EDTA-free `{base_buf_chem}` + `1–2 mM DTT / Dithionite` "
                "(avoid open-beaker Ammonium Sulfate stirring in air so `[Fe–S]` clusters do not oxidize!)."
            )
        elif abs(net_charge) >= 3.5:
            col_short = "HiTrap Q (Anion Exchange)" if net_charge < 0 else "HiTrap SP (Cation Exchange)"
            step1_title = f"Tagless Capture: {col_short} or (NH4)2SO4 Salting-Out"
            step1_detail = (
                f"Load low-conductivity clarified extract directly onto {col_short}{edta_free_tag} (`25 mM {base_buf_chem}`), "
                "or perform a gentle 35–65% Ammonium Sulfate cut."
            )
        else:
            step1_title = "Tagless Capture: Ammonium Sulfate (NH4)2SO4 Fractionation (40–70% Cut)"
            step1_detail = (
                f"Salt out bulk contaminants first{edta_free_tag}; redissolve your target pellet and load DIRECTLY onto a Hydrophobic Interaction (HIC) column with zero dialysis!"
            )

    # --- STEP 2: INTERMEDIATE ORTHOGONAL CLEANUP ---
    iex_short = "Anion Exchange (HiTrap Q)" if net_charge < 0 else "Cation Exchange (HiTrap SP)"
    if prof["requires_detergent_micelles"]:
        step2_title = f"Intermediate Cleanup: Detergent {iex_short} or Protease Tag-Cleavage"
        step2_detail = (
            f"Run {iex_short} in `0.03% DDM` (or `0.005% LMNG`)—never aqueous HIC!—or cleave the affinity tag (3C/TEV) "
            "and pass back over affinity resin (optional reconstitution into amphipols or lipid nanodiscs)."
        )
    elif "nucleic_acid_binding" in cats and abs(net_charge) >= 1.5:
        step2_title = f"Intermediate Cleanup: HiTrap Heparin HP OR {iex_short}"
        step2_detail = (
            f"Heparin Sepharose mimics the polyanionic DNA/RNA backbone—giving ultra-selective purification (`0.15 ➜ 1.2 M NaCl`) "
            f"while stripping residual host nucleic acids (`A260/A280 < 0.60`), or polish by charge on {iex_short}."
        )
    elif abs(net_charge) < 2.2 or abs(working_ph - pi) < 0.9:
        step2_title = f"Intermediate Cleanup: Hydrophobic Interaction — {hic['recommended_resin'].split('(')[0].strip()}"
        step2_detail = (
            f"Net charge ({net_charge:+.1f}) is weak near pI, so HIC (`{hic['starting_salt']}` ➜ `0 M` salt) is your best orthogonal step to remove co-eluting contaminants."
        )
    elif hic["is_required_or_strong"] and (not has_tag or imac_bypassed):
        step2_title = f"Intermediate Cleanup: Hydrophobic Interaction — {hic['recommended_resin'].split('(')[0].strip()}"
        step2_detail = (
            f"Separates by surface hydrophobicity (`{hic['starting_salt']}` ➜ `0 M` gradient)—completely orthogonal to charge-based Step 1 capture!"
        )
    elif hic["is_required_or_strong"] and has_tag:
        step2_title = f"Intermediate Cleanup: {iex_short} OR {hic['recommended_resin'].split('(')[0].strip()} (HIC)"
        eluate_src = "Step 1 affinity eluate" if (prof["caution_imidazole_imac"] or "his" not in tag_low) else "high-salt IMAC eluate"
        step2_detail = (
            f"Both work well! Use {iex_short} after diluting NaCl to `<30 mM`, OR adjust {eluate_src} into `{hic['starting_salt']}` for HIC with zero dialysis to strip misfolded/clipped species."
        )
    else:
        step2_title = f"Intermediate Cleanup: {iex_short}"
        step2_detail = f"High-resolution charge separation (Q = {net_charge:+.1f} at pH {working_ph:.1f}) using a `25 mM ➜ 1,000 mM NaCl` gradient."

    # --- STEP 3: FINAL POLISHING (SEC — ACCOUNTING FOR DETERGENT MICELLES, IDP RADIUS & COFACTORS) ---
    effective_sec_mw = mw_kda + 70.0 if prof["requires_detergent_micelles"] else (mw_kda * 1.6 if "idp_disordered" in cats else mw_kda)
    if effective_sec_mw < 45.0:
        sec_col = "Superdex 75 Increase 10/300 GL (or HiLoad 16/600 S75)"
        sec_window = "3 – 70 kDa"
    elif effective_sec_mw <= 400.0:
        sec_col = "Superdex 200 Increase 10/300 GL (or HiLoad 16/600 S200)"
        sec_window = "10 – 600 kDa"
    else:
        sec_col = "Superose 6 Increase 10/300 GL"
        sec_window = "5 kDa – 5 MDa"

    sec_tail_notes: List[str] = []
    if prof["requires_detergent_micelles"]:
        sec_tail_notes.append(f"includes ~70 kDa detergent micelle belt (`{mw_kda:.1f} kDa` protein + micelle ≈ `{effective_sec_mw:.0f} kDa` in `0.03% DDM` / `0.003% LMNG`)")
    if prof["is_chelator_sensitive"]:
        sec_tail_notes.append(f"run in **EDTA-free, Imidazole-free** `25–50 mM {base_buf_chem}` + `150 mM NaCl` to retain bound {prof['metal_summary']}")
    elif prof["avoid_tris_buffer"]:
        sec_tail_notes.append("run in `25–50 mM HEPES` (never Tris!) + `150 mM NaCl` + `10–20 µM PLP`")
    elif prof["avoid_reducing_agents"]:
        sec_tail_notes.append("omit DTT/TCEP so structural disulfide bridges remain oxidized")
    elif prof["require_reducing_agents"]:
        sec_tail_notes.append("include `0.5 mM TCEP` (or `1 mM DTT`) to protect free thiols")
    sec_extra_str = f" ({'; '.join(sec_tail_notes)})" if sec_tail_notes else ""

    step3_title = f"Final Polish: Size-Exclusion (SEC) on {sec_col.split('(')[0].strip()}"
    step3_detail = (
        f"Optimal for your {mw_kda:.1f} kDa protein ({sec_window} globular window). Removes soluble aggregates, resolves native oligomers, "
        f"and exchanges sample (<2% of column bed volume) into final storage/assay buffer (`≥150 mM NaCl`){sec_extra_str}."
    )

    # --- COMPLETE 7-METHOD EVALUATION MATRIX (TAILORED TO ACTIVE PROTEIN BIOCHEMISTRY) ---
    if prof["avoid_imidazole_imac"]:
        if has_tag and "his" in tag_low:
            m1_suit = "🚫 Avoid Ni-NTA / Imidazole"
            m1_resin = f"Bypass {effective_tag}/Ni-NTA ➜ Switch to Twin-Strep / MBP (EDTA-Free!)"
            m1_when = (
                f"Active-site/apo {prof['metal_summary']} pockets chelate resin Ni²⁺ (or suffer Ni²⁺ mismetallation) and 20–500 mM Imidazole "
                "competes with/strips the catalytic metal! Use EDTA-free Twin-Strep, MBP, FLAG, or native IEX/HIC instead."
            )
        elif has_tag:
            m1_suit = "⭐ Best Step 1 (EDTA-Free!)"
            m1_resin = f"{effective_tag} Resin (Strictly 0 mM EDTA / 0 mM Imidazole)"
            m1_when = f"Selective 1-step capture—MUST omit standard 1 mM EDTA from Strep/MBP buffers so bound {prof['metal_summary']} is never stripped!"
        else:
            m1_suit = "⚠️ Metal-Safe Tags Only"
            m1_resin = "Clone Twin-Strep, MBP, or FLAG (EDTA-Free — NEVER 6xHis / Ni-NTA!)"
            m1_when = (
                f"Do NOT fuse 6xHis or run Ni-NTA/Imidazole on {prof['metal_summary']} metalloproteins! "
                "If cloning an affinity tag, choose Twin-Strep or MBP in EDTA-free buffer."
            )
    elif prof["caution_imidazole_imac"] and has_tag and "his" in tag_low:
        m1_suit = "⚠️ Caution (Imidazole–Heme)"
        m1_resin = f"{effective_tag} (Use On-Column TEV/3C Cleavage or Immediate Desalting)"
        m1_when = "250–500 mM Imidazole coordinates axial heme Fe! Elute by protease cleavage (0 mM Imidazole) or switch to EDTA-free Strep/MBP."
    elif prof["avoid_reducing_agents"] and has_tag and "gst" in tag_low:
        m1_suit = "⚠️ Caution (GSH vs S–S Bonds)"
        m1_resin = "Glutathione Sepharose (Omit DTT; Brief 4°C GSH Elution)"
        m1_when = "10–20 mM Reduced Glutathione can scramble structural disulfides—Twin-Strep, FLAG, or reductant-free HisTrap is safer."
    else:
        m1_suit = "⭐ Best Step 1" if has_tag else "➕ Requires Tag"
        if has_tag:
            m1_resin = f"{effective_tag} Affinity Resin" + (" (EDTA-Free!)" if prof["is_chelator_sensitive"] else "")
        elif prof["is_heme_protein"]:
            m1_resin = "Prefer Twin-Strep, MBP, or Cleavable Tag (Avoid High Imidazole on Heme)"
        elif prof["avoid_reducing_agents"]:
            m1_resin = "Add Twin-Strep, FLAG, or Reductant-Free 6xHis (Avoid GST/DTT)"
        else:
            m1_resin = "Add Twin-Strep or 6xHis in sidebar prompt if cloning"
        m1_when = "Captures ~80–95% pure protein straight out of clarified cell lysate in one selective step."

    heparin_favored = (pi >= 8.3 and net_charge >= 3.5) or ("nucleic_acid_binding" in cats)
    all_methods = [
        {
            "Method": "1. Tag Affinity (Strep / MBP / FLAG / IMAC)",
            "Separates By": "Engineered Affinity Tag",
            "Suitability": m1_suit,
            "Resin / Reagent": m1_resin,
            "When to Use It": m1_when,
        },
        {
            "Method": "2. Heparin / Blue Pseudo-Affinity",
            "Separates By": "DNA-Mimic GAG / Dinucleotide Fold",
            "Suitability": "⭐ Premier Bio-Affinity" if "nucleic_acid_binding" in cats else ("✅ Great Tagless Option" if heparin_favored else "🔹 Specialized"),
            "Resin / Reagent": "HiTrap Heparin HP / HiTrap Blue HP",
            "When to Use It": (
                "Ideal for DNA/RNA-binding & basic proteins (Lysozyme, Cas9, Polymerases, p53, Transcription Factors; elute with 0.15 ➜ 1.5 M NaCl)."
                if heparin_favored
                else "Use for tagless DNA/RNA-binding proteins, coagulation factors (Heparin), or NAD+/ATP Rossmann-fold enzymes & albumin (Cibacron Blue)."
            ),
        },
        {
            "Method": "3. Ammonium Sulfate Salting-Out",
            "Separates By": "Bulk Solubility in Kosmotropic Salt",
            "Suitability": (
                "🚫 Avoid (Membrane Protein)" if prof["requires_detergent_micelles"]
                else ("⚠️ Avoid Aerobic Stirring" if prof["is_o2_sensitive"] else ("✅ Classic Native Step" if not has_tag else "🔹 Optional Bulk Step"))
            ),
            "Resin / Reagent": "Solid (NH4)2SO4 (typically 35% ➜ 70% saturation cut; 100% ≈ 4.1 M at 20°C)",
            "When to Use It": (
                "Avoid on integral membrane proteins—kosmotropic salt causes detergent micelle phase separation & irreversible aggregation!"
                if prof["requires_detergent_micelles"]
                else "Inexpensive classroom & native extract concentration step; redissolved pellet feeds DIRECTLY into HIC without dialysis!"
            ),
        },
        {
            "Method": "4. Ion-Exchange (IEX: AEX / CEX)",
            "Separates By": f"Net Surface Charge (Q = {net_charge:+.1f})",
            "Suitability": (
                "⭐ Highly Recommended" if abs(net_charge) >= 3.0
                else ("⚠️ Weak Near pI" if abs(net_charge) >= 1.5 else "❌ Avoid at This pH")
            ),
            "Resin / Reagent": iex["recommended_resins"].split(" or ")[0],
            "When to Use It": (
                f"Protein is {'negative ➜ Anion Exchange (Q)' if net_charge < 0 else 'positive ➜ Cation Exchange (SP)'} at pH {working_ph:.1f}"
                + (f" (use EDTA-free {base_buf_chem})." if prof["is_chelator_sensitive"] else ".")
                if abs(net_charge) >= 2.0
                else f"Net charge is too close to zero at pH {working_ph:.1f}; shift pH ≥1 unit from pI or use HIC instead!"
            ),
        },
        {
            "Method": "5. Hydrophobic Interaction (HIC)",
            "Separates By": f"Surface Hydrophobicity (GRAVY {gravy:+.2f})",
            "Suitability": hic["status_badge"].split("(")[0].strip(),
            "Resin / Reagent": hic["recommended_resin"],
            "When to Use It": hic["when_why"],
        },
        {
            "Method": "6. Hydroxyapatite / Multimodal (CHT / MMC)",
            "Separates By": "Ca²⁺ & PO₄³⁻ Sites / Charge + Hydrophobic",
            "Suitability": "⚠️ Caution (Metal/PO₄³⁻)" if prof["is_metalloprotein"] else "✅ Orthogonal Polish",
            "Resin / Reagent": "Capto MMC (Multimodal)" if prof["is_metalloprotein"] else "Ceramic Hydroxyapatite CHT [Ca10(PO4)6(OH)2] or Capto MMC",
            "When to Use It": (
                f"Prefer Capto MMC over CHT for {prof['metal_summary']} enzymes, as CHT's Ca²⁺/PO₄³⁻ lattice & phosphate gradients can perturb active-site metals."
                if prof["is_metalloprotein"]
                else "Salt-tolerant capture (Capto MMC) or phosphate-gradient CHT polish for antibodies, nucleic acid removal & clipped isoforms (avoid EDTA/citrate on CHT!)."
            ),
        },
        {
            "Method": "7. Size-Exclusion (SEC / Gel Filtration)",
            "Separates By": f"Hydrodynamic Size ({mw_kda:.1f} kDa)",
            "Suitability": "⭐ Essential Final Step",
            "Resin / Reagent": sec_col,
            "When to Use It": "Always run last on concentrated sample (0.5–2% of column bed volume) in ≥150 mM NaCl to remove aggregates and exchange into final buffer.",
        },
    ]

    # --- PRACTICAL, PROTEIN-SPECIFIC BIOCHEMICAL ALERTS ---
    alerts: List[str] = []
    if prof["is_metalloprotein"]:
        if "ni_metalloenzyme" in cats:
            alerts.append(
                f"🪙 **Nickel Metalloenzyme Protection ({prof['metal_summary']}):** Never use **Ni-NTA IMAC**, **Imidazole**, **EDTA**, **EGTA**, or **Citrate**! "
                "Apo/accessory sites chelate Ni²⁺ from IMAC beads, while millimolar Imidazole competes directly with active-site Ni²⁺ (and EDTA strips it). "
                "Use non-chelating Good's buffers (`HEPES`, `MOPS`, `MES`), **EDTA-free** protease inhibitors, and native IEX/HIC or EDTA-free Strep/MBP affinity."
            )
        elif "fe_s_cluster" in cats:
            alerts.append(
                "🫧 **Oxygen-Sensitive [Fe–S] Cluster Alert:** Purify strictly **anaerobically (< 2 ppm O₂)** using N₂/Ar-sparged degassed buffers + `1–2 mM DTT` "
                "(or `Sodium Dithionite`). Avoid **Ni-NTA**, **Imidazole**, and **EDTA** (which destroy/mismetallate `[Fe–S]` clusters), and track holoenzyme purity via `A410/A280`!"
            )
        else:
            alerts.append(
                f"🪙 **Metalloprotein Cofactor Protection ({prof['metal_summary']}):** Avoid **Ni-NTA IMAC** (leached Ni²⁺ causes irreversible Irving–Williams "
                f"**mismetallation** of native {prof['metal_summary']} active sites) and avoid **Imidazole**, **EDTA**, **EGTA**, `1,10-Phenanthroline`, and **Citrate** (which strip or inhibit "
                f"catalytic {prof['metal_summary']})! Use **EDTA-free** Good's buffers (`HEPES`, `MOPS`, `MES`) and EDTA-free protease inhibitors."
            )

    if prof["is_heme_protein"]:
        alerts.append(
            "🩸 **Hemoprotein / Soret Band Tip:** High **Imidazole (50–500 mM)** binds the 6th axial position of heme iron (shifting the Soret band ~410 ➜ ~426 nm "
            "and inhibiting catalysis). Avoid high-imidazole elution (use native IEX/HIC, EDTA-free Strep/MBP, or on-column protease cleavage), supplement `δ-ALA + Fe³⁺/hemin` "
            "during expression, and monitor holoprotein incorporation via the **Reinheitszahl (`Rz = A_Soret / A280`)**!"
        )

    if prof["needs_calcium"]:
        alerts.append(
            "🦴 **Ca²⁺ Cofactor & Phosphate Precipitation Alert:** Omit **EDTA / EGTA / Citrate**, include `1–2 mM CaCl2` if required for fold stability, "
            "and **NEVER combine Ca²⁺ with Sodium/Potassium Phosphate buffer** (precipitates insoluble `Ca3(PO4)2` chalk inside FPLC columns—use `HEPES` or `Tris-HCl` instead!)."
        )
    elif prof["needs_magnesium"] and not prof["is_metalloprotein"]:
        alerts.append(
            "⚡ **Mg²⁺ Cofactor Tip:** Omit **EDTA** from final polishing/assay buffers and include `1–5 mM MgCl2` in `HEPES` or `Tris-HCl` to stabilize nucleotide/catalytic binding."
        )

    if prof["requires_detergent_micelles"]:
        alerts.append(
            "🧴 **Integral Membrane Protein Protocol:** Keep a mild detergent **above its CMC** in EVERY buffer (e.g., extract in `1% w/v DDM`, purify in `0.03–0.05% DDM` "
            "or `0.005% LMNG + 0.001% CHS`, or reconstitute into nanodiscs). Never use aqueous Ammonium Sulfate salting-out or HIC!"
        )

    if prof["avoid_tris_buffer"]:
        alerts.append(
            "💊 **PLP (Vitamin B6) Cofactor Alert — Avoid Tris Buffer!** Primary amine buffers like **`Tris`** react covalently with the aldehyde of pyridoxal 5'-phosphate "
            "(forming a Schiff-base adduct that strips/quenches PLP). Always use **`HEPES` or `Phosphate`** + `10–50 µM PLP` and monitor the ~390–425 nm internal aldimine peak!"
        )

    if "flavin_dependent" in cats:
        alerts.append(
            "💛 **Flavoprotein (FAD / FMN) Tip:** Protect columns/fractions from bright light, avoid harsh chaotropes, optionally spike `10–50 µM FAD/FMN` before SEC "
            "if yellow cofactor partially dissociates during high-salt chromatography, and track holoenzyme ratio at `A450/A280`."
        )

    if "nucleic_acid_binding" in cats:
        alerts.append(
            "🧬 **Host DNA/RNA Co-Purification Guard:** Electropositive nucleic-acid-binding surfaces tightly pull down host DNA/RNA (`A260/A280 > 1.0`). "
            "Include a **`0.8–1.0 M NaCl` wash** during capture (or `HiTrap Heparin HP` chromatography) and verify **`A260/A280 < 0.60`** prior to SEC!"
        )

    if prof["protease_class"] == "serine":
        alerts.append(
            "✂️ **Active Serine Protease Caution:** If purifying catalytically active enzyme, **DO NOT add `PMSF` or `AEBSF`** (they irreversibly sulfonylate the catalytic Ser)! "
            "Purify at 4°C (or as inactive zymogen) to prevent autolysis."
        )
    elif prof["protease_class"] == "cysteine":
        alerts.append(
            "✂️ **Active Cysteine Protease Caution:** Do NOT add thiol-alkylating/epoxide inhibitors (`E-64`, `Iodoacetamide`, `NEM`), and include `0.5 mM TCEP` or `1 mM DTT` "
            "so the catalytic Cys thiol stays reduced and active!"
        )
    elif prof["protease_class"] == "metallo":
        alerts.append(
            "✂️ **Active Metalloprotease Caution:** Never add `EDTA`, `EGTA`, or `1,10-Phenanthroline` (use **EDTA-free** serine/cysteine protease inhibitor tablets only), "
            "and work at 4°C to prevent autoproteolysis!"
        )

    if prof["avoid_reducing_agents"]:
        alerts.append(
            f"🔗 **Disulfide Bridge Protection ({cys_count} Cysteines):** Structural `S–S` disulfide bonds stabilize this fold—**omit strong reducing agents (`DTT`, `TCEP`, `β-ME`)** "
            "from lysis, affinity, IEX, and SEC buffers so native disulfides stay intact!"
        )
    elif prof["require_reducing_agents"] and "fe_s_cluster" not in cats:
        alerts.append(
            f"🔗 **Free-Thiol Redox Tip ({cys_count} Cysteines):** Include `0.5 mM TCEP` (preferred on IMAC) or `1–2 mM DTT` (in IEX/SEC) to prevent unwanted air oxidation "
            "and intermolecular disulfide crosslinking."
        )
    elif cys_count >= 4:
        alerts.append(
            f"🔗 **Disulfide / Cysteine Alert ({cys_count} Cysteines detected):** If this is an extracellular/secreted protein (like Egg-White Lysozyme with 4 native disulfide bonds), "
            "omit strong reducing agents (DTT/TCEP/β-ME) so structural disulfide bridges stay intact! If it is a cytosolic protein with free thiols, include 0.5 mM TCEP or 1 mM DTT to prevent air oxidation."
        )

    ext_info = metrics.get("extinction") or {}
    if ext_info.get("molar_extinction_cystines") == 0:
        alerts.append(
            "🔭 **UV-280 Invisible Protein (No Trp/Tyr/S–S, ε280 = 0):** Common in short peptides & AI-designed (ProteinMPNN/RFdiffusion) scaffolds! "
            "Standard 280 nm FPLC traces will stay flat—monitor peptide-bond absorbance at **214 nm / 205 nm** "
            "(using low-UV buffer like Phosphate/MES rather than Tris/HEPES) or add a single C-terminal Trp/Tyr tag."
        )

    if abs(working_ph - pi) < 0.6:
        alerts.append(
            f"⚠️ **Isoelectric Precipitation Alert:** Your chosen pH ({working_ph:.1f}) is within 0.6 units of the protein's pI ({pi:.2f}). "
            "Adjust your buffer pH at least 1.0 unit higher or lower so molecules carry sufficient net charge to repel one another!"
        )
    if instab >= 40.0:
        inhib_type = "EDTA-free serine/cysteine" if prof["is_chelator_sensitive"] else ("non-PMSF" if prof["protease_class"] == "serine" else "EDTA-free")
        instab_tip = (
            f"🧊 **Cold-Room & Protease Inhibitor Tip:** Instability Index is {instab:.1f} (>40). Work at 4°C on ice, include 5–10% (v/v) Glycerol, and use {inhib_type} Protease Inhibitor Cocktail during cell lysis."
        )
        core_seq_wf = prof.get("core_sequence") or (sequence or "").strip().upper()
        if core_seq_wf:
            wf_core_len = len(core_seq_wf)
            wf_e_frac = core_seq_wf.count("E") / wf_core_len
            wf_ekr_frac = (core_seq_wf.count("E") + core_seq_wf.count("K") + core_seq_wf.count("R")) / wf_core_len
        else:
            wf_core_len = int(metrics.get("length") or 0)
            aa_pcts = metrics.get("aa_percentages") or {}
            wf_e_frac = float(aa_pcts.get("E", 0.0)) / 100.0
            wf_ekr_frac = (float(aa_pcts.get("E", 0.0)) + float(aa_pcts.get("K", 0.0)) + float(aa_pcts.get("R", 0.0))) / 100.0
        wf_name_low = " ".join(
            str(x) for x in (
                protein_name,
                (metadata or {}).get("protein_name", ""),
                (metadata or {}).get("identified_protein", ""),
            ) if x
        ).lower()
        is_denovo_saltbridge_ii = (
            (wf_core_len >= 40 and wf_ekr_frac >= 0.30 and wf_e_frac >= 0.11)
            or bool(re.search(r"\b(?:de[-\s]*novo|designed|proteinmpnn|mpnn|rfdiffusion|hallucinated)\b", wf_name_low))
        )
        if is_denovo_saltbridge_ii:
            instab_tip += (
                " (De Novo / ProteinMPNN Caveat: **Guruprasad's 1990 natural-proteome dipeptide index often flags "
                "Glu/Lys-rich AI-designed/thermostable helical bundles (`II ≥ 40`) even when experimentally hyper-stable**—"
                "verify stability via nanoDSF/CD rather than assuming instability.)"
            )
        alerts.append(instab_tip)

    return {
        "roadmap": [
            {"step": "Step 1 • Capture", "title": step1_title, "detail": step1_detail},
            {"step": "Step 2 • Intermediate (Orthogonal)", "title": step2_title, "detail": step2_detail},
            {"step": "Step 3 • Final Polishing", "title": step3_title, "detail": step3_detail},
        ],
        "iex": iex,
        "hic": hic,
        "all_methods": all_methods,
        "alerts": alerts,
        "protein_profile": prof,
        "tag_conflict": prof["tag_conflict"],
    }


def recommend_safe_buffer_preset_key(
    protein_profile: Optional[Dict[str, Any]],
    metrics: Optional[Dict[str, Any]] = None,
    working_ph: float = 7.4,
    net_charge: float = 0.0,
) -> str:
    """Returns the safest starter recipe template key in PRESET_RECIPE_TEMPLATES for the active protein."""
    prof = protein_profile or {}
    if (
        prof.get("is_metalloprotein")
        or prof.get("is_heme_protein")
        or prof.get("is_chelator_sensitive")
        or prof.get("avoid_tris_buffer")
        or prof.get("avoid_phosphate_buffer")
        or prof.get("avoid_reducing_agents")
    ):
        return "metalloprotein_safe"
    tag_low = str(prof.get("effective_tag") or "").lower()
    if "his" in tag_low and not prof.get("avoid_imidazole_imac"):
        return "his_lysis"
    if "strep" in tag_low:
        return "strep_elution"
    if "gst" in tag_low:
        return "gst_elution"
    if "mbp" in tag_low:
        return "mbp_elution"
    return "sec_running"


def build_compatibility_matrix(
    metrics: Dict[str, Any],
    working_ph: float,
    net_charge: float,
    workflow: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    Produces a crisp 6-tile visual Reagent & Column Traffic-Light Compatibility Board
    customized to the active protein's exact biochemistry (zero repetitive prose walls).
    Each item has keys: title, status ('good'|'warn'|'avoid'), badge, reason.
    """
    prof = workflow.get("protein_profile") or {}
    cats = set(prof.get("categories") or [])
    hic = workflow.get("hic") or {}
    pi = float(metrics.get("pi", 7.0))
    cys_cnt = int((metrics.get("counts") or {}).get("cysteines", 0))
    msum = prof.get("metal_summary", "metal cofactor")

    # 1. Ni-NTA / Imidazole
    if prof.get("avoid_imidazole_imac"):
        if "ni_metalloenzyme" in cats:
            imac_tile = {
                "title": "🧲 Ni-NTA / Imidazole",
                "status": "avoid",
                "badge": "🚫 Avoid",
                "reason": f"Resin Ni²⁺ chelation + Imidazole competes with active-site {msum}.",
            }
        elif "fe_s_cluster" in cats:
            imac_tile = {
                "title": "🧲 Ni-NTA / Imidazole",
                "status": "avoid",
                "badge": "🚫 Avoid",
                "reason": "Imidazole & leached Ni²⁺ destroy O₂-labile [Fe–S] clusters.",
            }
        else:
            imac_tile = {
                "title": "🧲 Ni-NTA / Imidazole",
                "status": "avoid",
                "badge": "🚫 Avoid",
                "reason": f"Ni²⁺ mismetallation & Imidazole extraction of {msum}.",
            }
    elif prof.get("caution_imidazole_imac"):
        imac_tile = {
            "title": "🧲 Ni-NTA / Imidazole",
            "status": "warn",
            "badge": "⚠️ Caution",
            "reason": "High Imidazole binds axial heme Fe (use protease cleavage or Strep).",
        }
    else:
        has_his = "his" in str(prof.get("effective_tag") or "").lower()
        imac_tile = {
            "title": "🧲 Ni-NTA / Imidazole",
            "status": "good",
            "badge": "✅ Safe" if has_his else "✅ Compatible",
            "reason": "Compatible with 6xHis IMAC (no catalytic metal/heme conflict)." if has_his else "No metal/heme conflict if you clone a 6xHis tag.",
        }

    # 2. Chelators (EDTA / EGTA / Citrate) — protein-specific AND resin-specific (Ni-NTA strips Ni2+!)
    has_his_tag = "his" in str(prof.get("effective_tag") or "").lower()
    if prof.get("is_chelator_sensitive"):
        chel_tile = {
            "title": "🧪 EDTA / Citrate",
            "status": "avoid",
            "badge": "🚫 Omit (0 mM)",
            "reason": f"Chelators strip essential {msum} into inactive apoprotein!",
        }
    elif has_his_tag:
        chel_tile = {
            "title": "🧪 EDTA / Citrate",
            "status": "warn",
            "badge": "⚠️ Omit for Ni-NTA",
            "reason": "Safe for the protein itself, but omit/remove EDTA before Ni-NTA IMAC (EDTA strips Ni²⁺ off standard resin; use EDTA-free protease inhibitors).",
        }
    else:
        chel_tile = {
            "title": "🧪 EDTA / Citrate",
            "status": "good",
            "badge": "✅ Non-IMAC (1 mM)",
            "reason": "1 mM EDTA is safe for this protein in IEX/HIC/SEC lysis, but omit or remove EDTA before any Ni-NTA/IMAC step (EDTA strips resin Ni²⁺).",
        }

    # 3. Reducing Agents (DTT / TCEP)
    if prof.get("avoid_reducing_agents"):
        red_tile = {
            "title": "🔗 DTT / TCEP",
            "status": "avoid",
            "badge": "🚫 Omit",
            "reason": f"Protects structural S–S disulfide bonds ({cys_cnt} Cys).",
        }
    elif prof.get("require_reducing_agents"):
        red_tile = {
            "title": "🔗 DTT / TCEP",
            "status": "good",
            "badge": "⭐ Include",
            "reason": f"Keep 0.5 mM TCEP or 1 mM DTT to protect free thiols ({cys_cnt} Cys).",
        }
    else:
        red_tile = {
            "title": "🔗 DTT / TCEP",
            "status": "warn" if cys_cnt >= 2 else "good",
            "badge": "🔹 Optional",
            "reason": "0.5 mM TCEP optional if free thiols oxidize; not strictly required.",
        }

    # 4. Ion-Exchange (IEX)
    if abs(working_ph - pi) < 0.6 or abs(net_charge) < 1.8:
        iex_tile = {
            "title": "⚡ Ion-Exchange (IEX)",
            "status": "warn",
            "badge": "⚠️ Weak Near pI",
            "reason": f"pH {working_ph:.1f} is close to pI {pi:.2f} (Q={net_charge:+.1f}); shift pH ≥1 unit or use HIC.",
        }
    elif net_charge < 0:
        iex_tile = {
            "title": "⚡ Ion-Exchange (IEX)",
            "status": "good",
            "badge": "✅ HiTrap Q (AEX)",
            "reason": f"Negative net charge ({net_charge:+.1f}) at pH {working_ph:.1f} binds positively charged Q beads.",
        }
    else:
        iex_tile = {
            "title": "⚡ Ion-Exchange (IEX)",
            "status": "good",
            "badge": "✅ HiTrap SP (CEX)",
            "reason": f"Positive net charge ({net_charge:+.1f}) at pH {working_ph:.1f} binds negatively charged SP beads.",
        }

    # 5. Hydrophobic Interaction (HIC)
    if prof.get("requires_detergent_micelles") or float(metrics.get("gravy", 0.0)) > 0.35:
        hic_tile = {
            "title": "💧 Hydrophobic (HIC)",
            "status": "avoid",
            "badge": "🚫 Avoid Aqueous",
            "reason": "Membrane detergents phase-separate in high salt; use Detergent IEX + SEC.",
        }
    elif hic.get("is_required_or_strong"):
        short_res = hic.get("recommended_resin", "Phenyl HP").split("(")[0].strip().replace("HiTrap ", "")
        hic_tile = {
            "title": "💧 Hydrophobic (HIC)",
            "status": "good",
            "badge": f"⭐ {short_res}",
            "reason": f"Strong orthogonal step (GRAVY {float(metrics.get('gravy', 0.0)):+.2f}); no dialysis after high salt!",
        }
    else:
        hic_tile = {
            "title": "💧 Hydrophobic (HIC)",
            "status": "warn",
            "badge": "🔹 Secondary",
            "reason": f"Very polar surface (GRAVY {float(metrics.get('gravy', 0.0)):+.2f}); try IEX before Octyl HIC.",
        }

    # 6. Ideal Buffer Family (strictly consistent with Method A IEX counterion polarity!)
    iex_is_cex = (abs(working_ph - pi) >= 0.6 and net_charge >= 1.8)
    iex_is_aex = (abs(working_ph - pi) >= 0.6 and net_charge <= -1.8)
    if prof.get("avoid_tris_buffer"):
        buf_tile = {
            "title": "🧫 Buffer Family",
            "status": "good",
            "badge": "⭐ HEPES / Bis-Tris" if iex_is_aex else "⭐ HEPES / PO₄",
            "reason": (
                "Avoid Tris (forms Schiff-base adduct with PLP) & avoid Phosphate on AEX Q!"
                if iex_is_aex
                else "Avoid Tris (forms Schiff-base adduct with PLP cofactor)!"
            ),
        }
    elif prof.get("avoid_phosphate_buffer"):
        buf_tile = {
            "title": "🧫 Buffer Family",
            "status": "good",
            "badge": "⭐ HEPES / MOPS" if iex_is_cex else "⭐ HEPES / Tris",
            "reason": (
                "Avoid Phosphate (precipitates Ca²⁺/Mn²⁺) & omit Tris on CEX SP (cationic Tris⁺ competes)."
                if iex_is_cex
                else "Avoid Phosphate (precipitates insoluble Ca²⁺/Mn²⁺ phosphate chalk)!"
            ),
        }
    elif prof.get("is_chelator_sensitive"):
        buf_tile = {
            "title": "🧫 Buffer Family",
            "status": "good",
            "badge": "⭐ HEPES / MOPS",
            "reason": "Non-chelating Good's buffers protect bound catalytic metal ions.",
        }
    elif prof.get("requires_detergent_micelles"):
        buf_tile = {
            "title": "🧫 Buffer Family",
            "status": "good",
            "badge": "⭐ HEPES + DDM",
            "reason": "Keep 0.03% DDM (or LMNG/CHS) above CMC in every column buffer.",
        }
    elif iex_is_cex:
        buf_tile = {
            "title": "🧫 Buffer Family",
            "status": "good",
            "badge": "✅ PO₄ / HEPES / MOPS",
            "reason": "Anionic/zwitterionic buffers for CEX (SP) — omit cationic Tris (competes for SP resin!).",
        }
    elif iex_is_aex:
        buf_tile = {
            "title": "🧫 Buffer Family",
            "status": "good",
            "badge": "✅ Tris / HEPES / Bis-Tris",
            "reason": "Cationic/zwitterionic buffers for AEX (Q) — omit Phosphate/Citrate (polyanions compete on Q!).",
        }
    else:
        buf_tile = {
            "title": "🧫 Buffer Family",
            "status": "good",
            "badge": "✅ HEPES / PO₄ / Tris",
            "reason": "Match buffer pKa within ±1 unit of working pH (pair with HIC/SEC near pI).",
        }

    return [imac_tile, chel_tile, red_tile, iex_tile, hic_tile, buf_tile]


def build_student_lab_challenge(
    protein_name: str,
    metrics: Dict[str, Any],
    working_ph: float,
    net_charge: float,
    workflow: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generates a single, 100%-scientifically-accurate interactive Lab Decision Challenge
    tailored specifically to the active protein's real cofactor, redox, membrane, or pI physics.
    """
    prof = workflow.get("protein_profile") or {}
    cats = set(prof.get("categories") or [])
    pi = float(metrics.get("pi", 7.0))
    msum = prof.get("metal_summary", "catalytic metal")
    short_name = protein_name.split("(")[0].strip() or "this protein"

    if "ni_metalloenzyme" in cats:
        return {
            "question": f"Your labmate wants to purify {short_name} on a Ni-NTA column using 300 mM Imidazole + 1 mM EDTA. What happens?",
            "options": [
                "It elutes at >98% purity with higher catalytic activity because extra Ni²⁺ boosts turnover.",
                f"Protein metal sites can chelate resin Ni²⁺, while Imidazole & EDTA compete with and strip the native {msum}!",
                "Imidazole cleaves all peptide bonds next to lysine residues.",
                "EDTA increases hydrophobic binding to Phenyl Sepharose.",
            ],
            "correct_index": 1,
            "explanation": (
                f"Correct! {short_name} relies on {msum}. Millimolar Imidazole directly competes with protein-bound Ni²⁺, "
                "apo/accessory pockets strip Ni²⁺ from Ni-NTA beads, and EDTA chelates essential divalent cations. "
                "Use EDTA-free Ion-Exchange (HiTrap Q/SP) + HIC + SEC in HEPES/MOPS—or EDTA-free Twin-Strep/MBP affinity."
            ),
        }
    if "fe_s_cluster" in cats:
        return {
            "question": f"{short_name} contains an [Fe–S] cluster. Which bench condition protects its cofactor during purification?",
            "options": [
                "Open-bench Ni-NTA with 300 mM Imidazole and vigorous aerobic stirring.",
                "Anaerobic glovebox (< 2 ppm O₂), N₂-sparged EDTA-free buffer + 1–2 mM DTT, monitoring A410/A280.",
                "Adding 10 mM EDTA and 5% hydrogen peroxide to oxidize impurities.",
                "Boiling lysate at 95°C for 10 minutes before SEC.",
            ],
            "correct_index": 1,
            "explanation": (
                "Correct! [Fe–S] clusters oxidize rapidly in air and are disrupted by Imidazole/EDTA or leached Ni²⁺. "
                "Anaerobic IEX/Strep + SEC in N₂-sparged reducing buffer preserves the intact holoenzyme (track A410/A280)."
            ),
        }
    if prof.get("is_metalloprotein"):
        return {
            "question": f"{short_name} requires {msum}. Why should you avoid standard Ni-NTA + Imidazole and EDTA?",
            "options": [
                f"Leached Ni²⁺ causes irreversible mismetallation, while Imidazole & EDTA extract native {msum} into inactive apoprotein.",
                "Ni-NTA only works on proteins larger than 200 kDa.",
                "EDTA precipitates Tris-HCl buffer into insoluble crystals.",
                "Imidazole permanently reverses the protein's isoelectric point (pI).",
            ],
            "correct_index": 0,
            "explanation": (
                f"Correct! By the Irving–Williams series, leached Ni²⁺ displaces native {msum} (mismetallation), "
                "and both Imidazole and EDTA strip divalent metal cofactors. Purify in EDTA-free HEPES/MOPS via native IEX/HIC or Twin-Strep/MBP."
            ),
        }
    if prof.get("is_heme_protein"):
        return {
            "question": f"{short_name} is a hemoprotein. What happens if you elute it from Ni-NTA using 300–500 mM Imidazole?",
            "options": [
                "Imidazole covalently cuts the porphyrin ring in half.",
                "Imidazole coordinates the 6th axial position of heme iron—shifting the Soret peak (~410 ➜ ~426 nm) and inhibiting catalysis.",
                "Imidazole converts heme iron into a disulfide bond.",
                "Nothing—Imidazole cannot interact with iron ions.",
            ],
            "correct_index": 1,
            "explanation": (
                "Correct! High Imidazole directly binds the axial site of heme Fe. Use native IEX/HIC, EDTA-free Strep/MBP, "
                "or imidazole-free on-column protease cleavage, and verify purity with the Reinheitszahl ratio (Rz = A_Soret / A280)."
            ),
        }
    if prof.get("requires_detergent_micelles"):
        return {
            "question": f"{short_name} is an integral membrane protein. Which step will RUIN your prep?",
            "options": [
                "Solubilizing membranes in 1% DDM and running SEC in 0.03% DDM.",
                "Running Ammonium Sulfate salting-out or detergent-free aqueous HIC.",
                "Using a Superdex 200 column to account for the ~70 kDa detergent micelle belt.",
                "Clarifying unsolubilized membrane debris at 100,000 × g.",
            ],
            "correct_index": 1,
            "explanation": (
                "Correct! High Ammonium Sulfate causes detergent micelles to phase-separate and makes exposed transmembrane helices aggregate irreversibly. "
                "Always maintain mild detergent above CMC (e.g., 0.03% DDM or LMNG/CHS) through affinity, IEX, and SEC!"
            ),
        }
    if prof.get("avoid_tris_buffer"):
        return {
            "question": f"{short_name} uses a PLP (Vitamin B6) aldehyde cofactor. Which buffer should you AVOID?",
            "options": [
                "50 mM HEPES (pH 7.5)",
                "50 mM Tris-HCl (pH 7.5) — Tris's primary amine forms a covalent Schiff-base adduct with PLP!",
                "50 mM Sodium Phosphate (pH 7.2)",
                "150 mM NaCl salt additive",
            ],
            "correct_index": 1,
            "explanation": (
                "Correct! Tris contains a reactive primary amine that condenses with the aldehyde group of PLP, stripping/quenching the cofactor. "
                "Use HEPES or Phosphate supplemented with 10–50 µM PLP instead."
            ),
        }
    if prof.get("avoid_reducing_agents"):
        cys_n = int((metrics.get("counts") or {}).get("cysteines", 0))
        iex_right = "Cation-Exchange (HiTrap SP)" if net_charge > 0 else "Anion-Exchange (HiTrap Q)"
        charge_word = "positively" if net_charge > 0 else "negatively"
        return {
            "question": f"At pH {working_ph:.1f}, {short_name} has pI {pi:.2f} (net charge {net_charge:+.1f}) and structural disulfide bonds ({cys_n} Cys). Which setup keeps it folded and bound?",
            "options": [
                f"{iex_right} in reductant-free buffer (0 mM DTT/TCEP) to preserve native S–S bridges.",
                f"{'Anion-Exchange (HiTrap Q)' if net_charge > 0 else 'Cation-Exchange (HiTrap SP)'} with 10 mM DTT at 60°C.",
                "Running IEX exactly at pH = pI so the net charge equals zero.",
                "Adding 20 mM β-Mercaptoethanol to strengthen disulfide bonds.",
            ],
            "correct_index": 0,
            "explanation": (
                f"Correct! At pH {working_ph:.1f}, {short_name} is {charge_word} charged ({net_charge:+.1f}), so it binds {iex_right}—and omitting DTT/TCEP "
                "keeps its stabilizing S–S disulfide bridges intact!"
            ),
        }
    if abs(working_ph - pi) < 0.9 or abs(net_charge) < 2.2:
        return {
            "question": f"At pH {working_ph:.1f}, {short_name} (pI {pi:.2f}) has weak net charge ({net_charge:+.1f}). Which chromatography method works WITHOUT relying on net charge?",
            "options": [
                "Anion-Exchange (HiTrap Q) at the same pH.",
                "Hydrophobic Interaction Chromatography (HIC) — separates by surface hydrophobic patches in high kosmotropic salt!",
                "Cation-Exchange (HiTrap SP) at the same pH.",
                "Adding 5 mM DTT to shift pI by +4 units.",
            ],
            "correct_index": 1,
            "explanation": (
                f"Correct! Near pI ({pi:.2f}), proteins carry almost zero net charge so IEX fails. "
                "HIC separates by surface hydrophobicity in kosmotropic salt (e.g., Ammonium Sulfate), rescuing purification near pI!"
            ),
        }
    correct_iex = "Anion-Exchange (HiTrap Q — positively charged resin)" if net_charge < 0 else "Cation-Exchange (HiTrap SP — negatively charged resin)"
    wrong_iex = "Cation-Exchange (HiTrap SP — negatively charged resin)" if net_charge < 0 else "Anion-Exchange (HiTrap Q — positively charged resin)"
    sign_txt = "negatively" if net_charge < 0 else "positively"
    cmp_txt = "above" if working_ph > pi else "below"
    return {
        "question": f"At working pH {working_ph:.1f} (which is {cmp_txt} pI {pi:.2f}), {short_name} has net charge {net_charge:+.1f}. Which Ion-Exchange resin binds it?",
        "options": [
            correct_iex,
            wrong_iex,
            "Size-Exclusion (Superdex) via electrostatic attraction.",
            "Neither — proteins only bind IEX when net charge equals 0.",
        ],
        "correct_index": 0,
        "explanation": (
            f"Correct! Because pH {working_ph:.1f} is {cmp_txt} pI {pi:.2f}, {short_name} is {sign_txt} charged ({net_charge:+.1f}) "
            f"and binds the oppositely charged resin: {correct_iex}!"
        ),
    }


def audit_buffer_compatibility_for_protein(
    rows: List[Dict[str, Any]],
    target_ph: float,
    protein_profile: Optional[Dict[str, Any]],
    protein_name: str = "Active Protein",
) -> List[str]:
    """
    Audits a custom buffer composition (Tab 3) against BOTH internal chemical incompatibilities
    (e.g., Phosphate + Ca2+/Zn2+ precipitation, >1 mM DTT in IMAC buffer) AND the active protein's
    biochemical profile (metalloprotein vs Imidazole/EDTA/Citrate, disulfide protein vs DTT/TCEP,
    PLP enzyme vs Tris, serine protease vs PMSF, membrane protein missing detergent).
    """
    prof = protein_profile or {}
    warnings: List[str] = []
    if not rows:
        return warnings

    reagent_names_low = [str(r.get("Reagent") or "").lower().strip() for r in rows if float(r.get("Value") or 0.0) > 0]
    has_imidazole = any("imidazole" in n for n in reagent_names_low)
    chelators_found = [r["Reagent"] for r in rows if float(r.get("Value") or 0.0) > 0 and any(k in str(r.get("Reagent") or "").lower() for k in ("edta", "egta", "citrate", "citric acid", "phenanthroline"))]
    reductants_found = [r["Reagent"] for r in rows if float(r.get("Value") or 0.0) > 0 and any(k in str(r.get("Reagent") or "").lower() for k in ("dtt", "dithiothreitol", "tcep", "mercaptoethanol", "β-me", "b-me", "glutathione", "gsh"))]
    has_dtt = any("dtt" in n or "dithiothreitol" in n for n in reagent_names_low)
    has_tris = any("tris" in n and "citrate" not in n for n in reagent_names_low)
    has_phosphate = any(k in n for n in reagent_names_low for k in ("hpo4", "h2po4", "phosphate"))
    has_divalent_salt = any(k in n for n in reagent_names_low for k in ("cacl2", "calcium", "znso4", "zncl2", "zinc", "mncl2", "mnso4"))
    has_detergent = any(k in n for n in reagent_names_low for k in ("ddm", "maltoside", "lmng", "triton", "tween", "polysorbate", "chaps", "octyl glucoside", "digitonin"))
    has_pmsf = any("pmsf" in n or "aebsf" in n for n in reagent_names_low)

    # 1. Metalloprotein / Heme vs Imidazole
    if has_imidazole and prof.get("avoid_imidazole_imac"):
        warnings.append(
            f"🛑 **Metalloprotein–Imidazole Conflict ({protein_name} — {prof.get('metal_summary', 'Metal Cofactor')}):** "
            "This recipe contains **Imidazole**, which coordinates/competes with the protein's bound metal cofactor (and Ni-NTA resin risks "
            "metal stripping or Ni²⁺ mismetallation)! Load the **🪙 Metalloprotein-Safe (EDTA-Free & Imidazole-Free)** template instead."
        )
    elif has_imidazole and prof.get("caution_imidazole_imac"):
        warnings.append(
            f"⚠️ **Hemoprotein–Imidazole Caution ({protein_name}):** Millimolar **Imidazole** binds axially to heme iron (shifting the Soret peak & "
            "inhibiting catalysis). Desalt out Imidazole immediately after elution or use an Imidazole-free purification buffer."
        )

    # 2. Metalloprotein / Ca2+ / Mg2+ vs EDTA / EGTA / Citrate
    if chelators_found and prof.get("is_chelator_sensitive"):
        ch_list = ", ".join(f"`{c}`" for c in chelators_found)
        warnings.append(
            f"🛑 **Chelator Stripping Hazard ({protein_name} — {prof.get('metal_summary', 'Divalent Cofactor')}):** "
            f"{ch_list} strongly chelates divalent cations and will strip essential **{prof.get('metal_summary')}**, converting your enzyme into inactive apoprotein! "
            "Remove chelators from the table above."
        )

    # 3. Secreted Disulfide Protein vs Reducing Agents
    if reductants_found and prof.get("avoid_reducing_agents"):
        red_list = ", ".join(f"`{c}`" for c in reductants_found)
        warnings.append(
            f"⚠️ **Disulfide Reduction Warning ({protein_name}):** {red_list} cleaves structural `S–S` disulfide bridges and can unfold secreted/extracellular proteins! "
            "Delete reducing agents unless intentionally denaturing."
        )

    # 4. PLP Enzyme vs Primary Amine Tris Buffer
    if has_tris and prof.get("avoid_tris_buffer"):
        warnings.append(
            f"🛑 **PLP–Tris Schiff-Base Conflict ({protein_name}):** Primary amine **`Tris`** reacts covalently with pyridoxal 5'-phosphate (PLP), quenching/stripping the cofactor! "
            "Replace `Tris` with `HEPES` or `Sodium/Potassium Phosphate`."
        )

    # 5. Phosphate + Ca2+/Zn2+ Insoluble Salt Precipitation
    if has_phosphate and (has_divalent_salt or prof.get("avoid_phosphate_buffer")):
        warnings.append(
            "⚠️ **Insoluble Metal-Phosphate Precipitation Hazard:** Mixing orthophosphate (`Na2HPO4`/`KH2PO4`) with `Ca²⁺`, `Zn²⁺`, or `Mn²⁺` precipitates insoluble "
            "`Ca3(PO4)2` / `Zn3(PO4)2` crystals that clog 0.22 µm filters and FPLC columns! Switch buffer to `HEPES` or `Tris-HCl`."
        )

    # 6. DTT + Imidazole (Ni-NTA resin reduction) & EDTA/Chelator + Imidazole (Ni-NTA resin stripping)
    if has_imidazole and has_dtt:
        warnings.append(
            "⚠️ **IMAC Resin Reduction Warning:** `DTT` reduces immobilized Ni²⁺ beads to brown/black metallic Ni⁰ (stripping the column)! "
            "Replace `DTT` with `0.5 mM TCEP-HCl` (or `≤ 5 mM 2-Mercaptoethanol`) if IMAC is used on a non-metalloprotein."
        )
    if has_imidazole and chelators_found and not prof.get("is_chelator_sensitive"):
        ch_list = ", ".join(f"`{c}`" for c in chelators_found)
        warnings.append(
            f"🛑 **IMAC Resin Stripping Hazard (Chelator + Imidazole):** {ch_list} chelates and strips immobilized `Ni²⁺`/`Co²⁺` off standard Ni-NTA / TALON resin "
            "(turning blue beads white and preventing His-tag binding)! Omit EDTA before IMAC (use EDTA-free protease inhibitors), desalt it out prior to loading, or switch to an EDTA-resistant resin (e.g., HisTrap Excel)."
        )

    # 7. Active Serine Protease vs PMSF
    if has_pmsf and prof.get("protease_class") == "serine":
        warnings.append(
            f"🛑 **Self-Inhibition Hazard ({protein_name}):** `PMSF` irreversibly sulfonylates the active-site Serine of serine proteases—omit `PMSF` if you need catalytically active enzyme!"
        )

    # 8. Integral Membrane Protein without Detergent
    if prof.get("requires_detergent_micelles") and not has_detergent:
        warnings.append(
            f"🧴 **Missing Detergent Micelle ({protein_name}):** Integral membrane proteins aggregate immediately in detergent-free aqueous buffer—include a mild detergent above CMC (e.g., `0.03% w/v DDM` or `0.05% Tween-20` / `Triton X-100`)."
        )

    return warnings


def _build_reagent_bench_warning(name_low: str, molarity_m: float, is_liquid: bool, liquid_ml: float, vol_ml: float, density: float, mw: float) -> str:
    """Returns reagent-specific solubility, pH-titration, volume-displacement, and lab safety warnings."""
    notes: List[str] = []

    if is_liquid and liquid_ml is not None and liquid_ml > vol_ml * 1.0001:
        neat_m = (1000.0 * density / mw) if mw > 0 else 0.0
        neat_str = f" (neat liquid = {neat_m:.2f} M)" if neat_m > 0 else ""
        return (
            f"⚠️ IMPOSSIBLE CONCENTRATION: Requires {liquid_ml:.2f} mL neat liquid in {vol_ml:g} mL final volume{neat_str}!"
        )

    # 1. PMSF (insoluble & unstable in water -> 100 mM isopropanol/ethanol stock)
    if "pmsf" in name_low or "phenylmethylsulfonyl" in name_low:
        stock_ml = (molarity_m / 0.100) * vol_ml if molarity_m > 0 else 0.0
        stock_str = (
            f"spike {stock_ml * 1000.0:.1f} µL" if 0 < stock_ml < 1.0
            else (f"spike {stock_ml:.2f} mL" if stock_ml >= 1.0 else "spike")
        )
        if molarity_m > 0.002:
            return (
                "⚠️ EXCEEDS AQUEOUS SOLUBILITY (~2 mM max; precipitates in water)! Use 0.1–1.0 mM from a 100 mM stock "
                "(17.42 mg/mL) in anhydrous isopropanol or absolute ethanol (fume hood; aqueous t½ ~35–55 min at pH 7.5–8.0)."
            )
        return (
            f"⚠️ Insoluble directly in water! Prepare a 100 mM stock (17.42 mg/mL) in anhydrous isopropanol/ethanol (fume hood) "
            f"and {stock_str} into cold buffer immediately before lysis (aqueous t½ ~35–55 min)."
        )

    # 2. Ammonium Sulfate [(NH4)2SO4] saturation (~4.1 M / ~541 g per L final solution at 20–25°C; ~3.9 M at 4°C)
    if "ammonium sulfate" in name_low or "(nh4)2so4" in name_low:
        if molarity_m > 4.10:
            return (
                "⚠️ EXCEEDS SATURATION: Max aqueous (NH4)2SO4 at 20–25°C is ~4.1 M (~541 g per L final solution; ~3.9 M at 4°C) — undissolved salt will remain!"
            )
        if molarity_m > 3.85:
            notes.append("⚠️ Near saturation (~4.1 M at 20°C, ~3.9 M at 4°C — may crystallize in cold room!); start with ~60% water & keep pH ≤ 8.0")
        elif molarity_m >= 0.8:
            notes.append("Start with ~65–70% water (salt displaces volume); keep pH ≤ 8.0 (NH4+ pKa = 9.25)")

    # 3. Sodium Sulfate [Na2SO4] saturation (~1.37 M at 20°C; ~1.98 M at 25°C; precipitates at 4°C!)
    elif "sodium sulfate" in name_low or "na2so4" in name_low:
        if molarity_m > 1.98:
            return "⚠️ EXCEEDS SATURATION: Max aqueous Na2SO4 is ~1.98 M at 25°C (~1.37 M at 20°C)!"
        if molarity_m > 1.37:
            notes.append("⚠️ Exceeds 20°C solubility (~1.37 M; max ~1.98 M at 25°C): keep at 22–25°C and NEVER chill to 4°C!")
        elif molarity_m >= 0.5:
            notes.append("Keep at room temp (20–25°C); Na2SO4·10H2O crystallizes if chilled to 4°C!")

    # 4. D-Biotin vs D-Desthiobiotin
    elif "desthiobiotin" in name_low:
        if molarity_m > 0.010:
            notes.append("⚠️ Limited water solubility (>10 mM): dissolve with NaOH/Tris titration to pH 8.0")
    elif "biotin" in name_low:
        if molarity_m > 0.50:
            return "⚠️ EXCEEDS SOLUBILITY: Even with NaOH titration, max D-Biotin concentrate is ~0.5 M (10× Buffer BXT)!"
        if molarity_m > 0.001:
            notes.append("⚠️ Pure H2O solubility is only ~0.9 mM: dissolve with NaOH titration (to pH ~8.0) before final pH adjustment!")

    # 5. Reduced L-Glutathione, Imidazole, EDTA, EGTA, PIPES
    elif "glutathione" in name_low or "gsh" in name_low:
        notes.append("Acidic tripeptide: re-titrate pH to 8.0 with NaOH AFTER dissolving & prepare fresh!")
    elif "imidazole" in name_low and molarity_m >= 0.050:
        notes.append("Alkaline base: titrate pH down with HCl AFTER dissolving!")
    elif "edta" in name_low:
        if molarity_m > 0.55:
            return "⚠️ EXCEEDS SOLUBILITY: Max EDTA disodium stock is ~0.5 M (requires NaOH titration to pH 8.0)!"
        if molarity_m >= 0.05:
            notes.append("Titrate with NaOH to pH ~8.0 to dissolve completely")
    elif "egta" in name_low:
        if molarity_m > 0.55:
            return "⚠️ EXCEEDS SOLUBILITY: Max EGTA stock is ~0.5 M (requires NaOH titration to pH 8.0)!"
        notes.append("Free acid is insoluble in pure H2O — titrate with NaOH to pH 7.5–8.0 to dissolve!")
    elif "pipes" in name_low:
        notes.append("Free acid requires NaOH titration to pH ~6.5 to dissolve in H2O")

    # 6. Urea & Guanidine-HCl volume displacement & saturation
    elif "urea" in name_low:
        if molarity_m > 10.5:
            return "⚠️ EXCEEDS SATURATION: Max Urea solubility at 20–25°C is ~10.0–10.5 M!"
        if molarity_m >= 4.0:
            disp_pct = molarity_m * 60.06 * 0.075
            notes.append(f"Displaces ~{disp_pct:.0f}% volume: start with ~55% water (NOT 80%) & NEVER heat >37°C (cyanate risk)!")
    elif "guanidine" in name_low:
        if molarity_m > 8.5:
            return "⚠️ EXCEEDS SATURATION: Max Guanidine-HCl solubility at 20°C is ~8.0–8.5 M!"
        if molarity_m >= 3.0:
            disp_pct = molarity_m * 95.53 * 0.075
            notes.append(f"Displaces ~{disp_pct:.0f}% volume: start with ~45–50% water (NOT 80%); precipitates with SDS!")

    # 7. NaCl & KCl saturation
    elif "nacl" in name_low or "sodium chloride" in name_low:
        if molarity_m > 5.4:
            return "⚠️ EXCEEDS SATURATION: Saturated NaCl brine at 20–25°C is ~5.4 M (~317 g/L final solution)!"
    elif "kcl" in name_low or "potassium chloride" in name_low:
        if molarity_m > 4.0:
            return "⚠️ EXCEEDS SATURATION: Saturated KCl at 20°C is ~4.0 M!"

    # 8. Liquid & Specialty Reagents (β-ME, Detergents, Glycerol, DTT, TCEP, SDS)
    elif "mercaptoethanol" in name_low or name_low in ("β-me", "b-me", "bme"):
        notes.append("Dispense in fume hood (neat = 14.26 M); add fresh before use")
    elif "triton" in name_low:
        notes.append("Viscous surfactant; absorbs strongly at UV 280 nm (prepare 10% v/v stock for small volumes)")
    elif "tween" in name_low or "polysorbate 20" in name_low:
        notes.append("Viscous surfactant (prepare 10% v/v aqueous stock for small volumes)")
    elif "glycerol" in name_low:
        notes.append("Viscous: use wide-bore tip / 50% v/v stock or weigh directly on balance")
    elif "dtt" in name_low or "dithiothreitol" in name_low:
        notes.append("Oxidizes in air: add fresh from -20°C 1 M stock right before use")
    elif "tcep" in name_low:
        notes.append("Acidic HCl salt: verify buffer pH (or use pH 7.0 neutralized 0.5 M stock)")
    elif "sds" in name_low or "dodecyl sulfate" in name_low:
        notes.append("Wear dust mask; precipitates at <15°C or with K+ salts (KCl/KH2PO4)!")

    return " • ".join(notes)


def compute_custom_buffer_recipe(rows: List[Dict[str, Any]], volume_ml: float) -> List[Dict[str, Any]]:
    """
    Computes exact laboratory recipes from user-editable component rows.
    Supports BOTH:
    - Concentration -> Weight in grams (g), milligrams (mg), AND neat liquid volume (mL / µL)
      for liquid reagents [mM, M, µM, % (w/v), % (v/v), mg/mL]
    - Direct Weight in grams (g) or milligrams (mg) -> Resulting Molarity (mM / M), % (w/v), and liquid volume!
    """
    vol_ml = max(0.1, float(volume_ml))
    vol_l = vol_ml / 1000.0
    computed = []

    for r in rows:
        name = str(r.get("Reagent", "") or "").strip()
        if not name:
            continue
        reg_info = _resolve_reagent_info(name)
        raw_mw = float(r.get("MW (g/mol)") or 0.0)
        mw = raw_mw if raw_mw > 0 else float(reg_info.get("mw", 0.0))
        val = float(r.get("Value") or 0.0)
        unit = str(r.get("Unit") or "mM").strip()

        density = float(reg_info.get("density_g_ml", 1.0))
        is_liquid = (reg_info.get("state") == "liquid") and (density > 0)

        mass_g = 0.0
        mass_mg = 0.0
        molarity_m = 0.0
        liquid_ml = None
        final_conc_str = ""

        if unit == "mM":
            molarity_m = val * 1e-3
            moles = molarity_m * vol_l
            mass_g = moles * mw
            mass_mg = mass_g * 1000.0
            if is_liquid and mass_g > 0:
                liquid_ml = mass_g / density
            final_conc_str = f"{val:g} mM"
        elif unit == "M":
            molarity_m = val
            moles = molarity_m * vol_l
            mass_g = moles * mw
            mass_mg = mass_g * 1000.0
            if is_liquid and mass_g > 0:
                liquid_ml = mass_g / density
            final_conc_str = f"{val:g} M ({val * 1000:g} mM)"
        elif unit in ("µM", "uM"):
            molarity_m = val * 1e-6
            moles = molarity_m * vol_l
            mass_g = moles * mw
            mass_mg = mass_g * 1000.0
            if is_liquid and mass_g > 0:
                liquid_ml = mass_g / density
            final_conc_str = f"{val:g} µM"
        elif unit == "% (w/v)":
            # 1% w/v = 1 g solute per 100 mL final solution
            mass_g = val * (vol_ml / 100.0)
            mass_mg = mass_g * 1000.0
            if is_liquid and mass_g > 0:
                liquid_ml = mass_g / density
            if mw > 0:
                molarity_m = (mass_g / mw) / vol_l
                final_conc_str = f"{val:g}% w/v ({molarity_m * 1000.0:.1f} mM)"
            else:
                final_conc_str = f"{val:g}% w/v"
        elif unit == "mg/mL":
            mass_mg = val * vol_ml
            mass_g = mass_mg / 1000.0
            if is_liquid and mass_g > 0:
                liquid_ml = mass_g / density
            if mw > 0:
                molarity_m = val / mw
                final_conc_str = f"{val:g} mg/mL ({molarity_m * 1000.0:.2f} mM)"
            else:
                final_conc_str = f"{val:g} mg/mL"
        elif unit == "% (v/v)":
            liquid_ml = val * (vol_ml / 100.0)
            mass_g = liquid_ml * density
            mass_mg = mass_g * 1000.0
            if mw > 0:
                molarity_m = (mass_g / mw) / vol_l
                if molarity_m >= 1.0:
                    final_conc_str = f"{val:g}% v/v ({molarity_m:.2f} M)"
                else:
                    final_conc_str = f"{val:g}% v/v ({molarity_m * 1000.0:.1f} mM)"
            else:
                final_conc_str = f"{val:g}% v/v"
        elif unit == "g":
            # User entered direct weight in grams (g) -> compute resulting concentration!
            mass_g = val
            mass_mg = val * 1000.0
            mg_per_ml = mass_mg / vol_ml
            if is_liquid and mass_g > 0:
                liquid_ml = mass_g / density
            if mw > 0:
                molarity_m = (mass_g / mw) / vol_l
                molarity_mm = molarity_m * 1000.0
                if molarity_mm >= 1000.0:
                    final_conc_str = f"{molarity_m:.3f} M ({mg_per_ml:.2f} mg/mL)"
                else:
                    final_conc_str = f"{molarity_mm:.2f} mM ({mg_per_ml:.2f} mg/mL)"
            else:
                final_conc_str = f"{mg_per_ml:.2f} mg/mL ({mass_g / (vol_ml / 100.0):.2f}% w/v)"
        elif unit == "mg":
            # User entered direct weight in milligrams (mg) -> compute resulting concentration!
            mass_mg = val
            mass_g = val / 1000.0
            mg_per_ml = mass_mg / vol_ml
            if is_liquid and mass_g > 0:
                liquid_ml = mass_g / density
            if mw > 0:
                molarity_m = (mass_g / mw) / vol_l
                molarity_mm = molarity_m * 1000.0
                final_conc_str = f"{molarity_mm:.2f} mM ({mg_per_ml:.3f} mg/mL)"
            else:
                final_conc_str = f"{mg_per_ml:.3f} mg/mL"
        else:
            molarity_m = val * 1e-3
            moles = molarity_m * vol_l
            mass_g = moles * mw
            mass_mg = mass_g * 1000.0
            if is_liquid and mass_g > 0:
                liquid_ml = mass_g / density
            final_conc_str = f"{val:g} {unit}"

        # Practical laboratory bench guidance + chemical solubility / hazard checks
        extra_warn = _build_reagent_bench_warning(
            name.lower(), molarity_m, is_liquid or (liquid_ml is not None), liquid_ml, vol_ml, density, mw
        )

        if extra_warn.startswith("⚠️ EXCEEDS") or extra_warn.startswith("⚠️ IMPOSSIBLE") or "pmsf" in name.lower():
            bench_tip = extra_warn
        else:
            if liquid_ml is not None:
                if 0 < liquid_ml < 0.002:
                    base_tip = f"Tiny volume ({liquid_ml * 1000.0:.2f} µL neat): prepare a 100× stock dilution to pipette accurately"
                elif liquid_ml < 1.0:
                    base_tip = f"Pipette {liquid_ml * 1000.0:.1f} µL neat liquid (ρ = {density:g} g/mL; {mass_mg:.1f} mg)"
                else:
                    base_tip = f"Measure {liquid_ml:.2f} mL liquid (ρ = {density:g} g/mL; or weigh {mass_g:.2f} g)"
            elif 0 < mass_mg < 25.0:
                base_tip = f"Tiny mass ({mass_mg:.2f} mg < 25 mg): prepare a 100× concentrated stock solution and pipette!"
            elif 25.0 <= mass_mg < 500.0:
                base_tip = f"Weigh {mass_mg:.1f} mg on an analytical balance"
            else:
                base_tip = f"Weigh {mass_g:.3f} g on a top-loading lab balance"

            bench_tip = f"{base_tip} • {extra_warn}" if extra_warn else base_tip

        computed.append({
            "Reagent": name,
            "MW (g/mol)": round(mw, 2) if mw > 0 else 0.0,
            "Final Concentration": final_conc_str,
            "Weight (g)": round(mass_g, 4),
            "Weight (mg)": round(mass_mg, 2),
            "Liquid Vol (mL)": round(liquid_ml, 6) if liquid_ml is not None else None,
            "Bench Tip": bench_tip
        })

    return computed


def convert_weight_and_molarity(
    mode: str,
    mw_g_mol: float,
    volume_ml: float,
    input_value: float,
    input_unit: str
) -> Dict[str, Any]:
    """
    Two-way laboratory quick converter:
    - mode == 'conc_to_weight': given concentration (M, mM, µM, or nM) + MW + Volume -> returns exact g and mg.
    - mode == 'weight_to_conc': given weight (g, mg, or µg) + MW + Volume -> returns exact mM, M, mg/mL, and % (w/v).
    """
    vol_ml = max(0.01, float(volume_ml))
    vol_l = vol_ml / 1000.0
    mw = max(0.01, float(mw_g_mol))
    u = str(input_unit).strip()

    if mode == "conc_to_weight":
        if u == "M":
            molar = float(input_value)
        elif u == "mM":
            molar = float(input_value) * 1e-3
        elif u in ("µM", "uM"):
            molar = float(input_value) * 1e-6
        elif u == "nM":
            molar = float(input_value) * 1e-9
        else:
            molar = float(input_value) * 1e-3
        mass_g = molar * vol_l * mw
        mass_mg = mass_g * 1000.0
    else:
        if u == "g":
            mass_g = float(input_value)
        elif u == "mg":
            mass_g = float(input_value) * 1e-3
        elif u in ("µg", "ug"):
            mass_g = float(input_value) * 1e-6
        else:
            mass_g = float(input_value) * 1e-3
        mass_mg = mass_g * 1000.0
        molar = (mass_g / mw) / vol_l

    return {
        "mass_g": round(mass_g, 5),
        "mass_mg": round(mass_mg, 4),
        "molarity_mm": round(molar * 1000.0, 4),
        "molarity_m": round(molar, 6),
        "mg_per_ml": round(mass_mg / vol_ml, 4),
        "pct_wv": round((mass_g / vol_ml) * 100.0, 4)
    }


def generate_affinity_recipes(tag_type: str, volume_ml: float = 500.0) -> Dict[str, Any]:
    """Generates standard lysis, wash, and elution buffer recipes (with exact g and mg) for purification systems."""
    v_l = volume_ml / 1000.0
    tag_clean = tag_type.lower()

    def _comp(name: str, conc_str: str, mM_val: float, mw: float) -> Dict[str, Any]:
        exact_g = mM_val * 0.001 * mw * v_l
        return {
            "name": name,
            "conc": conc_str,
            "mass_g": round(exact_g, 4),
            "mass_mg": round(exact_g * 1000.0, 2)
        }

    if "his" in tag_clean:
        glyc_ml = 0.05 * volume_ml
        glyc_g = glyc_ml * 1.261
        return {
            "system": "Immobilized Metal Affinity Chromatography (IMAC / Ni-NTA / HisTrap)",
            "resin": "Ni-NTA Agarose / HisTrap HP / TALON (Co2+ for higher selectivity)",
            "buffers": [
                {
                    "name": "Lysis / Binding Buffer",
                    "components": [
                        _comp("Tris-Base (Anhydrous)", "50 mM", 50, 121.14),
                        _comp("NaCl", "300 mM", 300, 58.44),
                        _comp("Imidazole", "20 mM", 20, 68.08),
                        {
                            "name": "Glycerol (ρ = 1.261 g/mL)",
                            "conc": "5% (v/v)",
                            "mass_g": round(glyc_g, 4),
                            "mass_mg": round(glyc_g * 1000.0, 2),
                            "liquid_ml": round(glyc_ml, 3)
                        },
                        _comp("TCEP-HCl (omit if disulfide-bonded)", "0.5 mM", 0.5, 286.65)
                    ],
                    "ph_target": "8.0 at working temp (titrate HCl after adding imidazole)"
                },
                {
                    "name": "Wash Buffer (High Stringency)",
                    "components": [
                        _comp("Tris-Base (Anhydrous)", "50 mM", 50, 121.14),
                        _comp("NaCl", "500 mM", 500, 58.44),
                        _comp("Imidazole", "50 mM", 50, 68.08)
                    ],
                    "ph_target": "8.0 at working temp"
                },
                {
                    "name": "Elution Buffer",
                    "components": [
                        _comp("Tris-Base (Anhydrous)", "50 mM", 50, 121.14),
                        _comp("NaCl", "300 mM", 300, 58.44),
                        _comp("Imidazole", "300 mM", 300, 68.08)
                    ],
                    "ph_target": "8.0 at working temp (re-adjust pH down with HCl after dissolving imidazole!)"
                }
            ]
        }
    elif "hic" in tag_clean or "hydrophobic" in tag_clean:
        return {
            "system": "Hydrophobic Interaction Chromatography (HIC)",
            "resin": "HiTrap Butyl HP / Phenyl HP / Octyl FF",
            "buffers": [
                {
                    "name": "Buffer A (High-Salt Binding Buffer)",
                    "components": [
                        _comp("Na2HPO4 (Anhydrous)", "50 mM", 50, 141.96),
                        _comp("Ammonium Sulfate (NH4)2SO4", "1,300 mM (1.3 M)", 1300, 132.14)
                    ],
                    "ph_target": "7.0 (adjust pH after dissolving salt; start with ~65–70% water)"
                },
                {
                    "name": "Buffer B (Salt-Free Elution Buffer)",
                    "components": [
                        _comp("Na2HPO4 (Anhydrous)", "50 mM", 50, 141.96)
                    ],
                    "ph_target": "7.0"
                }
            ]
        }
    elif "strep" in tag_clean:
        is_xt = "xt" in tag_clean
        return {
            "system": "Strep-Tactin / Strep-Tactin XT Affinity Chromatography",
            "resin": "Strep-Tactin XT 4Flow (50 mM Biotin elution) or Strep-Tactin Superflow (2.5 mM Desthiobiotin elution)",
            "buffers": [
                {
                    "name": "Buffer W (Binding / Wash Buffer)",
                    "components": [
                        _comp("Tris-Base (Anhydrous)", "100 mM", 100, 121.14),
                        _comp("NaCl", "150 mM", 150, 58.44),
                        _comp("EDTA Disodium·2H2O", "1 mM", 1, 372.24)
                    ],
                    "ph_target": "8.0"
                },
                (
                    {
                        "name": "Buffer BXT (Strep-Tactin XT Elution Buffer)",
                        "components": [
                            _comp("Tris-Base (Anhydrous)", "100 mM", 100, 121.14),
                            _comp("NaCl", "150 mM", 150, 58.44),
                            _comp("EDTA Disodium·2H2O", "1 mM", 1, 372.24),
                            _comp("D-Biotin (dissolve via NaOH titration)", "50 mM", 50.0, 244.31)
                        ],
                        "ph_target": "8.0 (pre-dissolve D-Biotin with NaOH before final pH 8.0 adjustment)"
                    }
                    if is_xt else
                    {
                        "name": "Buffer E (Classic Strep-Tactin Elution Buffer; use 50 mM D-Biotin for XT)",
                        "components": [
                            _comp("Tris-Base (Anhydrous)", "100 mM", 100, 121.14),
                            _comp("NaCl", "150 mM", 150, 58.44),
                            _comp("EDTA Disodium·2H2O", "1 mM", 1, 372.24),
                            _comp("D-Desthiobiotin", "2.5 mM", 2.5, 214.26)
                        ],
                        "ph_target": "8.0"
                    }
                )
            ]
        }
    elif "gst" in tag_clean:
        return {
            "system": "Glutathione Affinity Chromatography",
            "resin": "Glutathione Sepharose 4B / GSTrap HP",
            "buffers": [
                {
                    "name": "Binding / Wash Buffer (1× PBS + DTT)",
                    "components": [
                        _comp("NaCl", "137 mM", 137, 58.44),
                        _comp("KCl", "2.7 mM", 2.7, 74.55),
                        _comp("Na2HPO4 (Anhydrous)", "10 mM", 10, 141.96),
                        _comp("KH2PO4 (Anhydrous)", "1.8 mM", 1.8, 136.09),
                        _comp("DTT", "1 mM", 1, 154.25)
                    ],
                    "ph_target": "7.4"
                },
                {
                    "name": "Elution Buffer",
                    "components": [
                        _comp("Tris-Base (Anhydrous)", "50 mM", 50, 121.14),
                        _comp("NaCl", "150 mM", 150, 58.44),
                        _comp("Reduced L-Glutathione (GSH)", "15 mM", 15, 307.32)
                    ],
                    "ph_target": "8.0 (CRITICAL: re-adjust pH to 8.0 with NaOH after dissolving acidic GSH!)"
                }
            ]
        }
    elif "mbp" in tag_clean or "maltose" in tag_clean:
        return {
            "system": "Amylose Affinity Chromatography (MBP-Tag)",
            "resin": "Amylose Resin / MBPTrap HP",
            "buffers": [
                {
                    "name": "Column Binding / Wash Buffer",
                    "components": [
                        _comp("Tris-Base (Anhydrous)", "20 mM", 20, 121.14),
                        _comp("NaCl", "200 mM", 200, 58.44),
                        _comp("EDTA Disodium·2H2O", "1 mM", 1, 372.24),
                        _comp("DTT (optional)", "1 mM", 1, 154.25)
                    ],
                    "ph_target": "7.4"
                },
                {
                    "name": "Elution Buffer (10 mM D-Maltose)",
                    "components": [
                        _comp("Tris-Base (Anhydrous)", "20 mM", 20, 121.14),
                        _comp("NaCl", "200 mM", 200, 58.44),
                        _comp("EDTA Disodium·2H2O", "1 mM", 1, 372.24),
                        _comp("D-Maltose·H2O", "10 mM", 10, 360.31)
                    ],
                    "ph_target": "7.4"
                }
            ]
        }
    else:
        return {
            "system": "Size-Exclusion Chromatography (SEC) Final Polishing Buffer",
            "resin": "Superdex 75 / Superdex 200 / Superose 6 Increase",
            "buffers": [
                {
                    "name": "Running Buffer (SEC)",
                    "components": [
                        _comp("HEPES (Free Acid, Anhydrous)", "25 mM", 25, 238.30),
                        _comp("NaCl", "150 mM", 150, 58.44),
                        _comp("TCEP-HCl", "0.5 mM", 0.5, 286.65)
                    ],
                    "ph_target": "7.5 (0.22 µm filter & degas before use)"
                }
            ]
        }
