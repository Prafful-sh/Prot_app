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

from typing import Dict, Any, List

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
        "notes": "Low imidazole (20 mM) + 300 mM NaCl prevents non-specific host background binding onto Ni2+ beads (omit TCEP if purifying disulfide-bonded proteins).",
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


def advise_hic_strategy(metrics: Dict[str, Any], working_ph: float, net_charge: float) -> Dict[str, Any]:
    """
    Evaluates protein suitability and exact conditions for Hydrophobic Interaction Chromatography (HIC)
    per the Cytiva HIC Handbook:
    - Sepharose ligand hydrophobicity series (lowest -> highest):
      Butyl-S FF < Butyl HP (C4 alkyl) < Phenyl HP (mixed aromatic π–π / hydrophobic) < Octyl FF (C8 alkyl)
      (note: high-substitution Phenyl FF / Capto Phenyl [high sub] exceeds Octyl due to high ligand density,
      whereas Phenyl HP / Capto Phenyl ImpRes provides moderate mixed aromatic/hydrophobic retention).
    - Inverse rule: More hydrophobic proteins require MILDER ligands (Butyl-S / Butyl) and lower starting salt
      so they elute natively without on-column unfolding; polar proteins require stronger ligands (Octyl) + higher salt.
    """
    gravy = float(metrics["gravy"])
    aliphatic = float(metrics["aliphatic_index"])
    pi = float(metrics["pi"])
    length = max(1, int(metrics["length"]))
    cnts = metrics["counts"]

    aromatic_pct = (cnts["aromatic_residues"] / length) * 100.0
    hydrophobic_pct = (cnts["hydrophobic_residues"] / length) * 100.0
    near_pi = (abs(working_ph - pi) < 0.9) or (abs(net_charge) < 2.2)

    # 1. Decide Priority / When HIC is Required
    if gravy > 0.35:
        status_badge = "⚠️ Caution: Membrane / Very Hydrophobic"
        is_required = False
        when_why = (
            f"GRAVY is positive ({gravy:+.2f}), typical of transmembrane or poorly water-soluble proteins. "
            "Standard HIC resins risk irreversible binding without detergents—prefer Affinity + Detergent SEC, or screen Cytiva's mildest HiTrap Butyl-S FF resin."
        )
    elif near_pi:
        status_badge = "⭐ REQUIRED HERE (Rescues Purification Near pI!)"
        is_required = True
        when_why = (
            f"At pH {working_ph:.1f}, your protein is close to its pI ({pi:.2f}) with weak net charge ({net_charge:+.1f}), "
            "where Ion-Exchange (IEX) fails! HIC separates by surface hydrophobic patches in kosmotropic salt—independent of net charge—making HIC essential here."
        )
    elif gravy >= -0.35 or (gravy >= -0.58 and (aromatic_pct >= 7.5 or aliphatic >= 78.0 or hydrophobic_pct >= 38.0)):
        status_badge = "✅ Strongly Recommended Orthogonal Method"
        is_required = True
        when_why = (
            f"Your protein has an ideal amphipathic balance (GRAVY {gravy:+.2f}, {aromatic_pct:.1f}% aromatic Trp/Tyr/Phe, "
            f"{hydrophobic_pct:.1f}% hydrophobic residues). HIC gives crisp orthogonal resolution—especially right after Ammonium Sulfate precipitation "
            "or high-salt IMAC elution with ZERO dialysis needed!"
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

    if working_ph < 5.8:
        hic_buf_name = f"50 mM Sodium Acetate (pH {working_ph:.1f}; or run HIC at neutral pH 7.0 in 50 mM Sodium Phosphate)"
    elif working_ph <= 8.0:
        hic_buf_name = f"50 mM Sodium Phosphate (pH {working_ph:.1f})"
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
    tag_detected: str = "None"
) -> Dict[str, Any]:
    """
    Builds a complete, textbook-grade 3-Step Purification Roadmap (Capture -> Intermediate -> Polishing)
    and evaluates ALL 7 major protein purification methods per Cytiva CIPP (Capture, Intermediate, Polishing) principles.
    """
    mw_kda = float(metrics["mw_kda"])
    pi = float(metrics["pi"])
    gravy = float(metrics["gravy"])
    cys_count = int(metrics["counts"]["cysteines"])
    instab = float(metrics["instability_index"])

    iex = advise_iex_strategy(pi, working_ph, net_charge)
    hic = advise_hic_strategy(metrics, working_ph, net_charge)

    has_tag = bool(tag_detected and tag_detected.lower() not in ("none", "n/a", ""))
    tag_low = (tag_detected or "").lower()

    # --- STEP 1: CAPTURE ---
    if has_tag:
        if "his" in tag_low:
            step1_title = f"Affinity Capture: Ni-NTA / HisTrap HP ({tag_detected})"
            step1_detail = "Bind clarified lysate in 20–40 mM Imidazole + 300–500 mM NaCl (pH 7.4–8.0); wash at 50 mM Imidazole; elute at 250–500 mM Imidazole."
        elif "strep" in tag_low:
            step1_title = f"Affinity Capture: Strep-Tactin / Strep-Tactin XT ({tag_detected})"
            step1_detail = (
                "Bind in 100 mM Tris-HCl, 150 mM NaCl, 1 mM EDTA (pH 8.0); elute classic Strep-Tactin with 2.5 mM D-Desthiobiotin (Buffer E) "
                "OR high-affinity Strep-Tactin XT with 50 mM D-Biotin (Buffer BXT)."
            )
        elif "gst" in tag_low:
            step1_title = "Affinity Capture: Glutathione Sepharose 4B (GST-Tag)"
            step1_detail = "Bind in PBS (pH 7.4) + 1–5 mM DTT; elute natively in 50 mM Tris-HCl, 150 mM NaCl + 10–20 mM Reduced L-Glutathione (re-titrated to pH 8.0)."
        elif "mbp" in tag_low:
            step1_title = "Affinity Capture: Amylose Resin / MBPTrap HP (MBP-Tag)"
            step1_detail = "Bind in 20 mM Tris-HCl (pH 7.4), 200 mM NaCl, 1 mM EDTA; elute natively with 10 mM D-Maltose."
        elif "flag" in tag_low:
            step1_title = f"Affinity Capture: Anti-FLAG M2 Resin ({tag_detected})"
            step1_detail = "Bind in TBS (50 mM Tris-HCl pH 7.4, 150 mM NaCl); elute competitively under native conditions with 100–150 µg/mL 3× FLAG peptide."
        else:
            step1_title = f"Affinity Capture ({tag_detected})"
            step1_detail = "Capture recombinant tagged protein directly from clarified cell lysate using anti-epitope affinity resin."
    else:
        # Tagless Native Capture Logic
        if pi >= 8.4 and net_charge >= 4.0:
            step1_title = "Tagless Capture: Cation Exchange (HiTrap SP) or Heparin Sepharose"
            step1_detail = (
                f"Because your protein is strongly basic (pI {pi:.2f}, charge {net_charge:+.1f}), most host proteins are anionic and flow straight through! "
                "Capture directly on SP Sepharose (CEX) or HiTrap Heparin HP (DNA-mimic / sulfated GAG)."
            )
        elif abs(net_charge) >= 3.5:
            col_short = "HiTrap Q (Anion Exchange)" if net_charge < 0 else "HiTrap SP (Cation Exchange)"
            step1_title = f"Tagless Capture: {col_short} or (NH4)2SO4 Salting-Out"
            step1_detail = (
                f"Optional 35–65% Ammonium Sulfate cut to concentrate lysate, or load low-conductivity clarified extract directly onto {col_short}."
            )
        else:
            step1_title = "Tagless Capture: Ammonium Sulfate (NH4)2SO4 Fractionation (40–70% Cut)"
            step1_detail = (
                "Salt out bulk contaminants first; redissolve your target pellet and load DIRECTLY onto a Hydrophobic Interaction (HIC) column with zero dialysis!"
            )

    # --- STEP 2: INTERMEDIATE ORTHOGONAL CLEANUP ---
    if abs(net_charge) < 2.2 or abs(working_ph - pi) < 0.9:
        step2_title = f"Intermediate Cleanup: Hydrophobic Interaction — {hic['recommended_resin'].split('(')[0].strip()}"
        step2_detail = (
            f"Net charge ({net_charge:+.1f}) is weak near pI, so HIC ({hic['starting_salt']} ➜ 0 M salt) is your best orthogonal step to remove co-eluting contaminants."
        )
    elif hic["is_required_or_strong"] and not has_tag:
        step2_title = f"Intermediate Cleanup: Hydrophobic Interaction — {hic['recommended_resin'].split('(')[0].strip()}"
        step2_detail = (
            f"Separates by surface hydrophobicity ({hic['starting_salt']} ➜ 0 M gradient)—completely orthogonal to charge-based Step 1 capture!"
        )
    elif hic["is_required_or_strong"] and has_tag:
        iex_short = "Anion Exchange (HiTrap Q)" if net_charge < 0 else "Cation Exchange (HiTrap SP)"
        step2_title = f"Intermediate Cleanup: {iex_short} OR {hic['recommended_resin'].split('(')[0].strip()} (HIC)"
        step2_detail = (
            f"Both work well! Use {iex_short} after diluting NaCl to <30 mM, OR adjust high-salt IMAC eluate into {hic['starting_salt']} for HIC with zero dialysis to strip misfolded/clipped species."
        )
    else:
        iex_short = "Anion Exchange (HiTrap Q)" if net_charge < 0 else "Cation Exchange (HiTrap SP)"
        step2_title = f"Intermediate Cleanup: {iex_short}"
        step2_detail = f"High-resolution charge separation (Q = {net_charge:+.1f} at pH {working_ph:.1f}) using a 25 mM ➜ 1,000 mM NaCl gradient."

    # --- STEP 3: FINAL POLISHING (SEC) ---
    if mw_kda < 45.0:
        sec_col = "Superdex 75 Increase 10/300 GL (or HiLoad 16/600 S75)"
        sec_window = "3 – 70 kDa"
    elif mw_kda <= 400.0:
        sec_col = "Superdex 200 Increase 10/300 GL (or HiLoad 16/600 S200)"
        sec_window = "10 – 600 kDa"
    else:
        sec_col = "Superose 6 Increase 10/300 GL"
        sec_window = "5 kDa – 5 MDa"

    step3_title = f"Final Polish: Size-Exclusion (SEC) on {sec_col.split('(')[0].strip()}"
    step3_detail = (
        f"Optimal for your {mw_kda:.1f} kDa protein ({sec_window} globular window). Removes soluble aggregates, resolves oligomers, "
        "and exchanges sample (<2% of column bed volume) into final storage/assay buffer (≥150 mM NaCl)."
    )

    # --- COMPLETE 7-METHOD EVALUATION MATRIX ---
    all_methods = [
        {
            "Method": "1. Tag Affinity (Ni-NTA / Strep / GST / MBP)",
            "Separates By": "Engineered Affinity Tag",
            "Suitability": "⭐ Best Step 1" if has_tag else "➕ Requires Tag",
            "Resin / Reagent": f"{tag_detected} Affinity Resin" if has_tag else "Add 6xHis or Twin-Strep in sidebar prompt if cloning",
            "When to Use It": "Captures ~80–95% pure protein straight out of clarified cell lysate in one selective step."
        },
        {
            "Method": "2. Heparin / Blue Pseudo-Affinity",
            "Separates By": "DNA-Mimic GAG / Dinucleotide Fold",
            "Suitability": "✅ Great Tagless Option" if (pi >= 8.3 and net_charge >= 3.5) else "🔹 Specialized",
            "Resin / Reagent": "HiTrap Heparin HP / HiTrap Blue HP",
            "When to Use It": (
                "Ideal for tagless basic or DNA/RNA-binding proteins (Lysozyme, Cas9, Polymerases, Transcription Factors; elute with 0.1 ➜ 1.5 M NaCl)."
                if (pi >= 8.3 and net_charge >= 3.5)
                else "Use for tagless DNA/RNA-binding proteins, coagulation factors (Heparin), or NAD+/ATP Rossmann-fold enzymes & albumin (Cibacron Blue)."
            )
        },
        {
            "Method": "3. Ammonium Sulfate Salting-Out",
            "Separates By": "Bulk Solubility in Kosmotropic Salt",
            "Suitability": "✅ Classic Native Step" if not has_tag else "🔹 Optional Bulk Step",
            "Resin / Reagent": "Solid (NH4)2SO4 (typically 35% ➜ 70% saturation cut; 100% ≈ 4.1 M at 20°C)",
            "When to Use It": "Inexpensive classroom & native extract concentration step; redissolved pellet feeds DIRECTLY into HIC without dialysis!"
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
                f"Protein is {'negative ➜ Anion Exchange (Q)' if net_charge < 0 else 'positive ➜ Cation Exchange (SP)'} at pH {working_ph:.1f}."
                if abs(net_charge) >= 2.0
                else f"Net charge is too close to zero at pH {working_ph:.1f}; shift pH ≥1 unit from pI or use HIC instead!"
            )
        },
        {
            "Method": "5. Hydrophobic Interaction (HIC)",
            "Separates By": f"Surface Hydrophobicity (GRAVY {gravy:+.2f})",
            "Suitability": hic["status_badge"].split("(")[0].strip(),
            "Resin / Reagent": hic["recommended_resin"],
            "When to Use It": hic["when_why"]
        },
        {
            "Method": "6. Hydroxyapatite / Multimodal (CHT / MMC)",
            "Separates By": "Ca²⁺ & PO₄³⁻ Sites / Charge + Hydrophobic",
            "Suitability": "✅ Orthogonal Polish",
            "Resin / Reagent": "Ceramic Hydroxyapatite CHT [Ca10(PO4)6(OH)2] or Capto MMC",
            "When to Use It": "Salt-tolerant capture (Capto MMC) or phosphate-gradient CHT polish for antibodies, nucleic acid removal & clipped isoforms (avoid EDTA/citrate on CHT!)."
        },
        {
            "Method": "7. Size-Exclusion (SEC / Gel Filtration)",
            "Separates By": f"Hydrodynamic Size ({mw_kda:.1f} kDa)",
            "Suitability": "⭐ Essential Final Step",
            "Resin / Reagent": sec_col,
            "When to Use It": "Always run last on concentrated sample (0.5–2% of column bed volume) in ≥150 mM NaCl to remove aggregates and exchange into final buffer."
        }
    ]

    # --- PRACTICAL BIOCHEMICAL ALERTS FOR STUDENTS & RESEARCHERS ---
    alerts = []
    if cys_count >= 4:
        alerts.append(
            f"🔗 **Disulfide / Cysteine Alert ({cys_count} Cysteines detected):** If this is an extracellular/secreted protein (like Egg-White Lysozyme with 4 native disulfide bonds), "
            "omit strong reducing agents (DTT/TCEP/β-ME) so structural disulfide bridges stay intact! If it is a cytosolic protein with free thiols, include 0.5 mM TCEP or 1 mM DTT to prevent air oxidation."
        )
    if abs(working_ph - pi) < 0.6:
        alerts.append(
            f"⚠️ **Isoelectric Precipitation Alert:** Your chosen pH ({working_ph:.1f}) is within 0.6 units of the protein's pI ({pi:.2f}). "
            "Adjust your buffer pH at least 1.0 unit higher or lower so molecules carry sufficient net charge to repel one another!"
        )
    if instab >= 40.0:
        alerts.append(
            f"🧊 **Cold-Room & Protease Inhibitor Tip:** Instability Index is {instab:.1f} (>40). Work at 4°C on ice, include 5–10% (v/v) Glycerol, and add EDTA-free Protease Inhibitor Cocktail during cell lysis."
        )

    return {
        "roadmap": [
            {"step": "Step 1 • Capture", "title": step1_title, "detail": step1_detail},
            {"step": "Step 2 • Intermediate (Orthogonal)", "title": step2_title, "detail": step2_detail},
            {"step": "Step 3 • Final Polishing", "title": step3_title, "detail": step3_detail}
        ],
        "iex": iex,
        "hic": hic,
        "all_methods": all_methods,
        "alerts": alerts
    }


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
