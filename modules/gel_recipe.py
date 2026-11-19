"""
SDS-PAGE Gel Casting Recipe Calculator Module
=============================================
Textbook-accurate, research-grade handcast polyacrylamide gel calculator supporting:

1. Tris-Glycine SDS-PAGE (Laemmli, U.K. 1970, Nature 227:680-685; Bio-Rad Mini-PROTEAN /
   Sambrook & Russell Molecular Cloning protocols):
   - Resolving gel: 4.0%-22.0% T, 375 mM Tris-HCl pH 8.8 (25% of 1.5 M stock),
     0.1% (w/v) SDS (1% of 10% stock), 0.1% (w/v) APS (1% of 10% stock),
     gel-%-tuned TEMED (0.08% v/v for <=7.5%, 0.06% v/v for 8.0%-10.5%, 0.04% v/v for >=11.0%
     so low-% gels gel reliably and high-% gels do not polymerize prematurely/exothermically),
     and ddH2O to exact conserved total volume.
   - Stacking gel: 4.0% or 5.0% T (3.0%-6.0% supported), 125 mM Tris-HCl pH 6.8
     (25% of 0.5 M stock), 0.1% (w/v) SDS (1%), 0.1% (w/v) APS (1%), 0.10% (v/v) TEMED,
     and ddH2O to exact conserved total volume.

2. Tricine-SDS-PAGE (Schägger, H. 2006, Nature Protocols 1(1):16-22;
   Schägger & von Jagow 1987, Anal. Biochem. 166:368-379 — gold standard for 1-30 kDa
   proteins and <10 kDa peptides):
   - Resolving gel: 10.0%-20.0% T, 1.0 M Tris-HCl / 0.1% (w/v) SDS pH 8.45
     (33.33% of 3x Tricine Gel Buffer: 3.0 M Tris-HCl, 0.3% SDS, pH 8.45 — no separate SDS),
     optional Glycerol (default 10% v/v for >=16% T; 10%-13% v/v recommended),
     0.05% (w/v) APS (0.5% of 10% stock), 0.05% (v/v) TEMED, and ddH2O to exact volume.
   - Stacking gel: 4.0% T, 0.75 M Tris-HCl / 0.075% (w/v) SDS pH 8.45
     (25% of 3x Gel Buffer per Schägger 2006 Table 1 for Kohlrausch ionic-strength stacking),
     0.075% (w/v) APS (0.75% of 10% stock), 0.075% (v/v) TEMED, and ddH2O to exact volume.

Exact mathematical volume conservation (abs(sum(row['volume_ml']) - total_vol_ml) < 1e-4)
is strictly enforced on every resolving and stacking formulation, with explicit guards against
physically impossible stock/percentage combinations (e.g., 22% Laemmli or 18-20% Tricine with
30% Acrylamide/Bis stock).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


# Canonical table column order for UI DataFrame display
RECIPE_DISPLAY_COLUMNS: List[str] = [
    "Reagent",
    "Stock",
    "Final",
    "Volume (mL)",
    "Volume (µL)",
    "Pipette Display",
    "Order/Role",
]


# Supported SDS-PAGE buffer / gel systems
GEL_SYSTEMS: Dict[str, Dict[str, Any]] = {
    "laemmli": {
        "key": "laemmli",
        "name": "Tris-Glycine SDS-PAGE (Laemmli 1970)",
        "short_name": "Laemmli (Tris-Glycine)",
        "reference": "Laemmli, U.K. (1970) Nature 227:680–685; Bio-Rad Mini-PROTEAN Handcast Protocol",
        "min_gel_pct": 4.0,
        "max_gel_pct": 22.0,
        "default_stacking_pct": 4.0,
        "stacking_pct_range": (3.0, 6.0),
        "recommended_mw_range_kda": (10.0, 300.0),
        "supported_stocks": [30.0, 40.0],
        "default_stock_pct": 30.0,
        "resolving_ph": 8.8,
        "stacking_ph": 6.8,
    },
    "tricine": {
        "key": "tricine",
        "name": "Tricine-SDS-PAGE (Schägger 2006)",
        "short_name": "Tricine (Schägger)",
        "reference": "Schägger, H. (2006) Nature Protocols 1(1):16–22; Schägger & von Jagow (1987) Anal. Biochem. 166:368–379",
        "min_gel_pct": 10.0,
        "max_gel_pct": 20.0,
        "default_stacking_pct": 4.0,
        "stacking_pct_range": (3.5, 5.0),
        "recommended_mw_range_kda": (1.0, 30.0),
        "supported_stocks": [30.0, 40.0, 49.5],
        "default_stock_pct": 40.0,
        "resolving_ph": 8.45,
        "stacking_ph": 8.45,
    },
}


# Standard commercial and textbook Acrylamide:Bis-acrylamide stock solutions
ACRYLAMIDE_STOCK_PRESETS: Dict[float, Dict[str, Any]] = {
    30.0: {
        "pct": 30.0,
        "label": "30% (w/v) Acrylamide/Bis (37.5:1, 2.67% C)",
        "short_label": "30% (37.5:1)",
        "ratio": "37.5:1",
        "crosslinker_c_pct": 2.67,
        "notes": "Standard general-purpose Laemmli stock (e.g., Bio-Rad #1610158). Covers 4%–21.5% Laemmli and 10%–16% Tricine.",
    },
    40.0: {
        "pct": 40.0,
        "label": "40% (w/v) Acrylamide/Bis (29:1, 3.3% C)",
        "short_label": "40% (29:1)",
        "ratio": "29:1",
        "crosslinker_c_pct": 3.33,
        "notes": "Higher-concentration stock required for 22% Laemmli and 18%–20% Tricine (+ glycerol) gels.",
    },
    49.5: {
        "pct": 49.5,
        "label": "49.5% T, 3% C Acrylamide/Bis (Schägger AB-3, 32.3:1)",
        "short_label": "49.5% T (3% C)",
        "ratio": "32.3:1 (48 g Acr + 1.5 g Bis / 100 mL)",
        "crosslinker_c_pct": 3.0,
        "notes": "Canonical Schägger (2006) AB-3 stock for Tricine-SDS-PAGE (1–30 kDa). Use AB-6 (49.5% T, 6% C) for <5 kDa peptides.",
    },
}


# Standard cassette volume presets (Bio-Rad Mini-PROTEAN / Hoefer / Criterion)
# Includes practical ~15–20% casting margin above nominal plate cavity volume.
GEL_FORMAT_PRESETS: Dict[str, Dict[str, Any]] = {
    "mini_0.75mm": {
        "key": "mini_0.75mm",
        "label": "0.75 mm Mini-Gel (Bio-Rad Mini-PROTEAN / Hoefer)",
        "thickness_mm": 0.75,
        "resolving_vol_ml": 5.0,
        "stacking_vol_ml": 2.0,
        "dimensions_cm": "8.3 × 7.3 cm",
        "typical_well_load_ul": "10–15 µL (10-well) / 5–10 µL (15-well)",
        "notes": "Fastest electrophoresis & Western blot transfer; thinner slab requires gentle handling below 8% T.",
    },
    "mini_1.0mm": {
        "key": "mini_1.0mm",
        "label": "1.0 mm Mini-Gel (Bio-Rad Mini-PROTEAN — Standard)",
        "thickness_mm": 1.0,
        "resolving_vol_ml": 7.5,
        "stacking_vol_ml": 2.5,
        "dimensions_cm": "8.3 × 7.3 cm",
        "typical_well_load_ul": "15–25 µL (10-well) / 10–15 µL (15-well)",
        "notes": "Standard everyday format; ideal balance of band sharpness, durability, and sample load.",
    },
    "mini_1.5mm": {
        "key": "mini_1.5mm",
        "label": "1.5 mm Mini-Gel (Bio-Rad Mini-PROTEAN — High-Load)",
        "thickness_mm": 1.5,
        "resolving_vol_ml": 10.0,
        "stacking_vol_ml": 3.5,
        "dimensions_cm": "8.3 × 7.3 cm",
        "typical_well_load_ul": "25–40 µL (10-well) / 15–20 µL (15-well)",
        "notes": "High sample capacity for dilute fractions, Coomassie band excision, or fragile low-% (4%–6%) gels.",
    },
    "midi_1.0mm": {
        "key": "midi_1.0mm",
        "label": "1.0 mm Midi-Gel (Bio-Rad Criterion / Hoefer SE400)",
        "thickness_mm": 1.0,
        "resolving_vol_ml": 15.0,
        "stacking_vol_ml": 5.0,
        "dimensions_cm": "13.3 × 8.7 cm",
        "typical_well_load_ul": "20–30 µL (18-well) / 15–20 µL (26-well)",
        "notes": "Longer resolving distance and more lanes for closely spaced MW species or fraction screening.",
    },
    "custom": {
        "key": "custom",
        "label": "Custom Cassette Volume",
        "thickness_mm": None,
        "resolving_vol_ml": 7.5,
        "stacking_vol_ml": 2.5,
        "dimensions_cm": "User-defined",
        "typical_well_load_ul": "Varies by comb",
        "notes": "Specify custom per-gel resolving and stacking volumes (mL).",
    },
}


# Running buffer definitions for each gel system
RUNNING_BUFFERS: Dict[str, Dict[str, Any]] = {
    "laemmli": {
        "name": "1× Tris-Glycine-SDS Running Buffer (Laemmli)",
        "single_chamber": True,
        "ph": "8.3 (native, unadjusted)",
        "composition_1x": "25 mM Tris base, 192 mM Glycine, 0.1% (w/v) SDS, pH ~8.3",
        "anode_buffer": "1× Tris-Glycine-SDS: 25 mM Tris, 192 mM Glycine, 0.1% (w/v) SDS (pH ~8.3)",
        "cathode_buffer": "1× Tris-Glycine-SDS: 25 mM Tris, 192 mM Glycine, 0.1% (w/v) SDS (pH ~8.3)",
        "stock_10x_per_1l": (
            "10× Stock (1.0 L): 30.3 g Tris base + 144.1 g Glycine + 10.0 g SDS; "
            "dissolve in ~800 mL ddH2O and bring to 1.0 L. Dilute 100 mL 10× + 900 mL ddH2O for 1×."
        ),
        "run_conditions": (
            "Stack at 80–100 V constant (~15–20 min) until dye front enters resolving gel, "
            "then resolve at 150–180 V constant (~45–60 min) until Bromophenol Blue reaches ~0.5 cm from bottom."
        ),
        "notes": (
            "Do NOT adjust pH with HCl or NaOH — native pH (~8.3) is self-buffering. "
            "Adding acid/base raises ionic strength, causing overheating and smiling bands."
        ),
    },
    "tricine": {
        "name": "Tricine-SDS-PAGE Dual Running Buffers (Anode + Cathode, Schägger 2006)",
        "single_chamber": False,
        "ph": "Anode pH 8.9 / Cathode pH ~8.25",
        "composition_1x": (
            "Anode (lower tank): 0.1 M Tris-HCl, pH 8.9  |  "
            "Cathode (upper chamber): 0.1 M Tris, 0.1 M Tricine, 0.1% (w/v) SDS, pH ~8.25"
        ),
        "anode_buffer": (
            "1× Anode (outer/lower tank): 0.1 M Tris-HCl, pH 8.9 "
            "(10× stock per 1 L: 121.1 g Tris base, adjust to pH 8.9 with conc. HCl, ddH2O to 1.0 L)"
        ),
        "cathode_buffer": (
            "1× Cathode (inner/upper chamber): 0.1 M Tris, 0.1 M Tricine, 0.1% (w/v) SDS, pH ~8.25 "
            "(10× stock per 1 L: 121.1 g Tris base + 179.2 g Tricine + 10.0 g SDS, ddH2O to 1.0 L — do NOT adjust pH)"
        ),
        "stock_10x_per_1l": (
            "Anode 10× (1.0 L): 121.1 g Tris base (pH 8.9 with HCl)  |  "
            "Cathode 10× (1.0 L): 121.1 g Tris base + 179.2 g Tricine + 10.0 g SDS (native pH ~8.25). Dilute each 1:10."
        ),
        "run_conditions": (
            "Start at 30 V constant (~15–30 min) until sample fully enters stacking gel, then step to "
            "90–120 V constant (lower voltage than Laemmli — high 1.0 M Tris ionic strength produces heat; "
            "chill tank or run at 4 °C for >120 V)."
        ),
        "notes": (
            "Tricine requires TWO distinct buffers: fill inner/upper chamber with Cathode buffer "
            "(contains Tricine + SDS) and outer/lower tank with Anode buffer (0.1 M Tris pH 8.9, no SDS). "
            "Verify gasket seal so upper cathode buffer does not leak into lower anode tank."
        ),
    },
}


# Quick stock-solution preparation reference for bench sheets
STOCK_PREP_GUIDES: Dict[str, Dict[str, str]] = {
    "laemmli": {
        "1.5 M Tris-HCl (pH 8.8) — 500 mL": (
            "Dissolve 90.86 g Tris base in ~350 mL ddH2O, titrate to pH 8.8 with conc. HCl "
            "(cool to RT before final pH reading), bring to 500 mL with ddH2O, 0.22 µm filter; store at 4 °C."
        ),
        "0.5 M Tris-HCl (pH 6.8) — 500 mL": (
            "Dissolve 30.29 g Tris base in ~350 mL ddH2O, titrate to pH 6.8 with conc. HCl, "
            "bring to 500 mL with ddH2O, 0.22 µm filter; store at 4 °C."
        ),
        "10% (w/v) SDS — 100 mL": (
            "Dissolve 10.0 g electrophoresis-grade SDS in ~80 mL warm ddH2O (fume hood; avoid foaming), "
            "bring to 100 mL; store at room temperature (precipitates at 4 °C)."
        ),
        "10% (w/v) APS — 10 mL": (
            "Dissolve 1.0 g Ammonium Persulfate in 10.0 mL ddH2O; aliquot 200–500 µL into microfuge tubes "
            "at -20 °C (stable ~6 months) or keep at 4 °C for <= 1 week."
        ),
    },
    "tricine": {
        "3× Tricine Gel Buffer (3.0 M Tris-HCl, 0.3% SDS, pH 8.45) — 500 mL": (
            "Dissolve 181.7 g Tris base + 1.5 g SDS in ~300 mL ddH2O, titrate to pH 8.45 with conc. HCl "
            "(allow heat of neutralization to cool to RT), bring to 500 mL; store at room temperature (do not refrigerate SDS)."
        ),
        "49.5% T, 3% C Acrylamide/Bis (AB-3) — 100 mL": (
            "In fume hood: dissolve 48.0 g acrylamide + 1.5 g bis-acrylamide in ddH2O to 100 mL final volume "
            "(or use commercial 40% 29:1 stock); store amber bottle at 4 °C (warm briefly to dissolve crystals if needed)."
        ),
        "10% (w/v) APS — 10 mL": (
            "Dissolve 1.0 g Ammonium Persulfate in 10.0 mL ddH2O; store single-use aliquots at -20 °C "
            "or at 4 °C for <= 1 week."
        ),
    },
}


def _normalize_system(system: str) -> str:
    """Normalizes system identifier to canonical 'laemmli' or 'tricine'."""
    if not isinstance(system, str) or not system.strip():
        raise ValueError("Gel system must be 'laemmli' or 'tricine'.")
    key = system.strip().lower().replace(" ", "_").replace("-", "_")
    if key in ("laemmli", "tris_glycine", "trisglycine", "glycine"):
        return "laemmli"
    if key in ("tricine", "schagger", "schägger", "tricine_sds", "tricine_sds_page"):
        return "tricine"
    raise ValueError(
        f"Unsupported gel system {system!r}. Supported systems are 'laemmli' (Tris-Glycine) "
        f"and 'tricine' (Tricine-SDS-PAGE, Schägger 2006)."
    )


def _validate_finite_float(value: Any, name: str) -> float:
    """Coerces value to a finite float or raises a descriptive ValueError."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a numeric value, got boolean {value!r}.")
    try:
        fval = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric value, got {value!r}.") from exc
    if not math.isfinite(fval):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return fval


def _get_laemmli_resolving_temed_pct(gel_pct: float) -> Tuple[float, str]:
    """
    Returns gel-%-dependent TEMED concentration (% v/v) and rationale for Laemmli resolving gels
    (Sambrook & Russell Table 18-1 / Bio-Rad handcast calibration):
      - <= 7.5% T:  0.08% (v/v) (8 µL per 10 mL) — dilute monomer needs stronger catalysis to set completely
      - 8.0–10.5%:  0.06% (v/v) (6 µL per 10 mL) — balanced 20–30 min polymerization rate
      - >= 11.0% T: 0.04% (v/v) (4 µL per 10 mL) — prevents premature gelling & exothermic bubble formation
    """
    if gel_pct <= 7.5:
        return (
            0.08,
            "0.08% v/v TEMED for <=7.5% gels (dilute monomer requires higher catalyst for reliable, complete polymerization)",
        )
    if gel_pct <= 10.5:
        return (
            0.06,
            "0.06% v/v TEMED for 8%–10.5% gels (balanced ~20–30 min polymerization kinetics)",
        )
    return (
        0.04,
        "0.04% v/v TEMED for >=11% gels (high monomer concentration polymerizes rapidly & exothermically; lower TEMED prevents premature gelling and thermal bubbles)",
    )


def _get_stock_metadata(acrylamide_stock_pct: float) -> Dict[str, Any]:
    """Looks up standard stock preset metadata or synthesizes a clean custom label."""
    for preset_pct, meta in ACRYLAMIDE_STOCK_PRESETS.items():
        if abs(acrylamide_stock_pct - preset_pct) < 1e-6:
            return dict(meta)
    return {
        "pct": acrylamide_stock_pct,
        "label": f"{acrylamide_stock_pct:g}% (w/v) Acrylamide/Bis (custom stock)",
        "short_label": f"{acrylamide_stock_pct:g}% (custom)",
        "ratio": "Custom",
        "crosslinker_c_pct": None,
        "notes": f"Custom {acrylamide_stock_pct:g}% (w/v) Acrylamide/Bis stock solution.",
    }


def _fmt_clean_number(val: float, decimals: int = 1) -> str:
    """Formats a float with up to `decimals` places, trimming trailing '.0' when exact."""
    rounded = round(val, decimals)
    if abs(rounded - round(rounded)) < 1e-9:
        return f"{int(round(rounded))}"
    return f"{rounded:.{decimals}f}".rstrip("0").rstrip(".")


def _format_pipette_display(vol_ml: float) -> str:
    """
    Formats a reagent volume into a bench-ready pipetting display with recommended pipette class
    (P2, P10, P20, P200, P1000, or serological pipette).
    """
    vol_ul = vol_ml * 1000.0
    if vol_ml <= 1e-9:
        return "0 µL (none needed)"
    if vol_ul < 2.0:
        return f"{vol_ul:.1f} µL (P2)"
    if vol_ul < 10.0:
        return f"{vol_ul:.1f} µL (P10)"
    if vol_ul < 20.0:
        return f"{vol_ul:.1f} µL (P20)"
    if vol_ul < 200.0:
        return f"{_fmt_clean_number(vol_ul, 1)} µL (P200)"
    if vol_ul < 1000.0:
        return f"{_fmt_clean_number(vol_ul, 1)} µL (P1000)"
    if abs(vol_ml - 1.0) < 1e-4:
        return "1.00 mL (P1000)"
    if vol_ml <= 5.0:
        return f"{vol_ml:.2f} mL (5 mL pipet)"
    if vol_ml <= 10.0:
        return f"{vol_ml:.2f} mL (10 mL pipet)"
    if vol_ml <= 25.0:
        return f"{vol_ml:.2f} mL (25 mL pipet)"
    return f"{vol_ml:.2f} mL (50 mL pipet/cylinder)"


def _get_fixed_resolving_fraction(
    system: str,
    gel_pct: float,
    include_glycerol: bool,
    glycerol_pct: float,
) -> float:
    """Returns sum of all non-acrylamide, non-water volume fractions in the resolving gel."""
    if system == "laemmli":
        temed_pct, _ = _get_laemmli_resolving_temed_pct(gel_pct)
        # 1.5 M Tris (25%) + 10% SDS (1%) + 10% APS (1%) + TEMED (0.04–0.08%)
        return 0.25 + 0.01 + 0.01 + (temed_pct / 100.0)
    # Tricine (Schägger 2006): 3x Gel Buffer (1/3) + optional Glycerol + 10% APS (0.5%) + TEMED (0.05%)
    glyc_frac = (glycerol_pct / 100.0) if include_glycerol else 0.0
    return (1.0 / 3.0) + glyc_frac + 0.005 + 0.0005


def get_max_achievable_gel_pct(
    system: str = "laemmli",
    acrylamide_stock_pct: float = 30.0,
    include_glycerol: Optional[bool] = None,
    glycerol_pct: float = 10.0,
) -> float:
    """
    Calculates the theoretical maximum resolving gel %T achievable with a given Acrylamide/Bis
    stock before ddH2O volume reaches zero (capped by the system's valid %T upper limit).
    """
    sys_key = _normalize_system(system)
    stock = _validate_finite_float(acrylamide_stock_pct, "acrylamide_stock_pct")
    if stock <= 0:
        raise ValueError("acrylamide_stock_pct must be > 0.")

    if sys_key == "laemmli":
        # High-% Laemmli uses 0.04% TEMED -> fixed fraction = 0.2704
        fixed_frac = 0.25 + 0.01 + 0.01 + 0.0004
        raw_max = stock * (1.0 - fixed_frac)
        return round(min(GEL_SYSTEMS["laemmli"]["max_gel_pct"], max(0.0, raw_max)), 2)

    eff_glyc_pct = _validate_finite_float(glycerol_pct, "glycerol_pct")
    if eff_glyc_pct < 0.0 or eff_glyc_pct > 25.0:
        raise ValueError("glycerol_pct must be between 0.0 and 25.0% (v/v).")
    # Default Tricine ceiling assumes glycerol is enabled for >=16% unless explicitly disabled
    use_glyc = True if include_glycerol is None else bool(include_glycerol)
    fixed_frac = (1.0 / 3.0) + ((eff_glyc_pct / 100.0) if use_glyc else 0.0) + 0.005 + 0.0005
    raw_max = stock * (1.0 - fixed_frac)
    return round(min(GEL_SYSTEMS["tricine"]["max_gel_pct"], max(0.0, raw_max)), 2)


def get_linear_separation_range_kda(gel_pct: float, system: str = "laemmli") -> Tuple[float, float]:
    """
    Returns the approximate optimal linear MW separation window (low_kda, high_kda)
    for a given resolving gel percentage and buffer system (Bio-Rad / Sambrook / Schägger 2006).
    """
    sys_key = _normalize_system(system)
    pct = _validate_finite_float(gel_pct, "gel_pct")

    if sys_key == "tricine":
        if pct <= 11.0:
            return (5.0, 100.0)
        if pct <= 13.5:
            return (3.0, 70.0)
        if pct <= 16.5:
            return (1.0, 30.0)
        return (0.5, 18.0)

    # Laemmli (Tris-Glycine)
    if pct <= 5.5:
        return (100.0, 500.0)
    if pct <= 6.5:
        return (60.0, 250.0)
    if pct <= 8.5:
        return (40.0, 200.0)
    if pct <= 10.5:
        return (25.0, 150.0)
    if pct <= 13.0:
        return (15.0, 100.0)
    if pct <= 15.5:
        return (10.0, 60.0)
    if pct <= 18.5:
        return (8.0, 45.0)
    return (5.0, 30.0)


def recommend_gel_recipe_params(target_mw_kda: float) -> Dict[str, Any]:
    """
    Recommends the optimal SDS-PAGE gel system, resolving %T, and Acrylamide/Bis stock
    for a target protein molecular weight (kDa).
    """
    mw = _validate_finite_float(target_mw_kda, "target_mw_kda")
    if mw <= 0:
        raise ValueError("target_mw_kda must be > 0.")

    if mw < 6.0:
        return {
            "system": "tricine",
            "gel_pct": 18.0,
            "stacking_pct": 4.0,
            "acrylamide_stock_pct": 40.0,
            "include_glycerol": True,
            "rationale": (
                f"Ultra-low MW peptide ({mw:.1f} kDa < 6 kDa): 18% Tricine-SDS-PAGE (+10% glycerol, "
                f"40% or 49.5% T stock) resolves peptides sharply well above the dye front."
            ),
        }
    if mw < 10.0:
        return {
            "system": "tricine",
            "gel_pct": 16.0,
            "stacking_pct": 4.0,
            "acrylamide_stock_pct": 40.0,
            "include_glycerol": True,
            "rationale": (
                f"Low-MW peptide ({mw:.1f} kDa < 10 kDa): 16% Tricine-SDS-PAGE (+10% glycerol, Schägger 2006) "
                f"prevents SDS-micelle co-migration that blurs <10 kDa bands on Laemmli gels."
            ),
        }
    if mw < 22.0:
        return {
            "system": "laemmli",
            "gel_pct": 15.0,
            "stacking_pct": 4.0,
            "acrylamide_stock_pct": 30.0,
            "include_glycerol": False,
            "rationale": (
                f"Small protein ({mw:.1f} kDa): 15% Laemmli Tris-Glycine separates 10–60 kDa cleanly "
                f"(12%–16% Tricine is an excellent alternative near 10–15 kDa)."
            ),
        }
    if mw <= 110.0:
        return {
            "system": "laemmli",
            "gel_pct": 12.0,
            "stacking_pct": 4.0,
            "acrylamide_stock_pct": 30.0,
            "include_glycerol": False,
            "rationale": (
                f"Mid-range protein ({mw:.1f} kDa): 12% Laemmli Tris-Glycine places 20–100 kDa targets "
                f"squarely in the linear resolving zone."
            ),
        }
    if mw <= 250.0:
        return {
            "system": "laemmli",
            "gel_pct": 8.0,
            "stacking_pct": 4.0,
            "acrylamide_stock_pct": 30.0,
            "include_glycerol": False,
            "rationale": (
                f"Large protein ({mw:.1f} kDa): 8% Laemmli Tris-Glycine provides wide pores for clean "
                f"migration of 40–200 kDa proteins."
            ),
        }
    return {
        "system": "laemmli",
        "gel_pct": 6.0,
        "stacking_pct": 4.0,
        "acrylamide_stock_pct": 30.0,
        "include_glycerol": False,
        "rationale": (
            f"Very large protein ({mw:.1f} kDa > 250 kDa): 6% Laemmli (or 4%–5% for >350 kDa) ensures "
            f"the band migrates cleanly past the stacking/resolving boundary."
        ),
    }


def _build_reagent_row(
    reagent: str,
    stock: str,
    final: str,
    vol_ml: float,
    order_role: str,
) -> Dict[str, Any]:
    """Constructs a single standardized reagent row dict with exact mL/µL & pipette formatting."""
    clean_ml = round(vol_ml, 4)
    clean_ul = round(clean_ml * 1000.0, 1)
    return {
        "Reagent": reagent,
        "Stock": stock,
        "Final": final,
        "Volume (mL)": clean_ml,
        "Volume (µL)": clean_ul,
        "Pipette Display": _format_pipette_display(clean_ml),
        "Order/Role": order_role,
        "volume_ml": round(vol_ml, 6),
        "volume_ul": round(vol_ml * 1000.0, 3),
    }


def _build_conserved_rows(
    total_vol_ml: float,
    non_water_specs: List[Dict[str, Any]],
    water_order_role: str = "1. Base solvent (add first)",
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Computes reagent volumes with guaranteed exact volume conservation:
      water_ml = total_vol_ml - sum(non_water_vols)
    Non-water reagents are rounded to 0.0001 mL (0.1 µL, matching P2/P10 micropipette step size)
    before subtracting from total_vol_ml so both displayed Volume (mL), Volume (µL), and exact
    volume_ml sum to total_vol_ml within < 1e-4 mL (exact to floating-point precision for all
    standard cassette volumes).
    """
    raw_non_water_ml = [total_vol_ml * spec["fraction"] for spec in non_water_specs]
    raw_water_ml = total_vol_ml - sum(raw_non_water_ml)

    # Round non-water reagents half-up to 0.1 µL (0.0001 mL)
    rounded_non_water = [math.floor(v * 10000.0 + 0.5 + 1e-12) / 10000.0 for v in raw_non_water_ml]
    water_ml = total_vol_ml - sum(rounded_non_water)

    # Guard edge case where exact water >= 0 but banker's rounding nudged rounded sum > total_vol_ml
    if raw_water_ml >= -1e-9 and water_ml < 0.0:
        rounded_non_water = [math.floor(v * 10000.0 + 1e-9) / 10000.0 for v in raw_non_water_ml]
        water_ml = max(0.0, total_vol_ml - sum(rounded_non_water))

    rows: List[Dict[str, Any]] = [
        _build_reagent_row(
            reagent="ddH2O (Ultrapure Water)",
            stock="18.2 MΩ·cm",
            final="To volume",
            vol_ml=water_ml,
            order_role=water_order_role,
        )
    ]

    for spec, vol_ml in zip(non_water_specs, rounded_non_water):
        rows.append(
            _build_reagent_row(
                reagent=spec["reagent"],
                stock=spec["stock"],
                final=spec["final"],
                vol_ml=vol_ml,
                order_role=spec["order_role"],
            )
        )

    total_sum_ml = round(sum(r["volume_ml"] for r in rows), 6)
    return rows, total_sum_ml


def calculate_sds_gel_recipe(
    gel_pct: float,
    system: str = "laemmli",
    acrylamide_stock_pct: float = 30.0,
    resolving_vol_ml: float = 7.5,
    stacking_vol_ml: float = 2.5,
    stacking_pct: float = 4.0,
    num_gels: int = 1,
    target_mw_kda: Optional[float] = None,
    *,
    include_glycerol: Optional[bool] = None,
    glycerol_pct: float = 10.0,
    auto_upgrade_stock: bool = False,
) -> Dict[str, Any]:
    """
    Computes exact, volume-conserved SDS-PAGE casting recipes for BOTH resolving and stacking gels.

    Parameters
    ----------
    gel_pct : float
        Target resolving gel acrylamide percentage (% T).
        - Laemmli: 4.0% to 22.0% T
        - Tricine: 10.0% to 20.0% T
    system : str, default "laemmli"
        Buffer system: "laemmli" (Tris-Glycine, Laemmli 1970) or "tricine" (Schägger 2006).
    acrylamide_stock_pct : float, default 30.0
        Acrylamide:Bis stock concentration (% w/v), e.g., 30.0 (37.5:1), 40.0 (29:1), or
        49.5 (Schägger AB-3, 3% C).
    resolving_vol_ml : float, default 7.5
        Per-gel resolving volume in mL (e.g., 5.0 for 0.75 mm, 7.5 for 1.0 mm, 10.0 for 1.5 mm).
    stacking_vol_ml : float, default 2.5
        Per-gel stacking volume in mL (e.g., 2.0 for 0.75 mm, 2.5 for 1.0 mm, 3.5 for 1.5 mm).
    stacking_pct : float, default 4.0
        Stacking gel acrylamide percentage (% T), typically 4.0% or 5.0% (Laemmli: 3.0–6.0%,
        Tricine: 3.5–5.0%, canonical 4.0%).
    num_gels : int, default 1
        Number of gels to cast in a single master mix (integer >= 1).
    target_mw_kda : float, optional
        Target protein molecular weight in kDa for automated system/percentage advisories
        (e.g., auto-suggests Tricine-SDS-PAGE if target_mw_kda < 10.0 on Laemmli).
    include_glycerol : bool, optional
        Tricine only: None (default) automatically includes glycerol for >=16.0% T resolving gels;
        True/False explicitly enables or disables glycerol. Ignored for Laemmli.
    glycerol_pct : float, default 10.0
        Final glycerol concentration (% v/v) when glycerol is included in Tricine resolving gels
        (10.0%–13.0% v/v recommended per Schägger 2006).
    auto_upgrade_stock : bool, default False
        If False (strict lab-safety default), impossible combinations leaving negative water
        (e.g., 22% Laemmli or 18–20% Tricine with 30% stock) raise a clean ValueError advising
        40% or 49.5% Acrylamide/Bis stock. If True, automatically upgrades to the minimum standard
        stock (40.0% or 49.5%) that yields non-negative water and records a prominent warning.

    Returns
    -------
    Dict[str, Any]
        Structured recipe dictionary containing `resolving_rows`, `stacking_rows`,
        `running_buffer`, `stock_prep_guide`, `warnings`, `bench_notes`, `safety_notes`,
        and full casting metadata.
    """
    sys_key = _normalize_system(system)
    sys_spec = GEL_SYSTEMS[sys_key]

    pct = _validate_finite_float(gel_pct, "gel_pct")
    min_pct = float(sys_spec["min_gel_pct"])
    max_pct = float(sys_spec["max_gel_pct"])
    if pct < min_pct or pct > max_pct:
        raise ValueError(
            f"{sys_spec['short_name']} resolving gel percentage must be between "
            f"{min_pct:g}% and {max_pct:g}% T (received {pct:g}%)."
        )

    stk_pct = _validate_finite_float(stacking_pct, "stacking_pct")
    stk_min, stk_max = sys_spec["stacking_pct_range"]
    if stk_pct < stk_min or stk_pct > stk_max:
        raise ValueError(
            f"{sys_spec['short_name']} stacking gel percentage must be between "
            f"{stk_min:g}% and {stk_max:g}% T (received {stk_pct:g}%)."
        )

    req_stock = _validate_finite_float(acrylamide_stock_pct, "acrylamide_stock_pct")
    if req_stock <= 0.0 or req_stock > 60.0:
        raise ValueError(
            f"acrylamide_stock_pct must be between >0% and 60% (w/v) (received {req_stock:g}%)."
        )

    res_per_gel = _validate_finite_float(resolving_vol_ml, "resolving_vol_ml")
    stk_per_gel = _validate_finite_float(stacking_vol_ml, "stacking_vol_ml")
    if res_per_gel <= 0.0:
        raise ValueError(f"resolving_vol_ml must be > 0 (received {res_per_gel:g}).")
    if stk_per_gel <= 0.0:
        raise ValueError(f"stacking_vol_ml must be > 0 (received {stk_per_gel:g}).")

    if isinstance(num_gels, bool) or not isinstance(num_gels, int) or num_gels < 1:
        raise ValueError(f"num_gels must be an integer >= 1 (received {num_gels!r}).")

    mw_kda: Optional[float] = None
    if target_mw_kda is not None:
        mw_kda = _validate_finite_float(target_mw_kda, "target_mw_kda")
        if mw_kda <= 0.0:
            raise ValueError(f"target_mw_kda must be > 0 (received {mw_kda:g}).")

    # Glycerol handling (Tricine resolving gels only)
    eff_glycerol_pct = _validate_finite_float(glycerol_pct, "glycerol_pct")
    if eff_glycerol_pct < 0.0 or eff_glycerol_pct > 20.0:
        raise ValueError(
            f"glycerol_pct must be between 0.0% and 20.0% (v/v) (received {eff_glycerol_pct:g}%)."
        )
    if sys_key == "tricine":
        if include_glycerol is None:
            eff_include_glycerol = (pct >= 16.0) and (eff_glycerol_pct > 0.0)
        else:
            eff_include_glycerol = bool(include_glycerol) and (eff_glycerol_pct > 0.0)
        if not eff_include_glycerol:
            eff_glycerol_pct = 0.0
    else:
        eff_include_glycerol = False
        eff_glycerol_pct = 0.0

    # Total master-mix volumes across all gels
    resolving_total_ml = round(res_per_gel * num_gels, 6)
    stacking_total_ml = round(stk_per_gel * num_gels, 6)

    # Check resolving + stacking water feasibility (negative water guard)
    res_fixed_frac = _get_fixed_resolving_fraction(
        sys_key, pct, eff_include_glycerol, eff_glycerol_pct
    )
    if sys_key == "laemmli":
        stk_fixed_frac = 0.25 + 0.01 + 0.01 + 0.001
    else:
        stk_fixed_frac = 0.25 + 0.0075 + 0.00075

    def _water_fraction_for_stock(stock_pct: float) -> Tuple[float, float]:
        res_w = 1.0 - ((pct / stock_pct) + res_fixed_frac)
        stk_w = 1.0 - ((stk_pct / stock_pct) + stk_fixed_frac)
        return res_w, stk_w

    eff_stock = req_stock
    stock_upgraded = False
    res_water_frac, stk_water_frac = _water_fraction_for_stock(eff_stock)

    if res_water_frac < -1e-9 or stk_water_frac < -1e-9:
        # Determine minimum standard stock that makes the recipe physically possible
        candidate_stocks = [40.0, 49.5] if sys_key == "tricine" else [40.0, 49.5]
        viable_stock: Optional[float] = None
        for cand in candidate_stocks:
            if cand > req_stock:
                cw_res, cw_stk = _water_fraction_for_stock(cand)
                if cw_res >= -1e-9 and cw_stk >= -1e-9:
                    viable_stock = cand
                    break

        max_with_req = get_max_achievable_gel_pct(
            system=sys_key,
            acrylamide_stock_pct=req_stock,
            include_glycerol=eff_include_glycerol,
            glycerol_pct=eff_glycerol_pct if eff_include_glycerol else 10.0,
        )
        deficit_ml = abs(min(0.0, res_water_frac) * resolving_total_ml)
        non_water_pct = (1.0 - res_water_frac) * 100.0

        if auto_upgrade_stock and viable_stock is not None:
            eff_stock = viable_stock
            stock_upgraded = True
            res_water_frac, stk_water_frac = _water_fraction_for_stock(eff_stock)
        else:
            if viable_stock is not None:
                advice = (
                    f"Switch to {viable_stock:g}% Acrylamide/Bis stock (recommended)"
                    + (" or 49.5% T (3% C) Schägger AB-3 stock" if sys_key == "tricine" and viable_stock < 49.5 else "")
                    + f", or reduce resolving gel % to <= {max_with_req:g}% T"
                    + (" (or disable glycerol)." if eff_include_glycerol else ".")
                )
            else:
                advice = (
                    f"Reduce resolving gel % (<= {max_with_req:g}% T with {req_stock:g}% stock)"
                    + (" or reduce/disable glycerol." if eff_include_glycerol else ".")
                )
            raise ValueError(
                f"Impossible gel formulation: a {pct:g}% {sys_spec['short_name']} resolving gel "
                f"cannot be prepared using {req_stock:g}% Acrylamide/Bis stock "
                f"(non-water reagents require {non_water_pct:.2f}% of total volume, leaving "
                f"-{deficit_ml:.3f} mL water across {resolving_total_ml:g} mL). {advice}"
            )

    stock_meta = _get_stock_metadata(eff_stock)
    stock_short = stock_meta["short_label"]
    stock_full = stock_meta["label"]

    # Build resolving and stacking reagent specifications
    if sys_key == "laemmli":
        res_temed_pct, temed_rationale = _get_laemmli_resolving_temed_pct(pct)
        stk_temed_pct = 0.10

        resolving_specs = [
            {
                "reagent": f"Acrylamide/Bis {stock_short}",
                "stock": stock_full,
                "final": f"{pct:g}% T",
                "fraction": pct / eff_stock,
                "order_role": "2. Monomer / crosslinker matrix",
            },
            {
                "reagent": "Tris-HCl (pH 8.8)",
                "stock": "1.5 M (pH 8.8)",
                "final": "375 mM",
                "fraction": 0.25,
                "order_role": "3. Resolving gel buffer (25% v/v)",
            },
            {
                "reagent": "SDS",
                "stock": "10% (w/v)",
                "final": "0.1% (w/v)",
                "fraction": 0.01,
                "order_role": "4. Anionic denaturant (mix gently — avoid foam)",
            },
            {
                "reagent": "APS (Ammonium Persulfate)",
                "stock": "10% (w/v), fresh",
                "final": "0.1% (w/v)",
                "fraction": 0.01,
                "order_role": "5. Radical initiator (add just before pouring)",
            },
            {
                "reagent": "TEMED",
                "stock": "Neat (100%)",
                "final": f"{res_temed_pct:.2f}% (v/v)",
                "fraction": res_temed_pct / 100.0,
                "order_role": "6. Catalyst — ADD LAST, invert gently & pour immediately",
            },
        ]

        stacking_specs = [
            {
                "reagent": f"Acrylamide/Bis {stock_short}",
                "stock": stock_full,
                "final": f"{stk_pct:g}% T",
                "fraction": stk_pct / eff_stock,
                "order_role": "2. Large-pore stacking matrix",
            },
            {
                "reagent": "Tris-HCl (pH 6.8)",
                "stock": "0.5 M (pH 6.8)",
                "final": "125 mM",
                "fraction": 0.25,
                "order_role": "3. Stacking gel buffer (25% v/v)",
            },
            {
                "reagent": "SDS",
                "stock": "10% (w/v)",
                "final": "0.1% (w/v)",
                "fraction": 0.01,
                "order_role": "4. Anionic denaturant (avoid bubbles)",
            },
            {
                "reagent": "APS (Ammonium Persulfate)",
                "stock": "10% (w/v), fresh",
                "final": "0.1% (w/v)",
                "fraction": 0.01,
                "order_role": "5. Radical initiator (add just before pouring)",
            },
            {
                "reagent": "TEMED",
                "stock": "Neat (100%)",
                "final": f"{stk_temed_pct:.2f}% (v/v)",
                "fraction": stk_temed_pct / 100.0,
                "order_role": "6. Catalyst — ADD LAST, pour & insert comb immediately",
            },
        ]
    else:
        res_temed_pct = 0.05
        stk_temed_pct = 0.075
        temed_rationale = (
            "0.05% v/v TEMED + 0.05% w/v APS in resolving gel (0.075% each in 4% stacking gel) "
            "per Schägger (2006) Nature Protocols 1:16–22 Table 1"
        )

        resolving_specs = [
            {
                "reagent": f"Acrylamide/Bis {stock_short}",
                "stock": stock_full,
                "final": f"{pct:g}% T",
                "fraction": pct / eff_stock,
                "order_role": "2. Monomer / crosslinker matrix",
            },
            {
                "reagent": "3× Tricine Gel Buffer (pH 8.45)",
                "stock": "3.0 M Tris-HCl, 0.3% SDS",
                "final": "1.0 M Tris, 0.1% SDS",
                "fraction": 1.0 / 3.0,
                "order_role": "3. Gel buffer + SDS (33.3% v/v; no separate SDS needed)",
            },
        ]
        next_step = 4
        if eff_include_glycerol:
            resolving_specs.append(
                {
                    "reagent": "Glycerol (anhydrous)",
                    "stock": "100% (neat)",
                    "final": f"{eff_glycerol_pct:g}% (v/v)",
                    "fraction": eff_glycerol_pct / 100.0,
                    "order_role": f"{next_step}. Viscosity/density enhancer (sharpens <10 kDa bands)",
                }
            )
            next_step += 1

        resolving_specs.extend(
            [
                {
                    "reagent": "APS (Ammonium Persulfate)",
                    "stock": "10% (w/v), fresh",
                    "final": "0.05% (w/v)",
                    "fraction": 0.005,
                    "order_role": f"{next_step}. Radical initiator (0.5% v/v of 10% stock; add just before pouring)",
                },
                {
                    "reagent": "TEMED",
                    "stock": "Neat (100%)",
                    "final": f"{res_temed_pct:.2f}% (v/v)",
                    "fraction": res_temed_pct / 100.0,
                    "order_role": f"{next_step + 1}. Catalyst — ADD LAST, swirl gently & pour immediately",
                },
            ]
        )

        stacking_specs = [
            {
                "reagent": f"Acrylamide/Bis {stock_short}",
                "stock": stock_full,
                "final": f"{stk_pct:g}% T",
                "fraction": stk_pct / eff_stock,
                "order_role": "2. Large-pore stacking matrix",
            },
            {
                "reagent": "3× Tricine Gel Buffer (pH 8.45)",
                "stock": "3.0 M Tris-HCl, 0.3% SDS",
                "final": "0.75 M Tris, 0.075% SDS",
                "fraction": 0.25,
                "order_role": "3. Lower-ionic-strength stacking buffer (25% v/v, Schägger Table 1)",
            },
            {
                "reagent": "APS (Ammonium Persulfate)",
                "stock": "10% (w/v), fresh",
                "final": "0.075% (w/v)",
                "fraction": 0.0075,
                "order_role": "4. Radical initiator (0.75% v/v of 10% stock; add just before pouring)",
            },
            {
                "reagent": "TEMED",
                "stock": "Neat (100%)",
                "final": f"{stk_temed_pct:.3f}% (v/v)",
                "fraction": stk_temed_pct / 100.0,
                "order_role": "5. Catalyst — ADD LAST, pour & insert comb immediately",
            },
        ]

    resolving_rows, resolving_sum_ml = _build_conserved_rows(resolving_total_ml, resolving_specs)
    stacking_rows, stacking_sum_ml = _build_conserved_rows(stacking_total_ml, stacking_specs)

    # Hard internal conservation assertion (guarantees contract on every returned recipe)
    if abs(resolving_sum_ml - resolving_total_ml) >= 1e-4:
        raise RuntimeError(
            f"Internal volume conservation check failed for resolving gel: "
            f"{resolving_sum_ml} != {resolving_total_ml}"
        )
    if abs(stacking_sum_ml - stacking_total_ml) >= 1e-4:
        raise RuntimeError(
            f"Internal volume conservation check failed for stacking gel: "
            f"{stacking_sum_ml} != {stacking_total_ml}"
        )

    # Contextual warnings & advisories
    warnings: List[str] = []
    if stock_upgraded:
        warnings.append(
            f"Stock auto-upgraded from {req_stock:g}% to {eff_stock:g}% Acrylamide/Bis: "
            f"a {pct:g}% {sys_spec['short_name']} gel leaves negative water with {req_stock:g}% stock. "
            f"Verify you are pipetting from the {eff_stock:g}% bottle before mixing!"
        )

    if mw_kda is not None:
        if mw_kda < 10.0 and sys_key == "laemmli":
            warnings.append(
                f"Low-MW peptide ({mw_kda:.1f} kDa < 10 kDa): On Tris-Glycine (Laemmli) gels, "
                f"peptides below ~10 kDa co-migrate with the SDS micelle / dye front as diffuse bands. "
                f"Switch to Tricine-SDS-PAGE (16%–18% T with 10% glycerol, Schägger 2006) or "
                f"precast Bis-Tris/MES for crisp peptide resolution."
            )
        elif mw_kda < 5.0 and sys_key == "tricine" and pct < 16.0:
            warnings.append(
                f"Ultra-low MW peptide ({mw_kda:.1f} kDa < 5 kDa): Increase Tricine resolving gel "
                f"to 16%–20% T with 10% glycerol (ideally 49.5% T / 6% C AB-6 stock) for sharp bands."
            )
        elif mw_kda > 100.0 and sys_key == "tricine":
            warnings.append(
                f"High-MW protein ({mw_kda:.1f} kDa > 100 kDa): Tricine-SDS-PAGE is tailored for "
                f"1–30 kDa proteins/peptides. Switch to a 6%–8% Laemmli (Tris-Glycine) gel."
            )
        elif mw_kda > 150.0 and sys_key == "laemmli" and pct > 10.0:
            warnings.append(
                f"Large protein ({mw_kda:.1f} kDa) on a {pct:g}% Laemmli gel will migrate slowly near "
                f"the top of the resolving slab; consider 6%–8% T for optimal separation."
            )
        elif 10.0 <= mw_kda < 20.0 and sys_key == "laemmli" and pct < 12.0:
            warnings.append(
                f"Small protein ({mw_kda:.1f} kDa) on a {pct:g}% Laemmli gel will run very close to "
                f"the dye front; consider raising to 15% Laemmli (or 12%–16% Tricine)."
            )

    if 0.0 <= res_water_frac < 0.05:
        warnings.append(
            f"Tight water margin ({res_water_frac * 100.0:.1f}% of resolving volume): Mixing is "
            f"easier and more accurate with a higher-concentration stock "
            f"({'40% (29:1) or 49.5% T (3% C)' if sys_key == 'tricine' else '40% (29:1)'})."
        )

    if sys_key == "tricine" and pct >= 16.0 and not eff_include_glycerol:
        warnings.append(
            f"Glycerol is disabled on a {pct:g}% Tricine gel: Schägger (2006) recommends 10%–13% (v/v) "
            f"glycerol in >=16% resolving gels to sharpen <10 kDa bands and aid overlay layering."
        )

    res_temed_ul = resolving_rows[-1]["Volume (µL)"]
    stk_temed_ul = stacking_rows[-1]["Volume (µL)"]
    min_temed_ul = min(res_temed_ul, stk_temed_ul)
    if min_temed_ul < 2.0:
        warnings.append(
            f"Very small TEMED volume ({min_temed_ul:.1f} µL): Use a calibrated P2/P10 pipette, "
            f"immerse tip just below the meniscus and pipette up/down 2–3× to flush residual TEMED "
            f"(or prepare master mix for 2+ gels)."
        )

    # Practical wet-lab bench notes
    bench_notes: List[str] = []
    if mw_kda is not None and mw_kda < 10.0 and sys_key == "laemmli":
        bench_notes.append(
            f"System suggestion for {mw_kda:.1f} kDa target: Switch to Tricine-SDS-PAGE "
            f"(16%–18% T + 10% glycerol, Schägger 2006) for sharp resolution below 10 kDa."
        )

    bench_notes.append(
        "Order of addition: Mix ddH2O, Acrylamide/Bis, and buffer first; add APS and TEMED LAST "
        "immediately before pouring. Invert/swirl gently 5–6 times (never vortex — dissolved O2 "
        "inhibits free-radical polymerization and traps bubbles)."
    )
    bench_notes.append(f"TEMED calibration: {temed_rationale}.")
    bench_notes.append(
        "Meniscus overlay: Immediately layer 0.5–1.0 mL isopropanol, water-saturated n-butanol, "
        "or ddH2O onto the poured resolving gel to exclude O2 and form a flat interface. Allow "
        "30–45 min polymerization at RT, decant, rinse 3× with ddH2O, and wick dry with filter paper "
        "before pouring the stacking gel."
    )
    bench_notes.append(
        "APS potency: Use freshly prepared 10% (w/v) APS (4 °C <= 1 week) or single-use -20 °C "
        "aliquots; degraded APS is the primary cause of slow, uneven, or failed gel polymerization."
    )

    if pct >= 15.0:
        bench_notes.append(
            f"High-% gel ({pct:g}% T): Degas the monomer/buffer mixture under vacuum for 5–10 min "
            f"(or chill briefly on ice) before adding APS + TEMED to reduce exothermic heating and "
            f"prevent micro-bubbles or premature gelling."
        )
    elif pct <= 7.0:
        bench_notes.append(
            f"Low-% gel ({pct:g}% T): Cast slab is soft and elastic — use 1.0–1.5 mm spacers when "
            f"possible and pry glass plates apart gently under running buffer or transfer buffer."
        )

    if pct <= stk_pct:
        bench_notes.append(
            f"Matched %T ({pct:g}% resolving / {stk_pct:g}% stacking): Protein stacking still occurs "
            f"via the discontinuous pH 6.8 / pH 8.8 chloride-glycine ion boundary (Kohlrausch effect)."
        )

    if sys_key == "tricine":
        bench_notes.append(
            "No separate 10% SDS tube: In Tricine-SDS-PAGE, the 3× Gel Buffer (3.0 M Tris-HCl, "
            "0.3% SDS, pH 8.45) already supplies the exact final 0.1% SDS in the resolving gel "
            "(and 0.075% SDS in the 4% stacking gel)."
        )
        if eff_include_glycerol:
            bench_notes.append(
                "Pipetting neat glycerol: 100% glycerol is viscous — aspirate slowly with a wide-bore "
                "or trimmed tip (or weigh 1.26 g per mL). If using 50% (v/v) glycerol stock instead, "
                "double the glycerol volume and subtract that extra volume from ddH2O."
            )
        if pct >= 16.0:
            bench_notes.append(
                "Optional Schägger (2006) spacer gel: For ultra-sharp separation of 1–15 kDa peptides, "
                "you may overlay the resolving gel with a 1–2 cm 10% T Tricine spacer gel prior to the "
                "4% stacking gel (routine mini-gels also resolve cleanly with direct 4% stacking)."
            )

    # EHS Laboratory Safety Notes
    safety_notes: List[str] = [
        "NEUROTOXIN / CARCINOGEN (Acrylamide): Unpolymerized acrylamide monomer is a potent cumulative "
        "neurotoxin and probable human carcinogen absorbed through skin and inhalation. Always wear "
        "nitrile gloves, lab coat, and safety glasses, and dispense pre-made liquid stock solutions "
        "inside a certified chemical fume hood (avoid weighing dry acrylamide powder).",
        "TEMED (Flammable / Corrosive): Neat TEMED is volatile, corrosive, and pungent — pipette inside "
        "a fume hood and cap tightly immediately after use.",
        "SDS (Respiratory Irritant): Prepare 10% (w/v) SDS stock solutions from powder inside a fume "
        "hood or while wearing a particulate respirator.",
        "Waste Disposal: Polymerize leftover liquid acrylamide casting mix completely in its tube "
        "before discarding. Handle polymerized gels with gloves (trace residual monomer may remain) "
        "and dispose in designated chemical/solid lab waste.",
    ]

    return {
        "system": sys_key,
        "system_name": sys_spec["name"],
        "system_short_name": sys_spec["short_name"],
        "reference": sys_spec["reference"],
        "gel_pct": pct,
        "stacking_pct": stk_pct,
        "acrylamide_stock_pct": eff_stock,
        "acrylamide_stock_label": stock_full,
        "requested_acrylamide_stock_pct": req_stock,
        "stock_upgraded": stock_upgraded,
        "num_gels": num_gels,
        "resolving_vol_per_gel_ml": res_per_gel,
        "stacking_vol_per_gel_ml": stk_per_gel,
        "resolving_total_vol_ml": resolving_total_ml,
        "stacking_total_vol_ml": stacking_total_ml,
        "total_combined_vol_ml": round(resolving_total_ml + stacking_total_ml, 6),
        "resolving_temed_pct": res_temed_pct,
        "stacking_temed_pct": stk_temed_pct,
        "include_glycerol": eff_include_glycerol,
        "glycerol_pct": eff_glycerol_pct,
        "target_mw_kda": mw_kda,
        "linear_range_kda": get_linear_separation_range_kda(pct, sys_key),
        "resolving_rows": resolving_rows,
        "stacking_rows": stacking_rows,
        "resolving_sum_ml": resolving_sum_ml,
        "stacking_sum_ml": stacking_sum_ml,
        "running_buffer": dict(RUNNING_BUFFERS[sys_key]),
        "stock_prep_guide": dict(STOCK_PREP_GUIDES[sys_key]),
        "warnings": warnings,
        "bench_notes": bench_notes,
        "safety_notes": safety_notes,
    }


def _format_ascii_table(rows: List[Dict[str, Any]], total_ml: float) -> str:
    """Renders a clean fixed-width text table for bench printing / download."""
    headers = ["Reagent", "Stock", "Final", "Vol (mL)", "Vol (µL)", "Pipette", "Order / Role"]
    formatted_rows: List[List[str]] = []
    for r in rows:
        formatted_rows.append(
            [
                str(r["Reagent"]),
                str(r["Stock"]),
                str(r["Final"]),
                f"{r['Volume (mL)']:.4f}",
                f"{r['Volume (µL)']:.1f}",
                str(r["Pipette Display"]),
                str(r["Order/Role"]),
            ]
        )
    formatted_rows.append(
        [
            "TOTAL",
            "—",
            "—",
            f"{total_ml:.4f}",
            f"{total_ml * 1000.0:.1f}",
            f"{total_ml:.2f} mL total",
            "Exact conserved volume",
        ]
    )

    widths = [
        max(len(headers[i]), max(len(row[i]) for row in formatted_rows))
        for i in range(len(headers))
    ]
    align_right = {3, 4}

    def _fmt_line(vals: List[str]) -> str:
        cells = []
        for i, v in enumerate(vals):
            cells.append(v.rjust(widths[i]) if i in align_right else v.ljust(widths[i]))
        return " | ".join(cells)

    sep = "-+-".join("-" * w for w in widths)
    lines = [_fmt_line(headers), sep]
    for row in formatted_rows[:-1]:
        lines.append(_fmt_line(row))
    lines.append(sep)
    lines.append(_fmt_line(formatted_rows[-1]))
    return "\n".join(lines)


def format_gel_recipe_text(recipe_dict: Dict[str, Any]) -> str:
    """
    Formats a recipe dictionary produced by `calculate_sds_gel_recipe` into a clean,
    bench-ready plain-text casting sheet for 1-click download or lab notebook printing.
    """
    if not isinstance(recipe_dict, dict) or "resolving_rows" not in recipe_dict or "stacking_rows" not in recipe_dict:
        raise ValueError("recipe_dict must be a valid dictionary returned by calculate_sds_gel_recipe().")

    sys_name = recipe_dict["system_name"]
    gel_pct = recipe_dict["gel_pct"]
    stk_pct = recipe_dict["stacking_pct"]
    num_gels = recipe_dict["num_gels"]
    res_per = recipe_dict["resolving_vol_per_gel_ml"]
    stk_per = recipe_dict["stacking_vol_per_gel_ml"]
    res_tot = recipe_dict["resolving_total_vol_ml"]
    stk_tot = recipe_dict["stacking_total_vol_ml"]
    low_mw, high_mw = recipe_dict["linear_range_kda"]
    gel_word = "gel" if num_gels == 1 else "gels"

    lines: List[str] = [
        "=" * 88,
        f"SDS-PAGE HANDCAST GEL RECIPE SHEET — {sys_name}",
        "=" * 88,
        f"Reference          : {recipe_dict['reference']}",
        f"Gel Formulation    : {gel_pct:g}% T Resolving  /  {stk_pct:g}% T Stacking"
        + (f"  (+{recipe_dict['glycerol_pct']:g}% v/v Glycerol)" if recipe_dict.get("include_glycerol") else ""),
        f"Acrylamide Stock   : {recipe_dict['acrylamide_stock_label']}",
        f"Casting Scale      : {num_gels} {gel_word} "
        f"(Resolving: {res_per:g} mL × {num_gels} = {res_tot:g} mL | "
        f"Stacking: {stk_per:g} mL × {num_gels} = {stk_tot:g} mL)",
        f"Optimal MW Window  : ~{low_mw:g}–{high_mw:g} kDa (linear separation range)",
    ]

    if recipe_dict.get("target_mw_kda") is not None:
        lines.append(f"Target Protein MW  : {recipe_dict['target_mw_kda']:.2f} kDa")

    warnings = recipe_dict.get("warnings") or []
    if warnings:
        lines.extend(["", ">>> ADVISORIES & WARNINGS <<<"])
        for w in warnings:
            lines.append(f"  [!] {w}")

    lines.extend(
        [
            "",
            f"1. RESOLVING GEL — {gel_pct:g}% T  (Total: {res_tot:g} mL for {num_gels} {gel_word})",
            "-" * 88,
            _format_ascii_table(recipe_dict["resolving_rows"], res_tot),
            "",
            f"2. STACKING GEL — {stk_pct:g}% T  (Total: {stk_tot:g} mL for {num_gels} {gel_word})",
            "-" * 88,
            _format_ascii_table(recipe_dict["stacking_rows"], stk_tot),
        ]
    )

    rb = recipe_dict.get("running_buffer") or {}
    if rb:
        lines.extend(
            [
                "",
                f"3. RUNNING BUFFER — {rb.get('name', 'SDS-PAGE Running Buffer')}",
                "-" * 88,
                f"  1× Composition : {rb.get('composition_1x', '')}",
                f"  pH             : {rb.get('ph', '')}",
            ]
        )
        if not rb.get("single_chamber", True):
            lines.append(f"  Anode (Lower)  : {rb.get('anode_buffer', '')}")
            lines.append(f"  Cathode (Upper): {rb.get('cathode_buffer', '')}")
        else:
            lines.append(f"  10× Stock Prep : {rb.get('stock_10x_per_1l', '')}")
        lines.append(f"  Run Protocol   : {rb.get('run_conditions', '')}")
        if rb.get("notes"):
            lines.append(f"  Buffer Note    : {rb['notes']}")

    stock_guide = recipe_dict.get("stock_prep_guide") or {}
    if stock_guide:
        lines.extend(["", "4. STOCK SOLUTION PREPARATION QUICK REFERENCE", "-" * 88])
        for stock_title, stock_desc in stock_guide.items():
            lines.append(f"  • {stock_title}: {stock_desc}")

    bench_notes = recipe_dict.get("bench_notes") or []
    if bench_notes:
        lines.extend(["", "5. BENCH CASTING PROTOCOL & TIPS", "-" * 88])
        for idx, note in enumerate(bench_notes, start=1):
            lines.append(f"  {idx}. {note}")

    safety_notes = recipe_dict.get("safety_notes") or []
    if safety_notes:
        lines.extend(["", "6. LABORATORY SAFETY (EHS)", "-" * 88])
        for note in safety_notes:
            lines.append(f"  * {note}")

    lines.append("=" * 88)
    return "\n".join(lines) + "\n"


__all__ = [
    "RECIPE_DISPLAY_COLUMNS",
    "GEL_SYSTEMS",
    "ACRYLAMIDE_STOCK_PRESETS",
    "GEL_FORMAT_PRESETS",
    "RUNNING_BUFFERS",
    "STOCK_PREP_GUIDES",
    "calculate_sds_gel_recipe",
    "format_gel_recipe_text",
    "get_max_achievable_gel_pct",
    "get_linear_separation_range_kda",
    "recommend_gel_recipe_params",
]
