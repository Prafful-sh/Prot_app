"""
Size Exclusion Chromatography (SEC) Biophysical Simulation Module (v3.1)
========================================================================
Models hydrodynamic Stokes radii (Rh), accessible intraparticle pore partitioning
(Laurent–Killander / Ackers cumulative log-normal pore size distribution with solid
gel-matrix volume exclusion), plate-theory band broadening (N = 16 * [Ve / W_base]^2),
and dual-wavelength UV (A280 nm / A260 nm) chromatograms for proteins, oligomers,
intrinsically disordered proteins (IDPs), protein–protein complexes, and
protein–DNA/RNA (RNP/chromatin) assemblies across 3 kDa to 3+ MDa.

References & Calibration Sources:
- Cytiva Gel Filtration Calibration Kits (LMW 28-4038-41 & HMW 28-4038-42) and Column Datasheets
  (Superdex 75/200 Increase, Superose 6 Increase, HiLoad 16/600 & 26/600 pg)
- Erickson, H. P. (2009) Biol. Proced. Online 11:32–51 (PMC3055910) & La Verde et al. (2017) Bio-protocol 7:e2230
- Uversky, V. N. (1993) Biochemistry 32:13288–13298 & Uversky, V. N. (2002) Protein Sci. 11:739–756
- Wilkins, D. K. et al. (1999) Biochemistry 38:16424–16431
- Laurent, T. C. & Killander, J. (1964) J. Chromatogr. 14:317–330; Ackers, G. K. (1967) J. Biol. Chem. 242:3237–3238
- Tirado, M. M. & García de la Torre, J. (1980) J. Chem. Phys. 73:1986–1993 (rod/DNA hydrodynamics)
"""

import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# Calibrated quadratic log-hydrodynamic scaling for folded globular biomolecules (1.35 kDa – 3.0 MDa, R^2 = 0.9950):
# ln(Rh_Angstrom) = _RH_A0 + _RH_A1 * ln(MW_kDa) + _RH_A2 * [ln(MW_kDa)]^2
# Agrees closely with Uversky (1993, Biochemistry 32:13288-13298: Rs = 0.557 * M_Da^0.369) and
# Wilkins et al. (1999, Biochemistry 38:16424-16431: Rh = 4.75 * N_aa^0.29) across 5–200 kDa while
# converging smoothly toward spherical M^(1/3) scaling for MegaDalton cages and ribosomes.
_RH_A0 = 1.925344
_RH_A1 = 0.381009
_RH_A2 = -0.001666

# Conformation / Hydrodynamic Shape Profiles
CONFORMATION_PROFILES: Dict[str, Dict[str, Any]] = {
    "globular": {
        "name": "Compact Globular Fold (Standard Sphere/Cage)",
        "short": "Globular",
        "f_shape": 1.00,
        "description": "Typical single-domain globular enzymes, compact oligomers, spherical cages (Ferritin, Ribosome, Nanoparticles)."
    },
    "moderately_elongated": {
        "name": "Moderately Elongated / Multi-Domain (e.g., IgG, Fusion Tag)",
        "short": "Multi-Domain",
        "f_shape": 1.14,
        "description": "Two- or three-domain proteins connected by flexible linkers, Y-shaped antibodies, or dimeric coiled-coils."
    },
    "elongated": {
        "name": "Elongated / Asymmetric Rod-like (e.g., Full-length Parkin, Fibrillar)",
        "short": "Elongated",
        "f_shape": 1.24,
        "description": "Extended multi-domain scaffolds or asymmetric rods (frictional ratio f/f0 ~ 1.5); elutes noticeably earlier than spherical globes."
    },
    "glycoprotein_elongated": {
        "name": "Glycosylated Asymmetric Dimer (e.g., Thyroglobulin)",
        "short": "Glycoprotein",
        "f_shape": 1.16,
        "description": "Hydrated branched glycans increase hydrodynamic Stokes radius relative to bare polypeptide mass."
    },
    "idp_hybrid": {
        "name": "Folded Core + Disordered IDP Tails (~30–45% Disordered)",
        "short": "Core + IDP Tails",
        "f_shape": 1.36,
        "description": "Folded catalytic/binding core flanked by long disordered N/C-terminal tails (e.g., p53, full-length METTL3–METTL14)."
    },
    "disordered_idp": {
        "name": "Intrinsically Disordered Protein (Native Coil, Uversky Scaling)",
        "short": "IDP / Unfolded Coil",
        "f_shape": None,
        "description": "Uversky (2002) native-coil polymer hydrodynamics (log10 Rh = 0.493*log10 MW_Da - 0.551, i.e. Rh = 0.281 * MW_Da^0.493 Å); e.g., 14.5 kDa alpha-synuclein (Rh ~ 31.6 Å) elutes like a ~55 kDa globular protein."
    },
    "compact_pna": {
        "name": "Compact Protein–DNA/RNA Complex (RNP / Nucleosome)",
        "short": "Compact RNP/NCP",
        "f_shape": 1.04,
        "description": "Nucleic acid wrapped around or cradled inside protein scaffold (e.g., Nucleosome Core Particle, SpCas9–sgRNA, CRISPR Cascade)."
    },
    "extended_pna": {
        "name": "Extended Protein–DNA/RNA Complex (Protruding Duplex Overhangs)",
        "short": "Extended PNA",
        "f_shape": 1.20,
        "description": "Transcription factor or polymerase bound to long flanking DNA/RNA duplexes that project into solvent."
    },
    "dsdna": {
        "name": "Free Double-Stranded DNA (Tirado–García de la Torre Stiff Rod)",
        "short": "Free dsDNA",
        "f_shape": None,
        "description": "Rigid B-form DNA double helix (3.4 Å/bp, ~618 Da/bp, persistence length ~500 Å); has a much larger Stokes radius than a protein of equal mass."
    },
    "ssrna": {
        "name": "Free Structured Single-Stranded RNA (tRNA / sgRNA / crRNA)",
        "short": "Free ssRNA",
        "f_shape": None,
        "description": "Secondary-structure folded RNA (~321.5 Da/nt) with strong 260 nm UV absorbance (A260/A280 = 2.00)."
    }
}

# Standard molecular weight markers & experimental Stokes radii (Rh in Å) from Cytiva LMW/HMW Gel Filtration
# Calibration Kits (28-4038-41 / 28-4038-42), Erickson (2009, PMC3055910), and La Verde et al. (2017, PMC5507634)
SEC_STANDARD_MARKERS = [
    {"name": "Blue Dextran 2000 (Void, Vo)", "mw_kda": 2000.0, "rh_angstrom": 270.0, "color": "rgba(41, 128, 185, 0.45)", "is_void": True, "conformation": "globular"},
    {"name": "70S Ribosome", "mw_kda": 2350.0, "rh_angstrom": 116.0, "color": "rgba(109, 40, 217, 0.65)", "conformation": "globular", "megadalton_only": True},
    {"name": "GroEL (14-mer)", "mw_kda": 800.0, "rh_angstrom": 79.5, "color": "rgba(124, 58, 237, 0.65)", "conformation": "globular", "megadalton_only": True},
    {"name": "Thyroglobulin", "mw_kda": 669.0, "rh_angstrom": 85.0, "color": "rgba(142, 68, 173, 0.65)", "conformation": "glycoprotein_elongated"},
    {"name": "Ferritin", "mw_kda": 440.0, "rh_angstrom": 61.0, "color": "rgba(230, 126, 34, 0.65)", "conformation": "globular"},
    {"name": "Aldolase", "mw_kda": 158.0, "rh_angstrom": 48.1, "color": "rgba(39, 174, 96, 0.65)", "conformation": "globular"},
    {"name": "Conalbumin", "mw_kda": 75.0, "rh_angstrom": 36.4, "color": "rgba(241, 196, 15, 0.65)", "conformation": "globular"},
    {"name": "Ovalbumin", "mw_kda": 44.0, "rh_angstrom": 30.5, "color": "rgba(231, 76, 60, 0.65)", "conformation": "globular"},
    {"name": "Carbonic Anhydrase", "mw_kda": 29.0, "rh_angstrom": 23.6, "color": "rgba(22, 160, 133, 0.65)", "conformation": "globular"},
    {"name": "Ribonuclease A", "mw_kda": 13.7, "rh_angstrom": 16.4, "color": "rgba(52, 73, 94, 0.65)", "conformation": "globular"},
    {"name": "Aprotinin", "mw_kda": 6.5, "rh_angstrom": 13.5, "color": "rgba(127, 140, 141, 0.65)", "conformation": "globular"},
    {"name": "Vitamin B12", "mw_kda": 1.355, "rh_angstrom": 8.5, "color": "rgba(236, 72, 153, 0.60)", "conformation": "globular"},
]

# Calibrated Cytiva SEC Column Database (Ackers Log-Normal Pore Model + Accessible Liquid Volume V_perm)
# Note on plate_count_n: Cytiva's factory QC plate spec (qc_acetone_plates_n: >43,000–48,000 N/m for Increase,
# >13,000 N/m for HiLoad pg) is measured using 1% acetone (58 Da). For macromolecules (10–1,000+ kDa),
# slower intraparticle pore diffusion (van Deemter C-term) yields effective protein plate counts N_protein
# ~2,100–3,800, matching real ÄKTA FPLC base peak widths (W_base = 4*sigma ~ 0.8–1.6 mL on 24 mL columns; ~5–9 mL on 120 mL columns).
SEC_COLUMNS: Dict[str, Dict[str, Any]] = {
    "hiload_16_600_s200": {
        "id": "hiload_16_600_s200",
        "name": "HiLoad 16/600 Superdex 200 pg",
        "type": "Preparative (Prep Grade)",
        "bed_volume_vt_ml": 120.0,
        "void_volume_vo_ml": 40.5,
        "liquid_perm_volume_ml": 106.763,
        "rh50_angstrom": 34.522,
        "sigma_pore": 0.7802,
        "opt_range_min_kda": 10.0,
        "opt_range_max_kda": 600.0,
        "exclusion_limit_kda": 1300.0,
        "dimensions": "16 mm × 600 mm",
        "bead_size_um": 34,
        "typical_flow_rate_ml_min": 1.0,
        "plate_count_n": 2200,
        "qc_acetone_plates_n": 7800,
        "equivalent_analytical": "s200_increase_10_300"
    },
    "s200_increase_10_300": {
        "id": "s200_increase_10_300",
        "name": "Superdex 200 Increase 10/300 GL",
        "type": "Analytical (High Resolution)",
        "bed_volume_vt_ml": 24.0,
        "void_volume_vo_ml": 8.05,
        "liquid_perm_volume_ml": 21.771,
        "rh50_angstrom": 32.657,
        "sigma_pore": 0.8083,
        "opt_range_min_kda": 10.0,
        "opt_range_max_kda": 600.0,
        "exclusion_limit_kda": 1300.0,
        "dimensions": "10 mm × 300 mm",
        "bead_size_um": 8.6,
        "typical_flow_rate_ml_min": 0.5,
        "plate_count_n": 3800,
        "qc_acetone_plates_n": 14400,
        "equivalent_preparative": "hiload_16_600_s200"
    },
    "hiload_16_600_s75": {
        "id": "hiload_16_600_s75",
        "name": "HiLoad 16/600 Superdex 75 pg",
        "type": "Preparative (Prep Grade)",
        "bed_volume_vt_ml": 120.0,
        "void_volume_vo_ml": 40.0,
        "liquid_perm_volume_ml": 109.062,
        "rh50_angstrom": 15.922,
        "sigma_pore": 0.6680,
        "opt_range_min_kda": 3.0,
        "opt_range_max_kda": 70.0,
        "exclusion_limit_kda": 100.0,
        "dimensions": "16 mm × 600 mm",
        "bead_size_um": 34,
        "typical_flow_rate_ml_min": 1.0,
        "plate_count_n": 2200,
        "qc_acetone_plates_n": 7800,
        "equivalent_analytical": "s75_increase_10_300"
    },
    "s75_increase_10_300": {
        "id": "s75_increase_10_300",
        "name": "Superdex 75 Increase 10/300 GL",
        "type": "Analytical (High Resolution)",
        "bed_volume_vt_ml": 24.0,
        "void_volume_vo_ml": 7.80,
        "liquid_perm_volume_ml": 21.785,
        "rh50_angstrom": 15.922,
        "sigma_pore": 0.6680,
        "opt_range_min_kda": 3.0,
        "opt_range_max_kda": 70.0,
        "exclusion_limit_kda": 100.0,
        "dimensions": "10 mm × 300 mm",
        "bead_size_um": 8.6,
        "typical_flow_rate_ml_min": 0.5,
        "plate_count_n": 3600,
        "qc_acetone_plates_n": 12900,
        "equivalent_preparative": "hiload_16_600_s75"
    },
    "superose_6_increase_10_300": {
        "id": "superose_6_increase_10_300",
        "name": "Superose 6 Increase 10/300 GL",
        "type": "Analytical (MegaDalton Range)",
        "bed_volume_vt_ml": 24.0,
        "void_volume_vo_ml": 8.10,
        "liquid_perm_volume_ml": 20.198,
        "rh50_angstrom": 72.875,
        "sigma_pore": 0.9432,
        "opt_range_min_kda": 5.0,
        "opt_range_max_kda": 5000.0,
        "exclusion_limit_kda": 40000.0,
        "dimensions": "10 mm × 300 mm",
        "bead_size_um": 8.6,
        "typical_flow_rate_ml_min": 0.5,
        "plate_count_n": 3600,
        "qc_acetone_plates_n": 14400,
        "equivalent_preparative": "hiload_16_600_superose_6"
    },
    "hiload_16_600_superose_6": {
        "id": "hiload_16_600_superose_6",
        "name": "HiLoad 16/600 Superose 6 pg",
        "type": "Preparative (MegaDalton Range)",
        "bed_volume_vt_ml": 120.0,
        "void_volume_vo_ml": 40.5,
        "liquid_perm_volume_ml": 100.990,
        "rh50_angstrom": 72.875,
        "sigma_pore": 0.9432,
        "opt_range_min_kda": 5.0,
        "opt_range_max_kda": 5000.0,
        "exclusion_limit_kda": 40000.0,
        "dimensions": "16 mm × 600 mm",
        "bead_size_um": 30,
        "typical_flow_rate_ml_min": 1.0,
        "plate_count_n": 2100,
        "qc_acetone_plates_n": 7800,
        "equivalent_analytical": "superose_6_increase_10_300"
    },
    "hiload_26_600_s200": {
        "id": "hiload_26_600_s200",
        "name": "HiLoad 26/600 Superdex 200 pg",
        "type": "Large Preparative",
        "bed_volume_vt_ml": 320.0,
        "void_volume_vo_ml": 108.0,
        "liquid_perm_volume_ml": 284.701,
        "rh50_angstrom": 34.522,
        "sigma_pore": 0.7802,
        "opt_range_min_kda": 10.0,
        "opt_range_max_kda": 600.0,
        "exclusion_limit_kda": 1300.0,
        "dimensions": "26 mm × 600 mm",
        "bead_size_um": 34,
        "typical_flow_rate_ml_min": 2.6,
        "plate_count_n": 2100,
        "qc_acetone_plates_n": 7800,
        "equivalent_analytical": "s200_increase_10_300"
    },
    "superose_6_increase_3_2_300": {
        "id": "superose_6_increase_3_2_300",
        "name": "Superose 6 Increase 3.2/300",
        "type": "Micro-Analytical (Cryo-EM Scale)",
        "bed_volume_vt_ml": 2.4,
        "void_volume_vo_ml": 0.81,
        "liquid_perm_volume_ml": 2.020,
        "rh50_angstrom": 72.875,
        "sigma_pore": 0.9432,
        "opt_range_min_kda": 5.0,
        "opt_range_max_kda": 5000.0,
        "exclusion_limit_kda": 40000.0,
        "dimensions": "3.2 mm × 300 mm",
        "bead_size_um": 8.6,
        "typical_flow_rate_ml_min": 0.04,
        "plate_count_n": 3400,
        "qc_acetone_plates_n": 14400,
        "equivalent_preparative": "hiload_16_600_superose_6"
    }
}


def _get_extra_column_dispersion_ml(column_config: Dict[str, Any]) -> float:
    """
    Returns realistic extra-column & sample injection loop standard deviation sigma_inj (mL)
    on ÄKTA FPLC systems (variance = V_loop^2 / 12 + tubing/UV flow-cell dispersion):
      - 2.4 mL micro-analytical (3.2/300, ~10–25 µL load): ~0.015 mL
      - 24 mL analytical (10/300 GL, ~200–350 µL load):    ~0.15 mL -> W_base ~ 0.80–1.55 mL
      - 120 mL HiLoad 16/600 pg (~1.5–2.5 mL load):        ~0.72 mL -> W_base ~ 5.0–8.9 mL
      - 320 mL HiLoad 26/600 pg (~4–6 mL load):            ~1.90 mL -> W_base ~ 13.5–23.8 mL
    """
    col_id_low = column_config["id"].lower()
    if "26_600" in col_id_low:
        return 1.90
    elif "hiload" in col_id_low:
        return 0.72
    elif "3_2_300" in col_id_low:
        return 0.015
    return 0.15


def predict_stokes_radius_rh(
    mw_kda: float,
    conformation: str = "globular",
    na_fraction: float = 0.0
) -> float:
    """
    Predicts macromolecular hydrodynamic Stokes radius Rh (in Angstroms)
    from molecular weight (kDa) and 3D structural conformation.
    - Folded globular: quadratic log-MW calibration (1.35 kDa – 3 MDa, R^2 = 0.9950; consistent with
      Uversky 1993 Biochemistry 32:13288 & Wilkins et al. 1999 Biochemistry 38:16424).
    - Disordered IDP (native coil): Uversky (2002, Protein Sci. 11:739–756) log10(Rh) = 0.493*log10(M_Da) - 0.551
      (i.e. Rh = 0.28119 * M_Da^0.493 Å).
    - Free B-form dsDNA: Tirado & García de la Torre (1980) rod hydrodynamics with Kratky–Porod worm-like
      chain persistence-length correction (3.4 Å/bp, 617.96 Da/bp, Lp = 500 Å).
    """
    if mw_kda <= 0:
        return 1.0

    lmw = math.log(max(0.2, mw_kda))
    rh_glob = math.exp(_RH_A0 + _RH_A1 * lmw + _RH_A2 * (lmw ** 2))
    mw_da = mw_kda * 1000.0

    if conformation == "disordered_idp":
        # Uversky (2002, Protein Sci. 11:739-756) native-coil IDP scaling: 10^(-0.551) * M_Da^0.493
        return (10.0 ** -0.551) * (mw_da ** 0.493)
    elif conformation == "dsdna":
        n_bp = max(6.0, mw_kda / 0.61796)
        length_a = 3.4 * n_bp
        diam_a = 22.0
        aspect = max(1.2, length_a / diam_a)
        denom = math.log(aspect) + 0.312 + 0.565 / aspect - 0.100 / (aspect * aspect)
        rh_rod = (length_a / 2.0) / denom
        wlc_factor = (1.0 + length_a / (3.0 * 500.0)) ** (-0.22)
        return rh_rod * wlc_factor
    elif conformation == "ssrna":
        return 0.720 * (mw_da ** 0.365)
    elif conformation == "extended_pna":
        return rh_glob * (1.14 + 0.18 * min(1.0, max(0.0, na_fraction)))

    prof = CONFORMATION_PROFILES.get(conformation)
    if prof and prof.get("f_shape"):
        return rh_glob * float(prof["f_shape"])

    return rh_glob


def calculate_dual_uv_ratio(
    prot_mw_kda: float,
    prot_eps01: float = 1.0,
    na_mw_kda: float = 0.0,
    na_type: Optional[str] = None
) -> Dict[str, float]:
    """
    Computes physically grounded relative A280 and A260 detector signals and the A260/A280
    chromophore ratio for pure proteins, pure nucleic acids, and protein-DNA/RNA complexes
    using standard 1 mg/mL (0.1% w/v, 1 cm pathlength) mass extinction coefficients:
    - Pure folded protein: A260/A280 = 0.57 (canonical range 0.55–0.60; Glasel 1995 / Layne 1957)
    - Pure dsDNA (50 µg/mL = 1.0 A260): E260(0.1%) = 20.0 L/(g*cm), A260/A280 = 1.85 -> E280(0.1%) = 10.81
    - Pure ssRNA (40 µg/mL = 1.0 A260): E260(0.1%) = 25.0 L/(g*cm), A260/A280 = 2.00 -> E280(0.1%) = 12.50
    - Pure ssDNA (37 µg/mL = 1.0 A260): E260(0.1%) = 27.0 L/(g*cm), A260/A280 = 1.80 -> E280(0.1%) = 15.00
    """
    eff_eps_prot = max(0.15, float(prot_eps01)) if prot_mw_kda > 0 else 0.0
    a280_prot = max(0.0, prot_mw_kda) * eff_eps_prot
    a260_prot = a280_prot * 0.57

    a280_na = 0.0
    a260_na = 0.0
    if na_mw_kda > 0 and na_type:
        nt = na_type.lower()
        if "rna" in nt:
            a260_na = na_mw_kda * 25.0
            a280_na = na_mw_kda * 12.50
        elif "ssdna" in nt:
            a260_na = na_mw_kda * 27.0
            a280_na = na_mw_kda * 15.00
        else:
            a260_na = na_mw_kda * 20.0
            a280_na = na_mw_kda * (20.0 / 1.85)

    tot_280 = a280_prot + a280_na
    tot_260 = a260_prot + a260_na
    if tot_280 <= 1e-9:
        return {"a260_a280_ratio": 0.57, "a260_multiplier": 0.57}

    ratio = tot_260 / tot_280
    return {
        "a260_a280_ratio": round(ratio, 2),
        "a260_multiplier": round(ratio, 4)
    }


def recommend_optimal_columns(mw_kda: float) -> Tuple[str, str]:
    """Returns recommended (preparative_col_id, analytical_col_id) for a given biomolecule MW."""
    if mw_kda < 48.0:
        return "hiload_16_600_s75", "s75_increase_10_300"
    elif mw_kda <= 480.0:
        return "hiload_16_600_s200", "s200_increase_10_300"
    else:
        return "hiload_16_600_superose_6", "superose_6_increase_10_300"


def calculate_retention_volume(
    mw_kda: float,
    column_config: Dict[str, Any],
    conformation: str = "globular",
    rh_override_a: Optional[float] = None,
    na_fraction: float = 0.0
) -> Tuple[float, float, str]:
    """
    Calculates theoretical elution volume (Ve in mL) and geometric partition coefficient (Kav)
    using hydrodynamic Stokes radius Rh and Ackers log-normal pore size distribution:
        Ve(Rh) = Vo + (V_perm - Vo) * 0.5 * erfc( ln(Rh / Rh50) / (sqrt(2) * sigma_pore) )
        Kav    = (Ve - Vo) / (Vt - Vo)
    Returns: (Ve_ml, Kav, status_note)
    """
    vt = float(column_config["bed_volume_vt_ml"])
    vo = float(column_config["void_volume_vo_ml"])
    v_perm = float(column_config.get("liquid_perm_volume_ml", vt * 0.90))
    rh50 = float(column_config.get("rh50_angstrom", 32.657))
    sigma_pore = float(column_config.get("sigma_pore", 0.8083))
    max_kda = float(column_config["opt_range_max_kda"])
    min_kda = float(column_config["opt_range_min_kda"])
    excl_kda = float(column_config.get("exclusion_limit_kda", max_kda * 2.0))

    if mw_kda <= 0:
        return round(vo, 2), 0.0, "Invalid MW"

    rh_a = float(rh_override_a) if (rh_override_a is not None and rh_override_a > 0) else predict_stokes_radius_rh(
        mw_kda, conformation=conformation, na_fraction=na_fraction
    )

    # Check physical void-volume exclusion limit (based on both mass and equivalent exclusion Rh)
    rh_excl = predict_stokes_radius_rh(excl_kda, "globular")
    if rh_a >= rh_excl * 1.08 or mw_kda >= excl_kda * 1.15:
        return round(vo, 2), 0.0, f"Excluded (> {excl_kda:.0f} kDa limit; Rh = {rh_a:.1f} Å). Elutes in Void Volume (Vo)."

    z = (math.log(rh_a) - math.log(rh50)) / (math.sqrt(2.0) * sigma_pore)
    pore_access = 0.5 * math.erfc(z)

    ve = vo + (v_perm - vo) * pore_access
    if ve <= vo + 0.002 * vt:
        return round(vo, 2), 0.0, f"Exclusion limit reached (Rh = {rh_a:.1f} Å). Elutes at Void Volume (Vo)."

    ve = max(vo, min(vt, ve))
    kav = (ve - vo) / (vt - vo)

    dec = 3 if vt < 5.0 else 2
    ve_out = round(ve, dec)
    kav_out = round(kav, 3)

    if mw_kda > max_kda:
        note = f"Above optimal range ({max_kda:.0f} kDa); partial pore exclusion (Rh = {rh_a:.1f} Å, Kav = {kav_out})."
    elif mw_kda < min_kda:
        note = f"Below optimal range (< {min_kda:.0f} kDa); near salt/permeation boundary (Rh = {rh_a:.1f} Å)."
    else:
        note = f"Optimal fractionation window (Rh = {rh_a:.1f} Å, Kav = {kav_out})."

    return ve_out, kav_out, note


def get_column_marker_elutions(column_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Calculates elution volumes for all relevant Cytiva / MegaDalton standard calibration markers
    on the given column using their experimentally measured Stokes radii (Rh in Å).
    """
    results = []
    max_kda = float(column_config["opt_range_max_kda"])
    min_kda = float(column_config["opt_range_min_kda"])
    is_megadalton = max_kda >= 1500.0

    for marker in SEC_STANDARD_MARKERS:
        if marker.get("megadalton_only") and not is_megadalton:
            continue

        if marker.get("is_void"):
            results.append({
                "name": marker["name"],
                "mw_kda": marker["mw_kda"],
                "rh_angstrom": float(marker.get("rh_angstrom", 270.0)),
                "ve_ml": column_config["void_volume_vo_ml"],
                "kav": 0.0,
                "color": marker["color"],
                "is_void": True,
                "note": "Void Volume Marker"
            })
            continue

        if marker["mw_kda"] <= max_kda * 1.25 and marker["mw_kda"] >= min_kda * 0.40:
            conf = marker.get("conformation", "globular")
            exp_rh = marker.get("rh_angstrom")
            rh_a = float(exp_rh) if exp_rh is not None else predict_stokes_radius_rh(marker["mw_kda"], conf)
            ve, kav, _ = calculate_retention_volume(
                marker["mw_kda"],
                column_config,
                conformation=conf,
                rh_override_a=rh_a
            )
            results.append({
                "name": marker["name"],
                "mw_kda": marker["mw_kda"],
                "rh_angstrom": round(rh_a, 1),
                "ve_ml": ve,
                "kav": kav,
                "color": marker["color"],
                "is_void": False,
                "note": f"{marker['mw_kda']:g} kDa"
            })

    return results


def simulate_chromatogram_peaks(
    target_states: List[Dict[str, Any]],
    column_config: Dict[str, Any],
    resolution_points: int = 700
) -> Dict[str, Any]:
    """
    Generates high-resolution simulated SEC chromatogram profiles for both UV 280 nm and UV 260 nm.
    Models peak width from chromatographic plate theory:
        sigma_col = Ve / sqrt(N_protein)
        sigma     = sqrt(sigma_col^2 + sigma_inj^2)
        W_base    = 4 * sigma   <=>   N_eff = 16 * (Ve / W_base)^2
    Each item in target_states supports:
      - label (str), mw_kda (float), abundance (float), color (str)
      - optional: conformation (str), prot_mw_kda (float), na_mw_kda (float), na_type ('dsdna'|'ssrna'),
        prot_eps01 (float), is_aggregate (bool)
    """
    vt = float(column_config["bed_volume_vt_ml"])
    vo = float(column_config["void_volume_vo_ml"])
    plate_count = float(column_config["plate_count_n"])
    excl_kda_col = float(column_config.get("exclusion_limit_kda", 1300.0))
    dec = 3 if vt < 5.0 else 2

    x_max = vt * 1.12
    volumes = np.linspace(0.0, x_max, resolution_points)
    total_signal_280 = np.zeros_like(volumes)
    total_signal_260 = np.zeros_like(volumes)

    v_perm = float(column_config.get("liquid_perm_volume_ml", vt * 0.90))
    vinj_dispersion = _get_extra_column_dispersion_ml(column_config)
    individual_peaks = []

    for state in target_states:
        label = state["label"]
        mw = float(state["mw_kda"])
        abundance = float(state.get("abundance", 1.0))
        color = state.get("color", "#2563EB")
        conformation = state.get("conformation", "globular")
        is_aggregate = bool(state.get("is_aggregate", False))

        prot_mw = float(state.get("prot_mw_kda", mw if conformation not in ("dsdna", "ssrna") else 0.0))
        na_mw = float(state.get("na_mw_kda", mw if conformation in ("dsdna", "ssrna") else 0.0))
        na_type = state.get("na_type", conformation if conformation in ("dsdna", "ssrna") else None)
        raw_prot_eps01 = state.get("prot_eps01", 1.0)
        prot_eps01 = 1.0 if raw_prot_eps01 is None else float(raw_prot_eps01)
        uv280_invisible = (
            "prot_eps01" in state
            and raw_prot_eps01 is not None
            and prot_eps01 == 0.0
            and prot_mw > 0.0
            and na_mw == 0.0
        )

        rh_override = state.get("rh_override_a")
        if is_aggregate:
            excl_kda = excl_kda_col
            mw = max(excl_kda * 1.25, mw)
            if conformation not in ("dsdna", "ssrna") and "prot_mw_kda" not in state:
                prot_mw = mw
            rh_excl = predict_stokes_radius_rh(excl_kda, conformation="globular")
            rh_a = (
                float(rh_override)
                if (rh_override is not None and float(rh_override) > 0)
                else max(predict_stokes_radius_rh(mw, conformation="globular"), rh_excl * 1.10)
            )
            ve = round(vo, dec)
            kav = 0.0
            note = (
                f"Soluble HMW aggregate (> {excl_kda:g} kDa exclusion limit); "
                f"100% pore-excluded at Void Volume (Vo = {ve} mL, Kav = 0.000)."
            )
            # Mild polydispersity of soluble HMW void aggregates (~18% broader intraparticle/interparticle envelope)
            col_sigma = (ve / math.sqrt(plate_count)) * 1.18
        else:
            excl_kda = excl_kda_col
            na_frac = na_mw / max(1e-6, mw)
            rh_a = (
                float(rh_override)
                if (rh_override is not None and float(rh_override) > 0)
                else predict_stokes_radius_rh(mw, conformation=conformation, na_fraction=na_frac)
            )
            ve, kav, note = calculate_retention_volume(
                mw, column_config, conformation=conformation, rh_override_a=rh_override, na_fraction=na_frac
            )
            # Plate theory (sigma_col = Ve / sqrt(N)) + extra-column / sample-loop variance
            col_sigma = ve / math.sqrt(plate_count)

        sigma = math.sqrt(col_sigma ** 2 + vinj_dispersion ** 2)
        w_base = 4.0 * sigma
        eff_plates = int(round(16.0 * ((ve / max(1e-6, w_base)) ** 2)))

        if uv280_invisible:
            uv_blind_warn = (
                "⚠️ Zero Trp/Tyr/S–S (ε280 = 0): invisible at 280 nm on ÄKTA FPLC — "
                "monitor 214/205 nm or engineer a C-terminal Trp/Tyr!"
            )
            note = f"{note} {uv_blind_warn}" if note else uv_blind_warn

        uv_info = calculate_dual_uv_ratio(prot_mw, prot_eps01=prot_eps01, na_mw_kda=na_mw, na_type=na_type)
        a260_ratio = uv_info["a260_a280_ratio"]
        if is_aggregate:
            # Rayleigh/Mie UV light scattering (~ lambda^-4) elevates apparent A260/A280 of pure-protein
            # void aggregates from ~0.57 to ~0.92 (classic ÄKTA SEC diagnostic for misfolded HMW aggregates)
            a260_ratio = round(max(a260_ratio, 0.92), 2)

        # Scale A280 peak height (mAU); if A260 > A280 (nucleic-acid rich), normalize so complex stays cleanly on scale
        base_h_280 = 760.0 * abundance
        if a260_ratio > 1.25:
            # Keep max(A280, A260) visually well-proportioned (~850 mAU max)
            peak_height_260 = min(980.0, base_h_280 * 1.18)
            peak_height_280 = peak_height_260 / a260_ratio
        else:
            peak_height_280 = base_h_280
            peak_height_260 = peak_height_280 * a260_ratio

        # Realistic chromatographic peak with mild hydrodynamic tailing (Cytiva asymmetry factor As ~ 1.08)
        diff = volumes - ve
        sigma_eff = np.where(diff >= 0, sigma * 1.08, sigma)
        unit_peak = np.exp(-0.5 * (diff / sigma_eff) ** 2)

        curve_280 = peak_height_280 * unit_peak
        curve_260 = peak_height_260 * unit_peak

        total_signal_280 += curve_280
        total_signal_260 += curve_260

        if is_aggregate or kav <= 0.0:
            pore_access_pct = 0.0
        else:
            pore_access_pct = round(100.0 * min(1.0, max(0.0, (ve - vo) / max(1e-6, v_perm - vo))), 1)

        individual_peaks.append({
            "label": label,
            "mw_kda": round(mw, 2),
            "conformation": conformation,
            "rh_angstrom": round(rh_a, 1),
            "ve_ml": ve,
            "kav": kav,
            "pore_access_pct": pore_access_pct,
            "sigma_ml": round(sigma, dec),
            "fwhm_ml": round(2.3548 * sigma, dec),
            "peak_width_base_ml": round(w_base, dec),
            "effective_plates_n": eff_plates,
            "peak_height_mau": round(peak_height_280, 1),
            "peak_height_a260_mau": round(peak_height_260, 1),
            "a260_a280_ratio": a260_ratio,
            "uv280_invisible": uv280_invisible,
            "is_aggregate": is_aggregate,
            "exclusion_limit_kda": excl_kda,
            "exp_ve_ml": state.get("exp_ve_ml"),
            "color": color,
            "note": note,
            "curve_y": curve_280.tolist(),
            "curve_y_a260": curve_260.tolist(),
        })

    return {
        "column_id": column_config["id"],
        "column_name": column_config["name"],
        "volume_axis_ml": volumes.tolist(),
        "total_signal_mau": total_signal_280.tolist(),
        "total_signal_a260_mau": total_signal_260.tolist(),
        "peaks": individual_peaks,
        "vt_ml": vt,
        "vo_ml": vo,
        "v_perm_ml": float(column_config.get("liquid_perm_volume_ml", vt * 0.90)),
    }


def simulate_standard_mixture_chromatogram(
    column_config: Dict[str, Any],
    resolution_points: int = 700
) -> Dict[str, Any]:
    """
    Simulates the Cytiva / MegaDalton benchmark calibration mixture chromatogram
    for direct reference overlay on the selected column.
    """
    markers = get_column_marker_elutions(column_config)
    vt = float(column_config["bed_volume_vt_ml"])
    plate_count = float(column_config["plate_count_n"])
    volumes = np.linspace(0.0, vt * 1.12, resolution_points)
    total_mix_signal = np.zeros_like(volumes)
    marker_peaks = []

    vinj_dispersion = _get_extra_column_dispersion_ml(column_config)

    for m in markers:
        ve = float(m["ve_ml"])
        col_sigma = ve / math.sqrt(plate_count)
        sigma = math.sqrt(col_sigma ** 2 + vinj_dispersion ** 2)

        h = 230.0 if not m.get("is_void") else 155.0
        diff = volumes - ve
        sigma_eff = np.where(diff >= 0, sigma * 1.08, sigma)
        curve = h * np.exp(-0.5 * (diff / sigma_eff) ** 2)
        total_mix_signal += curve
        marker_peaks.append({
            "name": m["name"],
            "mw_kda": m["mw_kda"],
            "rh_angstrom": m["rh_angstrom"],
            "ve_ml": ve,
            "kav": m["kav"],
            "peak_width_base_ml": round(4.0 * sigma, 3 if vt < 5.0 else 2),
            "peak_height": h,
            "label": f"{m['name']} ({m['note']})"
        })

    return {
        "volume_axis_ml": volumes.tolist(),
        "signal_mau": total_mix_signal.tolist(),
        "marker_peaks": marker_peaks
    }


def _short_species_name(label: str) -> str:
    """Produces a compact badge-friendly species name (e.g., 'Monomer (14.3 kDa)' -> 'Monomer 14.3k')."""
    s = label.strip()
    m = None
    import re as _re
    m = _re.match(r"^([^\(]+)\(\s*([0-9.]+)\s*kDa\s*\)$", s)
    if m:
        base = m.group(1).strip()
        val = float(m.group(2))
        mw_txt = f"{val/1000.0:.2f}M" if val >= 1000.0 else f"{val:g}k"
        return f"{base} ({mw_txt})"
    return s[:22]


def compute_column_snapshot(
    sim_data: Dict[str, Any],
    column_config: Dict[str, Any],
    pumped_ml: float,
    include_reference_companions: bool = True
) -> Dict[str, Any]:
    """
    Computes an exact chromatographic snapshot of the packed SEC column, bead pore accessibility,
    live UV 280 nm detector flow-cell reading, and 8-tube fraction collector rack after pumping
    `pumped_ml` of running buffer into the top of the column.

    Physical Migration Model:
      - In linear size-exclusion chromatography, a band with retention volume Ve moves down a
        packed bed of normalized depth z in [0, 1] at constant fractional rate dz/dV = 1 / Ve,
        so z_center(V_pumped) = V_pumped / Ve.
      - Pore accessibility (% of liquid intraparticle pore volume entered, Ackers Kd * 100%):
        pore_access_pct = 100 * clip((Ve - Vo) / (V_perm - Vo), 0, 1).
    """
    vt = float(column_config["bed_volume_vt_ml"])
    vo = float(column_config["void_volume_vo_ml"])
    v_perm = float(column_config.get("liquid_perm_volume_ml", vt * 0.90))
    rh50 = float(column_config.get("rh50_angstrom", 32.657))
    excl_kda = float(column_config.get("exclusion_limit_kda", 1300.0))
    plate_n = float(column_config.get("plate_count_n", 2800.0))
    dec = 2 if vt < 5.0 else 1

    pumped = max(0.0, min(vt * 1.12, float(pumped_ml)))
    vol_axis = np.asarray(sim_data["volume_axis_ml"], dtype=float)
    sig_280 = np.asarray(sim_data["total_signal_mau"], dtype=float)
    sig_260 = np.asarray(sim_data.get("total_signal_a260_mau", sig_280 * 0.57), dtype=float)

    live_uv_280 = float(np.interp(pumped, vol_axis, sig_280)) if len(vol_axis) > 1 else 0.0
    live_uv_260 = float(np.interp(pumped, vol_axis, sig_260)) if len(vol_axis) > 1 else 0.0

    # 8 Fraction Collector Tubes spanning [0.85 * Vo, 1.02 * Vt] (pre-0.85*Vo goes to Waste)
    v_frac_start = round(vo * 0.85, 3)
    v_frac_end = round(vt * 1.02, 3)
    tube_span = (v_frac_end - v_frac_start) / 8.0

    def _assign_tube_id(ve_val: float) -> str:
        idx = int((ve_val - v_frac_start) / max(1e-6, tube_span))
        idx = max(0, min(7, idx))
        return f"F{idx + 1}"

    # Build active species list + optional non-colliding reference size companions for classroom contrast
    raw_items: List[Dict[str, Any]] = []
    real_peaks = list(sim_data.get("peaks", []))
    for pk in real_peaks:
        p_col = "#0F766E" if pk.get("color") == "#2563EB" else pk.get("color", "#0F766E")
        ve_v = float(pk["ve_ml"])
        kav_v = float(pk["kav"])
        is_agg = bool(pk.get("is_aggregate", False))
        p_pct = float(
            pk.get(
                "pore_access_pct",
                0.0 if (is_agg or kav_v <= 0.0) else round(100.0 * min(1.0, max(0.0, (ve_v - vo) / max(1e-6, v_perm - vo))), 1),
            )
        )
        raw_items.append({
            "label": pk["label"],
            "short_label": _short_species_name(pk["label"]),
            "mw_kda": float(pk["mw_kda"]),
            "rh_angstrom": float(pk["rh_angstrom"]),
            "ve_ml": ve_v,
            "kav": kav_v,
            "pore_access_pct": p_pct,
            "sigma_ml": float(pk.get("sigma_ml", max(0.02 * vt, ve_v / math.sqrt(plate_n)))),
            "peak_height_mau": float(pk.get("peak_height_mau", 700.0)),
            "color": p_col,
            "is_aggregate": is_agg,
            "is_reference_ghost": False,
        })

    if include_reference_companions and len(real_peaks) < 2:
        existing_ves = [it["ve_ml"] for it in raw_items]
        min_sep = 0.075 * vt

        # 1. Giant Excluded Reference (Void Vo, 0% pores)
        if all(abs(vo - ev) > min_sep for ev in existing_ves):
            excl_lbl = f"{excl_kda/1000.0:g} MDa" if excl_kda >= 1000.0 else f"{excl_kda:g} kDa"
            rh_excl_ref = max(110.0, predict_stokes_radius_rh(excl_kda * 1.3, "globular"))
            sig_vo = math.sqrt((vo / math.sqrt(plate_n)) ** 2 + _get_extra_column_dispersion_ml(column_config) ** 2)
            raw_items.append({
                "label": f"Ref: Big Excluded (>{excl_lbl})",
                "short_label": "Ref: Giant (Vo)",
                "mw_kda": excl_kda * 1.3,
                "rh_angstrom": round(rh_excl_ref, 1),
                "ve_ml": round(vo, 3 if vt < 5.0 else 2),
                "kav": 0.0,
                "pore_access_pct": 0.0,
                "sigma_ml": sig_vo,
                "peak_height_mau": 0.0,
                "color": "#DC2626",
                "is_aggregate": True,
                "is_reference_ghost": True,
            })
            existing_ves.append(vo)

        # 2. If the user's single species is itself at Vo (e.g. Aggregate alone or giant Titin), add a Mid-Size Reference
        ve_mid, kav_mid, _ = calculate_retention_volume(44.0, column_config, rh_override_a=30.5)
        if len(existing_ves) < 2 and all(abs(ve_mid - ev) > min_sep * 1.2 for ev in existing_ves):
            sig_mid = math.sqrt((ve_mid / math.sqrt(plate_n)) ** 2 + _get_extra_column_dispersion_ml(column_config) ** 2)
            p_pct_mid = round(100.0 * min(1.0, max(0.0, (ve_mid - vo) / max(1e-6, v_perm - vo))), 1)
            raw_items.append({
                "label": "Ref: Medium Protein (44 kDa)",
                "short_label": "Ref: Medium (44k)",
                "mw_kda": 44.0,
                "rh_angstrom": 30.5,
                "ve_ml": ve_mid,
                "kav": kav_mid,
                "pore_access_pct": p_pct_mid,
                "sigma_ml": sig_mid,
                "peak_height_mau": 0.0,
                "color": "#D97706",
                "is_aggregate": False,
                "is_reference_ghost": True,
            })
            existing_ves.append(ve_mid)

        # 3. Tiny Permeating Marker (Vitamin B12, 1.36 kDa, Rh = 8.5 Å -> near V_perm)
        ve_tiny, kav_tiny, _ = calculate_retention_volume(1.355, column_config, rh_override_a=8.5)
        if all(abs(ve_tiny - ev) > min_sep for ev in existing_ves):
            sig_tiny = math.sqrt((ve_tiny / math.sqrt(plate_n)) ** 2 + _get_extra_column_dispersion_ml(column_config) ** 2)
            p_pct_tiny = round(100.0 * min(1.0, max(0.0, (ve_tiny - vo) / max(1e-6, v_perm - vo))), 1)
            raw_items.append({
                "label": "Ref: Tiny Marker (1.4 kDa)",
                "short_label": "Ref: Tiny (1.4k)",
                "mw_kda": 1.36,
                "rh_angstrom": 8.5,
                "ve_ml": ve_tiny,
                "kav": kav_tiny,
                "pore_access_pct": p_pct_tiny,
                "sigma_ml": sig_tiny,
                "peak_height_mau": 0.0,
                "color": "#64748B",
                "is_aggregate": False,
                "is_reference_ghost": True,
            })

    # Sort travellers from fastest-eluting (largest Rh / smallest Ve) to slowest-eluting (smallest Rh / largest Ve)
    raw_items.sort(key=lambda x: (x["ve_ml"], 0 if not x["is_reference_ghost"] else 1))

    vinj = _get_extra_column_dispersion_ml(column_config)
    travellers: List[Dict[str, Any]] = []
    detector_active_peak: Optional[Dict[str, Any]] = None
    best_det_signal = 0.0

    for it in raw_items:
        ve_v = max(1e-6, it["ve_ml"])
        sig_v = max(1e-6, it["sigma_ml"])
        progress = pumped / ve_v
        z_col = min(1.0, max(0.0, progress))

        # Plate-theory fractional band dispersion inside the column
        sig_z0 = vinj / ve_v
        sig_z = math.sqrt(sig_z0 ** 2 + max(0.04, z_col) / max(400.0, plate_n))
        dz_band = min(0.095, max(0.024, 1.8 * sig_z))

        # Monotonic visual sphere radius (px) from hydrodynamic Stokes radius Rh (Å)
        rh_eff = max(8.0, it["rh_angstrom"])
        sphere_r_px = round(min(10.8, max(3.6, 3.4 + 2.55 * math.log(rh_eff / 7.5))), 2)
        if it["is_aggregate"] and not it["is_reference_ghost"]:
            sphere_r_px = max(sphere_r_px, 9.8)

        if pumped <= 0.002 * vt:
            status = "at_inlet"
        elif abs(pumped - ve_v) <= 1.15 * sig_v:
            status = "at_detector"
        elif pumped < ve_v:
            status = "in_column"
        else:
            status = "eluted"

        tube_id = _assign_tube_id(ve_v)
        entry = {
            **it,
            "progress": round(progress, 4),
            "z_col": round(z_col, 4),
            "pct_down": int(round(z_col * 100.0)),
            "dz_band": round(dz_band, 4),
            "sphere_r_px": sphere_r_px,
            "status": status,
            "assigned_tube": tube_id,
        }
        travellers.append(entry)

        if not it["is_reference_ghost"] and status == "at_detector":
            inst_sig = it["peak_height_mau"] * math.exp(-0.5 * ((pumped - ve_v) / sig_v) ** 2)
            if inst_sig > best_det_signal:
                best_det_signal = inst_sig
                detector_active_peak = entry

    # Compute 8 Fraction Collector Tubes F1..F8
    fractions: List[Dict[str, Any]] = []
    for k in range(8):
        v0_t = v_frac_start + k * tube_span
        v1_t = v0_t + tube_span
        v_mid = 0.5 * (v0_t + v1_t)
        if pumped <= v0_t:
            fill_frac = 0.0
        elif pumped >= v1_t:
            fill_frac = 1.0
        else:
            fill_frac = (pumped - v0_t) / max(1e-6, tube_span)

        is_filling_now = (v0_t <= pumped < v1_t) and (pumped > 0.002 * vt)

        # Determine dominant real species captured in this tube
        best_pk_color = "#CBD5E1"
        best_pk_weight = 0.0
        captured_labels: List[str] = []
        for tr in travellers:
            if tr["is_reference_ghost"]:
                continue
            # Overlap weight of Gaussian peak inside [v0_t, v1_t]
            dist = 0.0 if (v0_t <= tr["ve_ml"] <= v1_t) else min(abs(tr["ve_ml"] - v0_t), abs(tr["ve_ml"] - v1_t))
            w = math.exp(-0.5 * (dist / max(1e-6, tr["sigma_ml"])) ** 2)
            if v0_t <= tr["ve_ml"] < v1_t or (k == 7 and abs(tr["ve_ml"] - v1_t) <= 0.02 * vt):
                captured_labels.append(tr["short_label"])
            if w > 0.25 and w > best_pk_weight:
                best_pk_weight = w
                best_pk_color = tr["color"]

        tint_opacity = round(0.18 + 0.67 * min(1.0, best_pk_weight), 2) if best_pk_weight > 0.25 else 0.22
        fractions.append({
            "tube_id": f"F{k + 1}",
            "v_start_ml": round(v0_t, dec),
            "v_end_ml": round(v1_t, dec),
            "v_mid_ml": round(v_mid, dec),
            "fill_frac": round(fill_frac, 3),
            "is_filling_now": is_filling_now,
            "has_protein": best_pk_weight > 0.25,
            "tint_color": best_pk_color if best_pk_weight > 0.25 else "#BAE6FD",
            "tint_opacity": tint_opacity,
            "captured_labels": captured_labels,
        })

    # Build Plain-English Classroom Stage Narrative
    active_tube_obj = next((f for f in fractions if f["is_filling_now"]), None)
    active_tube_txt = active_tube_obj["tube_id"] if active_tube_obj else ("Waste" if pumped < v_frac_start else "Done")

    if pumped <= 0.005 * vt:
        stage_badge = "Stage 1 • Sample Injected at Top"
        stage_text = (
            "All molecules start mixed together in a thin sample band at the very top of the column (0% down). "
            "Move the pump slider right to push buffer downward and watch big molecules pull ahead of small ones!"
        )
    elif detector_active_peak is not None:
        pk_nm = detector_active_peak["label"]
        p_pct = detector_active_peak["pore_access_pct"]
        stage_badge = f"🔦 Peak Exiting Now into Tube {active_tube_txt}!"
        if detector_active_peak["is_aggregate"] or p_pct <= 1.0:
            stage_text = (
                f"At {pumped:.{dec}f} mL (= Void Volume Vo), {pk_nm} is exiting the bottom UV detector "
                f"({live_uv_280:.0f} mAU) into Tube {active_tube_txt}! Because it entered 0% of the bead tunnels, "
                f"it took the express outside highway around every bead."
            )
        else:
            stage_text = (
                f"At {pumped:.{dec}f} mL, {pk_nm} is crossing the UV 280 nm lamp ({live_uv_280:.0f} mAU) and dripping "
                f"into Tube {active_tube_txt}! It explored {p_pct:.0f}% of the microscopic bead tunnels along the way."
            )
    elif pumped < vo * 0.96:
        stage_badge = "Stage 2 • Pushing Pre-Void Buffer"
        stage_text = (
            f"At {pumped:.{dec}f} mL (< Vo = {vo:.{dec}f} mL), NO proteins have reached the bottom yet — only pure buffer "
            f"is exiting to Waste. Inside the column, big molecules are already pulling ahead because they skip the bead pores!"
        )
    elif abs(pumped - vo) <= 0.035 * vt:
        in_col_real = [t for t in travellers if not t["is_reference_ghost"] and t["status"] == "in_column"]
        stage_badge = f"🚀 Void Volume (Vo = {vo:.{dec}f} mL) • Giant Excluded Molecules Exit First!"
        if in_col_real:
            t0 = in_col_real[0]
            stage_text = (
                f"At {pumped:.{dec}f} mL (= Void Volume Vo), molecules too big for the bead pores (0% tunnels entered) "
                f"reach the bottom first (Tube {active_tube_txt}), while your {t0['label']} is only {t0['pct_down']}% down "
                f"the column because it weaves into {t0['pore_access_pct']:.0f}% of the bead tunnels!"
            )
        else:
            stage_text = (
                f"At {pumped:.{dec}f} mL (= Void Volume Vo), molecules larger than the bead pores bypass 100% of the "
                f"internal tunnels and exit first into Tube {active_tube_txt}."
            )
    elif pumped < vt * 0.98:
        in_col_real = [t["short_label"] for t in travellers if not t["is_reference_ghost"] and t["status"] == "in_column"]
        eluted_real = [t["short_label"] for t in travellers if not t["is_reference_ghost"] and t["status"] == "eluted"]
        stage_badge = f"Stage 3 • Separating Inside Bead Tunnels (Tube {active_tube_txt})"
        if eluted_real and in_col_real:
            stage_text = (
                f"At {pumped:.{dec}f} mL, {', '.join(eluted_real[:2])} already exited into early tubes, while smaller "
                f"{', '.join(in_col_real[:2])} is still weaving through internal bead tunnels before reaching the UV lamp!"
            )
        elif in_col_real:
            stage_text = (
                f"At {pumped:.{dec}f} mL (past Vo = {vo:.{dec}f} mL), {', '.join(in_col_real[:2])} is travelling through "
                f"the bead tunnel network and will reach the bottom UV detector at its elution volume (Ve)."
            )
        else:
            stage_text = (
                f"At {pumped:.{dec}f} mL, your target protein has already been collected cleanly in its fraction tube, "
                f"while tiny molecules and buffer salts finish their long maze path toward Vt ({vt:g} mL)."
            )
    else:
        stage_badge = "Stage 4 • Column Run Complete (Vt Reached)"
        stage_text = (
            f"At {pumped:.{dec}f} mL (≥ Total Bed Volume Vt = {vt:g} mL), even tiny salts have explored 100% of the bead "
            f"tunnels and exited. Every protein size has been cleanly sorted into its own numbered Fraction Tube!"
        )

    return {
        "column_id": column_config["id"],
        "column_name": column_config["name"],
        "pumped_ml": round(pumped, 3 if vt < 5.0 else 2),
        "vt_ml": vt,
        "vo_ml": vo,
        "v_perm_ml": v_perm,
        "v_frac_start_ml": v_frac_start,
        "v_frac_end_ml": v_frac_end,
        "rh50_angstrom": rh50,
        "exclusion_limit_kda": excl_kda,
        "live_uv_mau": round(live_uv_280, 1),
        "live_uv_a260_mau": round(live_uv_260, 1),
        "detector_active_peak": detector_active_peak,
        "active_tube_id": active_tube_txt,
        "travellers": travellers,
        "fractions": fractions,
        "stage_badge": stage_badge,
        "stage_text": stage_text,
    }
