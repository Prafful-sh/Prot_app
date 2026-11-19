"""
Purification Metrics & Physical Chemistry Module
Calculates:
- Molecular Weight (MW; ExPASy ProtParam / IUPAC average residue masses + 18.01528 Da H2O)
- Theoretical Isoelectric Point (pI; Bjellqvist et al., 1993, 1994 with terminal-specific pKa values)
- Net Charge across pH 2 - 12 (Henderson-Hasselbalch equation)
- Extinction Coefficients at 280 nm (reduced Cys and cystine disulfides; Pace et al., 1995) and Abs 0.1% (1 g/L)
- Instability Index (Guruprasad et al., 1990; 400-dipeptide DIWV matrix)
- Aliphatic Index (Ikai, 1980)
- GRAVY (Grand average of hydropathicity; Kyte & Doolittle, 1982)
- Recombinant Solubility Score in E. coli (Wilkinson & Harrison, 1991; Davis et al., 1999)
- Amino Acid Composition
"""

from typing import Dict, Any, List, Tuple

# Average isotopic amino acid residue masses (Da / g·mol^-1; ExPASy ProtParam / Gasteiger et al., 2005)
AA_MASSES = {
    'A': 71.0788,  'R': 156.1875, 'N': 114.1038, 'D': 115.0886,
    'C': 103.1388, 'E': 129.1155, 'Q': 128.1307, 'G': 57.0519,
    'H': 137.1411, 'I': 113.1594, 'L': 113.1594, 'K': 128.1741,
    'M': 131.1926, 'F': 147.1766, 'P': 97.1167,  'S': 87.0782,
    'T': 101.1051, 'W': 186.2132, 'Y': 163.1760, 'V': 99.1326
}
# Average molecular mass of terminal H + OH (2 * 1.00794 + 15.9994 = 18.01528 Da)
WATER_MASS = 18.01528

# Kyte & Doolittle (1982, J. Mol. Biol. 157:105-132) hydropathy values
KYTE_DOOLITTLE = {
    'A': 1.8,  'R': -4.5, 'N': -3.5, 'D': -3.5,
    'C': 2.5,  'E': -3.5, 'Q': -3.5, 'G': -0.4,
    'H': -3.2, 'I': 4.5,  'L': 3.8,  'K': -3.9,
    'M': 1.9,  'F': 2.8,  'P': -1.6, 'S': -0.8,
    'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# Bjellqvist et al. (1993, Electrophoresis 14:1023-1031; 1994, Electrophoresis 15:529-539)
# polypeptide pKa values as implemented in ExPASy ProtParam / Compute pI/Mw (Gasteiger et al., 2005)
BJELLQVIST_PKA = {
    'N_term': 7.50,
    'C_term': 3.55,
    'K': 10.00,
    'R': 12.00,
    'H': 5.98,
    'D': 4.05,
    'E': 4.45,
    'C': 9.00,
    'Y': 10.00
}

# Residue-specific N-terminal and C-terminal pKa adjustments (Bjellqvist et al., 1993, 1994)
BJELLQVIST_NTERM_PKA = {
    'A': 7.59,
    'M': 7.00,
    'S': 6.93,
    'P': 8.36,
    'T': 6.82,
    'V': 7.44,
    'E': 7.70
}
BJELLQVIST_CTERM_PKA = {
    'D': 4.55,
    'E': 4.75
}

# Dipeptide Instability Weight Values (Guruprasad, Reddy & Pandit, 1990, Protein Eng. 4:155-161)
# Verified across all 400 dipeptide pairs against ExPASy ProtParam
DIWV = {
    'AA': 1.0, 'AC': 44.94, 'AD': -7.49, 'AE': 1.0, 'AF': 1.0, 'AG': 1.0, 'AH': -7.49, 'AI': 1.0, 'AK': 1.0, 'AL': 1.0,
    'AM': 1.0, 'AN': 1.0, 'AP': 20.26, 'AQ': 1.0, 'AR': 1.0, 'AS': 1.0, 'AT': 1.0, 'AV': 1.0, 'AW': 1.0, 'AY': 1.0,
    'CA': 1.0, 'CC': 1.0, 'CD': 20.26, 'CE': 1.0, 'CF': 1.0, 'CG': 1.0, 'CH': 33.6, 'CI': 1.0, 'CK': 1.0, 'CL': 20.26,
    'CM': 33.6, 'CN': 1.0, 'CP': 20.26, 'CQ': -6.54, 'CR': 1.0, 'CS': 1.0, 'CT': 33.6, 'CV': -6.54, 'CW': 24.68, 'CY': 1.0,
    'DA': 1.0, 'DC': 1.0, 'DD': 1.0, 'DE': 1.0, 'DF': -6.54, 'DG': 1.0, 'DH': 1.0, 'DI': 1.0, 'DK': -7.49, 'DL': 1.0,
    'DM': 1.0, 'DN': 1.0, 'DP': 1.0, 'DQ': 1.0, 'DR': -6.54, 'DS': 20.26, 'DT': -14.03, 'DV': 1.0, 'DW': 1.0, 'DY': 1.0,
    'EA': 1.0, 'EC': 44.94, 'ED': 20.26, 'EE': 33.6, 'EF': 1.0, 'EG': 1.0, 'EH': -6.54, 'EI': 20.26, 'EK': 1.0, 'EL': 1.0,
    'EM': 1.0, 'EN': 1.0, 'EP': 20.26, 'EQ': 20.26, 'ER': 1.0, 'ES': 20.26, 'ET': 1.0, 'EV': 1.0, 'EW': -14.03, 'EY': 1.0,
    'FA': 1.0, 'FC': 1.0, 'FD': 13.34, 'FE': 1.0, 'FF': 1.0, 'FG': 1.0, 'FH': 1.0, 'FI': 1.0, 'FK': -14.03, 'FL': 1.0,
    'FM': 1.0, 'FN': 1.0, 'FP': 20.26, 'FQ': 1.0, 'FR': 1.0, 'FS': 1.0, 'FT': 1.0, 'FV': 1.0, 'FW': 1.0, 'FY': 33.6,
    'GA': -7.49, 'GC': 1.0, 'GD': 1.0, 'GE': -6.54, 'GF': 1.0, 'GG': 13.34, 'GH': 1.0, 'GI': -7.49, 'GK': -7.49, 'GL': 1.0,
    'GM': 1.0, 'GN': -7.49, 'GP': 1.0, 'GQ': 1.0, 'GR': 1.0, 'GS': 1.0, 'GT': -7.49, 'GV': 1.0, 'GW': 13.34, 'GY': -7.49,
    'HA': 1.0, 'HC': 1.0, 'HD': 1.0, 'HE': 1.0, 'HF': -9.37, 'HG': -9.37, 'HH': 1.0, 'HI': 44.94, 'HK': 24.68, 'HL': 1.0,
    'HM': 1.0, 'HN': 24.68, 'HP': -1.88, 'HQ': 1.0, 'HR': 1.0, 'HS': 1.0, 'HT': -6.54, 'HV': 1.0, 'HW': -1.88, 'HY': 44.94,
    'IA': 1.0, 'IC': 1.0, 'ID': 1.0, 'IE': 44.94, 'IF': 1.0, 'IG': 1.0, 'IH': 13.34, 'II': 1.0, 'IK': -7.49, 'IL': 20.26,
    'IM': 1.0, 'IN': 1.0, 'IP': -1.88, 'IQ': 1.0, 'IR': 1.0, 'IS': 1.0, 'IT': 1.0, 'IV': -7.49, 'IW': 1.0, 'IY': 1.0,
    'KA': 1.0, 'KC': 1.0, 'KD': 1.0, 'KE': 1.0, 'KF': 1.0, 'KG': -7.49, 'KH': 1.0, 'KI': -7.49, 'KK': 1.0, 'KL': -7.49,
    'KM': 33.6, 'KN': 1.0, 'KP': -6.54, 'KQ': 24.64, 'KR': 33.6, 'KS': 1.0, 'KT': 1.0, 'KV': -7.49, 'KW': 1.0, 'KY': 1.0,
    'LA': 1.0, 'LC': 1.0, 'LD': 1.0, 'LE': 1.0, 'LF': 1.0, 'LG': 1.0, 'LH': 1.0, 'LI': 1.0, 'LK': -7.49, 'LL': 1.0,
    'LM': 1.0, 'LN': 1.0, 'LP': 20.26, 'LQ': 33.6, 'LR': 20.26, 'LS': 1.0, 'LT': 1.0, 'LV': 1.0, 'LW': 24.68, 'LY': 1.0,
    'MA': 13.34, 'MC': 1.0, 'MD': 1.0, 'ME': 1.0, 'MF': 1.0, 'MG': 1.0, 'MH': 58.28, 'MI': 1.0, 'MK': 1.0, 'ML': 1.0,
    'MM': -1.88, 'MN': 1.0, 'MP': 44.94, 'MQ': -6.54, 'MR': -6.54, 'MS': 44.94, 'MT': -1.88, 'MV': 1.0, 'MW': 1.0, 'MY': 24.68,
    'NA': 1.0, 'NC': -1.88, 'ND': 1.0, 'NE': 1.0, 'NF': -14.03, 'NG': -14.03, 'NH': 1.0, 'NI': 44.94, 'NK': 24.68, 'NL': 1.0,
    'NM': 1.0, 'NN': 1.0, 'NP': -1.88, 'NQ': -6.54, 'NR': 1.0, 'NS': 1.0, 'NT': -7.49, 'NV': 1.0, 'NW': -9.37, 'NY': 1.0,
    'PA': 20.26, 'PC': -6.54, 'PD': -6.54, 'PE': 18.38, 'PF': 20.26, 'PG': 1.0, 'PH': 1.0, 'PI': 1.0, 'PK': 1.0, 'PL': 1.0,
    'PM': -6.54, 'PN': 1.0, 'PP': 20.26, 'PQ': 20.26, 'PR': -6.54, 'PS': 20.26, 'PT': 1.0, 'PV': 20.26, 'PW': -1.88, 'PY': 1.0,
    'QA': 1.0, 'QC': -6.54, 'QD': 20.26, 'QE': 20.26, 'QF': -6.54, 'QG': 1.0, 'QH': 1.0, 'QI': 1.0, 'QK': 1.0, 'QL': 1.0,
    'QM': 1.0, 'QN': 1.0, 'QP': 20.26, 'QQ': 20.26, 'QR': 1.0, 'QS': 44.94, 'QT': 1.0, 'QV': -6.54, 'QW': 1.0, 'QY': -6.54,
    'RA': 1.0, 'RC': 1.0, 'RD': 1.0, 'RE': 1.0, 'RF': 1.0, 'RG': -7.49, 'RH': 20.26, 'RI': 1.0, 'RK': 1.0, 'RL': 1.0,
    'RM': 1.0, 'RN': 13.34, 'RP': 20.26, 'RQ': 20.26, 'RR': 58.28, 'RS': 44.94, 'RT': 1.0, 'RV': 1.0, 'RW': 58.28, 'RY': -6.54,
    'SA': 1.0, 'SC': 33.6, 'SD': 1.0, 'SE': 20.26, 'SF': 1.0, 'SG': 1.0, 'SH': 1.0, 'SI': 1.0, 'SK': 1.0, 'SL': 1.0,
    'SM': 1.0, 'SN': 1.0, 'SP': 44.94, 'SQ': 20.26, 'SR': 20.26, 'SS': 20.26, 'ST': 1.0, 'SV': 1.0, 'SW': 1.0, 'SY': 1.0,
    'TA': 1.0, 'TC': 1.0, 'TD': 1.0, 'TE': 20.26, 'TF': 13.34, 'TG': -7.49, 'TH': 1.0, 'TI': 1.0, 'TK': 1.0, 'TL': 1.0,
    'TM': 1.0, 'TN': -14.03, 'TP': 1.0, 'TQ': -6.54, 'TR': 1.0, 'TS': 1.0, 'TT': 1.0, 'TV': 1.0, 'TW': -14.03, 'TY': 1.0,
    'VA': 1.0, 'VC': 1.0, 'VD': -14.03, 'VE': 1.0, 'VF': 1.0, 'VG': -7.49, 'VH': 1.0, 'VI': 1.0, 'VK': -1.88, 'VL': 1.0,
    'VM': 1.0, 'VN': 1.0, 'VP': 20.26, 'VQ': 1.0, 'VR': 1.0, 'VS': 1.0, 'VT': -7.49, 'VV': 1.0, 'VW': 1.0, 'VY': -6.54,
    'WA': -14.03, 'WC': 1.0, 'WD': 1.0, 'WE': 1.0, 'WF': 1.0, 'WG': -9.37, 'WH': 24.68, 'WI': 1.0, 'WK': 1.0, 'WL': 13.34,
    'WM': 24.68, 'WN': 13.34, 'WP': 1.0, 'WQ': 1.0, 'WR': 1.0, 'WS': 1.0, 'WT': -14.03, 'WV': -7.49, 'WW': 1.0, 'WY': 1.0,
    'YA': 24.68, 'YC': 1.0, 'YD': 24.68, 'YE': -6.54, 'YF': 1.0, 'YG': -7.49, 'YH': 13.34, 'YI': 1.0, 'YK': 1.0, 'YL': 1.0,
    'YM': 44.94, 'YN': 1.0, 'YP': 13.34, 'YQ': 1.0, 'YR': -15.91, 'YS': 1.0, 'YT': -7.49, 'YV': 1.0, 'YW': -9.37, 'YY': 13.34
}


def calculate_molecular_weight(sequence: str) -> float:
    """Calculates polypeptide molecular weight in Daltons (ExPASy ProtParam average isotopic mass)."""
    seq = sequence.strip().upper()
    if not seq:
        return 0.0
    mass = sum(AA_MASSES.get(aa, 110.0) for aa in seq)
    mass += WATER_MASS  # Free N-terminal H and C-terminal OH
    return round(mass, 2)


def calculate_net_charge_at_ph(sequence: str, ph: float, cystines_oxidized: bool = False) -> float:
    """
    Calculates net electrical charge of the polypeptide at a specific pH using
    Bjellqvist et al. (1993, 1994) side-chain and residue-specific terminal pKa values
    (the ExPASy ProtParam sequence-based model).
    When `cystines_oxidized=True`, Cys side chains locked in structural S-S disulfide bonds
    are treated as non-ionizable (omitting thiol deprotonation above pH ~9).
    """
    seq = sequence.strip().upper()
    if not seq:
        return 0.0

    counts = {aa: seq.count(aa) for aa in "KRHDECY"}
    n_pka = BJELLQVIST_NTERM_PKA.get(seq[0], BJELLQVIST_PKA['N_term'])
    c_pka = BJELLQVIST_CTERM_PKA.get(seq[-1], BJELLQVIST_PKA['C_term'])

    # Positive charges (basic groups: protonated below pKa -> 1 / (1 + 10^(pH - pKa)))
    n_term_pos = 1.0 / (1.0 + 10.0 ** (ph - n_pka))
    k_pos = counts['K'] / (1.0 + 10.0 ** (ph - BJELLQVIST_PKA['K']))
    r_pos = counts['R'] / (1.0 + 10.0 ** (ph - BJELLQVIST_PKA['R']))
    h_pos = counts['H'] / (1.0 + 10.0 ** (ph - BJELLQVIST_PKA['H']))
    total_pos = n_term_pos + k_pos + r_pos + h_pos

    # Negative charges (acidic groups: deprotonated above pKa -> 1 / (1 + 10^(pKa - pH)))
    c_term_neg = 1.0 / (1.0 + 10.0 ** (c_pka - ph))
    d_neg = counts['D'] / (1.0 + 10.0 ** (BJELLQVIST_PKA['D'] - ph))
    e_neg = counts['E'] / (1.0 + 10.0 ** (BJELLQVIST_PKA['E'] - ph))
    c_neg = 0.0 if cystines_oxidized else (counts['C'] / (1.0 + 10.0 ** (BJELLQVIST_PKA['C'] - ph)))
    y_neg = counts['Y'] / (1.0 + 10.0 ** (BJELLQVIST_PKA['Y'] - ph))
    total_neg = c_term_neg + d_neg + e_neg + c_neg + y_neg

    return total_pos - total_neg


def calculate_isoelectric_point(sequence: str, cystines_oxidized: bool = False) -> float:
    """Finds theoretical pI where net charge equals 0 using high-precision binary bisection."""
    seq = sequence.strip().upper()
    if not seq:
        return 0.0
    low_ph = 0.0
    high_ph = 14.0
    for _ in range(50):
        mid_ph = (low_ph + high_ph) / 2.0
        charge = calculate_net_charge_at_ph(seq, mid_ph, cystines_oxidized=cystines_oxidized)
        if charge > 0.0:
            low_ph = mid_ph
        else:
            high_ph = mid_ph
    return round((low_ph + high_ph) / 2.0, 2)


def generate_charge_ph_profile(sequence: str) -> Tuple[List[float], List[float]]:
    """Generates pH curve points from pH 2.0 to 12.0 in 0.1 increments."""
    ph_values = [round(2.0 + i * 0.1, 1) for i in range(101)]
    charge_values = [round(calculate_net_charge_at_ph(sequence, ph), 2) for ph in ph_values]
    return ph_values, charge_values


def calculate_extinction_coefficient(sequence: str, mw: float) -> Dict[str, Any]:
    """
    Calculates molar extinction coefficient at 280 nm (M^-1 cm^-1) per Pace et al. (1995,
    Protein Sci. 4:2411-2423) and Absorbance at 280 nm of a 0.1% (= 1 g/L = 1 mg/mL) solution.
    """
    seq = sequence.strip().upper()
    nW = seq.count('W')
    nY = seq.count('Y')
    nC = seq.count('C')

    # Pace et al. (1995): Trp = 5500, Tyr = 1490, Cystine (S-S bond) = 125 M^-1 cm^-1
    eps_reduced = (nW * 5500) + (nY * 1490)
    cystines = nC // 2
    eps_cystines = eps_reduced + (cystines * 125)

    a280_reduced = round(eps_reduced / mw, 3) if mw > 0 else 0.0
    a280_cystines = round(eps_cystines / mw, 3) if mw > 0 else 0.0

    return {
        "num_Trp": nW,
        "num_Tyr": nY,
        "num_Cys": nC,
        "potential_disulfides": cystines,
        "molar_extinction_reduced": eps_reduced,
        "molar_extinction_cystines": eps_cystines,
        "abs_01_reduced": a280_reduced,        # Abs 0.1% (= 1 g/L or 1 mg/mL, 1 cm pathlength)
        "abs_01_cystines": a280_cystines
    }


def calculate_instability_index(sequence: str) -> Tuple[float, str]:
    """Calculates Guruprasad et al. (1990) Instability Index (II). II < 40: Stable; II >= 40: Unstable."""
    seq = sequence.strip().upper()
    if len(seq) < 2:
        return 0.0, "Stable (in vitro)"
    score = 0.0
    for i in range(len(seq) - 1):
        dipeptide = seq[i:i + 2]
        score += DIWV.get(dipeptide, 1.0)
    instability_index = round((10.0 / len(seq)) * score, 2)
    classification = "Stable (in vitro)" if instability_index < 40.0 else "Unstable (potential degradation / short half-life)"
    return instability_index, classification


def calculate_aliphatic_index(sequence: str) -> float:
    """Calculates Ikai (1980, J. Biochem. 88:1895-1898) Aliphatic Index: AI = X(Ala) + 2.9*X(Val) + 3.9*(X(Ile) + X(Leu))."""
    seq = sequence.strip().upper()
    if not seq:
        return 0.0
    L = len(seq)
    xA = (seq.count('A') / L) * 100.0
    xV = (seq.count('V') / L) * 100.0
    xI = (seq.count('I') / L) * 100.0
    xL = (seq.count('L') / L) * 100.0
    ai = xA + (2.9 * xV) + (3.9 * (xI + xL))
    return round(ai, 2)


def calculate_gravy(sequence: str) -> float:
    """Calculates Kyte & Doolittle (1982) GRAVY (Grand average of hydropathicity). Negative = hydrophilic, positive = hydrophobic."""
    seq = sequence.strip().upper()
    if not seq:
        return 0.0
    total_hydro = sum(KYTE_DOOLITTLE.get(aa, 0.0) for aa in seq)
    return round(total_hydro / len(seq), 3)


def calculate_solubility_heuristic(sequence: str) -> Dict[str, Any]:
    """
    Revised Wilkinson-Harrison (1991, Bio/Technology 9:443-448) / Davis et al. (1999,
    Biotechnol. Bioeng. 65:382-388) model for recombinant protein solubility in E. coli cytoplasm at 37°C.

    Canonical Variable:
        CV = 15.43 * ((N + G + P + S) / L) - 29.56 * |((R + K - D - E) / L) - 0.03|
    Discriminant cutoff:
        CV' = 1.71  (CV - CV' < 0 predicts soluble; CV - CV' > 0 predicts inclusion bodies)
    Classification confidence probability:
        P = 0.4934 + 0.276 * |CV - 1.71| - 0.0392 * (CV - 1.71)^2
    """
    seq = sequence.strip().upper()
    L = len(seq)
    if L == 0:
        return {
            "solubility_class": "Unknown",
            "description": "No sequence provided.",
            "composite_score": 0.0,
            "canonical_cv": 0.0,
            "cv_discriminant": 0.0,
            "solubility_probability_pct": 0.0,
            "turn_residue_fraction": 0.0,
            "net_charge_per_residue": 0.0
        }

    K = seq.count('K')
    R = seq.count('R')
    D = seq.count('D')
    E = seq.count('E')

    # Signed and absolute net charge per residue (Arg, Lys vs Asp, Glu)
    signed_charge_per_res = ((R + K) - (D + E)) / L
    q = abs(signed_charge_per_res)

    # Turn-forming residue fraction (Asn, Gly, Pro, Ser)
    N = seq.count('N')
    G = seq.count('G')
    P = seq.count('P')
    S = seq.count('S')
    turn_fraction = (N + G + P + S) / L

    # Revised Wilkinson-Harrison / Davis et al. (1999) Canonical Variable & Probability
    cv = 15.43 * turn_fraction - 29.56 * abs(signed_charge_per_res - 0.03)
    cv_prime = 1.71
    cv_diff = cv - cv_prime

    # Quadratic confidence saturates at vertex x = 0.276 / (2 * 0.0392) = 3.5204 (~97.9% max confidence)
    x = min(abs(cv_diff), 3.5204)
    class_confidence = min(1.0, max(0.4934, 0.4934 + (0.276 * x) - (0.0392 * (x ** 2))))
    sol_prob = class_confidence if cv_diff <= 0.0 else (1.0 - class_confidence)
    sol_pct = round(sol_prob * 100.0, 1)
    insol_pct = round((1.0 - sol_prob) * 100.0, 1)

    sol_pct_coarse = int(round(sol_pct / 5.0) * 5)
    insol_pct_coarse = int(round(insol_pct / 5.0) * 5)

    if cv_diff <= -0.35:
        sol_class = "High Solubility Propensity (Heuristic)"
        desc = (
            f"Coarse sequence heuristic (Wilkinson–Harrison 1991 / Davis 1999, trained on 81 proteins in E. coli cytoplasm at 37°C): "
            f"CV = {cv:.2f} < 1.71 (model score ≈{sol_pct_coarse}% soluble). Treat as a rough screening cue, not a measured probability—"
            "actual soluble yield depends strongly on temperature, promoter, host strain, and construct design."
        )
    elif cv_diff <= 0.35:
        sol_class = "Borderline Solubility (Heuristic)"
        desc = (
            f"Coarse sequence heuristic (Wilkinson–Harrison 1991 / Davis 1999, trained on 81 proteins in E. coli cytoplasm at 37°C): "
            f"CV = {cv:.2f} sits near the 1.71 cutoff (model score ≈{sol_pct_coarse}% soluble). Treat as an approximate indicator—"
            "actual solubility depends strongly on expression temperature (16–20°C), host strain, and fusion tags."
        )
    else:
        sol_class = "Elevated Inclusion-Body Risk (Heuristic)"
        desc = (
            f"Coarse sequence heuristic (Wilkinson–Harrison 1991 / Davis 1999, trained on 81 proteins in E. coli cytoplasm at 37°C): "
            f"CV = {cv:.2f} > 1.71 (model score ≈{insol_pct_coarse}% insoluble). This is an empirical model estimate, not a measured outcome—"
            "actual solubility varies widely with low-temperature induction (16–18°C), oxidizing strains (SHuffle/Origami) or periplasmic export for disulfides, and MBP/SUMO tags."
        )

    return {
        "solubility_class": sol_class,
        "description": desc,
        "composite_score": round(cv_diff, 2),
        "canonical_cv": round(cv, 2),
        "cv_discriminant": round(cv_diff, 2),
        "solubility_probability_pct": sol_pct,
        "turn_residue_fraction": round(turn_fraction * 100.0, 1),
        "net_charge_per_residue": round(q, 3)
    }


def analyze_all_parameters(sequence: str) -> Dict[str, Any]:
    """Runs all physicochemical and purification parameters on the input sequence."""
    seq = sequence.strip().upper()
    mw_da = calculate_molecular_weight(seq)
    mw_kda = round(mw_da / 1000.0, 2)
    pi = calculate_isoelectric_point(seq, cystines_oxidized=False)
    pi_oxidized = calculate_isoelectric_point(seq, cystines_oxidized=True)
    ph_curve, charge_curve = generate_charge_ph_profile(seq)
    ext_coeffs = calculate_extinction_coefficient(seq, mw_da)
    instability, instab_class = calculate_instability_index(seq)
    aliphatic = calculate_aliphatic_index(seq)
    gravy = calculate_gravy(seq)
    solubility = calculate_solubility_heuristic(seq)

    # Composition breakdown
    total_aa = len(seq)
    aa_counts = {aa: seq.count(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}

    pos_count = aa_counts['K'] + aa_counts['R'] + aa_counts['H']
    neg_count = aa_counts['D'] + aa_counts['E']
    aromatic_count = aa_counts['F'] + aa_counts['W'] + aa_counts['Y']
    hydrophobic_count = (
        aa_counts['A'] + aa_counts['V'] + aa_counts['I'] + aa_counts['L'] +
        aa_counts['M'] + aa_counts['F'] + aa_counts['W']
    )

    return {
        "length": total_aa,
        "mw_da": mw_da,
        "mw_kda": mw_kda,
        "pi": pi,
        "pi_oxidized": pi_oxidized,
        "ph_values": ph_curve,
        "charge_values": charge_curve,
        "extinction": ext_coeffs,
        "instability_index": instability,
        "instability_class": instab_class,
        "aliphatic_index": aliphatic,
        "gravy": gravy,
        "solubility": solubility,
        "counts": {
            "positive_residues": pos_count,
            "negative_residues": neg_count,
            "aromatic_residues": aromatic_count,
            "hydrophobic_residues": hydrophobic_count,
            "cysteines": aa_counts['C']
        },
        "aa_percentages": {
            aa: round((c / total_aa) * 100.0, 1) if total_aa > 0 else 0.0
            for aa, c in aa_counts.items()
        }
    }
