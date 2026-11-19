"""
Comprehensive Golden Scientific Verification & Benchmark Test Suite for ProPurify Studio
========================================================================================
Literature-referenced verification tests covering:
1. ExPASy ProtParam Golden Reference on Mature Hen Egg-White Lysozyme (UniProt P00698,
   chain 19–147, 129 aa, formula C613H959N193O185S10):
   - Gasteiger et al. (2005) / IUPAC average isotopic MW = 14313.14 Da (14.31 kDa)
   - Bjellqvist et al. (1993, 1994) theoretical pI = 9.32
   - Pace et al. (1995) extinction at 280 nm: eps_reduced = 37470 M^-1 cm^-1 (Abs 0.1% = 2.618),
     eps_cystines = 37970 M^-1 cm^-1 (Abs 0.1% = 2.653; 4 disulfide pairs)
   - Guruprasad et al. (1990) Instability Index = 16.09 (400-dipeptide DIWV matrix)
   - Ikai (1980) Aliphatic Index = 65.12 (exact 8400 / 129 = 65.1163)
   - Kyte & Doolittle (1982) GRAVY = -0.472 (exact -60.9 / 129 = -0.4721)
2. Classroom Presets, Standard Genetic Code (64 codons), Synthetic Bacterial Lysozyme ORF
   (ATG + 129 aa + TAA = 393 bp / 130 aa), Construct Tags & Protease Cleavage Sites, and
   Zero eIF4E / Q9FLB8 Check.
3. IUPAC Chemical Formula Weights, Buffer pKa / Buffering Windows, Solid & Neat-Liquid
   Stoichiometry (g, mg, mM, M, µM, % w/v, % v/v, mg/mL; sub-mL µL dispensing for
   2-Mercaptoethanol & Triton X-100), and Solubility / Safety Bench Warnings.
4. All 7 Protein Purification Methods: Affinity (His / Strep / Strep-XT / GST / MBP / FLAG),
   Pseudo-Affinity (Heparin / Blue), Ammonium Sulfate Salting-Out, Ion-Exchange (AEX / CEX +
   near-pI precipitation alert), Hydrophobic Interaction (Cytiva Butyl < Phenyl < Octyl +
   near-pI rescue + pH > 8.0 Na2SO4 rule), Multimodal (CHT / Capto MMC), and SEC.
5. SEC Hydrodynamic Stokes Radii (Rh), 8 Calibrated Columns, Standard Calibration Markers,
   Nucleic Acid MW Conventions (617.96 Da/bp dsDNA; 321.47 Da/nt ssRNA), Dual-UV A260/A280,
   and 1,254-Protein / 7-Paper Benchmark Integrity.
6. SDS-PAGE Ladder Ordering, 2D Rf Monotonicity (across MW & Gel % Acrylamide), Coomassie vs.
   Silver Stain Simulation, and Live Headless Streamlit UI Smoke Test of app.py.
"""

import json
import os
import re

from modules.sequence_parser import (
    LYSOZYME_MATURE_SEQ,
    CLASSROOM_PRESET_PROTEINS,
    STANDARD_TAGS,
    STANDARD_CLEAVAGE_SITES,
    CODON_TABLE,
    clean_amino_acid_sequence,
    extract_fasta_header_metadata,
    fetch_uniprot_by_id,
    fetch_ncbi_by_id,
    find_longest_orf,
    translate_nucleotide,
    reverse_complement,
    _normalize_uniprot_input,
    _normalize_ncbi_input,
    _classify_ncbi_accession_db,
    _parse_ncbi_fasta_header,
    _extract_uniprot_biochem_annotations,
    parse_natural_language_prompt,
)
from modules.purification_metrics import (
    AA_MASSES,
    WATER_MASS,
    KYTE_DOOLITTLE,
    DIWV,
    analyze_all_parameters,
    calculate_net_charge_at_ph,
)
from modules.buffer_advisor import (
    BUFFER_SYSTEMS,
    COMMON_LAB_REAGENTS,
    REAGENT_LOOKUP,
    PRESET_RECIPE_TEMPLATES,
    advise_iex_strategy,
    advise_hic_strategy,
    recommend_full_purification_workflow,
    compute_custom_buffer_recipe,
    convert_weight_and_molarity,
    generate_affinity_recipes,
)
from modules.sec_simulator import (
    SEC_COLUMNS,
    SEC_STANDARD_MARKERS,
    CONFORMATION_PROFILES,
    predict_stokes_radius_rh,
    calculate_dual_uv_ratio,
    calculate_retention_volume,
    recommend_optimal_columns,
    get_column_marker_elutions,
    simulate_chromatogram_peaks,
)
from modules.sds_page_simulator import (
    STANDARD_LADDERS,
    calculate_sds_rf,
    generate_sds_page_figure,
)
from modules.secondary_structure import (
    predict_secondary_structure,
    generate_secondary_structure_figure,
    format_ss_sequence_blocks,
    compute_q3,
    compute_sov99,
)


# Authoritative 2021 IUPAC Standard Atomic Weights for stoichiometric verification
IUPAC_ATOMIC_WEIGHTS = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Na": 22.990,
    "Mg": 24.305,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "K": 39.098,
    "Ca": 40.078,
    "Zn": 65.38,
}


def _compute_iupac_mw(element_counts: dict) -> float:
    return sum(IUPAC_ATOMIC_WEIGHTS[el] * cnt for el, cnt in element_counts.items())


def test_protparam_golden_reference_and_sequences(root_dir: str) -> None:
    print("=== 1. Testing ExPASy ProtParam Golden Reference (Hen Egg-White Lysozyme, P00698) ===")
    assert len(LYSOZYME_MATURE_SEQ) == 129
    assert LYSOZYME_MATURE_SEQ.startswith("KVFGRCELAA") and LYSOZYME_MATURE_SEQ.endswith("QAWIRGCRL")

    # Verify foundational constant tables
    assert len(AA_MASSES) == 20 and abs(WATER_MASS - 18.01528) < 1e-5
    assert len(KYTE_DOOLITTLE) == 20
    assert len(DIWV) == 400, f"Expected 400 dipeptide instability entries, got {len(DIWV)}"

    lyso = analyze_all_parameters(LYSOZYME_MATURE_SEQ)
    ext = lyso["extinction"]
    print(
        f"HEWL (129 aa): MW={lyso['mw_da']:.2f} Da ({lyso['mw_kda']:.2f} kDa) | "
        f"pI={lyso['pi']:.2f} | eps_red={ext['molar_extinction_reduced']} "
        f"(A0.1%={ext['abs_01_reduced']:.3f}) | eps_ox={ext['molar_extinction_cystines']} "
        f"(A0.1%={ext['abs_01_cystines']:.3f}) | II={lyso['instability_index']:.2f} | "
        f"AI={lyso['aliphatic_index']:.2f} | GRAVY={lyso['gravy']:+.3f}"
    )

    # Exact ExPASy ProtParam golden checks on mature HEWL (P00698 residues 19-147)
    assert abs(lyso["mw_da"] - 14313.14) <= 0.02, f"Expected MW 14313.14 Da, got {lyso['mw_da']}"
    assert lyso["mw_kda"] == 14.31, f"Expected 14.31 kDa, got {lyso['mw_kda']}"
    assert abs(lyso["pi"] - 9.32) <= 0.01, f"Expected Bjellqvist ProtParam pI 9.32, got {lyso['pi']}"

    # Pace et al. (1995): 6 Trp (5500), 3 Tyr (1490), 8 Cys -> 4 cystines (125)
    assert ext["num_Trp"] == 6 and ext["num_Tyr"] == 3 and ext["num_Cys"] == 8
    assert ext["potential_disulfides"] == 4
    assert ext["molar_extinction_reduced"] == 37470
    assert ext["molar_extinction_cystines"] == 37970
    assert abs(ext["abs_01_reduced"] - 2.618) <= 0.001
    assert abs(ext["abs_01_cystines"] - 2.653) <= 0.001

    # Guruprasad et al. (1990) Instability Index = 16.09 (Stable, < 40)
    assert abs(lyso["instability_index"] - 16.09) <= 0.01, f"Expected II 16.09, got {lyso['instability_index']}"
    assert "Stable" in lyso["instability_class"]
    assert "canonical_cv" in lyso["solubility"] and 0.0 <= lyso["solubility"]["solubility_probability_pct"] <= 100.0

    # Ikai (1980) Aliphatic Index: (12 + 2.9*6 + 3.9*(6+8)) / 129 * 100 = 8400 / 129 = 65.1163 -> 65.12
    expected_ai = round(((12 + 2.9 * 6 + 3.9 * (6 + 8)) / 129.0) * 100.0, 2)
    assert expected_ai == 65.12
    assert abs(lyso["aliphatic_index"] - 65.12) <= 0.01, f"Expected AI 65.12, got {lyso['aliphatic_index']}"

    # Kyte & Doolittle (1982) GRAVY: -60.9 / 129 = -0.47209 -> -0.472
    assert abs(lyso["gravy"] - (-0.472)) <= 0.001, f"Expected GRAVY -0.472, got {lyso['gravy']}"

    # Charge titration monotonicity across pH 2.0 -> 12.0 & zero-crossing at pI
    charges = lyso["charge_values"]
    assert all(charges[i] >= charges[i + 1] - 1e-9 for i in range(len(charges) - 1))
    assert abs(calculate_net_charge_at_ph(LYSOZYME_MATURE_SEQ, lyso["pi"])) < 0.05
    assert calculate_net_charge_at_ph(LYSOZYME_MATURE_SEQ, 7.0) > 7.0

    # Classroom presets & exact ProtParam cross-checks + zero legacy eIF4E / Q9FLB8 check
    assert len(CLASSROOM_PRESET_PROTEINS) >= 7
    valid_aa = set(AA_MASSES.keys())
    expected_presets = {
        "lysozyme_native": ("P00698", 129, 14313.14, 9.32, 16.09),
        "lysozyme_his_tev": ("P00698", 143, 16119.09, 9.16, 18.91),
        "gfp": ("P42212", 238, 26886.32, 5.67, 31.06),
        "carbonic_anhydrase": ("P00918", 259, 29114.86, 6.86, 20.03),
        "myoglobin": ("P02185", 153, 17199.90, 8.71, 15.09),
        "lipase_a": ("P37957", 181, 19361.88, 9.25, 10.56),
        "bsa": ("P02769", 583, 66432.96, 5.60, 40.11),
    }
    assert "PGDFG" in CLASSROOM_PRESET_PROTEINS["myoglobin"]["sequence"]
    for key, (exp_uid, exp_len, exp_mw_da, exp_pi, exp_ii) in expected_presets.items():
        preset = CLASSROOM_PRESET_PROTEINS[key]
        assert preset["uniprot_id"] == exp_uid
        seq = preset["sequence"]
        assert len(seq) == exp_len and set(seq).issubset(valid_aa), f"Invalid sequence in preset {key}"
        pm = analyze_all_parameters(seq)
        assert abs(pm["mw_da"] - exp_mw_da) <= 0.05, f"MW mismatch on {key}: {pm['mw_da']} vs {exp_mw_da}"
        assert abs(pm["pi"] - exp_pi) <= 0.02, f"pI mismatch on {key}: {pm['pi']} vs {exp_pi}"
        assert abs(pm["instability_index"] - exp_ii) <= 0.02, f"II mismatch on {key}: {pm['instability_index']} vs {exp_ii}"

    for rel_path in ["app.py", os.path.join("modules", "sequence_parser.py")]:
        with open(os.path.join(root_dir, rel_path), "r", encoding="utf-8") as fh:
            txt = fh.read()
            assert "eif4e" not in txt.lower(), f"Legacy eIF4E found in {rel_path}"
            assert "q9flb8" not in txt.lower(), f"Legacy Q9FLB8 found in {rel_path}"

    # NCBI Standard Genetic Code (64 codons: 61 sense + 3 stop codons)
    assert len(CODON_TABLE) == 64
    stop_codons = {c for c, aa in CODON_TABLE.items() if aa == "*"}
    assert stop_codons == {"TAA", "TAG", "TGA"}
    assert CODON_TABLE["ATG"] == "M" and CODON_TABLE["TGG"] == "W"

    # Synthetic bacterial Lysozyme coding ORF (ATG + 129 aa + TAA = 393 bp -> 130 aa, 14.44 kDa)
    aa_to_codon = {
        "A": "GCT", "C": "TGC", "D": "GAT", "E": "GAA", "F": "TTT",
        "G": "GGT", "H": "CAT", "I": "ATT", "K": "AAA", "L": "CTG",
        "M": "ATG", "N": "AAC", "P": "CCG", "Q": "CAG", "R": "CGT",
        "S": "TCT", "T": "ACT", "V": "GTT", "W": "TGG", "Y": "TAT",
    }
    synth_orf = "ATG" + "".join(aa_to_codon[aa] for aa in LYSOZYME_MATURE_SEQ) + "TAA"
    assert len(synth_orf) == 393
    orf_res = find_longest_orf(synth_orf)
    assert orf_res["orf_length_aa"] == 130
    assert orf_res["sequence"] == "M" + LYSOZYME_MATURE_SEQ
    assert analyze_all_parameters(orf_res["sequence"])["mw_kda"] == 14.44
    # mRNA (U instead of T) must translate identically
    rna_orf = synth_orf.replace("T", "U")
    assert translate_nucleotide(rna_orf, frame=1) == "M" + LYSOZYME_MATURE_SEQ + "*"

    # FASTA cleaning & Natural Language Construct Builder + Protease Sites
    cleaned, hdr, warns = clean_amino_acid_sequence(">sp|P00698|LYSC_CHICK\nKVFGR 123 CELAA*\n")
    assert cleaned == "KVFGRCELAA" and hdr == "sp|P00698|LYSC_CHICK" and len(warns) == 0
    _, _, warns_nonstd = clean_amino_acid_sequence("KVFGRXCELAA")
    assert len(warns_nonstd) == 1

    nl_his = parse_natural_language_prompt("Chicken egg white lysozyme with N-terminal 6xHis tag and TEV site")
    assert nl_his["tag_detected"] == "6xHis"
    assert "TEV" in nl_his["cleavage_detected"]
    assert nl_his["length"] == 7 + 7 + 129  # MHHHHHH + ENLYFQG + 129 aa

    nl_strep = parse_natural_language_prompt("Green fluorescent protein with C-terminal Strep tag")
    assert "Strep" in nl_strep["tag_detected"] and nl_strep["tag_position"] == "C-terminal"
    assert nl_strep["full_construct_sequence"].endswith("WSHPQFEK")

    assert STANDARD_CLEAVAGE_SITES["tev"]["seq"] == "ENLYFQG" and STANDARD_CLEAVAGE_SITES["tev"]["cut_offset"] == 6
    assert STANDARD_CLEAVAGE_SITES["3c"]["seq"] == "LEVLFQGP" and STANDARD_CLEAVAGE_SITES["3c"]["cut_offset"] == 6
    assert STANDARD_CLEAVAGE_SITES["thrombin"]["seq"] == "LVPRGS" and STANDARD_CLEAVAGE_SITES["thrombin"]["cut_offset"] == 4
    assert STANDARD_CLEAVAGE_SITES["factor_xa"]["seq"] == "IEGR" and STANDARD_CLEAVAGE_SITES["factor_xa"]["cut_offset"] == 4
    assert STANDARD_CLEAVAGE_SITES["enterokinase"]["seq"] == "DDDDK" and STANDARD_CLEAVAGE_SITES["enterokinase"]["cut_offset"] == 5
    assert len(STANDARD_TAGS) >= 10


def test_chemical_mw_stoichiometry_and_liquids() -> None:
    print("\n=== 2. Testing IUPAC Chemical Formula Weights, Molarity/Mass (g & mg) & Neat Liquids ===")
    assert len(BUFFER_SYSTEMS) == 12
    assert len(COMMON_LAB_REAGENTS) >= 45

    # Verify buffering windows in BUFFER_SYSTEMS
    for buf in BUFFER_SYSTEMS:
        assert buf["min_ph"] < buf["max_ph"], f"Invalid pH window for {buf['name']}"
        if "Citrate" in buf["name"]:
            # Polyprotic citrate: pKa1=3.13, pKa2=4.76, pKa3=6.40 -> working range 3.0-6.2
            assert buf["min_ph"] == 3.0 and buf["max_ph"] == 6.2 and abs(buf["pka"] - 6.40) <= 0.02
        else:
            assert buf["min_ph"] < buf["pka"] < buf["max_ph"], f"pKa outside range for {buf['name']}"

    # Independent IUPAC formula-weight audit across key crystalline solids & pure liquids
    expected_formulas = [
        ("Tris-Base", {"C": 4, "H": 11, "N": 1, "O": 3}),
        ("Tris-HCl", {"C": 4, "H": 12, "Cl": 1, "N": 1, "O": 3}),
        ("HEPES (Free Acid)", {"C": 8, "H": 18, "N": 2, "O": 4, "S": 1}),
        ("HEPES Sodium Salt", {"C": 8, "H": 17, "N": 2, "Na": 1, "O": 4, "S": 1}),
        ("MES (Free Acid, Anhydrous)", {"C": 6, "H": 13, "N": 1, "O": 4, "S": 1}),
        ("MES (Monohydrate)", {"C": 6, "H": 15, "N": 1, "O": 5, "S": 1}),
        ("MOPS (Free Acid, Anhydrous)", {"C": 7, "H": 15, "N": 1, "O": 4, "S": 1}),
        ("PIPES (Free Acid, Anhydrous)", {"C": 8, "H": 18, "N": 2, "O": 6, "S": 2}),
        ("Bis-Tris (Free Base, Anhydrous)", {"C": 8, "H": 19, "N": 1, "O": 5}),
        ("Na2HPO4 (Anhydrous)", {"Na": 2, "H": 1, "P": 1, "O": 4}),
        ("Na2HPO4·7H2O", {"Na": 2, "H": 15, "P": 1, "O": 11}),
        ("NaH2PO4 (Anhydrous)", {"Na": 1, "H": 2, "P": 1, "O": 4}),
        ("NaH2PO4·H2O", {"Na": 1, "H": 4, "P": 1, "O": 5}),
        ("KH2PO4 (Anhydrous)", {"K": 1, "H": 2, "P": 1, "O": 4}),
        ("Sodium Acetate (Anhydrous)", {"C": 2, "H": 3, "Na": 1, "O": 2}),
        ("Sodium Acetate (Trihydrate)", {"C": 2, "H": 9, "Na": 1, "O": 5}),
        ("Trisodium Citrate·2H2O", {"C": 6, "H": 9, "Na": 3, "O": 9}),
        ("Citric Acid (Anhydrous)", {"C": 6, "H": 8, "O": 7}),
        ("Bicine (Anhydrous)", {"C": 6, "H": 13, "N": 1, "O": 4}),
        ("CHES (Free Acid, Anhydrous)", {"C": 8, "H": 17, "N": 1, "O": 3, "S": 1}),
        ("CAPS (Free Acid, Anhydrous)", {"C": 9, "H": 19, "N": 1, "O": 3, "S": 1}),
        ("NaCl (Sodium Chloride)", {"Na": 1, "Cl": 1}),
        ("KCl (Potassium Chloride)", {"K": 1, "Cl": 1}),
        ("Ammonium Sulfate (NH4)2SO4 [HIC Salt]", {"N": 2, "H": 8, "S": 1, "O": 4}),
        ("Sodium Sulfate (Na2SO4, Anhydrous) [HIC Salt]", {"Na": 2, "S": 1, "O": 4}),
        ("MgCl2·6H2O", {"Mg": 1, "Cl": 2, "H": 12, "O": 6}),
        ("CaCl2·2H2O", {"Ca": 1, "Cl": 2, "H": 4, "O": 2}),
        ("ZnSO4·7H2O", {"Zn": 1, "S": 1, "O": 11, "H": 14}),
        ("Imidazole (His-Tag Eluant)", {"C": 3, "H": 4, "N": 2}),
        ("D-Desthiobiotin (Strep Eluant)", {"C": 10, "H": 18, "N": 2, "O": 3}),
        ("D-Biotin (Strep-Tactin XT Eluant)", {"C": 10, "H": 16, "N": 2, "O": 3, "S": 1}),
        ("Reduced L-Glutathione (GSH • GST Eluant)", {"C": 10, "H": 17, "N": 3, "O": 6, "S": 1}),
        ("D-Maltose·H2O (MBP Eluant)", {"C": 12, "H": 24, "O": 12}),
        ("DTT (Dithiothreitol)", {"C": 4, "H": 10, "O": 2, "S": 2}),
        ("TCEP-HCl", {"C": 9, "H": 16, "Cl": 1, "O": 6, "P": 1}),
        ("2-Mercaptoethanol (β-ME)", {"C": 2, "H": 6, "O": 1, "S": 1}),
        ("EDTA Disodium·2H2O", {"C": 10, "H": 18, "N": 2, "Na": 2, "O": 10}),
        ("EGTA (Free Acid)", {"C": 14, "H": 24, "N": 2, "O": 10}),
        ("Glycerol (Liquid, ρ = 1.261 g/mL)", {"C": 3, "H": 8, "O": 3}),
        ("Sucrose", {"C": 12, "H": 22, "O": 11}),
        ("L-Arginine HCl (Anti-Aggregation)", {"C": 6, "H": 15, "Cl": 1, "N": 4, "O": 2}),
        ("SDS (Sodium Dodecyl Sulfate)", {"C": 12, "H": 25, "Na": 1, "O": 4, "S": 1}),
        ("Urea (Denaturant)", {"C": 1, "H": 4, "N": 2, "O": 1}),
        ("Guanidine-HCl (Denaturant)", {"C": 1, "H": 6, "Cl": 1, "N": 3}),
        ("PMSF (Protease Inhibitor)", {"C": 7, "H": 7, "F": 1, "O": 2, "S": 1}),
    ]
    for r_name, counts in expected_formulas:
        assert r_name in REAGENT_LOOKUP, f"Missing reagent {r_name} in REAGENT_LOOKUP"
        iupac_mw = round(_compute_iupac_mw(counts), 2)
        code_mw = float(REAGENT_LOOKUP[r_name]["mw"])
        assert abs(code_mw - iupac_mw) <= 0.02, f"MW mismatch for {r_name}: code={code_mw}, IUPAC={iupac_mw}"

    # Verify solid & neat-liquid buffer recipe stoichiometry (500 mL)
    rows = [
        {"Reagent": "Tris-Base", "MW (g/mol)": 121.14, "Value": 50.0, "Unit": "mM"},
        {"Reagent": "NaCl (Sodium Chloride)", "MW (g/mol)": 58.44, "Value": 0.5, "Unit": "M"},
        {"Reagent": "Ammonium Sulfate (NH4)2SO4 [HIC Salt]", "MW (g/mol)": 132.14, "Value": 66.07, "Unit": "g"},
        {"Reagent": "Imidazole (His-Tag Eluant)", "MW (g/mol)": 68.08, "Value": 680.8, "Unit": "mg"},
        {"Reagent": "2-Mercaptoethanol (β-ME)", "MW (g/mol)": 78.13, "Value": 5.0, "Unit": "mM"},
        {"Reagent": "Triton X-100", "MW (g/mol)": 625.00, "Value": 0.1, "Unit": "% (v/v)"},
        {"Reagent": "Glycerol (Liquid, ρ = 1.261 g/mL)", "MW (g/mol)": 92.09, "Value": 10.0, "Unit": "% (v/v)"},
        {"Reagent": "Sucrose", "MW (g/mol)": 342.30, "Value": 5.0, "Unit": "% (w/v)"},
        {"Reagent": "PMSF (Protease Inhibitor)", "MW (g/mol)": 174.19, "Value": 1.0, "Unit": "mM"},
    ]
    calc = compute_custom_buffer_recipe(rows, volume_ml=500.0)
    assert len(calc) == 9

    # 1. Solid mM: 50 mM Tris-Base in 0.5 L -> 3.0285 g / 3028.50 mg, Liquid Vol = None
    assert abs(calc[0]["Weight (g)"] - 3.0285) < 1e-4 and abs(calc[0]["Weight (mg)"] - 3028.50) < 0.05
    assert calc[0]["Liquid Vol (mL)"] is None

    # 2. Solid M: 0.5 M NaCl in 0.5 L -> 14.6100 g / 14610.00 mg
    assert abs(calc[1]["Weight (g)"] - 14.61) < 1e-3 and abs(calc[1]["Weight (mg)"] - 14610.0) < 0.1

    # 3. Direct grams (g) -> Molarity: 66.07 g (NH4)2SO4 in 0.5 L -> 1.000 M
    assert "1.000 M" in calc[2]["Final Concentration"]

    # 4. Direct milligrams (mg) -> Molarity: 680.8 mg Imidazole in 0.5 L -> 20.00 mM
    assert "20.00 mM" in calc[3]["Final Concentration"] and abs(calc[3]["Weight (g)"] - 0.6808) < 1e-4

    # 5. Neat liquid in mM: 5 mM β-ME (MW 78.13, ρ=1.114 g/mL) in 0.5 L -> 0.1953 g (195.33 mg) -> 0.175337 mL (175.3 µL)
    assert abs(calc[4]["Weight (mg)"] - 195.325) < 0.05
    assert calc[4]["Liquid Vol (mL)"] is not None
    assert abs(calc[4]["Liquid Vol (mL)"] * 1000.0 - 175.337) < 0.05  # 175.3 µL (< 1 mL!)
    assert "175.3 µL" in calc[4]["Bench Tip"] and "fume hood" in calc[4]["Bench Tip"].lower()

    # 6. Neat liquid in % (v/v): 0.1% Triton X-100 (ρ=1.07 g/mL) in 500 mL -> 0.5 mL (500.0 µL), 535.0 mg
    assert abs(calc[5]["Liquid Vol (mL)"] - 0.5) < 1e-5
    assert abs(calc[5]["Weight (mg)"] - 535.0) < 0.05

    # 7. Neat liquid >= 1 mL: 10% (v/v) Glycerol (ρ=1.261 g/mL) in 500 mL -> 50.0 mL, 63.0500 g
    assert abs(calc[6]["Liquid Vol (mL)"] - 50.0) < 1e-4
    assert abs(calc[6]["Weight (g)"] - 63.05) < 1e-3

    # 8. % (w/v) solid: 5% w/v Sucrose in 500 mL -> 25.0000 g (25000.00 mg)
    assert abs(calc[7]["Weight (g)"] - 25.0) < 1e-4

    # 9. PMSF aqueous insolubility warning (100 mM isopropanol/ethanol stock)
    assert "isopropanol" in calc[8]["Bench Tip"].lower()

    # Over-saturation warning check (> 4.1 M Ammonium Sulfate)
    sat_check = compute_custom_buffer_recipe(
        [{"Reagent": "Ammonium Sulfate (NH4)2SO4 [HIC Salt]", "MW (g/mol)": 132.14, "Value": 5.0, "Unit": "M"}],
        volume_ml=100.0,
    )
    assert "EXCEEDS SATURATION" in sat_check[0]["Bench Tip"]

    # Two-way converter round-trip checks (M, mM, µM, nM <-> g, mg, µg)
    c2w = convert_weight_and_molarity("conc_to_weight", mw_g_mol=58.44, volume_ml=1000.0, input_value=150.0, input_unit="mM")
    assert abs(c2w["mass_g"] - 8.766) < 1e-3 and abs(c2w["mass_mg"] - 8766.0) < 0.1
    w2c = convert_weight_and_molarity("weight_to_conc", mw_g_mol=58.44, volume_ml=1000.0, input_value=8766.0, input_unit="mg")
    assert abs(w2c["molarity_mm"] - 150.0) < 0.01 and abs(w2c["pct_wv"] - 0.8766) < 0.001

    # Affinity & HIC preset recipe generators (His, Strep, Strep-XT, GST, MBP, HIC, SEC)
    for sys_key in ["6xHis", "Strep-tag II", "Strep-Tactin XT", "GST", "MBP", "HIC", "SEC"]:
        rec = generate_affinity_recipes(sys_key, volume_ml=500.0)
        assert len(rec["buffers"]) >= 1
    assert any("Biotin" in c["name"] and "50 mM" in c["conc"] for c in generate_affinity_recipes("Strep-Tactin XT")["buffers"][1]["components"])
    assert any("KCl" in c["name"] for c in generate_affinity_recipes("GST")["buffers"][0]["components"])
    assert len(PRESET_RECIPE_TEMPLATES) >= 10


def test_all_seven_purification_methods() -> None:
    print("\n=== 3. Testing All 7 Purification Methods (Affinity, Heparin, AS, IEX, HIC, CHT/MMC, SEC) ===")
    lyso_metrics = analyze_all_parameters(LYSOZYME_MATURE_SEQ)

    # A. Tagless Hen Egg-White Lysozyme at pH 7.4 (pI 9.32, basic Q > +7 -> CEX + Heparin + Phenyl HIC + Disulfide alert)
    q_lyso_74 = calculate_net_charge_at_ph(LYSOZYME_MATURE_SEQ, 7.4)
    wf_lyso = recommend_full_purification_workflow(lyso_metrics, 7.4, q_lyso_74, tag_detected="None")
    assert len(wf_lyso["all_methods"]) == 7
    assert "Cation Exchange" in wf_lyso["iex"]["recommended_column"]
    assert "Heparin" in wf_lyso["roadmap"][0]["title"] or "SP" in wf_lyso["roadmap"][0]["title"]
    assert "Phenyl" in wf_lyso["hic"]["recommended_resin"]
    assert wf_lyso["hic"]["is_required_or_strong"] is True
    assert "Superdex 75" in wf_lyso["roadmap"][2]["title"]
    assert any("Disulfide" in a for a in wf_lyso["alerts"])
    assert any("Hydroxyapatite" in m["Method"] for m in wf_lyso["all_methods"])
    assert any("Ammonium Sulfate" in m["Method"] for m in wf_lyso["all_methods"])

    # B. Isoelectric precipitation risk alert near pI (|pH - pI| < 0.5)
    q_lyso_pi = calculate_net_charge_at_ph(LYSOZYME_MATURE_SEQ, 9.3)
    iex_near_pi = advise_iex_strategy(lyso_metrics["pi"], 9.3, q_lyso_pi)
    assert iex_near_pi["risk_level"] == "HIGH RISK OF PRECIPITATION"

    # C. Acidic GFP at pH 8.0 -> Anion Exchange (AEX: HiTrap Q / DEAE)
    gfp_seq = CLASSROOM_PRESET_PROTEINS["gfp"]["sequence"]
    gfp_metrics = analyze_all_parameters(gfp_seq)
    q_gfp_80 = calculate_net_charge_at_ph(gfp_seq, 8.0)
    iex_gfp = advise_iex_strategy(gfp_metrics["pi"], 8.0, q_gfp_80)
    assert q_gfp_80 < -5.0 and "Anion Exchange" in iex_gfp["recommended_column"]

    # D. Carbonic Anhydrase II near its neutral pI (6.86) -> HIC REQUIRED rescue!
    ca2_seq = CLASSROOM_PRESET_PROTEINS["carbonic_anhydrase"]["sequence"]
    ca2_metrics = analyze_all_parameters(ca2_seq)
    near_ph = round(ca2_metrics["pi"] + 0.2, 1)
    hic_ca2 = advise_hic_strategy(ca2_metrics, near_ph, calculate_net_charge_at_ph(ca2_seq, near_ph))
    assert "REQUIRED" in hic_ca2["status_badge"] and hic_ca2["is_required_or_strong"] is True

    # E. Cytiva Sepharose HIC ligand hydrophobicity series (Butyl HP < Phenyl HP < Octyl FF) & alkaline Na2SO4 rule
    lip_seq = CLASSROOM_PRESET_PROTEINS["lipase_a"]["sequence"]
    lip_metrics = analyze_all_parameters(lip_seq)
    hic_lip = advise_hic_strategy(lip_metrics, 7.5, calculate_net_charge_at_ph(lip_seq, 7.5))
    assert "Butyl" in hic_lip["recommended_resin"]

    polar_seq = "DKEQNRSTGH" * 15  # Highly hydrophilic, zero aromatics (GRAVY = -2.35)
    polar_metrics = analyze_all_parameters(polar_seq)
    hic_polar = advise_hic_strategy(polar_metrics, 7.0, calculate_net_charge_at_ph(polar_seq, 7.0))
    assert "Octyl" in hic_polar["recommended_resin"]

    hic_alkaline = advise_hic_strategy(lyso_metrics, 8.5, calculate_net_charge_at_ph(LYSOZYME_MATURE_SEQ, 8.5))
    assert "Na2SO4" in hic_alkaline["starting_salt"] or "Sodium Sulfate" in hic_alkaline["starting_salt"]

    # F. Affinity capture routing for 6xHis, Twin-Strep, GST, MBP, FLAG + SEC column scaling (S75 / S200 / Superose 6)
    for tag_label, expected_kw in [
        ("6xHis", "Ni-NTA"),
        ("Twin-Strep-tag", "Strep-Tactin"),
        ("GST", "Glutathione"),
        ("MBP", "Amylose"),
        ("FLAG-tag", "FLAG"),
    ]:
        wf_tag = recommend_full_purification_workflow(lyso_metrics, 7.5, q_lyso_74, tag_detected=tag_label)
        assert expected_kw in wf_tag["roadmap"][0]["title"]

    bsa_metrics = analyze_all_parameters(CLASSROOM_PRESET_PROTEINS["bsa"]["sequence"])
    wf_bsa = recommend_full_purification_workflow(bsa_metrics, 7.5, -15.0, tag_detected="None")
    assert "Superdex 200" in wf_bsa["roadmap"][2]["title"]

    mega_metrics = dict(bsa_metrics, mw_kda=650.0)
    wf_mega = recommend_full_purification_workflow(mega_metrics, 7.5, -20.0, tag_detected="None")
    assert "Superose 6" in wf_mega["roadmap"][2]["title"]


def test_sec_hydrodynamics_markers_and_benchmark(root_dir: str) -> None:
    print("\n=== 4. Testing SEC Stokes Radii, 8 Columns, Calibration Markers, Dual-UV & Benchmark ===")
    assert len(SEC_COLUMNS) == 8
    for cid, col in SEC_COLUMNS.items():
        vo = col["void_volume_vo_ml"]
        vp = col["liquid_perm_volume_ml"]
        vt = col["bed_volume_vt_ml"]
        assert 0.0 < vo < vp <= vt, f"Invalid volume hierarchy on {cid}: Vo={vo}, Vp={vp}, Vt={vt}"
        assert 0.30 <= (vo / vt) <= 0.34, f"Vo/Vt ratio out of Cytiva 30-34% spec on {cid}: {vo/vt:.3f}"
        assert 0.0 < col["opt_range_min_kda"] < col["opt_range_max_kda"] <= col["exclusion_limit_kda"]

    # Column recommendation logic across size regimes
    assert recommend_optimal_columns(14.31) == ("hiload_16_600_s75", "s75_increase_10_300")
    assert recommend_optimal_columns(150.0) == ("hiload_16_600_s200", "s200_increase_10_300")
    assert recommend_optimal_columns(800.0) == ("hiload_16_600_superose_6", "superose_6_increase_10_300")

    # Stokes radius (Rh) strict shape ordering at fixed 50 kDa polypeptide mass
    shape_order = ["globular", "moderately_elongated", "elongated", "idp_hybrid", "disordered_idp"]
    rhs_50k = [predict_stokes_radius_rh(50.0, c) for c in shape_order]
    assert all(rhs_50k[i] < rhs_50k[i + 1] for i in range(len(rhs_50k) - 1)), f"Shape Rh order violated: {rhs_50k}"

    # Alpha-synuclein (14.46 kDa IDP, Uversky 2002 native coil ~31.6 Å) vs globular 14.46 kDa
    rh_asyn_idp = predict_stokes_radius_rh(14.46, "disordered_idp")
    rh_asyn_glob = predict_stokes_radius_rh(14.46, "globular")
    assert 30.0 <= rh_asyn_idp <= 33.5 and rh_asyn_idp > rh_asyn_glob * 1.5

    # Free nucleic acids have larger hydrodynamic radii than globular proteins of equal mass
    assert predict_stokes_radius_rh(90.8, "dsdna") > predict_stokes_radius_rh(90.8, "globular") * 1.4
    assert predict_stokes_radius_rh(32.1, "ssrna") > predict_stokes_radius_rh(32.1, "globular")

    # Standard calibration marker experimental Rh values & strict hydrodynamic elution monotonicity (Ve increases as Rh decreases)
    marker_rh_lookup = {m["name"]: m["rh_angstrom"] for m in SEC_STANDARD_MARKERS}
    assert marker_rh_lookup["Thyroglobulin"] == 85.0 and marker_rh_lookup["GroEL (14-mer)"] == 79.5
    assert marker_rh_lookup["Ferritin"] == 61.0 and marker_rh_lookup["Aldolase"] == 48.1
    assert marker_rh_lookup["Ovalbumin"] == 30.5 and marker_rh_lookup["Ribonuclease A"] == 16.4
    for cid in ["s75_increase_10_300", "s200_increase_10_300", "superose_6_increase_10_300"]:
        col = SEC_COLUMNS[cid]
        elutions = [m for m in get_column_marker_elutions(col) if not m.get("is_void")]
        elutions_by_rh_desc = sorted(elutions, key=lambda x: x["rh_angstrom"], reverse=True)
        ves = [m["ve_ml"] for m in elutions_by_rh_desc]
        assert all(ves[i] <= ves[i + 1] + 1e-6 for i in range(len(ves) - 1)), f"Non-monotonic Rh->Ve on {cid}: {ves}"
        assert all(col["void_volume_vo_ml"] <= v <= col["bed_volume_vt_ml"] for v in ves)

    # Retention anchors across 3 kDa -> 2.95 MDa & plate-theory peak widths
    s75 = SEC_COLUMNS["s75_increase_10_300"]
    s200 = SEC_COLUMNS["s200_increase_10_300"]
    sup6 = SEC_COLUMNS["superose_6_increase_10_300"]
    hiload_s200 = SEC_COLUMNS["hiload_16_600_s200"]
    ve_3k_s75, _, _ = calculate_retention_volume(3.0, s75)
    ve_29k_s200, _, _ = calculate_retention_volume(29.0, s200)
    ve_3mda_s200, kav_3mda_s200, _ = calculate_retention_volume(2950.0, s200)
    ve_3mda_sup6, kav_3mda_sup6, _ = calculate_retention_volume(2950.0, sup6)
    assert 17.0 < ve_3k_s75 < 21.5
    assert abs(ve_29k_s200 - 16.95) < 0.40
    assert ve_3mda_s200 == s200["void_volume_vo_ml"] and kav_3mda_s200 == 0.0
    assert 10.7 <= ve_3mda_sup6 <= 11.7 and kav_3mda_sup6 > 0.15

    pk_24 = simulate_chromatogram_peaks([{"label": "Lysozyme", "mw_kda": 14.31}], s75)["peaks"][0]
    pk_120 = simulate_chromatogram_peaks([{"label": "BSA", "mw_kda": 66.43}], hiload_s200)["peaks"][0]
    assert 0.75 <= pk_24["peak_width_base_ml"] <= 1.60
    assert 5.0 <= pk_120["peak_width_base_ml"] <= 9.0

    # Nucleic acid MW conversion consistency (147 bp Widom 601 = 90.84 kDa; 100 nt sgRNA = 32.15 kDa) & Dual-UV ratios
    assert abs(147 * 0.61796 - 90.84) < 0.01
    assert abs(100 * 0.32147 - 32.15) < 0.01
    uv_prot = calculate_dual_uv_ratio(108.5, prot_eps01=0.45, na_mw_kda=0.0, na_type=None)
    uv_dna_only = calculate_dual_uv_ratio(0.0, prot_eps01=0.0, na_mw_kda=90.8, na_type="dsdna")
    uv_rna_only = calculate_dual_uv_ratio(0.0, prot_eps01=0.0, na_mw_kda=32.1, na_type="ssrna")
    uv_ncp = calculate_dual_uv_ratio(108.5, prot_eps01=0.45, na_mw_kda=90.8, na_type="dsdna")
    uv_cas9_rnp = calculate_dual_uv_ratio(158.4, prot_eps01=0.76, na_mw_kda=32.1, na_type="ssrna")
    assert uv_prot["a260_a280_ratio"] == 0.57
    assert abs(uv_dna_only["a260_a280_ratio"] - 1.85) <= 0.02
    assert abs(uv_rna_only["a260_a280_ratio"] - 2.00) <= 0.02
    assert 1.70 <= uv_ncp["a260_a280_ratio"] <= 1.85
    assert 1.60 <= uv_cas9_rnp["a260_a280_ratio"] <= 1.75

    # Benchmark JSON & all 7 published open-access case studies
    bench_path = os.path.join(root_dir, "data", "sec_benchmark_1200.json")
    assert os.path.exists(bench_path)
    with open(bench_path, "r", encoding="utf-8") as bf:
        bench = json.load(bf)
    assert bench["summary"]["total_pdb_biomolecules"] >= 1000
    assert bench["summary"]["literature_complex_mae_ml"] <= 0.25
    assert len(bench["literature_cases"]) >= 7
    for case in bench["literature_cases"]:
        col_cfg = SEC_COLUMNS[case["column_id"]]
        sim = simulate_chromatogram_peaks(case["species"], col_cfg)
        assert len(sim["peaks"]) == len(case["species"])


def test_sds_page_monotonicity_and_live_app_ui(root_dir: str) -> None:
    print("\n=== 5. Testing SDS-PAGE 2D Rf Monotonicity & Headless Streamlit App UI ===")
    ladder_bands = STANDARD_LADDERS["precision_plus"]["bands"]
    ladder_mws = [b["mw_kda"] for b in ladder_bands]
    assert ladder_mws == [250.0, 150.0, 100.0, 75.0, 50.0, 37.0, 25.0, 20.0, 15.0, 10.0]
    ladder_colors = {b["mw_kda"]: b["color_type"] for b in ladder_bands}
    assert ladder_colors[75.0] == "pink_ref" and ladder_colors[25.0] == "pink_ref"
    assert "kaleidoscope" in STANDARD_LADDERS

    # 1. Strict Rf monotonicity across MW at fixed gel % (4% - 22%: smaller polypeptides migrate faster -> larger Rf)
    for gel_pct in [4.0, 6.0, 10.0, 12.0, 15.0, 20.0, 22.0]:
        rfs = [calculate_sds_rf(mw, gel_pct) for mw in ladder_mws]
        assert all(0.06 <= rf <= 0.94 for rf in rfs)
        assert all(rfs[i] < rfs[i + 1] for i in range(len(rfs) - 1)), f"Rf not strictly increasing at {gel_pct}%: {rfs}"

    # 2. Strict Rf monotonicity across gel % (4% - 22%) at fixed MW (tighter acrylamide sieve -> smaller Rf)
    gel_grid = [4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 22.0]
    for test_mw in [14.31, 29.0, 66.43, 150.0]:
        rfs_across_gels = [calculate_sds_rf(test_mw, g) for g in gel_grid]
        assert all(
            rfs_across_gels[i] > rfs_across_gels[i + 1] for i in range(len(rfs_across_gels) - 1)
        ), f"Rf did not decrease monotonically with gel % for {test_mw} kDa: {rfs_across_gels}"

    # 3. Coomassie & Silver Stain Plotly figure generation
    fig_coom = generate_sds_page_figure(14.31, gel_pct=15.0, stain_type="Coomassie Blue", show_purification_stages=True)
    fig_silv = generate_sds_page_figure(14.31, gel_pct=15.0, stain_type="Silver Stain", show_purification_stages=False)
    assert len(fig_coom.data) > 0 and len(fig_silv.data) > 0

    # 4. Audit hardcoded constants, accessions & safety notes in app.py + headless AppTest execution
    app_path = os.path.join(root_dir, "app.py")
    with open(app_path, "r", encoding="utf-8") as af:
        app_src = af.read()

    assert "use_container_width" not in app_src, "Deprecated use_container_width found in app.py"
    assert "NP_990612.2" in app_src and "NM_205281.2" in app_src
    assert "P00698" in app_src and "P61626" in app_src and "P42212" in app_src
    assert re.search(r"DSDNA_MW_KDA_PER_BP\s*=\s*0\.61796", app_src)
    assert re.search(r"SSRNA_MW_KDA_PER_NT\s*=\s*0\.32147", app_src)
    assert "Met + Mature Lysozyme Coding ORF, 130 aa" in app_src
    assert "Always add acid to water, NEVER pour water into concentrated acid" in app_src

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(app_path).run(timeout=30)
    assert len(at.exception) == 0, f"Streamlit AppTest raised exceptions: {at.exception}"


def test_natural_language_disambiguation_and_ui(root_dir: str) -> None:
    print("\n=== 6. Testing Natural Language Target vs. Tag/Cleavage Disambiguation & Clean UI ===")
    from unittest.mock import patch
    import requests as _requests
    from streamlit.testing.v1 import AppTest

    def _verify_nl_suite() -> None:
        # 1. Untagged Human GST ("GST human" and "human GST") — MUST return Human GST, NOT GST-tagged random protein!
        for q_gst_hum in ["GST human", "human GST"]:
            r_gh = parse_natural_language_prompt(q_gst_hum)
            assert r_gh["uniprot_id"] in {"P09211", "P08263", "P09488"}, f"Unexpected Human GST ID: {r_gh}"
            assert "glutathione s-transferase" in r_gh["identified_protein"].lower()
            assert "homo sapiens" in r_gh["identified_organism"].lower()
            assert r_gh["tag_detected"] == "None" and r_gh["tag_position"] == "N/A"
            assert r_gh["cleavage_detected"] == "None"
            assert r_gh["full_construct_sequence"] == r_gh["canonical_sequence"]

        # 2. Standalone "GST" -> untagged Glutathione S-transferase
        r_gst = parse_natural_language_prompt("GST")
        assert r_gst["uniprot_id"] in {"P08515", "P09211", "P08263", "P09488"}
        assert "glutathione s-transferase" in r_gst["identified_protein"].lower()
        assert r_gst["tag_detected"] == "None" and r_gst["cleavage_detected"] == "None"

        # 3. "Human GST with N-terminal 6xHis tag" -> Human GST target WITH 6xHis tag (must NOT strip GST from target!)
        r_his_gst = parse_natural_language_prompt("Human GST with N-terminal 6xHis tag")
        assert r_his_gst["uniprot_id"] in {"P09211", "P08263", "P09488"}
        assert "glutathione s-transferase" in r_his_gst["identified_protein"].lower()
        assert r_his_gst["tag_detected"] == "6xHis" and r_his_gst["tag_position"] == "N-terminal"
        assert r_his_gst["cleavage_detected"] == "None"
        assert r_his_gst["full_construct_sequence"].startswith("MHHHHHH")
        assert r_his_gst["length"] == len(r_his_gst["canonical_sequence"]) + 7

        # 4. Dual-meaning proteins & proteases asked as target proteins (untagged!)
        r_mbp_ec = parse_natural_language_prompt("MBP E. coli")
        assert r_mbp_ec["uniprot_id"] == "P0AEX9"
        assert r_mbp_ec["tag_detected"] == "None" and r_mbp_ec["cleavage_detected"] == "None"

        r_mbp_hu = parse_natural_language_prompt("Human MBP")
        assert r_mbp_hu["uniprot_id"] == "P02686"
        assert "myelin basic" in r_mbp_hu["identified_protein"].lower()
        assert r_mbp_hu["tag_detected"] == "None" and r_mbp_hu["cleavage_detected"] == "None"

        r_myc = parse_natural_language_prompt("Human c-Myc")
        assert r_myc["uniprot_id"] == "P01106"
        assert "myc" in r_myc["identified_protein"].lower()
        assert r_myc["tag_detected"] == "None" and r_myc["cleavage_detected"] == "None"

        r_thr = parse_natural_language_prompt("Human thrombin")
        assert r_thr["uniprot_id"] == "P00734"
        assert "thrombin" in r_thr["identified_protein"].lower()
        assert r_thr["tag_detected"] == "None" and r_thr["cleavage_detected"] == "None"

        r_tev = parse_natural_language_prompt("TEV protease")
        assert r_tev["uniprot_id"] == "P04517"
        assert r_tev["length"] == 242
        assert r_tev["tag_detected"] == "None" and r_tev["cleavage_detected"] == "None"

        # 5. Tagged / cleavable constructs across varied grammar forms
        r_lyso_full = parse_natural_language_prompt("Chicken egg white lysozyme with N-terminal 6xHis tag and TEV site")
        assert r_lyso_full["uniprot_id"] == "P00698" and r_lyso_full["length"] == 143
        assert r_lyso_full["tag_detected"] == "6xHis" and "TEV" in r_lyso_full["cleavage_detected"]

        r_lyso_hyph = parse_natural_language_prompt("6xHis-TEV-lysozyme")
        assert r_lyso_hyph["uniprot_id"] == "P00698" and r_lyso_hyph["length"] == 143
        assert r_lyso_hyph["tag_detected"] == "6xHis" and "TEV" in r_lyso_hyph["cleavage_detected"]

        r_lyso_inline = parse_natural_language_prompt("His-tagged lysozyme")
        assert r_lyso_inline["uniprot_id"] == "P00698" and r_lyso_inline["length"] == 136
        assert r_lyso_inline["tag_detected"] == "6xHis" and r_lyso_inline["cleavage_detected"] == "None"

        r_gfp_strep = parse_natural_language_prompt("Green fluorescent protein with C-terminal Strep tag")
        assert r_gfp_strep["uniprot_id"] == "P42212"
        assert r_gfp_strep["tag_detected"] == "Strep-tag II" and r_gfp_strep["tag_position"] == "C-terminal"
        assert r_gfp_strep["full_construct_sequence"].endswith("WSHPQFEK")

        for q_gst_p53 in ["GST-tagged human p53", "Human p53 with N-terminal GST tag", "GST-p53"]:
            r_gp53 = parse_natural_language_prompt(q_gst_p53)
            assert r_gp53["uniprot_id"] == "P04637", f"Expected P04637 for {q_gst_p53}, got {r_gp53}"
            assert r_gp53["tag_detected"] == "GST (Glutathione S-transferase)"
            assert r_gp53["tag_position"] == "N-terminal"
            assert r_gp53["length"] == 220 + 393

        r_flag_p53 = parse_natural_language_prompt("human p53 C-terminal FLAG tag")
        assert r_flag_p53["uniprot_id"] == "P04637"
        assert r_flag_p53["tag_detected"] == "FLAG-tag" and r_flag_p53["tag_position"] == "C-terminal"
        assert r_flag_p53["full_construct_sequence"].endswith("DYKDDDDK")

        # 6. C-terminal hyphenated synthetic & dual-meaning constructs + cleavage preservation
        for q_cterm, exp_acc, exp_tag, exp_cleav, exp_suffix in [
            ("GFP-6xHis", "P42212", "6xHis", "None", "HHHHHH"),
            ("GFP-8xHis", "P42212", "8xHis", "None", "HHHHHHHH"),
            ("GFP-10xHis", "P42212", "10xHis", "None", "HHHHHHHHHH"),
            ("GFP-His6", "P42212", "6xHis", "None", "HHHHHH"),
            ("GFP-His", "P42212", "6xHis", "None", "HHHHHH"),
            ("GFP-Twin-Strep", "P42212", "Twin-Strep-tag", "None", "WSHPQFEK"),
            ("lysozyme-TEV-6xHis", "P00698", "6xHis", "TEV Protease", "ENLYFQGHHHHHH"),
            ("p53-c-Myc", "P04637", "c-Myc-tag", "None", "EQKLISEEDL"),
            ("p53-Myc", "P04637", "c-Myc-tag", "None", "EQKLISEEDL"),
            ("GST-FLAG human", "P09211", "FLAG-tag", "None", "DYKDDDDK"),
            ("c-Myc-FLAG human", "P01106", "FLAG-tag", "None", "DYKDDDDK"),
            ("GST-6xHis human", "P09211", "6xHis", "None", "HHHHHH"),
        ]:
            r_ct = parse_natural_language_prompt(q_cterm)
            assert r_ct["uniprot_id"] == exp_acc, f"Unexpected ID on {q_cterm}: {r_ct}"
            assert r_ct["tag_detected"] == exp_tag and r_ct["tag_position"] == "C-terminal", f"Bad C-term tag on {q_cterm}: {r_ct}"
            assert r_ct["cleavage_detected"] == exp_cleav, f"Bad cleavage on {q_cterm}: {r_ct}"
            assert r_ct["full_construct_sequence"].endswith(exp_suffix), f"Bad suffix on {q_cterm}: {r_ct}"

    # Run suite under current network environment AND explicitly under forced offline mode
    _verify_nl_suite()
    with patch("modules.sequence_parser.requests.get", side_effect=_requests.ConnectionError("Simulated offline")):
        _verify_nl_suite()

    # 7. Verify '-containing' Swiss-Prot protein names & '<GENE>-<DIGIT>' names are never truncated by grammar splitter
    from modules.sequence_parser import _split_target_and_modifiers, _strip_known_organisms, _normalize_protein_query

    for raw_q, exp_norm, exp_tax, exp_has_mod in [
        ("HA-1 human", "HA-1", 9606, False),
        ("Bromodomain-containing protein 4 human", "Bromodomain-containing protein 4", 9606, False),
        ("Valosin-containing protein human", "Valosin-containing protein", 9606, False),
        ("Bromodomain-containing protein 4 human with N-terminal 6xHis tag", "Bromodomain-containing protein 4", 9606, True),
    ]:
        t_phr, m_txt, _ = _split_target_and_modifiers(raw_q)
        sp_p, tx_id, _ = _strip_known_organisms(t_phr)
        assert _normalize_protein_query(sp_p) == exp_norm, f"Truncated target on {raw_q!r}: {t_phr!r}"
        assert tx_id == exp_tax, f"Lost organism on {raw_q!r}: {tx_id}"
        assert bool(m_txt.strip()) == exp_has_mod, f"Unexpected modifier {m_txt!r} on {raw_q!r}"

    # Verify clean UI ribbon rendering & non-duplicated organism parentheses in app.py
    app_path = os.path.join(root_dir, "app.py")
    at = AppTest.from_file(app_path).run(timeout=30)
    at.sidebar.radio[0].set_value("💬 Describe in Plain English").run(timeout=30)
    at.sidebar.text_area[0].set_value("GST human").run(timeout=30)
    at.sidebar.button[0].click().run(timeout=30)
    assert len(at.exception) == 0, f"AppTest exception on 'GST human': {at.exception}"
    assert at.session_state["construct_parts"] is None
    assert at.session_state["sequence_metadata"]["tag"] == "None"
    assert "Glutathione S-transferase" in at.session_state["sequence_name"]
    all_md = "\n".join(m.value for m in at.markdown)
    assert "🧬 Active Protein: Glutathione S-transferase" in all_md
    assert "🧬 Active Construct:" not in all_md

    at.sidebar.text_area[0].set_value("Green fluorescent protein with C-terminal Strep tag").run(timeout=30)
    at.sidebar.button[0].click().run(timeout=30)
    assert len(at.exception) == 0
    seq_nm_gfp = at.session_state["sequence_name"]
    assert seq_nm_gfp.lower().count("aequorea victoria") == 1, f"Duplicate organism in sequence_name: {seq_nm_gfp}"

    at.sidebar.text_area[0].set_value("Human GST with N-terminal 6xHis tag").run(timeout=30)
    at.sidebar.button[0].click().run(timeout=30)
    assert len(at.exception) == 0
    assert at.session_state["construct_parts"] is not None and len(at.session_state["construct_parts"]) == 2
    all_md_tagged = "\n".join(m.value for m in at.markdown)
    assert "🧬 Active Construct: Glutathione S-transferase" in all_md_tagged


def test_tree_of_life_nl_parser_and_deterministic_ranking() -> None:
    print("\n=== 7. Testing Universal Tree-of-Life Organism Extraction, Zero-Collision Grammar & Deterministic Ranking ===")
    import random
    from modules.sequence_parser import (
        extract_nl_query_components,
        score_uniprot_candidate,
        rank_uniprot_candidates,
    )

    # 1. Curated Tree-of-Life lab shorthands & common names across all domains
    shorthand_cases = [
        ("Taq DNA polymerase I", 271, "DNA polymerase I"),
        ("Pfu Triosephosphate isomerase", 186497, "Triosephosphate isomerase"),
        ("KOD DNA polymerase", 69014, "DNA polymerase"),
        ("T4 DNA ligase", 10665, "DNA ligase"),
        ("Phage T7 RNA polymerase", 10760, "RNA polymerase"),
        ("Spinach ferredoxin", 3562, "ferredoxin"),
        ("Rice Glutathione S-transferase", 39947, "Glutathione S-transferase"),
        ("Maize catalase", 4577, "catalase"),
        ("Fission yeast cdc2", 284812, "cdc2"),
        ("Baker's yeast hexokinase", 559292, "hexokinase"),
        ("Malaria lactate dehydrogenase", 36329, "lactate dehydrogenase"),
        ("Slime mold myosin", 44689, "myosin"),
        ("SARS-CoV-2 Spike glycoprotein", 2697049, "Spike glycoprotein"),
        ("HIV-1 Integrase", 11676, "Integrase"),
        ("M. tuberculosis katG", 83332, "katG"),
        ("S. aureus Protein A", 93061, "Protein A"),
        ("M. jannaschii DNA topoisomerase", 243232, "DNA topoisomerase"),
    ]
    for prompt, exp_tax, exp_pq in shorthand_cases:
        c = extract_nl_query_components(prompt)
        assert c["organism_taxid"] == exp_tax, f"Expected taxid {exp_tax} on {prompt!r}, got {c}"
        assert c["protein_query"] == exp_pq, f"Expected protein_query {exp_pq!r} on {prompt!r}, got {c['protein_query']!r}"
        assert c["tag_detected"] == "None" and c["cleavage_detected"] == "None"

    # 2. General binomial / trinomial / Candidatus / 'Genus sp.' / viral species extraction + protein guard
    general_org_cases = [
        ("Triosephosphate isomerase Archaeoglobus fulgidus (strain ATCC 49558 / DSM 4304)", "Archaeoglobus fulgidus", "Triosephosphate isomerase"),
        ("Haloferax volcanii Dihydrofolate reductase", "Haloferax volcanii", "Dihydrofolate reductase"),
        ("Cytochrome c from Neurospora crassa", "Neurospora crassa", "Cytochrome c"),
        ("Telomerase reverse transcriptase (Tetrahymena thermophila)", "Tetrahymena thermophila", "Telomerase reverse transcriptase"),
        ("Photosystem II protein D1 Synechocystis sp. PCC 6803", "Synechocystis sp. PCC 6803", "Photosystem II protein D1"),
        ("Coat protein Tobacco mosaic virus (strain vulgare)", "Tobacco mosaic virus", "Coat protein"),
        ("Ammonia monooxygenase Candidatus Nitrosopumilus maritimus (strain SCM1)", "Candidatus Nitrosopumilus maritimus", "Ammonia monooxygenase"),
        ("Oenococcus oeni Malolactic enzyme", "Oenococcus oeni", "Malolactic enzyme"),
    ]
    for prompt, exp_org_sub, exp_pq in general_org_cases:
        c = extract_nl_query_components(prompt)
        assert c["organism_name"] and exp_org_sub.lower() in c["organism_name"].lower(), f"Bad org on {prompt!r}: {c}"
        assert c["protein_query"] == exp_pq, f"Bad protein_query on {prompt!r}: {c}"

    # Verify protein phrases without organisms NEVER trigger false-positive Latin binomials
    for pure_protein in [
        "Factor Xa",
        "Histone H3",
        "Cytochrome c",
        "Protein kinase",
        "Glutathione S-transferase",
        "Calcium and integrin-binding protein",
        "WD repeat-containing protein",
        "Protein-tyrosine phosphatase with PEST-domain",
        "plus-end-directed kinesin",
        "3C-like proteinase",
        "His-tRNA ligase",
        "HA-binding protein",
        "Myc-associated factor X",
    ]:
        cp = extract_nl_query_components(pure_protein)
        assert cp["organism_taxid"] is None and cp["organism_name"] is None, f"False organism on {pure_protein!r}: {cp}"
        assert cp["tag_detected"] == "None" and cp["cleavage_detected"] == "None", f"False tag/cleavage on {pure_protein!r}: {cp}"
        assert cp["protein_query"] == pure_protein, f"Truncated {pure_protein!r} -> {cp['protein_query']!r}"

    # 3. Hard gene/tag/cleavage collision checks (hisA..hisS, HIS6, fliC/Flagellin, HA, H3C1/RAB3C, CAT, double-'with', beta'-6xHis)
    collision_checks = [
        ("hisA Escherichia coli", "hisA", "None", "N/A", "None"),
        ("HIS6 Saccharomyces cerevisiae", "HIS6", "None", "N/A", "None"),
        ("Flagellin Salmonella typhimurium", "Flagellin", "None", "N/A", "None"),
        ("Hemagglutinin Influenza A virus (strain A/Puerto Rico/8/1934 H1N1)", "Hemagglutinin", "None", "N/A", "None"),
        ("Streptavidin Streptomyces avidinii", "Streptavidin", "None", "N/A", "None"),
        ("RAB3C Homo sapiens", "RAB3C", "None", "N/A", "None"),
        ("H3C1 Homo sapiens", "H3C1", "None", "N/A", "None"),
        ("CAT Homo sapiens", "CAT", "None", "N/A", "None"),
        (
            "Interferon-induced protein with tetratricopeptide repeats 3 Homo sapiens with N-terminal 6xHis tag and TEV site",
            "Interferon-induced protein with tetratricopeptide repeats 3",
            "6xHis",
            "N-terminal",
            "TEV Protease",
        ),
        (
            "DNA-directed RNA polymerase subunit beta'-6xHis Bacillus subtilis (strain 168)",
            "DNA-directed RNA polymerase subunit beta'",
            "6xHis",
            "C-terminal",
            "None",
        ),
        (
            "Triosephosphate isomerase with C-terminal FLAG tag and HRV 3C site from Pyrococcus furiosus",
            "Triosephosphate isomerase",
            "FLAG-tag",
            "C-terminal",
            "PreScission / HRV 3C Protease",
        ),
    ]
    for prompt, exp_pq, exp_tag, exp_pos, exp_clv in collision_checks:
        cc = extract_nl_query_components(prompt)
        assert cc["protein_query"] == exp_pq, f"Collision truncated {prompt!r}: {cc}"
        assert (cc["tag_detected"], cc["tag_position"], cc["cleavage_detected"]) == (exp_tag, exp_pos, exp_clv), f"Bad modifier on {prompt!r}: {cc}"

    # 4. Deterministic candidate ranking & permutation-invariance across adversarial decoys
    target_entry = {
        "accession": "P00942",
        "recommended_name": "Triosephosphate isomerase",
        "alt_names": ["TIM"],
        "short_names": ["TPI"],
        "gene_primary": "TPI1",
        "gene_synonyms": [],
        "organism_scientific": "Saccharomyces cerevisiae (strain ATCC 204508 / S288c)",
        "organism_common": "Baker's yeast",
        "tax_id": 559292,
        "annotation_score": 5.0,
        "is_fragment": False,
        "reviewed": True,
    }
    decoys = [
        dict(target_entry, accession="A0A001TREMBL", reviewed=False, annotation_score=5.0),
        dict(target_entry, accession="P00939", organism_scientific="Oryctolagus cuniculus", tax_id=9986),
        dict(target_entry, accession="P99991", recommended_name="Triosephosphate isomerase-binding protein", gene_primary="TPIB1"),
        dict(target_entry, accession="P99992", recommended_name="Triosephosphate isomerase-like receptor", gene_primary="TPIL1"),
        dict(target_entry, accession="P00001", is_fragment=True),
    ]
    pool = [target_entry] + decoys
    rng = random.Random(42)
    for _ in range(10):
        shuffled = list(pool)
        rng.shuffle(shuffled)
        ranked = rank_uniprot_candidates(shuffled, "Baker's yeast Triosephosphate isomerase with N-terminal 6xHis tag")
        assert ranked[0]["accession"] == "P00942", f"Expected P00942 first, got {ranked[0]['accession']}"
        assert score_uniprot_candidate(target_entry, "Baker's yeast Triosephosphate isomerase") > score_uniprot_candidate(decoys[0], "Baker's yeast Triosephosphate isomerase")

    # 5. Regression checks for English protein words + common/clause organisms, 'Genus sp. <STRAIN>', viral short genes,
    #    serotype/subspecies/HIV-subtype ranking, and parenthesized Drosophila gene symbols ('fs(1)K10') + token multiplicity
    edge_extraction_cases = [
        ("Myc proto-oncogene protein rat", 10116, "Rattus norvegicus", "Myc proto-oncogene protein", "None"),
        ("Mechanosensitive ion channel protein 8 from Arabidopsis", 3702, "Arabidopsis thaliana", "Mechanosensitive ion channel protein 8", "None"),
        ("Telomere length regulation protein TEN1 in yeast", 559292, "Saccharomyces cerevisiae", "Telomere length regulation protein TEN1", "None"),
        ("Neuron navigator 3 (zebrafish)", 7955, "Danio rerio", "Neuron navigator 3", "None"),
        ("Pre-protein-C8 from Halobacterium sp. AS7092 fused to C-terminal Strep-tag II", None, "Halobacterium sp.", "Pre-protein-C8", "Strep-tag II"),
        ("Bacillus sp. PS3 ATP synthase subunit c with N-terminal 6xHis tag and TEV site", None, "Bacillus sp.", "ATP synthase subunit c", "6xHis"),
        ("Cocaine esterase (Rhodococcus sp. MB1 Bresler)", None, "Rhodococcus sp.", "Cocaine esterase", "None"),
        ("psbA-TEV-6xHis Synechocystis sp. PCC 6714", 1147, "Synechocystis sp. PCC 6714", "psbA", "6xHis"),
        ("S-methyl-5'-thioadenosine phosphorylase Synechocystis sp. (strain PCC 6803 / GT-S)", None, "Synechocystis sp.", "S-methyl-5'-thioadenosine phosphorylase", "None"),
        ("ATP phosphoribosyltransferase (Salmonella paratyphi A)", None, "Salmonella paratyphi", "ATP phosphoribosyltransferase", "None"),
        ("Newcastle disease virus HN", None, "Newcastle disease virus", "HN", "None"),
        ("Pfu tfb", 186497, "Pyrococcus furiosus", "tfb", "None"),
        ("Xenopus tpr", 8355, "Xenopus laevis", "tpr", "None"),
        ("Picrinine-N-methytransferase from Vinca minor with C-terminal 6xHis tag", 60093, "Vinca minor", "Picrinine-N-methytransferase", "6xHis"),
        ("Integrase from Escherichia phage lambda", 10710, "Escherichia phage lambda", "Integrase", "None"),
    ]
    for prompt, exp_taxid, exp_org_sub, exp_pq, exp_tag in edge_extraction_cases:
        ec = extract_nl_query_components(prompt)
        assert ec["organism_taxid"] == exp_taxid, f"Expected organism_taxid={exp_taxid} on {prompt!r}, got {ec}"
        assert ec["organism_name"] and exp_org_sub.lower() in ec["organism_name"].lower(), f"Expected org {exp_org_sub!r} on {prompt!r}, got {ec}"
        assert ec["protein_query"] == exp_pq, f"Expected protein_query={exp_pq!r} on {prompt!r}, got {ec['protein_query']!r}"
        assert ec["tag_detected"] == exp_tag, f"Expected tag={exp_tag!r} on {prompt!r}, got {ec['tag_detected']!r}"

    # Serotype / viral-subtype ranking over default reference strain
    m18_target = {
        "accession": "P68885",
        "recommended_name": "Streptopain",
        "gene_primary": "speB",
        "organism_scientific": "Streptococcus pyogenes serotype M18 (strain MGAS8232)",
        "tax_id": 186103,
        "annotation_score": 5.0,
        "is_fragment": False,
        "reviewed": True,
    }
    m1_decoy = dict(m18_target, accession="P0C0J0", organism_scientific="Streptococcus pyogenes serotype M1", tax_id=160490)
    ranked_sero = rank_uniprot_candidates([m1_decoy, m18_target], "Streptopain Streptococcus pyogenes serotype M18")
    assert ranked_sero[0]["accession"] == "P68885", f"Expected serotype M18 P68885 first, got {ranked_sero[0]['accession']}"

    # Parenthesized Drosophila gene 'fs(1)K10' + repeated-token '-binding protein' decoy trap
    k10_comp = extract_nl_query_components("Drosophila melanogaster fs(1)K10 (DNA-binding protein K10)")
    assert k10_comp["gene_hint"] == "fs(1)K10" and k10_comp["protein_name_hint"] == "DNA-binding protein K10", f"Bad split on fs(1)K10: {k10_comp}"
    k10_target = {
        "accession": "P13468",
        "recommended_name": "DNA-binding protein K10",
        "gene_primary": "fs(1)K10",
        "organism_scientific": "Drosophila melanogaster",
        "tax_id": 7227,
        "annotation_score": 5.0,
        "is_fragment": False,
        "reviewed": True,
    }
    k10_decoy = dict(k10_target, accession="D_BIND_P13468", recommended_name="DNA-binding protein K10-binding protein", gene_primary="fs(1)K10BP")
    ranked_k10 = rank_uniprot_candidates([k10_decoy, k10_target], k10_comp)
    assert ranked_k10[0]["accession"] == "P13468", f"Expected P13468 over repeated-token binding decoy, got {ranked_k10[0]['accession']}"

    # 6. Full-length reference protein outranking legacy species-node Fragment ('Plasmodium falciparum L-lactate dehydrogenase')
    #    and non-None string gene_name on ORF-only phage entries ('P00573' T7 RNA polymerase)
    from modules.sequence_parser import _build_uniprot_search_plan, _parse_uniprot_entry_dict

    pf_full = {
        "primaryAccession": "Q27743",
        "entryType": "UniProtKB reviewed (Swiss-Prot)",
        "annotationScore": 5.0,
        "proteinDescription": {"recommendedName": {"fullName": {"value": "L-lactate dehydrogenase"}}},
        "genes": [{"geneName": {"value": "LDH"}}],
        "organism": {"scientificName": "Plasmodium falciparum (isolate 3D7)", "taxonId": 36329},
        "sequence": {"length": 316, "value": "M" * 316},
    }
    pf_frag = {
        "primaryAccession": "P13774",
        "entryType": "UniProtKB reviewed (Swiss-Prot)",
        "annotationScore": 2.0,
        "proteinDescription": {"recommendedName": {"fullName": {"value": "L-lactate dehydrogenase"}}, "flag": "Fragment"},
        "genes": [],
        "organism": {"scientificName": "Plasmodium falciparum", "taxonId": 5833},
        "sequence": {"length": 42, "value": "M" * 42, "fragment": "single"},
    }
    for pf_prompt in [
        "Plasmodium falciparum L-lactate dehydrogenase",
        "Plasmodium falciparum L-lactate dehydrogenase with N-terminal 6xHis tag and TEV site",
    ]:
        pf_comp = extract_nl_query_components(pf_prompt)
        pf_queries, _, _ = _build_uniprot_search_plan(pf_comp)
        assert "NOT (fragment:true)" in pf_queries[0], f"Expected non-fragment priority in top search tier: {pf_queries[0]}"
        assert any("NOT (fragment:true)" not in q and "(reviewed:true)" in q for q in pf_queries), "Expected fragment-inclusive reviewed fallback tier"
        ranked_pf = rank_uniprot_candidates([pf_frag, pf_full], pf_comp)
        assert ranked_pf[0]["primaryAccession"] == "Q27743", f"Fragment P13774 outranked full-length Q27743: {ranked_pf[0]}"

    t7_orf_only = {
        "primaryAccession": "P00573",
        "proteinDescription": {"recommendedName": {"fullName": {"value": "DNA-directed RNA polymerase"}}},
        "organism": {"scientificName": "Enterobacteria phage T7"},
        "genes": [{"orfNames": [{"value": "1"}]}],
        "sequence": {"length": 883, "value": "M" * 883},
    }
    parsed_t7 = _parse_uniprot_entry_dict(t7_orf_only)
    assert isinstance(parsed_t7["gene_name"], str) and parsed_t7["gene_name"] == "1", f"Expected ORF fallback '1', got {parsed_t7['gene_name']!r}"
    parsed_empty_gene = _parse_uniprot_entry_dict(dict(t7_orf_only, genes=[{}]))
    assert isinstance(parsed_empty_gene["gene_name"], str) and parsed_empty_gene["gene_name"] == "N/A", f"Expected 'N/A', got {parsed_empty_gene['gene_name']!r}"

    # 7. Colloquial periodless organism binomials, <ALPHA><NUM> <-> <ALPHA>-<NUM> dual-form search/scoring,
    #    and canonical biochemical shorthand resolution ("E coli IF3", "E coli EFTu", "B subtilis GroEL", "S cerevisiae eIF4E", "H sapiens IL2")
    colloquial_cases = [
        ("E coli IF3", 83333, "Escherichia coli", "IF3", "None", "None", ["P0A707"], ["INFC"]),
        ("E. coli IF3", 83333, "Escherichia coli", "IF3", "None", "None", ["P0A707"], ["INFC"]),
        ("IF3 E coli", 83333, "Escherichia coli", "IF3", "None", "None", ["P0A707"], ["INFC"]),
        ("E coli K12 IF3", 83333, "Escherichia coli", "IF3", "None", "None", ["P0A707"], ["INFC"]),
        ("E coli BL21 IF3", 83333, "Escherichia coli", "IF3", "None", "None", ["P0A707"], ["INFC"]),
        ("E coli IF3 with N-terminal 6xHis tag", 83333, "Escherichia coli", "IF3", "6xHis", "None", ["P0A707"], ["INFC"]),
        ("E coli IF3 with N-terminal 6xHis tag and TEV site", 83333, "Escherichia coli", "IF3", "6xHis", "TEV Protease", ["P0A707"], ["INFC"]),
        ("E coli IF1", 83333, "Escherichia coli", "IF1", "None", "None", ["P69222"], ["INFA"]),
        ("E coli IF2", 83333, "Escherichia coli", "IF2", "None", "None", ["P0A705"], ["INFB"]),
        ("E coli EFTu", 83333, "Escherichia coli", "EFTu", "None", "None", ["P0CE47"], ["TUFA"]),
        ("E coli EFG", 83333, "Escherichia coli", "EFG", "None", "None", ["P0A6M8"], ["FUSA"]),
        ("E coli RF1", 83333, "Escherichia coli", "RF1", "None", "None", ["P0A7I0"], ["PRFA"]),
        ("E coli RRF", 83333, "Escherichia coli", "RRF", "None", "None", ["P0A805"], ["FRR"]),
        ("B subtilis GroEL", 224308, "Bacillus subtilis", "GroEL", "None", "None", [], ["GROL", "GROEL"]),
        ("S cerevisiae eIF4E", 559292, "Saccharomyces cerevisiae", "eIF4E", "None", "None", [], []),
        ("H sapiens IL2", 9606, "Homo sapiens", "IL2", "None", "None", [], []),
        ("A thaliana Rubisco", 3702, "Arabidopsis thaliana", "Rubisco", "None", "None", ["O03042"], ["RBCL"]),
        ("S aureus Protein A", 93061, "Staphylococcus aureus", "Protein A", "None", "None", ["P02976"], ["SPA"]),
        ("D melanogaster Notch", 7227, "Drosophila melanogaster", "Notch", "None", "None", ["P07207"], ["N"]),
    ]
    for prompt, exp_tax, exp_org_sub, exp_pq, exp_tag, exp_clv, req_accs, req_genes in colloquial_cases:
        cc = extract_nl_query_components(prompt)
        assert cc["organism_taxid"] == exp_tax, f"Expected taxid {exp_tax} on {prompt!r}, got {cc}"
        assert cc["organism_name"] and exp_org_sub.lower() in cc["organism_name"].lower(), f"Bad org on {prompt!r}: {cc}"
        assert cc["protein_query"] == exp_pq, f"Expected protein_query={exp_pq!r} on {prompt!r}, got {cc['protein_query']!r}"
        assert cc["tag_detected"] == exp_tag and cc["cleavage_detected"] == exp_clv, f"Bad modifiers on {prompt!r}: {cc}"
        for ra in req_accs:
            assert ra in cc["preferred_accessions"], f"Missing {ra} in preferred_accessions for {prompt!r}: {cc}"
        for rg in req_genes:
            assert rg in cc["preferred_genes"], f"Missing {rg} in preferred_genes for {prompt!r}: {cc}"

    # Guard against periodless false-positive organism collisions on 'G max' / 'L major' / 'V minor'
    for non_org_phrase in ["Protein G MAX", "Type IV major pilin", "Type V minor pilin"]:
        nc = extract_nl_query_components(non_org_phrase)
        assert nc["organism_taxid"] is None and nc["organism_name"] is None, f"False periodless org on {non_org_phrase!r}: {nc}"

    # Verify dual-form <ALPHA><NUM> <-> <ALPHA>-<NUM> search plan generation across arbitrary proteins
    for q_dual, exp_fused, exp_hyph in [
        ("H sapiens IL2", '"IL2"', '"IL-2"'),
        ("human Bcl2", '"Bcl2"', '"Bcl-2"'),
        ("human FGF2", '"FGF2"', '"FGF-2"'),
        ("human CDK2", '"CDK2"', '"CDK-2"'),
        ("S cerevisiae eIF4E", '"eIF4E"', '"eIF-4E"'),
        ("B subtilis IF3", '"IF-3"', "infC"),
    ]:
        dq, _, _ = _build_uniprot_search_plan(extract_nl_query_components(q_dual))
        joined_dq = " || ".join(dq)
        assert exp_fused in joined_dq and exp_hyph in joined_dq, f"Missing dual-form variants ({exp_fused}, {exp_hyph}) in {joined_dq}"

    # Verify de-hyphenated shortName & token fusion scoring on P0A707 (IF-3), P07260 (eIF-4E), and P60568 (IL-2)
    ec_if3 = {
        "primaryAccession": "P0A707",
        "entryType": "UniProtKB reviewed (Swiss-Prot)",
        "annotationScore": 5.0,
        "proteinDescription": {
            "recommendedName": {
                "fullName": {"value": "Translation initiation factor IF-3"},
                "shortNames": [{"value": "IF-3"}],
            }
        },
        "genes": [{"geneName": {"value": "infC"}}],
        "organism": {"scientificName": "Escherichia coli (strain K12)", "taxonId": 83333},
        "sequence": {"length": 180, "value": "M" * 180},
    }
    ec_decoy = {
        "primaryAccession": "P77777",
        "entryType": "UniProtKB reviewed (Swiss-Prot)",
        "annotationScore": 5.0,
        "proteinDescription": {"recommendedName": {"fullName": {"value": "Putative factor YjgF"}}},
        "genes": [{"geneName": {"value": "yjgF"}}],
        "organism": {"scientificName": "Escherichia coli (strain K12)", "taxonId": 83333},
        "sequence": {"length": 128, "value": "M" * 128},
    }
    sc_eif4e = {
        "primaryAccession": "P07260",
        "entryType": "UniProtKB reviewed (Swiss-Prot)",
        "annotationScore": 5.0,
        "proteinDescription": {
            "recommendedName": {
                "fullName": {"value": "Eukaryotic translation initiation factor 4E"},
                "shortNames": [{"value": "eIF-4E"}],
            }
        },
        "genes": [{"geneName": {"value": "CDC33"}}],
        "organism": {"scientificName": "Saccharomyces cerevisiae (strain ATCC 204508 / S288c)", "taxonId": 559292},
        "sequence": {"length": 213, "value": "M" * 213},
    }
    sc_eif4e_bind = dict(
        sc_eif4e,
        primaryAccession="P99998",
        proteinDescription={"recommendedName": {"fullName": {"value": "Eukaryotic translation initiation factor 4E-binding protein"}}},
        genes=[{"geneName": {"value": "EAP1"}}],
    )
    assert rank_uniprot_candidates([ec_decoy, ec_if3], "E coli IF3")[0]["primaryAccession"] == "P0A707"
    assert rank_uniprot_candidates([sc_eif4e_bind, sc_eif4e], "S cerevisiae eIF4E")[0]["primaryAccession"] == "P07260"

    # End-to-end parse_natural_language_prompt checks on 'E coli IF3' variants (works both online & offline)
    for q_if3, exp_len, exp_tag, exp_clv in [
        ("E coli IF3", 180, "None", "None"),
        ("E. coli IF3", 180, "None", "None"),
        ("IF3 E coli", 180, "None", "None"),
        ("E coli IF3 with N-terminal 6xHis tag", 187, "6xHis", "None"),
        ("E coli IF3 with N-terminal 6xHis tag and TEV site", 194, "6xHis", "TEV Protease"),
    ]:
        r_if3 = parse_natural_language_prompt(q_if3)
        assert r_if3["uniprot_id"] == "P0A707", f"Expected P0A707 on {q_if3!r}, got {r_if3}"
        assert r_if3["gene_name"] == "infC", f"Expected infC on {q_if3!r}, got {r_if3}"
        assert "initiation factor if-3" in r_if3["identified_protein"].lower()
        assert r_if3["length"] == exp_len, f"Expected length {exp_len} on {q_if3!r}, got {r_if3['length']}"
        assert r_if3["tag_detected"] == exp_tag and r_if3["cleavage_detected"] == exp_clv


def test_soluble_aggregate_void_peak_and_ui(root_dir: str) -> None:
    print("\n=== 8. Testing Soluble HMW Aggregate (Void Peak Vo) Biophysical Modeling & Tab 1 UI ===")
    import math
    from modules.sec_simulator import _get_extra_column_dispersion_ml
    from streamlit.testing.v1 import AppTest

    # 1. Verify strict Vo elution, Kav = 0.0, polydisperse broadening, and A260/A280 light-scattering
    #    elevation (~0.92 vs 0.57 monomer) across ALL 8 SEC_COLUMNS (including 40 MDa Superose 6!)
    assert len(SEC_COLUMNS) == 8
    for cid, col_cfg in SEC_COLUMNS.items():
        vo_expected = float(col_cfg["void_volume_vo_ml"])
        excl_expected = float(col_cfg["exclusion_limit_kda"])
        states = [
            {"label": "Monomer (14.3 kDa)", "mw_kda": 14.31, "prot_eps01": 2.618, "abundance": 1.0, "color": "#0F766E"},
            {
                "label": "Aggregate (Vo)",
                "mw_kda": 14.31 * 20.0,
                "is_aggregate": True,
                "prot_eps01": 2.618,
                "abundance": 0.35,
                "color": "#B91C1C",
            },
        ]
        sim = simulate_chromatogram_peaks(states, col_cfg)
        pk_mono, pk_agg = sim["peaks"][0], sim["peaks"][1]

        assert pk_mono["is_aggregate"] is False
        assert pk_agg["is_aggregate"] is True
        assert pk_agg["ve_ml"] == vo_expected, f"Aggregate Ve {pk_agg['ve_ml']} != Vo {vo_expected} on {cid}"
        assert pk_agg["kav"] == 0.0, f"Aggregate Kav {pk_agg['kav']} != 0.0 on {cid}"
        assert pk_agg["exclusion_limit_kda"] == excl_expected
        assert pk_agg["mw_kda"] >= excl_expected * 1.25
        assert pk_mono["a260_a280_ratio"] == 0.57
        assert abs(pk_agg["a260_a280_ratio"] - 0.92) <= 0.01, f"Expected ~0.92 light-scattering ratio, got {pk_agg['a260_a280_ratio']}"
        assert pk_agg["a260_a280_ratio"] > pk_mono["a260_a280_ratio"]
        assert abs(pk_agg["peak_height_a260_mau"] - round(pk_agg["peak_height_mau"] * pk_agg["a260_a280_ratio"], 1)) <= 0.2
        assert "Soluble HMW aggregate" in pk_agg["note"] and "Kav = 0.000" in pk_agg["note"]

        # Verify 1.18x polydispersity broadening relative to an unbroadened Vo Gaussian
        dec = 3 if col_cfg["bed_volume_vt_ml"] < 5.0 else 2
        mono_vo_sigma = math.sqrt((vo_expected / math.sqrt(col_cfg["plate_count_n"])) ** 2 + _get_extra_column_dispersion_ml(col_cfg) ** 2)
        agg_vo_sigma = math.sqrt(((vo_expected / math.sqrt(col_cfg["plate_count_n"])) * 1.18) ** 2 + _get_extra_column_dispersion_ml(col_cfg) ** 2)
        assert pk_agg["sigma_ml"] == round(agg_vo_sigma, dec)
        assert agg_vo_sigma > mono_vo_sigma

    # 2. Headless Streamlit AppTest verification of Tab 1 Compare Oligomer Peaks
    app_path = os.path.join(root_dir, "app.py")
    at = AppTest.from_file(app_path).run(timeout=30)
    assert len(at.exception) == 0

    def _chk_by_label(lbl: str):
        matches = [c for c in at.checkbox if c.label == lbl]
        assert len(matches) == 1, f"Expected 1 checkbox labeled {lbl!r}, found {len(matches)}"
        return matches[0]

    # Default clean view: 1x Mono ON, 2x..4x and Aggregate OFF
    assert _chk_by_label("1× Mono").value is True
    assert _chk_by_label("2× Dimer").value is False
    assert _chk_by_label("3× Trimer").value is False
    assert _chk_by_label("4× Tetra").value is False
    assert _chk_by_label("Aggregate").value is False
    assert len(at.dataframe[0].value) == 1 and len(at.dataframe[1].value) == 1
    assert not any("Void Aggregate Peak" in inf.value for inf in at.info)

    # Check ALL 5 peaks ON (1x Mono + 2x Dimer + 3x Trimer + 4x Tetra + Aggregate)
    _chk_by_label("2× Dimer").set_value(True)
    _chk_by_label("3× Trimer").set_value(True)
    _chk_by_label("4× Tetra").set_value(True)
    _chk_by_label("Aggregate").set_value(True).run(timeout=30)
    assert len(at.exception) == 0, f"AppTest exception with all 5 peaks checked: {at.exception}"

    df_prep_s75 = at.dataframe[0].value
    df_anal_s75 = at.dataframe[1].value
    assert len(df_prep_s75) == 5 and len(df_anal_s75) == 5
    assert df_prep_s75.iloc[0]["Peak"].startswith("Monomer")
    agg_row_s75 = df_prep_s75[df_prep_s75["Peak"] == "Aggregate (Vo)"].iloc[0]
    assert agg_row_s75["Size"] == "> 100 kDa (HMW)"
    assert agg_row_s75["3D Radius (Rh)"] == "Excluded (Vo)"
    assert agg_row_s75["Elutes At (Ve)"] == "40.00 mL"
    assert agg_row_s75["A260/A280"] == "0.92"
    assert any("Void Aggregate Peak" in inf.value and "SDS-PAGE" in inf.value for inf in at.info)

    # Switch Prep -> S200 (1.3 MDa) and Analytical -> Superose 6 (40 MDa) with all 5 peaks checked
    prep_sb = [s for s in at.selectbox if "Large Prep Column" in s.label][0]
    anal_sb = [s for s in at.selectbox if "Benchtop Analytical Column" in s.label][0]
    prep_sb.set_value("hiload_16_600_s200")
    anal_sb.set_value("superose_6_increase_10_300").run(timeout=30)
    assert len(at.exception) == 0

    agg_row_s200 = at.dataframe[0].value[at.dataframe[0].value["Peak"] == "Aggregate (Vo)"].iloc[0]
    agg_row_sup6 = at.dataframe[1].value[at.dataframe[1].value["Peak"] == "Aggregate (Vo)"].iloc[0]
    assert agg_row_s200["Size"] == "> 1.3 MDa (HMW)" and agg_row_s200["Elutes At (Ve)"] == "40.50 mL"
    assert agg_row_sup6["Size"] == "> 40 MDa (HMW)" and agg_row_sup6["Elutes At (Ve)"] == "8.10 mL"

    # Verify distinct colors between 4x Tetra and Aggregate (Vo) & hovertext on the rendered Plotly SEC chart
    sec_fig_spec = json.loads(at.get("plotly_chart")[0].proto.spec)
    tetra_traces = [tr for tr in sec_fig_spec["data"] if tr.get("name", "").startswith("Tetramer")]
    agg_traces = [tr for tr in sec_fig_spec["data"] if tr.get("name") == "Aggregate (Vo)"]
    assert len(tetra_traces) == 1 and len(agg_traces) == 1
    assert tetra_traces[0]["marker"]["color"] != agg_traces[0]["marker"]["color"]
    assert "> 1.3 MDa" in agg_traces[0]["hovertext"] and "UV Light Scattering" in agg_traces[0]["hovertext"]
    peak_annots = [a for a in sec_fig_spec["layout"]["annotations"] if a.get("showarrow")]
    assert len(peak_annots) == 5

    # Also verify 320 mL HiLoad 26/600 & 2.4 mL Superose 6 3.2/300 scales with all 5 peaks checked
    prep_sb.set_value("hiload_26_600_s200")
    anal_sb.set_value("superose_6_increase_3_2_300").run(timeout=30)
    assert len(at.exception) == 0
    agg_row_26_600 = at.dataframe[0].value[at.dataframe[0].value["Peak"] == "Aggregate (Vo)"].iloc[0]
    agg_row_3_2 = at.dataframe[1].value[at.dataframe[1].value["Peak"] == "Aggregate (Vo)"].iloc[0]
    assert agg_row_26_600["Size"] == "> 1.3 MDa (HMW)" and agg_row_26_600["Elutes At (Ve)"] == "108.00 mL"
    assert agg_row_3_2["Size"] == "> 40 MDa (HMW)" and agg_row_3_2["Elutes At (Ve)"] == "0.81 mL"

    # Switch back to S200 / Superose 6 10/300 and verify ONLY Aggregate checked plots ONLY Aggregate (Vo) cleanly
    prep_sb.set_value("hiload_16_600_s200")
    anal_sb.set_value("superose_6_increase_10_300")

    # Verify ONLY Aggregate checked (with 1x Mono..4x Tetra ALL unchecked) plots ONLY Aggregate (Vo) cleanly
    _chk_by_label("1× Mono").set_value(False)
    _chk_by_label("2× Dimer").set_value(False)
    _chk_by_label("3× Trimer").set_value(False)
    _chk_by_label("4× Tetra").set_value(False)
    _chk_by_label("Aggregate").set_value(True).run(timeout=30)
    assert len(at.exception) == 0, f"AppTest exception with Aggregate alone: {at.exception}"

    df_only_agg_prep = at.dataframe[0].value
    df_only_agg_anal = at.dataframe[1].value
    assert len(df_only_agg_prep) == 1 and df_only_agg_prep.iloc[0]["Peak"] == "Aggregate (Vo)"
    assert len(df_only_agg_anal) == 1 and df_only_agg_anal.iloc[0]["Peak"] == "Aggregate (Vo)"
    md_joined = "\n".join(m.value for m in at.markdown)
    assert "🎯 Aggregate (Vo, > 1.3 MDa)" in md_joined
    assert "🎯 Aggregate (Vo, > 40 MDa)" in md_joined


def test_secondary_structure_engine_benchmarks_and_ui(root_dir: str) -> None:
    print("\n=== 9. Testing Secondary Structure Engine, ≥10,000-Protein & Web-Server Benchmarks, and Tab 4/5 UI ===")
    import time
    import numpy as np
    from streamlit.testing.v1 import AppTest

    # 1. Warm up cached NumPy weights, then verify 100% determinism & fast runtime (< 0.15s)
    lyso_seq = CLASSROOM_PRESET_PROTEINS["lysozyme_native"]["sequence"]
    res_warm = predict_secondary_structure(lyso_seq)

    t0 = time.perf_counter()
    res_det_1 = predict_secondary_structure(lyso_seq)
    dt_lyso = time.perf_counter() - t0
    res_det_2 = predict_secondary_structure(lyso_seq)
    assert res_det_1 == res_warm == res_det_2, "Non-deterministic output across repeated predict_secondary_structure calls!"
    assert dt_lyso < 0.15, f"Secondary structure runtime {dt_lyso:.4f}s exceeded 0.15s budget on Lysozyme!"

    # Also verify runtime < 0.15s on large 583-aa BSA
    bsa_seq = CLASSROOM_PRESET_PROTEINS["bsa"]["sequence"]
    t_bsa0 = time.perf_counter()
    res_bsa = predict_secondary_structure(bsa_seq)
    dt_bsa = time.perf_counter() - t_bsa0
    assert dt_bsa < 0.15, f"Secondary structure runtime {dt_bsa:.4f}s exceeded 0.15s budget on 583-aa BSA!"

    # 2. Structural & mathematical invariants across all classroom presets
    for preset_key, preset_obj in CLASSROOM_PRESET_PROTEINS.items():
        seq = preset_obj["sequence"]
        ss = predict_secondary_structure(seq)
        L = len(seq)
        assert ss["sequence"] == seq and ss["length"] == L
        assert len(ss["ss_string"]) == L and set(ss["ss_string"]).issubset({"H", "E", "C"})
        assert len(ss["prob_h"]) == L and len(ss["prob_e"]) == L and len(ss["prob_c"]) == L
        for ph, pe, pc in zip(ss["prob_h"], ss["prob_e"], ss["prob_c"]):
            assert 0.0 <= ph <= 1.0 and 0.0 <= pe <= 1.0 and 0.0 <= pc <= 1.0
            assert abs((ph + pe + pc) - 1.0) <= 0.01
        assert abs((ss["helix_pct"] + ss["strand_pct"] + ss["coil_pct"]) - 100.0) <= 0.05

        # Contiguous 1-based inclusive segments covering 1..L + physical min lengths (H >= 4, E >= 2)
        segs = ss["segments"]
        assert len(segs) >= 1 and segs[0]["start"] == 1 and segs[-1]["end"] == L
        cursor = 1
        for seg in segs:
            assert seg["start"] == cursor
            assert seg["end"] - seg["start"] + 1 == seg["length"]
            assert seg["sequence"] == seq[seg["start"] - 1 : seg["end"]]
            if seg["type"] == "H":
                assert seg["length"] >= 4, f"Helix shorter than 4 aa on {preset_key}: {seg}"
                assert seg["label"].startswith("α")
            elif seg["type"] == "E":
                assert seg["length"] >= 2, f"Strand shorter than 2 aa on {preset_key}: {seg}"
                assert seg["label"].startswith("β")
            cursor = seg["end"] + 1
        assert cursor == L + 1

    # 3. Known textbook protein fold checks (both mature classroom presets & full UniProt corpus entries)
    myo_res = predict_secondary_structure(CLASSROOM_PRESET_PROTEINS["myoglobin"]["sequence"])
    assert "All-α" in myo_res["fold_class"], f"Expected All-α for Myoglobin P02185, got {myo_res}"
    assert myo_res["helix_pct"] >= 65.0 and myo_res["strand_pct"] <= 5.0, f"Unexpected Myoglobin composition: {myo_res}"

    gfp_res = predict_secondary_structure(CLASSROOM_PRESET_PROTEINS["gfp"]["sequence"])
    assert "All-β" in gfp_res["fold_class"], f"Expected All-β for GFP P42212, got {gfp_res}"
    assert gfp_res["strand_pct"] >= 40.0, f"Expected GFP strand_pct >= 40%, got {gfp_res}"

    lyso_res = res_det_1
    assert "Mixed α/β" in lyso_res["fold_class"], f"Expected Mixed α/β for Lysozyme P00698, got {lyso_res}"
    assert lyso_res["helix_count"] >= 1 and lyso_res["strand_count"] >= 1, f"Expected both α & β segments in Lysozyme: {lyso_res}"

    lip_res = predict_secondary_structure(CLASSROOM_PRESET_PROTEINS["lipase_a"]["sequence"])
    assert "Mixed α/β" in lip_res["fold_class"], f"Expected Mixed α/β for Lipase A P37957, got {lip_res}"
    assert lip_res["helix_count"] >= 1 and lip_res["strand_count"] >= 1

    assert "All-α" in res_bsa["fold_class"] and res_bsa["helix_pct"] >= 60.0

    # Also verify full-length UniProt sequences directly from data/ss_10k_corpus.npz
    corpus_path = os.path.join(root_dir, "data", "ss_10k_corpus.npz")
    assert os.path.exists(corpus_path), "Missing data/ss_10k_corpus.npz"
    with np.load(corpus_path, allow_pickle=False) as c_npz:
        accs_list = [str(a) for a in c_npz["accessions"]]
        splits_arr = c_npz["splits"]
        seqs_arr = c_npz["sequences"]
        for c_acc in ("P02185", "P42212", "P00698", "P37957", "P00918", "P02769"):
            idx_c = accs_list.index(c_acc)
            assert str(splits_arr[idx_c]) == "test", f"Classroom protein {c_acc} must be in held-out test split!"
        u_myo = predict_secondary_structure(str(seqs_arr[accs_list.index("P02185")]))
        u_gfp = predict_secondary_structure(str(seqs_arr[accs_list.index("P42212")]))
        u_lys = predict_secondary_structure(str(seqs_arr[accs_list.index("P00698")]))
        u_lip = predict_secondary_structure(str(seqs_arr[accs_list.index("P37957")]))
        assert "All-α" in u_myo["fold_class"] and u_myo["helix_pct"] >= 65.0 and u_myo["strand_pct"] <= 5.0
        assert "All-β" in u_gfp["fold_class"] and u_gfp["strand_pct"] >= 40.0
        assert "Mixed α/β" in u_lys["fold_class"] and u_lys["helix_count"] >= 1 and u_lys["strand_count"] >= 1
        assert "Mixed α/β" in u_lip["fold_class"] and u_lip["helix_count"] >= 1 and u_lip["strand_count"] >= 1

    # 4. Plotly 2-track topology figure & monospace alignment block formatter
    fig_lyso = generate_secondary_structure_figure(lyso_res, "Hen Egg-White Lysozyme")
    assert len(fig_lyso.data) >= 5, f"Expected >=5 traces in 2-track SS figure, got {len(fig_lyso.data)}"
    fig_json = json.loads(fig_lyso.to_json())
    assert (
        fig_json["layout"]["xaxis"].get("matches") == "x2"
        or fig_json["layout"]["xaxis2"].get("matches") == "x"
    ), "Expected linked x-axes across topology & propensity panels"

    aligned_txt = format_ss_sequence_blocks(lyso_res["sequence"], lyso_res["ss_string"], block_width=60)
    assert "Legend:" in aligned_txt and "Pos" in aligned_txt
    aa_lines = [ln.split()[2] for ln in aligned_txt.splitlines() if ln.startswith("AA ")]
    ss_lines = [ln[10:] for ln in aligned_txt.splitlines() if ln.startswith("SS        ")]
    assert "".join(aa_lines) == lyso_res["sequence"]
    assert "".join(ss_lines) == lyso_res["ss_string"].replace("C", "·")
    assert compute_q3(lyso_res["ss_string"], lyso_res["ss_string"]) == 100.0
    assert compute_sov99(lyso_res["ss_string"], lyso_res["ss_string"]) == 100.0

    # 5. Verify data/ss_10k_benchmark_results.json (>=10,000 test proteins across all domains & 7 strata)
    ss_bench_path = os.path.join(root_dir, "data", "ss_10k_benchmark_results.json")
    assert os.path.exists(ss_bench_path), "Missing data/ss_10k_benchmark_results.json"
    with open(ss_bench_path, "r", encoding="utf-8") as sbf:
        ss_bench = json.load(sbf)
    assert int(ss_bench["test_count"]) >= 10000, f"Expected test_count >= 10000, got {ss_bench['test_count']}"
    assert float(ss_bench["Q3"]) >= 60.0 and float(ss_bench["SOV_99"]) >= 50.0
    expected_domains = {"Bacteria", "Archaea", "Eukaryota", "Viruses"}
    expected_strata = {"Bacteria", "Archaea", "Animals", "Plants", "Fungi", "Protists", "Viruses"}
    assert expected_domains.issubset(set(ss_bench["by_domain"].keys()))
    assert expected_strata.issubset(set(ss_bench["by_stratum"].keys()))
    assert sum(int(ss_bench["by_stratum"][s]["n_proteins"]) for s in expected_strata) == int(ss_bench["test_count"])

    # 6. Verify data/ss_webserver_crossref.json (n_proteins >= 60 across all 7 strata)
    ss_xref_path = os.path.join(root_dir, "data", "ss_webserver_crossref.json")
    assert os.path.exists(ss_xref_path), "Missing data/ss_webserver_crossref.json"
    with open(ss_xref_path, "r", encoding="utf-8") as sxf:
        ss_xref = json.load(sxf)
    assert int(ss_xref["n_proteins"]) >= 60, f"Expected n_proteins >= 60, got {ss_xref['n_proteins']}"
    assert len(ss_xref["proteins"]) == int(ss_xref["n_proteins"])
    xref_strata_seen = {p["stratum"] for p in ss_xref["proteins"]}
    assert expected_strata.issubset(xref_strata_seen), f"Missing strata in crossref: {expected_strata - xref_strata_seen}"
    for s_name in expected_strata:
        assert sum(1 for p in ss_xref["proteins"] if p["stratum"] == s_name) >= 8
    xref_accs = {p["accession"] for p in ss_xref["proteins"]}
    for req_preset in ("P00698", "P42212", "P00918", "P02185", "P37957", "P02769"):
        assert req_preset in xref_accs, f"Missing classroom preset {req_preset} in ss_webserver_crossref.json"
    assert ss_xref["summary"]["mean_q3_vs_exp_dssp"]["propurify_ss"] is not None

    # 7. Audit strict typography rules (ZERO ad-hoc inline HTML font-size: outside <style> & Material icon safeguard)
    app_path = os.path.join(root_dir, "app.py")
    with open(app_path, "r", encoding="utf-8") as af:
        app_src = af.read()
    style_end_idx = app_src.find("</style>")
    assert style_end_idx > 0
    assert '[data-testid="stIconMaterial"]' in app_src[:style_end_idx], "Material icon safeguard missing from <style>!"
    after_style_src = app_src[style_end_idx + len("</style>") :]
    inline_fs_hits = re.findall(r"font-size\s*:", after_style_src, flags=re.IGNORECASE)
    assert len(inline_fs_hits) == 0, f"Found forbidden inline font-size: outside <style> in app.py: {inline_fs_hits}"

    # 8. Headless Streamlit AppTest verifying Tab 4 & Tab 5 secondary structure cards, diagram, expander & presets
    at = AppTest.from_file(app_path).run(timeout=30)
    assert len(at.exception) == 0, f"AppTest raised exceptions: {at.exception}"

    tab_labels = [t.label for t in at.tabs]
    assert "🔬 4. Amino Acids, Structure & Solubility" in tab_labels

    non_style_md = "\n".join(m.value for m in at.markdown if "<style>" not in m.value)
    assert len(re.findall(r"font-size\s*:", non_style_md, flags=re.IGNORECASE)) == 0, "Inline font-size: found in rendered markdown!"
    md_all = "\n".join(m.value for m in at.markdown)
    assert "🧭 Predicted Secondary Structure & Topology Diagram" in md_all
    assert "🌀 α-Helix Content" in md_all
    assert "➔ β-Strand Content" in md_all
    assert "〰️ Coil / Loop / Turn" in md_all
    assert "🧭 Predicted Fold Class" in md_all
    assert "Mixed α/β" in md_all
    assert "≥10,500-Protein PDB/DSSP Benchmark" in md_all
    assert "NPS@ SOPMA" in md_all and "AlphaFold DB" in md_all

    # Verify segment table & aligned Pos/AA/SS code block inside Tab 4 expander
    ss_expanders = [
        ex for ex in at.expander
        if "View Residue-by-Residue Secondary Structure Alignment" in ex.label
    ]
    assert len(ss_expanders) == 1, "Expected Tab 4 Secondary Structure alignment expander!"
    seg_dfs = [
        df.value for df in at.dataframe
        if list(df.value.columns) == ["Segment", "Type", "Start–End", "Length (aa)", "Mean Confidence", "Sequence"]
    ]
    assert len(seg_dfs) == 1 and len(seg_dfs[0]) == len(lyso_res["segments"])
    assert any("Legend:  H = α-Helix" in c.value for c in at.code)

    # Switch classroom preset -> Myoglobin (All-α) and GFP (All-β) in Sidebar and verify live Tab 4 updates
    preset_sb = at.sidebar.selectbox[0]
    preset_sb.set_value("myoglobin").run(timeout=30)
    assert len(at.exception) == 0
    assert "All-α (Helix-Rich)" in "\n".join(m.value for m in at.markdown)

    preset_sb.set_value("gfp").run(timeout=30)
    assert len(at.exception) == 0
    assert "All-β (Beta-Sheet Rich)" in "\n".join(m.value for m in at.markdown)


def test_giant_multidomain_titin_and_ui(root_dir: str) -> None:
    print("\n=== 10. Testing Giant Multi-Domain Proteins (Human Titin Q8WZ42, 34,350 aa) & Domain Explorer UI ===")
    import time
    from unittest.mock import patch
    import numpy as np
    import requests as _requests
    from streamlit.testing.v1 import AppTest

    # 1. Verify "Titin human" / "human titin" / "TTN human" -> canonical Swiss-Prot Q8WZ42 (34,350 aa, Homo sapiens, untagged)
    for q_titin in ["Titin human", "human titin", "TTN human"]:
        r_tt = parse_natural_language_prompt(q_titin)
        assert r_tt["uniprot_id"] == "Q8WZ42", f"Expected Q8WZ42 for {q_titin!r}, got {r_tt}"
        assert r_tt["identified_protein"] == "Titin"
        assert "homo sapiens" in r_tt["identified_organism"].lower()
        assert r_tt["gene_name"] == "TTN"
        assert r_tt["tag_detected"] == "None" and r_tt["cleavage_detected"] == "None"
        assert r_tt["length"] == 34350

    with patch("modules.sequence_parser.requests.get", side_effect=_requests.ConnectionError("Simulated offline")):
        r_tt_off = parse_natural_language_prompt("Titin human")
        assert r_tt_off["uniprot_id"] == "Q8WZ42" and r_tt_off["length"] == 34350
        r_dys_off = parse_natural_language_prompt("Dystrophin human")
        assert r_dys_off["uniprot_id"] == "P11532" and r_dys_off["length"] == 3685

    titin_seq = r_tt_off["canonical_sequence"]
    dystrophin_seq = r_dys_off["canonical_sequence"]
    titin_metrics = analyze_all_parameters(titin_seq)
    assert abs(titin_metrics["mw_kda"] - 3816.03) <= 0.5
    assert recommend_optimal_columns(titin_metrics["mw_kda"]) == (
        "hiload_16_600_superose_6",
        "superose_6_increase_10_300",
    )

    # 2. Warm up SS engine & Plotly validators, then verify Titin (34,350 aa), Dystrophin (3,685 aa), and synthetic 34,350-aa chain
    warm_ss = predict_secondary_structure(LYSOZYME_MATURE_SEQ)
    generate_secondary_structure_figure(warm_ss, "Warmup")

    t_titin0 = time.perf_counter()
    titin_ss = predict_secondary_structure(titin_seq)
    dt_titin = time.perf_counter() - t_titin0
    print(
        f"Human Titin Q8WZ42 (34,350 aa): H={titin_ss['helix_pct']:.2f}% ({titin_ss['helix_count']}), "
        f"E={titin_ss['strand_pct']:.2f}% ({titin_ss['strand_count']}), "
        f"C={titin_ss['coil_pct']:.2f}% ({titin_ss['coil_count']}) | "
        f"{titin_ss['fold_class']} | runtime={dt_titin:.3f}s"
    )
    assert dt_titin < 1.0, f"Titin 34,350-aa SS runtime {dt_titin:.3f}s exceeded 1.0s budget!"
    assert 35.0 <= titin_ss["strand_pct"] <= 55.0 and titin_ss["strand_count"] >= 1500
    assert 2.0 <= titin_ss["helix_pct"] <= 8.0 and titin_ss["helix_count"] >= 80
    assert 40.0 <= titin_ss["coil_pct"] <= 60.0
    assert "All-β" in titin_ss["fold_class"]

    # Verify sub-domain biophysical fidelity inside Q8WZ42: PEVK spring (~Coil) vs C-terminal Titin Kinase (Mixed alpha/beta)
    pevk_slice = titin_ss["ss_string"][10500:11500]
    kinase_slice = titin_ss["ss_string"][32150:32500]
    assert pevk_slice.count("C") / len(pevk_slice) >= 0.90, "Expected PEVK elastic region to predict >=90% Coil!"
    assert kinase_slice.count("H") / len(kinase_slice) >= 0.20 and kinase_slice.count("E") / len(kinase_slice) >= 0.15

    # Determinism check on 34,350-aa Titin
    titin_ss_repeat = predict_secondary_structure(titin_seq)
    assert titin_ss["ss_string"] == titin_ss_repeat["ss_string"]
    assert titin_ss["prob_h"] == titin_ss_repeat["prob_h"]

    # Human Dystrophin (P11532, 3,685 aa): spectrin-repeat coiled-coil rod -> All-alpha (>60% H)
    t_dys0 = time.perf_counter()
    dys_ss = predict_secondary_structure(dystrophin_seq)
    dt_dys = time.perf_counter() - t_dys0
    assert dt_dys < 0.5, f"Dystrophin 3,685-aa runtime {dt_dys:.3f}s exceeded 0.5s!"
    assert "All-α" in dys_ss["fold_class"] and dys_ss["helix_pct"] >= 60.0

    # Synthetic 34,350-aa multi-domain chimera (GFP beta-barrels + Myoglobin alpha-bundles + flexible linkers)
    synth_unit = (
        CLASSROOM_PRESET_PROTEINS["gfp"]["sequence"]
        + "GGGGSGGGGS"
        + CLASSROOM_PRESET_PROTEINS["myoglobin"]["sequence"]
        + "PQPQPQPEVKPEVK"
    )
    synth_34k = (synth_unit * ((34350 // len(synth_unit)) + 1))[:34350]
    assert len(synth_34k) == 34350
    t_syn0 = time.perf_counter()
    synth_ss = predict_secondary_structure(synth_34k)
    dt_syn = time.perf_counter() - t_syn0
    assert dt_syn < 1.0, f"Synthetic 34,350-aa runtime {dt_syn:.3f}s exceeded 1.0s!"
    assert synth_ss["helix_pct"] >= 20.0 and synth_ss["strand_pct"] >= 15.0

    # Confirm zero regression on L <= 1,500 held-out test split in data/ss_10k_corpus.npz
    with open(os.path.join(root_dir, "data", "ss_10k_benchmark_results.json"), "r", encoding="utf-8") as bf:
        bench_meta = json.load(bf)
    with np.load(os.path.join(root_dir, "data", "ss_10k_corpus.npz"), allow_pickle=False) as c_npz:
        test_idx = np.flatnonzero(c_npz["splits"] == "test")
        rng = np.random.default_rng(2026)
        sample_idx = rng.choice(test_idx, size=100, replace=False)
        for s_i in sample_idx:
            seq_s = str(c_npz["sequences"][s_i])
            assert len(seq_s) <= 1500
            pred_s = predict_secondary_structure(seq_s)
            assert pred_s["length"] == len(seq_s)
    assert round(float(bench_meta["models"]["propurify_ss"]["mean_protein_q3_pct"]), 2) == 69.50

    # 3. Verify high-performance figure generation (< 0.3s, compact JSON < 450 KB) on 34,350-aa Titin
    t_fig0 = time.perf_counter()
    fig_titin_macro = generate_secondary_structure_figure(titin_ss, "Titin (Homo sapiens)")
    dt_fig_macro = time.perf_counter() - t_fig0
    macro_json = fig_titin_macro.to_json()
    print(f"Titin 34,350-aa Macro Figure: build={dt_fig_macro*1000:.1f}ms, JSON={len(macro_json)/1024:.1f} KB")
    assert dt_fig_macro < 0.50, f"Titin macro figure took {dt_fig_macro:.3f}s (expected < 0.50s)!"
    assert len(macro_json) < 450 * 1024, f"Titin macro figure JSON too large: {len(macro_json)/1024:.1f} KB!"
    assert len(fig_titin_macro.layout.shapes) == 0, "Macro figure must use batched polygon traces instead of add_shape!"

    fig_titin_win = generate_secondary_structure_figure(titin_ss, "Titin (Homo sapiens)", region_start=32101, region_end=32700)
    assert len(fig_titin_win.layout.shapes) >= 1, "600-aa domain window must render high-resolution helix cylinder shapes!"

    # 4. Headless Streamlit AppTest loading 34,350-aa Human Titin via Plain English ("Titin human")
    app_path = os.path.join(root_dir, "app.py")
    at = AppTest.from_file(app_path).run(timeout=30)
    at.sidebar.radio[0].set_value("💬 Describe in Plain English").run(timeout=30)
    at.sidebar.text_area[0].set_value("Titin human").run(timeout=30)
    at.sidebar.button[0].click().run(timeout=30)
    assert len(at.exception) == 0, f"AppTest raised exceptions on 34,350-aa 'Titin human': {at.exception}"
    assert at.session_state["sequence_metadata"]["uniprot_id"] == "Q8WZ42"
    assert len(at.session_state["active_sequence"]) == 34350

    non_style_md = "\n".join(m.value for m in at.markdown if "<style>" not in m.value)
    assert len(re.findall(r"font-size\s*:", non_style_md, flags=re.IGNORECASE)) == 0, "Inline font-size: found on Titin UI!"
    md_titin = "\n".join(m.value for m in at.markdown)
    assert "34350 aa" in md_titin or "34,350 aa" in md_titin
    assert "All-β (Beta-Sheet Rich)" in md_titin
    assert "Giant Multi-Domain Protein Window Explorer" in md_titin

    # Move 600-aa domain window slider to C-terminal Titin Kinase (32,101) and also test Full-Chain Macro Overview
    win_sliders = [sl for sl in at.slider if "Start Residue" in sl.label]
    assert len(win_sliders) == 1
    win_sliders[0].set_value(32101).run(timeout=30)
    assert len(at.exception) == 0
    assert any("32101" in c.value for c in at.code), "Expected paginated alignment code block to display window 32101!"

    view_radios = [rd for rd in at.radio if "Topology Diagram View Mode" in rd.label]
    assert len(view_radios) == 1
    view_radios[0].set_value("🌐 Full-Chain Macro Overview").run(timeout=30)
    assert len(at.exception) == 0
    non_style_md_macro = "\n".join(m.value for m in at.markdown if "<style>" not in m.value)
    assert len(re.findall(r"font-size\s*:", non_style_md_macro, flags=re.IGNORECASE)) == 0


def test_interactive_sec_column_visualizer_and_intuitive_plots(root_dir: str) -> None:
    print("\n=== 11. Testing Interactive Inside-the-SEC-Column Studio, Pore Entry % & Zone Chromatograms ===")
    import xml.etree.ElementTree as ET
    from modules.sec_simulator import compute_column_snapshot
    from streamlit.testing.v1 import AppTest

    # 1. Verify pore_access_pct monotonicity & compute_column_snapshot physics across all 8 SEC columns
    for cid, col_cfg in SEC_COLUMNS.items():
        vo = float(col_cfg["void_volume_vo_ml"])
        vt = float(col_cfg["bed_volume_vt_ml"])
        states = [
            {"label": "Aggregate (Vo)", "mw_kda": 300.0, "is_aggregate": True, "abundance": 0.35, "color": "#B91C1C"},
            {"label": "Tetramer (57.2 kDa)", "mw_kda": 57.24, "abundance": 0.25, "color": "#EA580C"},
            {"label": "Dimer (28.6 kDa)", "mw_kda": 28.62, "abundance": 0.45, "color": "#D97706"},
            {"label": "Monomer (14.3 kDa)", "mw_kda": 14.31, "abundance": 1.0, "color": "#0F766E"},
        ]
        sim = simulate_chromatogram_peaks(states, col_cfg)
        pks = sim["peaks"]
        assert pks[0]["pore_access_pct"] == 0.0, f"Aggregate pore_access_pct must be 0.0 on {cid}"
        # Larger oligomer -> smaller or equal pore_access_pct and smaller or equal Ve
        for idx_p in range(len(pks) - 1):
            assert pks[idx_p]["pore_access_pct"] <= pks[idx_p + 1]["pore_access_pct"] + 1e-6
            assert pks[idx_p]["ve_ml"] <= pks[idx_p + 1]["ve_ml"] + 1e-6

        # Snapshot at 0 mL (all at_inlet), Vo (Aggregate at_detector), Monomer Ve (Aggregate eluted, Monomer at_detector), Vt (all eluted)
        snap_0 = compute_column_snapshot(sim, col_cfg, 0.0)
        assert all(t["status"] == "at_inlet" and t["z_col"] == 0.0 for t in snap_0["travellers"])
        assert all(f["fill_frac"] == 0.0 for f in snap_0["fractions"])

        snap_vo = compute_column_snapshot(sim, col_cfg, vo)
        tr_agg_vo = [t for t in snap_vo["travellers"] if t["is_aggregate"]][0]
        tr_mono_vo = [t for t in snap_vo["travellers"] if t["label"].startswith("Monomer")][0]
        assert tr_agg_vo["status"] == "at_detector" and tr_agg_vo["pct_down"] == 100
        assert tr_mono_vo["status"] == "in_column" and 25 <= tr_mono_vo["pct_down"] <= 85

        mono_ve = pks[-1]["ve_ml"]
        snap_ve = compute_column_snapshot(sim, col_cfg, mono_ve)
        tr_agg_ve = [t for t in snap_ve["travellers"] if t["is_aggregate"]][0]
        tr_mono_ve = [t for t in snap_ve["travellers"] if t["label"].startswith("Monomer")][0]
        assert tr_agg_ve["status"] == "eluted"
        assert tr_mono_ve["status"] == "at_detector" and snap_ve["detector_active_peak"] is not None

        snap_vt = compute_column_snapshot(sim, col_cfg, vt * 1.03)
        assert all(t["status"] == "eluted" for t in snap_vt["travellers"])
        assert all(f["fill_frac"] == 1.0 for f in snap_vt["fractions"])

    # 2. Headless Streamlit AppTest verification of Tab 1 Interactive Column Studio, Quick-Jump Buttons & SVG validity
    app_path = os.path.join(root_dir, "app.py")
    at = AppTest.from_file(app_path).run(timeout=30)
    assert len(at.exception) == 0

    studio_mds = [m.value for m in at.markdown if "<style>" not in m.value and "sec-studio-svg" in m.value]
    assert len(studio_mds) == 1, f"Expected 1 rendered SEC Column SVG Studio card, found {len(studio_mds)}"

    # Verify visual placement: "See Inside the SEC Column..." immediately follows "Easy Rule of SEC" (sec-rule-card)
    # and precedes Compare Oligomer Peaks (ui-control-label) and elution-strip
    non_style_md_list = [m.value for m in at.markdown if "<style>" not in m.value]
    idx_rule = next(i for i, v in enumerate(non_style_md_list) if "sec-rule-card" in v)
    idx_inside_hdr = next(i for i, v in enumerate(non_style_md_list) if "See Inside the SEC Column: Watch How Size Controls Elution Volume" in v)
    idx_inside_svg = next(i for i, v in enumerate(non_style_md_list) if "sec-studio-svg" in v)
    idx_olig_lbl = next(i for i, v in enumerate(non_style_md_list) if "Compare Oligomer Peaks" in v)
    idx_elution_strip = next(i for i, v in enumerate(non_style_md_list) if "elution-strip" in v)
    assert idx_inside_hdr == idx_rule + 1, (
        f"Expected 'See Inside the SEC Column' right after 'sec-rule-card' ({idx_rule}), got index {idx_inside_hdr}"
    )
    assert idx_rule < idx_inside_hdr < idx_inside_svg < idx_olig_lbl < idx_elution_strip, (
        f"Unexpected Tab 1 section ordering: rule={idx_rule}, hdr={idx_inside_hdr}, svg={idx_inside_svg}, olig={idx_olig_lbl}, strip={idx_elution_strip}"
    )
    svg_match = re.search(r"(<svg\b.*?</svg>)", studio_mds[0], flags=re.DOTALL)
    assert svg_match is not None, "SVG tag missing inside sec-studio-card!"
    # Parse SVG with XML ElementTree to guarantee 100% well-formed XML (no unescaped > or &!)
    svg_root = ET.fromstring(svg_match.group(1))

    # Verify cohesive sidebar ordering:
    # Load Your Protein -> Input Method radio -> active input dialog & action button -> Try an Example -> Choose an Example
    sidebar_md_vals = [m.value for m in at.sidebar.markdown]
    assert sidebar_md_vals[0] == "### 📥 Load Your Protein", f"Expected 'Load Your Protein' first in sidebar, got {sidebar_md_vals[0]!r}"
    assert at.sidebar.radio[0].label == "Input Method:"
    assert at.sidebar.selectbox[0].label == "Choose an Example:", f"Unexpected example selectbox label: {at.sidebar.selectbox[0].label!r}"
    sidebar_elem_types = [getattr(el, "type", type(el).__name__) for el in at.sidebar]
    idx_sb_hdr = next(i for i, el in enumerate(at.sidebar) if getattr(el, "type", "") == "markdown" and "Load Your Protein" in el.value)
    idx_sb_rad = next(i for i, el in enumerate(at.sidebar) if getattr(el, "type", "") == "radio" and el.label == "Input Method:")
    idx_sb_inp = next(i for i, el in enumerate(at.sidebar) if getattr(el, "type", "") == "text_input" and el.label == "UniProt Accession ID:")
    idx_sb_btn = next(i for i, el in enumerate(at.sidebar) if getattr(el, "type", "") == "button" and el.label == "Fetch from UniProt")
    idx_sb_ex_hdr = next(i for i, el in enumerate(at.sidebar) if getattr(el, "type", "") == "markdown" and "Try an Example" in el.value)
    idx_sb_ex = next(i for i, el in enumerate(at.sidebar) if getattr(el, "type", "") == "selectbox" and el.label == "Choose an Example:")
    assert idx_sb_hdr < idx_sb_rad < idx_sb_inp < idx_sb_btn < idx_sb_ex_hdr < idx_sb_ex, (
        f"Unexpected sidebar ordering: hdr={idx_sb_hdr}, rad={idx_sb_rad}, inp={idx_sb_inp}, "
        f"btn={idx_sb_btn}, ex_hdr={idx_sb_ex_hdr}, ex={idx_sb_ex} ({sidebar_elem_types})"
    )
    sidebar_opts = list(at.sidebar.radio[0].options)
    assert sidebar_opts[0] == "🔍 UniProt ID", f"Expected UniProt ID first, got {sidebar_opts}"
    assert sidebar_opts[-1] == "💬 Describe in Plain English", f"Expected Plain English last, got {sidebar_opts}"

    # Verify removed Classroom labels & spotlight/preset text do NOT appear anywhere in UI, and zero raw <div> leaks as code blocks
    all_rendered_ui = "\n".join(
        [m.value for m in at.markdown if "<style>" not in m.value]
        + [m.value for m in at.sidebar.markdown]
        + [c.value for c in at.sidebar.caption]
    )
    for forbidden_txt in (
        "1-Click Classroom Proteins",
        "Instant examples for students",
        "Classroom Spotlight",
        "Classroom Preset",
        "Classroom Example",
    ):
        assert forbidden_txt not in all_rendered_ui, f"Removed UI text {forbidden_txt!r} still rendered!"
    assert not any("<div" in c.value for c in at.code), "Raw HTML <div> rendered in code block!"
    for _md_el in at.markdown:
        if "<style>" in _md_el.value:
            continue
        _md_s = _md_el.value.strip()
        assert not re.match(r"^[ \t]{4,}<", _md_s), f"Indented opening HTML tag risk: {_md_s[:120]!r}"
        assert not re.search(r"\n[ \t]*\n[ \t]{4,}<", _md_s), f"Blank-line + indented HTML tag CommonMark trap: {_md_s[:240]!r}"

    # Verify every line inside Panel 3 cheat-sheet box (x="746") fits comfortably (<= 55 chars)
    p3_texts = ["".join(el.itertext()) for el in svg_root.iter("{http://www.w3.org/2000/svg}text") if el.attrib.get("x") == "746"]
    assert len(p3_texts) == 5, f"Expected 5 lines in Panel 3 cheat-sheet box, got {len(p3_texts)}"
    for t_line in p3_texts:
        assert len(t_line) <= 55, f"Panel 3 cheat-sheet text too long ({len(t_line)} chars): {t_line!r}"
    assert "below to race" in p3_texts[-1]

    # Verify Bead Pores Entered column in summary dataframe
    assert "Bead Pores Entered" in list(at.dataframe[0].value.columns)
    assert "Kav" in at.dataframe[0].value.iloc[0]["Bead Pores Entered"]

    # Click all 4 Quick-Stage Jump buttons & switch Cutaway Column to Large Prep
    for btn_key_prefix in ["💉 1. Inject", "🚀 2. Void", "🎯 3. Target", "🏁 4. End"]:
        btns = [b for b in at.button if b.label.startswith(btn_key_prefix)]
        assert len(btns) == 1, f"Expected button starting with {btn_key_prefix!r}"
        btns[0].click().run(timeout=30)
        assert len(at.exception) == 0

    cutaway_radios = [r for r in at.radio if "Column Shown Inside Cutaway" in r.label]
    assert len(cutaway_radios) == 1
    cutaway_radios[0].set_value("Large Prep").run(timeout=30)
    assert len(at.exception) == 0

    # Verify SVG stays 100% well-formed XML even when Aggregate (> 1.3 MDa) is enabled!
    agg_chk = [c for c in at.checkbox if c.label == "Aggregate"][0]
    agg_chk.set_value(True).run(timeout=30)
    assert len(at.exception) == 0
    studio_mds_agg = [m.value for m in at.markdown if "<style>" not in m.value and "sec-studio-svg" in m.value]
    svg_match_agg = re.search(r"(<svg\b.*?</svg>)", studio_mds_agg[0], flags=re.DOTALL)
    assert svg_match_agg is not None
    ET.fromstring(svg_match_agg.group(1))
    non_style_md_sec = "\n".join(m.value for m in at.markdown if "<style>" not in m.value)
    assert len(re.findall(r"font-size\s*:", non_style_md_sec, flags=re.IGNORECASE)) == 0


def test_biochemical_protein_intelligence_and_metalloprotein_safety(root_dir: str) -> None:
    print("\n=== 12. Testing Biochemical Protein Intelligence, Metalloprotein/Cofactor Guards & Wise Strategy Routing ===")
    from unittest.mock import patch
    import requests as _requests
    from streamlit.testing.v1 import AppTest
    from modules.sequence_parser import (
        detect_sequence_terminal_tag,
        extract_nl_query_components,
        _extract_uniprot_biochem_annotations,
        _candidate_matches_requested_protein,
    )
    from modules.buffer_advisor import (
        infer_protein_biochemical_profile,
        audit_buffer_compatibility_for_protein,
        build_compatibility_matrix,
        build_student_lab_challenge,
        recommend_safe_buffer_preset_key,
    )

    # 1. Exact user scenarios: ALL spaced/punctuated Urease + 6xHis phrasings MUST resolve to catalytic Ni2+ Urease (NEVER Zn2+!)
    for ure_prompt in (
        "Urease with 6x His tag",
        "6x His tagged urease",
        "Urease + 6x His tag",
        "Urease 6x His tag",
        "Urease with 6x His",
        "His-6 urease",
        "Urease enzyme with 6x His tag",
        "Plant urease with 6x His tag",
    ):
        ucomp = extract_nl_query_components(ure_prompt)
        assert ucomp["tag_detected"] == "6xHis", f"Tag missed on {ure_prompt!r}: {ucomp['tag_detected']}"
        assert "6x" not in ucomp["protein_query"].lower() and "+" not in ucomp["protein_query"]
        assert "P07374" in ucomp["preferred_accessions"], f"Expected P07374 preferred for {ure_prompt!r}, got {ucomp}"

    bact_comp = extract_nl_query_components("Bacterial urease with 6x His tag")
    assert bact_comp["tag_detected"] == "6xHis" and "P18314" in bact_comp["preferred_accessions"]

    # Semantic non-existence guard: must REJECT unrelated Zn2+ Dihydropyrimidinase (Q14117) for "Human urease"!
    hum_ure_comp = extract_nl_query_components("Human urease with 6x His tag")
    fake_dhp_hit = {
        "primaryAccession": "Q14117",
        "proteinDescription": {"recommendedName": {"fullName": {"value": "Dihydropyrimidinase"}}},
        "genes": [{"geneName": {"value": "DPYS"}}],
        "organism": {"scientificName": "Homo sapiens", "taxonId": 9606},
    }
    assert _candidate_matches_requested_protein(fake_dhp_hit, hum_ure_comp) is False
    real_ure_hit = {
        "primaryAccession": "P07374",
        "proteinDescription": {"recommendedName": {"fullName": {"value": "Urease"}}},
        "genes": [{"geneName": {"value": "UREA"}}],
        "organism": {"scientificName": "Canavalia ensiformis", "taxonId": 3823},
    }
    assert _candidate_matches_requested_protein(real_ure_hit, extract_nl_query_components("Urease with 6x His tag")) is True

    with patch("modules.sequence_parser.requests.get", side_effect=_requests.ConnectionError("Simulated offline")):
        nl_urease = parse_natural_language_prompt("Urease with 6x His tag")
        nl_urease_spaced = parse_natural_language_prompt("6x His tagged urease")
    assert nl_urease["uniprot_id"] == "P07374" and nl_urease_spaced["uniprot_id"] == "P07374"
    assert nl_urease["tag_detected"] == "6xHis" and nl_urease_spaced["tag_detected"] == "6xHis"
    ure_seq = nl_urease["full_construct_sequence"]
    ure_metrics = analyze_all_parameters(ure_seq)
    ure_q74 = calculate_net_charge_at_ph(ure_seq, 7.4)

    ure_prof = infer_protein_biochemical_profile(
        sequence=ure_seq,
        protein_name=nl_urease["identified_protein"],
        metadata={
            "uniprot_id": nl_urease["uniprot_id"],
            "biochem_annotations": nl_urease.get("biochem_annotations", {}),
            "tag": "6xHis",
        },
        metrics=ure_metrics,
        tag_detected="6xHis",
        construct_parts=nl_urease["construct_parts"],
    )
    assert "ni_metalloenzyme" in ure_prof["categories"]
    assert "zn_metalloprotein" not in ure_prof["categories"]
    assert ure_prof["metal_labels"] == ["Ni²⁺ (Dinuclear Active Site)"]
    assert ure_prof["is_metalloprotein"] is True
    assert ure_prof["avoid_imidazole_imac"] is True
    assert ure_prof["is_chelator_sensitive"] is True
    assert ure_prof["tag_conflict"] is not None and ure_prof["tag_conflict"]["severity"] == "critical"
    assert "Imidazole" in ure_prof["tag_conflict"]["explanation"] and "Ni²⁺" in ure_prof["tag_conflict"]["explanation"]
    assert "Zn²⁺" not in ure_prof["tag_conflict"]["explanation"]

    # Classroom Preset Jack Bean Urease (8th preset) must also be strictly Ni2+ only
    ure_preset = CLASSROOM_PRESET_PROTEINS["urease_jack_bean"]
    assert ure_preset["uniprot_id"] == "P07374" and len(ure_preset["sequence"]) == 840
    ure_preset_prof = infer_protein_biochemical_profile(
        sequence=ure_preset["sequence"],
        protein_name=ure_preset["name"],
        metadata={"uniprot_id": "P07374", "biochem_annotations": ure_preset["biochem_annotations"]},
    )
    assert ure_preset_prof["metal_labels"] == ["Ni²⁺ (Dinuclear Active Site)"]

    # Structured UniProt JSON precision regressions:
    # (i) Carbonic Anhydrase II (P00918) prose note says 'not copper(II), nickel(II) and manganese(II)' + in-vitro Co(2+) -> STRICTLY Zn2+ ONLY!
    ca2_live_ann = _extract_uniprot_biochem_annotations({
        "comments": [{
            "commentType": "COFACTOR",
            "cofactors": [
                {"name": "Zn(2+)", "cofactorCrossReference": {"id": "CHEBI:29105"}},
                {"name": "Co(2+)", "cofactorCrossReference": {"id": "CHEBI:48828"}},
            ],
            "note": {"texts": [{"value": "Zinc. Can also use cobalt(II) with lower efficiency, but not copper(II), nickel(II) and manganese(II)."}]},
        }],
        "features": [
            {"type": "Binding site", "ligand": {"name": "Zn(2+)", "id": "CHEBI:29105"}},
            {"type": "Binding site", "ligand": {"name": "substrate"}},
        ],
        "keywords": [{"name": "Metal-binding"}, {"name": "Zinc"}],
        "proteinDescription": {"recommendedName": {"ecNumbers": [{"value": "4.2.1.1"}]}},
    })
    ca2_live_prof = infer_protein_biochemical_profile(
        CLASSROOM_PRESET_PROTEINS["carbonic_anhydrase"]["sequence"],
        "Carbonic anhydrase 2",
        {"uniprot_id": "P00918", "biochem_annotations": ca2_live_ann},
        tag_detected="6xHis",
    )
    assert ca2_live_prof["metal_labels"] == ["Zn²⁺ Cofactor"], f"CA II leaked extra metals: {ca2_live_prof['metal_labels']}"
    assert "ni_metalloenzyme" not in ca2_live_prof["categories"]

    # (ii) BSA (P02769) serum cargo carrier binding sites must NOT misclassify BSA as a catalytic metalloprotein!
    bsa_live_ann = _extract_uniprot_biochem_annotations({
        "comments": [],
        "features": [
            {"type": "Binding site", "ligand": {"name": "Cu cation"}},
            {"type": "Binding site", "ligand": {"name": "Zn(2+)", "id": "CHEBI:29105"}},
            {"type": "Binding site", "ligand": {"name": "Ca(2+)", "id": "CHEBI:29108"}},
        ],
        "keywords": [{"name": "Calcium"}, {"name": "Copper"}, {"name": "Disulfide bond"}, {"name": "Metal-binding"}, {"name": "Secreted"}, {"name": "Zinc"}],
    })
    bsa_live_prof = infer_protein_biochemical_profile(
        CLASSROOM_PRESET_PROTEINS["bsa"]["sequence"],
        "Albumin",
        {"uniprot_id": "P02769", "biochem_annotations": bsa_live_ann},
        tag_detected="6xHis",
    )
    assert bsa_live_prof["is_metalloprotein"] is False and "disulfide_structural" in bsa_live_prof["categories"]

    # (iii) Live TEV Protease P04517 viral polyprotein 'Disulfide bond' keyword must NOT suppress DTT/TCEP requirement!
    tev_live_prof = infer_protein_biochemical_profile(
        "", "Genome polyprotein", {"uniprot_id": "P04517", "biochem_annotations": {"keywords": ["Disulfide bond"]}}
    )
    assert tev_live_prof["protease_class"] == "cysteine"
    assert tev_live_prof["require_reducing_agents"] is True and tev_live_prof["avoid_reducing_agents"] is False

    # (iv) Acid-activated urea channel (Q09068 ureI) & MCR methyltransferase (Q8THG6) must NOT trigger ni_metalloenzyme!
    urei_prof = infer_protein_biochemical_profile("", "Acid-activated urea channel", {"uniprot_id": "Q09068", "gene_name": "ureI"})
    assert urei_prof["is_metalloprotein"] is False and "membrane_integral" in urei_prof["categories"]
    mcr_mt_prof = infer_protein_biochemical_profile("", "[Methyl coenzyme M reductase]-L-arginine C-5-methyltransferase", {"uniprot_id": "Q8THG6"})
    assert "ni_metalloenzyme" not in mcr_mt_prof["categories"]

    wf_urease_his = recommend_full_purification_workflow(
        ure_metrics, 7.4, ure_q74, tag_detected="6xHis",
        sequence=ure_seq, protein_name="Urease", protein_profile=ure_prof,
    )
    assert "Bypass Ni-NTA" in wf_urease_his["roadmap"][0]["title"]
    assert "Affinity Capture: Ni-NTA" not in wf_urease_his["roadmap"][0]["title"]
    assert "Avoid Ni-NTA" in wf_urease_his["all_methods"][0]["Suitability"]
    assert "IMAC eluate" not in wf_urease_his["roadmap"][1]["detail"]
    assert any("Nickel Metalloenzyme Protection" in a and "EDTA" in a for a in wf_urease_his["alerts"])

    # Compatibility Board (6 tiles) & Interactive Student Lab Challenge
    ure_tiles = build_compatibility_matrix(ure_metrics, 7.4, ure_q74, wf_urease_his)
    assert len(ure_tiles) == 6 and ure_tiles[0]["status"] == "avoid" and ure_tiles[1]["status"] == "avoid"
    ure_chal = build_student_lab_challenge("Urease", ure_metrics, 7.4, ure_q74, wf_urease_his)
    assert "Ni²⁺" in ure_chal["options"][ure_chal["correct_index"]] and "Zn²⁺" not in ure_chal["explanation"]
    assert recommend_safe_buffer_preset_key(ure_prof, ure_metrics, 7.4, ure_q74) == "metalloprotein_safe"

    # Verify Strep-tag & MBP-tag on Urease automatically switch to 0 mM EDTA (EDTA-FREE!)
    for safe_tag in ("Twin-Strep-tag", "MBP"):
        wf_ure_safe = recommend_full_purification_workflow(
            ure_metrics, 7.4, ure_q74, tag_detected=safe_tag,
            sequence=nl_urease["canonical_sequence"], protein_name="Urease",
            metadata={"uniprot_id": "P07374"},
        )
        assert wf_ure_safe["tag_conflict"] is None
        assert "0 mM EDTA (EDTA-FREE" in wf_ure_safe["roadmap"][0]["detail"]
        assert "1 mM EDTA" not in wf_ure_safe["roadmap"][0]["detail"]

    # Verify raw unnamed Urease FASTA with terminal His-tag is STILL caught purely from sequence + active-site motif!
    raw_mystery_urease = "MHHHHHH" + nl_urease["canonical_sequence"]
    det_tag = detect_sequence_terminal_tag(raw_mystery_urease)
    assert det_tag["tag_detected"] == "6xHis" and det_tag["core_sequence"] == nl_urease["canonical_sequence"]
    mystery_prof = infer_protein_biochemical_profile(
        sequence=raw_mystery_urease, protein_name="Unnamed_Sample_1", metrics=analyze_all_parameters(raw_mystery_urease)
    )
    assert "ni_metalloenzyme" in mystery_prof["categories"] and mystery_prof["tag_conflict"] is not None

    # 2. Broad Metalloprotein & Cofactor Matrix (Zn, Fe-S, Cu/Mn, Heme, Ca2+, Membrane, PLP, Proteases, Non-Metalloprotein Controls)
    # (a) Zn2+ Carbonic Anhydrase II (P00918) + 6xHis -> Mismetallation & Imidazole block + Citrate warning
    ca2_seq = "MHHHHHH" + CLASSROOM_PRESET_PROTEINS["carbonic_anhydrase"]["sequence"]
    ca2_m = analyze_all_parameters(ca2_seq)
    wf_ca2_his = recommend_full_purification_workflow(
        ca2_m, 6.0, calculate_net_charge_at_ph(ca2_seq, 6.0),
        tag_detected="6xHis", sequence=ca2_seq, protein_name="Carbonic Anhydrase 2",
        metadata={"uniprot_id": "P00918"},
    )
    assert wf_ca2_his["tag_conflict"] is not None and wf_ca2_his["tag_conflict"]["severity"] == "critical"
    assert "Bypass Ni-NTA" in wf_ca2_his["roadmap"][0]["title"]
    citrate_notes = [b["notes"] for b in wf_ca2_his["iex"]["recommended_buffers"] if "Citrate" in b["name"]]
    assert citrate_notes and "AVOID" in citrate_notes[0]

    # (b) [Fe-S] O2-sensitive Ferredoxin + 6xHis -> Anaerobic < 2 ppm O2 + IMAC block
    fd_prof = infer_protein_biochemical_profile(protein_name="Spinach ferredoxin", tag_detected="6xHis")
    assert "fe_s_cluster" in fd_prof["categories"] and fd_prof["is_o2_sensitive"] is True and fd_prof["avoid_imidazole_imac"] is True
    assert fd_prof["tag_conflict"] is not None and "anaerobic" in fd_prof["tag_conflict"]["better_strategy"].lower()

    # (c) Cu / Mn Metalloenzymes (Laccase, Arginase)
    for m_name in ("Fungal laccase", "Human arginase-1", "Matrix metalloproteinase-9"):
        mp = infer_protein_biochemical_profile(protein_name=m_name, tag_detected="6xHis")
        assert mp["is_metalloprotein"] is True and mp["avoid_imidazole_imac"] is True and mp["tag_conflict"] is not None

    # (d) Hemoproteins (Myoglobin P02185 & Cytochrome P450) -> Axial Heme-Imidazole coordination caution
    myo_seq = "MHHHHHH" + CLASSROOM_PRESET_PROTEINS["myoglobin"]["sequence"]
    myo_m = analyze_all_parameters(myo_seq)
    wf_myo_his = recommend_full_purification_workflow(
        myo_m, 7.4, calculate_net_charge_at_ph(myo_seq, 7.4),
        tag_detected="6xHis", sequence=myo_seq, protein_name="Myoglobin", metadata={"uniprot_id": "P02185"},
    )
    assert wf_myo_his["protein_profile"]["is_heme_protein"] is True
    assert wf_myo_his["tag_conflict"] is not None and wf_myo_his["tag_conflict"]["severity"] == "warning"
    assert "Heme-Safe" in wf_myo_his["roadmap"][0]["title"]

    # (e) Ca2+-dependent proteins (Calmodulin / Subtilisin) -> No EDTA & No Sodium Phosphate precipitation!
    lyso_m = analyze_all_parameters(LYSOZYME_MATURE_SEQ)
    cam_prof = infer_protein_biochemical_profile(protein_name="Calmodulin", metrics=lyso_m)
    assert cam_prof["needs_calcium"] is True and cam_prof["avoid_phosphate_buffer"] is True and cam_prof["is_chelator_sensitive"] is True
    hic_cam = advise_hic_strategy(lyso_m, 7.0, -8.0, protein_profile=cam_prof)
    assert "avoid Sodium Phosphate" in hic_cam["elution_protocol"]

    # (f) Integral Membrane Protein (Bacteriorhodopsin) -> Detergent micelles required, HIC & (NH4)2SO4 forbidden, SEC +70 kDa micelle shift!
    mem_m = dict(lyso_m, mw_kda=28.0, gravy=0.65)
    wf_mem = recommend_full_purification_workflow(mem_m, 7.5, -4.0, tag_detected="Twin-Strep-tag", protein_name="Bacteriorhodopsin")
    assert wf_mem["protein_profile"]["requires_detergent_micelles"] is True
    assert "DDM" in wf_mem["roadmap"][0]["detail"]
    assert "Avoid" in wf_mem["all_methods"][2]["Suitability"]  # Ammonium sulfate forbidden
    assert "Avoid" in wf_mem["all_methods"][4]["Suitability"]  # Aqueous HIC forbidden
    assert "Superdex 200" in wf_mem["roadmap"][2]["title"]     # 28 kDa + ~70 kDa DDM micelle -> S200, not S75!

    # (g) PLP Holoenzyme (Alanine aminotransferase) -> Avoid primary amine Tris buffer!
    plp_prof = infer_protein_biochemical_profile(protein_name="Alanine aminotransferase")
    assert plp_prof["avoid_tris_buffer"] is True

    # (h) Active Serine Protease (Trypsin) vs Cysteine Protease (TEV P04517) vs Non-Protease 6xHis-TEV-Lysozyme Linker Control!
    tryp_prof = infer_protein_biochemical_profile(protein_name="Trypsin")
    assert tryp_prof["protease_class"] == "serine"
    tev_prof = infer_protein_biochemical_profile(protein_name="TEV Protease", metadata={"uniprot_id": "P04517"})
    assert tev_prof["protease_class"] == "cysteine" and tev_prof["require_reducing_agents"] is True

    # Control: 6xHis-TEV-Lysozyme & 6xHis-GST & 6xHis-GFP MUST NOT trigger false-positive metalloprotein or TEV-protease rules!
    his_tev_lyso = CLASSROOM_PRESET_PROTEINS["lysozyme_his_tev"]
    htl_prof = infer_protein_biochemical_profile(
        sequence=his_tev_lyso["sequence"],
        protein_name=his_tev_lyso["name"],
        metadata={"uniprot_id": "P00698", "tag": "6xHis"},
        metrics=analyze_all_parameters(his_tev_lyso["sequence"]),
        tag_detected="6xHis",
        construct_parts=his_tev_lyso["construct_parts"],
    )
    assert htl_prof["is_metalloprotein"] is False
    assert htl_prof["protease_class"] is None
    assert htl_prof["avoid_reducing_agents"] is True
    assert htl_prof["tag_conflict"] is None

    for non_metal_ctrl in ("Glutathione S-transferase P", "Green fluorescent protein"):
        ctrl_p = infer_protein_biochemical_profile(protein_name=non_metal_ctrl, tag_detected="6xHis")
        assert ctrl_p["is_metalloprotein"] is False and ctrl_p["tag_conflict"] is None

    # 3. Tab 3 Custom Buffer Recipe Compatibility Auditor
    ure_his_lysis_warns = audit_buffer_compatibility_for_protein(
        PRESET_RECIPE_TEMPLATES["his_lysis"]["rows"], 8.0, ure_prof, "Urease"
    )
    assert any("Metalloprotein–Imidazole Conflict" in w for w in ure_his_lysis_warns)
    ure_strep_edta_warns = audit_buffer_compatibility_for_protein(
        PRESET_RECIPE_TEMPLATES["strep_elution"]["rows"], 8.0, ure_prof, "Urease"
    )
    assert any("Chelator Stripping Hazard" in w and "EDTA" in w for w in ure_strep_edta_warns)
    ure_safe_warns = audit_buffer_compatibility_for_protein(
        PRESET_RECIPE_TEMPLATES["metalloprotein_safe"]["rows"], 7.5, ure_prof, "Urease"
    )
    assert len(ure_safe_warns) == 0

    # 4. Simulate Streamlit Cloud hot-reload with stale sys.modules missing newly added functions
    import sys as _sys
    delattr(_sys.modules["modules.sequence_parser"], "detect_sequence_terminal_tag")
    delattr(_sys.modules["modules.buffer_advisor"], "infer_protein_biochemical_profile")

    # 5. Headless Streamlit AppTest end-to-end verification of "Urease with 6x His tag"
    app_path = os.path.join(root_dir, "app.py")
    at_reload = AppTest.from_file(app_path).run(timeout=30)
    assert len(at_reload.exception) == 0, f"Hot-reload recovery failed: {at_reload.exception}"
    assert hasattr(_sys.modules["modules.sequence_parser"], "detect_sequence_terminal_tag")
    assert hasattr(_sys.modules["modules.buffer_advisor"], "infer_protein_biochemical_profile")

    with patch("modules.sequence_parser.requests.get", side_effect=_requests.ConnectionError("Simulated offline")):
        at = AppTest.from_file(app_path).run(timeout=30)
        at.sidebar.radio[0].set_value("💬 Describe in Plain English").run(timeout=30)
        at.sidebar.text_area[0].set_value("Urease with 6x His tag").run(timeout=30)
        at.sidebar.button[0].click().run(timeout=30)
    assert len(at.exception) == 0, f"AppTest exception on 'Urease with 6x His tag': {at.exception}"
    assert at.session_state["sequence_metadata"]["uniprot_id"] == "P07374"
    assert at.session_state["sequence_metadata"]["tag"] == "6xHis"
    assert len(at.error) >= 1 and any("Metalloprotein–IMAC Conflict" in e.value for e in at.error)
    assert any("Metalloprotein–Imidazole Conflict" in w.value for w in at.warning)
    all_ure_md = "\n".join(m.value for m in at.markdown if "<style>" not in m.value)
    assert "Bypass Ni-NTA / Imidazole" in all_ure_md
    assert "Affinity Capture: Ni-NTA" not in all_ure_md
    assert "Quick Compatibility Board" in all_ure_md
    assert "compat-grid" in all_ure_md
    assert "Ni²⁺ (Dinuclear Active Site)" in all_ure_md
    assert "This protein coordinates **Zn²⁺" not in all_ure_md
    assert len(re.findall(r"font-size\s*:", all_ure_md, flags=re.IGNORECASE)) == 0

    # Click the 1-Click Safe Buffer Switch button in Tab 3 and verify the Imidazole warning clears!
    safe_btns = [b for b in at.button if "Switch to Safe Buffer" in b.label]
    assert len(safe_btns) == 1, f"Expected 1-Click Safe Buffer button, found {[b.label for b in at.button]}"
    safe_btns[0].click().run(timeout=30)
    assert len(at.exception) == 0
    assert at.session_state["active_recipe_key"] == "metalloprotein_safe"
    assert not any("Metalloprotein–Imidazole Conflict" in w.value for w in at.warning)


def test_all_5_input_methods_hardening_and_minimal_chrome(root_dir: str) -> None:
    print("\n=== 13. Testing Hardening of All 5 Input Methods, ChEBI Normalization & Minimal Branding Chrome ===")
    import time
    from unittest.mock import patch
    from streamlit.testing.v1 import AppTest

    # 1. Zero author/creator branding and clean Streamlit minimal chrome config
    cfg_path = os.path.join(root_dir, ".streamlit", "config.toml")
    with open(cfg_path, "r", encoding="utf-8") as cf:
        cfg_txt = cf.read()
    assert 'toolbarMode = "minimal"' in cfg_txt

    app_path = os.path.join(root_dir, "app.py")
    with open(app_path, "r", encoding="utf-8") as af:
        app_txt = af.read()
    assert "menu_items={}" in app_txt
    assert "viewerBadge" in app_txt and "#MainMenu" in app_txt

    for folder, fnames in ((root_dir, ("app.py",)), (os.path.join(root_dir, "modules"), os.listdir(os.path.join(root_dir, "modules")))):
        for fn in fnames:
            if not fn.endswith(".py"):
                continue
            with open(os.path.join(folder, fn), "r", encoding="utf-8") as fobj:
                content = fobj.read()
            assert not re.search(r"prafful|created\s+by", content, flags=re.IGNORECASE), f"Branding text found in {fn}"
    assert not re.search(r"prafful|created\s+by", cfg_txt, flags=re.IGNORECASE)

    # 2. Method 1 (UniProt ID): URLs, FASTA pipes, isoforms, .1 versions, entry names, ChEBI normalization, mocked API paths
    assert _normalize_uniprot_input("https://www.uniprot.org/uniprotkb/P00698/entry")["query"] == "P00698"
    assert _normalize_uniprot_input("sp|P00698|LYSC_CHICK")["query"] == "P00698"
    assert _normalize_uniprot_input("UniProt: P07374.1")["query"] == "P07374"
    iso_norm = _normalize_uniprot_input("P04637-2")
    assert iso_norm == {"query": "P04637-2", "base_accession": "P04637", "isoform_num": "2", "is_entry_name": False}
    ename_norm = _normalize_uniprot_input("LYSC_CHICK")
    assert ename_norm["query"] == "LYSC_CHICK" and ename_norm["is_entry_name"] is True
    try:
        _normalize_uniprot_input("   ")
        raise AssertionError("Blank UniProt ID should raise ValueError")
    except ValueError:
        pass

    # Live UniProt double-prefix ChEBI normalization ('ChEBI:CHEBI:49786' -> 'CHEBI:49786') & CHEBI:25516 Ni cation
    ann_live_style = _extract_uniprot_biochem_annotations({
        "comments": [{"commentType": "COFACTOR", "cofactors": [{"name": "Ni cation", "cofactorCrossReference": {"id": "CHEBI:25516"}}]}],
        "features": [{"type": "Binding site", "ligand": {"name": "Ni(2+)", "id": "ChEBI:CHEBI:49786"}}],
    })
    assert "CHEBI:25516" in ann_live_style["cofactor_chebi_ids"]
    assert "CHEBI:49786" in ann_live_style["metal_ligand_chebi_ids"]

    class _MockResp:
        def __init__(self, code: int, text: str = "", json_obj=None):
            self.status_code = code
            self.text = text
            self._json = json_obj

        def json(self):
            return self._json

    def _mock_m1_m2_get(url: str, **kwargs):
        if "P04637-2.fasta" in url:
            return _MockResp(200, text=">sp|P04637-2|P53_HUMAN Isoform 2 OS=Homo sapiens GN=TP53\n" + ("MEEPQSDPSV" * 20) + "\n")
        if "P04637.json" in url:
            return _MockResp(200, json_obj={
                "primaryAccession": "P04637",
                "proteinDescription": {"recommendedName": {"fullName": {"value": "Cellular tumor antigen p53"}}},
                "organism": {"scientificName": "Homo sapiens"},
                "genes": [{"geneName": {"value": "TP53"}}],
                "sequence": {"value": "M" * 393, "length": 393},
                "features": [{"type": "Metal binding", "ligand": {"name": "Zn(2+)", "id": "ChEBI:CHEBI:29105"}}],
            })
        if "OLD001.json" in url:
            return _MockResp(200, json_obj={
                "entryType": "Inactive",
                "primaryAccession": "OLD001",
                "inactiveReason": {"inactiveReasonType": "MERGED", "mergeDemergeTo": ["P00698"]},
            })
        if "P00698.json" in url:
            return _MockResp(200, json_obj={
                "primaryAccession": "P00698",
                "proteinDescription": {"recommendedName": {"fullName": {"value": "Lysozyme C"}}},
                "organism": {"scientificName": "Gallus gallus"},
                "genes": [{"geneName": {"value": "LYZ"}}],
                "sequence": {"value": "MRSLLILVLCFLPLAALG" + LYSOZYME_MATURE_SEQ, "length": 147},
            })
        if "search?query=id:TEST_ENAME" in url:
            return _MockResp(200, json_obj={"results": [{
                "primaryAccession": "P99901",
                "proteinDescription": {"recommendedName": {"fullName": {"value": "Entry Name Search Protein"}}},
                "organism": {"scientificName": "Escherichia coli"},
                "genes": [{"geneName": {"value": "tesE"}}],
                "sequence": {"value": "M" * 120, "length": 120},
            }]})
        if "db=protein" in url and "NP_990612.2" in url:
            return _MockResp(200, text=">NP_990612.2 lysozyme C precursor [Gallus gallus]\nMRSLLILVLCFLPLAALG" + LYSOZYME_MATURE_SEQ + "\n")
        if "db=protein" in url and "DUAL_TEST_1" in url:
            return _MockResp(400, text="")
        if "db=nuccore" in url and "DUAL_TEST_1" in url:
            return _MockResp(200, text=">DUAL_TEST_1 Gallus gallus synthetic lysozyme mRNA\nCCCTAAATGGCTGCTCGTTAA\n")
        return _MockResp(404, text="")

    with patch("modules.sequence_parser.requests.get", side_effect=_mock_m1_m2_get):
        r_iso = fetch_uniprot_by_id("P04637-2")
        assert r_iso["accession"] == "P04637-2" and r_iso["length"] == 200
        assert "CHEBI:29105" in r_iso["biochem_annotations"]["metal_ligand_chebi_ids"]

        r_merge = fetch_uniprot_by_id("OLD001")
        assert r_merge["accession"] == "P00698" and r_merge["length"] == 147

        r_ename = fetch_uniprot_by_id("TEST_ENAME")
        assert r_ename["accession"] == "P99901" and r_ename["length"] == 120

        # 3. Method 2 (NCBI Accession): protein vs nuccore classification, nested [Organism], dual-DB fallback
        for acc_p in ("NP_990612.2", "WP_000002298.1", "QHD43416.1", "AAA12345.1", "1LYZ_A"):
            assert _classify_ncbi_accession_db(acc_p) == "protein", f"Expected protein for {acc_p}"
        for acc_n in ("NM_205281.2", "M21110.1", "BC004147", "MN908947.3", "U00096.3", "NZ_CP012345.1"):
            assert _classify_ncbi_accession_db(acc_n) == "nuccore", f"Expected nuccore for {acc_n}"
        assert _normalize_ncbi_input("https://www.ncbi.nlm.nih.gov/nuccore/M21110.1") == "M21110.1"
        assert _normalize_ncbi_input("ref|NP_990612.2|") == "NP_990612.2"
        assert _normalize_ncbi_input("pdb|1LYZ|A") == "1LYZ_A"

        pn_rot, org_rot = _parse_ncbi_fasta_header(
            "NP_114261.1 RNA-directed RNA polymerase [Rotavirus A (strain RVA/SA11-Patton/G3P[X])]",
            "NP_114261.1",
        )
        assert pn_rot == "RNA-directed RNA polymerase"
        assert org_rot == "Rotavirus A (strain RVA/SA11-Patton/G3P[X])"

        r_dual = fetch_ncbi_by_id("DUAL_TEST_1")
        assert r_dual["sequence"] == "MAAR" and "Nucleotide ORF" in r_dual["source"]

    # 4. Method 3 (Paste Amino Acid / FASTA): Multi-FASTA stop+warning & metadata extraction
    ure_seq = CLASSROOM_PRESET_PROTEINS["urease_jack_bean"]["sequence"]
    multi_fasta = (
        f">sp|P07374|UREA_CANEN Urease OS=Canavalia ensiformis OX=3823 GN=UREA PE=1 SV=2\n{ure_seq}\n"
        f">sp|P00698|LYSC_CHICK Lysozyme C OS=Gallus gallus GN=LYZ\n{LYSOZYME_MATURE_SEQ}\n"
    )
    cleaned_mf, hdr_mf, warns_mf = clean_amino_acid_sequence(multi_fasta)
    assert cleaned_mf == ure_seq, "Multi-FASTA concatenated second sequence instead of stopping at 2nd '>'!"
    assert any("Multiple FASTA entries" in w for w in warns_mf)
    hdr_meta = extract_fasta_header_metadata(hdr_mf)
    assert hdr_meta["uniprot_id"] == "P07374" and hdr_meta["gene_name"] == "UREA"
    assert hdr_meta["organism"] == "Canavalia ensiformis" and "Ni(2+)" in hdr_meta["biochem_annotations"].get("cofactors", [])

    # 5. Method 4 (Translate DNA / RNA): FASTA header strip, IUPAC N frame preservation, 6-frame minus strand, O(L) speed
    fasta_nt = ">NM_205281.2 Gallus gallus lysozyme (AUG UAA test)\nATGGCTGCTTAA\n"
    assert translate_nucleotide(fasta_nt, frame=1) == "MAA*"
    assert translate_nucleotide("ATGNNNGCTTAA", frame=1) == "MXA*"
    assert reverse_complement("ATGGCT") == "AGCCAT"
    minus_dna = reverse_complement("CCCTTT" + "ATGGCTGCTAAAGGGTTTTAA" + "CCCTTT")
    minus_orf = find_longest_orf(minus_dna)
    assert minus_orf["sequence"] == "MAAKGF" and minus_orf["strand"] == "-" and minus_orf["frame"] in (-1, -2, -3)
    assert translate_nucleotide(minus_dna[:-6], frame=-1).split("*")[0] == "MAAKGF"

    t0 = time.perf_counter()
    long_orf = find_longest_orf("ATG" + "GCT" * 10000)
    assert long_orf["orf_length_aa"] == 10001 and (time.perf_counter() - t0) < 0.25

    # 6. AppTest UI end-to-end checks for Method 2 (NCBI badge prefix) & Method 3 (Urease FASTA intelligence)
    at = AppTest.from_file(app_path).run(timeout=30)
    at.sidebar.radio[0].set_value("🔤 Paste Amino Acid / FASTA").run(timeout=30)
    at.sidebar.text_area[0].set_value(multi_fasta).run(timeout=30)
    at.sidebar.button[0].click().run(timeout=30)
    assert len(at.exception) == 0, f"AppTest FASTA exception: {at.exception}"
    assert at.session_state["active_sequence"] == ure_seq
    assert len(at.warning) >= 1 and any("Multiple FASTA entries" in w.value for w in at.warning)
    md_fasta = "\n".join(m.value for m in at.markdown if "<style>" not in m.value)
    assert "Ni²⁺" in md_fasta and "UniProt P07374" in md_fasta

    with patch("modules.sequence_parser.requests.get", side_effect=_mock_m1_m2_get):
        at_ncbi = AppTest.from_file(app_path).run(timeout=30)
        at_ncbi.sidebar.radio[0].set_value("🏛️ NCBI Accession").run(timeout=30)
        at_ncbi.sidebar.text_input[0].set_value("NP_990612.2").run(timeout=30)
        at_ncbi.sidebar.button[0].click().run(timeout=30)
    assert len(at_ncbi.exception) == 0, f"AppTest NCBI exception: {at_ncbi.exception}"
    md_ncbi = "\n".join(m.value for m in at_ncbi.markdown if "<style>" not in m.value)
    assert "NCBI NP_990612.2" in md_ncbi
    assert "UniProt NP_990612.2" not in md_ncbi
    assert len(re.findall(r"font-size\s*:", md_ncbi, flags=re.IGNORECASE)) == 0

    # 7. Scientific precision & caveat checks (disulfide-oxidized pI, resin-specific EDTA/Ni-NTA warning, heuristic solubility, benchmark transparency)
    from modules.buffer_advisor import build_compatibility_matrix, audit_buffer_compatibility_for_protein, infer_protein_biochemical_profile

    lyso_m = analyze_all_parameters(LYSOZYME_MATURE_SEQ)
    assert abs(lyso_m["pi"] - 9.32) <= 0.01
    assert abs(lyso_m["pi_oxidized"] - 10.46) <= 0.02
    assert "Heuristic" in lyso_m["solubility"]["solubility_class"]
    assert "87.5%" not in lyso_m["solubility"]["description"] and "81 proteins" in lyso_m["solubility"]["description"]

    # Resin-specific EDTA warnings for untagged & His-tagged non-metalloproteins
    lyso_prof_untagged = infer_protein_biochemical_profile(
        sequence=LYSOZYME_MATURE_SEQ,
        protein_name="Lysozyme C",
        metadata={"uniprot_id": "P00698"},
        metrics=lyso_m,
        tag_detected="None",
    )
    wf_lyso_untagged = recommend_full_purification_workflow(
        lyso_m, 7.4, calculate_net_charge_at_ph(LYSOZYME_MATURE_SEQ, 7.4),
        tag_detected="None", sequence=LYSOZYME_MATURE_SEQ, protein_name="Lysozyme C", protein_profile=lyso_prof_untagged,
    )
    chel_tile_untagged = [t for t in build_compatibility_matrix(lyso_m, 7.4, 7.38, wf_lyso_untagged) if "EDTA" in t["title"]][0]
    assert "Ni-NTA" in chel_tile_untagged["reason"] and "strips resin Ni²⁺" in chel_tile_untagged["reason"]

    lyso_prof_his = infer_protein_biochemical_profile(
        sequence="MHHHHHH" + LYSOZYME_MATURE_SEQ,
        protein_name="Lysozyme C",
        metadata={"uniprot_id": "P00698"},
        metrics=lyso_m,
        tag_detected="6xHis",
    )
    wf_lyso_his = recommend_full_purification_workflow(
        lyso_m, 7.4, calculate_net_charge_at_ph(LYSOZYME_MATURE_SEQ, 7.4),
        tag_detected="6xHis", sequence="MHHHHHH" + LYSOZYME_MATURE_SEQ, protein_name="Lysozyme C", protein_profile=lyso_prof_his,
    )
    chel_tile_his = [t for t in build_compatibility_matrix(lyso_m, 7.4, 7.38, wf_lyso_his) if "EDTA" in t["title"]][0]
    assert chel_tile_his["status"] == "warn" and "Omit for Ni-NTA" in chel_tile_his["badge"]

    imac_edta_warns = audit_buffer_compatibility_for_protein(
        [
            {"Reagent": "Tris-Base", "Value": 50.0, "Unit": "mM"},
            {"Reagent": "Imidazole (His-Tag Eluant)", "Value": 20.0, "Unit": "mM"},
            {"Reagent": "EDTA Disodium·2H2O", "Value": 1.0, "Unit": "mM"},
        ],
        8.0,
        lyso_prof_his,
        "Lysozyme C",
    )
    assert any("IMAC Resin Stripping Hazard" in w for w in imac_edta_warns)

    # Default Lysozyme page shows oxidized S-S pI estimate & removes undefined '~99% volume accuracy' claim
    assert "~99% volume accuracy" not in app_txt
    assert "Benchmark Methodology, Data Splits & Practical Scope" in app_txt
    at_def = AppTest.from_file(app_path).run(timeout=30)
    assert len(at_def.exception) == 0
    md_def = "\n".join(m.value for m in at_def.markdown if "<style>" not in m.value)
    cap_def = "\n".join(c.value for c in at_def.caption)
    assert "S–S oxidized: ~10.46" in md_def
    assert "Bjellqvist" in cap_def and "10.46" in cap_def
    assert "500 mL (500 mL" not in md_def and "Exact Weigh-Out Sheet for **500 mL** (Target pH" in md_def
    assert 'initial_sidebar_state="auto"' in app_txt
    assert len(re.findall(r"font-size\s*:", md_def, flags=re.IGNORECASE)) == 0

    # 8. Superdex 75 prep vs analytical Kav consistency, CEX Tris-free buffer tile, low-MW peptide gel & UniProt mature-chain toggle
    s75_anal = SEC_COLUMNS["s75_increase_10_300"]
    s75_prep = SEC_COLUMNS["hiload_16_600_s75"]
    for std_mw in (75.0, 44.0, 29.0, 14.31, 13.7, 6.5, 1.355):
        _, kav_a, _ = calculate_retention_volume(std_mw, s75_anal)
        _, kav_p, _ = calculate_retention_volume(std_mw, s75_prep)
        assert abs(kav_a - kav_p) <= 0.005, f"S75 Kav mismatch at {std_mw} kDa: anal={kav_a:.4f}, prep={kav_p:.4f}"

    # CEX buffer tile must NOT recommend Tris when routing Lysozyme to SP Sepharose
    buf_tile_lyso = [t for t in build_compatibility_matrix(lyso_m, 7.4, 7.38, wf_lyso_untagged) if "Buffer Family" in t["title"]][0]
    assert "Tris" not in buf_tile_lyso["badge"] and "omit cationic Tris" in buf_tile_lyso["reason"]

    # UniProt live-style 'Signal' + 'Chain' feature extraction & 1-click Mature Chain toggle in UI
    from modules.sequence_parser import _parse_uniprot_entry_dict
    parsed_p00698 = _parse_uniprot_entry_dict({
        "primaryAccession": "P00698",
        "proteinDescription": {"recommendedName": {"fullName": {"value": "Lysozyme C"}}},
        "organism": {"scientificName": "Gallus gallus"},
        "genes": [{"geneName": {"value": "LYZ"}}],
        "sequence": {"value": "MRSLLILVLCFLPLAALG" + LYSOZYME_MATURE_SEQ, "length": 147},
        "features": [
            {"type": "Signal", "location": {"start": {"value": 1}, "end": {"value": 18}}},
            {"type": "Chain", "location": {"start": {"value": 19}, "end": {"value": 147}}, "description": "Lysozyme C"},
        ],
    })
    assert parsed_p00698["biochem_annotations"]["has_signal_peptide"] is True
    assert parsed_p00698["mature_chain"]["sequence"] == LYSOZYME_MATURE_SEQ
    assert parsed_p00698["mature_chain"]["length"] == 129

    # Test 1-click Mature Chain toggle in AppTest (fetching offline P00698 precursor -> toggle to 129 aa mature chain)
    import requests as _req_mod
    with patch("modules.sequence_parser.requests.get", side_effect=_req_mod.ConnectionError("Offline")):
        at_mat = AppTest.from_file(app_path).run(timeout=30)
        at_mat.sidebar.button[0].click().run(timeout=30)  # Fetch P00698 (147 aa precursor)
    assert len(at_mat.exception) == 0 and len(at_mat.session_state["active_sequence"]) == 147
    mat_btns = [b for b in at_mat.button if "Use Mature Chain (129 aa)" in b.label]
    assert len(mat_btns) == 1, f"Expected Mature Chain toggle button, found {[b.label for b in at_mat.button]}"
    mat_btns[0].click().run(timeout=30)
    assert len(at_mat.exception) == 0 and at_mat.session_state["active_sequence"] == LYSOZYME_MATURE_SEQ

    # Test 3.2 kDa peptide gel advisory (never green-checks Laemmli; recommends Tricine / Bis-Tris MES!)
    at_pep = AppTest.from_file(app_path).run(timeout=30)
    at_pep.sidebar.radio[0].set_value("🔤 Paste Amino Acid / FASTA").run(timeout=30)
    at_pep.sidebar.text_area[0].set_value(">Tiny_Peptide_3kDa\nACDEFGHIKLMNPQRSTVWYACDEFGHIKL\n").run(timeout=30)
    at_pep.sidebar.button[0].click().run(timeout=30)
    assert len(at_pep.exception) == 0
    pep_info_warn = "\n".join([i.value for i in at_pep.info] + [w.value for w in at_pep.warning])
    assert "Tricine" in pep_info_warn and "Bis-Tris" in pep_info_warn
    assert not any("gives crisp band separation" in s.value for s in at_pep.success)


def test_gel_recipes_3d_viewer_feedback_and_ai_proteins(root_dir: str) -> None:
    print("\n=== 14. Testing Gel Recipes, 3D Viewer, Feedback Modal, AI-Designed Protein Fixes & AppTest ===")
    import glob
    import gzip
    import py_compile
    import sys as _sys
    import time
    from unittest.mock import patch
    import requests as _requests
    from streamlit.testing.v1 import AppTest

    from modules.buffer_advisor import infer_protein_biochemical_profile
    from modules.gel_recipe import (
        RECIPE_DISPLAY_COLUMNS,
        GEL_FORMAT_PRESETS,
        calculate_sds_gel_recipe,
        format_gel_recipe_text,
        get_max_achievable_gel_pct,
        recommend_gel_recipe_params,
    )
    from modules.protein_3d import (
        SCHEMATIC_3D_DISCLAIMER,
        SUPPORTED_3DMOL_STYLES,
        SUPPORTED_3DMOL_COLOR_SCHEMES,
        normalize_uniprot_accession,
        parse_uploaded_structure_file,
        fetch_3d_structure_for_uniprot,
        build_3dmol_viewer_html,
        build_3d_biophysical_backbone_figure,
    )

    # ------------------------------------------------------------------
    # 1. Static, Syntax & Branding Audit
    # ------------------------------------------------------------------
    py_files = [
        os.path.join(root_dir, "app.py"),
        os.path.join(root_dir, "test_modules.py"),
    ] + sorted(glob.glob(os.path.join(root_dir, "modules", "*.py")))
    for pf in py_files:
        py_compile.compile(pf, doraise=True)

    app_path = os.path.join(root_dir, "app.py")
    with open(app_path, "r", encoding="utf-8") as af:
        app_src = af.read()

    assert "use_container_width" not in app_src, "Deprecated use_container_width found in app.py"
    app_no_style = re.sub(r"<style>.*?</style>", "", app_src, flags=re.S | re.I)
    assert not re.search(r"font-size\s*:", app_no_style, flags=re.I), "Inline font-size: outside <style> in app.py"

    # Zero blank-line + 4-space-indent CommonMark HTML code-block traps in app.py
    for m_html in re.finditer(r'st\.markdown\s*\(\s*f?"""(.*?)"""\s*,\s*unsafe_allow_html\s*=\s*True', app_src, flags=re.S):
        body = m_html.group(1)
        if "<style>" not in body:
            assert not re.search(r"\n[ \t]*\n[ ]{4,}<", body), f"CommonMark HTML trap near offset {m_html.start()}"

    brand_files = [
        app_path,
        os.path.join(root_dir, ".streamlit", "config.toml"),
    ] + sorted(glob.glob(os.path.join(root_dir, "modules", "*.py")))
    for bf in brand_files:
        with open(bf, "r", encoding="utf-8") as fh:
            btxt = fh.read()
        assert not re.search(r"prafful|created\s+by", btxt, flags=re.I), f"Forbidden branding text in {bf}"

    # ------------------------------------------------------------------
    # 2. SDS-PAGE Handcast Gel Recipe Calculator (modules/gel_recipe.py)
    # ------------------------------------------------------------------
    # Laemmli 4%-22% across 30%/40%/49.5% stocks, 4%/5% stacking, formats & 1-4 gels
    for pct in (4.0, 6.0, 7.5, 8.0, 10.0, 10.5, 11.0, 12.0, 15.0, 18.0, 20.0, 21.5, 22.0):
        for stock in (30.0, 40.0, 49.5):
            for stk_pct in (4.0, 5.0):
                for fmt_k in ("mini_0.75mm", "mini_1.0mm", "mini_1.5mm", "midi_1.0mm"):
                    fmt = GEL_FORMAT_PRESETS[fmt_k]
                    max_p = get_max_achievable_gel_pct("laemmli", stock)
                    if pct > max_p + 1e-6:
                        try:
                            calculate_sds_gel_recipe(
                                pct, "laemmli", stock, fmt["resolving_vol_ml"], fmt["stacking_vol_ml"],
                                stacking_pct=stk_pct, num_gels=2,
                            )
                            raise AssertionError(f"Expected ValueError for Laemmli {pct}% with {stock}% stock")
                        except ValueError as ve:
                            assert "Impossible gel formulation" in str(ve)
                        rec = calculate_sds_gel_recipe(
                            pct, "laemmli", stock, fmt["resolving_vol_ml"], fmt["stacking_vol_ml"],
                            stacking_pct=stk_pct, num_gels=2, auto_upgrade_stock=True,
                        )
                        assert rec["stock_upgraded"] is True and rec["acrylamide_stock_pct"] > stock
                        assert any("auto-upgraded" in w.lower() for w in rec["warnings"])
                    else:
                        rec = calculate_sds_gel_recipe(
                            pct, "laemmli", stock, fmt["resolving_vol_ml"], fmt["stacking_vol_ml"],
                            stacking_pct=stk_pct, num_gels=2,
                        )
                        assert rec["stock_upgraded"] is False
                    for rows_key, tot_key in (("resolving_rows", "resolving_total_vol_ml"), ("stacking_rows", "stacking_total_vol_ml")):
                        rows = rec[rows_key]
                        tot = rec[tot_key]
                        assert abs(sum(r["volume_ml"] for r in rows) - tot) < 1e-4
                        assert abs(sum(r["Volume (mL)"] for r in rows) - tot) < 1e-4
                        assert all(r["volume_ml"] >= -1e-9 for r in rows)
                        assert all(col in rows[0] for col in RECIPE_DISPLAY_COLUMNS)

    # Tricine 10%-20% across 30%/40%/49.5% stocks & glycerol toggles (including 10% and 13%)
    for pct in (10.0, 12.0, 14.0, 16.0, 18.0, 20.0):
        for stock in (30.0, 40.0, 49.5):
            for inc_glyc, glyc_pct in ((None, 10.0), (True, 10.0), (True, 13.0), (False, 10.0)):
                eff_g = (pct >= 16.0) if inc_glyc is None else inc_glyc
                max_p = get_max_achievable_gel_pct("tricine", stock, include_glycerol=eff_g, glycerol_pct=glyc_pct)
                if pct > max_p + 1e-6:
                    try:
                        calculate_sds_gel_recipe(
                            pct, "tricine", stock, include_glycerol=inc_glyc, glycerol_pct=glyc_pct
                        )
                        raise AssertionError(f"Expected ValueError for Tricine {pct}% w/ {stock}% stock")
                    except ValueError:
                        pass
                    rec = calculate_sds_gel_recipe(
                        pct, "tricine", stock, include_glycerol=inc_glyc, glycerol_pct=glyc_pct,
                        auto_upgrade_stock=True,
                    )
                    assert rec["stock_upgraded"] is True and rec["acrylamide_stock_pct"] > stock
                else:
                    rec = calculate_sds_gel_recipe(
                        pct, "tricine", stock, include_glycerol=inc_glyc, glycerol_pct=glyc_pct
                    )
                assert abs(sum(r["volume_ml"] for r in rec["resolving_rows"]) - rec["resolving_total_vol_ml"]) < 1e-4
                assert abs(sum(r["volume_ml"] for r in rec["stacking_rows"]) - rec["stacking_total_vol_ml"]) < 1e-4
                assert all(r["volume_ml"] >= -1e-9 for r in rec["resolving_rows"])

    # Textbook Laemmli 12% (10 mL, 30% stock) & gel-%-tuned TEMED + <10 kDa Tricine advisory + text sheet
    r_12 = calculate_sds_gel_recipe(12.0, "laemmli", 30.0, resolving_vol_ml=10.0, stacking_vol_ml=5.0, target_mw_kda=5.5)
    rmap = {r["Reagent"]: r["volume_ml"] for r in r_12["resolving_rows"]}
    assert abs(rmap["Acrylamide/Bis 30% (37.5:1)"] - 4.0) < 1e-4
    assert abs(rmap["Tris-HCl (pH 8.8)"] - 2.5) < 1e-4
    assert abs(rmap["SDS"] - 0.1) < 1e-4 and abs(rmap["APS (Ammonium Persulfate)"] - 0.1) < 1e-4
    assert abs(rmap["TEMED"] - 0.004) < 1e-6 and abs(rmap["ddH2O (Ultrapure Water)"] - 3.296) < 1e-4
    assert any("Tricine" in w and "10 kDa" in w for w in r_12["warnings"])
    assert recommend_gel_recipe_params(5.5)["system"] == "tricine"

    # Schägger (2006) Table 1 16% Tricine 30 mL (49.5% T stock, 10% glycerol) & dual buffer text export
    r_tri16 = calculate_sds_gel_recipe(16.0, "tricine", 49.5, resolving_vol_ml=30.0, stacking_vol_ml=12.0)
    tmap = {r["Reagent"]: r["volume_ml"] for r in r_tri16["resolving_rows"]}
    assert abs(tmap["3× Tricine Gel Buffer (pH 8.45)"] - 10.0) < 1e-4
    assert abs(tmap["Glycerol (anhydrous)"] - 3.0) < 1e-4
    assert abs(tmap["APS (Ammonium Persulfate)"] - 0.15) < 1e-4
    assert abs(tmap["TEMED"] - 0.015) < 1e-6

    laem_txt = format_gel_recipe_text(r_12)
    tri_txt = format_gel_recipe_text(r_tri16)
    assert "RESOLVING GEL" in laem_txt and "STACKING GEL" in laem_txt and "NEUROTOXIN" in laem_txt
    assert "Anode (Lower)" in tri_txt and "Cathode (Upper)" in tri_txt

    for bad_kwargs in (
        {"gel_pct": 3.0, "system": "laemmli"},
        {"gel_pct": 8.0, "system": "tricine"},
        {"gel_pct": 12.0, "num_gels": 0},
        {"gel_pct": 12.0, "resolving_vol_ml": -1.0},
        {"gel_pct": 12.0, "system": "invalid_sys"},
    ):
        try:
            calculate_sds_gel_recipe(**bad_kwargs)
            raise AssertionError(f"Expected ValueError for {bad_kwargs}")
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # 3. 3D Protein Structure Viewer & Parser (modules/protein_3d.py)
    # ------------------------------------------------------------------
    lyso_m = analyze_all_parameters(LYSOZYME_MATURE_SEQ)
    lyso_ss = predict_secondary_structure(LYSOZYME_MATURE_SEQ)
    # Warm up Plotly scatter3d validator schema once so standalone timings reflect true runtime
    _ = build_3d_biophysical_backbone_figure(LYSOZYME_MATURE_SEQ, lyso_ss, lyso_m, color_by="ss")
    for cmode in ("ss", "hydrophobicity", "charge"):
        t0 = time.perf_counter()
        fig_lyso_3d = build_3d_biophysical_backbone_figure(
            LYSOZYME_MATURE_SEQ, lyso_ss, lyso_m, color_by=cmode, protein_name="Lysozyme"
        )
        dt_lyso = time.perf_counter() - t0
        assert dt_lyso < 0.30, f"Offline 3D Lysozyme ({cmode}) too slow: {dt_lyso:.3f}s"
        assert len(fig_lyso_3d.data) >= 3
        assert SCHEMATIC_3D_DISCLAIMER in str(fig_lyso_3d.layout.title.subtitle.text)

    titin_34k = (LYSOZYME_MATURE_SEQ * 267)[:34350]
    titin_ss = predict_secondary_structure(titin_34k)
    titin_m = analyze_all_parameters(titin_34k)
    for cmode in ("ss", "hydrophobicity", "charge"):
        t0 = time.perf_counter()
        fig_titin_3d = build_3d_biophysical_backbone_figure(
            titin_34k, titin_ss, titin_m, color_by=cmode, protein_name="Titin"
        )
        dt_titin = time.perf_counter() - t0
        assert dt_titin < 0.30, f"Offline 3D Titin 34,350 aa ({cmode}) too slow: {dt_titin:.3f}s"
        assert SCHEMATIC_3D_DISCLAIMER in str(fig_titin_3d.layout.title.subtitle.text)

    def _pdb_atom(serial: int, name: str, resn: str, chain: str, resi: int, x: float, y: float, z: float, bfac: float, elem: str) -> str:
        return f"ATOM  {serial:5d} {name:^4s} {resn:>3s} {chain:1s}{resi:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfac:6.2f}          {elem:>2s}"

    af2_pdb = "\n".join([
        "HEADER    ALPHAFOLD MONOMER V2.0 PREDICTION",
        _pdb_atom(1, "CA", "LYS", "A", 1, 1.0, 2.0, 3.0, 95.4, "C"),
        _pdb_atom(2, "CA", "VAL", "A", 2, 4.8, 2.0, 3.0, 82.6, "C"),
        _pdb_atom(3, "CA", "PHE", "A", 3, 8.6, 2.0, 3.0, 61.0, "C"),
        "END",
    ]).encode("utf-8")
    p_af2 = parse_uploaded_structure_file(af2_pdb, "af2_model.pdb")
    assert p_af2["valid"] and p_af2["looks_like_plddt"] is True and p_af2["plddt_scale"] == "0-100"
    assert abs(p_af2["mean_plddt"] - 79.67) < 0.05

    esm_pdb = "\n".join([
        _pdb_atom(1, "CA", "ALA", "A", 1, 1.0, 2.0, 3.0, 0.94, "C"),
        _pdb_atom(2, "CA", "GLY", "A", 2, 4.8, 2.0, 3.0, 0.78, "C"),
        _pdb_atom(3, "CA", "LEU", "A", 3, 8.6, 2.0, 3.0, 0.62, "C"),
        "END",
    ]).encode("utf-8")
    p_esm = parse_uploaded_structure_file(esm_pdb, "esmfold_design.pdb")
    assert p_esm["valid"] and p_esm["looks_like_plddt"] is True and p_esm["plddt_scale"] == "0-1"
    assert abs(p_esm["mean_plddt"] - 78.0) < 0.1 and any("0–1 pLDDT" in w for w in p_esm["warnings"])

    rfd_pdb = "\n".join([
        _pdb_atom(1, "CA", "ALA", "A", 1, 1.0, 2.0, 3.0, 0.00, "C"),
        _pdb_atom(2, "CA", "GLU", "A", 2, 4.8, 2.0, 3.0, 0.00, "C"),
        _pdb_atom(3, "CA", "LYS", "A", 3, 8.6, 2.0, 3.0, 0.00, "C"),
        "END",
    ]).encode("utf-8")
    p_rfd = parse_uploaded_structure_file(rfd_pdb, "rfdiffusion_backbone.pdb")
    assert p_rfd["valid"] and p_rfd["looks_like_plddt"] is False and p_rfd["plddt_scale"] is None and p_rfd["mean_plddt"] is None

    xray_pdb = "\n".join([
        "EXPDTA    X-RAY DIFFRACTION",
        "REMARK   2 RESOLUTION.    1.45 ANGSTROMS.",
        "SSBOND   1 CYS A    6    CYS A  127",
        _pdb_atom(1, "CA", "HIS", "A", 1, 1.0, 2.0, 3.0, 24.5, "C"),
        _pdb_atom(2, "CA", "CYS", "A", 2, 4.8, 2.0, 3.0, 31.2, "C"),
        "HETATM    3 FE   HEM A 201       5.000   5.000   5.000  1.00 18.00          FE",
        "HETATM    4 ZN    ZN A 202       9.000   5.000   5.000  1.00 15.00          ZN",
        "HETATM    5  O   HOH A 301      12.000   5.000   5.000  1.00 35.00           O",
        "END",
    ]).encode("utf-8")
    p_xray = parse_uploaded_structure_file(xray_pdb, "holo_enzyme.pdb")
    assert p_xray["valid"] and p_xray["looks_like_plddt"] is False
    assert p_xray["resolution_angstrom"] == 1.45 and p_xray["disulfide_bonds"] == 1
    assert "HEM" in p_xray["cofactors"] and "FE" in p_xray["metal_ions"] and "ZN" in p_xray["metal_ions"]
    assert "HOH" not in p_xray["hetatm_ligands"]

    af3_cif = (
        "data_AF3_BINDER\n"
        "_struct.title 'AlphaFold3 Designed Binder'\n"
        "_ma_qa_metric_global.metric_value 91.5\n"
        "loop_\n"
        "_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n_atom_site.label_atom_id\n"
        "_atom_site.label_comp_id\n_atom_site.label_asym_id\n_atom_site.auth_seq_id\n"
        "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n_atom_site.B_iso_or_equiv\n"
        "ATOM 1 C CA ALA A 1 1.0 2.0 3.0 92.0\n"
        "ATOM 2 C O5' GLU A 2 4.8 2.0 3.0 91.0\n"
        "HETATM 3 ZN ZN ZN A 101 8.0 2.0 3.0 88.0\n"
    ).encode("utf-8")
    p_cif = parse_uploaded_structure_file(af3_cif, "binder_af3.cif")
    assert p_cif["valid"] and p_cif["format"] == "cif" and p_cif["looks_like_plddt"] is True
    assert "ZN" in p_cif["metal_ions"] and p_cif["residue_count"] == 2

    p_gz = parse_uploaded_structure_file(gzip.compress(af2_pdb), "af2_model.pdb.gz")
    assert p_gz["valid"] and p_gz["format"] == "pdb" and p_gz["looks_like_plddt"] is True

    for bad_bytes, bad_name in (
        (b"", "empty.pdb"),
        (b"plain text", "notes.txt"),
        (b"\x00\x01\x02binary", "fake.pdb"),
        (b"REMARK ONLY NO ATOMS\nEND\n", "no_atoms.pdb"),
        (b"not a valid gzip stream", "corrupt.pdb.gz"),
        (b"A" * (10 * 1024 * 1024 + 5), "oversize.pdb"),
    ):
        r_bad = parse_uploaded_structure_file(bad_bytes, bad_name)
        assert r_bad["valid"] is False and bool(r_bad["error"]), f"Should reject malformed upload {bad_name}"

    # 3Dmol HTML escaping against </script>, HTML comments, quotes, backticks & line separators
    hostile_pdb = (
        "TITLE     HOSTILE </script><script>alert('xss')</script> <!-- comment --> \"dbl\" 'sgl' `bt`\u2028\u2029\n"
        + af2_pdb.decode("utf-8")
    )
    html_esc = build_3dmol_viewer_html(hostile_pdb, fmt="pdb", style="cartoon", color_scheme="plddt", bg_color="javascript:alert(1)")
    assert len(re.findall(r"</script>", html_esc, flags=re.I)) == 1
    assert "<!-- comment -->" not in html_esc and "javascript:alert" not in html_esc
    m_cfg = re.search(r"var CFG = (\{.*?\});\n", html_esc, flags=re.S)
    assert m_cfg is not None
    parsed_cfg = json.loads(m_cfg.group(1).replace("<\\/", "</").replace("<\\!--", "<!--"))
    assert "</script>" in parsed_cfg["structure"] and parsed_cfg["bgColor"] == "#F8FBF9"
    for st_name in SUPPORTED_3DMOL_STYLES:
        for cs_name in SUPPORTED_3DMOL_COLOR_SCHEMES:
            h_combo = build_3dmol_viewer_html(af2_pdb.decode("utf-8"), style=st_name, color_scheme=cs_name)
            assert "$3Dmol" in h_combo

    # Mocked AlphaFold -> RCSB fallback (including >4 MB skip), prefer_source='rcsb', and offline None
    assert normalize_uniprot_accession("P04637-2") == "P04637"

    class _Mock3DResp:
        def __init__(self, code: int, text: str = "", json_obj=None, headers=None):
            self.status_code = code
            self.text = text
            self.content = text.encode("utf-8") if isinstance(text, str) else b""
            self._json = json_obj
            self.headers = headers or {}

        def json(self):
            return self._json

    def _mock_3d_fetch(url: str, **kwargs):
        if "alphafold.ebi.ac.uk/api/prediction/P04637" in url:
            return _Mock3DResp(404)
        if "search.rcsb.org" in url:
            return _Mock3DResp(200, json_obj={"result_set": [{"identifier": "9BIG"}, {"identifier": "1TUP"}]})
        if "files.rcsb.org/download/9BIG.pdb" in url:
            return _Mock3DResp(200, text="ATOM", headers={"Content-Length": str(5 * 1024 * 1024)})
        if "files.rcsb.org/download/1TUP.pdb" in url:
            return _Mock3DResp(200, text=xray_pdb.decode("utf-8"))
        return _Mock3DResp(404)

    with patch("requests.get", side_effect=_mock_3d_fetch):
        fb_3d = fetch_3d_structure_for_uniprot("P04637-2")
        assert fb_3d is not None and fb_3d["source"] == "RCSB PDB" and fb_3d["pdb_id"] == "1TUP"
        assert fb_3d["accession"] == "P04637" and "ZN" in fb_3d["metal_ions"]

    with patch("requests.get", side_effect=_requests.ConnectionError("Simulated offline")):
        assert fetch_3d_structure_for_uniprot("P00698") is None
    assert fetch_3d_structure_for_uniprot("NOT_VALID_ID") is None

    # ------------------------------------------------------------------
    # 4. AI-Designed / De Novo Protein Discrepancy Fixes (#95, #99)
    # ------------------------------------------------------------------
    seq_2kl8 = "MEMDIRFRGDDLEAFEKALKEMIRQARKFAGTVTYTLDGNDLEIRITGVPEQVRKELAKEAERLAKEFNITVTYTIRLEHHHHHH"
    m_2kl8 = analyze_all_parameters(seq_2kl8)
    prof_2kl8 = infer_protein_biochemical_profile(
        sequence=seq_2kl8, protein_name="De novo ferredoxin fold protein Di-I_5", metrics=m_2kl8, tag_detected="6xHis"
    )
    assert "fe_s_cluster" not in prof_2kl8["categories"]
    assert prof_2kl8["is_o2_sensitive"] is False and prof_2kl8["avoid_imidazole_imac"] is False
    assert prof_2kl8["tag_conflict"] is None

    # Name-only "Spinach ferredoxin" with empty sequence STILL flags [Fe-S] & O2 sensitivity
    prof_spinach = infer_protein_biochemical_profile(protein_name="Spinach ferredoxin", sequence="", tag_detected="6xHis")
    assert "fe_s_cluster" in prof_spinach["categories"] and prof_spinach["is_o2_sensitive"] is True
    assert prof_spinach["avoid_imidazole_imac"] is True and prof_spinach["tag_conflict"] is not None

    # 6D0T (12.6% aromatic folded de novo beta-barrel) -> NOT IDP, NOT membrane
    seq_6d0t = "MVDAAQYFPGTWEFRFRSSDGKEYRGTVEMQPRTPTEIEIRFKGQSSDGRPVEGRGSIEVRSPYEYRFEMQSSDGARWEGTLQVRSPDSVEVRFKSSDGREYSGEFRRQEG"
    m_6d0t = analyze_all_parameters(seq_6d0t)
    prof_6d0t = infer_protein_biochemical_profile(
        sequence=seq_6d0t, protein_name="De novo fluorescence-activating beta-barrel mFAP2b", metrics=m_6d0t
    )
    assert "idp_disordered" not in prof_6d0t["categories"]
    assert prof_6d0t["requires_detergent_micelles"] is False

    # Contrast: true low-aromatic polar IDP sequence still flags idp_disordered
    seq_idp = "EDKQSGNTPQEDKQSGNTPQ" * 5
    prof_idp = infer_protein_biochemical_profile(
        sequence=seq_idp, protein_name="Unnamed_Disordered_Chain", metrics=analyze_all_parameters(seq_idp)
    )
    assert "idp_disordered" in prof_idp["categories"]

    # 6X1K (transmembrane beta-barrel) -> requires_detergent_micelles=True + S200 micelle shift + Avoid HIC
    seq_6x1k = "MQDGPGTLDVFVAAGWNTDNTIEITGGATYQLSPYIMVKAGYGWNNSSLNRFEFGGGLQYKVTPDLEPYAWAGATYNTDNTLVPAAGAGFRYKVSPEVKLVVEYGWNNSSLQFLQAGLSYRIQP"
    m_6x1k = analyze_all_parameters(seq_6x1k)
    prof_6x1k = infer_protein_biochemical_profile(
        sequence=seq_6x1k, protein_name="De novo transmembrane beta-barrel TMB2.3", metrics=m_6x1k
    )
    assert prof_6x1k["requires_detergent_micelles"] is True
    wf_6x1k = recommend_full_purification_workflow(
        m_6x1k, 7.5, calculate_net_charge_at_ph(seq_6x1k, 7.5),
        tag_detected="Twin-Strep-tag", sequence=seq_6x1k,
        protein_name="De novo transmembrane beta-barrel TMB2.3", protein_profile=prof_6x1k,
    )
    assert "Superdex 200" in wf_6x1k["roadmap"][2]["title"]
    assert "Avoid" in wf_6x1k["all_methods"][4]["Suitability"]
    assert infer_protein_biochemical_profile(protein_name="Transcriptional regulatory protein OmpR")["requires_detergent_micelles"] is False

    # Zero-Trp/Tyr sequence (3Q7X, eps280 = 0) -> uv280_invisible=True + SEC warning + workflow alert
    seq_3q7x = "HEVLLRSTETGQFLRINPDGTVDGTRDRSDPGIQFQISPEGNGEVLLRSTETGQFLRINPDGTVDGTRDRSDPGIQFQISPEGNGEVLLRSTETGQFLRINPDGTVDGTRDRSDPGIQFQISPEG"
    m_3q7x = analyze_all_parameters(seq_3q7x)
    assert m_3q7x["extinction"]["molar_extinction_cystines"] == 0 and m_3q7x["extinction"]["abs_01_cystines"] == 0.0
    sim_3q7x = simulate_chromatogram_peaks(
        [{"label": "3Q7X", "mw_kda": m_3q7x["mw_kda"], "prot_eps01": m_3q7x["extinction"]["abs_01_cystines"]}],
        SEC_COLUMNS["s75_increase_10_300"],
    )
    pk_3q7x = sim_3q7x["peaks"][0]
    assert pk_3q7x["uv280_invisible"] is True and pk_3q7x["peak_height_mau"] > 0 and "214/205 nm" in pk_3q7x["note"]
    wf_3q7x = recommend_full_purification_workflow(
        m_3q7x, 7.4, calculate_net_charge_at_ph(seq_3q7x, 7.4),
        sequence=seq_3q7x, protein_name="De novo symmetric beta-trefoil Symfoil-4P",
    )
    assert any("UV-280 Invisible" in a for a in wf_3q7x["alerts"])
    assert simulate_chromatogram_peaks([{"label": "DefaultEps", "mw_kda": 15.0}], SEC_COLUMNS["s75_increase_10_300"])["peaks"][0]["uv280_invisible"] is False
    assert simulate_chromatogram_peaks([{"label": "RNP", "mw_kda": 40.0, "prot_mw_kda": 15.0, "na_mw_kda": 25.0, "na_type": "ssrna", "prot_eps01": 0.0}], SEC_COLUMNS["s75_increase_10_300"])["peaks"][0]["uv280_invisible"] is False

    # Glu/Lys-rich MPNN/de novo II>=40 caveat vs BSA negative control (natural II=40.11)
    seq_ek_mpnn = "EEVVVEIRIRVQREEKVRRLIKRILEEVKRESNSVEVHVETRKRNGEVEVHVRIRHDDKETIERLVERILREIKKLDKNSEVEVRTTTKR"
    m_ek = analyze_all_parameters(seq_ek_mpnn)
    assert m_ek["instability_index"] >= 40.0
    wf_ek_raw = recommend_full_purification_workflow(
        m_ek, 7.4, calculate_net_charge_at_ph(seq_ek_mpnn, 7.4),
        sequence=seq_ek_mpnn, protein_name="Unnamed_Design",
    )
    assert any("De Novo / ProteinMPNN Caveat" in a for a in wf_ek_raw["alerts"])

    bsa_seq = CLASSROOM_PRESET_PROTEINS["bsa"]["sequence"]
    m_bsa = analyze_all_parameters(bsa_seq)
    assert m_bsa["instability_index"] >= 40.0
    wf_bsa = recommend_full_purification_workflow(
        m_bsa, 7.4, calculate_net_charge_at_ph(bsa_seq, 7.4),
        sequence=bsa_seq, protein_name="Bovine Serum Albumin",
    )
    assert any("Cold-Room" in a for a in wf_bsa["alerts"])
    assert not any("De Novo / ProteinMPNN Caveat" in a for a in wf_bsa["alerts"])

    # ------------------------------------------------------------------
    # 5. Headless AppTest Audit of app.py (#91, #99)
    # ------------------------------------------------------------------
    # 5a. Default load: 0 exceptions, 0 outbound HTTP calls, instant offline 3D Plotly scatter3d present
    def _forbid_http(*args, **kwargs):
        raise AssertionError(f"Unexpected outbound HTTP call on default load: args={args}")

    with patch("requests.get", side_effect=_forbid_http), \
         patch("requests.post", side_effect=_forbid_http), \
         patch("requests.Session.request", side_effect=_forbid_http):
        at = AppTest.from_file(app_path).run(timeout=30)

    assert len(at.exception) == 0, f"Default load exceptions: {at.exception}"
    mode_radios = [r for r in at.radio if r.key == "viewer_3d_mode_radio"]
    assert len(mode_radios) == 1 and "Offline" in mode_radios[0].value
    found_offline_3d = False
    for ch in at.get("plotly_chart"):
        spec = json.loads(ch.proto.spec)
        if any(tr.get("type") == "scatter3d" for tr in spec.get("data", [])):
            assert SCHEMATIC_3D_DISCLAIMER in str(spec.get("layout", {}).get("title", {}).get("subtitle", {}).get("text", ""))
            found_offline_3d = True
            break
    assert found_offline_3d, "Offline 3D scatter3d Plotly figure not found on default load!"
    rendered_md = "\n".join(m.value for m in at.markdown if "<style>" not in m.value)
    assert "Schematic 3D Note:" in rendered_md
    assert len(re.findall(r"font-size\s*:", rendered_md, flags=re.I)) == 0
    assert not any("UV-280 Blind" in w.value for w in at.warning)

    # 5b. Gel Recipe Pop-Out on Lysozyme (Laemmli + switch to Tricine)
    gel_btn = [b for b in at.button if b.key == "btn_open_gel_recipe_dialog"][0]
    gel_btn.click().run(timeout=30)
    assert len(at.exception) == 0
    gel_dfs = [df.value for df in at.dataframe if "Pipette Display" in list(df.value.columns)]
    assert len(gel_dfs) == 2
    assert abs(float(gel_dfs[0]["Volume (mL)"].sum()) - 7.5) < 1e-3
    assert abs(float(gel_dfs[1]["Volume (mL)"].sum()) - 2.5) < 1e-3

    # Switch to Tricine inside the pop-out dialog (re-trigger dialog open in headless AppTest)
    [r for r in at.radio if r.key == "gel_calc_system_radio"][0].set_value("tricine")
    [b for b in at.button if b.key == "btn_open_gel_recipe_dialog"][0].click()
    at.run(timeout=30)
    assert len(at.exception) == 0
    tri_dfs = [df.value for df in at.dataframe if "Pipette Display" in list(df.value.columns)]
    assert len(tri_dfs) == 2 and any("3× Tricine Gel Buffer" in str(x) for x in tri_dfs[0]["Reagent"].tolist())

    # 5c. <10 kDa zero-Trp/Tyr peptide -> UV-280 Blind SEC warning + Auto-Tricine Gel Recipe Pop-Out (and W clears warning)
    at_pep = AppTest.from_file(app_path).run(timeout=30)
    at_pep.sidebar.radio[0].set_value("🔤 Paste Amino Acid / FASTA").run(timeout=30)
    at_pep.sidebar.text_area[0].set_value(">DeNovo_UVBlind_Peptide\nAEKLAEKLAEKLAEKLAEKLAEKLAEKL\n").run(timeout=30)
    at_pep.sidebar.button[0].click().run(timeout=30)
    assert len(at_pep.exception) == 0
    assert any("UV-280 Blind" in w.value for w in at_pep.warning)

    [b for b in at_pep.button if b.key == "btn_open_gel_recipe_dialog"][0].click().run(timeout=30)
    assert len(at_pep.exception) == 0
    pep_sys = [r for r in at_pep.radio if r.key == "gel_calc_system_radio"]
    assert len(pep_sys) == 1 and pep_sys[0].value == "tricine"
    assert any("Auto-Selected Tricine-SDS-PAGE" in i.value for i in at_pep.info)
    pep_dfs = [df.value for df in at_pep.dataframe if "Pipette Display" in list(df.value.columns)]
    assert len(pep_dfs) == 2 and abs(float(pep_dfs[0]["Volume (mL)"].sum()) - 7.5) < 1e-3

    # Appending single C-terminal Trp clears UV-280 Blind SEC warning
    at_pep.sidebar.text_area[0].set_value(">DeNovo_TrpTag_Peptide\nAEKLAEKLAEKLAEKLAEKLAEKLAEKLW\n").run(timeout=30)
    at_pep.sidebar.button[0].click().run(timeout=30)
    assert len(at_pep.exception) == 0
    assert not any("UV-280 Blind" in w.value for w in at_pep.warning)

    # 5d. Live 3D Viewer: 0 HTTP before Fetch click, mocked AlphaFold fetch + cache after click
    with patch("requests.get", side_effect=_forbid_http), \
         patch("requests.Session.request", side_effect=_forbid_http):
        at_live = AppTest.from_file(app_path).run(timeout=30)
        [r for r in at_live.radio if r.key == "viewer_3d_mode_radio"][0].set_value("🌐 Live AlphaFold / PDB 3D").run(timeout=30)
    assert len(at_live.exception) == 0

    def _mock_live_af(url: str, **kwargs):
        if "alphafold.ebi.ac.uk/api/prediction/P00698" in url:
            return _Mock3DResp(200, json_obj=[{
                "entryId": "AF-P00698-F1",
                "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P00698-F1-model_v4.pdb",
                "globalMetricValue": 93.8,
                "latestVersion": 4,
                "uniprotDescription": "Lysozyme C",
            }])
        if "AF-P00698-F1-model_v4.pdb" in url:
            return _Mock3DResp(200, text=af2_pdb.decode("utf-8"))
        return _Mock3DResp(404)

    with patch("requests.get", side_effect=_mock_live_af):
        [b for b in at_live.button if b.key == "btn_fetch_3d_structure"][0].click().run(timeout=30)
    assert len(at_live.exception) == 0
    assert ("P00698", "alphafold") in at_live.session_state["structure_3d_cache"]
    assert at_live.session_state["structure_3d_cache"][("P00698", "alphafold")]["mean_plddt"] == 93.8

    # 5e. Feedback Modal: open & submit cleanly with guaranteed cleanup of data/user_feedback_reports.jsonl
    fb_log_path = os.path.join(root_dir, "data", "user_feedback_reports.jsonl")
    fb_existed_before = os.path.exists(fb_log_path)
    fb_prior_bytes = None
    if fb_existed_before:
        with open(fb_log_path, "rb") as fh_prior:
            fb_prior_bytes = fh_prior.read()

    try:
        at_fb = AppTest.from_file(app_path).run(timeout=30)
        [b for b in at_fb.sidebar.button if b.key == "btn_open_feedback_dialog"][0].click().run(timeout=30)
        assert len(at_fb.exception) == 0
        [ta for ta in at_fb.text_area if ta.key == "fb_description_input"][0].set_value(
            "Suite 14 verification feedback report (automatically cleaned up)."
        )
        [ti for ti in at_fb.text_input if ti.key == "fb_reference_input"][0].set_value("UniProt P00698")
        [b for b in at_fb.button if b.key == "fb_submit_btn"][0].click()
        [b for b in at_fb.sidebar.button if b.key == "btn_open_feedback_dialog"][0].click()
        at_fb.run(timeout=30)
        assert len(at_fb.exception) == 0
        assert "last_feedback_report" in at_fb.session_state
        assert at_fb.session_state["last_feedback_report"]["diagnostic_context"]["uniprot_or_accession_id"] == "P00698"
        assert any("Anonymous report captured" in s.value for s in at_fb.success)
    finally:
        if fb_existed_before and fb_prior_bytes is not None:
            with open(fb_log_path, "wb") as fh_restore:
                fh_restore.write(fb_prior_bytes)
        elif os.path.exists(fb_log_path):
            os.remove(fb_log_path)
    assert os.path.exists(fb_log_path) == fb_existed_before

    # 5f. Hot-reload recovery for modules.gel_recipe & modules.protein_3d
    delattr(_sys.modules["modules.gel_recipe"], "calculate_sds_gel_recipe")
    delattr(_sys.modules["modules.protein_3d"], "build_3d_biophysical_backbone_figure")
    at_hr = AppTest.from_file(app_path).run(timeout=30)
    assert len(at_hr.exception) == 0, f"Hot-reload recovery failed for new modules: {at_hr.exception}"
    assert hasattr(_sys.modules["modules.gel_recipe"], "calculate_sds_gel_recipe")
    assert hasattr(_sys.modules["modules.protein_3d"], "build_3d_biophysical_backbone_figure")


def test_pipeline() -> None:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    test_protparam_golden_reference_and_sequences(root_dir)
    test_chemical_mw_stoichiometry_and_liquids()
    test_all_seven_purification_methods()
    test_sec_hydrodynamics_markers_and_benchmark(root_dir)
    test_sds_page_monotonicity_and_live_app_ui(root_dir)
    test_natural_language_disambiguation_and_ui(root_dir)
    test_tree_of_life_nl_parser_and_deterministic_ranking()
    test_soluble_aggregate_void_peak_and_ui(root_dir)
    test_secondary_structure_engine_benchmarks_and_ui(root_dir)
    test_giant_multidomain_titin_and_ui(root_dir)
    test_interactive_sec_column_visualizer_and_intuitive_plots(root_dir)
    test_biochemical_protein_intelligence_and_metalloprotein_safety(root_dir)
    test_all_5_input_methods_hardening_and_minimal_chrome(root_dir)
    test_gel_recipes_3d_viewer_feedback_and_ai_proteins(root_dir)
    print("\n=== ALL 14 GOLDEN PROTPARAM, STOICHIOMETRY, 7-METHOD, SEC, SDS-PAGE, NL PARSER, SECONDARY STRUCTURE, GIANT TITIN, INTERACTIVE SEC COLUMN, BIOCHEMICAL INTELLIGENCE, ALL-5-INPUT-METHOD, GEL RECIPES, 3D VIEWER, FEEDBACK & AI-PROTEIN TESTS PASSED! ===")


if __name__ == "__main__":
    test_pipeline()
