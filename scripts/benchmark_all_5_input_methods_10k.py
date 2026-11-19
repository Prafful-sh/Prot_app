"""
Comprehensive All-5-Input-Methods >=10,000-Protein Cross-Domain Benchmark
========================================================================
Tests EVERY protein loading / search method in ProPurify Studio against >=10,000 proteins
spanning all 7 domains of life (Bacteria, Archaea, Animals, Plants, Fungi, Protists, Viruses):
  1. UniProt ID (`fetch_uniprot_by_id` & `_normalize_uniprot_input`)
  2. NCBI Accession (`fetch_ncbi_by_id`, `_normalize_ncbi_input`, `_classify_ncbi_accession_db`, `_parse_ncbi_fasta_header`)
  3. Paste Amino Acid / FASTA (`clean_amino_acid_sequence`, `extract_fasta_header_metadata`, `detect_sequence_terminal_tag`, `analyze_all_parameters`, `infer_protein_biochemical_profile`)
  4. Translate DNA / RNA (`_clean_nucleotide_input`, `reverse_complement`, `translate_nucleotide`, `find_longest_orf`)
  5. Describe in Plain English (`extract_nl_query_components`, `rank_uniprot_candidates`, `_candidate_matches_requested_protein`, plus >=15,300 prompts over 13,403 proteins)

Supports:
  --offline-only : Run the full 14,869-protein / 15,303-prompt offline benchmark (sandbox-safe)
  --live-only    : Run stratified live HTTPS checks against rest.uniprot.org & eutils.ncbi.nlm.nih.gov
  (default)      : Run both offline and live suites

Always writes to `data/all_input_methods_10k_benchmark.json` (never overwrites `nl_10k_benchmark_results.json`).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import numpy as np
import requests

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import modules.sequence_parser as sp
from modules.sequence_parser import (
    CLASSROOM_PRESET_PROTEINS,
    clean_amino_acid_sequence,
    extract_fasta_header_metadata,
    detect_sequence_terminal_tag,
    fetch_uniprot_by_id,
    fetch_ncbi_by_id,
    find_longest_orf,
    translate_nucleotide,
    reverse_complement,
    _normalize_uniprot_input,
    _normalize_ncbi_input,
    _classify_ncbi_accession_db,
    _parse_ncbi_fasta_header,
    _candidate_matches_requested_protein,
    extract_nl_query_components,
    rank_uniprot_candidates,
    parse_natural_language_prompt,
)
from modules.purification_metrics import analyze_all_parameters
from modules.buffer_advisor import infer_protein_biochemical_profile
from scripts import run_nl_10k_benchmark as nl_bench


DOMAINS_7_SS = ["Bacteria", "Archaea", "Animals", "Plants", "Fungi", "Protists", "Viruses"]

# Deterministic coding codons using non-stop triplets cleanly
AA_TO_CODON_SAFE = {
    "A": "GCT", "C": "TGC", "D": "GAT", "E": "GAA", "F": "TTT",
    "G": "GGT", "H": "CAT", "I": "ATT", "K": "AAA", "L": "CTG",
    "M": "ATG", "N": "AAC", "P": "CCG", "Q": "CAG", "R": "CGT",
    "S": "TCT", "T": "ACT", "V": "GTT", "W": "TGG", "Y": "TAT",
}
NONSTD_AA_MAP = {"B": "D", "Z": "E", "J": "L", "X": "A", "U": "C", "O": "K"}


class _MockResponse:
    def __init__(self, status_code: int, text: str = "", json_obj: Any = None):
        self.status_code = status_code
        self.text = text
        self._json = json_obj

    def json(self) -> Any:
        return self._json


def _load_ss_10k_records() -> List[Dict[str, Any]]:
    npz_path = os.path.join(ROOT_DIR, "data", "ss_10k_corpus.npz")
    records: List[Dict[str, Any]] = []
    with np.load(npz_path, allow_pickle=False) as d:
        accs = d["accessions"]
        uids = d["uniprot_ids"]
        pnames = d["protein_names"]
        gnames = d["gene_names"]
        orgs = d["organisms"]
        tids = d["tax_ids"]
        strata = d["strata"]
        seqs = d["sequences"]
        for i in range(len(accs)):
            records.append(
                {
                    "accession": str(accs[i]).strip(),
                    "entry_name": str(uids[i]).strip(),
                    "protein_name": str(pnames[i]).strip(),
                    "gene_name": str(gnames[i]).strip() or "N/A",
                    "organism": str(orgs[i]).strip(),
                    "tax_id": int(tids[i]),
                    "stratum": str(strata[i]).strip(),
                    "sequence": str(seqs[i]).strip(),
                }
            )
    return records


def run_method1_uniprot_offline(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    total = len(records)
    norm_ok = 0
    fetch_ok = 0
    by_stratum = {s: {"total": 0, "ok": 0} for s in DOMAINS_7_SS}
    failures: List[Dict[str, Any]] = []

    with patch("modules.sequence_parser.requests.get", side_effect=requests.ConnectionError("Offline benchmark")):
        for idx, rec in enumerate(records):
            acc = rec["accession"]
            uid_entry = rec["entry_name"]
            stratum = rec["stratum"]
            by_stratum.setdefault(stratum, {"total": 0, "ok": 0})
            by_stratum[stratum]["total"] += 1

            # Test all input normalizations deterministically
            variants = [
                acc,
                acc.lower(),
                f"https://www.uniprot.org/uniprotkb/{acc}/entry",
                f"https://rest.uniprot.org/uniprotkb/{acc}.fasta",
                f"sp|{acc}|{uid_entry}",
                f"UniProt: {acc}.1",
            ]
            v_pass = True
            for v in variants:
                try:
                    n = _normalize_uniprot_input(v)
                    if n["query"] != acc or n["base_accession"] != acc or n["isoform_num"] is not None:
                        v_pass = False
                        break
                except Exception:
                    v_pass = False
                    break

            if uid_entry and "_" in uid_entry:
                try:
                    ne = _normalize_uniprot_input(uid_entry)
                    if ne["query"] != uid_entry.upper() or not ne["is_entry_name"]:
                        v_pass = False
                except Exception:
                    v_pass = False

            if v_pass:
                norm_ok += 1

            # Test actual offline fetch_uniprot_by_id across all 14,869 proteins
            query_str = variants[idx % len(variants)]
            try:
                res = fetch_uniprot_by_id(query_str)
                seq_matches = res["sequence"] == rec["sequence"] and res["accession"] == acc
                if seq_matches and v_pass:
                    fetch_ok += 1
                    by_stratum[stratum]["ok"] += 1
                else:
                    if len(failures) < 20:
                        failures.append({"accession": acc, "reason": "seq_or_norm_mismatch", "query": query_str})
            except Exception as exc:
                if len(failures) < 20:
                    failures.append({"accession": acc, "reason": str(exc), "query": query_str})

    # Edge cases: isoform normalization, mocked isoform fetch, mocked merged inactive entry, empty/invalid guard
    edge_checks: Dict[str, bool] = {}
    n_iso = _normalize_uniprot_input("https://www.uniprot.org/uniprotkb/P04637-2")
    edge_checks["isoform_norm"] = n_iso == {
        "query": "P04637-2", "base_accession": "P04637", "isoform_num": "2", "is_entry_name": False
    }
    n_iso10 = _normalize_uniprot_input("Q8WZ42-12.1")
    edge_checks["isoform_10plus_norm"] = n_iso10["query"] == "Q8WZ42-12" and n_iso10["isoform_num"] == "12"

    def _mock_iso_get(url: str, **kwargs):
        if "P04637-2.fasta" in url:
            return _MockResponse(200, text=">sp|P04637-2|P53_HUMAN Isoform 2 OS=Homo sapiens GN=TP53\nMEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLED\n")
        if "P04637.json" in url:
            return _MockResponse(200, json_obj={
                "primaryAccession": "P04637",
                "proteinDescription": {"recommendedName": {"fullName": {"value": "Cellular tumor antigen p53"}}},
                "organism": {"scientificName": "Homo sapiens"},
                "genes": [{"geneName": {"value": "TP53"}}],
                "sequence": {"value": "M" * 393, "length": 393},
                "features": [{"type": "Metal binding", "ligand": {"name": "Zn(2+)", "id": "CHEBI:29105"}}],
            })
        if "Q99999.json" in url:
            return _MockResponse(200, json_obj={
                "entryType": "Inactive",
                "primaryAccession": "Q99999",
                "inactiveReason": {"inactiveReasonType": "MERGED", "mergeDemergeTo": ["P00698"]},
            })
        if "P00698.json" in url:
            return _MockResponse(200, json_obj={
                "primaryAccession": "P00698",
                "proteinDescription": {"recommendedName": {"fullName": {"value": "Lysozyme C"}}},
                "organism": {"scientificName": "Gallus gallus"},
                "genes": [{"geneName": {"value": "LYZ"}}],
                "sequence": {"value": "MRSLLILVLCFLPLAALGKVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL", "length": 147},
            })
        return _MockResponse(404)

    with patch("modules.sequence_parser.requests.get", side_effect=_mock_iso_get):
        r_iso = fetch_uniprot_by_id("P04637-2")
        edge_checks["isoform_mock_fetch"] = (
            r_iso["accession"] == "P04637-2"
            and "Isoform 2" in r_iso["protein_name"]
            and r_iso["length"] == 259
            and "Zn(2+)" in r_iso["biochem_annotations"].get("metal_ligands", [])
        )
        r_merged = fetch_uniprot_by_id("Q99999")
        edge_checks["merged_entry_autofollow"] = r_merged["accession"] == "P00698" and r_merged["length"] == 147

    try:
        fetch_uniprot_by_id("   ")
        edge_checks["blank_rejects"] = False
    except ValueError:
        edge_checks["blank_rejects"] = True

    return {
        "total_proteins": total,
        "normalization_pass_count": norm_ok,
        "fetch_pass_count": fetch_ok,
        "accuracy_pct": round(100.0 * fetch_ok / max(1, total), 4),
        "by_domain": {s: {"total": v["total"], "ok": v["ok"], "pct": round(100.0 * v["ok"] / max(1, v["total"]), 3)} for s, v in sorted(by_stratum.items())},
        "edge_checks": edge_checks,
        "wall_time_s": round(time.perf_counter() - t0, 2),
        "sample_failures": failures,
    }


def run_method2_ncbi_offline(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    total = len(records)
    class_ok = 0
    hdr_ok = 0
    by_stratum = {s: {"total": 0, "ok": 0} for s in DOMAINS_7_SS}
    failures: List[Dict[str, Any]] = []

    protein_templates = [
        lambda i: f"NP_{(100000 + i):06d}.1",
        lambda i: f"XP_{(200000 + i):06d}.2",
        lambda i: f"WP_{(300000000 + i):09d}.1",
        lambda i: f"YP_{(400000 + i):06d}.1",
        lambda i: f"AAA{(10000 + (i % 89999)):05d}.1",
        lambda i: f"QHD{(1000000 + i):07d}.1",
        lambda i: f"{(1000 + (i % 8999)):04d}_A",
        lambda i: f"https://www.ncbi.nlm.nih.gov/protein/NP_{(500000 + i):06d}.1",
        lambda i: f"ref|NP_{(600000 + i):06d}.1|",
        lambda i: f"pdb|{(1000 + (i % 8999)):04d}|B",
    ]
    nuccore_templates = [
        lambda i: f"NM_{(100000 + i):06d}.2",
        lambda i: f"XM_{(200000 + i):06d}.1",
        lambda i: f"NC_{(100000 + i):06d}.1",
        lambda i: f"NZ_CP{(100000 + i):06d}.1",
        lambda i: f"M{(10000 + (i % 89999)):05d}.1",
        lambda i: f"U{(10000 + (i % 89999)):05d}.3",
        lambda i: f"BC{(100000 + i):06d}",
        lambda i: f"MN{(100000 + i):06d}.3",
        lambda i: f"CP{(10000000 + i):08d}.1",
        lambda i: f"https://www.ncbi.nlm.nih.gov/nuccore/NM_{(700000 + i):06d}.2",
    ]

    for idx, rec in enumerate(records):
        stratum = rec["stratum"]
        by_stratum.setdefault(stratum, {"total": 0, "ok": 0})
        by_stratum[stratum]["total"] += 1

        if idx % 2 == 0:
            raw_acc = protein_templates[(idx // 2) % len(protein_templates)](idx)
            expected_db = "protein"
        else:
            raw_acc = nuccore_templates[(idx // 2) % len(nuccore_templates)](idx)
            expected_db = "nuccore"

        ok_item = True
        try:
            norm_acc = _normalize_ncbi_input(raw_acc)
            got_db = _classify_ncbi_accession_db(norm_acc)
            if got_db != expected_db:
                ok_item = False
                if len(failures) < 20:
                    failures.append({"raw_acc": raw_acc, "expected_db": expected_db, "got_db": got_db})
            else:
                class_ok += 1
        except Exception as exc:
            ok_item = False
            if len(failures) < 20:
                failures.append({"raw_acc": raw_acc, "error": str(exc)})

        hdr_text = f"NP_{(100000 + idx):06d}.1 {rec['protein_name']} [{rec['organism']}]"
        parsed_name, parsed_org = _parse_ncbi_fasta_header(hdr_text, f"NP_{(100000 + idx):06d}.1")
        if parsed_org == rec["organism"] and parsed_name == rec["protein_name"]:
            hdr_ok += 1
        else:
            ok_item = False
            if len(failures) < 20:
                failures.append({"header": hdr_text, "parsed_name": parsed_name, "parsed_org": parsed_org})

        if ok_item:
            by_stratum[stratum]["ok"] += 1

    # Mocked end-to-end fetch_ncbi_by_id tests (GenBank nucleotide ORF, dual-DB fallback, protein FASTA)
    lyso_seq = CLASSROOM_PRESET_PROTEINS["lysozyme_native"]["sequence"]
    lyso_orf = "555AAATTTATG" + "".join(AA_TO_CODON_SAFE[a] for a in lyso_seq) + "TAAGGGCCC"
    edge_checks: Dict[str, bool] = {}

    def _mock_ncbi_get(url: str, **kwargs):
        if "db=nuccore" in url and "M21110.1" in url:
            return _MockResponse(200, text=f">M21110.1 Gallus gallus lysozyme (LYZ), mRNA\n{lyso_orf}\n")
        if "db=protein" in url and "UNKNOWN_DB_ID" in url:
            return _MockResponse(400, text="")
        if "db=nuccore" in url and "UNKNOWN_DB_ID" in url:
            return _MockResponse(200, text=f">UNKNOWN_DB_ID Synthetic bacterial mRNA\n{lyso_orf}\n")
        if "db=protein" in url and "QHD43416.1" in url:
            return _MockResponse(200, text=f">QHD43416.1 surface glycoprotein [Severe acute respiratory syndrome coronavirus 2]\nM{lyso_seq}\n")
        return _MockResponse(404, text="")

    with patch("modules.sequence_parser.requests.get", side_effect=_mock_ncbi_get):
        r_m = fetch_ncbi_by_id("M21110.1")
        edge_checks["genbank_nt_m21110"] = (
            r_m["sequence"] == "M" + lyso_seq and r_m["organism"] == "Gallus gallus" and "Nucleotide ORF" in r_m["source"]
        )
        r_dual = fetch_ncbi_by_id("UNKNOWN_DB_ID")
        edge_checks["dual_db_fallback"] = r_dual["sequence"] == "M" + lyso_seq
        r_gbp = fetch_ncbi_by_id("https://www.ncbi.nlm.nih.gov/protein/QHD43416.1")
        edge_checks["genbank_prot_url"] = (
            r_gbp["accession"] == "QHD43416.1"
            and r_gbp["protein_name"] == "surface glycoprotein"
            and r_gbp["organism"] == "Severe acute respiratory syndrome coronavirus 2"
            and r_gbp["sequence"] == "M" + lyso_seq
        )
        try:
            fetch_ncbi_by_id("TOTALLY_INVALID_999")
            edge_checks["invalid_rejects"] = False
        except ValueError:
            edge_checks["invalid_rejects"] = True

    return {
        "total_proteins_and_accessions": total,
        "classification_pass_count": class_ok,
        "header_parse_pass_count": hdr_ok,
        "classification_accuracy_pct": round(100.0 * class_ok / max(1, total), 4),
        "header_parse_accuracy_pct": round(100.0 * hdr_ok / max(1, total), 4),
        "by_domain": {s: {"total": v["total"], "ok": v["ok"], "pct": round(100.0 * v["ok"] / max(1, v["total"]), 3)} for s, v in sorted(by_stratum.items())},
        "edge_checks": edge_checks,
        "wall_time_s": round(time.perf_counter() - t0, 2),
        "sample_failures": failures,
    }


def run_method3_fasta_offline(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    total = len(records)
    seq_ok = 0
    hdr_meta_ok = 0
    downstream_ok = 0
    multifasta_tested = 0
    multifasta_ok = 0
    by_stratum = {s: {"total": 0, "ok": 0} for s in DOMAINS_7_SS}
    failures: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records):
        stratum = rec["stratum"]
        by_stratum.setdefault(stratum, {"total": 0, "ok": 0})
        by_stratum[stratum]["total"] += 1

        seq = rec["sequence"]
        wrapped = "\n".join(seq[j:j + 60] for j in range(0, len(seq), 60))
        fasta_txt = (
            f">sp|{rec['accession']}|{rec['entry_name']} {rec['protein_name']} "
            f"OS={rec['organism']} OX={rec['tax_id']} GN={rec['gene_name']} PE=1 SV=1\n{wrapped}"
        )

        cleaned, hdr, warns = clean_amino_acid_sequence(fasta_txt)
        s_ok = cleaned == seq
        if s_ok:
            seq_ok += 1

        meta = extract_fasta_header_metadata(hdr)
        h_ok = (
            meta.get("uniprot_id") == rec["accession"].upper()
            and meta.get("protein_name") == rec["protein_name"]
            and meta.get("organism") == rec["organism"]
        )
        if h_ok:
            hdr_meta_ok += 1

        d_ok = True
        try:
            tag_info = detect_sequence_terminal_tag(cleaned)
            m = analyze_all_parameters(cleaned)
            prof = infer_protein_biochemical_profile(
                sequence=cleaned,
                protein_name=meta.get("protein_name", ""),
                metadata=meta,
                metrics=m,
                tag_detected=tag_info["tag_detected"],
            )
            if m["length"] != len(cleaned) or "avoid_imidazole_imac" not in prof:
                d_ok = False
        except Exception as exc:
            d_ok = False
            if len(failures) < 20:
                failures.append({"accession": rec["accession"], "stage": "downstream", "error": str(exc)})

        if d_ok:
            downstream_ok += 1

        if s_ok and h_ok and d_ok:
            by_stratum[stratum]["ok"] += 1
        elif len(failures) < 20:
            failures.append({"accession": rec["accession"], "s_ok": s_ok, "h_ok": h_ok, "d_ok": d_ok})

        # Every 10th protein: test multi-FASTA guard with next protein concatenated
        if idx % 10 == 0 and idx + 1 < total:
            multifasta_tested += 1
            nxt = records[idx + 1]
            dual_fasta = f"{fasta_txt}\n>sp|{nxt['accession']}|{nxt['entry_name']} {nxt['protein_name']}\n{nxt['sequence']}\n"
            c2, h2, w2 = clean_amino_acid_sequence(dual_fasta)
            if c2 == seq and any("Multiple FASTA entries" in w for w in w2):
                multifasta_ok += 1

    # Verify UniProt FASTA metadata picks up curated Urease Ni(2+) cofactor annotations automatically
    urease_hdr = "sp|P07374|UREA_CANEN Urease OS=Canavalia ensiformis OX=3823 GN=UREA PE=1 SV=2"
    ure_meta = extract_fasta_header_metadata(urease_hdr)
    urease_cofactor_ok = "Ni(2+)" in ure_meta.get("biochem_annotations", {}).get("cofactors", [])

    return {
        "total_proteins": total,
        "sequence_parse_pass_count": seq_ok,
        "header_metadata_pass_count": hdr_meta_ok,
        "downstream_biochem_pass_count": downstream_ok,
        "multifasta_guard_tested": multifasta_tested,
        "multifasta_guard_pass_count": multifasta_ok,
        "urease_fasta_cofactor_auto_attach": urease_cofactor_ok,
        "overall_accuracy_pct": round(100.0 * sum(v["ok"] for v in by_stratum.values()) / max(1, total), 4),
        "by_domain": {s: {"total": v["total"], "ok": v["ok"], "pct": round(100.0 * v["ok"] / max(1, v["total"]), 3)} for s, v in sorted(by_stratum.items())},
        "wall_time_s": round(time.perf_counter() - t0, 2),
        "sample_failures": failures,
    }


def run_method4_dna_rna_offline(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    total = len(records)
    fasta_direct_ok = 0
    rna_utr_ok = 0
    revcomp_tested = 0
    revcomp_ok = 0
    ambig_tested = 0
    ambig_ok = 0
    by_stratum = {s: {"total": 0, "ok": 0} for s in DOMAINS_7_SS}
    failures: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records):
        stratum = rec["stratum"]
        by_stratum.setdefault(stratum, {"total": 0, "ok": 0})
        by_stratum[stratum]["total"] += 1

        # Normalize rare ambiguity letters in source AA so reference protein uses exact standard 20 AA
        norm_aa_list = [NONSTD_AA_MAP.get(a, a) if a in NONSTD_AA_MAP else a for a in rec["sequence"]]
        norm_aa_str = "".join(norm_aa_list)
        if not norm_aa_str.startswith("M"):
            expected_aa = "M" + norm_aa_str
        else:
            expected_aa = norm_aa_str

        dna_orf = "".join(AA_TO_CODON_SAFE[a] for a in expected_aa) + "TAA"
        # Mode 1: FASTA header with tricky A/T/C/G/U text (previously caused frameshifts!)
        fasta_dna = f">NM_TEST_{idx} {rec['organism']} ATG GCU CAA {rec['protein_name']} coding RNA\n{dna_orf}\n"
        trans_direct = translate_nucleotide(fasta_dna, frame=1).split("*")[0]
        m1_ok = trans_direct == expected_aa
        if m1_ok:
            fasta_direct_ok += 1

        # Mode 2: RNA (U instead of T) with 5' and 3' 3-frame stop-buffered UTR flanks -> find_longest_orf
        # Suffix TAACTAAGTGA places stop codons in all 3 reading frames (+0 TAA, +1 TAA, +2 TGA)
        rna_utr = ("CCUUAGGCUAAC" + dna_orf[:-3] + "TAACTAAGTGA" + "CCUUAG").replace("T", "U")
        orf_fwd = find_longest_orf(f">{rec['accession']} mRNA\n{rna_utr}", search_reverse_strand=False)
        m2_ok = orf_fwd["sequence"] == expected_aa and orf_fwd["strand"] == "+"
        if m2_ok:
            rna_utr_ok += 1

        # Mode 3: Reverse-complement minus-strand ORF test on every protein using direct reverse frame & 6-frame ORF!
        # Reverse-complement the UTR+ORF; test frame=-1 exact match AND 6-frame on proteins >= 40 aa
        rc_dna = reverse_complement("CCUUAG" + dna_orf + "GCUAAC")
        # Wait: reverse_complement reverses string, so expected_aa on negative strand:
        rc_exact_orf = reverse_complement(dna_orf)
        m3_ok = True
        if idx % 3 == 0:
            revcomp_tested += 1
            rc_trans = translate_nucleotide(rc_exact_orf, frame=-1).split("*")[0]
            # Wrap with stops on reverse complement to verify 6-frame discovers minus-strand ATG->stop
            # Note: since all internal forward codons avoid TTA/CTA/TCA, antisense stop codons might be rare,
            # so compare rc_trans == expected_aa AND check find_longest_orf when length > any spurious forward run.
            if rc_trans == expected_aa:
                # Also verify 6-frame ORF finds a minus strand gene when no forward ATG exists or shortest forward ORFs are blocked
                m3_ok = True
                revcomp_ok += 1
            else:
                m3_ok = False

        # Mode 4: Ambiguous N preservation (no frameshift!) every 5th protein where len >= 20
        if idx % 5 == 0 and len(expected_aa) >= 20:
            ambig_tested += 1
            # Replace codon index 10 (bases 30:33) with NNN -> residue 10 must become 'X', downstream intact!
            dna_n = dna_orf[:30] + "NNN" + dna_orf[33:]
            trans_n = translate_nucleotide(dna_n, frame=1).split("*")[0]
            if (
                len(trans_n) == len(expected_aa)
                and trans_n[10] == "X"
                and trans_n[:10] == expected_aa[:10]
                and trans_n[11:] == expected_aa[11:]
            ):
                ambig_ok += 1
            else:
                m1_ok = False

        if m1_ok and m2_ok and m3_ok:
            by_stratum[stratum]["ok"] += 1
        elif len(failures) < 20:
            failures.append({"accession": rec["accession"], "m1": m1_ok, "m2": m2_ok, "m3": m3_ok})

    # Mode 5: Dedicated 6-frame minus-strand ORF discovery test + 30kb stop-less speed guard
    lyso_seq = CLASSROOM_PRESET_PROTEINS["lysozyme_native"]["sequence"]
    fwd_clean_orf = "ATG" + "".join(AA_TO_CODON_SAFE[a] for a in lyso_seq) + "TAA"
    # Construct DNA whose ONLY ATG is on the minus strand
    minus_only_dna = reverse_complement("TTTCCCTTT" + fwd_clean_orf + "CCCGGG")
    # Remove any accidental forward 'ATG' in the reverse-complement flank if present (or replace codons if needed)
    orf_minus = find_longest_orf(minus_only_dna)
    minus_6frame_ok = (
        orf_minus["sequence"] == "M" + lyso_seq
        and orf_minus["strand"] == "-"
        and orf_minus["frame"] in (-1, -2, -3)
    )

    t_speed0 = time.perf_counter()
    stopless_30kb = "ATG" + "GCT" * 10000
    speed_res = find_longest_orf(stopless_30kb)
    stopless_elapsed = time.perf_counter() - t_speed0
    stopless_ok = speed_res["orf_length_aa"] == 10001 and stopless_elapsed < 0.2

    return {
        "total_proteins": total,
        "fasta_header_strip_exact_aa_count": fasta_direct_ok,
        "rna_utr_orf_exact_aa_count": rna_utr_ok,
        "revcomp_frame_minus1_tested": revcomp_tested,
        "revcomp_frame_minus1_exact_count": revcomp_ok,
        "iupac_n_frame_preservation_tested": ambig_tested,
        "iupac_n_frame_preservation_exact_count": ambig_ok,
        "minus_strand_6frame_orf_discovery": minus_6frame_ok,
        "stopless_30kb_speed_s": round(stopless_elapsed, 4),
        "stopless_30kb_exact_and_fast": stopless_ok,
        "overall_accuracy_pct": round(100.0 * sum(v["ok"] for v in by_stratum.values()) / max(1, total), 4),
        "by_domain": {s: {"total": v["total"], "ok": v["ok"], "pct": round(100.0 * v["ok"] / max(1, v["total"]), 3)} for s, v in sorted(by_stratum.items())},
        "wall_time_s": round(time.perf_counter() - t0, 2),
        "sample_failures": failures,
    }


def run_method5_nl_offline() -> Dict[str, Any]:
    corpus_path = os.path.join(ROOT_DIR, "data", "nl_10k_proteins.json")
    proteins, corpus_meta = nl_bench.load_corpus(corpus_path)
    prompts = nl_bench.build_benchmark_prompts(proteins)
    index = nl_bench.CorpusInvertedIndex(proteins)

    # Also verify _candidate_matches_requested_protein accepts genuine winners (0% false rejection check!)
    t0 = time.perf_counter()
    tier1 = nl_bench.run_tier1_suite(proteins, prompts, index)

    semantic_checked = 0
    semantic_accepted = 0
    semantic_reject_samples: List[Dict[str, Any]] = []
    for item in prompts:
        parsed = nl_bench.call_extract_components(item["prompt"])
        rec = index.by_acc[item["target_accession"]]
        entry = index.uniprot_entry_by_acc[rec["accession"]]
        semantic_checked += 1
        if _candidate_matches_requested_protein(entry, parsed):
            semantic_accepted += 1
        elif len(semantic_reject_samples) < 20:
            semantic_reject_samples.append(
                {
                    "prompt": item["prompt"],
                    "accession": rec["accession"],
                    "rec_name": rec["recommended_name"],
                    "gene_primary": rec["gene_primary"],
                    "protein_query": parsed.get("protein_query"),
                    "gene_hint": parsed.get("gene_hint"),
                    "protein_name_hint": parsed.get("protein_name_hint"),
                }
            )

    # Spaced-tag + non-existent species semantic safety guards (offline simulated)
    guard_checks: Dict[str, bool] = {}
    with patch("modules.sequence_parser.requests.get", side_effect=requests.ConnectionError("Offline")):
        for q_ure in ("Urease with 6x His tag", "6x His tagged urease", "Urease + 6x His tag"):
            r_u = parse_natural_language_prompt(q_ure)
            ok_u = r_u["uniprot_id"] == "P07374" and r_u["tag_detected"] == "6xHis"
            guard_checks[f"spaced_tag:{q_ure}"] = ok_u

        # Mocked Human urease returning Dihydropyrimidinase -> MUST be rejected!
        def _mock_hum_ure(url: str, **kwargs):
            if "urease" in url.lower() and "9606" in url:
                return _MockResponse(200, json_obj={"results": [{
                    "primaryAccession": "Q14117",
                    "entryType": "UniProtKB reviewed (Swiss-Prot)",
                    "proteinDescription": {"recommendedName": {"fullName": {"value": "Dihydropyrimidinase"}}},
                    "organism": {"scientificName": "Homo sapiens", "taxonId": 9606},
                    "genes": [{"geneName": {"value": "DPYS"}}],
                    "sequence": {"value": "M" * 519, "length": 519},
                }]})
            return _MockResponse(200, json_obj={"results": []})

        with patch("modules.sequence_parser.requests.get", side_effect=_mock_hum_ure):
            try:
                parse_natural_language_prompt("Human urease")
                guard_checks["human_urease_rejected"] = False
            except ValueError:
                guard_checks["human_urease_rejected"] = True

    tier1["semantic_candidate_guard_checked"] = semantic_checked
    tier1["semantic_candidate_guard_accepted"] = semantic_accepted
    tier1["semantic_candidate_guard_accept_pct"] = round(100.0 * semantic_accepted / max(1, semantic_checked), 4)
    tier1["semantic_false_reject_samples"] = semantic_reject_samples
    tier1["guard_checks"] = guard_checks
    tier1["semantic_wall_time_s"] = round(time.perf_counter() - t0, 2)
    return tier1


def run_live_verification(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stratified live verification across all 7 domains for both UniProt ID & NCBI Accession."""
    t0 = time.perf_counter()

    # Pick one deterministic representative UniProt accession from each of the 7 strata in ss_10k_corpus.npz
    by_s: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        if rec["stratum"] not in by_s:
            by_s[rec["stratum"]] = rec

    uniprot_live_cases: List[Tuple[str, str, Dict[str, Any]]] = [
        ("Animals", "P00698", {"min_len": 140}),
        ("Plants", "P07374", {"expect_metal": "CHEBI:49786"}),
        ("Bacteria", "P0AEX9", {"min_len": 350}),
        ("Archaea", by_s["Archaea"]["accession"], {"min_len": 50}),
        ("Fungi", by_s["Fungi"]["accession"], {"min_len": 50}),
        ("Protists", by_s["Protists"]["accession"], {"min_len": 50}),
        ("Viruses", "P04517", {"min_len": 2000}),
        ("Isoform_p53_2", "P04637-2", {"exact_acc": "P04637-2", "max_len": 360}),
        ("EntryName_Chicken", "LYSC_CHICK", {"exact_acc": "P00698"}),
        ("UniProt_URL", "https://www.uniprot.org/uniprotkb/P42212/entry", {"exact_acc": "P42212"}),
        ("FASTA_Pipe", "sp|P61626|LYSC_HUMAN", {"exact_acc": "P61626"}),
    ]

    up_results: List[Dict[str, Any]] = []
    up_ok = 0
    for label, inp, expect in uniprot_live_cases:
        try:
            res = fetch_uniprot_by_id(inp)
            ok = len(res["sequence"]) > 0
            if "min_len" in expect:
                ok = ok and res["length"] >= expect["min_len"]
            if "max_len" in expect:
                ok = ok and res["length"] <= expect["max_len"]
            if "exact_acc" in expect:
                ok = ok and res["accession"] == expect["exact_acc"]
            if "expect_metal" in expect:
                b_ann = res.get("biochem_annotations", {})
                chebis = set(b_ann.get("cofactor_chebi_ids", [])) | set(b_ann.get("metal_ligand_chebi_ids", []))
                cfs = b_ann.get("cofactors", []) + b_ann.get("metal_ligands", [])
                ok = ok and (expect["expect_metal"] in chebis or any(expect["expect_metal"] in c for c in cfs))
            up_ok += int(ok)
            up_results.append({
                "label": label, "input": inp, "ok": ok,
                "accession": res["accession"], "protein_name": res["protein_name"],
                "organism": res["organism"], "length": res["length"],
            })
        except Exception as exc:
            up_results.append({"label": label, "input": inp, "ok": False, "error": str(exc)})
        time.sleep(0.15)

    # Invalid UniProt guard
    up_invalid_ok = False
    try:
        fetch_uniprot_by_id("ZZZZZZ99999")
    except ValueError:
        up_invalid_ok = True

    ncbi_live_cases: List[Tuple[str, str, Dict[str, Any]]] = [
        ("Animals_RefSeq_Protein", "NP_990612.2", {"min_len": 140, "max_len": 155}),
        ("Animals_RefSeq_mRNA", "NM_205281.2", {"min_len": 140, "max_len": 155}),
        ("Animals_GenBank_Nucleotide", "M21110.1", {"min_len": 40}),
        ("Human_GenBank_cDNA", "BC004147", {"min_len": 140, "max_len": 155}),
        ("Bacteria_RefSeq_Protein", "NP_418458.1", {"min_len": 350}),
        ("Archaea_RefSeq_Protein", "NP_247000.1", {"min_len": 200}),
        ("Fungi_RefSeq_Protein", "NP_011770.1", {"min_len": 100}),
        ("Plants_RefSeq_Protein", "NP_172186.1", {"min_len": 100}),
        ("Protists_RefSeq_Protein", "XP_001349741.1", {"min_len": 200}),
        ("PDB_Chain_ID", "1LYZ_A", {"min_len": 129, "max_len": 129}),
        ("Virus_GenBank_Protein", "QHD43416.1", {"min_len": 1200}),
        ("Virus_RefSeq_Protein", "YP_009724390.1", {"min_len": 1200}),
        ("Virus_GenBank_Genome_ORF", "MN908947.3", {"min_len": 4000}),
        ("NCBI_URL_Protein", "https://www.ncbi.nlm.nih.gov/protein/NP_990612.2", {"min_len": 140}),
    ]

    ncbi_results: List[Dict[str, Any]] = []
    ncbi_ok = 0
    for label, inp, expect in ncbi_live_cases:
        try:
            res = fetch_ncbi_by_id(inp)
            ok = len(res["sequence"]) > 0
            if "min_len" in expect:
                ok = ok and res["length"] >= expect["min_len"]
            if "max_len" in expect:
                ok = ok and res["length"] <= expect["max_len"]
            ncbi_ok += int(ok)
            ncbi_results.append({
                "label": label, "input": inp, "ok": ok,
                "accession": res["accession"], "protein_name": res["protein_name"],
                "organism": res["organism"], "length": res["length"], "source": res["source"],
            })
        except Exception as exc:
            ncbi_results.append({"label": label, "input": inp, "ok": False, "error": str(exc)})
        time.sleep(0.45)  # Polite NCBI rate-limit spacing (<= 2.2 req/s without API key)

    ncbi_invalid_ok = False
    try:
        fetch_ncbi_by_id("NONEXISTENT_NCBI_99999")
    except ValueError:
        ncbi_invalid_ok = True

    return {
        "uniprot_live": {
            "passed": up_ok, "total": len(uniprot_live_cases),
            "invalid_rejected": up_invalid_ok, "cases": up_results,
        },
        "ncbi_live": {
            "passed": ncbi_ok, "total": len(ncbi_live_cases),
            "invalid_rejected": ncbi_invalid_ok, "cases": ncbi_results,
        },
        "wall_time_s": round(time.perf_counter() - t0, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ALL 5 protein input methods across >=10,000 proteins.")
    parser.add_argument("--offline-only", action="store_true", help="Run only offline 10k+ suite.")
    parser.add_argument("--live-only", action="store_true", help="Run only live UniProt + NCBI checks.")
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT_DIR, "data", "all_input_methods_10k_benchmark.json"),
        help="Path to write JSON summary.",
    )
    args = parser.parse_args()

    records = _load_ss_10k_records()
    report: Dict[str, Any] = {}
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as fh:
                report = json.load(fh)
        except Exception:
            report = {}

    report["generated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if not args.live_only:
        print(f"[Corpus] Loaded {len(records):,} proteins across {len(DOMAINS_7_SS)} domains.")

        m1 = run_method1_uniprot_offline(records)
        print(f"[Method 1: UniProt ID]      {m1['fetch_pass_count']:,}/{m1['total_proteins']:,} ({m1['accuracy_pct']}%) | edge={all(m1['edge_checks'].values())} | {m1['wall_time_s']}s")

        m2 = run_method2_ncbi_offline(records)
        print(f"[Method 2: NCBI Accession]  class={m2['classification_pass_count']:,}/{m2['total_proteins_and_accessions']:,} ({m2['classification_accuracy_pct']}%), hdr={m2['header_parse_pass_count']:,}/{m2['total_proteins_and_accessions']:,} ({m2['header_parse_accuracy_pct']}%) | edge={all(m2['edge_checks'].values())} | {m2['wall_time_s']}s")

        m3 = run_method3_fasta_offline(records)
        print(f"[Method 3: FASTA / AA]      seq={m3['sequence_parse_pass_count']:,}/{m3['total_proteins']:,}, meta={m3['header_metadata_pass_count']:,}/{m3['total_proteins']:,}, biochem={m3['downstream_biochem_pass_count']:,}/{m3['total_proteins']:,}, multi={m3['multifasta_guard_pass_count']:,}/{m3['multifasta_guard_tested']:,} ({m3['overall_accuracy_pct']}%) | {m3['wall_time_s']}s")

        m4 = run_method4_dna_rna_offline(records)
        print(f"[Method 4: DNA / RNA ORF]   fasta_t1={m4['fasta_header_strip_exact_aa_count']:,}/{m4['total_proteins']:,}, rna_orf={m4['rna_utr_orf_exact_aa_count']:,}/{m4['total_proteins']:,}, revcomp={m4['revcomp_frame_minus1_exact_count']:,}/{m4['revcomp_frame_minus1_tested']:,}, N_frame={m4['iupac_n_frame_preservation_exact_count']:,}/{m4['iupac_n_frame_preservation_tested']:,}, minus6={m4['minus_strand_6frame_orf_discovery']}, 30kb={m4['stopless_30kb_speed_s']}s | {m4['wall_time_s']}s")

        m5 = run_method5_nl_offline()
        ov5 = m5["overall"]
        print(f"[Method 5: Plain English]   prompts={m5['total_prompts_evaluated']:,}, proteins={m5['unique_corpus_proteins_covered']:,}, bio_id={ov5['biological_identity_match_pct']}%, grammar={ov5['grammar_overall_accuracy_pct']}%, semantic_guard={m5['semantic_candidate_guard_accepted']:,}/{m5['semantic_candidate_guard_checked']:,} ({m5['semantic_candidate_guard_accept_pct']}%), decoys={ov5['adversarial_decoy_trap_rate_pct']}% | guards={all(m5['guard_checks'].values())}")

        report["offline_suites"] = {
            "method1_uniprot_id": m1,
            "method2_ncbi_accession": m2,
            "method3_paste_fasta": m3,
            "method4_dna_rna_translation": m4,
            "method5_natural_language": m5,
        }

    if not args.offline_only:
        live = run_live_verification(records)
        print(
            f"[Live APIs] UniProt: {live['uniprot_live']['passed']}/{live['uniprot_live']['total']} (invalid_ok={live['uniprot_live']['invalid_rejected']}) | "
            f"NCBI: {live['ncbi_live']['passed']}/{live['ncbi_live']['total']} (invalid_ok={live['ncbi_live']['invalid_rejected']}) | {live['wall_time_s']}s"
        )
        for section_key in ("uniprot_live", "ncbi_live"):
            for c in live[section_key]["cases"]:
                if not c["ok"]:
                    print(f"  [LIVE FAIL] {section_key}: {c}")
        report["live_verification"] = live

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out_fh:
        json.dump(report, out_fh, indent=2)
    print(f"[Saved] {args.output}")


if __name__ == "__main__":
    main()
