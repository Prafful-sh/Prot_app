"""
10,000+ Protein Multi-Domain & Live Reproducibility Benchmark for ProPurify Studio
==================================================================================
Evaluates `modules/sequence_parser.py` across ALL domains of life:
  - Bacteria
  - Archaea
  - Eukaryota: Animals (Metazoa)
  - Eukaryota: Plants (Viridiplantae)
  - Eukaryota: Fungi
  - Eukaryota: Protists
  - Viruses & Bacteriophages

Two-Tier Rigorous Benchmark Architecture:
1. Tier 1 — Full >=11,000-Protein Multi-Domain Grammar, Collision & Adversarial Candidate Ranking Suite:
   - Generates >=12,500 diverse plain-language student & researcher prompts covering 100% of unique
     Swiss-Prot proteins in `data/nl_10k_proteins.json` plus a dedicated hard-collision & adversarial
     grammar suite (`hisA..hisZ`, `hisS`, `Flagellin`, `Hemagglutinin`, `GST`, `MBP`, `Myc`,
     `Streptavidin`, `3C`, and Swiss-Prot protein names containing `with`, `plus`, `containing`, `and`).
   - Runs every prompt through `sequence_parser.py`'s exact production NL parser and
     `rank_uniprot_candidates` against an indexed candidate pool containing:
       * Simulated UniProt retrieval over the full >=11,000-protein corpus (never cheating when
         organism or protein extraction fails)
       * Cross-organism ortholog decoys from the corpus (identical protein/gene in other species)
       * Same-organism paralog / substring decoys from the corpus
       * Same-organism `-binding` / `-like` / `receptor` / `inhibitor` derivative traps
       * Same-organism identical-name unreviewed TrEMBL decoys (`A0A000`) & Fragment decoys
2. Tier 2 — Live End-to-End UniProt REST API Benchmark + 100-Prompt Shuffled Reproducibility Suite:
   - Executes >=635 stratified live `parse_natural_language_prompt(prompt)` calls directly against
     `https://rest.uniprot.org` via `ThreadPoolExecutor(max_workers=8)` across all 7 domain strata.
   - Re-runs 100 stratified prompts in shuffled order live against `https://rest.uniprot.org` to
     verify 100.0% deterministic reproducibility (`uniprot_id`, `full_construct_sequence`,
     `tag_detected`, `tag_position`, `cleavage_detected`).
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import modules.sequence_parser as sp
from modules.sequence_parser import (
    CLASSROOM_PRESET_PROTEINS,
    OFFLINE_UNIPROT_REFERENCES,
    STANDARD_CLEAVAGE_SITES,
    STANDARD_TAGS,
    parse_natural_language_prompt,
)

CANONICAL_STRATA = [
    "Bacteria",
    "Archaea",
    "Eukaryota: Animals",
    "Eukaryota: Plants",
    "Eukaryota: Fungi",
    "Eukaryota: Protists",
    "Viruses & Phages",
]

# Well-known common & laboratory shorthand names used by students and researchers
MODEL_ORGANISM_COMMON_NAMES: Dict[int, List[str]] = {
    9606: ["human", "Homo sapiens"],
    10090: ["mouse", "Mus musculus"],
    10116: ["rat", "Rattus norvegicus"],
    9031: ["chicken", "Gallus gallus"],
    9913: ["bovine", "Bos taurus"],
    9823: ["porcine", "Sus scrofa"],
    9986: ["rabbit", "Oryctolagus cuniculus"],
    7955: ["zebrafish", "Danio rerio"],
    7227: ["fruit fly", "Drosophila melanogaster"],
    6239: ["C. elegans", "Caenorhabditis elegans"],
    8355: ["Xenopus", "Xenopus laevis"],
    3702: ["Arabidopsis", "Arabidopsis thaliana"],
    39947: ["rice", "Oryza sativa"],
    4530: ["rice", "Oryza sativa"],
    4577: ["maize", "Zea mays"],
    3562: ["spinach", "Spinacia oleracea"],
    4081: ["tomato", "Solanum lycopersicum"],
    559292: ["yeast", "budding yeast", "Saccharomyces cerevisiae"],
    4932: ["yeast", "Saccharomyces cerevisiae"],
    284812: ["fission yeast", "S. pombe", "Schizosaccharomyces pombe"],
    4896: ["fission yeast", "Schizosaccharomyces pombe"],
    83333: ["E. coli", "Escherichia coli K-12", "Escherichia coli"],
    562: ["E. coli", "Escherichia coli"],
    224308: ["B. subtilis", "Bacillus subtilis"],
    1423: ["B. subtilis", "Bacillus subtilis"],
    271: ["Taq", "Thermus aquaticus"],
    65393: ["Taq", "Thermus aquaticus"],
    274: ["T. thermophilus", "Thermus thermophilus"],
    300852: ["T. thermophilus", "Thermus thermophilus"],
    2261: ["Pfu", "Pyrococcus furiosus"],
    186497: ["Pfu", "Pyrococcus furiosus"],
    69014: ["Thermococcus kodakarensis"],
    243232: ["M. jannaschii", "Methanocaldococcus jannaschii"],
    93061: ["S. aureus", "Staphylococcus aureus"],
    1280: ["S. aureus", "Staphylococcus aureus"],
    83332: ["M. tuberculosis", "Mycobacterium tuberculosis"],
    1773: ["M. tuberculosis", "Mycobacterium tuberculosis"],
    208964: ["P. aeruginosa", "Pseudomonas aeruginosa"],
    36329: ["P. falciparum", "Plasmodium falciparum"],
    44689: ["Dictyostelium discoideum", "slime mold"],
    10665: ["phage T4", "Bacteriophage T4", "Enterobacteria phage T4"],
    10760: ["phage T7", "Bacteriophage T7", "Enterobacteria phage T7"],
    2697049: ["SARS-CoV-2", "Severe acute respiratory syndrome coronavirus 2"],
    12227: ["Tobacco etch virus"],
}


def normalize_text(s: str) -> str:
    """Normalizes protein/organism strings for deterministic biological identity comparison."""
    s = (s or "").lower().strip()
    s = re.sub(r"\((?:mature[^)]*|chain[^)]*|fragment[^)]*)\)", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def extract_species_root(org_sci: str) -> str:
    """Extracts clean species root (stripping strain parentheses when informative)."""
    raw = (org_sci or "").strip()
    if not raw:
        return ""
    before_paren = re.split(r"\s*\(", raw, maxsplit=1)[0].strip()
    # Preserve strain identifier for 'Genus sp. (strain ...)' (normalizing PCC codes across substrains)
    if re.search(r"\bspp?\.$", before_paren, flags=re.IGNORECASE):
        m_pcc = re.search(r"\b(PCC\s*\d+)\b", raw, flags=re.IGNORECASE)
        if m_pcc:
            pcc_norm = re.sub(r"\s+", " ", m_pcc.group(1).upper())
            return f"{before_paren} {pcc_norm}"
        m_strain = re.search(r"\((?:strain\s+)?([^)/]+)", raw, flags=re.IGNORECASE)
        if m_strain:
            return f"{before_paren} {m_strain.group(1).strip()}"
        return raw
    return before_paren if len(before_paren) >= 4 else raw


def classify_stratum(rec: Dict[str, Any]) -> str:
    """Maps a corpus protein record to one of the 7 canonical domain/kingdom strata."""
    dom_exact = (rec.get("domain") or "").strip()
    sub_exact = (rec.get("subkingdom") or "").strip()
    if dom_exact == "Bacteria":
        return "Bacteria"
    if dom_exact == "Archaea":
        return "Archaea"
    if dom_exact == "Viruses":
        return "Viruses & Phages"
    if dom_exact == "Eukaryota":
        if sub_exact == "Metazoa":
            return "Eukaryota: Animals"
        if sub_exact == "Viridiplantae":
            return "Eukaryota: Plants"
        if sub_exact == "Fungi":
            return "Eukaryota: Fungi"
        if sub_exact == "Protists":
            return "Eukaryota: Protists"

    dom = dom_exact.lower()
    sub = (sub_exact or rec.get("kingdom") or rec.get("category") or "").lower()
    lineage = " ".join(rec.get("lineage") or []).lower()
    blob = f"{dom} {sub} {lineage}"

    if "virus" in blob or "viridae" in blob or "phage" in blob:
        return "Viruses & Phages"
    if "archaea" in blob:
        return "Archaea"
    if "bacteria" in blob:
        return "Bacteria"
    if any(k in blob for k in ("metazoa", "animal", "chordata", "arthropoda", "nematoda", "mollusca", "cnidaria")):
        return "Eukaryota: Animals"
    if any(k in blob for k in ("viridiplantae", "plant", "streptophyta", "chlorophyta")):
        return "Eukaryota: Plants"
    if any(k in blob for k in ("fungi", "ascomycota", "basidiomycota")):
        return "Eukaryota: Fungi"
    if "eukaryot" in blob or "protist" in blob:
        return "Eukaryota: Protists"
    return "Bacteria"


def load_corpus(corpus_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Loads and validates `nl_10k_proteins.json`."""
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if isinstance(payload, dict):
        proteins = payload.get("proteins") or payload.get("entries") or payload.get("records") or []
        meta = {k: v for k, v in payload.items() if k not in {"proteins", "entries", "records"}}
    elif isinstance(payload, list):
        proteins = payload
        meta = {}
    else:
        raise ValueError("Unexpected JSON structure in corpus file.")

    seen_acc: Set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for rec in proteins:
        acc = (rec.get("accession") or rec.get("primaryAccession") or "").strip()
        if not acc or acc in seen_acc:
            continue
        seen_acc.add(acc)
        rec = dict(rec)
        rec["accession"] = acc
        rec["recommended_name"] = (rec.get("recommended_name") or rec.get("protein_name") or "").strip()
        rec["alt_names"] = list(rec.get("alt_names") or [])
        rec["short_names"] = list(rec.get("short_names") or [])
        rec["gene_primary"] = (rec.get("gene_primary") or rec.get("gene_name") or "").strip()
        if rec["gene_primary"] == "N/A":
            rec["gene_primary"] = ""
        rec["gene_synonyms"] = list(rec.get("gene_synonyms") or [])
        rec["organism_scientific"] = (rec.get("organism_scientific") or rec.get("organism") or "").strip()
        rec["organism_common"] = (rec.get("organism_common") or "").strip()
        rec["tax_id"] = int(rec.get("tax_id") or rec.get("taxonId") or 0)
        rec["length"] = int(rec.get("length") or 250)
        rec["annotation_score"] = float(rec.get("annotation_score") or 5.0)
        rec["is_fragment"] = bool(rec.get("is_fragment", False))
        rec["stratum"] = classify_stratum(rec)
        rec["species_root"] = extract_species_root(rec["organism_scientific"])
        rec["norm_rec_name"] = normalize_text(rec["recommended_name"])
        deduped.append(rec)

    return deduped, meta


def corpus_rec_to_uniprot_entry(
    rec: Dict[str, Any],
    *,
    reviewed: bool = True,
    override_acc: Optional[str] = None,
    override_rec_name: Optional[str] = None,
    override_gene: Optional[str] = None,
    override_annot: Optional[float] = None,
    override_fragment: Optional[bool] = None,
    override_org_sci: Optional[str] = None,
    override_tax_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Formats a corpus record into the exact UniProtKB REST JSON structure expected by `sequence_parser.py`."""
    acc = override_acc or rec["accession"]
    rec_name = override_rec_name if override_rec_name is not None else rec["recommended_name"]
    gene_prim = override_gene if override_gene is not None else rec["gene_primary"]
    annot = override_annot if override_annot is not None else rec["annotation_score"]
    is_frag = override_fragment if override_fragment is not None else rec["is_fragment"]
    org_sci = override_org_sci if override_org_sci is not None else rec["organism_scientific"]
    tax_id = override_tax_id if override_tax_id is not None else rec["tax_id"]

    pdesc: Dict[str, Any] = {
        "recommendedName": {
            "fullName": {"value": rec_name},
            "shortNames": [{"value": s} for s in (rec["short_names"] if override_rec_name is None else []) if s],
        },
        "alternativeNames": [
            {"fullName": {"value": a}} for a in (rec["alt_names"] if override_rec_name is None else []) if a
        ],
    }
    if is_frag:
        pdesc["flag"] = "Fragment"

    genes_block: List[Dict[str, Any]] = []
    if gene_prim or (override_gene is None and rec["gene_synonyms"]):
        g_obj: Dict[str, Any] = {}
        if gene_prim:
            g_obj["geneName"] = {"value": gene_prim}
        if override_gene is None and rec["gene_synonyms"]:
            g_obj["synonyms"] = [{"value": s} for s in rec["gene_synonyms"] if s]
        genes_block.append(g_obj)

    seq_len = max(10, int(rec.get("length") or 200))
    if is_frag and override_fragment is True:
        seq_len = max(12, seq_len // 4)
    seq_val = rec.get("sequence") or ("M" + ("ACDEFGHIKLMNPQRSTVWY" * ((seq_len // 20) + 1))[: seq_len - 1])

    seq_block: Dict[str, Any] = {
        "length": len(seq_val),
        "value": seq_val,
    }
    if is_frag:
        seq_block["fragment"] = "single"

    return {
        "entryType": "UniProtKB reviewed (Swiss-Prot)" if reviewed else "UniProtKB unreviewed (TrEMBL)",
        "primaryAccession": acc,
        "uniProtkbId": rec.get("uniProtkbId") or f"{acc}_BENCH",
        "annotationScore": annot,
        "proteinDescription": pdesc,
        "genes": genes_block,
        "organism": {
            "scientificName": org_sci,
            "commonName": rec.get("organism_common", ""),
            "taxonId": tax_id,
            "lineage": rec.get("lineage") or [],
        },
        "sequence": seq_block,
    }


def is_hard_collision_record(rec: Dict[str, Any]) -> List[str]:
    """Identifies all collision/adversarial categories triggered by a real Swiss-Prot record."""
    reasons: List[str] = []
    gp = rec["gene_primary"]
    gp_low = gp.lower()
    rn = rec["recommended_name"]
    rn_low = rn.lower()
    org_low = rec["organism_scientific"].lower()

    if re.match(r"^his[a-z0-9]?$", gp_low) or gp_low == "hiss" or re.search(r"\bhistid", rn_low):
        reasons.append("his_pathway_or_histidine")
    if "flagellin" in rn_low or gp_low in {"flic", "flaa", "flab"} or rn_low.startswith("flag"):
        reasons.append("flagellin_flag")
    if "hemagglutinin" in rn_low or gp_low == "ha" or gp_low.startswith("ha-"):
        reasons.append("hemagglutinin_ha")
    if "glutathione s-transferase" in rn_low or gp_low.startswith("gst"):
        reasons.append("gst_enzyme")
    if "maltose" in rn_low or "myelin basic" in rn_low or gp_low in {"male", "mbp"}:
        reasons.append("mbp_collision")
    if gp_low in {"myc", "c-myc", "mycn", "mycl"} or "myc proto-oncogene" in rn_low or "mycobacterium" in org_low or "mycoplasma" in org_low:
        reasons.append("myc_or_mycobacterium")
    if "streptavidin" in rn_low or "streptomyces" in org_low or "streptococcus" in org_low:
        reasons.append("streptavidin_or_streptomyces")
    if gp_low.endswith("3c") or re.search(r"\b3c\b", rn_low):
        reasons.append("3c_gene_or_protease")
    if any(k in rn_low for k in ("thrombin", "coagulation factor x", "enteropeptidase", "nuclear inclusion")):
        reasons.append("protease_collision")
    if re.search(r"(?<![a-z0-9-])(?:with|plus|containing|carrying|bearing|having)(?![a-z0-9-])|-containing\b|\bplus-", rn_low):
        reasons.append("grammar_keyword_in_name")
    return reasons


def is_clean_single_token_symbol(sym: str) -> bool:
    """Checks if a gene symbol is suitable for hyphenated constructs like `6xHis-TEV-<sym>` or `<sym>-6xHis`."""
    if not sym or len(sym) < 3 or len(sym) > 12:
        return False
    if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", sym):
        return False
    # Exclude raw ORF locus tags containing underscores with long digit blocks when avoidable
    if sym.lower() in {"tag", "his", "gst", "mbp", "myc", "ha", "flag", "tev", "strep"}:
        return False
    return True


# 14 diverse construct templates covering all 10 STANDARD_TAGS, both positions, and all 5 cleavage sites
TAGGED_CONSTRUCT_SPECS: List[Dict[str, str]] = [
    {
        "kind": "clause_nterm_6xhis_tev",
        "fmt": "{org} {protein} with N-terminal 6xHis tag and TEV site",
        "exp_tag": "6xHis",
        "exp_pos": "N-terminal",
        "exp_cleav": "TEV Protease",
    },
    {
        "kind": "clause_cterm_6xhis",
        "fmt": "{protein} from {org} with C-terminal 6xHis tag",
        "exp_tag": "6xHis",
        "exp_pos": "C-terminal",
        "exp_cleav": "None",
    },
    {
        "kind": "prefix_gst_tagged",
        "fmt": "GST-tagged {org} {protein}",
        "exp_tag": "GST (Glutathione S-transferase)",
        "exp_pos": "N-terminal",
        "exp_cleav": "None",
    },
    {
        "kind": "prefix_mbp_fxa",
        "fmt": "MBP-tagged {protein} from {org} with Factor Xa cleavage site",
        "exp_tag": "MBP (Maltose-Binding Protein)",
        "exp_pos": "N-terminal",
        "exp_cleav": "Factor Xa",
    },
    {
        "kind": "inline_nterm_twinstrep",
        "fmt": "N-terminal Twin-Strep {protein} from {org}",
        "exp_tag": "Twin-Strep-tag",
        "exp_pos": "N-terminal",
        "exp_cleav": "None",
    },
    {
        "kind": "clause_cterm_flag_3c",
        "fmt": "{protein} from {org} with C-terminal FLAG tag and HRV 3C site",
        "exp_tag": "FLAG-tag",
        "exp_pos": "C-terminal",
        "exp_cleav": "PreScission / HRV 3C Protease",
    },
    {
        "kind": "carrying_nterm_8xhis_thrombin",
        "fmt": "{org} {protein} carrying an N-terminal 8xHis tag and Thrombin cleavage site",
        "exp_tag": "8xHis",
        "exp_pos": "N-terminal",
        "exp_cleav": "Thrombin",
    },
    {
        "kind": "paren_nterm_10xhis_prescission",
        "fmt": "{protein} ({org}) with N-terminal 10xHis tag and PreScission site",
        "exp_tag": "10xHis",
        "exp_pos": "N-terminal",
        "exp_cleav": "PreScission / HRV 3C Protease",
    },
    {
        "kind": "fused_cterm_strep",
        "fmt": "{protein} from {org} fused to C-terminal Strep-tag II",
        "exp_tag": "Strep-tag II",
        "exp_pos": "C-terminal",
        "exp_cleav": "None",
    },
    {
        "kind": "clause_nterm_ha_ek",
        "fmt": "{protein} from {org} with N-terminal HA tag and Enterokinase site",
        "exp_tag": "HA-tag",
        "exp_pos": "N-terminal",
        "exp_cleav": "Enterokinase",
    },
    {
        "kind": "bearing_cterm_myc",
        "fmt": "{org} {protein} bearing a C-terminal c-Myc tag",
        "exp_tag": "c-Myc-tag",
        "exp_pos": "C-terminal",
        "exp_cleav": "None",
    },
    {
        "kind": "hyph_cterm_6xhis",
        "fmt_sym": "{sym}-6xHis from {org}",
        "fmt_fallback": "{protein} ({org}) with C-terminal His6 tag",
        "exp_tag": "6xHis",
        "exp_pos": "C-terminal",
        "exp_cleav": "None",
    },
    {
        "kind": "hyph_cterm_tev_6xhis",
        "fmt_sym": "{sym}-TEV-6xHis {org}",
        "fmt_fallback": "{protein} from {org} with C-terminal 6xHis tag and TEV cleavage site",
        "exp_tag": "6xHis",
        "exp_pos": "C-terminal",
        "exp_cleav": "TEV Protease",
    },
    {
        "kind": "hyph_nterm_6xhis_tev",
        "fmt_sym": "6xHis-TEV-{sym} from {org}",
        "fmt_fallback": "6xHis-tagged {protein} from {org} with TEV site",
        "exp_tag": "6xHis",
        "exp_pos": "N-terminal",
        "exp_cleav": "TEV Protease",
    },
]


def choose_org_display(
    rec: Dict[str, Any],
    species_root_counts: Dict[str, int],
    variant_idx: int,
) -> Tuple[str, bool]:
    """
    Returns `(org_display, used_strain_specific_name)`.
    Uses realistic common/lab names for major model organisms, clean `Genus species` when unique
    or canonical, and full strain-qualified names on a subset or when multiple strains coexist.
    """
    tax_id = rec["tax_id"]
    full_sci = rec["organism_scientific"]
    root_sci = rec["species_root"]
    multi_strain = species_root_counts.get(root_sci.lower(), 0) > 1

    if tax_id in MODEL_ORGANISM_COMMON_NAMES and not multi_strain:
        aliases = MODEL_ORGANISM_COMMON_NAMES[tax_id]
        chosen = aliases[variant_idx % len(aliases)]
        return chosen, False

    if multi_strain:
        # Keep strain in 70% of prompts so strict accession is unambiguous, and use species root in 30%
        if variant_idx % 3 != 0:
            return full_sci, True
        return root_sci, False

    if variant_idx % 5 == 0:
        return full_sci, True
    return root_sci, False


def build_benchmark_prompts(proteins: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generates >=12,500 diverse plain-language prompts covering 100% of `proteins` plus
    a dedicated hard-collision & adversarial grammar suite.
    """
    taxids_by_root: Dict[str, Set[int]] = collections.defaultdict(set)
    for r in proteins:
        taxids_by_root[r["species_root"].lower()].add(r["tax_id"])
    species_root_counts = {k: len(v) for k, v in taxids_by_root.items()}

    prompts: List[Dict[str, Any]] = []

    # Part A: 100% Corpus Coverage (1 primary prompt per unique Swiss-Prot protein)
    for idx, rec in enumerate(proteins):
        org_disp, strain_specific = choose_org_display(rec, species_root_counts, idx)
        rec_name = rec["recommended_name"]
        gp = rec["gene_primary"]
        bucket = idx % 20

        if bucket < 5:
            # 25%: Untagged `<Recommended Protein Name> <Organism>`
            p_text = f"{rec_name} {org_disp}"
            p_family = "untagged_protein_org"
            exp_tag, exp_pos, exp_cleav = "None", "N/A", "None"
            used_gene_only = False
        elif bucket < 10:
            # 25%: Untagged `<Organism> <Recommended Name>` & `<Protein> from/in/(<Organism>)`
            sub = idx % 5
            if sub == 0:
                p_text = f"{org_disp} {rec_name}"
            elif sub == 1:
                p_text = f"{rec_name} from {org_disp}"
            elif sub == 2:
                p_text = f"{rec_name} in {org_disp}"
            elif sub == 3:
                p_text = f"{rec_name} ({org_disp})"
            else:
                p_text = f"Untagged {rec_name} from {org_disp}"
            p_family = "untagged_org_protein_and_clauses"
            exp_tag, exp_pos, exp_cleav = "None", "N/A", "None"
            used_gene_only = False
        elif bucket < 14:
            # 20%: Gene symbol + organism / combined gene + protein name
            if gp and len(gp) >= 2:
                if idx % 2 == 0 and re.match(r"^[A-Za-z][A-Za-z0-9_-]{1,14}$", gp):
                    p_text = f"{gp} from {org_disp}" if (idx % 4 == 0) else f"{org_disp} {gp}"
                    used_gene_only = True
                else:
                    p_text = f"{org_disp} {gp} ({rec_name})"
                    used_gene_only = False
                p_family = "untagged_gene_and_combined"
            else:
                p_text = f"{rec_name} from {org_disp}"
                p_family = "untagged_org_protein_and_clauses"
                used_gene_only = False
            exp_tag, exp_pos, exp_cleav = "None", "N/A", "None"
        else:
            # 30%: Tagged & Cleavage constructs across all 14 construct specs
            spec = TAGGED_CONSTRUCT_SPECS[idx % len(TAGGED_CONSTRUCT_SPECS)]
            if "fmt_sym" in spec:
                if is_clean_single_token_symbol(gp):
                    p_text = spec["fmt_sym"].format(sym=gp, org=org_disp, protein=rec_name)
                    used_gene_only = True
                else:
                    p_text = spec["fmt_fallback"].format(org=org_disp, protein=rec_name)
                    used_gene_only = False
            else:
                p_text = spec["fmt"].format(org=org_disp, protein=rec_name)
                used_gene_only = False
            p_family = "tagged_and_cleavage"
            exp_tag = spec["exp_tag"]
            exp_pos = spec["exp_pos"]
            exp_cleav = spec["exp_cleav"]

        prompts.append(
            {
                "prompt_id": f"T1-{len(prompts) + 1:05d}",
                "prompt": p_text,
                "family": p_family,
                "collision_tags": is_hard_collision_record(rec),
                "target_accession": rec["accession"],
                "target_tax_id": rec["tax_id"],
                "target_species_root": rec["species_root"],
                "target_stratum": rec["stratum"],
                "target_recommended_name": rec_name,
                "target_norm_rec_name": rec["norm_rec_name"],
                "target_gene_primary": gp,
                "strain_specific_prompt": strain_specific,
                "used_gene_only": used_gene_only,
                "expected_tag": exp_tag,
                "expected_pos": exp_pos,
                "expected_cleavage": exp_cleav,
            }
        )

    # Part B: Dedicated Hard-Collision & Adversarial Grammar Suite (Untagged + Tagged)
    collision_candidates = [r for r in proteins if is_hard_collision_record(r)]
    rng = random.Random(20261116)
    rng.shuffle(collision_candidates)
    # Cap extra collision prompts around 1,800 so total Tier-1 suite is ~12,800-13,500 prompts
    selected_collisions = collision_candidates[: min(len(collision_candidates), 950)]

    for c_idx, rec in enumerate(selected_collisions):
        reasons = is_hard_collision_record(rec)
        org_disp, strain_specific = choose_org_display(rec, species_root_counts, c_idx + 1)
        rec_name = rec["recommended_name"]
        gp = rec["gene_primary"]

        # 1. Untagged hard-collision prompt (verifies zero false tag/cleavage extraction & zero truncation)
        if "his_pathway_or_histidine" in reasons and gp and gp.lower().startswith("his"):
            u_prompt = f"{gp} {org_disp}"
            u_gene_only = True
        elif "3c_gene_or_protease" in reasons and gp and gp.lower().endswith("3c"):
            u_prompt = f"{gp} from {org_disp}"
            u_gene_only = True
        else:
            u_prompt = f"{rec_name} from {org_disp}" if (c_idx % 2 == 0) else f"{org_disp} {rec_name}"
            u_gene_only = False

        prompts.append(
            {
                "prompt_id": f"T1-{len(prompts) + 1:05d}",
                "prompt": u_prompt,
                "family": "hard_collision_untagged",
                "collision_tags": reasons,
                "target_accession": rec["accession"],
                "target_tax_id": rec["tax_id"],
                "target_species_root": rec["species_root"],
                "target_stratum": rec["stratum"],
                "target_recommended_name": rec_name,
                "target_norm_rec_name": rec["norm_rec_name"],
                "target_gene_primary": gp,
                "strain_specific_prompt": strain_specific,
                "used_gene_only": u_gene_only,
                "expected_tag": "None",
                "expected_pos": "N/A",
                "expected_cleavage": "None",
            }
        )

        # 2. Tagged hard-collision prompt (especially testing 'with ... with 6xHis tag' & dual-meaning tags!)
        if c_idx % 2 == 0:
            t_prompt = f"{rec_name} {org_disp} with N-terminal 6xHis tag and TEV site"
            e_tag, e_pos, e_clv = "6xHis", "N-terminal", "TEV Protease"
        else:
            t_prompt = f"{org_disp} {rec_name} with C-terminal FLAG tag and HRV 3C site"
            e_tag, e_pos, e_clv = "FLAG-tag", "C-terminal", "PreScission / HRV 3C Protease"

        prompts.append(
            {
                "prompt_id": f"T1-{len(prompts) + 1:05d}",
                "prompt": t_prompt,
                "family": "hard_collision_tagged",
                "collision_tags": reasons,
                "target_accession": rec["accession"],
                "target_tax_id": rec["tax_id"],
                "target_species_root": rec["species_root"],
                "target_stratum": rec["stratum"],
                "target_recommended_name": rec_name,
                "target_norm_rec_name": rec["norm_rec_name"],
                "target_gene_primary": gp,
                "strain_specific_prompt": strain_specific,
                "used_gene_only": False,
                "expected_tag": e_tag,
                "expected_pos": e_pos,
                "expected_cleavage": e_clv,
            }
        )

    return prompts


class CorpusInvertedIndex:
    """Fast in-memory UniProt search & hard-negative decoy index over all >=11,000 Swiss-Prot records."""

    def __init__(self, proteins: List[Dict[str, Any]]) -> None:
        self.proteins = proteins
        self.by_acc: Dict[str, Dict[str, Any]] = {}
        self.uniprot_entry_by_acc: Dict[str, Dict[str, Any]] = {}
        self.by_tax_id: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
        self.by_species_root: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        self.by_norm_name: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        self.by_gene_low: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        self.by_token: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)

        for rec in proteins:
            acc = rec["accession"]
            self.by_acc[acc] = rec
            self.uniprot_entry_by_acc[acc] = corpus_rec_to_uniprot_entry(rec, reviewed=True)
            self.by_tax_id[rec["tax_id"]].append(rec)
            self.by_species_root[rec["species_root"].lower()].append(rec)
            self.by_norm_name[rec["norm_rec_name"]].append(rec)

            genes = [rec["gene_primary"]] + list(rec["gene_synonyms"])
            for g in {x.strip().lower() for x in genes if x and x.strip()}:
                self.by_gene_low[g].append(rec)

            all_names = [rec["recommended_name"]] + rec["alt_names"] + rec["short_names"] + genes
            toks = {
                t
                for nm in all_names
                for t in re.split(r"[^a-z0-9]+", (nm or "").lower())
                if len(t) >= 2
            }
            for tok in toks:
                self.by_token[tok].append(rec)

    def parser_retrieves_target(self, parsed: Dict[str, Any], target_rec: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Verifies that `sequence_parser`'s extracted `(protein_query, organism_taxid, organism_name, organism_raw)`
        genuinely matches `target_rec` in UniProtKB without leaking unstripped organism/modifier noise.
        """
        pq = (parsed.get("protein_query") or "").strip()
        if not pq:
            return False, "empty_protein_query"

        ext_taxid = parsed.get("organism_taxid")
        ext_clade = parsed.get("organism_clade_taxid")
        ext_taxids: Set[int] = set(parsed.get("organism_taxids") or [])
        if ext_taxid is not None:
            ext_taxids.add(ext_taxid)
        if ext_clade is not None:
            ext_taxids.add(ext_clade)

        ext_org_name = (parsed.get("organism_name") or parsed.get("organism_scientific") or "").strip()
        ext_org_raw = (parsed.get("organism_raw") or parsed.get("organism_hint") or "").strip()
        ext_strain = (parsed.get("organism_strain") or "").strip()

        # 1. Organism consistency check
        if ext_taxids or ext_org_name or ext_org_raw:
            tgt_tax = target_rec["tax_id"]
            tgt_sci_low = target_rec["organism_scientific"].lower()
            tgt_com_low = (target_rec.get("organism_common") or "").lower()
            tgt_root_low = target_rec["species_root"].lower()

            # Guard against catalog taxon ID typos pointing to a completely different genus/species (e.g. 4058 vs 60093)
            if ext_taxids and tgt_tax not in ext_taxids:
                corpus_hits_for_taxids = [r for tx in ext_taxids for r in self.by_tax_id.get(tx, [])]
                if corpus_hits_for_taxids:
                    tgt_genus = tgt_root_low.split()[0]
                    if not any(tgt_genus in r["organism_scientific"].lower() for r in corpus_hits_for_taxids):
                        return False, f"catalog_taxid_wrong_genus:{sorted(ext_taxids)}_vs_{target_rec['organism_scientific']!r}({tgt_tax})"

            # If a specific strain was extracted (e.g. PCC 6714), ensure primary taxid wasn't locked to a conflicting strain
            if ext_strain and ext_taxid is not None and tgt_tax not in ext_taxids and ext_strain.lower() not in ext_org_name.lower():
                if ext_strain.lower() in tgt_sci_low:
                    return False, f"conflicting_strain_taxid:{ext_taxid}_for_strain_{ext_strain!r}"

            tax_ok = tgt_tax in ext_taxids
            name_ok = False
            for cand_h in (ext_org_raw.lower(), ext_org_name.lower()):
                if cand_h and (
                    cand_h in tgt_sci_low
                    or tgt_root_low in cand_h
                    or (tgt_com_low and cand_h in tgt_com_low)
                ):
                    name_ok = True

            if not (tax_ok or name_ok):
                clade_species_prefixes = {
                    11320: "influenza a virus",
                    11676: "human immunodeficiency virus",
                    11103: "hepatitis c virus",
                    10407: "hepatitis b virus",
                    12092: "hepatitis a virus",
                    10376: "epstein-barr virus",
                    10298: "herpes simplex virus",
                    10245: "vaccinia virus",
                    12637: "dengue virus",
                    12080: "poliovirus",
                    10665: "phage t4",
                    10760: "phage t7",
                    10710: "phage lambda",
                    562: "escherichia coli",
                    1423: "bacillus subtilis",
                    1280: "staphylococcus aureus",
                    1773: "mycobacterium tuberculosis",
                    287: "pseudomonas aeruginosa",
                    1314: "streptococcus pyogenes",
                    1313: "streptococcus pneumoniae",
                    1148: "synechocystis",
                    666: "vibrio cholerae",
                    210: "helicobacter pylori",
                }
                for cid, pref in clade_species_prefixes.items():
                    if cid in ext_taxids and pref in tgt_sci_low:
                        name_ok = True
                        break

            if not (tax_ok or name_ok):
                return (
                    False,
                    f"wrong_organism:taxids={sorted(ext_taxids)},name={ext_org_name!r},raw={ext_org_raw!r}_vs_{target_rec['organism_scientific']!r}",
                )

        # 2. Protein / gene query match check (simulating UniProt Tier 1-3 fielded search)
        protein_blob = normalize_text(
            " ".join(
                [
                    target_rec["recommended_name"],
                    *target_rec["alt_names"],
                    *target_rec["short_names"],
                    target_rec["gene_primary"],
                    *target_rec["gene_synonyms"],
                    target_rec.get("uniProtkbId", ""),
                ]
            )
        )
        blob_tokens = set(protein_blob.split())

        pq_norm = normalize_text(pq)
        pq_tokens = [
            t
            for t in pq_norm.split()
            if (len(t) >= 2 or t.isdigit()) and t not in {"the", "and", "or", "of", "in", "from"}
        ]
        if len(pq_tokens) > 1:
            pq_tokens = [t for t in pq_tokens if t != "protein" or "protein" in blob_tokens]
        if not pq_tokens:
            return False, f"no_searchable_tokens:{pq!r}"

        known_synonyms = {"p53", "gfp", "bsa", "hewl", "mbp", "gst", "myc", "tev", "thrombin", "enterokinase"}
        missing = [
            t
            for t in pq_tokens
            if t not in blob_tokens and t not in protein_blob and t not in known_synonyms
        ]
        if missing:
            return False, f"unmatched_query_tokens_in_protein_fields:{missing}_from_pq={pq!r}"

        return True, "ok"

    def build_adversarial_pool(
        self,
        prompt_item: Dict[str, Any],
        parsed: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], bool, str]:
        """
        Builds a deterministic 15-25 candidate pool containing:
          1. True target Swiss-Prot entry (ONLY if `parser_retrieves_target` succeeds!)
          2. Same-organism real Swiss-Prot entries from the corpus sharing tokens/genes
          3. Cross-organism real Swiss-Prot ortholog decoys (same protein/gene in other species)
          4. Same-organism adversarial `-binding`, `-like`, `receptor`, `inhibitor` Swiss-Prot traps
          5. Same-organism identical-name unreviewed TrEMBL decoy (`A0A000`) + Fragment decoy (`A0A001`)
        """
        target_rec = self.by_acc[prompt_item["target_accession"]]
        retrievable, retrieve_reason = self.parser_retrieves_target(parsed, target_rec)

        pool_by_acc: Dict[str, Dict[str, Any]] = {}
        if retrievable:
            pool_by_acc[target_rec["accession"]] = self.uniprot_entry_by_acc[target_rec["accession"]]

        pq = (parsed.get("protein_query") or "").strip()
        pq_toks = [t for t in normalize_text(pq).split() if len(t) >= 3]

        # A. Real same-organism corpus entries sharing query tokens or random paralogs in same tax_id
        same_org_recs = self.by_tax_id.get(target_rec["tax_id"], [])
        overlap_same_org: List[Dict[str, Any]] = []
        other_same_org: List[Dict[str, Any]] = []
        for r in same_org_recs:
            if r["accession"] == target_rec["accession"]:
                continue
            r_blob = f"{r['norm_rec_name']} {r['gene_primary'].lower()}"
            if any(t in r_blob for t in pq_toks):
                overlap_same_org.append(r)
            else:
                other_same_org.append(r)

        for r in overlap_same_org[:8]:
            pool_by_acc[r["accession"]] = self.uniprot_entry_by_acc[r["accession"]]
        for r in other_same_org[:2]:
            pool_by_acc[r["accession"]] = self.uniprot_entry_by_acc[r["accession"]]

        # B. Cross-organism orthologs with exact same `recommended_name` or `gene_primary`
        orthologs = [
            r
            for r in self.by_norm_name.get(target_rec["norm_rec_name"], [])
            if r["tax_id"] != target_rec["tax_id"]
        ]
        if target_rec["gene_primary"]:
            gene_orth = [
                r
                for r in self.by_gene_low.get(target_rec["gene_primary"].lower(), [])
                if r["tax_id"] != target_rec["tax_id"]
            ]
            orthologs.extend(gene_orth)
        for r in orthologs[:5]:
            if r["accession"] not in pool_by_acc:
                pool_by_acc[r["accession"]] = self.uniprot_entry_by_acc[r["accession"]]

        # C. Same-organism derivative traps (`-binding protein`, `-like protein`, `inhibitor`, `receptor`)
        base_name = target_rec["recommended_name"]
        base_gene = target_rec["gene_primary"] or "PROT"
        deriv_specs = [
            ("D_BIND_" + target_rec["accession"], f"{base_name}-binding protein", f"{base_gene}BP"),
            ("D_LIKE_" + target_rec["accession"], f"{base_name}-like protein", f"{base_gene}L1"),
            ("D_INHB_" + target_rec["accession"], f"{base_name} inhibitor", f"{base_gene}I"),
            ("D_RECP_" + target_rec["accession"], f"{base_name} receptor", f"{base_gene}R"),
        ]
        for d_acc, d_name, d_gene in deriv_specs:
            if normalize_text(d_name) != target_rec["norm_rec_name"]:
                pool_by_acc[d_acc] = corpus_rec_to_uniprot_entry(
                    target_rec,
                    reviewed=True,
                    override_acc=d_acc,
                    override_rec_name=d_name,
                    override_gene=d_gene,
                    override_annot=5.0,
                    override_fragment=False,
                )

        # D. Same-organism identical-name unreviewed TrEMBL decoy (lexicographically first accession 'A0A000'!)
        trembl_acc = f"A0A000_{target_rec['accession']}"
        pool_by_acc[trembl_acc] = corpus_rec_to_uniprot_entry(
            target_rec,
            reviewed=False,
            override_acc=trembl_acc,
            override_rec_name=base_name,
            override_gene=target_rec["gene_primary"],
            override_annot=3.0,
            override_fragment=False,
        )

        # E. Same-organism identical-name Fragment decoys (only when true target is full-length!)
        if not target_rec["is_fragment"]:
            frag_acc = f"A0A001_{target_rec['accession']}"
            pool_by_acc[frag_acc] = corpus_rec_to_uniprot_entry(
                target_rec,
                reviewed=True,
                override_acc=frag_acc,
                override_rec_name=base_name,
                override_gene=target_rec["gene_primary"],
                override_annot=target_rec["annotation_score"],
                override_fragment=True,
            )
            # Also inject an un-isolated bare-species legacy Edman Fragment decoy (e.g. P13774 vs Q27743)
            if target_rec["organism_scientific"] != target_rec["species_root"]:
                bare_frag_acc = f"A0A002_{target_rec['accession']}"
                pool_by_acc[bare_frag_acc] = corpus_rec_to_uniprot_entry(
                    target_rec,
                    reviewed=True,
                    override_acc=bare_frag_acc,
                    override_rec_name=base_name,
                    override_gene=target_rec["gene_primary"],
                    override_annot=2.0,
                    override_fragment=True,
                    override_org_sci=target_rec["species_root"],
                    override_tax_id=9999999,
                )

        # Deterministically shuffle candidate pool so input order never leaks ground truth
        entries = list(pool_by_acc.values())
        seed_int = int(hashlib.md5(prompt_item["prompt_id"].encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed_int)
        rng.shuffle(entries)
        return entries, retrievable, retrieve_reason


def call_extract_components(prompt: str) -> Dict[str, Any]:
    """Calls `sequence_parser.extract_nl_query_components` (adapting cleanly to its return dict)."""
    if hasattr(sp, "extract_nl_query_components"):
        res = sp.extract_nl_query_components(prompt)
        return res
    # Fallback adapter for pre-upgrade `sequence_parser.py` during early local smoke checks
    target_phrase, modifier_text, force_untagged = sp._split_target_and_modifiers(prompt)
    if force_untagged:
        det_tag, tag_pos, det_cleav = None, "N/A", None
    else:
        det_tag, tag_pos, det_cleav = sp._detect_tag_and_cleavage(modifier_text)
    if (det_tag or det_cleav) and not sp._has_real_protein_tokens(target_phrase):
        det_tag, tag_pos, det_cleav = None, "N/A", None
        target_phrase = prompt.strip()
    stripped_prot, org_taxid, org_sci = sp._strip_known_organisms(target_phrase)
    pq = sp._normalize_protein_query(stripped_prot)
    return {
        "prompt": prompt,
        "target_phrase": target_phrase,
        "modifier_text": modifier_text,
        "protein_query": pq,
        "organism_taxid": org_taxid,
        "organism_scientific": org_sci,
        "organism_hint": org_sci,
        "tag_detected": det_tag["name"] if det_tag else "None",
        "tag_position": tag_pos if det_tag else "N/A",
        "cleavage_detected": det_cleav["name"] if det_cleav else "None",
    }


def call_rank_candidates(entries: List[Dict[str, Any]], parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Calls `sequence_parser.rank_uniprot_candidates` directly."""
    if hasattr(sp, "rank_uniprot_candidates"):
        return sp.rank_uniprot_candidates(entries, parsed)
    # Fallback adapter for pre-upgrade `sequence_parser.py`
    pq = parsed.get("protein_query") or ""
    _, pref_accs, pref_genes = sp._build_uniprot_search_plan(pq, parsed.get("organism_taxid"))
    ranked = sorted(
        entries,
        key=lambda e: (
            sp._rank_uniprot_hit(e, pq, pref_accs, pref_genes),
            float(e.get("annotationScore", 0.0) or 0.0),
            e.get("primaryAccession", ""),
        ),
        reverse=True,
    )
    return ranked


def get_parsed_tag_fields(parsed: Dict[str, Any]) -> Tuple[str, str, str]:
    """Extracts standardized `(tag_name, tag_position, cleavage_name)` strings from `parsed`."""
    raw_tag = parsed.get("tag_detected")
    if isinstance(raw_tag, dict):
        tag_name = raw_tag.get("name", "None")
    elif raw_tag:
        tag_name = str(raw_tag)
    else:
        tag_name = "None"

    tag_pos = str(parsed.get("tag_position") or ("N/A" if tag_name == "None" else "N-terminal"))
    if tag_name == "None":
        tag_pos = "N/A"

    raw_clv = parsed.get("cleavage_detected")
    if isinstance(raw_clv, dict):
        clv_name = raw_clv.get("name", "None")
    elif raw_clv:
        clv_name = str(raw_clv)
    else:
        clv_name = "None"

    return tag_name, tag_pos, clv_name


def is_biological_identity_match_tier1(
    winner_entry: Dict[str, Any],
    prompt_item: Dict[str, Any],
    index: CorpusInvertedIndex,
) -> bool:
    """
    Checks exact biological identity in Tier 1:
      - Exact accession match, OR
      - Genuine reviewed Swiss-Prot corpus entry (NEVER a synthetic/TrEMBL/Fragment decoy!) with
        matching organism (exact `tax_id`, or exact `species_root` when strain was omitted from prompt)
        AND exact normalized `recommended_name` (or exact `gene_primary` when prompt was gene-only).
    """
    win_acc = winner_entry.get("primaryAccession", "")
    if win_acc == prompt_item["target_accession"]:
        return True
    if win_acc not in index.by_acc:
        # Decoy won -> NEVER a valid biological match!
        return False

    win_rec = index.by_acc[win_acc]
    same_tax = win_rec["tax_id"] == prompt_item["target_tax_id"]
    same_species_when_unstrained = (
        not prompt_item["strain_specific_prompt"]
        and win_rec["species_root"].lower() == prompt_item["target_species_root"].lower()
    )
    if not (same_tax or same_species_when_unstrained):
        return False

    if win_rec["norm_rec_name"] == prompt_item["target_norm_rec_name"]:
        return True

    if prompt_item["used_gene_only"] and prompt_item["target_gene_primary"]:
        t_gp = prompt_item["target_gene_primary"].lower()
        w_genes = {win_rec["gene_primary"].lower(), *(g.lower() for g in win_rec["gene_synonyms"])}
        if t_gp in w_genes:
            return True

    return False


def run_tier1_suite(
    proteins: List[Dict[str, Any]],
    prompts: List[Dict[str, Any]],
    index: CorpusInvertedIndex,
) -> Dict[str, Any]:
    """Executes the full >=12,500-prompt Tier 1 grammar, collision & ranking benchmark."""
    t0 = time.perf_counter()

    total = len(prompts)
    grammar_ok = 0
    tag_ok = 0
    pos_ok = 0
    cleav_ok = 0
    query_clean_ok = 0
    strict_acc_ok = 0
    bio_id_ok = 0
    top3_acc_ok = 0
    decoy_trap_hits = 0

    by_stratum: Dict[str, Dict[str, int]] = {
        s: {"total": 0, "grammar_ok": 0, "strict_acc_ok": 0, "bio_id_ok": 0, "top3_acc_ok": 0}
        for s in CANONICAL_STRATA
    }
    by_family: Dict[str, Dict[str, int]] = collections.defaultdict(
        lambda: {"total": 0, "grammar_ok": 0, "strict_acc_ok": 0, "bio_id_ok": 0, "top3_acc_ok": 0}
    )
    by_collision: Dict[str, Dict[str, int]] = collections.defaultdict(
        lambda: {"total": 0, "grammar_ok": 0, "strict_acc_ok": 0, "bio_id_ok": 0}
    )

    grammar_failures: List[Dict[str, Any]] = []
    ranking_failures: List[Dict[str, Any]] = []

    for item in prompts:
        p_text = item["prompt"]
        stratum = item["target_stratum"]
        fam = item["family"]

        try:
            parsed = call_extract_components(p_text)
        except Exception as exc:
            parsed = {
                "protein_query": "",
                "tag_detected": "ERROR",
                "tag_position": "ERROR",
                "cleavage_detected": str(exc),
            }

        det_tag, det_pos, det_cleav = get_parsed_tag_fields(parsed)
        t_match = det_tag == item["expected_tag"]
        p_match = det_pos == item["expected_pos"]
        c_match = det_cleav == item["expected_cleavage"]

        pool, retrievable, retrieve_reason = index.build_adversarial_pool(item, parsed)
        q_clean = bool((parsed.get("protein_query") or "").strip()) and retrievable

        g_all = t_match and p_match and c_match and q_clean

        ranked = call_rank_candidates(pool, parsed) if pool else []
        winner = ranked[0] if ranked else {}
        win_acc = winner.get("primaryAccession", "")
        top3_accs = {r.get("primaryAccession", "") for r in ranked[:3]}

        is_strict = win_acc == item["target_accession"]
        is_bio = is_biological_identity_match_tier1(winner, item, index) if winner else False
        is_top3 = item["target_accession"] in top3_accs

        if win_acc.startswith(("D_", "A0A000_", "A0A001_", "A0A002_")):
            decoy_trap_hits += 1

        tag_ok += int(t_match)
        pos_ok += int(p_match)
        cleav_ok += int(c_match)
        query_clean_ok += int(q_clean)
        grammar_ok += int(g_all)
        strict_acc_ok += int(is_strict)
        bio_id_ok += int(is_bio)
        top3_acc_ok += int(is_top3)

        for bucket_dict, key in ((by_stratum, stratum), (by_family, fam)):
            b = bucket_dict[key]
            b["total"] += 1
            b["grammar_ok"] += int(g_all)
            b["strict_acc_ok"] += int(is_strict)
            b["bio_id_ok"] += int(is_bio)
            b["top3_acc_ok"] += int(is_top3)

        for c_tag in item["collision_tags"]:
            cb = by_collision[c_tag]
            cb["total"] += 1
            cb["grammar_ok"] += int(g_all)
            cb["strict_acc_ok"] += int(is_strict)
            cb["bio_id_ok"] += int(is_bio)

        if not g_all and len(grammar_failures) < 120:
            grammar_failures.append(
                {
                    "prompt_id": item["prompt_id"],
                    "prompt": p_text,
                    "family": fam,
                    "stratum": stratum,
                    "target_accession": item["target_accession"],
                    "target_recommended_name": item["target_recommended_name"],
                    "target_gene": item["target_gene_primary"],
                    "expected": {
                        "tag": item["expected_tag"],
                        "pos": item["expected_pos"],
                        "cleavage": item["expected_cleavage"],
                    },
                    "got": {
                        "tag": det_tag,
                        "pos": det_pos,
                        "cleavage": det_cleav,
                        "protein_query": parsed.get("protein_query"),
                        "organism_taxid": parsed.get("organism_taxid"),
                        "organism_name": parsed.get("organism_name"),
                        "organism_raw": parsed.get("organism_raw"),
                    },
                    "retrieve_reason": retrieve_reason,
                }
            )

        if not is_bio and len(ranking_failures) < 120:
            win_desc = (
                winner.get("proteinDescription", {})
                .get("recommendedName", {})
                .get("fullName", {})
                .get("value", "")
            )
            ranking_failures.append(
                {
                    "prompt_id": item["prompt_id"],
                    "prompt": p_text,
                    "family": fam,
                    "stratum": stratum,
                    "target_accession": item["target_accession"],
                    "target_recommended_name": item["target_recommended_name"],
                    "target_gene": item["target_gene_primary"],
                    "winner_accession": win_acc,
                    "winner_name": win_desc,
                    "protein_query": parsed.get("protein_query"),
                    "organism_taxid": parsed.get("organism_taxid"),
                    "organism_name": parsed.get("organism_name"),
                    "organism_raw": parsed.get("organism_raw"),
                    "retrieve_reason": retrieve_reason,
                }
            )

    elapsed = time.perf_counter() - t0

    def _fmt_bucket(b: Dict[str, int]) -> Dict[str, Any]:
        n = max(1, b["total"])
        out: Dict[str, Any] = {
            "total_prompts": b["total"],
            "grammar_accuracy_pct": round(100.0 * b["grammar_ok"] / n, 3),
            "strict_accession_match_pct": round(100.0 * b["strict_acc_ok"] / n, 3),
            "biological_identity_match_pct": round(100.0 * b["bio_id_ok"] / n, 3),
        }
        if "top3_acc_ok" in b:
            out["top3_strict_accession_pct"] = round(100.0 * b["top3_acc_ok"] / n, 3)
        return out

    euk_keys = ["Eukaryota: Animals", "Eukaryota: Plants", "Eukaryota: Fungi", "Eukaryota: Protists"]
    euk_rollup = {
        k: sum(by_stratum[s][k] for s in euk_keys)
        for k in ("total", "grammar_ok", "strict_acc_ok", "bio_id_ok", "top3_acc_ok")
    }

    covered_accessions = {p["target_accession"] for p in prompts}

    return {
        "total_unique_corpus_proteins": len(proteins),
        "unique_corpus_proteins_covered": len(covered_accessions),
        "corpus_coverage_pct": round(100.0 * len(covered_accessions) / max(1, len(proteins)), 2),
        "total_prompts_evaluated": total,
        "wall_time_seconds": round(elapsed, 2),
        "overall": {
            "grammar_overall_accuracy_pct": round(100.0 * grammar_ok / total, 4),
            "tag_accuracy_pct": round(100.0 * tag_ok / total, 4),
            "tag_position_accuracy_pct": round(100.0 * pos_ok / total, 4),
            "cleavage_accuracy_pct": round(100.0 * cleav_ok / total, 4),
            "non_truncated_retrievable_query_pct": round(100.0 * query_clean_ok / total, 4),
            "strict_accession_match_pct": round(100.0 * strict_acc_ok / total, 4),
            "biological_identity_match_pct": round(100.0 * bio_id_ok / total, 4),
            "top3_strict_accession_pct": round(100.0 * top3_acc_ok / total, 4),
            "adversarial_decoy_trap_rate_pct": round(100.0 * decoy_trap_hits / total, 4),
            "grammar_error_count": total - grammar_ok,
            "biological_identity_mismatch_count": total - bio_id_ok,
        },
        "by_domain_stratum": {
            **{s: _fmt_bucket(by_stratum[s]) for s in CANONICAL_STRATA},
            "Eukaryota (All Kingdoms Combined)": _fmt_bucket(euk_rollup),
        },
        "by_prompt_family": {k: _fmt_bucket(v) for k, v in sorted(by_family.items())},
        "by_collision_category": {k: _fmt_bucket(v) for k, v in sorted(by_collision.items())},
        "sample_grammar_failures": grammar_failures[:40],
        "sample_ranking_failures": ranking_failures[:40],
    }


def select_stratified_live_prompts(prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Selects 640 stratified prompts across all 7 domain strata exceeding every Tier-2 minimum:
      - Bacteria: 160 (>=150)
      - Archaea: 105 (>=100)
      - Eukaryota: Animals: 105 (>=100)
      - Eukaryota: Plants: 85 (>=80)
      - Eukaryota: Fungi: 65 (>=60)
      - Eukaryota: Protists: 55 (>=50)
      - Viruses & Phages: 65 (>=60)
      Total = 640 live prompts spanning untagged, tagged+cleavage, gene, and collision classes.
    """
    quotas = {
        "Bacteria": 160,
        "Archaea": 105,
        "Eukaryota: Animals": 105,
        "Eukaryota: Plants": 85,
        "Eukaryota: Fungi": 65,
        "Eukaryota: Protists": 55,
        "Viruses & Phages": 65,
    }
    by_strat: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for p in prompts:
        by_strat[p["target_stratum"]].append(p)

    rng = random.Random(20261117)
    selected: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    for strat, quota in quotas.items():
        pool = list(by_strat.get(strat, []))
        rng.shuffle(pool)
        # Prioritize balanced mix of untagged, tagged_and_cleavage, and hard_collision prompts
        tagged = [x for x in pool if x["expected_tag"] != "None"]
        collision = [x for x in pool if x["collision_tags"] and x["expected_tag"] == "None"]
        untagged = [x for x in pool if not x["collision_tags"] and x["expected_tag"] == "None"]

        n_tag = quota // 3
        n_col = min(len(collision), quota // 5)
        n_unt = quota - n_tag - n_col

        pick = tagged[:n_tag] + collision[:n_col] + untagged[:n_unt]
        if len(pick) < quota:
            for extra in pool:
                if extra["prompt_id"] not in {z["prompt_id"] for z in pick}:
                    pick.append(extra)
                    if len(pick) >= quota:
                        break

        for item in pick[:quota]:
            if item["prompt_id"] not in seen_ids:
                seen_ids.add(item["prompt_id"])
                selected.append(item)

    return selected


def check_live_biological_match(
    res: Dict[str, Any],
    item: Dict[str, Any],
    target_rec: Dict[str, Any],
) -> bool:
    """
    Evaluates whether a live `parse_natural_language_prompt` result resolves to the exact
    intended biological protein in UniProtKB:
      - Exact accession match, OR
      - Matching species/organism AND matching normalized protein name (or alt/short name, or
        shared gene identity when prompt only supplied a gene symbol).
    """
    got_acc = (res.get("uniprot_id") or "").strip()
    canon_seq = res.get("canonical_sequence") or ""
    prompt_low = item["prompt"].lower()
    is_intentional_mature_tev = got_acc == "P04517" and len(canon_seq) == 242
    if (
        not target_rec["is_fragment"]
        and "fragment" not in prompt_low
        and not is_intentional_mature_tev
        and len(canon_seq) < 0.5 * target_rec["length"]
    ):
        return False

    if got_acc == item["target_accession"]:
        return True

    got_org = (res.get("identified_organism") or "").lower()
    tgt_root = item["target_species_root"].lower()
    tgt_sci = target_rec["organism_scientific"].lower()

    # Check species equivalence (handling strain variations or viral lineage synonyms)
    org_ok = (
        tgt_root in got_org
        or got_org in tgt_sci
        or normalize_text(tgt_root.split()[0]) in normalize_text(got_org)
    )
    if not org_ok:
        return False

    got_prot_norm = normalize_text(res.get("identified_protein") or "")
    valid_names = {
        target_rec["norm_rec_name"],
        *(normalize_text(a) for a in target_rec["alt_names"] if a),
        *(normalize_text(s) for s in target_rec["short_names"] if s),
    }
    if got_prot_norm in valid_names:
        return True

    # Mature preset suffixes (e.g. "Lysozyme C (Mature Egg-White Chain)" -> "lysozyme c")
    for vn in valid_names:
        if vn and (vn == got_prot_norm or (len(vn) >= 6 and vn in got_prot_norm)):
            # Guard against derivative hits like '-binding' or 'inhibitor' when not in target
            deriv_words = {"binding", "inhibitor", "receptor", "like", "interacting", "regulator"}
            extra_words = set(got_prot_norm.split()) - set(vn.split())
            if not (extra_words & deriv_words):
                return True

    # If prompt ONLY gave a gene symbol (e.g. 'tufA' / 'adh1' / 'hisA'), check returned gene_name first
    if item["used_gene_only"] and item["target_gene_primary"]:
        t_gp_low = item["target_gene_primary"].strip().lower()
        res_gn_low = (res.get("gene_name") or "").strip().lower()
        if res_gn_low and res_gn_low == t_gp_low:
            return True
        try:
            live_meta = sp.fetch_uniprot_by_id(got_acc)
            if (live_meta.get("gene_name") or "").strip().lower() == t_gp_low:
                return True
        except Exception:
            pass

    return False


def verify_construct_assembly(res: Dict[str, Any], item: Dict[str, Any]) -> bool:
    """Verifies exact sequence assembly of tags, cleavage sites, non-None gene_name, and canonical sequence."""
    if not isinstance(res.get("gene_name"), str) or not res.get("gene_name").strip():
        return False
    if res.get("tag_detected") != item["expected_tag"]:
        return False
    if res.get("tag_position") != item["expected_pos"]:
        return False
    if res.get("cleavage_detected") != item["expected_cleavage"]:
        return False

    canon = res.get("canonical_sequence") or ""
    full = res.get("full_construct_sequence") or ""
    if not canon or not full or res.get("length") != len(full):
        return False

    if item["expected_tag"] == "None" and item["expected_cleavage"] == "None":
        return full == canon

    return canon in full and len(full) > len(canon)


def run_tier2_live_and_reproducibility(
    live_prompts: List[Dict[str, Any]],
    index: CorpusInvertedIndex,
    max_workers: int = 8,
) -> Dict[str, Any]:
    """
    Executes >=635 live end-to-end `parse_natural_language_prompt` calls against
    `https://rest.uniprot.org` + 100 shuffled reproducibility repeats.
    """
    print(f"\n[Tier 2] Running {len(live_prompts)} stratified live UniProt queries (max_workers={max_workers})...")
    t0 = time.perf_counter()

    def _exec_one(item: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[str], float]:
        s_t = time.perf_counter()
        try:
            res = parse_natural_language_prompt(item["prompt"])
            ms = (time.perf_counter() - s_t) * 1000.0
            return item, res, None, ms
        except Exception as exc:
            ms = (time.perf_counter() - s_t) * 1000.0
            return item, None, f"{type(exc).__name__}: {exc}", ms

    run1_by_id: Dict[str, Dict[str, Any]] = {}
    latencies_ms: List[float] = []
    http_ok = 0
    grammar_construct_ok = 0
    strict_acc_ok = 0
    bio_id_ok = 0

    by_stratum: Dict[str, Dict[str, int]] = {
        s: {"total": 0, "http_ok": 0, "grammar_construct_ok": 0, "strict_acc_ok": 0, "bio_id_ok": 0}
        for s in CANONICAL_STRATA
    }
    live_failures: List[Dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_exec_one, it) for it in live_prompts]
        for done_idx, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            item, res, err, ms = fut.result()
            latencies_ms.append(ms)
            strat = item["target_stratum"]
            sb = by_stratum[strat]
            sb["total"] += 1

            if err is not None or res is None:
                live_failures.append(
                    {
                        "prompt_id": item["prompt_id"],
                        "prompt": item["prompt"],
                        "stratum": strat,
                        "target_accession": item["target_accession"],
                        "error": err,
                    }
                )
                continue

            run1_by_id[item["prompt_id"]] = res
            http_ok += 1
            sb["http_ok"] += 1

            target_rec = index.by_acc[item["target_accession"]]
            gc_ok = verify_construct_assembly(res, item)
            s_ok = res.get("uniprot_id") == item["target_accession"]
            b_ok = check_live_biological_match(res, item, target_rec)

            grammar_construct_ok += int(gc_ok)
            strict_acc_ok += int(s_ok)
            bio_id_ok += int(b_ok)
            sb["grammar_construct_ok"] += int(gc_ok)
            sb["strict_acc_ok"] += int(s_ok)
            sb["bio_id_ok"] += int(b_ok)

            if (not gc_ok or not b_ok) and len(live_failures) < 100:
                live_failures.append(
                    {
                        "prompt_id": item["prompt_id"],
                        "prompt": item["prompt"],
                        "family": item["family"],
                        "stratum": strat,
                        "expected": {
                            "accession": item["target_accession"],
                            "protein": item["target_recommended_name"],
                            "gene": item["target_gene_primary"],
                            "organism": target_rec["organism_scientific"],
                            "tag": item["expected_tag"],
                            "pos": item["expected_pos"],
                            "cleavage": item["expected_cleavage"],
                        },
                        "got": {
                            "uniprot_id": res.get("uniprot_id"),
                            "gene_name": res.get("gene_name"),
                            "identified_protein": res.get("identified_protein"),
                            "identified_organism": res.get("identified_organism"),
                            "tag_detected": res.get("tag_detected"),
                            "tag_position": res.get("tag_position"),
                            "cleavage_detected": res.get("cleavage_detected"),
                            "length": res.get("length"),
                        },
                        "grammar_construct_ok": gc_ok,
                        "biological_match_ok": b_ok,
                    }
                )

            if done_idx % 100 == 0 or done_idx == len(live_prompts):
                print(
                    f"  Live progress: {done_idx}/{len(live_prompts)} | "
                    f"HTTP={http_ok} | Grammar={grammar_construct_ok} | BioMatch={bio_id_ok}",
                    flush=True,
                )

    live_wall_s = time.perf_counter() - t0

    # Reproducibility Test: 100 stratified live prompts re-run in shuffled order
    repro_candidates = [it for it in live_prompts if it["prompt_id"] in run1_by_id]
    rng_rep = random.Random(20261118)
    by_s_rep: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for it in repro_candidates:
        by_s_rep[it["target_stratum"]].append(it)

    repro_sample: List[Dict[str, Any]] = []
    rep_quotas = {
        "Bacteria": 22,
        "Archaea": 15,
        "Eukaryota: Animals": 16,
        "Eukaryota: Plants": 14,
        "Eukaryota: Fungi": 11,
        "Eukaryota: Protists": 10,
        "Viruses & Phages": 12,
    }
    for strat, rq in rep_quotas.items():
        lst = list(by_s_rep.get(strat, []))
        rng_rep.shuffle(lst)
        repro_sample.extend(lst[:rq])
    if len(repro_sample) < 100:
        used_r = {x["prompt_id"] for x in repro_sample}
        for extra in repro_candidates:
            if extra["prompt_id"] not in used_r:
                repro_sample.append(extra)
                if len(repro_sample) >= 100:
                    break
    repro_sample = repro_sample[:100]
    rng_rep.shuffle(repro_sample)

    print(f"\n[Reproducibility] Re-running {len(repro_sample)} shuffled prompts live against UniProt...")
    t_rep0 = time.perf_counter()
    repro_exact = 0
    repro_mismatches: List[Dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs_rep = [pool.submit(_exec_one, it) for it in repro_sample]
        for fut in concurrent.futures.as_completed(futs_rep):
            item, res2, err2, _ = fut.result()
            res1 = run1_by_id[item["prompt_id"]]
            identical = (
                err2 is None
                and res2 is not None
                and res1.get("uniprot_id") == res2.get("uniprot_id")
                and res1.get("full_construct_sequence") == res2.get("full_construct_sequence")
                and res1.get("canonical_sequence") == res2.get("canonical_sequence")
                and res1.get("tag_detected") == res2.get("tag_detected")
                and res1.get("tag_position") == res2.get("tag_position")
                and res1.get("cleavage_detected") == res2.get("cleavage_detected")
            )
            if identical:
                repro_exact += 1
            else:
                repro_mismatches.append(
                    {
                        "prompt_id": item["prompt_id"],
                        "prompt": item["prompt"],
                        "run1_uniprot_id": res1.get("uniprot_id"),
                        "run2_uniprot_id": res2.get("uniprot_id") if res2 else None,
                        "run2_error": err2,
                    }
                )
    repro_wall_s = time.perf_counter() - t_rep0

    n_live = max(1, len(live_prompts))
    lat_sorted = sorted(latencies_ms)
    p95_ms = lat_sorted[min(len(lat_sorted) - 1, int(0.95 * len(lat_sorted)))] if lat_sorted else 0.0

    def _fmt_live_stratum(b: Dict[str, int]) -> Dict[str, Any]:
        tot = max(1, b["total"])
        return {
            "total_prompts": b["total"],
            "http_success_pct": round(100.0 * b["http_ok"] / tot, 3),
            "grammar_and_construct_accuracy_pct": round(100.0 * b["grammar_construct_ok"] / tot, 3),
            "strict_accession_match_pct": round(100.0 * b["strict_acc_ok"] / tot, 3),
            "biological_identity_match_pct": round(100.0 * b["bio_id_ok"] / tot, 3),
        }

    return {
        "total_live_prompts": len(live_prompts),
        "max_workers": max_workers,
        "wall_time_seconds": round(live_wall_s, 2),
        "latency_ms": {
            "mean": round(statistics.mean(latencies_ms), 1) if latencies_ms else 0.0,
            "median": round(statistics.median(latencies_ms), 1) if latencies_ms else 0.0,
            "p95": round(p95_ms, 1),
        },
        "overall": {
            "http_success_pct": round(100.0 * http_ok / n_live, 4),
            "grammar_and_construct_accuracy_pct": round(100.0 * grammar_construct_ok / n_live, 4),
            "strict_accession_match_pct": round(100.0 * strict_acc_ok / n_live, 4),
            "biological_identity_match_pct": round(100.0 * bio_id_ok / n_live, 4),
            "failure_count": len(live_prompts) - bio_id_ok,
        },
        "by_domain_stratum": {s: _fmt_live_stratum(by_stratum[s]) for s in CANONICAL_STRATA},
        "reproducibility_100_shuffled": {
            "total_retested_prompts": len(repro_sample),
            "wall_time_seconds": round(repro_wall_s, 2),
            "exact_match_count": repro_exact,
            "reproducibility_pct": round(100.0 * repro_exact / max(1, len(repro_sample)), 4),
            "mismatches": repro_mismatches,
        },
        "sample_live_failures": live_failures[:40],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run >=10,000-protein multi-domain NL search benchmark.")
    parser.add_argument(
        "--corpus",
        default=os.path.join(ROOT_DIR, "data", "nl_10k_proteins.json"),
        help="Path to 11,000+ Swiss-Prot corpus JSON.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT_DIR, "data", "nl_10k_benchmark_results.json"),
        help="Path to output benchmark results JSON.",
    )
    parser.add_argument(
        "--tier1-only",
        action="store_true",
        help="Run only offline Tier 1 >=12,500-prompt suite (skipping live HTTPS UniProt calls).",
    )
    parser.add_argument(
        "--live-workers",
        type=int,
        default=8,
        help="Max concurrent workers for polite live UniProt REST API queries (default: 8).",
    )
    args = parser.parse_args()

    print(f"[Corpus] Loading Swiss-Prot corpus from {args.corpus}...")
    proteins, corpus_meta = load_corpus(args.corpus)
    stratum_counts = collections.Counter(p["stratum"] for p in proteins)
    unique_species = len({p["tax_id"] for p in proteins})
    print(f"[Corpus] Loaded {len(proteins):,} unique Swiss-Prot proteins across {unique_species:,} taxa:")
    for s in CANONICAL_STRATA:
        print(f"  - {s:22s}: {stratum_counts[s]:,}")

    prompts = build_benchmark_prompts(proteins)
    print(f"\n[Tier 1] Generated {len(prompts):,} plain-language prompts covering 100% of corpus + collision suite.")

    index = CorpusInvertedIndex(proteins)
    tier1_results = run_tier1_suite(proteins, prompts, index)
    t1_ov = tier1_results["overall"]
    print(
        f"[Tier 1] Completed in {tier1_results['wall_time_seconds']}s:\n"
        f"  - Grammar & Construct Accuracy : {t1_ov['grammar_overall_accuracy_pct']:.3f}% "
        f"(errors: {t1_ov['grammar_error_count']})\n"
        f"  - Strict Accession Match       : {t1_ov['strict_accession_match_pct']:.3f}%\n"
        f"  - Biological Identity Match    : {t1_ov['biological_identity_match_pct']:.3f}% "
        f"(mismatches: {t1_ov['biological_identity_mismatch_count']})\n"
        f"  - Top-3 Strict Accession Recall: {t1_ov['top3_strict_accession_pct']:.3f}%\n"
        f"  - Adversarial Decoy Trap Rate  : {t1_ov['adversarial_decoy_trap_rate_pct']:.4f}%"
    )

    tier2_results: Optional[Dict[str, Any]] = None
    if not args.tier1_only:
        live_prompts = select_stratified_live_prompts(prompts)
        tier2_results = run_tier2_live_and_reproducibility(
            live_prompts=live_prompts,
            index=index,
            max_workers=args.live_workers,
        )
        t2_ov = tier2_results["overall"]
        rep = tier2_results["reproducibility_100_shuffled"]
        print(
            f"\n[Tier 2 Live Summary] ({tier2_results['total_live_prompts']} queries in {tier2_results['wall_time_seconds']}s):\n"
            f"  - HTTP Success                 : {t2_ov['http_success_pct']:.2f}%\n"
            f"  - Grammar & Construct Assembly : {t2_ov['grammar_and_construct_accuracy_pct']:.2f}%\n"
            f"  - Strict Accession Match       : {t2_ov['strict_accession_match_pct']:.2f}%\n"
            f"  - Biological Identity Match    : {t2_ov['biological_identity_match_pct']:.2f}%\n"
            f"  - 100-Prompt Reproducibility   : {rep['reproducibility_pct']:.2f}% ({rep['exact_match_count']}/{rep['total_retested_prompts']})"
        )

    full_report = {
        "benchmark_name": "ProPurify Studio >=10,000-Protein Multi-Domain Plain-Language Search & Reproducibility Benchmark",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_summary": {
            "total_unique_swissprot_proteins": len(proteins),
            "unique_taxon_ids": unique_species,
            "counts_by_stratum": {s: stratum_counts[s] for s in CANONICAL_STRATA},
            "source_metadata": corpus_meta,
        },
        "tier1_full_corpus_benchmark": tier1_results,
        "tier2_live_uniprot_benchmark": tier2_results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out_fh:
        json.dump(full_report, out_fh, indent=2)
    print(f"\n[Saved] Benchmark report written to {args.output}")


if __name__ == "__main__":
    main()
