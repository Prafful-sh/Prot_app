#!/usr/bin/env python3
"""
Stratified Multi-Domain Swiss-Prot Downloader for ProPurify Studio Benchmark.

Assembles >=11,000 unique, authentic, reviewed UniProtKB/Swiss-Prot (reviewed:true)
protein records across ALL domains of life (Bacteria, Archaea, Eukaryota [Metazoa,
Viridiplantae, Fungi, Protists], and Viruses/Bacteriophages) plus hard-collision
protein classes into `data/nl_10k_proteins.json`.

Uses UniProt REST API bulk cursor pagination (`size=500`, following `Link` headers).
Zero synthetic entries; every record is retrieved directly from https://rest.uniprot.org.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
FIELDS_PRIMARY = (
    "accession,id,protein_name,gene_names,organism_name,organism_id,"
    "lineage,length,annotation_score,fragment"
)
FIELDS_FALLBACK = (
    "accession,id,protein_name,gene_names,organism_name,organism_id,"
    "lineage,length,annotation_score"
)

VALID_DOMAINS = {"Bacteria", "Archaea", "Eukaryota", "Viruses"}

GENERIC_UNCHARACTERIZED_PATTERNS = re.compile(
    r"^(?:uncharacterized\s+protein|putative\s+uncharacterized\s+protein|"
    r"hypothetical\s+protein|probable\s+uncharacterized\s+protein|"
    r"upf\d+\s+protein\b|duf\d+\s+domain-containing\s+protein$)",
    flags=re.IGNORECASE,
)

# Canonical accessions referenced in sequence_parser.py (always keep)
CANONICAL_REFERENCE_ACCESSIONS: Set[str] = {
    "P00698", "P42212", "P00918", "P02185", "P37957", "P02769",
    "P09211", "P08263", "P09488", "P08515", "P04637", "P0AEX9",
    "P02686", "P01106", "P00734", "P04517", "P61626", "P22629",
}


def build_session() -> requests.Session:
    """Builds a polite HTTP session with connection pooling and exponential retry."""
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=4,
        read=4,
        status=5,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "ProPurifyStudio-SwissProt10kBenchmark/1.0 (Python requests)",
        }
    )
    return session


def _dedup_preserve_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for raw in items:
        val = (raw or "").strip()
        if val and val not in seen:
            seen.add(val)
            out.append(val)
    return out


def classify_domain_and_subkingdom(
    lineage: List[str], fallback_domain: Optional[str] = None
) -> Tuple[Optional[str], str]:
    """Determines canonical top-level domain and eukaryotic kingdom / sub-clade from UniProt lineage."""
    clean_lin = [x.strip() for x in (lineage or []) if x and x.strip()]
    domain: Optional[str] = None

    for item in clean_lin:
        if item in VALID_DOMAINS:
            domain = item
            break
        if item == "Viruses" or item.endswith("viria") or "phage" in item.lower():
            domain = "Viruses"
            break

    if domain is None and fallback_domain in VALID_DOMAINS:
        domain = fallback_domain

    if domain is None:
        return None, "Unknown"

    if domain == "Eukaryota":
        lin_set = set(clean_lin)
        if "Metazoa" in lin_set:
            return "Eukaryota", "Metazoa"
        if "Viridiplantae" in lin_set:
            return "Eukaryota", "Viridiplantae"
        if "Fungi" in lin_set:
            return "Eukaryota", "Fungi"
        return "Eukaryota", "Protists"

    # For Bacteria, Archaea, Viruses: use immediate major phylum/realm from lineage if present
    subkingdom = clean_lin[1] if len(clean_lin) >= 2 else domain
    return domain, subkingdom


def parse_uniprot_record(
    entry: Dict[str, Any],
    fallback_domain: Optional[str] = None,
    allow_uncharacterized: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Converts a raw UniProtKB JSON search hit into the compact benchmark record schema.
    Returns None if mandatory fields are missing or generic uncharacterized filter triggers.
    """
    acc = (entry.get("primaryAccession") or "").strip()
    uid = (entry.get("uniProtkbId") or "").strip()
    if not acc or not uid:
        return None

    # Enforce Swiss-Prot reviewed status if entryType is present in response
    entry_type = entry.get("entryType") or ""
    if entry_type and "unreviewed" in entry_type.lower():
        return None

    pdesc = entry.get("proteinDescription") or {}
    rec_block = pdesc.get("recommendedName") or {}
    sub_list = pdesc.get("submissionNames") or []

    rec_name = (rec_block.get("fullName") or {}).get("value", "").strip()
    if not rec_name and sub_list:
        rec_name = (sub_list[0].get("fullName") or {}).get("value", "").strip()
    if not rec_name:
        return None

    if (
        not allow_uncharacterized
        and acc not in CANONICAL_REFERENCE_ACCESSIONS
        and GENERIC_UNCHARACTERIZED_PATTERNS.search(rec_name)
    ):
        return None

    alt_blocks = pdesc.get("alternativeNames") or []
    alt_names = _dedup_preserve_order(
        (alt.get("fullName") or {}).get("value", "") for alt in alt_blocks
    )

    short_candidates: List[str] = []
    for s in rec_block.get("shortNames") or []:
        if s.get("value"):
            short_candidates.append(s["value"])
    for alt in alt_blocks:
        for s in alt.get("shortNames") or []:
            if s.get("value"):
                short_candidates.append(s["value"])
    short_names = _dedup_preserve_order(short_candidates)

    genes_raw = entry.get("genes") or []
    gene_primary = ""
    syn_candidates: List[str] = []
    for g in genes_raw:
        g_val = (g.get("geneName") or {}).get("value", "").strip()
        if g_val:
            if not gene_primary:
                gene_primary = g_val
            else:
                syn_candidates.append(g_val)
        for syn in g.get("synonyms") or []:
            s_val = (syn.get("value") or "").strip()
            if s_val:
                syn_candidates.append(s_val)
    gene_synonyms = [s for s in _dedup_preserve_order(syn_candidates) if s != gene_primary]

    org = entry.get("organism") or {}
    org_sci = (org.get("scientificName") or "").strip()
    org_com = (org.get("commonName") or "").strip()
    tax_id = int(org.get("taxonId") or 0)
    if not org_sci or tax_id <= 0:
        return None

    lineage = org.get("lineage") or []
    if "synthetic" in org_sci.lower() or any("artificial" in str(x).lower() or "other sequences" in str(x).lower() for x in lineage):
        return None
    domain, subkingdom = classify_domain_and_subkingdom(lineage, fallback_domain=fallback_domain)
    if domain not in VALID_DOMAINS:
        return None

    seq_block = entry.get("sequence") or {}
    length = int(seq_block.get("length") or entry.get("length") or 0)
    if length <= 0:
        return None

    annot_score = float(entry.get("annotationScore") or 0.0)

    flag_str = (pdesc.get("flag") or "").lower()
    is_fragment = bool(
        "fragment" in flag_str
        or seq_block.get("fragment")
        or (entry.get("fragment") is True)
    )

    return {
        "accession": acc,
        "uniProtkbId": uid,
        "recommended_name": rec_name,
        "alt_names": alt_names,
        "short_names": short_names,
        "gene_primary": gene_primary,
        "gene_synonyms": gene_synonyms,
        "organism_scientific": org_sci,
        "organism_common": org_com,
        "tax_id": tax_id,
        "domain": domain,
        "subkingdom": subkingdom,
        "length": length,
        "annotation_score": annot_score,
        "is_fragment": is_fragment,
    }


def extract_next_link(resp: requests.Response) -> Optional[str]:
    """Extracts the cursor-paginated next URL from RFC 5988 Link headers."""
    if "next" in resp.links and resp.links["next"].get("url"):
        return resp.links["next"]["url"]
    raw_link = resp.headers.get("Link") or resp.headers.get("link") or ""
    m = re.search(r'<([^>]+)>;\s*rel="next"', raw_link)
    return m.group(1) if m else None


def normalize_species_binomial(scientific_name: str) -> str:
    """Strips strain/isolate parenthetical suffixes to count unique biological species."""
    base = re.sub(
        r"\s*\((?:strain|isolate|serovar|serotype|subsp\.|biovar|pv\.|ecotype)\b.*",
        "",
        scientific_name.strip(),
        flags=re.IGNORECASE,
    )
    return " ".join(base.split())


class CorpusCollector:
    """Manages deduplicated protein collection, per-taxon diversity caps, and HTTP telemetry."""

    def __init__(self) -> None:
        self.records_by_acc: Dict[str, Dict[str, Any]] = {}
        self.taxid_counts: Counter[int] = Counter()
        self.domain_counts: Counter[str] = Counter()
        self.subkingdom_counts: Counter[str] = Counter()
        self.http_requests: int = 0
        self.fields_param: str = FIELDS_PRIMARY

    def add_entry(
        self,
        raw_entry: Dict[str, Any],
        fallback_domain: Optional[str] = None,
        max_per_taxid: Optional[int] = None,
        bypass_cap_for_canonical: bool = True,
    ) -> bool:
        acc = (raw_entry.get("primaryAccession") or "").strip()
        if not acc or acc in self.records_by_acc:
            return False

        parsed = parse_uniprot_record(raw_entry, fallback_domain=fallback_domain)
        if parsed is None:
            return False

        tid = parsed["tax_id"]
        if (
            max_per_taxid is not None
            and self.taxid_counts[tid] >= max_per_taxid
            and not (bypass_cap_for_canonical and acc in CANONICAL_REFERENCE_ACCESSIONS)
        ):
            return False

        self.records_by_acc[acc] = parsed
        self.taxid_counts[tid] += 1
        self.domain_counts[parsed["domain"]] += 1
        self.subkingdom_counts[parsed["subkingdom"]] += 1
        return True

    def fetch_query_pages(
        self,
        session: requests.Session,
        label: str,
        query: str,
        max_pages: int = 1,
        page_size: int = 500,
        fallback_domain: Optional[str] = None,
        max_per_taxid: Optional[int] = None,
        stop_predicate: Optional[Callable[["CorpusCollector"], bool]] = None,
        polite_delay_s: float = 0.12,
    ) -> int:
        """Executes a UniProt search query across up to `max_pages` cursor pages."""
        added_total = 0
        next_url: Optional[str] = None

        for page_idx in range(1, max_pages + 1):
            if stop_predicate is not None and stop_predicate(self):
                break

            if page_idx == 1:
                params = {
                    "query": query,
                    "fields": self.fields_param,
                    "size": str(page_size),
                    "format": "json",
                }
                resp = session.get(UNIPROT_SEARCH_URL, params=params, timeout=30)
                if resp.status_code == 400 and self.fields_param != FIELDS_FALLBACK:
                    self.fields_param = FIELDS_FALLBACK
                    params["fields"] = self.fields_param
                    resp = session.get(UNIPROT_SEARCH_URL, params=params, timeout=30)
            else:
                assert next_url is not None
                resp = session.get(next_url, timeout=30)

            self.http_requests += 1

            if resp.status_code != 200:
                print(
                    f"  [WARN] {label} page {page_idx} returned HTTP {resp.status_code}: "
                    f"{resp.text[:200]}",
                    file=sys.stderr,
                )
                break

            payload = resp.json()
            results = payload.get("results") or []
            page_added = 0
            for item in results:
                if self.add_entry(
                    item,
                    fallback_domain=fallback_domain,
                    max_per_taxid=max_per_taxid,
                ):
                    page_added += 1
            added_total += page_added

            print(
                f"  [{self.http_requests:02d}] {label} (p.{page_idx}): "
                f"fetched={len(results)}, new_added={page_added}, "
                f"corpus_total={len(self.records_by_acc)}"
            )

            next_url = extract_next_link(resp)
            if not next_url or len(results) < page_size:
                break

            if polite_delay_s > 0:
                time.sleep(polite_delay_s)

        return added_total


def compute_collision_counts(proteins: List[Dict[str, Any]]) -> Dict[str, int]:
    """Computes representation counts for all hard-collision protein classes."""
    his_genes = {"hisa", "hisb", "hisc", "hisd", "hisf", "hisg", "hish", "hisi", "hiss"}
    picorna_orgs = (
        "rhinovirus", "poliovirus", "enterovirus", "encephalomyocarditis",
        "hepatovirus", "aphthovirus", "foot-and-mouth", "coxsackie", "cardiovirus",
    )
    counts = {
        "histidine_pathway_his": 0,
        "flagellin_fliC": 0,
        "viral_hemagglutinin_HA": 0,
        "viral_3C_protease_and_picornaviruses": 0,
        "genes_with_3C": 0,
        "gst_family": 0,
        "mbp_family": 0,
        "myc_family": 0,
        "streptavidin_avidin": 0,
        "name_contains_with": 0,
        "name_contains_plus": 0,
        "name_contains_and": 0,
        "name_contains_containing": 0,
    }

    for p in proteins:
        gp = p["gene_primary"]
        gp_low = gp.lower()
        all_genes_low = ({gp_low} if gp_low else set()) | {g.lower() for g in p["gene_synonyms"]}
        rec = p["recommended_name"]
        rec_low = rec.lower()
        org_low = p["organism_scientific"].lower()
        all_names_low = " ; ".join(
            [rec_low] + [a.lower() for a in p["alt_names"]] + [s.lower() for s in p["short_names"]]
        )

        if all_genes_low & his_genes:
            counts["histidine_pathway_his"] += 1
        if "flic" in all_genes_low or "flagellin" in all_names_low:
            counts["flagellin_fliC"] += 1
        if p["domain"] == "Viruses" and ("hemagglutinin" in all_names_low or "ha" in all_genes_low):
            counts["viral_hemagglutinin_HA"] += 1
        if p["domain"] == "Viruses" and (
            re.search(r"\b3c(?:-like)?\b", all_names_low)
            or "picornain" in all_names_low
            or ("polyprotein" in rec_low and any(v in org_low for v in picorna_orgs))
        ):
            counts["viral_3C_protease_and_picornaviruses"] += 1
        if any(len(g) >= 3 and (g.endswith("3c") or g.startswith("h3c")) for g in all_genes_low):
            counts["genes_with_3C"] += 1
        if "glutathione s-transferase" in all_names_low or gp_low.startswith("gst"):
            counts["gst_family"] += 1
        if (
            "maltose/maltodextrin-binding" in all_names_low
            or "maltose-binding" in all_names_low
            or "myelin basic protein" in all_names_low
            or gp_low in {"male", "mbp"}
        ):
            counts["mbp_family"] += 1
        if gp_low in {"myc", "mycn", "mycl"} or "myc proto-oncogene" in all_names_low:
            counts["myc_family"] += 1
        if "streptavidin" in all_names_low or "avidin" in rec_low:
            counts["streptavidin_avidin"] += 1

        if re.search(r"\bwith\b", all_names_low):
            counts["name_contains_with"] += 1
        if re.search(r"\bplus\b", all_names_low):
            counts["name_contains_plus"] += 1
        if re.search(r"\band\b", rec_low):
            counts["name_contains_and"] += 1
        if "containing" in rec_low:
            counts["name_contains_containing"] += 1

    return counts


# Required star clades across all domains of life: (label, domain, match_substrings_in_sci_name, topup_query, min_desired)
REQUIRED_STAR_CLADES: List[Tuple[str, str, Tuple[str, ...], str, int]] = [
    # Bacteria
    ("Escherichia coli", "Bacteria", ("escherichia coli",), "taxonomy_id:562", 50),
    ("Bacillus subtilis", "Bacteria", ("bacillus subtilis",), "taxonomy_id:1423", 40),
    ("Staphylococcus aureus", "Bacteria", ("staphylococcus aureus",), "taxonomy_id:1280", 30),
    ("Mycobacterium tuberculosis", "Bacteria", ("mycobacterium tuberculosis",), "taxonomy_id:1773", 30),
    ("Pseudomonas aeruginosa", "Bacteria", ("pseudomonas aeruginosa",), "taxonomy_id:287", 30),
    ("Thermus aquaticus", "Bacteria", ("thermus aquaticus",), "taxonomy_id:271", 15),
    ("Streptococcus pyogenes", "Bacteria", ("streptococcus pyogenes",), "taxonomy_id:1314", 20),
    ("Synechocystis", "Bacteria", ("synechocystis",), "taxonomy_id:1142", 25),
    ("Vibrio cholerae", "Bacteria", ("vibrio cholerae",), "taxonomy_id:666", 25),
    ("Helicobacter pylori", "Bacteria", ("helicobacter pylori",), "taxonomy_id:210", 25),
    ("Streptomyces coelicolor", "Bacteria", ("streptomyces coelicolor",), "taxonomy_id:1902", 25),
    ("Caulobacter", "Bacteria", ("caulobacter",), "taxonomy_id:75", 20),
    # Archaea
    ("Methanocaldococcus jannaschii", "Archaea", ("methanocaldococcus jannaschii",), "taxonomy_id:2190", 40),
    ("Sulfolobus acidocaldarius", "Archaea", ("sulfolobus acidocaldarius",), "taxonomy_id:2285", 20),
    ("Saccharolobus solfataricus", "Archaea", ("saccharolobus solfataricus", "sulfolobus solfataricus"), "taxonomy_id:2287", 25),
    ("Pyrococcus furiosus", "Archaea", ("pyrococcus furiosus",), "taxonomy_id:2261", 30),
    ("Thermococcus kodakarensis", "Archaea", ("thermococcus kodakarensis",), "taxonomy_id:69014", 25),
    ("Haloferax volcanii", "Archaea", ("haloferax volcanii",), "taxonomy_id:2246", 20),
    ("Halobacterium salinarum", "Archaea", ("halobacterium salinarum",), "taxonomy_id:2242", 25),
    ("Archaeoglobus fulgidus", "Archaea", ("archaeoglobus fulgidus",), "taxonomy_id:2234", 25),
    ("Nitrosopumilus", "Archaea", ("nitrosopumilus",), "(taxonomy_id:338190 OR taxonomy_id:436308)", 10),
    # Animals / Metazoa
    ("Homo sapiens", "Eukaryota", ("homo sapiens",), "taxonomy_id:9606", 80),
    ("Mus musculus", "Eukaryota", ("mus musculus",), "taxonomy_id:10090", 50),
    ("Rattus norvegicus", "Eukaryota", ("rattus norvegicus",), "taxonomy_id:10116", 35),
    ("Gallus gallus", "Eukaryota", ("gallus gallus",), "taxonomy_id:9031", 25),
    ("Bos taurus", "Eukaryota", ("bos taurus",), "taxonomy_id:9913", 30),
    ("Danio rerio", "Eukaryota", ("danio rerio",), "taxonomy_id:7955", 30),
    ("Drosophila melanogaster", "Eukaryota", ("drosophila melanogaster",), "taxonomy_id:7227", 40),
    ("Caenorhabditis elegans", "Eukaryota", ("caenorhabditis elegans",), "taxonomy_id:6239", 40),
    ("Xenopus laevis", "Eukaryota", ("xenopus laevis",), "taxonomy_id:8355", 25),
    # Plants & Green Algae / Viridiplantae
    ("Arabidopsis thaliana", "Eukaryota", ("arabidopsis thaliana",), "taxonomy_id:3702", 60),
    ("Oryza sativa", "Eukaryota", ("oryza sativa",), "taxonomy_id:4530", 40),
    ("Zea mays", "Eukaryota", ("zea mays",), "taxonomy_id:4577", 25),
    ("Spinacia oleracea", "Eukaryota", ("spinacia oleracea",), "taxonomy_id:3562", 20),
    ("Solanum lycopersicum", "Eukaryota", ("solanum lycopersicum",), "taxonomy_id:4081", 20),
    ("Chlamydomonas reinhardtii", "Eukaryota", ("chlamydomonas reinhardtii",), "taxonomy_id:3055", 25),
    # Fungi
    ("Saccharomyces cerevisiae", "Eukaryota", ("saccharomyces cerevisiae",), "taxonomy_id:4932", 60),
    ("Schizosaccharomyces pombe", "Eukaryota", ("schizosaccharomyces pombe",), "taxonomy_id:4896", 50),
    ("Neurospora crassa", "Eukaryota", ("neurospora crassa",), "taxonomy_id:5141", 25),
    ("Aspergillus", "Eukaryota", ("aspergillus",), "taxonomy_id:5052", 30),
    ("Candida albicans", "Eukaryota", ("candida albicans",), "taxonomy_id:5476", 25),
    # Protists
    ("Plasmodium falciparum", "Eukaryota", ("plasmodium falciparum",), "(taxonomy_id:5833 OR taxonomy_id:36329)", 25),
    ("Trypanosoma brucei", "Eukaryota", ("trypanosoma brucei",), "taxonomy_id:5691", 25),
    ("Dictyostelium discoideum", "Eukaryota", ("dictyostelium discoideum",), "taxonomy_id:44689", 40),
    ("Tetrahymena thermophila", "Eukaryota", ("tetrahymena thermophila",), "taxonomy_id:5911", 20),
    ("Diatoms (Thalassiosira/Phaeodactylum)", "Eukaryota", ("thalassiosira", "phaeodactylum"), "(taxonomy_id:35127 OR taxonomy_id:2850)", 10),
    # Viruses & Phages
    ("Bacteriophage T4", "Viruses", ("phage t4", "tequatrovirus t4"), '(taxonomy_id:10665 OR organism_name:"phage T4")', 20),
    ("Bacteriophage T7", "Viruses", ("phage t7", "teseptimavirus t7"), '(taxonomy_id:10760 OR organism_name:"phage T7")', 15),
    ("Phage Lambda", "Viruses", ("phage lambda", "lambdavirus lambda"), '(taxonomy_id:10710 OR taxonomy_id:2681611 OR organism_name:"phage lambda")', 15),
    ("Tobacco etch virus", "Viruses", ("tobacco etch virus",), "taxonomy_id:12227", 1),
    ("SARS-CoV-2", "Viruses", ("severe acute respiratory syndrome coronavirus 2", "sars-cov-2"), "taxonomy_id:2697049", 10),
    ("Influenza A virus", "Viruses", ("influenza a virus",), "taxonomy_id:11320", 25),
    ("HIV-1", "Viruses", ("human immunodeficiency virus type 1", "human immunodeficiency virus 1", "hiv-1"), "taxonomy_id:11676", 15),
    ("Vaccinia virus", "Viruses", ("vaccinia virus",), "taxonomy_id:10245", 25),
    ("Herpesvirus", "Viruses", ("herpesvirus", "simplexvirus", "cytomegalovirus", "epstein-barr", "hhv-"), '(taxonomy_id:10239 AND (organism_name:"herpesvirus" OR organism_name:"Epstein-Barr"))', 25),
    ("Tobacco mosaic virus", "Viruses", ("tobacco mosaic virus",), "taxonomy_id:12242", 4),
    ("Sulfolobus viruses", "Viruses", ("sulfolobus", "saccharolobus"), '(taxonomy_id:10239 AND organism_name:"Sulfolobus")', 5),
]


def count_clade_matches(proteins: Iterable[Dict[str, Any]], substrings: Tuple[str, ...], domain: str) -> int:
    c = 0
    for p in proteins:
        if p["domain"] != domain:
            continue
        sci_low = p["organism_scientific"].lower()
        com_low = p["organism_common"].lower()
        if any(sub in sci_low or sub in com_low for sub in substrings):
            c += 1
    return c


def download_full_corpus(output_path: Path) -> Dict[str, Any]:
    """Downloads >=11,500 stratified Swiss-Prot proteins across all domains of life."""
    session = build_session()
    collector = CorpusCollector()
    base_clean = '(reviewed:true) AND NOT (fragment:true) AND NOT (protein_name:"Uncharacterized protein")'

    print("=== Phase 1: Hard-Collision & Canonical Reference Sweeps ===")
    collision_queries = [
        (
            "Collisions-1-Canonical-GST-MBP-Myc-Strep",
            '(reviewed:true) AND (accession:P00698 OR accession:P42212 OR accession:P00918 '
            'OR accession:P02185 OR accession:P37957 OR accession:P02769 OR accession:P09211 '
            'OR accession:P08263 OR accession:P09488 OR accession:P08515 OR accession:P04637 '
            'OR accession:P0AEX9 OR accession:P02686 OR accession:P01106 OR accession:P00734 '
            'OR accession:P04517 OR accession:P61626 OR accession:P22629 '
            'OR protein_name:"Streptavidin" OR protein_name:"Glutathione S-transferase" '
            'OR gene:malE OR gene:MBP OR protein_name:"Myelin basic protein" '
            'OR gene:MYC OR protein_name:"Myc proto-oncogene")',
        ),
        (
            "Collisions-2-HisPathway-Flagellin",
            f"{base_clean} AND (gene:hisA OR gene:hisB OR gene:hisC OR gene:hisD OR gene:hisF "
            "OR gene:hisG OR gene:hisH OR gene:hisI OR gene:hisS OR gene:fliC "
            'OR protein_name:"Flagellin")',
        ),
        (
            "Collisions-3-HA-3CProtease-3CGenes",
            f'{base_clean} AND ((taxonomy_id:10239 AND (protein_name:"Hemagglutinin" '
            'OR protein_name:"3C" OR protein_name:"picornain" OR gene:HA)) '
            "OR gene:H3C1 OR gene:H3C2 OR gene:H3C4 OR gene:RAB3C OR gene:UBE3C "
            "OR gene:KIF3C OR gene:FAM3C OR gene:UNC13C OR gene:VPS33C OR gene:PPP3CA)",
        ),
        (
            "Collisions-4-With-Plus-ConnectorNames",
            f'{base_clean} AND (protein_name:"with" OR protein_name:"plus")',
        ),
    ]
    for label, q in collision_queries:
        collector.fetch_query_pages(session, label, q, max_pages=1, page_size=500, max_per_taxid=250)

    print("\n=== Phase 2: Bacteria (Target >= 3,450 across diverse phyla) ===")
    collector.fetch_query_pages(
        session,
        "Bacteria-SmallStarClades",
        f"{base_clean} AND (taxonomy_id:271 OR taxonomy_id:274 OR taxonomy_id:1314 "
        "OR taxonomy_id:75 OR taxonomy_id:210 OR taxonomy_id:1142)",
        max_pages=2,
        page_size=500,
        fallback_domain="Bacteria",
        max_per_taxid=220,
    )
    collector.fetch_query_pages(
        session,
        "Bacteria-MidStarClades",
        f"{base_clean} AND (taxonomy_id:287 OR taxonomy_id:1280 OR taxonomy_id:1773 "
        "OR taxonomy_id:666 OR taxonomy_id:1902)",
        max_pages=2,
        page_size=500,
        fallback_domain="Bacteria",
        max_per_taxid=240,
    )
    collector.fetch_query_pages(
        session,
        "Bacteria-Ecoli-Bsubtilis",
        f"{base_clean} AND (taxonomy_id:562 OR taxonomy_id:1423)",
        max_pages=2,
        page_size=500,
        fallback_domain="Bacteria",
        max_per_taxid=380,
    )
    collector.fetch_query_pages(
        session,
        "Bacteria-BroadPhyla",
        f"{base_clean} AND (taxonomy_id:2) AND NOT (taxonomy_id:562) AND NOT (taxonomy_id:1423)",
        max_pages=5,
        page_size=500,
        fallback_domain="Bacteria",
        max_per_taxid=160,
        stop_predicate=lambda c: c.domain_counts["Bacteria"] >= 3450,
    )

    print("\n=== Phase 3: Archaea (Target >= 2,150 across Cren/Eury/Thaumarchaeota) ===")
    collector.fetch_query_pages(
        session,
        "Archaea-Sulfolobus-Halophiles-Thaum",
        f"{base_clean} AND (taxonomy_id:2285 OR taxonomy_id:2287 OR taxonomy_id:2246 "
        "OR taxonomy_id:2242 OR taxonomy_id:338190 OR taxonomy_id:436308)",
        max_pages=2,
        page_size=500,
        fallback_domain="Archaea",
        max_per_taxid=260,
    )
    collector.fetch_query_pages(
        session,
        "Archaea-Hyperthermophiles",
        f"{base_clean} AND (taxonomy_id:2261 OR taxonomy_id:69014 OR taxonomy_id:2234 OR taxonomy_id:2190)",
        max_pages=2,
        page_size=500,
        fallback_domain="Archaea",
        max_per_taxid=320,
    )
    collector.fetch_query_pages(
        session,
        "Archaea-Broad",
        f"{base_clean} AND (taxonomy_id:2157) AND NOT (taxonomy_id:2190) AND NOT (taxonomy_id:2261)",
        max_pages=8,
        page_size=500,
        fallback_domain="Archaea",
        max_per_taxid=350,
        stop_predicate=lambda c: c.domain_counts["Archaea"] >= 2150,
    )

    print("\n=== Phase 4: Eukaryota (Target >= 4,600 across Metazoa, Viridiplantae, Fungi, Protists) ===")
    # 4a. Metazoa (Target >= 1,650)
    collector.fetch_query_pages(
        session,
        "Metazoa-NonMammalModels",
        f"{base_clean} AND (taxonomy_id:9031 OR taxonomy_id:7955 OR taxonomy_id:7227 "
        "OR taxonomy_id:6239 OR taxonomy_id:8355 OR taxonomy_id:6100 OR taxonomy_id:6182 OR taxonomy_id:9755)",
        max_pages=2,
        page_size=500,
        fallback_domain="Eukaryota",
        max_per_taxid=220,
    )
    collector.fetch_query_pages(
        session,
        "Metazoa-NonHumanMammals",
        f"{base_clean} AND (taxonomy_id:10090 OR taxonomy_id:10116 OR taxonomy_id:9913 "
        "OR taxonomy_id:9823 OR taxonomy_id:9986)",
        max_pages=1,
        page_size=500,
        fallback_domain="Eukaryota",
        max_per_taxid=240,
    )
    collector.fetch_query_pages(
        session,
        "Metazoa-Human",
        f"{base_clean} AND (taxonomy_id:9606)",
        max_pages=1,
        page_size=500,
        fallback_domain="Eukaryota",
        max_per_taxid=380,
    )
    collector.fetch_query_pages(
        session,
        "Metazoa-BroadOther",
        f"{base_clean} AND (taxonomy_id:33208) AND NOT (taxonomy_id:9606) AND NOT (taxonomy_id:10090)",
        max_pages=4,
        page_size=500,
        fallback_domain="Eukaryota",
        max_per_taxid=240,
        stop_predicate=lambda c: c.subkingdom_counts["Metazoa"] >= 1650,
    )

    # 4b. Viridiplantae (Target >= 1,100)
    collector.fetch_query_pages(
        session,
        "Plants-CropsSpinachAlgae",
        f"{base_clean} AND (taxonomy_id:4530 OR taxonomy_id:4577 OR taxonomy_id:3562 "
        "OR taxonomy_id:4081 OR taxonomy_id:3055)",
        max_pages=2,
        page_size=500,
        fallback_domain="Eukaryota",
        max_per_taxid=250,
    )
    collector.fetch_query_pages(
        session,
        "Plants-Arabidopsis",
        f"{base_clean} AND (taxonomy_id:3702)",
        max_pages=1,
        page_size=500,
        fallback_domain="Eukaryota",
        max_per_taxid=350,
    )
    collector.fetch_query_pages(
        session,
        "Plants-BroadOther",
        f"{base_clean} AND (taxonomy_id:33090) AND NOT (taxonomy_id:3702) AND NOT (taxonomy_id:4530)",
        max_pages=4,
        page_size=500,
        fallback_domain="Eukaryota",
        max_per_taxid=220,
        stop_predicate=lambda c: c.subkingdom_counts["Viridiplantae"] >= 1100,
    )

    # 4c. Fungi (Target >= 1,050)
    collector.fetch_query_pages(
        session,
        "Fungi-FilamentousAndPathogenic",
        f"{base_clean} AND (taxonomy_id:5141 OR taxonomy_id:5052 OR taxonomy_id:5476)",
        max_pages=1,
        page_size=500,
        fallback_domain="Eukaryota",
        max_per_taxid=220,
    )
    collector.fetch_query_pages(
        session,
        "Fungi-Scerevisiae",
        f"{base_clean} AND (taxonomy_id:4932)",
        max_pages=1,
        page_size=500,
        fallback_domain="Eukaryota",
        max_per_taxid=340,
    )
    collector.fetch_query_pages(
        session,
        "Fungi-SpombeAndBroad",
        f"{base_clean} AND (taxonomy_id:4751) AND NOT (taxonomy_id:4932)",
        max_pages=4,
        page_size=500,
        fallback_domain="Eukaryota",
        max_per_taxid=320,
        stop_predicate=lambda c: c.subkingdom_counts["Fungi"] >= 1050,
    )

    # 4d. Protists (Target >= 800)
    collector.fetch_query_pages(
        session,
        "Protists-ParasitesCiliatesDiatoms",
        f"{base_clean} AND (taxonomy_id:5833 OR taxonomy_id:36329 OR taxonomy_id:5691 "
        "OR taxonomy_id:5911 OR taxonomy_id:35127 OR taxonomy_id:2850)",
        max_pages=2,
        page_size=500,
        fallback_domain="Eukaryota",
        max_per_taxid=250,
    )
    collector.fetch_query_pages(
        session,
        "Protists-Dictyostelium",
        f"{base_clean} AND (taxonomy_id:44689)",
        max_pages=1,
        page_size=500,
        fallback_domain="Eukaryota",
        max_per_taxid=280,
    )
    collector.fetch_query_pages(
        session,
        "Protists-BroadOther",
        f"{base_clean} AND (taxonomy_id:2759) AND NOT (taxonomy_id:33208) "
        "AND NOT (taxonomy_id:33090) AND NOT (taxonomy_id:4751) AND NOT (taxonomy_id:44689)",
        max_pages=4,
        page_size=500,
        fallback_domain="Eukaryota",
        max_per_taxid=220,
        stop_predicate=lambda c: c.subkingdom_counts["Protists"] >= 800,
    )

    print("\n=== Phase 5: Viruses & Bacteriophages (Target >= 1,650) ===")
    collector.fetch_query_pages(
        session,
        "Viruses-PhagesPlantSARSArchaea",
        f'{base_clean} AND (taxonomy_id:10239) AND (taxonomy_id:10665 OR taxonomy_id:10760 '
        'OR taxonomy_id:10710 OR taxonomy_id:2681611 OR taxonomy_id:12227 OR taxonomy_id:12242 '
        'OR taxonomy_id:2697049 OR organism_name:"phage T4" OR organism_name:"phage T7" '
        'OR organism_name:"phage lambda" OR organism_name:"Sulfolobus")',
        max_pages=1,
        page_size=500,
        fallback_domain="Viruses",
        max_per_taxid=220,
    )
    collector.fetch_query_pages(
        session,
        "Viruses-InfluenzaHIVVacciniaHerpesPicorna",
        f"{base_clean} AND (taxonomy_id:11320 OR taxonomy_id:11676 OR taxonomy_id:10245 "
        "OR taxonomy_id:10292 OR taxonomy_id:12058)",
        max_pages=2,
        page_size=500,
        fallback_domain="Viruses",
        max_per_taxid=220,
    )
    collector.fetch_query_pages(
        session,
        "Viruses-Broad",
        f"{base_clean} AND (taxonomy_id:10239)",
        max_pages=7,
        page_size=500,
        fallback_domain="Viruses",
        max_per_taxid=220,
        stop_predicate=lambda c: c.domain_counts["Viruses"] >= 1650,
    )

    print("\n=== Phase 6: Star-Clade Coverage Audit & Auto Top-Up ===")
    for clade_name, dom, match_subs, topup_clause, min_desired in REQUIRED_STAR_CLADES:
        current_n = count_clade_matches(collector.records_by_acc.values(), match_subs, dom)
        if current_n < min_desired:
            q_top = f'(reviewed:true) AND NOT (protein_name:"Uncharacterized protein") AND ({topup_clause})'
            collector.fetch_query_pages(
                session,
                f"TopUp-{clade_name}",
                q_top,
                max_pages=1,
                page_size=max(100, min_desired * 2),
                fallback_domain=dom,
                max_per_taxid=450,
            )

    # Sort deterministically by domain, subkingdom, tax_id, accession for 100% reproducible ordering
    proteins_sorted = sorted(
        collector.records_by_acc.values(),
        key=lambda r: (r["domain"], r["subkingdom"], r["tax_id"], r["accession"]),
    )

    # Strict post-conditions
    total_proteins = len(proteins_sorted)
    unique_accs = {p["accession"] for p in proteins_sorted}
    assert len(unique_accs) == total_proteins, "Duplicate accessions detected!"
    assert total_proteins >= 11000, f"Expected >=11,000 proteins, got {total_proteins}"

    dom_summary = {
        d: sum(1 for p in proteins_sorted if p["domain"] == d)
        for d in ("Bacteria", "Archaea", "Eukaryota", "Viruses")
    }
    assert dom_summary["Bacteria"] >= 3200, f"Bacteria < 3200: {dom_summary['Bacteria']}"
    assert dom_summary["Archaea"] >= 2000, f"Archaea < 2000: {dom_summary['Archaea']}"
    assert dom_summary["Eukaryota"] >= 4300, f"Eukaryota < 4300: {dom_summary['Eukaryota']}"
    assert dom_summary["Viruses"] >= 1500, f"Viruses < 1500: {dom_summary['Viruses']}"

    euk_kingdom_summary = {
        k: sum(1 for p in proteins_sorted if p["domain"] == "Eukaryota" and p["subkingdom"] == k)
        for k in ("Metazoa", "Viridiplantae", "Fungi", "Protists")
    }
    for k, cnt in euk_kingdom_summary.items():
        assert cnt >= 300, f"Eukaryotic kingdom {k} under-represented: {cnt}"

    unique_species = {normalize_species_binomial(p["organism_scientific"]) for p in proteins_sorted}
    unique_tax_ids = {p["tax_id"] for p in proteins_sorted}
    collision_summary = compute_collision_counts(proteins_sorted)
    clade_audit = {
        clade_name: count_clade_matches(proteins_sorted, match_subs, dom)
        for clade_name, dom, match_subs, _, _ in REQUIRED_STAR_CLADES
    }

    output_doc = {
        "metadata": {
            "dataset_name": "ProPurify Studio Multi-Domain Swiss-Prot 10K+ Natural-Language Benchmark Corpus",
            "source_api": "https://rest.uniprot.org/uniprotkb/search (reviewed:true)",
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "http_requests_made": collector.http_requests,
            "total_unique_proteins": total_proteins,
            "unique_species_count": len(unique_species),
            "unique_tax_id_count": len(unique_tax_ids),
            "fragment_count": sum(1 for p in proteins_sorted if p["is_fragment"]),
            "with_primary_gene_count": sum(1 for p in proteins_sorted if p["gene_primary"]),
            "domain_counts": dom_summary,
            "eukaryotic_kingdom_counts": euk_kingdom_summary,
            "collision_class_counts": collision_summary,
            "star_clade_counts": clade_audit,
        },
        "proteins": proteins_sorted,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(output_doc, fh, indent=2, ensure_ascii=False)
    os.replace(tmp_path, output_path)

    print("\n=== Corpus Assembly Complete ===")
    print(json.dumps(output_doc["metadata"], indent=2))
    return output_doc


def run_offline_self_test() -> None:
    """Validates record parsing, lineage classification, and collision logic offline."""
    sample_hit = {
        "entryType": "UniProtKB reviewed (Swiss-Prot)",
        "primaryAccession": "P0A6F5",
        "uniProtkbId": "CH60_ECOLI",
        "annotationScore": 5.0,
        "proteinDescription": {
            "recommendedName": {
                "fullName": {"value": "Chaperonin GroEL"},
                "shortNames": [{"value": "GroEL"}],
            },
            "alternativeNames": [
                {"fullName": {"value": "60 kDa chaperonin"}},
                {"fullName": {"value": "Protein Cpn60"}},
            ],
        },
        "genes": [
            {
                "geneName": {"value": "groL"},
                "synonyms": [{"value": "groEL"}, {"value": "mopA"}],
            }
        ],
        "organism": {
            "scientificName": "Escherichia coli (strain K12)",
            "commonName": "E. coli",
            "taxonId": 83333,
            "lineage": ["Bacteria", "Pseudomonadota", "Gammaproteobacteria"],
        },
        "sequence": {"length": 548},
    }
    rec = parse_uniprot_record(sample_hit)
    assert rec is not None
    assert rec["accession"] == "P0A6F5"
    assert rec["domain"] == "Bacteria"
    assert rec["subkingdom"] == "Pseudomonadota"
    assert rec["gene_primary"] == "groL"
    assert rec["gene_synonyms"] == ["groEL", "mopA"]
    assert rec["short_names"] == ["GroEL"]
    assert rec["is_fragment"] is False

    # Verify protist classification
    dom, sub = classify_domain_and_subkingdom(["Eukaryota", "Sar", "Alveolata", "Apicomplexa"])
    assert (dom, sub) == ("Eukaryota", "Protists")

    # Verify uncharacterized rejection
    unchar = dict(sample_hit, primaryAccession="P99999")
    unchar["proteinDescription"] = {"recommendedName": {"fullName": {"value": "Uncharacterized protein"}}}
    assert parse_uniprot_record(unchar) is None

    print("Offline self-test passed.")


def recompute_corpus_metadata(corpus_path: Path) -> Dict[str, Any]:
    """Recomputes and validates summary metadata for an existing corpus JSON file offline."""
    with corpus_path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)

    proteins_sorted = doc["proteins"]
    total_proteins = len(proteins_sorted)
    unique_accs = {p["accession"] for p in proteins_sorted}
    assert len(unique_accs) == total_proteins, "Duplicate accessions detected!"
    assert total_proteins >= 11000, f"Expected >=11,000 proteins, got {total_proteins}"

    dom_summary = {
        d: sum(1 for p in proteins_sorted if p["domain"] == d)
        for d in ("Bacteria", "Archaea", "Eukaryota", "Viruses")
    }
    euk_kingdom_summary = {
        k: sum(1 for p in proteins_sorted if p["domain"] == "Eukaryota" and p["subkingdom"] == k)
        for k in ("Metazoa", "Viridiplantae", "Fungi", "Protists")
    }
    unique_species = {normalize_species_binomial(p["organism_scientific"]) for p in proteins_sorted}
    unique_tax_ids = {p["tax_id"] for p in proteins_sorted}
    collision_summary = compute_collision_counts(proteins_sorted)
    clade_audit = {
        clade_name: count_clade_matches(proteins_sorted, match_subs, dom)
        for clade_name, dom, match_subs, _, _ in REQUIRED_STAR_CLADES
    }

    old_meta = doc.get("metadata") or {}
    doc["metadata"] = {
        "dataset_name": old_meta.get(
            "dataset_name",
            "ProPurify Studio Multi-Domain Swiss-Prot 10K+ Natural-Language Benchmark Corpus",
        ),
        "source_api": old_meta.get(
            "source_api", "https://rest.uniprot.org/uniprotkb/search (reviewed:true)"
        ),
        "generated_at_utc": old_meta.get(
            "generated_at_utc",
            dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
        "http_requests_made": old_meta.get("http_requests_made", 40),
        "total_unique_proteins": total_proteins,
        "unique_species_count": len(unique_species),
        "unique_tax_id_count": len(unique_tax_ids),
        "fragment_count": sum(1 for p in proteins_sorted if p["is_fragment"]),
        "with_primary_gene_count": sum(1 for p in proteins_sorted if p["gene_primary"]),
        "domain_counts": dom_summary,
        "eukaryotic_kingdom_counts": euk_kingdom_summary,
        "collision_class_counts": collision_summary,
        "star_clade_counts": clade_audit,
    }

    tmp_path = corpus_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    os.replace(tmp_path, corpus_path)
    return doc["metadata"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch >=11,000 stratified Swiss-Prot proteins.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "nl_10k_proteins.json",
        help="Destination JSON file path.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run offline parser & classification unit checks without network calls.",
    )
    parser.add_argument(
        "--recompute-metadata",
        action="store_true",
        help="Recompute and validate summary metadata on an existing corpus JSON offline.",
    )
    args = parser.parse_args()

    if args.self_test:
        run_offline_self_test()
        return

    if args.recompute_metadata:
        meta = recompute_corpus_metadata(args.output)
        print(json.dumps(meta, indent=2))
        return

    download_full_corpus(args.output)


if __name__ == "__main__":
    main()
