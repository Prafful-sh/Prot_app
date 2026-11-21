"""
Sequence Parser & Acquisition Module
Handles:
- Instant Classroom & Teaching-Lab Preset Proteins (Lysozyme default, GFP, Carbonic Anhydrase II, Myoglobin, Lipase A, BSA)
- Direct FASTA/Raw protein sequence
- UniProt ID lookup via REST API
- NCBI ID lookup via Entrez REST API
- DNA/RNA sequence translation (ORF finder & direct 6-frame, NCBI Standard Genetic Code transl_table=1)
- Natural Language construct parser (Organism + Gene + Tag + Cleavage site, with offline classroom fallback)
"""

import base64
import json
import re
import threading
import zlib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, Any, Optional, Tuple, List, Union, Set


def _decompress_offline_seq(b64_payload: str) -> str:
    """Decompresses a compact zlib+base64 canonical Swiss-Prot reference sequence."""
    return zlib.decompress(base64.b64decode(b64_payload.encode("ascii"))).decode("ascii")

_ORIG_REQUESTS_GET = requests.get
_THREAD_LOCAL_HTTP = threading.local()


def _get_http_session() -> requests.Session:
    """Returns a thread-local persistent requests.Session with retry/backoff on 429/5xx."""
    sess = getattr(_THREAD_LOCAL_HTTP, "session", None)
    if sess is None:
        sess = requests.Session()
        retry_strategy = Retry(
            total=3,
            connect=0,
            read=2,
            status=3,
            backoff_factor=0.4,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=retry_strategy)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        sess.headers.update({
            "Accept": "application/json",
            "User-Agent": "ProPurifyStudio/2.0 (Scientific Protein Construct Parser)",
        })
        _THREAD_LOCAL_HTTP.session = sess
    return sess


def _http_get(url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 12.0) -> requests.Response:
    """Executes HTTP GET via shared session with retry, while honoring unittest.mock patches on requests.get."""
    if requests.get is not _ORIG_REQUESTS_GET:
        return requests.get(url, headers=headers, timeout=timeout)
    return _get_http_session().get(url, headers=headers, timeout=timeout)

# Canonical mature Hen Egg-White Lysozyme sequence (UniProt P00698, residues 19-147, 129 aa, 14,313.14 Da / 14.31 kDa)
LYSOZYME_MATURE_SEQ = (
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTP"
    "GSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
)

# Classroom & Teaching-Lab Protein Presets (Offline-ready verified mature chains)
CLASSROOM_PRESET_PROTEINS: Dict[str, Dict[str, Any]] = {
    "lysozyme_native": {
        "label": "🥚 Egg-White Lysozyme (Chicken • 14.3 kDa • Basic Enzyme)",
        "name": "Hen Egg-White Lysozyme C (Gallus gallus • Mature Chain)",
        "uniprot_id": "P00698",
        "organism": "Gallus gallus (Chicken)",
        "sequence": LYSOZYME_MATURE_SEQ,
        "tag": "None",
        "construct_parts": None,
        "student_story": (
            "Found abundantly in chicken egg whites (with a closely related homolog in human tears), Lysozyme C "
            "hydrolyzes bacterial peptidoglycan cell walls and is stabilized by 4 disulfide bridges. Because it is "
            "strongly positively charged (pI 9.32) and compact (14.31 kDa), it is the #1 classic enzyme used in school "
            "and college labs to teach Cation-Exchange (CEX), Hydrophobic Interaction (HIC), and Gel Filtration (SEC)!"
        )
    },
    "lysozyme_his_tev": {
        "label": "🏷️ 6xHis-TEV-Lysozyme (Recombinant • 16.1 kDa • Tagged)",
        "name": "6xHis-TEV-Lysozyme (Recombinant Construct)",
        "uniprot_id": "P00698",
        "organism": "Gallus gallus (Recombinant E. coli construct)",
        "sequence": "MHHHHHHENLYFQG" + LYSOZYME_MATURE_SEQ,
        "tag": "6xHis",
        "construct_parts": [
            ("Tag (6xHis)", "MHHHHHH"),
            ("Cleavage Site (TEV)", "ENLYFQG"),
            ("Target Protein (Lysozyme)", LYSOZYME_MATURE_SEQ)
        ],
        "student_story": (
            "Engineered Lysozyme (16.12 kDa, pI 9.16) carrying an N-terminal 6xHis affinity tag and a TEV protease "
            "cut site (ENLYFQ|G). Capture it in one step on a Ni-NTA column, clip off the His-tag with TEV protease "
            "(leaving 14.37 kDa Gly-Lysozyme), and polish on SEC!"
        )
    },
    "gfp": {
        "label": "🟢 Green Fluorescent Protein (GFP • 26.9 kDa • Acidic)",
        "name": "Green Fluorescent Protein (Aequorea victoria)",
        "uniprot_id": "P42212",
        "organism": "Aequorea victoria (Crystal Jellyfish)",
        "sequence": (
            "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQC"
            "FSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHK"
            "LEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKD"
            "PNEKRDHMVLLEFVTAAGITHGMDELYK"
        ),
        "tag": "None",
        "construct_parts": None,
        "student_story": (
            "2008 Nobel Prize-winning glowing jellyfish beta-barrel protein (26.89 kDa, acidic pI 5.67)! In college "
            "teaching labs, GFP is famously purified using Ammonium Sulfate salting-out followed directly by "
            "Hydrophobic Interaction Chromatography (HIC)—students can literally watch the bright green band bind and "
            "elute under a UV lamp!"
        )
    },
    "carbonic_anhydrase": {
        "label": "🫁 Carbonic Anhydrase II (29.1 kDa • Near-Neutral pI 6.86 -> Needs HIC!)",
        "name": "Carbonic Anhydrase 2 (Homo sapiens • Mature Chain, P00918)",
        "uniprot_id": "P00918",
        "organism": "Homo sapiens (Human)",
        "sequence": (
            "SHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFD"
            "DSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAV"
            "LGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKE"
            "PISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK"
        ),
        "tag": "None",
        "construct_parts": None,
        "student_story": (
            "One of nature's fastest zinc metalloenzymes—hydrating ~1 million CO2 molecules per second in human red "
            "blood cells! In mature native CA II (residues 2–260, 259 aa, initiator Met1 cleaved; 29.11 kDa), its "
            "near-neutral pI (6.86) sits close to physiological pH 7.4, giving a weak net charge (-1.4 at pH 7.4)—a "
            "textbook case where Ion-Exchange (IEX) binds weakly and Hydrophobic Interaction Chromatography (HIC) "
            "rescues the purification!"
        )
    },
    "myoglobin": {
        "label": "💪 Myoglobin (Whale Muscle • 17.2 kDa • Oxygen Carrier)",
        "name": "Myoglobin (Physeter macrocephalus • Mature Chain)",
        "uniprot_id": "P02185",
        "organism": "Physeter macrocephalus (Sperm Whale)",
        "sequence": (
            "VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLT"
            "ALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRK"
            "DIAAKYKELGYQG"
        ),
        "tag": "None",
        "construct_parts": None,
        "student_story": (
            "The very first protein ever solved in 3D by X-ray crystallography (John Kendrew, 1958; Nobel Prize in "
            "Chemistry 1962)! Its compact 153-aa, 8-alpha-helix apoprotein chain (17.20 kDa, pI 8.71; ~17.8 kDa as "
            "holomyoglobin with its non-covalent iron-heme b cofactor) stores oxygen in deep-diving sperm whale muscle "
            "and serves as a classic globular standard on Cation-Exchange (pH 6.0–6.8), Phenyl HIC, and Superdex 75 SEC."
        )
    },
    "lipase_a": {
        "label": "🧴 Bacillus Lipase A (19.4 kDa • Hydrophobic Surface -> HIC)",
        "name": "Lipase A (Bacillus subtilis • Mature Extracellular Enzyme)",
        "uniprot_id": "P37957",
        "organism": "Bacillus subtilis",
        "sequence": (
            "AEHNPVVMVHGIGGASFNFAGIKSYLVSQGWSRDKLYAVDFWDKTGTNYNNGPVLSRFVQKVLDETGAKK"
            "VDIVAHSMGGANTLYYIKNLDGGNKVANVVTLGGANRLTTGKALPGTDPNQKILYTSIYSSADMIVMNYL"
            "SRLDGARNVQIHGVGHIGLLYSSQVNSLIKEGLNGGGQNTN"
        ),
        "tag": "None",
        "construct_parts": None,
        "student_story": (
            "A compact 181-aa extracellular esterase/lipase (19.36 kDa minimal α/β-hydrolase fold lacking a lid domain) "
            "with exposed hydrophobic ('oil-loving') patches around its active site. Because of these surface patches "
            "(GRAVY -0.183), it binds strongly to Mild Butyl / Phenyl Hydrophobic Interaction (HIC) columns!"
        )
    },
    "bsa": {
        "label": "🧪 Serum Albumin (BSA • 66.4 kDa • Lab Standard)",
        "name": "Bovine Serum Albumin (Bos taurus • Mature Chain)",
        "uniprot_id": "P02769",
        "organism": "Bos taurus (Bovine)",
        "sequence": (
            "DTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLF"
            "GDELCKVASLRETYGDMADCCEKQEPERNECFLSHKDDSPDLPKLKPDPNTLCDEFKADEKKFWGKYLYE"
            "IARRHPYFYAPELLYYANKYNGVFQECCQAEDKGACLLPKIETMREKVLASSARQRLRCASIQKFGERAL"
            "KAWSVARLSQKFPKAEFVEVTKLVTDLTKVHKECCHGDLLECADDRADLAKYICDNQDTISSKLKECCDK"
            "PLLEKSHCIAEVEKDAIPENLPPLTADFAEDKDVCKNYQEAKDAFLGSFLYEYSRRHPEYAVSVLLRLAK"
            "EYEATLEECCAKDDPHACYSTVFDKLKHLVDEPQNLIKQNCDQFEKLGEYGFQNALIVRYTRKVPQVSTP"
            "TLVEVSRSLGKVGTRCCTKPESERMPCTEDYLSLILNRLCVLHEKTPVSEKVTKCCTESLVNRRPCFSAL"
            "TPDETYVPKAFDEKLFTFHADICTLPDTEKQIKKQTALVELLKHKPKATEEQLKTVMENFVAFVDKCCAA"
            "DDKEACFAVEGPKLVVSTQTALA"
        ),
        "tag": "None",
        "construct_parts": None,
        "student_story": (
            "The universal 583-aa laboratory standard protein (66.43 kDa, acidic pI 5.60, stabilized by 17 disulfide "
            "bridges) used in Bradford/BCA protein assays, SDS-PAGE markers, and Superdex 200 SEC column calibration."
        )
    }
}

# Standard affinity & solubility tag dictionary with verified vector/epitope sequences
STANDARD_TAGS = {
    "6xhis": {
        "name": "6xHis",
        "n_term": "MGSSHHHHHHSSG",      # Standard pET-derived His-tag leader
        "n_term_short": "MHHHHHH",
        "c_term": "LEHHHHHH",           # Standard pET XhoI-His6 C-terminal fusion
        "c_term_short": "HHHHHH"
    },
    "8xhis": {
        "name": "8xHis",
        "n_term": "MGSSHHHHHHHHSSG",
        "n_term_short": "MHHHHHHHH",
        "c_term": "LEHHHHHHHH",
        "c_term_short": "HHHHHHHH"
    },
    "10xhis": {
        "name": "10xHis",
        "n_term": "MGSSHHHHHHHHHHSSG",
        "n_term_short": "MHHHHHHHHHH",
        "c_term": "LEHHHHHHHHHH",
        "c_term_short": "HHHHHHHHHH"
    },
    "twin-strep": {
        "name": "Twin-Strep-tag",
        "n_term": "MWSHPQFEKGGGSGGGSGGSAWSHPQFEKGS",
        "n_term_short": "MWSHPQFEKGGGSGGGSGGSAWSHPQFEK",
        "c_term": "GSWSHPQFEKGGGSGGGSGGSAWSHPQFEK",
        "c_term_short": "WSHPQFEKGGGSGGGSGGSAWSHPQFEK"
    },
    "strep": {
        "name": "Strep-tag II",
        "n_term": "MWSHPQFEKGS",
        "n_term_short": "MWSHPQFEK",
        "c_term": "GSWSHPQFEK",
        "c_term_short": "WSHPQFEK"
    },
    "flag": {
        "name": "FLAG-tag",
        "n_term": "MDYKDDDDKGS",
        "n_term_short": "MDYKDDDDK",
        "c_term": "GSDYKDDDDK",
        "c_term_short": "DYKDDDDK"
    },
    "ha": {
        "name": "HA-tag",
        "n_term": "MYPYDVPDYAGS",
        "n_term_short": "MYPYDVPDYA",
        "c_term": "GSYPYDVPDYA",
        "c_term_short": "YPYDVPDYA"
    },
    "myc": {
        "name": "c-Myc-tag",
        "n_term": "MEQKLISEEDLGS",
        "n_term_short": "MEQKLISEEDL",
        "c_term": "GSEQKLISEEDL",
        "c_term_short": "EQKLISEEDL"
    },
    "gst": {
        "name": "GST (Glutathione S-transferase)",
        # Schistosoma japonicum GST (UniProt P08515, 218 aa) + pGEX-4T/6P SD linker (220 aa)
        "n_term": (
            "MSPILGYWKIKGLVQPTRLLLEYLEEKYEEHLYERDEGDKWRNKKFELGLEFPNLPYYIDGDVKLTQSMAIIRYIADKHNMLGGCP"
            "KERAEISMLEGAVLDIRYGVSRIAYSKDFETLKVDFLSKLPEMLKMFEDRLCHKTYLNGDHVTHPDFMLYDALDVVLYMDPMCLDAFP"
            "KLVCFKKRIEAIPQIDKYLKSSKYIAWPLQGWQATFGGGDHPPKSD"
        ),
        "c_term": (
            "GSSPILGYWKIKGLVQPTRLLLEYLEEKYEEHLYERDEGDKWRNKKFELGLEFPNLPYYIDGDVKLTQSMAIIRYIADKHNMLGGCP"
            "KERAEISMLEGAVLDIRYGVSRIAYSKDFETLKVDFLSKLPEMLKMFEDRLCHKTYLNGDHVTHPDFMLYDALDVVLYMDPMCLDAFP"
            "KLVCFKKRIEAIPQIDKYLKSSKYIAWPLQGWQATFGGGDHPPKSD"
        )
    },
    "mbp": {
        "name": "MBP (Maltose-Binding Protein)",
        # E. coli K-12 mature MalE (UniProt P0AEX9) engineered pMAL-c2X/c5X fusion sequence (371 aa)
        "n_term": (
            "MKIEEGKLVIWINGDKGYNGLAEVGKKFEKDTGIKVTVEHPDKLEEKFPQVAATGDGPDIIFWAHDRFGGYAQSGLLAEITPDKAFQDKL"
            "YPFTWDAVRYNGKLIAYPIAVEALSLIYNKDLLPNPPKTWEEIPALDKELKAKGKSALMFNLQEPYFTWPLIAADGGYAFKYENGKYDIK"
            "DVGVDNAGAKAGLTFLVDLIKNKHMNADTDYSIAEAAFNKGETAMTINGPWAWSNIDTSKVNYGVTVLPTFKGQPSKPFVGVLSAGINAA"
            "SPNKELAKEFLENYLLTDEGLEAVNKDKPLGAVALKSYEEELAKDPRIAATMENAQKGEIMPNIPQMSAFWYAVRTAVINAASGRQTVDE"
            "ALKDAQTNSSS"
        ),
        "c_term": (
            "GSKIEEGKLVIWINGDKGYNGLAEVGKKFEKDTGIKVTVEHPDKLEEKFPQVAATGDGPDIIFWAHDRFGGYAQSGLLAEITPDKAFQDKL"
            "YPFTWDAVRYNGKLIAYPIAVEALSLIYNKDLLPNPPKTWEEIPALDKELKAKGKSALMFNLQEPYFTWPLIAADGGYAFKYENGKYDIK"
            "DVGVDNAGAKAGLTFLVDLIKNKHMNADTDYSIAEAAFNKGETAMTINGPWAWSNIDTSKVNYGVTVLPTFKGQPSKPFVGVLSAGINAA"
            "SPNKELAKEFLENYLLTDEGLEAVNKDKPLGAVALKSYEEELAKDPRIAATMENAQKGEIMPNIPQMSAFWYAVRTAVINAASGRQTVDE"
            "ALKDAQTNSSS"
        )
    }
}

STANDARD_CLEAVAGE_SITES = {
    "tev": {
        "name": "TEV Protease",
        "seq": "ENLYFQG",
        "cut_offset": 6,  # between Q6 and G7
        "description": "Cleaves after Gln: Glu-Asn-Leu-Tyr-Phe-Gln | Gly"
    },
    "3c": {
        "name": "PreScission / HRV 3C Protease",
        "seq": "LEVLFQGP",
        "cut_offset": 6,  # between Q6 and G7
        "description": "Cleaves after Gln: Leu-Glu-Val-Leu-Phe-Gln | Gly-Pro"
    },
    "thrombin": {
        "name": "Thrombin",
        "seq": "LVPRGS",
        "cut_offset": 4,  # between R4 and G5
        "description": "Cleaves after Arg: Leu-Val-Pro-Arg | Gly-Ser"
    },
    "factor_xa": {
        "name": "Factor Xa",
        "seq": "IEGR",
        "cut_offset": 4,  # after R4
        "description": "Cleaves after Arg: Ile-Glu-Gly-Arg |"
    },
    "enterokinase": {
        "name": "Enterokinase",
        "seq": "DDDDK",
        "cut_offset": 5,  # after K5
        "description": "Cleaves after Lys: Asp-Asp-Asp-Asp-Lys |"
    }
}

# NCBI Standard Genetic Code (transl_table=1; all 64 codons verified)
CODON_TABLE = {
    'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
    'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
    'TAC': 'Y', 'TAT': 'Y', 'TAA': '*', 'TAG': '*',
    'TGC': 'C', 'TGT': 'C', 'TGA': '*', 'TGG': 'W',
}


def _parse_ncbi_fasta_header(header: Optional[str], fallback_id: str) -> Tuple[str, str]:
    """
    Parses an NCBI FASTA definition line into (clean_protein_name, organism).
    Handles protein headers like 'NP_990612.2 lysozyme C precursor [Gallus gallus]',
    pipe-delimited headers ('ref|NP_990612.2| ...'), and mRNA/nucleotide headers
    like 'NM_205281.2 Gallus gallus lysozyme (LYZ), mRNA'.
    """
    hdr = (header or "").strip()
    if not hdr:
        return fallback_id, "Unknown Organism"

    body = re.sub(
        r"^(?:ref|gb|emb|dbj|tpg|tpe|tpd|pir|prf|pdb)\|[^|\s]+\|(?:\S+\|)?\s*",
        "",
        hdr,
    )
    if body == hdr:
        if re.match(
            r"^(?:[A-Z]{1,4}_\d+(?:\.\d+)?|[A-Za-z]{1,4}\d{5,10}(?:\.\d+)?|[0-9][A-Za-z0-9]{3}_[A-Za-z0-9]+)(?:\s+|$)",
            hdr,
        ):
            body = re.sub(r"^\S+\s*", "", hdr, count=1)

    organism = "Unknown Organism"
    # Extract the outermost balanced [...] bracket group at/near the end of the definition line
    # so organisms containing nested genotype brackets like [Rotavirus A (strain ... G3P[X])] stay intact.
    top_brackets: List[Tuple[int, int, str]] = []
    depth = 0
    b_start = -1
    for idx_ch, ch in enumerate(body):
        if ch == "[":
            if depth == 0:
                b_start = idx_ch
            depth += 1
        elif ch == "]" and depth > 0:
            depth -= 1
            if depth == 0 and b_start >= 0:
                top_brackets.append((b_start, idx_ch + 1, body[b_start + 1:idx_ch].strip()))
                b_start = -1

    chosen_bracket: Optional[Tuple[int, int, str]] = None
    for b_s, b_e, b_txt in reversed(top_brackets):
        if b_txt and re.search(r"[A-Za-z]", b_txt):
            chosen_bracket = (b_s, b_e, b_txt)
            break

    if chosen_bracket is not None:
        b_s, b_e, b_txt = chosen_bracket
        organism = b_txt
        body = (body[:b_s] + body[b_e:]).strip(" ,;")
    else:
        org_details = _extract_organism_details(body)
        if org_details.get("organism_name"):
            organism = org_details["organism_name"]
        else:
            organism = "See Header"

    clean_name = re.sub(r"\s{2,}", " ", body).strip() or fallback_id
    return clean_name, organism


def extract_fasta_header_metadata(header: Optional[str]) -> Dict[str, Any]:
    """
    Extracts structured accession, gene name, organism, protein name, and curated offline
    biochemical annotations from UniProt ('>sp|P07374|UREA_CANEN ... OS=... GN=...') or
    NCBI ('>NP_990612.2 ... [Gallus gallus]') FASTA headers.
    """
    hdr = (header or "").strip()
    if not hdr:
        return {}

    # 1. UniProtKB FASTA format: sp|P07374|UREA_CANEN Urease OS=Canavalia ensiformis OX=3823 GN=UREA PE=1 SV=2
    m_up = re.match(r"^(?:sp|tr)\|([A-Za-z0-9\-]+)\|(\S+)(?:\s+(.*))?$", hdr)
    if m_up:
        uid_raw = m_up.group(1).strip().upper()
        base_uid = uid_raw.split("-")[0]
        entry_name = m_up.group(2).strip().upper()
        rest = (m_up.group(3) or "").strip()

        pname_match = re.split(r"\s+(?:OS|OX|GN|PE|SV)=", rest, maxsplit=1)
        clean_pname = (pname_match[0].strip() if pname_match else "") or entry_name or uid_raw
        m_os = re.search(r"(?:^|\s)OS=(.+?)(?=\s+(?:OX|GN|PE|SV)=|$)", rest)
        m_gn = re.search(r"(?:^|\s)GN=(\S+)", rest)
        org = m_os.group(1).strip() if m_os else "Unknown Organism"
        gene = m_gn.group(1).strip() if m_gn else "N/A"

        biochem_ann: Dict[str, Any] = {}
        if base_uid in OFFLINE_UNIPROT_REFERENCES:
            biochem_ann = dict(OFFLINE_UNIPROT_REFERENCES[base_uid].get("biochem_annotations", {}))
        else:
            for preset in CLASSROOM_PRESET_PROTEINS.values():
                if preset.get("uniprot_id") == base_uid and preset.get("tag") == "None":
                    biochem_ann = dict(preset.get("biochem_annotations", {}))
                    break

        return {
            "source_format": "UniProt FASTA",
            "uniprot_id": uid_raw,
            "accession": uid_raw,
            "entry_name": entry_name,
            "protein_name": clean_pname,
            "organism": org,
            "gene_name": gene,
            "biochem_annotations": biochem_ann,
        }

    # 2. NCBI FASTA format: NP_990612.2 lysozyme C precursor [Gallus gallus] / ref|...| / [Organism]
    m_pipe = re.match(r"^(?:ref|gb|emb|dbj|tpg|tpe|tpd|pir|prf)\|([^|\s]+)\|", hdr)
    m_acc = re.match(
        r"^([A-Z]{1,4}_\d+(?:\.\d+)?|[A-Za-z]{1,4}\d{5,10}(?:\.\d+)?|[0-9][A-Za-z0-9]{3}_[A-Za-z0-9]+)(?:\s+|$)",
        hdr,
    )
    has_org_bracket = bool(re.search(r"\[[^\[\]]+\]\s*$", hdr))
    if m_pipe or m_acc or has_org_bracket:
        acc_token = (m_pipe.group(1) if m_pipe else (m_acc.group(1) if m_acc else "")).strip()
        clean_pname, org = _parse_ncbi_fasta_header(hdr, acc_token or hdr)
        out: Dict[str, Any] = {
            "source_format": "NCBI FASTA",
            "protein_name": clean_pname,
            "organism": org,
        }
        if acc_token:
            out["accession"] = acc_token
        return out

    return {
        "source_format": "Custom FASTA",
        "protein_name": hdr,
    }


def clean_amino_acid_sequence(raw_text: str) -> Tuple[str, Optional[str], List[str]]:
    """
    Cleans raw text or FASTA format into a clean uppercase single-letter amino acid sequence.
    When multiple FASTA entries are pasted together, stops at the second '>' header and warns
    so adjacent proteins are never concatenated into a chimeric sequence.
    Returns: (cleaned_sequence, fasta_header, list_of_warnings)
    """
    lines = (raw_text or "").strip().splitlines()
    header = None
    seq_lines: List[str] = []
    additional_entries = 0
    stop_reading = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is None and not seq_lines:
                header = line[1:].strip()
            else:
                additional_entries += 1
                stop_reading = True
            continue
        if not stop_reading:
            seq_lines.append(line)

    full_str = "".join(seq_lines).upper()
    # Remove whitespace, tabs, numbers, gap/stop symbols
    full_str = re.sub(r'[\s\d\.\-\*]', '', full_str)

    # Check for invalid characters
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    non_standard = set(full_str) - valid_aa
    warnings: List[str] = []
    if additional_entries > 0:
        entry_label = header or "Entry 1"
        if len(entry_label) > 60:
            entry_label = entry_label[:57] + "..."
        warnings.append(
            f"Multiple FASTA entries detected ({additional_entries + 1} total); loaded only the first entry "
            f"('{entry_label}'). Paste one sequence at a time."
        )
    if non_standard:
        warnings.append(
            f"Sequence contains non-standard amino acid characters: {', '.join(sorted(non_standard))}. "
            "Calculations may exclude or approximate them."
        )

    return full_str, header, warnings


def detect_sequence_terminal_tag(sequence: str) -> Dict[str, Any]:
    """
    Inspects the N-terminal and C-terminal ends (within 35 residues) of an amino acid sequence
    to auto-detect engineered affinity tags (6xHis/8xHis/10xHis, Twin-Strep, Strep-tag II, FLAG)
    even on pasted FASTA or translated ORF sequences, and returns the tag-stripped core sequence
    so engineered tag residues never bias catalytic motif detection.
    """
    seq = (sequence or "").strip().upper()
    if len(seq) < 20:
        return {"tag_detected": "None", "tag_position": "N/A", "core_sequence": seq}

    head = seq[:38]
    tail = seq[-38:]

    # 1. Twin-Strep-tag
    for ts in ("WSHPQFEKGGGSGGGSGGSAWSHPQFEK", "WSHPQFEKGGGSGGGSGGSWSHPQFEK"):
        if ts in head:
            idx_end = head.find(ts) + len(ts)
            if seq[idx_end:idx_end + 2] == "GS":
                idx_end += 2
            return {"tag_detected": "Twin-Strep-tag", "tag_position": "N-terminal", "core_sequence": seq[idx_end:]}
        if ts in tail:
            idx_start = len(seq) - len(tail) + tail.find(ts)
            if idx_start >= 2 and seq[idx_start - 2:idx_start] == "GS":
                idx_start -= 2
            return {"tag_detected": "Twin-Strep-tag", "tag_position": "C-terminal", "core_sequence": seq[:idx_start]}

    # 2. Strep-tag II
    if "WSHPQFEK" in head[:24]:
        idx_end = head.find("WSHPQFEK") + 8
        if seq[idx_end:idx_end + 2] == "GS":
            idx_end += 2
        return {"tag_detected": "Strep-tag II", "tag_position": "N-terminal", "core_sequence": seq[idx_end:]}
    if "WSHPQFEK" in tail[-16:]:
        idx_start = len(seq) - 16 + tail[-16:].find("WSHPQFEK")
        if idx_start >= 2 and seq[idx_start - 2:idx_start] == "GS":
            idx_start -= 2
        return {"tag_detected": "Strep-tag II", "tag_position": "C-terminal", "core_sequence": seq[:idx_start]}

    # 3. FLAG-tag
    if "DYKDDDDK" in head[:24]:
        idx_end = head.find("DYKDDDDK") + 8
        return {"tag_detected": "FLAG-tag", "tag_position": "N-terminal", "core_sequence": seq[idx_end:]}
    if "DYKDDDDK" in tail[-16:]:
        idx_start = len(seq) - 16 + tail[-16:].find("DYKDDDDK")
        return {"tag_detected": "FLAG-tag", "tag_position": "C-terminal", "core_sequence": seq[:idx_start]}

    # 4. Poly-Histidine tags (10xHis, 8xHis, 6xHis) within 18 aa of N- or C-terminus
    m_n = re.match(r"^(M?[A-Z]{0,10}?(H{6,12})(?:SSGLVPRGSH|SSGENLYFQG|ENLYFQG|LEVLFQGP|SSG|GSG|GS)?)", head)
    if m_n:
        hlen = len(m_n.group(2))
        tname = "10xHis" if hlen >= 10 else ("8xHis" if hlen >= 8 else "6xHis")
        return {"tag_detected": tname, "tag_position": "N-terminal", "core_sequence": seq[len(m_n.group(1)):]}

    m_c = re.search(r"((?:LE|GS|GGS|AAALE)?(H{6,12}))$", tail[-22:])
    if m_c:
        hlen = len(m_c.group(2))
        tname = "10xHis" if hlen >= 10 else ("8xHis" if hlen >= 8 else "6xHis")
        cut_len = len(m_c.group(1))
        return {"tag_detected": tname, "tag_position": "C-terminal", "core_sequence": seq[:-cut_len]}

    return {"tag_detected": "None", "tag_position": "N/A", "core_sequence": seq}


_METAL_OR_COFACTOR_LIGAND_RX = re.compile(
    r"(?:\b(?:ni|zn|fe|cu|mn|co|mo|ca|mg|w|v)\s*\(\s*[123]?\+\s*\)|\b(?:ni|zn|fe|cu|mn|co|ca|mg)\s+cation\b|"
    r"\b(?:nickel|zinc|iron|copper|manganese|cobalt|molybdenum|tungsten|vanadium|calcium|magnesium)\b|"
    r"\[[234]fe-[234]s\]|iron-sulfur|heme|protoheme|siroheme|hemin|cobalamin|molybdopterin|"
    r"pyridoxal|flavin|\bfad\b|\bfmn\b|\bf430\b)",
    flags=re.IGNORECASE,
)

_NON_COFACTOR_LIGAND_WORDS = {
    "substrate", "inhibitor", "transition state", "transition-state", "product",
    "atp", "gtp", "adp", "gdp", "amp", "gmp", "nad", "nadh", "nadp", "nadph",
    "glutathione", "o2", "dioxygen", "nitrite", "water", "h2o", "coa", "acetyl-coa",
    "dna", "rna", "peptide", "lipid", "cholesterol", "glucose", "maltose", "urea",
}

_KNOWN_COFACTOR_CHEBI_IDS = {
    "CHEBI:49786",  # Ni(2+)
    "CHEBI:25516",  # Ni cation (UniProt COFACTOR notation)
    "CHEBI:29105",  # Zn(2+)
    "CHEBI:29033",  # Fe(2+)
    "CHEBI:29034",  # Fe(3+)
    "CHEBI:29036",  # Cu(2+)
    "CHEBI:49552",  # Cu(+)
    "CHEBI:23378",  # Cu cation
    "CHEBI:29035",  # Mn(2+)
    "CHEBI:48828",  # Co(2+)
    "CHEBI:30411",  # cobalamin / B12
    "CHEBI:18420",  # Mg(2+)
    "CHEBI:29108",  # Ca(2+)
    "CHEBI:49883",  # [4Fe-4S] cluster
    "CHEBI:49601",  # [2Fe-2S] cluster
    "CHEBI:21137",  # [3Fe-4S] cluster
    "CHEBI:30408",  # iron-sulfur cluster
    "CHEBI:60344",  # heme b
    "CHEBI:26355",  # heme
    "CHEBI:61717",  # heme c
    "CHEBI:60342",  # heme a
    "CHEBI:17627",  # heme / protoheme
    "CHEBI:597326", # pyridoxal 5'-phosphate (PLP)
    "CHEBI:57692",  # FAD
    "CHEBI:58210",  # FMN
    "CHEBI:57618",  # FMNH2
    "CHEBI:60240",  # divalent metal cation
    "CHEBI:60539",  # molybdopterin cofactor
    "CHEBI:71302",  # Moco
}


def _norm_chebi_id(raw_id: str) -> str:
    """Normalizes UniProt ChEBI IDs (collapses 'ChEBI:CHEBI:49786' -> 'CHEBI:49786')."""
    up = (raw_id or "").strip().upper()
    if not up:
        return ""
    return re.sub(r"^(?:CHEBI:)+", "CHEBI:", up)


def _extract_uniprot_biochem_annotations(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts structured biochemical, cofactor (with ChEBI IDs), metal-ligand, redox, and membrane
    topology annotations from a UniProtKB REST JSON entry.
    Strictly filters out generic substrates/inhibitors from metal_ligands and keeps free-text
    cofactor_notes separate from structured cofactors so prose negations ('not nickel(II)') never
    cause false-positive metal calls.
    """
    cofactors: List[str] = []
    cofactor_chebi_ids: List[str] = []
    cofactor_notes: List[str] = []
    subcellular_locations: List[str] = []
    for comm in (data.get("comments") or []):
        ctype = comm.get("commentType", "")
        if ctype == "COFACTOR":
            for cf in (comm.get("cofactors") or []):
                cf_name = (cf.get("name") or "").strip()
                if cf_name and cf_name not in cofactors:
                    cofactors.append(cf_name)
                cf_chebi = _norm_chebi_id((cf.get("cofactorCrossReference") or {}).get("id") or "")
                if cf_chebi and cf_chebi not in cofactor_chebi_ids:
                    cofactor_chebi_ids.append(cf_chebi)
            for nt in ((comm.get("note") or {}).get("texts") or []):
                val = (nt.get("value") or "").strip()
                if val:
                    cofactor_notes.append(val)
        elif ctype == "SUBCELLULAR LOCATION":
            for sl in (comm.get("subcellularLocations") or []):
                loc_val = ((sl.get("location") or {}).get("value") or "").strip()
                if loc_val and loc_val not in subcellular_locations:
                    subcellular_locations.append(loc_val)
                top_val = ((sl.get("topology") or {}).get("value") or "").strip()
                if top_val and top_val not in subcellular_locations:
                    subcellular_locations.append(top_val)

    metal_ligands: List[str] = []
    metal_ligand_chebi_ids: List[str] = []
    disulfide_count = 0
    transmembrane_count = 0
    has_signal_peptide = False
    zinc_finger_count = 0
    for feat in (data.get("features") or []):
        ftype = feat.get("type", "")
        if ftype == "Disulfide bond":
            disulfide_count += 1
        elif ftype in ("Transmembrane", "Intramembrane"):
            transmembrane_count += 1
        elif ftype in ("Signal", "Signal peptide"):
            has_signal_peptide = True
        elif ftype == "Zinc finger":
            zinc_finger_count += 1
            if "Zn(2+) [Zinc finger]" not in metal_ligands:
                metal_ligands.append("Zn(2+) [Zinc finger]")
            if "CHEBI:29105" not in metal_ligand_chebi_ids:
                metal_ligand_chebi_ids.append("CHEBI:29105")
        elif ftype in ("Binding site", "Metal binding"):
            lig_obj = feat.get("ligand") or {}
            lig_name = (lig_obj.get("name") or feat.get("description") or "").strip()
            lig_id = _norm_chebi_id(lig_obj.get("id") or "")
            feat_desc_low = str(feat.get("description") or "").lower()
            lig_low = lig_name.lower()
            # Skip inhibitors, non-physiological crystallization artifacts, and generic small-molecule substrates
            if any(w in feat_desc_low for w in ("inhibitor", "non-physiological", "artifact")):
                continue
            if lig_low in _NON_COFACTOR_LIGAND_WORDS:
                continue
            is_true_metal_or_cof = (
                (lig_id in _KNOWN_COFACTOR_CHEBI_IDS)
                or bool(_METAL_OR_COFACTOR_LIGAND_RX.search(lig_name))
            )
            if is_true_metal_or_cof:
                if lig_name and lig_name not in metal_ligands:
                    metal_ligands.append(lig_name)
                if lig_id and lig_id not in metal_ligand_chebi_ids:
                    metal_ligand_chebi_ids.append(lig_id)

    keywords: List[str] = []
    for kw in (data.get("keywords") or []):
        kw_name = (kw.get("name") or "").strip()
        if kw_name and kw_name not in keywords:
            keywords.append(kw_name)

    ec_numbers: List[str] = []
    pdesc = data.get("proteinDescription") or {}
    for ec_obj in ((pdesc.get("recommendedName") or {}).get("ecNumbers") or []):
        ec_v = (ec_obj.get("value") or "").strip()
        if ec_v and ec_v not in ec_numbers:
            ec_numbers.append(ec_v)

    return {
        "cofactors": cofactors,
        "cofactor_chebi_ids": cofactor_chebi_ids,
        "cofactor_notes": cofactor_notes,
        "metal_ligands": metal_ligands,
        "metal_ligand_chebi_ids": metal_ligand_chebi_ids,
        "disulfide_count": disulfide_count,
        "transmembrane_count": transmembrane_count,
        "has_signal_peptide": has_signal_peptide,
        "zinc_finger_count": zinc_finger_count,
        "keywords": keywords,
        "subcellular_locations": subcellular_locations,
        "ec_numbers": ec_numbers,
    }


def _extract_uniprot_mature_chain(data: Dict[str, Any], sequence: str) -> Optional[Dict[str, Any]]:
    """
    Inspects UniProtKB REST JSON features ('Signal'/'Signal peptide', 'Transit peptide',
    'Propeptide', 'Chain') to identify an unambiguous cleaved mature polypeptide chain
    when it is strictly shorter than the full precursor (e.g., P00698 147 aa -> 129 aa;
    P02769 BSA 607 aa -> 583 aa). Returns None for cytosolic non-cleaved proteins or
    multi-chain polyproteins (where len(chains) > 1).
    """
    seq = (sequence or "").strip().upper()
    L = len(seq)
    if L < 30:
        return None

    chains: List[Tuple[int, int, str]] = []
    cleaved_regions: List[str] = []
    signal_end: Optional[int] = None

    for feat in (data.get("features") or []):
        ftype = feat.get("type", "")
        loc = feat.get("location") or {}
        s_val = (loc.get("start") or {}).get("value")
        e_val = (loc.get("end") or {}).get("value")
        if not isinstance(s_val, int) or not isinstance(e_val, int):
            continue
        if s_val < 1 or e_val > L or s_val > e_val:
            continue
        if ftype == "Chain":
            chains.append((s_val, e_val, (feat.get("description") or "").strip()))
        elif ftype in ("Signal", "Signal peptide"):
            cleaved_regions.append(f"Signal peptide {s_val}–{e_val}")
            if s_val == 1:
                signal_end = e_val
        elif ftype == "Transit peptide":
            cleaved_regions.append(f"Transit peptide {s_val}–{e_val}")
            if s_val == 1 and signal_end is None:
                signal_end = e_val
        elif ftype == "Propeptide":
            cleaved_regions.append(f"Propeptide {s_val}–{e_val}")

    if len(chains) == 1:
        c_start, c_end, c_desc = chains[0]
        if (c_start > 1 or c_end < L) and (c_end - c_start + 1) >= 20:
            mat_seq = seq[c_start - 1 : c_end]
            summary = ", ".join(cleaved_regions) if cleaved_regions else f"N/C-terminal processing (mature {c_start}–{c_end})"
            return {
                "start": c_start,
                "end": c_end,
                "sequence": mat_seq,
                "length": len(mat_seq),
                "description": c_desc or "Mature Chain",
                "cleaved_summary": summary,
            }
    elif len(chains) == 0 and signal_end is not None and (L - signal_end) >= 20:
        mat_seq = seq[signal_end:]
        return {
            "start": signal_end + 1,
            "end": L,
            "sequence": mat_seq,
            "length": len(mat_seq),
            "description": "Mature Chain (Signal Peptide Removed)",
            "cleaved_summary": ", ".join(cleaved_regions) or f"Signal peptide 1–{signal_end}",
        }

    return None


def _parse_uniprot_entry_dict(data: Dict[str, Any], fallback_uid: str = "") -> Dict[str, Any]:
    """Extracts standardized metadata, sequence, mature-chain features, and biochemical annotations from a UniProtKB REST JSON entry."""
    accession = data.get("primaryAccession", fallback_uid)
    protein_desc = data.get("proteinDescription") or {}
    rec_name_obj = protein_desc.get("recommendedName") or {}
    sub_names = protein_desc.get("submissionNames") or []
    alt_names = protein_desc.get("alternativeNames") or []
    sub_first = sub_names[0] if (sub_names and isinstance(sub_names[0], dict)) else {}
    alt_first = alt_names[0] if (alt_names and isinstance(alt_names[0], dict)) else {}
    recommended_name = (
        (rec_name_obj.get("fullName") or {}).get("value")
        or (sub_first.get("fullName") or {}).get("value")
        or (alt_first.get("fullName") or {}).get("value")
        or data.get("uniProtkbId")
        or fallback_uid
        or "Unknown Protein"
    )
    organism_name = (data.get("organism") or {}).get("scientificName", "Unknown Organism")
    gene_list = data.get("genes") or []
    gene_name = "N/A"
    for g_obj in gene_list:
        cand_g = (
            (g_obj.get("geneName") or {}).get("value")
            or ((g_obj.get("synonyms") or [{}])[0].get("value") if g_obj.get("synonyms") else None)
            or ((g_obj.get("orderedLocusNames") or [{}])[0].get("value") if g_obj.get("orderedLocusNames") else None)
            or ((g_obj.get("orfNames") or [{}])[0].get("value") if g_obj.get("orfNames") else None)
        )
        if cand_g is not None and str(cand_g).strip():
            gene_name = str(cand_g).strip()
            break
    sequence_info = data.get("sequence", {})
    sequence = sequence_info.get("value", "")
    length = sequence_info.get("length", len(sequence))
    mature_chain = _extract_uniprot_mature_chain(data, sequence)
    out = {
        "source": "UniProt",
        "accession": accession,
        "protein_name": recommended_name,
        "gene_name": gene_name,
        "organism": organism_name,
        "length": length,
        "sequence": sequence,
        "biochem_annotations": _extract_uniprot_biochem_annotations(data),
    }
    if mature_chain is not None:
        out["mature_chain"] = mature_chain
    return out


# Authoritative Swiss-Prot reference sequences for canonical proteins & dual-meaning tag/protease names (offline fallback)
OFFLINE_UNIPROT_REFERENCES: Dict[str, Dict[str, Any]] = {
    "P09211": {
        "accession": "P09211",
        "protein_name": "Glutathione S-transferase P",
        "gene_name": "GSTP1",
        "organism": "Homo sapiens",
        "sequence": (
            "MPPYTVVYFPVRGRCAALRMLLADQGQSWKEEVVTVETWQEGSLKASCLYGQLPKFQDGDLTLYQSNTILRHLGRTLGLYGKDQQEAALV"
            "DMVNDGVEDLRCKYISLIYTNYEAGKDDYVKALPGQLKPFETLLSQNQGGKTFIVGDQISFADYNLLDLLLIHEVLAPGCLDAFPLLSAY"
            "VGRLSARPKLKAFLASPEYVNLPINGNGKQ"
        )
    },
    "P08263": {
        "accession": "P08263",
        "protein_name": "Glutathione S-transferase A1",
        "gene_name": "GSTA1",
        "organism": "Homo sapiens",
        "sequence": (
            "MAEKPKLHYFNARGRMESTRWLLAAAGVEFEEKFIKSAEDLDKLRNDGYLMFQQVPMVEIDGMKLVQTRAILNYIASKYNLYGKDIKERA"
            "LIDMYIEGIADLGEMILLLPVCPPEEKDAKLALIKEKIKNRYFPAFEKVLKSHGQDYLVGNKLSRADIHLVELLYYVEELDSSLISSFPL"
            "LKALKTRISNLPTVKKFLQPGSPRKPPMDEKSLEEARKIFRF"
        )
    },
    "P09488": {
        "accession": "P09488",
        "protein_name": "Glutathione S-transferase Mu 1",
        "gene_name": "GSTM1",
        "organism": "Homo sapiens",
        "sequence": (
            "MPMILGYWDIRGLAHAIRLLLEYTDSSYEEKKYTMGDAPDYDRSQWLNEKFKLGLDFPNLPYLIDGAHKITQSNAILCYIARKHNLCGET"
            "EEEKIRVDILENQTMDNHMQLGMICYNPEFEKLKPKYLEELPEKLKLYSEFLGKRPWFAGNKITFVDFLVYDVLDLHRIFEPKCLDAFPN"
            "LKDFISRFEGLEKISAYMKSSRFLPRPVFSKMAVWGNK"
        )
    },
    "P08515": {
        "accession": "P08515",
        "protein_name": "Glutathione S-transferase class-mu 26 kDa isozyme",
        "gene_name": "GST26",
        "organism": "Schistosoma japonicum",
        "sequence": (
            "MSPILGYWKIKGLVQPTRLLLEYLEEKYEEHLYERDEGDKWRNKKFELGLEFPNLPYYIDGDVKLTQSMAIIRYIADKHNMLGGCPKERA"
            "EISMLEGAVLDIRYGVSRIAYSKDFETLKVDFLSKLPEMLKMFEDRLCHKTYLNGDHVTHPDFMLYDALDVVLYMDPMCLDAFPKLVCFK"
            "KRIEAIPQIDKYLKSSKYIAWPLQGWQATFGGGDHPPK"
        )
    },
    "P04637": {
        "accession": "P04637",
        "protein_name": "Cellular tumor antigen p53",
        "gene_name": "TP53",
        "organism": "Homo sapiens",
        "sequence": (
            "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPS"
            "WPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHE"
            "RCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSF"
            "EVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPG"
            "GSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
        )
    },
    "P0AEX9": {
        "accession": "P0AEX9",
        "protein_name": "Maltose/maltodextrin-binding periplasmic protein",
        "gene_name": "malE",
        "organism": "Escherichia coli (strain K12)",
        "sequence": (
            "MKIKTGARILALSALTTMMFSASALAKIEEGKLVIWINGDKGYNGLAEVGKKFEKDTGIKVTVEHPDKLEEKFPQVAATGDGPDIIFWAH"
            "DRFGGYAQSGLLAEITPDKAFQDKLYPFTWDAVRYNGKLIAYPIAVEALSLIYNKDLLPNPPKTWEEIPALDKELKAKGKSALMFNLQEP"
            "YFTWPLIAADGGYAFKYENGKYDIKDVGVDNAGAKAGLTFLVDLIKNKHMNADTDYSIAEAAFNKGETAMTINGPWAWSNIDTSKVNYGV"
            "TVLPTFKGQPSKPFVGVLSAGINAASPNKELAKEFLENYLLTDEGLEAVNKDKPLGAVALKSYEEELAKDPRIAATMENAQKGEIMPNIP"
            "QMSAFWYAVRTAVINAASGRQTVDEALKDAQTRITK"
        )
    },
    "P02686": {
        "accession": "P02686",
        "protein_name": "Myelin basic protein",
        "gene_name": "MBP",
        "organism": "Homo sapiens",
        "sequence": (
            "MGNHAGKRELNAEKASTNSETNRGESEKKRNLGELSRTTSEDNEVFGEADANQNNGTSSQDTAVTDSKRTADPKNAWQDAHPADPGSRPH"
            "LIRLFSRDAPGREDNTFKDRPSESDELQTIQEDSAATSESLDVMASQKRPSQRHGSKYLATASTMDHARHGFLPRHRDTGILDSIGRFFG"
            "GDRGAPKRGSGKDSHHPARTAHYGSLPQKSHGRTQDENPVVHFFKNIVTPRTPPPSQGKGRGLSLSRFSWGAEGQRPGFGYGGRASDYKS"
            "AHKGFKGVDAQGTLSKIFKLGGRDSRSGSPMARR"
        )
    },
    "P01106": {
        "accession": "P01106",
        "protein_name": "Myc proto-oncogene protein",
        "gene_name": "MYC",
        "organism": "Homo sapiens",
        "sequence": (
            "MDFFRVVENQQPPATMPLNVSFTNRNYDLDYDSVQPYFYCDEEENFYQQQQQSELQPPAPSEDIWKKFELLPTPPLSPSRRSGLCSPSYV"
            "AVTPFSLRGDNDGGGGSFSTADQLEMVTELLGGDMVNQSFICDPDDETFIKNIIIQDCMWSGFSAAAKLVSEKLASYQAARKDSGSPNPA"
            "RGHSVCSTSSLYLQDLSAAASECIDPSVVFPYPLNDSSSPKSCASQDSSAFSPSSDSLLSSTESSPQGSPEPLVLHEETPPTTSSDSEEE"
            "QEDEEEIDVVSVEKRQAPGKRSESGSPSAGGHSKPPHSPLVLKRCHVSTHQHNYAAPPSTRKDYPAAKRVKLDSVRVLRQISNNRKCTSP"
            "RSSDTEENVKRRTHNVLERQRRNELKRSFFALRDQIPELENNEKAPKVVILKKATAYILSVQAEEQKLISEEDLLRKRREQLKHKLEQLR"
            "NSCA"
        )
    },
    "P00734": {
        "accession": "P00734",
        "protein_name": "Prothrombin",
        "gene_name": "F2",
        "organism": "Homo sapiens",
        "sequence": (
            "MAHVRGLQLPGCLALAALCSLVHSQHVFLAPQQARSLLQRVRRANTFLEEVRKGNLERECVEETCSYEEAFEALESSTATDVFWAKYTAC"
            "ETARTPRDKLAACLEGNCAEGLGTNYRGHVNITRSGIECQLWRSRYPHKPEINSTTHPGADLQENFCRNPDSSTTGPWCYTTDPTVRRQE"
            "CSIPVCGQDQVTVAMTPRSEGSSVNLSPPLEQCVPDRGQQYQGRLAVTTHGLPCLAWASAQAKALSKHQDFNSAVQLVENFCRNPDGDEE"
            "GVWCYVAGKPGDFGYCDLNYCEEAVEEETGDGLDEDSDRAIEGRTATSEYQTFFNPRTFGSGEADCGLRPLFEKKSLEDKTERELLESYI"
            "DGRIVEGSDAEIGMSPWQVMLFRKSPQELLCGASLISDRWVLTAAHCLLYPPWDKNFTENDLLVRIGKHSRTRYERNIEKISMLEKIYIH"
            "PRYNWRENLDRDIALMKLKKPVAFSDYIHPVCLPDRETAASLLQAGYKGRVTGWGNLKETWTANVGKGQPSVLQVVNLPIVERPVCKDST"
            "RIRITDNMFCAGYKPDEGKRGDACEGDSGGPFVMKSPFNNRWYQMGIVSWGEGCDRDGKYGFYTHVFRLKKWIQKVIDQFGE"
        )
    },
    "P04517": {
        "accession": "P04517",
        "protein_name": "TEV Protease (Nuclear inclusion protein A)",
        "gene_name": "NIa-Pro",
        "organism": "Tobacco etch virus",
        # UniProt P04517 catalytic NIa protease mature chain (residues 2038-2279, 242 aa)
        "sequence": (
            "GESLFKGPRDYNPISSTICHLTNESDGHTTSLYGIGFGPFIITNKHLFRRNNGTLLVQSLHGVFKVKNTTTLQQHLIDGRDMIIIRMPKD"
            "FPPFPQKLKFREPQREERICLVTTNFQTKSMSSMVSDTSCTFPSSDGIFWKHWIQTKDGQCGSPLVSTRDGFIVGIHSASNFTNTNNYFT"
            "SVPKNFMELLTNQEAQQWVSGWRLNADSVLWGGHKVFMSKPEEPFQPVKEATQLMNELVYSQ"
        )
    },
    "P61626": {
        "accession": "P61626",
        "protein_name": "Lysozyme C",
        "gene_name": "LYZ",
        "organism": "Homo sapiens",
        "sequence": (
            "MKALIVLGLVLLSVTVQGKVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRATNYNAGDRSTDYGIFQINSRYWCNDGKTPG"
            "AVNACHLSCSALLQDNIADAVACAKRVVRDPQGIRAWVAWRNRCQNRDVRQYVQGCGV"
        )
    },
    "P0A707": {
        "accession": "P0A707",
        "protein_name": "Translation initiation factor IF-3",
        "gene_name": "infC",
        "organism": "Escherichia coli (strain K12)",
        "sequence": (
            "MKGGKRVQTARPNRINGEIRAQEVRLTGLEGEQLGIVSLREALEKAEEAGVDLVEISPNAEPPVCRIMDYGKFLYEKSKSSKEQKKKQKV"
            "IQVKEIKFRPGTDEGDYQVKLRSLIRFLEEGDKAKITLRFRGREMAHQQIGMEVLNRVKDDLQELAVVESFPTKIEGRQMIMVLAPKKKQ"
        )
    },
    "Q8WZ42": {
        "accession": "Q8WZ42",
        "protein_name": "Titin",
        "gene_name": "TTN",
        "organism": "Homo sapiens",
        "sequence": _decompress_offline_seq(
            "eNqFnQlu5TqybccmqAclQRIJHmj+E/lrBXVs530F/FtZ2dg+ashoduxouJdydWeZynVuV661buOcS1emsVvWPE9nPceaP9M9"
            "zFddc8llO+d6rXnKw3x3aSvr2dWSuiPP95O31JVjzvPV8ZPduG01dWPpzvOY6nVzh73U+5rzVflHvWuZ17Octabp4Q7duF45"
            "b8PE9a9xHjYuuK3d2D3nOOS5PLkeXD9X7vy9/jWP41jPLqWSylr5Kh8e832Vex1TWnmPaejuvHaljntduUs3X9d2Lqmc572e"
            "ZzpTvjOPcfJDXequrbuvK5/5XO/FP+q91JtFir/mms+uu10Kvp/5wrlt+51YxlxqV2bW6zzPT+IFntrlnMdu3O+xlK3wzNd6"
            "l3G8PyOrxTqOVy0sM4/UdR2X7vjf0/HGXqlLY03X0A0pd36hVv4Y9u6u93iyGV0dL173LvzX1WW9zs4L3ok7puKuVD8yJC50"
            "jRtvWbjiWgrPdu1LLcvI46Sx8NNTPflprtm1n2b18jiuJaUr8YzX2K03X+UrPOfudUuXcknbWOo5c19esFzXwHW3/Ixp3bku"
            "q1K97spP8wfXHbaa7/keZ5/i5mGvlWun/fvMWy5rF9duT7Fud2Ht9q5w92vlSZY51aHOKbVFamtytTXh17xs48gSXhcLPj7z"
            "k8Y73qp243le99nl8Vx8qq6m864r0p/HsqyplOFK847U52tN/pMNG++tLnVI93mXLp/LVGpOa+bjZVzmZ+wQmTl3CNd2pXHL"
            "iGXursS6ZST/LNyDq/Nw3L2c23lN59QN5RzKkxCMubLD7Mlc81rmgm4gG9NYt4U14pkLunMj3Gcd17MioYX1m7d01FLXcR5z"
            "5SV7FfVBOEstn+ceh+dax5xX1ahMV57X7t72lQtOnWp0T6VHdg5uz9uU3D9bV6/Kdk8jm6bgjGni3cd0TxVVGmrdy1Dyxpeu"
            "bj7nEQ04n4m3RHWvtNVxnrEd09xfdZ4PNGqpzyelPNeTRZyf+0k1P0e6yjz2/PyKHWAdhm4eH5S23kda5rGE+G9sYDc86Pae"
            "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"
        )
    },
    "P11532": {
        "accession": "P11532",
        "protein_name": "Dystrophin",
        "gene_name": "DMD",
        "organism": "Homo sapiens",
        "sequence": _decompress_offline_seq(
            "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"
        )
    },
    "P07374": {
        "accession": "P07374",
        "protein_name": "Urease",
        "gene_name": "UREA",
        "organism": "Canavalia ensiformis",
        "biochem_annotations": {
            "cofactors": ["Ni(2+)"],
            "cofactor_chebi_ids": ["CHEBI:49786"],
            "cofactor_notes": ["Binds 2 Ni(2+) ions per subunit (carbamylated Lys-bridged dinuclear nickel active site)."],
            "metal_ligands": ["Ni(2+)"],
            "metal_ligand_chebi_ids": ["CHEBI:49786"],
            "disulfide_count": 0,
            "transmembrane_count": 0,
            "has_signal_peptide": False,
            "zinc_finger_count": 0,
            "keywords": ["Hydrolase", "Metal-binding", "Nickel"],
            "subcellular_locations": ["Cytoplasm"],
            "ec_numbers": ["3.5.1.5"],
        },
        "sequence": _decompress_offline_seq(
            "eNoVkluOBSEIRNdGGkQCGhTCxP1vZLh/bYdHnaKWWvilIjW2uYGfwdFrcLmu7ZcEBSYQRxY9uMikWXBsfcZnmvG9p8yhvB8b"
            "6hRBDkdOtcqa6BKXNpMdAhscZl4ehjqAUgn3FWcS+wjps7QtfBVKTCtDGa/LKY755hCq7c9y4O2Sx6sVCgAn1B3kjJ9GpVUI"
            "8daSy7xBANlrU24jgDW7tscOnkSkCEE8fjJ8fz52DplXG3TrY89kVLnG2M0GJN3wwB630hrM/NuBvPjEx9MdpLmyJNsHEXzJ"
            "IgooLIr8M1FYgbejrM01Yi8Rhp1UAkxdUQkMgt+s+eTzY/WohwZLNlYvTPYf74XMLz08T3uw7EQimtseDcPKEaGOZJN+AhgW"
            "m/YD/ziyBwjudlqA5kQ5smUmtvEEPIpmtKVD+Wb/frP9gV48oVWL1tcsu+8dkdvzumXsFKRpuGzVN/vjkjihwYAZV65qrwLC"
            "st1WNI40EcZpVbe42ePm30lAXQqnCU0/7IqNe3T7fQL6UnbbC5tHnNdxKCpWA7Raf+oxRufNaZUo8yr4+7neN23LPZ28dF1f"
            "j9N+F2CwhhwVCmB4bjXRY9t6O7wVu65axxeXtmQwJ8lCp3hZfc7ea19FxzzLLe5+w8Y/mVH35w=="
        ),
    },
}

# Add Jack Bean Urease (first enzyme ever crystallized — Sumner, 1926; classic Ni2+ metalloenzyme) to Classroom Presets
CLASSROOM_PRESET_PROTEINS["urease_jack_bean"] = {
    "label": "🪙 Jack Bean Urease (90.8 kDa • Ni²⁺ Metalloenzyme — Avoid Ni-NTA!)",
    "name": "Urease (Canavalia ensiformis • Dinuclear Ni²⁺ Metalloenzyme)",
    "uniprot_id": "P07374",
    "organism": "Canavalia ensiformis (Jack bean)",
    "sequence": OFFLINE_UNIPROT_REFERENCES["P07374"]["sequence"],
    "tag": "None",
    "construct_parts": None,
    "biochem_annotations": dict(OFFLINE_UNIPROT_REFERENCES["P07374"]["biochem_annotations"]),
    "student_story": (
        "The very first enzyme ever crystallized (James Sumner, 1926 — proving enzymes are proteins; 1946 Nobel Prize)! "
        "Each 90.8 kDa subunit houses a carbamylated-Lys-bridged dinuclear Ni²⁺ active site (native hexamer ~545 kDa). "
        "A vital lab lesson: never expose Urease to Ni-NTA, Imidazole, or EDTA—purify natively via EDTA-free Anion-Exchange (HiTrap Q), HIC, and SEC!"
    ),
}

_LOCAL_CORPUS_CACHE: Optional[List[Dict[str, Any]]] = None
_LOCAL_CORPUS_BY_ACC: Optional[Dict[str, Dict[str, Any]]] = None


def _ensure_local_corpus_loaded() -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Lazily loads the 14,869-protein local PDB/Swiss-Prot corpus (data/ss_10k_corpus.npz) for offline fallback."""
    global _LOCAL_CORPUS_CACHE, _LOCAL_CORPUS_BY_ACC
    if _LOCAL_CORPUS_CACHE is not None and _LOCAL_CORPUS_BY_ACC is not None:
        return _LOCAL_CORPUS_CACHE, _LOCAL_CORPUS_BY_ACC

    import os as _os
    import numpy as _np

    npz_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data", "ss_10k_corpus.npz")
    records: List[Dict[str, Any]] = []
    by_acc: Dict[str, Dict[str, Any]] = {}
    if _os.path.exists(npz_path):
        try:
            with _np.load(npz_path, allow_pickle=False) as d:
                accs = d["accessions"]
                pnames = d["protein_names"]
                gnames = d["gene_names"]
                orgs = d["organisms"]
                tids = d["tax_ids"]
                seqs = d["sequences"]
                for idx in range(len(accs)):
                    acc_s = str(accs[idx]).strip()
                    seq_s = str(seqs[idx]).strip()
                    if not acc_s or not seq_s:
                        continue
                    rec = {
                        "accession": acc_s,
                        "primaryAccession": acc_s,
                        "reviewed": True,
                        "annotationScore": 5.0,
                        "is_fragment": False,
                        "recommended_name": str(pnames[idx]).strip(),
                        "protein_name": str(pnames[idx]).strip(),
                        "gene_primary": str(gnames[idx]).strip(),
                        "gene_name": str(gnames[idx]).strip() or "N/A",
                        "organism_scientific": str(orgs[idx]).strip(),
                        "organism": str(orgs[idx]).strip(),
                        "tax_id": int(tids[idx]) if int(tids[idx]) > 0 else None,
                        "length": len(seq_s),
                        "sequence": seq_s,
                    }
                    records.append(rec)
                    if acc_s not in by_acc:
                        by_acc[acc_s] = rec
        except Exception:
            pass

    _LOCAL_CORPUS_CACHE = records
    _LOCAL_CORPUS_BY_ACC = by_acc
    return records, by_acc


def _search_local_corpus_offline(comp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Searches the 14,869-protein local corpus when network access is unavailable."""
    records, by_acc = _ensure_local_corpus_loaded()
    if not records:
        return None
    pref_accs = comp.get("preferred_accessions") or []
    for pa in pref_accs:
        if pa in by_acc:
            r = by_acc[pa]
            return {
                "source": "Built-in Local Reference Corpus (UniProt)",
                "accession": r["accession"],
                "protein_name": r["protein_name"],
                "gene_name": r["gene_name"],
                "organism": r["organism"],
                "length": r["length"],
                "sequence": r["sequence"],
            }

    pq_toks = _tokenize_for_matching(comp.get("protein_name_hint") or comp.get("protein_query") or "")
    gh = (comp.get("gene_hint") or "").strip().lower()
    if not pq_toks and not gh:
        return None

    shortlist: List[Dict[str, Any]] = []
    for r in records:
        pn_low = r["protein_name"].lower()
        gn_low = r["gene_primary"].lower()
        if (gh and gn_low == gh) or any(tok in pn_low or tok == gn_low for tok in pq_toks if len(tok) >= 3):
            shortlist.append(r)

    if not shortlist:
        return None
    ranked = rank_uniprot_candidates(shortlist, comp)
    best = ranked[0]
    if score_uniprot_candidate(best, comp) < 120.0:
        return None
    return {
        "source": "Built-in Local Reference Corpus (UniProt)",
        "accession": best["accession"],
        "protein_name": best["protein_name"],
        "gene_name": best["gene_name"],
        "organism": best["organism"],
        "length": best["length"],
        "sequence": best["sequence"],
    }


_UNIPROT_OFFLINE_ENTRY_NAMES: Dict[str, str] = {
    "LYSC_CHICK": "P00698",
    "LYSC_HUMAN": "P61626",
    "GFP_AEQVI": "P42212",
    "MYG_EQUCA": "P68082",
    "ALBU_BOVIN": "P02769",
    "GSTP1_HUMAN": "P09211",
    "CAH2_HUMAN": "P00918",
    "UREA_CANEN": "P07374",
    "DYST_HUMAN": "P11532",
    "TITIN_HUMAN": "Q8WZ42",
    "THIO_ECOLI": "P0AA25",
    "SUMO1_HUMAN": "P63165",
    "MBP_ECOLI": "P0AEX9",
    "MALE_ECOLI": "P0AEX9",
}


def _normalize_uniprot_input(raw_id: str) -> Dict[str, Any]:
    """
    Normalizes user-supplied UniProt inputs across canonical accessions (P00698, A0A024RBG1),
    isoforms (P04637-2), entry names (LYSC_CHICK), FASTA pipe tokens ('sp|P00698|LYSC_CHICK'),
    label prefixes ('UniProt: P00698.1'), and full UniProtKB/Rest URLs.
    """
    s = (raw_id or "").strip()
    if not s:
        raise ValueError(
            "Please enter a valid UniProt ID (e.g., P00698, P04637-2, or LYSC_CHICK)."
        )

    # Take first line / token if user pasted a FASTA header line
    first_line = s.splitlines()[0].strip()
    if first_line.startswith(">"):
        first_line = first_line[1:].strip()

    # Extract accession/entry name from UniProt URLs
    m_url = re.search(
        r"uniprot\.org/(?:uniprotkb|uniprot)/([A-Za-z0-9_.\-]+)",
        first_line,
        flags=re.IGNORECASE,
    )
    if m_url:
        token = m_url.group(1)
        token = re.sub(r"\.(?:json|fasta|xml|txt|rdf|gff)$", "", token, flags=re.IGNORECASE)
    else:
        token = re.sub(r"^(?:uniprotkb|uniprot|id|acc)\s*:\s*", "", first_line, flags=re.IGNORECASE)
        m_pipe = re.match(r"^(?:sp|tr)\|([A-Za-z0-9\-]+)\|", token, flags=re.IGNORECASE)
        if m_pipe:
            token = m_pipe.group(1)
        else:
            token = token.split()[0]

    # Strip file/version suffixes like P00698.1 (without harming isoforms like P04637-2)
    token = re.sub(r"\.\d+$", "", token.strip().strip("|,;"))
    uid = token.upper()

    raw_seq_cand = re.sub(r"[\s*]+", "", (raw_id or "").strip())
    if (
        len(raw_seq_cand) >= 20
        and re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", raw_seq_cand, flags=re.IGNORECASE)
    ):
        raise ValueError(
            "This looks like a raw amino acid sequence rather than a UniProt ID. "
            "Paste it in '🔤 Paste Amino Acid / FASTA' or '💬 Describe in Plain English' instead."
        )

    if not uid or not re.match(r"^[A-Z0-9][A-Z0-9_\-]{2,}$", uid):
        raise ValueError(
            f"Unrecognized UniProt ID format '{raw_id.strip()}'. Expected e.g. P00698, P04637-2, or LYSC_CHICK."
        )

    m_iso = re.match(
        r"^((?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}))-(\d+)$",
        uid,
    )
    if m_iso:
        return {
            "query": uid,
            "base_accession": m_iso.group(1),
            "isoform_num": m_iso.group(2),
            "is_entry_name": False,
        }

    is_entry_name = bool(re.match(r"^[A-Z0-9]{1,10}_[A-Z0-9]{1,5}$", uid))
    return {
        "query": uid,
        "base_accession": uid,
        "isoform_num": None,
        "is_entry_name": is_entry_name,
    }


def _fetch_uniprot_json_following_merge(
    uid: str,
    headers: Dict[str, str],
    _depth: int = 0,
) -> Optional[Dict[str, Any]]:
    """Fetches a UniProtKB JSON entry and automatically resolves merged/inactive accessions."""
    url = f"https://rest.uniprot.org/uniprotkb/{uid}.json"
    response = _http_get(url, headers=headers, timeout=12)
    if response.status_code != 200:
        return None
    data = response.json()
    has_seq = bool((data.get("sequence") or {}).get("value"))
    if data.get("entryType") == "Inactive" or not has_seq:
        ir = data.get("inactiveReason") or {}
        merge_targets = ir.get("mergeDemergeTo") or []
        if merge_targets and _depth < 2:
            target_acc = str(merge_targets[0]).strip().upper()
            resolved = _fetch_uniprot_json_following_merge(target_acc, headers, _depth=_depth + 1)
            if resolved is not None:
                resolved["note"] = f"Merged UniProt accession '{uid}' resolved to '{resolved['accession']}'."
                resolved["uniprot_id"] = resolved["accession"]
                return resolved
        raise ValueError(
            f"UniProt entry '{uid}' is inactive/obsolete and has no active sequence."
        )
    parsed = _parse_uniprot_entry_dict(data, fallback_uid=uid)
    parsed["uniprot_id"] = parsed["accession"]
    return parsed


def fetch_uniprot_by_id(uniprot_id: str) -> Dict[str, Any]:
    """
    Fetches full metadata and sequence for a UniProt ID, isoform (P04637-2), entry name (LYSC_CHICK),
    FASTA token ('sp|P00698|LYSC_CHICK'), or UniProt URL. Follows merged inactive accessions
    and falls back gracefully to offline Swiss-Prot & 14,869-protein corpus references.
    """
    norm = _normalize_uniprot_input(uniprot_id)
    uid = norm["query"]
    base_acc = norm["base_accession"]
    isoform_num = norm["isoform_num"]
    is_entry_name = norm["is_entry_name"]
    headers = {"Accept": "application/json"}

    try:
        # 1. Explicit isoform fetch (REST JSON endpoint rejects '-N' isoforms; use .fasta + parent .json)
        if isoform_num:
            iso_url = f"https://rest.uniprot.org/uniprotkb/{uid}.fasta"
            iso_resp = _http_get(iso_url, headers={"Accept": "text/plain"}, timeout=12)
            if iso_resp.status_code == 200 and iso_resp.text.strip().startswith(">"):
                iso_seq, iso_hdr, _ = clean_amino_acid_sequence(iso_resp.text)
                if iso_seq:
                    parent_meta: Optional[Dict[str, Any]] = None
                    try:
                        parent_meta = _fetch_uniprot_json_following_merge(base_acc, headers)
                    except (requests.RequestException, ValueError):
                        parent_meta = None
                    if parent_meta:
                        return {
                            "source": "UniProt",
                            "accession": uid,
                            "uniprot_id": uid,
                            "protein_name": f"{parent_meta['protein_name']} (Isoform {isoform_num})",
                            "gene_name": parent_meta["gene_name"],
                            "organism": parent_meta["organism"],
                            "length": len(iso_seq),
                            "sequence": iso_seq,
                            "biochem_annotations": parent_meta.get("biochem_annotations", {}),
                        }
                    hdr_meta = extract_fasta_header_metadata(iso_hdr)
                    return {
                        "source": "UniProt",
                        "accession": uid,
                        "uniprot_id": uid,
                        "protein_name": hdr_meta.get("protein_name") or f"{base_acc} (Isoform {isoform_num})",
                        "gene_name": hdr_meta.get("gene_name", "N/A"),
                        "organism": hdr_meta.get("organism", "Unknown Organism"),
                        "length": len(iso_seq),
                        "sequence": iso_seq,
                        "biochem_annotations": hdr_meta.get("biochem_annotations", {}),
                    }
        else:
            # 2. Direct JSON lookup (also follows inactive merged entries automatically)
            direct_hit = _fetch_uniprot_json_following_merge(uid, headers)
            if direct_hit is not None:
                return direct_hit

            # 3. Entry-name search fallback (e.g. LYSC_CHICK / UREA_CANEN)
            if is_entry_name:
                q_url = f"https://rest.uniprot.org/uniprotkb/search?query=id:{requests.utils.quote(uid)}&size=1&format=json"
                q_resp = _http_get(q_url, headers=headers, timeout=12)
                if q_resp.status_code == 200:
                    q_results = q_resp.json().get("results") or []
                    if q_results:
                        parsed_hit = _parse_uniprot_entry_dict(q_results[0], fallback_uid=uid)
                        if parsed_hit.get("sequence"):
                            parsed_hit["uniprot_id"] = parsed_hit["accession"]
                            return parsed_hit
    except requests.RequestException:
        pass

    # Offline fallback for built-in UniProt references & 14,869-protein corpus
    lookup_candidates = [uid]
    if base_acc not in lookup_candidates:
        lookup_candidates.append(base_acc)
    if is_entry_name and uid in _UNIPROT_OFFLINE_ENTRY_NAMES:
        mapped_acc = _UNIPROT_OFFLINE_ENTRY_NAMES[uid]
        if mapped_acc not in lookup_candidates:
            lookup_candidates.append(mapped_acc)

    for cand_acc in lookup_candidates:
        curated_preset_ann: Dict[str, Any] = {}
        curated_preset_obj: Optional[Dict[str, Any]] = None
        for preset in CLASSROOM_PRESET_PROTEINS.values():
            if preset["uniprot_id"] == cand_acc and preset["tag"] == "None":
                curated_preset_ann = dict(preset.get("biochem_annotations", {}))
                curated_preset_obj = preset
                break

        if cand_acc in OFFLINE_UNIPROT_REFERENCES:
            ref = OFFLINE_UNIPROT_REFERENCES[cand_acc]
            return {
                "source": "Built-in Swiss-Prot Reference (UniProt)",
                "accession": ref["accession"],
                "uniprot_id": ref["accession"],
                "protein_name": ref["protein_name"],
                "gene_name": ref["gene_name"],
                "organism": ref["organism"],
                "length": len(ref["sequence"]),
                "sequence": ref["sequence"],
                "biochem_annotations": ref.get("biochem_annotations", {}) or curated_preset_ann,
            }

        _, by_acc = _ensure_local_corpus_loaded()
        if cand_acc in by_acc:
            r = by_acc[cand_acc]
            out_corp: Dict[str, Any] = {
                "source": "Built-in Local Reference Corpus (UniProt)",
                "accession": r["accession"],
                "uniprot_id": r["accession"],
                "protein_name": r["protein_name"],
                "gene_name": r["gene_name"],
                "organism": r["organism"],
                "length": r["length"],
                "sequence": r["sequence"],
                "biochem_annotations": curated_preset_ann,
            }
            if curated_preset_obj is not None:
                p_seq = curated_preset_obj["sequence"]
                pos_idx = r["sequence"].find(p_seq)
                if pos_idx >= 0 and len(p_seq) < len(r["sequence"]):
                    cleaved_lbl = f"Signal/Propeptide 1–{pos_idx}" if pos_idx > 0 else "N/C-terminal cleavage"
                    out_corp["mature_chain"] = {
                        "start": pos_idx + 1,
                        "end": pos_idx + len(p_seq),
                        "sequence": p_seq,
                        "length": len(p_seq),
                        "description": r["protein_name"],
                        "cleaved_summary": cleaved_lbl,
                    }
            return out_corp

        for preset in CLASSROOM_PRESET_PROTEINS.values():
            if preset["uniprot_id"] == cand_acc and preset["tag"] == "None":
                return {
                    "source": "UniProt (Offline Reference)",
                    "accession": cand_acc,
                    "uniprot_id": cand_acc,
                    "protein_name": preset["name"],
                    "gene_name": cand_acc,
                    "organism": preset["organism"],
                    "length": len(preset["sequence"]),
                    "sequence": preset["sequence"],
                    "biochem_annotations": preset.get("biochem_annotations", {}),
                }

    if re.fullmatch(r"[1-9][A-Z0-9]{3}(?:_[A-Z0-9])?", uid):
        raise ValueError(
            f"Could not retrieve UniProt ID '{uid}'. Tip: '{uid}' looks like an RCSB PDB ID "
            "(common for de novo & structural proteins) — switch to '🧊 RCSB PDB ID (De Novo / 3D)' "
            "or '💬 Describe in Plain English'."
        )

    raise ValueError(
        f"Could not retrieve UniProt ID '{uid}'. Please verify the accession code (e.g., P00698, P04637-2, or LYSC_CHICK) or check internet connectivity."
    )


_DE_NOVO_PDB_KEYWORDS = (
    "de novo",
    "computationally designed",
    "designed protein",
    "synthetic construct",
    "rfdiffusion",
    "proteinmpnn",
    "top7",
    "hallucinat",
    "artificial protein",
    "in silico design",
)


def _normalize_pdb_input(raw_id: str) -> Dict[str, Any]:
    """
    Normalizes user-supplied RCSB PDB inputs across 4-char IDs ('1QYS', '2KL8', '6D0T'),
    prefixed strings ('PDB:1QYS', 'PDB 2KL8', 'rcsb:6D0T'), chain selectors ('1QYS_A', '6D0T:B',
    '7B3A.A', '6D0T chain B'), FASTA headers ('>1QYS_A|Chain A|TOP7'), and RCSB URLs.
    """
    s = (raw_id or "").strip()
    if not s:
        raise ValueError(
            "Please enter a valid 4-character RCSB PDB ID (e.g., 1QYS, 2KL8, 6D0T, or 2VB1)."
        )

    first_line = s.splitlines()[0].strip()
    if first_line.startswith(">"):
        first_line = first_line[1:].strip()

    m_url = re.search(
        r"rcsb\.org/(?:structure|download|pdb/explore\.do\?structureId=)/?([1-9][A-Za-z0-9]{3})(?:[._]([A-Za-z0-9]{1,2}))?",
        first_line,
        flags=re.IGNORECASE,
    )
    if m_url:
        pdb_code = m_url.group(1).upper()
        req_chain = m_url.group(2).upper() if m_url.group(2) else None
        return {"pdb_id": pdb_code, "requested_chain": req_chain}

    work = re.sub(r"^(?:rcsb\s+)?pdb\s*(?:id)?\s*[:\-\s]+", "", first_line, flags=re.IGNORECASE).strip()
    work = re.sub(r"\.(?:pdb|cif|mmcif|ent)(?:\.gz)?\b", "", work, flags=re.IGNORECASE)

    m_pdb = re.match(
        r"^([1-9][A-Za-z0-9]{3})(?:(?:\s*[_:.\-|]\s*|\s+chain\s+)([A-Za-z0-9]{1,2}))?(?:\s|\||$)",
        work,
        flags=re.IGNORECASE,
    )
    if not m_pdb:
        raise ValueError(
            f"Unrecognized RCSB PDB ID format '{raw_id.strip()}'. "
            "Expected a 4-character PDB ID starting with 1–9 (e.g., 1QYS, 2KL8, 6D0T, or 6D0T_A)."
        )

    pdb_code = m_pdb.group(1).upper()
    req_chain = m_pdb.group(2).upper() if m_pdb.group(2) else None
    return {"pdb_id": pdb_code, "requested_chain": req_chain}


def fetch_pdb_by_id(pdb_id: str) -> Dict[str, Any]:
    """
    Fetches polypeptide sequence and structural metadata from RCSB PDB REST Data API v1
    given a 4-character PDB ID (with optional chain selector), supporting de novo / AI-designed
    entries (Top7 1QYS, 2KL8, 6D0T) as well as natural experimental structures (2VB1).
    """
    norm = _normalize_pdb_input(pdb_id)
    clean_pdb = norm["pdb_id"]
    requested_chain = norm["requested_chain"]
    headers = {"Accept": "application/json"}

    try:
        entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{clean_pdb}"
        entry_resp = _http_get(entry_url, headers=headers, timeout=12.0)
        if entry_resp.status_code != 200:
            raise ValueError(
                f"RCSB PDB entry '{clean_pdb}' was not found (HTTP {entry_resp.status_code}). "
                "Verify the 4-character PDB ID (e.g., 1QYS, 2KL8, 6D0T)."
            )
        entry_data = entry_resp.json() or {}
        title = str(((entry_data.get("struct") or {}).get("title")) or "").strip()
        kw_obj = entry_data.get("struct_keywords") or {}
        kw_str = " ".join(
            str(kw_obj.get(k) or "") for k in ("pdbx_keywords", "text")
        ).strip()
        poly_count = int(((entry_data.get("rcsb_entry_info") or {}).get("polymer_entity_count")) or 1)
        poly_count = max(1, min(poly_count, 8))

        candidates: List[Dict[str, Any]] = []
        all_chains: List[str] = []
        for ent_id in range(1, poly_count + 1):
            ent_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{clean_pdb}/{ent_id}"
            ent_resp = _http_get(ent_url, headers=headers, timeout=12.0)
            if ent_resp.status_code != 200:
                continue
            ent = ent_resp.json() or {}
            epoly = ent.get("entity_poly") or {}
            poly_type = str(epoly.get("type") or "").lower()
            raw_seq = str(epoly.get("pdbx_seq_one_letter_code_can") or epoly.get("pdbx_seq_one_letter_code") or "").strip()
            if "polypeptide" not in poly_type or not raw_seq:
                continue
            clean_seq, _, _ = clean_amino_acid_sequence(raw_seq)
            if not clean_seq:
                continue

            strand_str = str(epoly.get("pdbx_strand_id") or "")
            chains = [c.strip().upper() for c in re.split(r"[\s,;]+", strand_str) if c.strip()]
            for ch in chains:
                if ch not in all_chains:
                    all_chains.append(ch)
            candidates.append({
                "entity_id": ent_id,
                "ent": ent,
                "clean_seq": clean_seq,
                "chains": chains,
            })
    except ValueError:
        raise
    except requests.RequestException as exc:
        raise ValueError(
            f"Could not contact RCSB PDB for '{clean_pdb}' ({exc}). "
            "Check internet connectivity or paste the sequence in '🔤 Paste Amino Acid / FASTA'."
        ) from exc

    if not candidates:
        raise ValueError(
            f"RCSB PDB entry '{clean_pdb}' contains no polypeptide protein chains (it may be nucleic-acid or non-polymer only)."
        )

    chosen = None
    if requested_chain:
        for cand in candidates:
            if requested_chain in cand["chains"]:
                chosen = cand
                break
        if chosen is None:
            raise ValueError(
                f"Chain '{requested_chain}' was not found in polypeptide chains of PDB '{clean_pdb}' "
                f"(available: {', '.join(all_chains) or 'None'})."
            )
    else:
        chosen = candidates[0]
        if len(chosen["clean_seq"]) < 15:
            for cand in candidates[1:]:
                if len(cand["clean_seq"]) >= 15:
                    chosen = cand
                    break

    ent = chosen["ent"]
    clean_seq = chosen["clean_seq"]
    chains = chosen["chains"]
    selected_chain = requested_chain or (chains[0] if chains else "A")

    desc = str(((ent.get("rcsb_polymer_entity") or {}).get("pdbx_description")) or "").strip()
    if desc:
        if len(desc) <= 12 and title:
            short_title = title.split(":")[0].strip() if ":" in title and len(title.split(":")[0].strip()) >= 8 else title
            if len(short_title) > 75:
                short_title = short_title[:75].rstrip() + "…"
            protein_name = f"{desc} — {short_title}"
        else:
            protein_name = desc
    elif title:
        protein_name = title[:80].strip() + ("…" if len(title) > 80 else "")
    else:
        protein_name = f"PDB {clean_pdb} Chain {selected_chain}"

    orgs = [
        str(o.get("ncbi_scientific_name") or "").strip()
        for o in (ent.get("rcsb_entity_source_organism") or [])
        if o and str(o.get("ncbi_scientific_name") or "").strip()
    ]
    ctx_lower = f"{title} {kw_str} {desc}".lower()
    is_de_novo = (
        len(orgs) == 0
        or any(any(tok in o.lower() for tok in ("synthetic", "unidentified", "artificial")) for o in orgs)
        or any(kw in ctx_lower for kw in _DE_NOVO_PDB_KEYWORDS)
    )
    organism = "De Novo / Synthetic Construct" if is_de_novo else orgs[0]

    uids = (ent.get("rcsb_polymer_entity_container_identifiers") or {}).get("uniprot_ids") or []
    linked_uid = str(uids[0]).strip().upper() if (uids and str(uids[0]).strip() and not is_de_novo) else ""

    return {
        "source": f"RCSB PDB {clean_pdb}" + (f" (Chain {selected_chain})" if selected_chain else ""),
        "accession": f"{clean_pdb}_{selected_chain}" if requested_chain else clean_pdb,
        "pdb_id": clean_pdb,
        "chain_id": selected_chain,
        "chains": chains,
        "uniprot_id": linked_uid,
        "protein_name": protein_name,
        "title": title,
        "gene_name": "Synthetic / De Novo" if is_de_novo else "N/A",
        "organism": organism,
        "length": len(clean_seq),
        "sequence": clean_seq,
        "is_de_novo": is_de_novo,
        "biochem_annotations": {},
    }


_NCBI_REFSEQ_PROTEIN_PREFIXES = ("NP_", "XP_", "WP_", "YP_", "AP_", "ZP_")
_NCBI_REFSEQ_NUCCORE_PREFIXES = ("NM_", "XM_", "NC_", "NG_", "NR_", "XR_", "NT_", "NW_", "NZ_", "AC_")


def _normalize_ncbi_input(raw_id: str) -> str:
    """
    Normalizes NCBI accession inputs across RefSeq (NP_990612.2, NM_205281.2, WP_...),
    GenBank/EMBL/DDBJ protein (QHD43416.1, AAA12345.1) and nucleotide (M21110.1, BC004147,
    MN908947.3), PDB chains (1LYZ_A), FASTA pipe IDs ('ref|NP_990612.2|'), and NCBI URLs.
    """
    s = (raw_id or "").strip()
    if not s:
        raise ValueError(
            "Please enter a valid NCBI accession (e.g., NP_990612.2, NM_205281.2, or M21110.1)."
        )

    first_line = s.splitlines()[0].strip()
    if first_line.startswith(">"):
        first_line = first_line[1:].strip()

    m_url = re.search(
        r"ncbi\.nlm\.nih\.gov/(?:protein|nuccore|nucleotide|ipg)/([A-Za-z0-9_.\-]+)",
        first_line,
        flags=re.IGNORECASE,
    )
    if m_url:
        token = m_url.group(1)
    else:
        token = re.sub(r"^(?:ncbi|refseq|genbank|id|acc)\s*:\s*", "", first_line, flags=re.IGNORECASE)
        m_pdb = re.match(r"^pdb\|([0-9][A-Za-z0-9]{3})\|([A-Za-z0-9]+)", token, flags=re.IGNORECASE)
        if m_pdb:
            token = f"{m_pdb.group(1).upper()}_{m_pdb.group(2)}"
        else:
            m_pipe = re.match(
                r"^(?:ref|gb|emb|dbj|tpg|tpe|tpd|pir|prf)\|([A-Za-z0-9_.\-]+)\|",
                token,
                flags=re.IGNORECASE,
            )
            if m_pipe:
                token = m_pipe.group(1)
            else:
                token = token.split()[0]

    clean_id = token.strip().strip("|,;/?#")
    if not clean_id or not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{3,}$", clean_id):
        raise ValueError(
            f"Unrecognized NCBI accession format '{raw_id.strip()}'. Expected e.g. NP_990612.2, NM_205281.2, or M21110.1."
        )
    return clean_id


def _classify_ncbi_accession_db(clean_id: str) -> str:
    """
    Deterministically classifies an NCBI/INSDC accession into Entrez database 'protein' or 'nuccore'
    according to NCBI RefSeq & INSDC accession prefix rules.
    """
    u = clean_id.strip().upper()
    if u.startswith(_NCBI_REFSEQ_NUCCORE_PREFIXES):
        return "nuccore"
    if u.startswith(_NCBI_REFSEQ_PROTEIN_PREFIXES):
        return "protein"
    # PDB chain format (e.g. 1LYZ_A, 6VXX_B)
    if re.match(r"^[0-9][A-Z0-9]{3}_[A-Z0-9]+$", u):
        return "protein"
    # INSDC / GenBank / EMBL / DDBJ protein: strictly 3 letters + 5 or 7 digits (+ optional .version)
    if re.match(r"^[A-Z]{3}\d{5}(?:\.\d+)?$", u) or re.match(r"^[A-Z]{3}\d{7}(?:\.\d+)?$", u):
        return "protein"
    # INSDC nucleotide: 1 letter + 5 digits (M21110.1), 2 letters + 6 or 8 digits (BC004147, MN908947.3),
    # or WGS/TSA nucleotide contigs (4-6 letters + 8-10 digits)
    if (
        re.match(r"^[A-Z]\d{5}(?:\.\d+)?$", u)
        or re.match(r"^[A-Z]{2}\d{6}(?:\.\d+)?$", u)
        or re.match(r"^[A-Z]{2}\d{8}(?:\.\d+)?$", u)
        or re.match(r"^[A-Z]{4,6}\d{8,10}(?:\.\d+)?$", u)
    ):
        return "nuccore"
    return "protein"


def fetch_ncbi_by_id(ncbi_id: str) -> Dict[str, Any]:
    """
    Fetches protein or nucleotide sequences from NCBI by accession using Entrez E-utilities.
    Supports RefSeq (NP_/XP_/WP_/YP_ & NM_/XM_/NC_/NG_/NR_/NZ_), GenBank/EMBL/DDBJ
    (e.g. QHD43416.1, M21110.1, BC004147, MN908947.3), PDB chains (1LYZ_A), FASTA pipe IDs,
    and NCBI URLs — with automatic retry/backoff and dual-DB (protein <-> nuccore) fallback.
    """
    clean_id = _normalize_ncbi_input(ncbi_id)
    primary_db = _classify_ncbi_accession_db(clean_id)
    alt_db = "nuccore" if primary_db == "protein" else "protein"

    fasta_text = ""
    used_db = primary_db
    last_exc: Optional[Exception] = None
    for db in (primary_db, alt_db):
        url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            f"?db={db}&id={requests.utils.quote(clean_id)}&rettype=fasta&retmode=text&tool=ProPurifyStudio"
        )
        try:
            response = _http_get(url, headers={"Accept": "text/plain"}, timeout=12)
            body = (response.text or "").strip()
            if response.status_code == 200 and body.startswith(">"):
                fasta_text = response.text
                used_db = db
                break
        except requests.RequestException as exc:
            last_exc = exc

    if not fasta_text:
        if last_exc is not None:
            raise ValueError(
                f"Could not connect to NCBI to retrieve '{clean_id}': {last_exc}"
            ) from last_exc
        raise ValueError(
            f"Could not retrieve NCBI ID '{clean_id}'. Please verify the accession ID (e.g., NP_990612.2, NM_205281.2, M21110.1)."
        )

    is_nucleotide = used_db == "nuccore"
    if is_nucleotide:
        orf_result = find_longest_orf(fasta_text)
        protein_seq = orf_result["sequence"]
        if not protein_seq:
            raise ValueError(
                f"Retrieved nucleotide entry '{clean_id}' from NCBI, but no translatable coding sequence was found."
            )
        raw_header = orf_result.get("header")
        strand_label = "+" if orf_result.get("frame", 1) > 0 else "-"
        extra_note = (
            f"Translated from NCBI nucleotide (Frame {orf_result['frame']}, "
            f"{strand_label} strand, ORF length: {len(protein_seq)} aa)"
        )
        source_label = "NCBI (Nucleotide ORF)"
    else:
        protein_seq, raw_header, _ = clean_amino_acid_sequence(fasta_text)
        if not protein_seq:
            raise ValueError(
                f"Retrieved entry '{clean_id}' from NCBI, but it contained no valid amino acid sequence."
            )
        extra_note = ""
        source_label = "NCBI"

    clean_name, organism = _parse_ncbi_fasta_header(raw_header, clean_id)
    return {
        "source": source_label,
        "accession": clean_id,
        "protein_name": clean_name,
        "gene_name": clean_id,
        "organism": organism,
        "length": len(protein_seq),
        "sequence": protein_seq,
        "note": extra_note,
        "biochem_annotations": {},
    }


_NT_COMPLEMENT_TRANS = str.maketrans(
    "ATCGRYSWKMBDHVN",
    "TAGCYRSWMKVHDBN",
)


def _clean_nucleotide_input(raw_text: str) -> Tuple[str, Optional[str], List[str]]:
    """
    Cleans raw DNA/RNA or FASTA input into an uppercase DNA string (U -> T):
    - Properly skips '>' FASTA definition lines so header letters never leak into nucleotides
      and frameshift codons.
    - Stops at a second '>' FASTA header with a warning so multi-FASTA inputs never fuse.
    - Preserves IUPAC ambiguity bases (N, R, Y, S, W, K, M, B, D, H, V) in place so
      ambiguous codons translate to 'X' without causing downstream frameshifts.
    """
    lines = (raw_text or "").strip().splitlines()
    header: Optional[str] = None
    seq_lines: List[str] = []
    additional_entries = 0
    stop_reading = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is None and not seq_lines:
                header = line[1:].strip()
            else:
                additional_entries += 1
                stop_reading = True
            continue
        if not stop_reading:
            seq_lines.append(line)

    raw_seq = "".join(seq_lines).upper().replace("U", "T")
    raw_seq = re.sub(r"[\s\d\.\-\*]", "", raw_seq)

    iupac_nt = set("ATCGRYSWKMBDHVN")
    invalid_chars = sorted(set(raw_seq) - iupac_nt)
    clean_seq = re.sub(r"[^ATCGRYSWKMBDHVN]", "", raw_seq)

    warnings: List[str] = []
    if additional_entries > 0:
        entry_label = header or "Entry 1"
        if len(entry_label) > 60:
            entry_label = entry_label[:57] + "..."
        warnings.append(
            f"Multiple FASTA entries detected ({additional_entries + 1} total); translated only the first entry "
            f"('{entry_label}'). Paste one sequence at a time."
        )
    if invalid_chars:
        warnings.append(
            f"Ignored non-IUPAC nucleotide characters: {', '.join(invalid_chars)}."
        )

    return clean_seq, header, warnings


def reverse_complement(dna_rna_seq: str) -> str:
    """Returns the IUPAC-aware reverse complement of a DNA/RNA sequence (uppercase T alphabet)."""
    if ">" in (dna_rna_seq or ""):
        clean_nt, _, _ = _clean_nucleotide_input(dna_rna_seq)
    else:
        clean_nt = re.sub(r"[^ATCGRYSWKMBDHVN]", "", (dna_rna_seq or "").strip().upper().replace("U", "T"))
    return clean_nt.translate(_NT_COMPLEMENT_TRANS)[::-1]


def translate_nucleotide(dna_rna_seq: str, frame: int = 1) -> str:
    """
    Translates DNA or RNA sequence directly using the NCBI Standard Genetic Code (transl_table=1).
    Supports 6 reading frames: +1, +2, +3 (forward strand) and -1, -2, -3 (reverse complement).
    Strips FASTA headers cleanly and preserves IUPAC ambiguous bases (N -> 'X') without frameshifting.
    """
    if frame not in (1, 2, 3, -1, -2, -3):
        raise ValueError(f"Reading frame must be 1, 2, 3, -1, -2, or -3 (got {frame}).")
    seq, _, _ = _clean_nucleotide_input(dna_rna_seq)
    if frame < 0:
        seq = seq.translate(_NT_COMPLEMENT_TRANS)[::-1]
        start_idx = abs(frame) - 1
    else:
        start_idx = frame - 1
    protein: List[str] = []
    for i in range(start_idx, len(seq) - 2, 3):
        codon = seq[i:i + 3]
        protein.append(CODON_TABLE.get(codon, "X"))
    return "".join(protein)


def find_longest_orf(dna_rna_seq: str, search_reverse_strand: bool = True) -> Dict[str, Any]:
    """
    Finds the longest Open Reading Frame (ATG to Stop codon) across all 6 reading frames
    (+1, +2, +3 forward and -1, -2, -3 reverse complement, preferring forward strand on exact ties).
    Properly strips FASTA headers and preserves IUPAC ambiguous bases (N -> 'X') without frameshifting.
    """
    dna_fwd, header, warnings = _clean_nucleotide_input(dna_rna_seq)
    seq_len = len(dna_fwd)

    best_orf = ""
    best_frame = 1
    best_strand = "+"
    best_start = 0
    best_end = 0

    strands: List[Tuple[str, str]] = [("+", dna_fwd)]
    if search_reverse_strand and dna_fwd:
        strands.append(("-", dna_fwd.translate(_NT_COMPLEMENT_TRANS)[::-1]))

    codon_get = CODON_TABLE.get
    for strand_sign, seq_str in strands:
        s_len = len(seq_str)
        for frame_offset in (0, 1, 2):
            i = frame_offset
            while i < s_len - 2:
                if seq_str[i:i + 3] == "ATG":
                    orf_codons: List[str] = []
                    j = i
                    found_stop = False
                    while j < s_len - 2:
                        aa = codon_get(seq_str[j:j + 3], "X")
                        if aa == "*":
                            found_stop = True
                            break
                        orf_codons.append(aa)
                        j += 3
                    orf_protein = "".join(orf_codons)
                    if len(orf_protein) > len(best_orf):
                        best_orf = orf_protein
                        best_strand = strand_sign
                        best_frame = (frame_offset + 1) if strand_sign == "+" else -(frame_offset + 1)
                        rc_end = (j + 3) if found_stop else j
                        if strand_sign == "+":
                            best_start = i + 1
                            best_end = rc_end
                        else:
                            best_start = s_len - rc_end + 1
                            best_end = s_len - i
                    if found_stop:
                        i = j + 3
                    else:
                        break
                else:
                    i += 3

    if not best_orf:
        fallback = translate_nucleotide(dna_fwd, frame=1).split("*")[0]
        return {
            "sequence": fallback,
            "frame": 1,
            "strand": "+",
            "has_start_codon": False,
            "orf_length_aa": len(fallback),
            "start_bp": 1 if seq_len > 0 else 0,
            "end_bp": seq_len,
            "header": header,
            "warnings": warnings,
        }

    return {
        "sequence": best_orf,
        "frame": best_frame,
        "strand": best_strand,
        "has_start_codon": True,
        "orf_length_aa": len(best_orf),
        "start_bp": best_start,
        "end_bp": best_end,
        "header": header,
        "warnings": warnings,
    }


# ==============================================================================
# UNIVERSAL TREE-OF-LIFE ORGANISM EXTRACTION & ZERO-COLLISION CONSTRUCT PARSER
# ==============================================================================
# Each tuple in _ORGANISM_CATALOG is:
#   (regex_pattern, primary_taxid, clade_taxid, canonical_sci_name, extra_taxids, context_guarded)
# Ordered longest / most-specific scientific & viral phrases FIRST, short common shorthands LAST.
_ORGANISM_CATALOG: List[Tuple[str, int, int, str, Tuple[int, ...], bool]] = [
    # --- Viruses & Bacteriophages (placed BEFORE plant/host names like "tobacco", "tomato", "human") ---
    (r"severe\s+acute\s+respiratory\s+syndrome\s+coronavirus\s+2", 2697049, 2697049, "Severe acute respiratory syndrome coronavirus 2", (2697049, 694009), False),
    (r"severe\s+acute\s+respiratory\s+syndrome\s+coronavirus", 694009, 694009, "Severe acute respiratory syndrome coronavirus", (694009,), False),
    (r"middle\s+east\s+respiratory\s+syndrome[-\s]+related\s+coronavirus", 1335626, 1335626, "Middle East respiratory syndrome-related coronavirus", (1335626,), False),
    (r"sars-?cov-?2", 2697049, 2697049, "Severe acute respiratory syndrome coronavirus 2", (2697049, 694009), False),
    (r"covid-?19\s+virus", 2697049, 2697049, "Severe acute respiratory syndrome coronavirus 2", (2697049,), False),
    (r"sars-?cov-?1", 694009, 694009, "Severe acute respiratory syndrome coronavirus", (694009,), False),
    (r"human\s+immunodeficiency\s+virus(?:\s+type)?\s+1", 11676, 11676, "Human immunodeficiency virus 1", (11676, 11706), False),
    (r"human\s+immunodeficiency\s+virus(?:\s+type)?\s+2", 11709, 11709, "Human immunodeficiency virus 2", (11709,), False),
    (r"hiv-?1", 11676, 11676, "Human immunodeficiency virus 1", (11676, 11706), False),
    (r"tobacco\s+etch\s+virus", 12227, 12227, "Tobacco etch virus", (12227,), False),
    (r"tobacco\s+mosaic\s+virus", 12242, 12242, "Tobacco mosaic virus", (12242,), False),
    (r"turnip\s+mosaic\s+virus", 12230, 12230, "Turnip mosaic virus", (12230,), False),
    (r"cucumber\s+mosaic\s+virus", 12305, 12305, "Cucumber mosaic virus", (12305,), False),
    (r"tomato\s+bushy\s+stunt\s+virus", 12145, 12145, "Tomato bushy stunt virus", (12145,), False),
    (r"cauliflower\s+mosaic\s+virus", 10641, 10641, "Cauliflower mosaic virus", (10641,), False),
    (r"influenza\s+a\s+virus", 11320, 11320, "Influenza A virus", (11320, 211044, 382835), False),
    (r"influenza\s+b\s+virus", 11520, 11520, "Influenza B virus", (11520,), False),
    (r"influenza\s+a", 11320, 11320, "Influenza A virus", (11320, 211044, 382835), True),
    (r"influenza\s+virus", 11320, 11320, "Influenza A virus", (11320, 211044, 382835), False),
    (r"influenza", 11320, 11320, "Influenza A virus", (11320, 211044, 382835), True),
    (r"(?:(?:human|simian|avian|duck)\s+)?hepatitis\s+c\s+virus", 11103, 11103, "Hepatitis C virus", (11103, 3052230), False),
    (r"(?:(?:human|simian|avian|duck)\s+)?hepatitis\s+b\s+virus", 10407, 10407, "Hepatitis B virus", (10407,), False),
    (r"(?:(?:human|simian|avian|duck)\s+)?hepatitis\s+a\s+virus", 12092, 12092, "Hepatitis A virus", (12092, 208726), False),
    (r"epstein-?barr\s+virus", 10376, 10376, "Human gammaherpesvirus 4", (10376, 10377), False),
    (r"human\s+(?:gamma)?herpesvirus\s+4", 10376, 10376, "Human gammaherpesvirus 4", (10376, 10377), False),
    (r"human\s+(?:alpha)?herpesvirus\s+1", 10298, 10298, "Human alphaherpesvirus 1", (10298, 10299), False),
    (r"herpes\s+simplex\s+virus(?:\s+1)?", 10298, 10298, "Human alphaherpesvirus 1", (10298, 10299), False),
    (r"human\s+(?:beta)?herpesvirus\s+5", 10359, 10359, "Human betaherpesvirus 5", (10359,), False),
    (r"vaccinia\s+virus", 10245, 10245, "Vaccinia virus", (10245, 10249, 10254), False),
    (r"zaire\s+ebolavirus", 186538, 186536, "Zaire ebolavirus", (186538, 186536, 128952), False),
    (r"ebola\s+virus", 186538, 186536, "Zaire ebolavirus", (186538, 186536, 128952), False),
    (r"dengue\s+virus(?:\s+type\s+[1-4])?", 12637, 12637, "Dengue virus", (12637, 11060), False),
    (r"zika\s+virus", 64320, 64320, "Zika virus", (64320,), False),
    (r"west\s+nile\s+virus", 11082, 11082, "West Nile virus", (11082,), False),
    (r"yellow\s+fever\s+virus", 11089, 11089, "Yellow fever virus", (11089,), False),
    (r"(?:human\s+)?poliovirus(?:\s+type)?(?:\s+[1-3])?", 12080, 12080, "Human poliovirus 1", (12080, 138948), False),
    (r"vesicular\s+stomatitis(?:\s+indiana)?\s+virus", 11276, 11276, "Vesicular stomatitis Indiana virus", (11276,), False),
    (r"rabies\s+(?:lyssa)?virus", 11292, 11292, "Rabies virus", (11292,), False),
    (r"measles\s+(?:morbilli)?virus", 11234, 11234, "Measles virus", (11234,), False),
    (r"simian\s+virus\s+40", 1891767, 1891767, "Macaca mulatta polyomavirus 1", (1891767, 10633), False),
    (r"autographa\s+californica\s+(?:nuclear\s+polyhedrosis\s+virus|nucleopolyhedrovirus)", 46015, 46015, "Autographa californica nuclear polyhedrosis virus", (46015,), False),
    (r"(?:enterobacteria|escherichia)\s+(?:bacterio)?phage\s+t4", 10665, 10665, "Enterobacteria phage T4", (10665, 2681598), False),
    (r"tequatrovirus\s+t4", 10665, 10665, "Enterobacteria phage T4", (10665, 2681598), False),
    (r"(?:bacterio)?phage\s+t4", 10665, 10665, "Enterobacteria phage T4", (10665, 2681598), False),
    (r"t4\s+(?:bacterio)?phage", 10665, 10665, "Enterobacteria phage T4", (10665, 2681598), False),
    (r"(?:enterobacteria|escherichia)\s+(?:bacterio)?phage\s+t7", 10760, 10760, "Enterobacteria phage T7", (10760,), False),
    (r"teseptimavirus\s+t7", 10760, 10760, "Enterobacteria phage T7", (10760,), False),
    (r"(?:bacterio)?phage\s+t7", 10760, 10760, "Enterobacteria phage T7", (10760,), False),
    (r"t7\s+(?:bacterio)?phage", 10760, 10760, "Enterobacteria phage T7", (10760,), False),
    (r"(?:enterobacteria|escherichia)\s+(?:bacterio)?phage\s+lambda", 10710, 2681611, "Escherichia phage lambda", (10710, 2681611), False),
    (r"lambdavirus\s+lambda", 10710, 2681611, "Escherichia phage lambda", (10710, 2681611), False),
    (r"(?:bacterio)?phage\s+lambda", 10710, 2681611, "Escherichia phage lambda", (10710, 2681611), False),
    (r"lambda\s+(?:bacterio)?phage", 10710, 2681611, "Escherichia phage lambda", (10710, 2681611), False),
    (r"(?:enterobacteria|escherichia)?\s*(?:bacterio)?phage\s+phix174", 10847, 10847, "Enterobacteria phage phiX174", (10847,), False),
    (r"(?:enterobacteria|escherichia)?\s*(?:bacterio)?phage\s+m13", 10870, 10870, "Enterobacteria phage M13", (10870,), False),
    (r"(?:bacillus\s+)?(?:bacterio)?phage\s+phi-?29", 10756, 10756, "Bacillus phage phi29", (10756,), False),

    # --- Bacteria ---
    (r"escherichia\s+coli\s+(?:strain\s+)?k-?12", 83333, 83333, "Escherichia coli (strain K12)", (83333, 562), False),
    (r"e(?:\.\s*|\s+)coli\s+(?:strain\s+)?k-?12", 83333, 83333, "Escherichia coli (strain K12)", (83333, 562), False),
    (r"escherichia\s+coli\s+(?:strain\s+)?bl21(?:\s*\(de3\))?", 83333, 562, "Escherichia coli", (469008, 511693, 83333, 562), False),
    (r"e(?:\.\s*|\s+)coli\s+(?:strain\s+)?bl21(?:\s*\(de3\))?", 83333, 562, "Escherichia coli", (469008, 511693, 83333, 562), False),
    (r"escherichia\s+coli", 83333, 562, "Escherichia coli", (83333, 562, 83334, 199310), False),
    (r"e(?:\.\s*|\s+)coli", 83333, 562, "Escherichia coli", (83333, 562, 83334, 199310), False),
    (r"ecoli", 83333, 562, "Escherichia coli", (83333, 562), False),
    (r"bacillus\s+subtilis", 224308, 1423, "Bacillus subtilis", (224308, 1423), False),
    (r"b(?:\.\s*|\s+)subtilis", 224308, 1423, "Bacillus subtilis", (224308, 1423), False),
    (r"thermus\s+aquaticus", 271, 271, "Thermus aquaticus", (271, 65393, 498848), False),
    (r"t(?:\.\s*|\s+)aquaticus", 271, 271, "Thermus aquaticus", (271, 65393, 498848), False),
    (r"thermus\s+thermophilus", 274, 274, "Thermus thermophilus", (274, 300852, 262724), False),
    (r"t(?:\.\s*|\s+)thermophilus", 274, 274, "Thermus thermophilus", (274, 300852, 262724), False),
    (r"staphylococcus\s+aureus", 93061, 1280, "Staphylococcus aureus", (93061, 1280, 158879, 93062, 282458, 367830, 46170), False),
    (r"s(?:\.\s*|\s+)aureus", 93061, 1280, "Staphylococcus aureus", (93061, 1280, 158879, 93062, 282458, 367830, 46170), False),
    (r"staph(?:\.\s*|\s+)aureus", 93061, 1280, "Staphylococcus aureus", (93061, 1280, 158879), False),
    (r"mycobacterium\s+tuberculosis", 83332, 1773, "Mycobacterium tuberculosis", (83332, 1773), False),
    (r"m(?:\.\s*|\s+)tuberculosis", 83332, 1773, "Mycobacterium tuberculosis", (83332, 1773), False),
    (r"(?:mycolicibacterium|mycobacterium)\s+smegmatis", 246196, 1772, "Mycolicibacterium smegmatis", (246196, 1772), False),
    (r"m(?:\.\s*|\s+)smegmatis", 246196, 1772, "Mycolicibacterium smegmatis", (246196, 1772), False),
    (r"pseudomonas\s+aeruginosa", 208964, 287, "Pseudomonas aeruginosa", (208964, 287), False),
    (r"p(?:\.\s*|\s+)aeruginosa", 208964, 287, "Pseudomonas aeruginosa", (208964, 287), False),
    (r"streptococcus\s+pyogenes", 160490, 1314, "Streptococcus pyogenes", (160490, 1314, 301447), False),
    (r"s(?:\.\s*|\s+)pyogenes", 160490, 1314, "Streptococcus pyogenes", (160490, 1314, 301447), False),
    (r"strep(?:\.\s*|\s+)pyogenes", 160490, 1314, "Streptococcus pyogenes", (160490, 1314), False),
    (r"streptococcus\s+pneumoniae", 171101, 1313, "Streptococcus pneumoniae", (171101, 170187, 1313), False),
    (r"s(?:\.\s*|\s+)pneumoniae", 171101, 1313, "Streptococcus pneumoniae", (171101, 170187, 1313), False),
    (r"synechocystis\s+sp\.?\s+pcc\s*6803\b", 1111708, 1148, "Synechocystis sp. PCC 6803", (1111708, 1148), False),
    (r"synechocystis\s+sp\.?\s+pcc\s*6714\b", 1147, 1148, "Synechocystis sp. PCC 6714", (1147, 1148), False),
    (r"synechocystis", 1111708, 1148, "Synechocystis sp. PCC 6803", (1111708, 1147, 1148), True),
    (r"vibrio\s+cholerae", 243277, 666, "Vibrio cholerae", (243277, 666), False),
    (r"v(?:\.\s*|\s+)cholerae", 243277, 666, "Vibrio cholerae", (243277, 666), False),
    (r"helicobacter\s+pylori", 85962, 210, "Helicobacter pylori", (85962, 210), False),
    (r"h(?:\.\s*|\s+)pylori", 85962, 210, "Helicobacter pylori", (85962, 210), False),
    (r"streptomyces\s+coelicolor", 100226, 1902, "Streptomyces coelicolor", (100226, 1902), False),
    (r"s(?:\.\s*|\s+)coelicolor", 100226, 1902, "Streptomyces coelicolor", (100226, 1902), False),
    (r"streptomyces\s+avidinii", 1895, 1895, "Streptomyces avidinii", (1895,), False),
    (r"s(?:\.\s*|\s+)avidinii", 1895, 1895, "Streptomyces avidinii", (1895,), False),
    (r"caulobacter\s+(?:vibrioides|crescentus)", 190650, 155892, "Caulobacter vibrioides", (190650, 155892, 565050), False),
    (r"c(?:\.\s*|\s+)(?:crescentus|vibrioides)", 190650, 155892, "Caulobacter vibrioides", (190650, 155892, 565050), False),
    (r"salmonella\s+(?:enterica(?:\s+subsp\.\s+enterica)?(?:\s+serovar\s+typhimurium)?|typhimurium)", 99287, 28901, "Salmonella typhimurium", (99287, 28901, 90371, 590), False),
    (r"s(?:\.\s*|\s+)(?:typhimurium|enterica)", 99287, 28901, "Salmonella typhimurium", (99287, 28901, 90371, 590), False),
    (r"klebsiella\s+pneumoniae", 272620, 573, "Klebsiella pneumoniae", (272620, 573), False),
    (r"k(?:\.\s*|\s+)pneumoniae", 272620, 573, "Klebsiella pneumoniae", (272620, 573), False),
    (r"listeria\s+monocytogenes", 169963, 1639, "Listeria monocytogenes", (169963, 1639), False),
    (r"l(?:\.\s*|\s+)monocytogenes", 169963, 1639, "Listeria monocytogenes", (169963, 1639), False),
    (r"corynebacterium\s+glutamicum", 196627, 1718, "Corynebacterium glutamicum", (196627, 1718), False),
    (r"c(?:\.\s*|\s+)glutamicum", 196627, 1718, "Corynebacterium glutamicum", (196627, 1718), False),
    (r"deinococcus\s+radiodurans", 243230, 1299, "Deinococcus radiodurans", (243230, 1299), False),
    (r"d(?:\.\s*|\s+)radiodurans", 243230, 1299, "Deinococcus radiodurans", (243230, 1299), False),
    (r"thermotoga\s+maritima", 243274, 2336, "Thermotoga maritima", (243274, 2336), False),
    (r"t(?:\.\s*|\s+)maritima", 243274, 2336, "Thermotoga maritima", (243274, 2336), False),
    (r"aquifex\s+aeolicus", 224324, 63363, "Aquifex aeolicus", (224324, 63363), False),
    (r"a(?:\.\s*|\s+)aeolicus", 224324, 63363, "Aquifex aeolicus", (224324, 63363), False),
    (r"haemophilus\s+influenzae", 71421, 727, "Haemophilus influenzae", (71421, 727), False),
    (r"h(?:\.\s*|\s+)influenzae", 71421, 727, "Haemophilus influenzae", (71421, 727), False),
    (r"neisseria\s+meningitidis", 122586, 487, "Neisseria meningitidis", (122586, 487), False),
    (r"n(?:\.\s*|\s+)meningitidis", 122586, 487, "Neisseria meningitidis", (122586, 487), False),
    (r"campylobacter\s+jejuni", 192222, 197, "Campylobacter jejuni", (192222, 197), False),
    (r"c(?:\.\s*|\s+)jejuni", 192222, 197, "Campylobacter jejuni", (192222, 197), False),
    (r"yersinia\s+pestis", 214092, 632, "Yersinia pestis", (214092, 632), False),
    (r"y(?:\.\s*|\s+)pestis", 214092, 632, "Yersinia pestis", (214092, 632), False),
    (r"acinetobacter\s+baumannii", 400667, 470, "Acinetobacter baumannii", (400667, 470), False),
    (r"a(?:\.\s*|\s+)baumannii", 400667, 470, "Acinetobacter baumannii", (400667, 470), False),
    (r"(?:mycoplasma|mycoplasmoides)\s+genitalium", 243273, 2097, "Mycoplasma genitalium", (243273, 2097), False),
    (r"m(?:\.\s*|\s+)genitalium", 243273, 2097, "Mycoplasma genitalium", (243273, 2097), False),
    (r"chlamydia\s+trachomatis", 272561, 813, "Chlamydia trachomatis", (272561, 813), False),
    (r"c(?:\.\s*|\s+)trachomatis", 272561, 813, "Chlamydia trachomatis", (272561, 813), False),
    (r"(?:klebsiella|enterobacter)\s+aerogenes", 548, 548, "Klebsiella aerogenes", (548,), False),
    (r"(?:sporosarcina|bacillus)\s+pasteurii", 1474, 1474, "Sporosarcina pasteurii", (1474,), False),

    # --- Archaea ---
    (r"pyrococcus\s+furiosus", 186497, 2261, "Pyrococcus furiosus", (186497, 2261), False),
    (r"p(?:\.\s*|\s+)furiosus", 186497, 2261, "Pyrococcus furiosus", (186497, 2261), False),
    (r"pyrococcus\s+horikoshii", 70601, 53953, "Pyrococcus horikoshii", (70601, 53953), False),
    (r"p(?:\.\s*|\s+)horikoshii", 70601, 53953, "Pyrococcus horikoshii", (70601, 53953), False),
    (r"pyrococcus\s+abyssi", 272844, 29292, "Pyrococcus abyssi", (272844, 29292), False),
    (r"p(?:\.\s*|\s+)abyssi", 272844, 29292, "Pyrococcus abyssi", (272844, 29292), False),
    (r"thermococcus\s+kodakara?ensis", 69014, 311400, "Thermococcus kodakarensis", (69014, 311400), False),
    (r"t(?:\.\s*|\s+)kodakara?ensis", 69014, 311400, "Thermococcus kodakarensis", (69014, 311400), False),
    (r"(?:methanocaldococcus|methanococcus)\s+jannaschii", 243232, 2190, "Methanocaldococcus jannaschii", (243232, 2190), False),
    (r"m(?:\.\s*|\s+)jannaschii", 243232, 2190, "Methanocaldococcus jannaschii", (243232, 2190), False),
    (r"sulfolobus\s+acidocaldarius", 330779, 2285, "Sulfolobus acidocaldarius", (330779, 2285), False),
    (r"s(?:\.\s*|\s+)acidocaldarius", 330779, 2285, "Sulfolobus acidocaldarius", (330779, 2285), False),
    (r"(?:saccharolobus|sulfolobus)\s+solfataricus", 273057, 2287, "Saccharolobus solfataricus", (273057, 2287), False),
    (r"s(?:\.\s*|\s+)solfataricus", 273057, 2287, "Saccharolobus solfataricus", (273057, 2287), False),
    (r"haloferax\s+volcanii", 309800, 2246, "Haloferax volcanii", (309800, 2246), False),
    (r"h(?:\.\s*|\s+)volcanii", 309800, 2246, "Haloferax volcanii", (309800, 2246), False),
    (r"halobacterium\s+salinarum", 64091, 2242, "Halobacterium salinarum", (64091, 478009, 2242), False),
    (r"h(?:\.\s*|\s+)salinarum", 64091, 2242, "Halobacterium salinarum", (64091, 478009, 2242), False),
    (r"archaeoglobus\s+fulgidus", 224325, 2234, "Archaeoglobus fulgidus", (224325, 2234), False),
    (r"a(?:\.\s*|\s+)fulgidus", 224325, 2234, "Archaeoglobus fulgidus", (224325, 2234), False),
    (r"methanothermobacter\s+thermautotrophicus", 187420, 145262, "Methanothermobacter thermautotrophicus", (187420, 145262), False),
    (r"m(?:\.\s*|\s+)thermautotrophicus", 187420, 145262, "Methanothermobacter thermautotrophicus", (187420, 145262), False),
    (r"methanosarcina\s+acetivorans", 188937, 2214, "Methanosarcina acetivorans", (188937, 2214), False),
    (r"m(?:\.\s*|\s+)acetivorans", 188937, 2214, "Methanosarcina acetivorans", (188937, 2214), False),
    (r"methanosarcina\s+mazei", 192952, 2209, "Methanosarcina mazei", (192952, 2209), False),
    (r"thermoplasma\s+acidophilum", 273075, 2303, "Thermoplasma acidophilum", (273075, 2303), False),
    (r"t(?:\.\s*|\s+)acidophilum", 273075, 2303, "Thermoplasma acidophilum", (273075, 2303), False),
    (r"aeropyrum\s+pernix", 272557, 56636, "Aeropyrum pernix", (272557, 56636), False),
    (r"a(?:\.\s*|\s+)pernix", 272557, 56636, "Aeropyrum pernix", (272557, 56636), False),
    (r"(?:candidatus\s+)?nitrosopumilus\s+maritimus", 436308, 338190, "Candidatus Nitrosopumilus maritimus", (436308, 338190), False),
    (r"(?:uncultured|antarctic|unknown)\s+(?:marine\s+)?(?:bacterium|prokaryotic\s+organism|archaeon|euryarchaeote|crenarchaeote)(?:\s+(?-i:(?=[A-Za-z0-9_-]*[0-9])(?![A-Za-z0-9_-]*[a-z]{3,})[A-Z0-9][A-Za-z0-9_-]*))?", None, None, "Uncultured microorganism", (), False),
    (r"sulfolobus", 330779, 2284, "Sulfolobus", (330779, 2285, 273057, 2287, 2284), True),
    (r"halobacterium", 64091, 2242, "Halobacterium salinarum", (64091, 478009, 2242, 2239), True),

    # --- Eukaryota: Animals / Metazoa ---
    (r"chicken\s+egg[-\s]+white", 9031, 9031, "Gallus gallus", (9031,), False),
    (r"hen\s+egg[-\s]+white", 9031, 9031, "Gallus gallus", (9031,), False),
    (r"egg[-\s]+white", 9031, 9031, "Gallus gallus", (9031,), False),
    (r"gallus\s+gallus", 9031, 9031, "Gallus gallus", (9031,), False),
    (r"g(?:\.\s*|\s+)gallus", 9031, 9031, "Gallus gallus", (9031,), False),
    (r"homo\s+sapiens", 9606, 9606, "Homo sapiens", (9606,), False),
    (r"h(?:\.\s*|\s+)sapiens", 9606, 9606, "Homo sapiens", (9606,), False),
    (r"mus\s+musculus", 10090, 10090, "Mus musculus", (10090,), False),
    (r"m(?:\.\s*|\s+)musculus", 10090, 10090, "Mus musculus", (10090,), False),
    (r"rattus\s+norvegicus", 10116, 10116, "Rattus norvegicus", (10116,), False),
    (r"r(?:\.\s*|\s+)norvegicus", 10116, 10116, "Rattus norvegicus", (10116,), False),
    (r"bos\s+taurus", 9913, 9913, "Bos taurus", (9913,), False),
    (r"b(?:\.\s*|\s+)taurus", 9913, 9913, "Bos taurus", (9913,), False),
    (r"sus\s+scrofa(?:\s+domesticus)?", 9823, 9823, "Sus scrofa", (9823, 9825), False),
    (r"s(?:\.\s*|\s+)scrofa", 9823, 9823, "Sus scrofa", (9823,), False),
    (r"ovis\s+aries", 9940, 9940, "Ovis aries", (9940,), False),
    (r"o(?:\.\s*|\s+)aries", 9940, 9940, "Ovis aries", (9940,), False),
    (r"capra\s+hircus", 9925, 9925, "Capra hircus", (9925,), False),
    (r"c(?:\.\s*|\s+)hircus", 9925, 9925, "Capra hircus", (9925,), False),
    (r"equus\s+caballus", 9796, 9796, "Equus caballus", (9796,), False),
    (r"e(?:\.\s*|\s+)caballus", 9796, 9796, "Equus caballus", (9796,), False),
    (r"canis\s+lupus(?:\s+familiaris)?", 9615, 9612, "Canis lupus familiaris", (9615, 9612), False),
    (r"canis\s+familiaris", 9615, 9612, "Canis lupus familiaris", (9615, 9612), False),
    (r"c(?:\.\s*|\s+)familiaris", 9615, 9612, "Canis lupus familiaris", (9615, 9612), False),
    (r"felis\s+catus", 9685, 9685, "Felis catus", (9685,), False),
    (r"f(?:\.\s*|\s+)catus", 9685, 9685, "Felis catus", (9685,), False),
    (r"domestic\s+cat", 9685, 9685, "Felis catus", (9685,), False),
    (r"oryctolagus\s+cuniculus", 9986, 9986, "Oryctolagus cuniculus", (9986,), False),
    (r"o(?:\.\s*|\s+)cuniculus", 9986, 9986, "Oryctolagus cuniculus", (9986,), False),
    (r"cavia\s+porcellus", 10141, 10141, "Cavia porcellus", (10141,), False),
    (r"guinea\s+pig", 10141, 10141, "Cavia porcellus", (10141,), False),
    (r"mesocricetus\s+auratus", 10036, 10036, "Mesocricetus auratus", (10036,), False),
    (r"(?:golden|syrian)\s+hamster", 10036, 10036, "Mesocricetus auratus", (10036,), False),
    (r"cricetulus\s+griseus", 10029, 10029, "Cricetulus griseus", (10029,), False),
    (r"chinese\s+hamster", 10029, 10029, "Cricetulus griseus", (10029,), False),
    (r"pan\s+troglodytes", 9598, 9598, "Pan troglodytes", (9598,), False),
    (r"macaca\s+mulatta", 9544, 9544, "Macaca mulatta", (9544,), False),
    (r"physeter\s+(?:macrocephalus|catodon)", 9755, 9755, "Physeter macrocephalus", (9755,), False),
    (r"sperm\s+whale", 9755, 9755, "Physeter macrocephalus", (9755,), False),
    (r"danio\s+rerio", 7955, 7955, "Danio rerio", (7955,), False),
    (r"d(?:\.\s*|\s+)rerio", 7955, 7955, "Danio rerio", (7955,), False),
    (r"drosophila\s+melanogaster", 7227, 7227, "Drosophila melanogaster", (7227,), False),
    (r"d(?:\.\s*|\s+)melanogaster", 7227, 7227, "Drosophila melanogaster", (7227,), False),
    (r"fruit\s*fly", 7227, 7227, "Drosophila melanogaster", (7227,), False),
    (r"caenorhabditis\s+elegans", 6239, 6239, "Caenorhabditis elegans", (6239,), False),
    (r"c(?:\.\s*|\s+)elegans", 6239, 6239, "Caenorhabditis elegans", (6239,), False),
    (r"xenopus\s+laevis", 8355, 8355, "Xenopus laevis", (8355,), False),
    (r"x(?:\.\s*|\s+)laevis", 8355, 8355, "Xenopus laevis", (8355,), False),
    (r"xenopus\s+tropicalis", 8364, 8364, "Xenopus tropicalis", (8364,), False),
    (r"x(?:\.\s*|\s+)tropicalis", 8364, 8364, "Xenopus tropicalis", (8364,), False),
    (r"(?:african\s+)?clawed\s+frog", 8355, 8355, "Xenopus laevis", (8355, 8364), False),
    (r"apis\s+mellifera", 7460, 7460, "Apis mellifera", (7460,), False),
    (r"honey\s*bee", 7460, 7460, "Apis mellifera", (7460,), False),
    (r"bombyx\s+mori", 7091, 7091, "Bombyx mori", (7091,), False),
    (r"silkworm", 7091, 7091, "Bombyx mori", (7091,), True),
    (r"anopheles\s+gambiae", 7165, 7165, "Anopheles gambiae", (7165, 180454), False),
    (r"schistosoma\s+japonicum", 6182, 6182, "Schistosoma japonicum", (6182,), False),
    (r"s(?:\.\s*|\s+)japonicum", 6182, 6182, "Schistosoma japonicum", (6182,), False),
    (r"schistosoma\s+mansoni", 6183, 6183, "Schistosoma mansoni", (6183,), False),
    (r"s(?:\.\s*|\s+)mansoni", 6183, 6183, "Schistosoma mansoni", (6183,), False),
    (r"aequorea\s+victoria", 6100, 6100, "Aequorea victoria", (6100,), False),
    (r"a(?:\.\s*|\s+)victoria", 6100, 6100, "Aequorea victoria", (6100,), False),
    (r"crystal\s+jellyfish", 6100, 6100, "Aequorea victoria", (6100,), False),
    (r"strongylocentrotus\s+purpuratus", 7668, 7668, "Strongylocentrotus purpuratus", (7668,), False),

    # --- Eukaryota: Plants & Green Algae ---
    (r"arabidopsis\s+thaliana", 3702, 3702, "Arabidopsis thaliana", (3702,), False),
    (r"a(?:\.\s*|\s+)thaliana", 3702, 3702, "Arabidopsis thaliana", (3702,), False),
    (r"thale\s+cress", 3702, 3702, "Arabidopsis thaliana", (3702,), False),
    (r"beta\s+vulgaris", 3555, 3555, "Beta vulgaris", (3555, 161934), False),
    (r"vinca\s+minor", 60093, 60093, "Vinca minor", (60093,), False),
    (r"catharanthus\s+roseus", 4058, 4058, "Catharanthus roseus", (4058,), False),
    (r"rosa\s+hybrid(?:\s+cultivar)?", 128735, 128735, "Rosa hybrid cultivar", (128735,), False),
    (r"oryza\s+sativa", 39947, 4530, "Oryza sativa", (39947, 4530, 39946), False),
    (r"o(?:\.\s*|\s+)sativa", 39947, 4530, "Oryza sativa", (39947, 4530, 39946), False),
    (r"zea\s+mays", 4577, 4577, "Zea mays", (4577,), False),
    (r"z(?:\.\s*|\s+)mays", 4577, 4577, "Zea mays", (4577,), False),
    (r"spinacia\s+oleracea", 3562, 3562, "Spinacia oleracea", (3562,), False),
    (r"s(?:\.\s*|\s+)oleracea", 3562, 3562, "Spinacia oleracea", (3562,), False),
    (r"solanum\s+lycopersicum", 4081, 4081, "Solanum lycopersicum", (4081,), False),
    (r"lycopersicon\s+esculentum", 4081, 4081, "Solanum lycopersicum", (4081,), False),
    (r"s(?:\.\s*|\s+)lycopersicum", 4081, 4081, "Solanum lycopersicum", (4081,), False),
    (r"solanum\s+tuberosum", 4113, 4113, "Solanum tuberosum", (4113,), False),
    (r"s(?:\.\s*|\s+)tuberosum", 4113, 4113, "Solanum tuberosum", (4113,), False),
    (r"nicotiana\s+tabacum", 4097, 4097, "Nicotiana tabacum", (4097,), False),
    (r"n(?:\.\s*|\s+)tabacum", 4097, 4097, "Nicotiana tabacum", (4097,), False),
    (r"nicotiana\s+benthamiana", 4100, 4100, "Nicotiana benthamiana", (4100,), False),
    (r"n(?:\.\s*|\s+)benthamiana", 4100, 4100, "Nicotiana benthamiana", (4100,), False),
    (r"glycine\s+max", 3847, 3847, "Glycine max", (3847,), False),
    (r"g\.\s*max", 3847, 3847, "Glycine max", (3847,), False),
    (r"triticum\s+aestivum", 4565, 4565, "Triticum aestivum", (4565,), False),
    (r"t(?:\.\s*|\s+)aestivum", 4565, 4565, "Triticum aestivum", (4565,), False),
    (r"hordeum\s+vulgare", 4513, 4513, "Hordeum vulgare", (4513, 112509), False),
    (r"h(?:\.\s*|\s+)vulgare", 4513, 4513, "Hordeum vulgare", (4513, 112509), False),
    (r"pisum\s+sativum", 3888, 3888, "Pisum sativum", (3888,), False),
    (r"p(?:\.\s*|\s+)sativum", 3888, 3888, "Pisum sativum", (3888,), False),
    (r"garden\s+pea", 3888, 3888, "Pisum sativum", (3888,), False),
    (r"medicago\s+truncatula", 3880, 3880, "Medicago truncatula", (3880,), False),
    (r"m(?:\.\s*|\s+)truncatula", 3880, 3880, "Medicago truncatula", (3880,), False),
    (r"vitis\s+vinifera", 29760, 29760, "Vitis vinifera", (29760,), False),
    (r"brassica\s+napus", 3708, 3708, "Brassica napus", (3708,), False),
    (r"canavalia\s+ensiformis", 3823, 3823, "Canavalia ensiformis", (3823,), False),
    (r"jack\s+bean", 3823, 3823, "Canavalia ensiformis", (3823,), False),
    (r"chlamydomonas\s+reinhardtii", 3055, 3055, "Chlamydomonas reinhardtii", (3055,), False),
    (r"c(?:\.\s*|\s+)reinhardtii", 3055, 3055, "Chlamydomonas reinhardtii", (3055,), False),
    (r"chlamydomonas", 3055, 3055, "Chlamydomonas reinhardtii", (3055,), True),
    (r"physcomitrium\s+patens|physcomitrella\s+patens", 3218, 3218, "Physcomitrium patens", (3218,), False),

    # --- Eukaryota: Fungi ---
    (r"saccharomyces\s+cerevisiae", 559292, 4932, "Saccharomyces cerevisiae", (559292, 4932), False),
    (r"s(?:\.\s*|\s+)cerevisiae", 559292, 4932, "Saccharomyces cerevisiae", (559292, 4932), False),
    (r"(?:baker'?s|budding|brewer'?s)\s+yeast", 559292, 4932, "Saccharomyces cerevisiae", (559292, 4932), False),
    (r"schizosaccharomyces\s+pombe", 284812, 4896, "Schizosaccharomyces pombe", (284812, 4896), False),
    (r"s(?:\.\s*|\s+)pombe", 284812, 4896, "Schizosaccharomyces pombe", (284812, 4896), False),
    (r"fission\s+yeast", 284812, 4896, "Schizosaccharomyces pombe", (284812, 4896), False),
    (r"neurospora\s+crassa", 367110, 5141, "Neurospora crassa", (367110, 5141), False),
    (r"n(?:\.\s*|\s+)crassa", 367110, 5141, "Neurospora crassa", (367110, 5141), False),
    (r"candida\s+albicans", 237561, 5476, "Candida albicans", (237561, 5476), False),
    (r"c(?:\.\s*|\s+)albicans", 237561, 5476, "Candida albicans", (237561, 5476), False),
    (r"(?:aspergillus|emericella)\s+nidulans", 227321, 162425, "Aspergillus nidulans", (227321, 162425), False),
    (r"a(?:\.\s*|\s+)nidulans", 227321, 162425, "Aspergillus nidulans", (227321, 162425), False),
    (r"aspergillus\s+niger", 425011, 5061, "Aspergillus niger", (425011, 5061), False),
    (r"a\.\s*niger", 425011, 5061, "Aspergillus niger", (425011, 5061), False),
    (r"aspergillus\s+fumigatus", 330879, 746128, "Aspergillus fumigatus", (330879, 746128, 5085), False),
    (r"a(?:\.\s*|\s+)fumigatus", 330879, 746128, "Aspergillus fumigatus", (330879, 746128, 5085), False),
    (r"(?:komagataella\s+phaffii|pichia\s+pastoris)", 644223, 4922, "Komagataella phaffii", (644223, 4922, 460519), False),
    (r"p(?:\.\s*|\s+)pastoris", 644223, 4922, "Komagataella phaffii", (644223, 4922, 460519), False),
    (r"k(?:\.\s*|\s+)phaffii", 644223, 4922, "Komagataella phaffii", (644223, 4922, 460519), False),
    (r"yarrowia\s+lipolytica", 284591, 4952, "Yarrowia lipolytica", (284591, 4952), False),
    (r"y(?:\.\s*|\s+)lipolytica", 284591, 4952, "Yarrowia lipolytica", (284591, 4952), False),
    (r"cryptococcus\s+neoformans", 214684, 5207, "Cryptococcus neoformans", (214684, 5207), False),
    (r"c(?:\.\s*|\s+)neoformans", 214684, 5207, "Cryptococcus neoformans", (214684, 5207), False),
    (r"ustilago\s+maydis", 237631, 5270, "Ustilago maydis", (237631, 5270), False),

    # --- Eukaryota: Protists ---
    (r"plasmodium\s+falciparum", 36329, 5833, "Plasmodium falciparum", (36329, 5833), False),
    (r"p(?:\.\s*|\s+)falciparum", 36329, 5833, "Plasmodium falciparum", (36329, 5833), False),
    (r"malaria(?:\s+parasite)?", 36329, 5833, "Plasmodium falciparum", (36329, 5833), True),
    (r"toxoplasma\s+gondii", 508771, 5811, "Toxoplasma gondii", (508771, 5811), False),
    (r"t(?:\.\s*|\s+)gondii", 508771, 5811, "Toxoplasma gondii", (508771, 5811), False),
    (r"trypanosoma\s+brucei", 185431, 5691, "Trypanosoma brucei", (185431, 5691, 5690, 31285, 31286, 679716), False),
    (r"t(?:\.\s*|\s+)brucei", 185431, 5691, "Trypanosoma brucei", (185431, 5691, 5690), False),
    (r"trypanosoma\s+cruzi", 353153, 5693, "Trypanosoma cruzi", (353153, 5693), False),
    (r"t(?:\.\s*|\s+)cruzi", 353153, 5693, "Trypanosoma cruzi", (353153, 5693), False),
    (r"trypanosoma", 185431, 5690, "Trypanosoma brucei", (185431, 5691, 353153, 5693, 5690), True),
    (r"leishmania\s+major", 347515, 5664, "Leishmania major", (347515, 5664), False),
    (r"l\.\s*major", 347515, 5664, "Leishmania major", (347515, 5664), False),
    (r"dictyostelium\s+discoideum", 44689, 44689, "Dictyostelium discoideum", (44689, 352472), False),
    (r"d(?:\.\s*|\s+)discoideum", 44689, 44689, "Dictyostelium discoideum", (44689, 352472), False),
    (r"dictyostelium", 44689, 44689, "Dictyostelium discoideum", (44689, 352472), True),
    (r"slime\s+mold", 44689, 44689, "Dictyostelium discoideum", (44689, 352472), False),
    (r"tetrahymena\s+thermophila", 312017, 5911, "Tetrahymena thermophila", (312017, 5911), False),
    (r"t(?:\.\s*|\s+)thermophila", 312017, 5911, "Tetrahymena thermophila", (312017, 5911), False),
    (r"tetrahymena", 312017, 5911, "Tetrahymena thermophila", (312017, 5911), True),
    (r"paramecium\s+tetraurelia", 5888, 5888, "Paramecium tetraurelia", (5888, 412030), False),
    (r"phaeodactylum\s+tricornutum", 556484, 2850, "Phaeodactylum tricornutum", (556484, 2850), False),
    (r"thalassiosira\s+pseudonana", 296543, 35128, "Thalassiosira pseudonana", (296543, 35128), False),
    (r"giardia\s+(?:intestinalis|lamblia)", 184922, 5741, "Giardia intestinalis", (184922, 5741), False),
    (r"entamoeba\s+histolytica", 294381, 5759, "Entamoeba histolytica", (294381, 5759), False),

    # --- Single-Word Common Host Names, Bare Genera & Lab Shorthands (Evaluated AFTER Full Viral & Binomial Names!) ---
    (r"human", 9606, 9606, "Homo sapiens", (9606,), True),
    (r"mouse", 10090, 10090, "Mus musculus", (10090,), True),
    (r"murine", 10090, 10090, "Mus musculus", (10090,), True),
    (r"rat", 10116, 10116, "Rattus norvegicus", (10116,), True),
    (r"chicken", 9031, 9031, "Gallus gallus", (9031,), True),
    (r"hen", 9031, 9031, "Gallus gallus", (9031,), True),
    (r"bovine", 9913, 9913, "Bos taurus", (9913,), True),
    (r"cow", 9913, 9913, "Bos taurus", (9913,), True),
    (r"cattle", 9913, 9913, "Bos taurus", (9913,), True),
    (r"porcine", 9823, 9823, "Sus scrofa", (9823,), True),
    (r"pig", 9823, 9823, "Sus scrofa", (9823,), True),
    (r"swine", 9823, 9823, "Sus scrofa", (9823,), True),
    (r"ovine", 9940, 9940, "Ovis aries", (9940,), True),
    (r"sheep", 9940, 9940, "Ovis aries", (9940,), True),
    (r"caprine", 9925, 9925, "Capra hircus", (9925,), True),
    (r"goat", 9925, 9925, "Capra hircus", (9925,), True),
    (r"equine", 9796, 9796, "Equus caballus", (9796,), True),
    (r"horse", 9796, 9796, "Equus caballus", (9796,), True),
    (r"canine", 9615, 9612, "Canis lupus familiaris", (9615, 9612), True),
    (r"dog", 9615, 9612, "Canis lupus familiaris", (9615, 9612), True),
    (r"feline", 9685, 9685, "Felis catus", (9685,), True),
    (r"cat", 9685, 9685, "Felis catus", (9685,), True),
    (r"rabbit", 9986, 9986, "Oryctolagus cuniculus", (9986,), True),
    (r"chimpanzee", 9598, 9598, "Pan troglodytes", (9598,), True),
    (r"whale", 9755, 9755, "Physeter macrocephalus", (9755,), True),
    (r"zebrafish", 7955, 7955, "Danio rerio", (7955,), True),
    (r"drosophila", 7227, 7227, "Drosophila melanogaster", (7227,), True),
    (r"roundworm", 6239, 6239, "Caenorhabditis elegans", (6239,), True),
    (r"nematode", 6239, 6239, "Caenorhabditis elegans", (6239,), True),
    (r"xenopus", 8355, 8355, "Xenopus laevis", (8355, 8364), True),
    (r"schistosoma", 6182, 6182, "Schistosoma japonicum", (6182, 6183), True),
    (r"jellyfish", 6100, 6100, "Aequorea victoria", (6100,), True),
    (r"arabidopsis", 3702, 3702, "Arabidopsis thaliana", (3702,), True),
    (r"rice", 39947, 4530, "Oryza sativa", (39947, 4530, 39946), True),
    (r"maize", 4577, 4577, "Zea mays", (4577,), True),
    (r"corn", 4577, 4577, "Zea mays", (4577,), True),
    (r"spinach", 3562, 3562, "Spinacia oleracea", (3562,), True),
    (r"tomato", 4081, 4081, "Solanum lycopersicum", (4081,), True),
    (r"potato", 4113, 4113, "Solanum tuberosum", (4113,), True),
    (r"tobacco", 4097, 4097, "Nicotiana tabacum", (4097,), True),
    (r"soybean", 3847, 3847, "Glycine max", (3847,), True),
    (r"wheat", 4565, 4565, "Triticum aestivum", (4565,), True),
    (r"barley", 4513, 4513, "Hordeum vulgare", (4513, 112509), True),
    (r"pea", 3888, 3888, "Pisum sativum", (3888,), True),
    (r"yeast", 559292, 4932, "Saccharomyces cerevisiae", (559292, 4932), True),
    (r"taq", 271, 271, "Thermus aquaticus", (271, 65393, 498848), True),
    (r"pfu", 186497, 2261, "Pyrococcus furiosus", (186497, 2261), True),
    (r"kod", 69014, 311400, "Thermococcus kodakarensis", (69014, 311400), True),
    (r"staph", 93061, 1280, "Staphylococcus aureus", (93061, 1280, 158879), True),
    (r"tb", 83332, 1773, "Mycobacterium tuberculosis", (83332, 1773), True),
    (r"t4", 10665, 10665, "Enterobacteria phage T4", (10665, 2681598), True),
    (r"t7", 10760, 10760, "Enterobacteria phage T7", (10760,), True),
    (r"tmv", 12242, 12242, "Tobacco mosaic virus", (12242,), True),
    (r"sv40", 1891767, 1891767, "Macaca mulatta polyomavirus 1", (1891767, 10633), True),
]

# Backwards-compatible 3-tuple alias table
ORGANISM_TAXON_ALIASES: List[Tuple[str, Optional[int], str]] = [
    (pat, prim_id, sci_name) for pat, prim_id, _, sci_name, _, _ in _ORGANISM_CATALOG if prim_id is not None
]

# Pre-compiled catalog regexes for fast scanning across 10,000+ prompts
_COMPILED_ORGANISM_CATALOG = [
    (
        re.compile(r'(?<![A-Za-z0-9.-])' + pat + r'(?![A-Za-z0-9-])', flags=re.IGNORECASE),
        prim_id,
        clade_id,
        sci_name,
        extra_ids,
        ctx_guard,
    )
    for pat, prim_id, clade_id, sci_name, extra_ids, ctx_guard in _ORGANISM_CATALOG
]

_NON_PROTEIN_STOPWORDS = {
    "protein", "polypeptide", "chain", "subunit", "recombinant", "native", "mature",
    "wild-type", "wildtype", "wt", "full-length", "full", "length", "canonical",
    "purified", "expressed", "in", "from", "of", "the", "a", "an", "and", "or",
    "with", "without", "tag", "tags", "tagged", "untagged", "tagless", "tag-free",
    "n-terminal", "c-terminal", "n-term", "c-term", "n-terminally", "c-terminally",
    "terminal", "terminus", "site", "cleavage", "fusion", "fused", "to", "at",
    "carrying", "having", "bearing", "containing", "plus"
}

# Strict guard lists preventing protein words from EVER being mistaken for a Latin "Genus species" binomial
_NON_GENUS_PROTEIN_WORDS = {
    "protein", "proteins", "uncharacterized", "putative", "probable", "possible", "predicted",
    "hypothetical", "conserved", "bifunctional", "multifunctional", "trifunctional",
    "factor", "coagulation", "complement", "interleukin", "interferon", "tumor", "transforming",
    "epidermal", "fibroblast", "vascular", "platelet", "insulin", "glucagon", "growth", "colony",
    "chemokine", "cytokine", "nuclear", "cellular", "cytoplasmic", "cytosolic", "mitochondrial",
    "chloroplastic", "peroxisomal", "endoplasmic", "golgi", "lysosomal", "vacuolar", "extracellular",
    "periplasmic", "membrane", "transmembrane", "integral", "peripheral", "outer", "inner", "surface",
    "coat", "capsid", "envelope", "matrix", "spike", "tegument", "nonstructural", "structural",
    "major", "minor", "small", "large", "general", "specific", "transcriptional", "translational",
    "ribosomal", "proteasomal", "spliceosomal", "chromosomal", "centromeric", "telomeric",
    "flagellar", "ciliary", "bacterial", "archaeal", "viral", "eukaryotic", "fungal", "mammalian",
    "green", "red", "blue", "yellow", "fluorescent", "luminescent", "heat", "cold", "acid", "alkaline",
    "neutral", "basic", "acidic", "soluble", "insoluble", "serum", "plasma", "blood", "liver", "muscle",
    "brain", "heart", "kidney", "pancreatic", "gastric", "intestinal", "salivary", "milk", "egg", "venom",
    "seed", "leaf", "root", "pollen", "cell", "photosystem", "reaction", "electron", "proton",
    "sodium", "potassium", "calcium", "magnesium", "iron", "zinc", "copper", "manganese", "cobalt",
    "nickel", "molybdenum", "tungsten", "vanadium", "selenium", "chloride", "phosphate", "sulfate",
    "nitrate", "nitrite", "ammonium", "carbonic", "formate", "acetate", "propionate", "butyrate",
    "pyruvate", "lactate", "malate", "fumarate", "succinate", "citrate", "isocitrate", "aconitate",
    "oxaloacetate", "oxoglutarate", "ketoglutarate", "glyoxylate", "glycolate", "glycerate",
    "phosphoglycerate", "bisphosphoglycerate", "triosephosphate", "glucose", "fructose", "galactose",
    "mannose", "ribose", "xylose", "arabinose", "erythrose", "sedoheptulose", "glucosamine", "muramic",
    "sialic", "glucuronate", "gluconate", "shikimate", "chorismate", "prephenate", "anthranilate",
    "indole", "imidazole", "histidinol", "homoserine", "cystathionine", "ornithine", "citrulline",
    "argininosuccinate", "carbamoyl", "dihydroorotate", "orotate", "inosine", "xanthosine",
    "adenosine", "guanosine", "cytidine", "uridine", "thymidine", "adenylate", "guanylate",
    "cytidylate", "uridylate", "thymidylate", "ribonucleoside", "deoxyribonucleoside",
    "glutathione", "thioredoxin", "glutaredoxin", "peroxiredoxin", "ferredoxin", "flavodoxin",
    "rubredoxin", "plastocyanin", "azurin", "cytochrome", "ubiquinol", "plastoquinol", "menaquinol",
    "acyl", "acetyl", "malonyl", "succinyl", "propionyl", "palmitoyl", "myristoyl", "farnesyl",
    "mevalonate", "squalene", "lanosterol", "cholesterol", "ceramide", "peptidoglycan", "cellulose",
    "chitin", "xylan", "pectin", "starch", "glycogen", "trehalose", "sucrose", "maltose", "lactose",
    "ubiquitin", "actin", "tubulin", "myosin", "kinesin", "dynein", "spectrin", "ankyrin", "collagen",
    "keratin", "lamin", "vimentin", "histone", "nucleosome", "chromatin", "ribosome", "proteasome",
    "signal", "nascent", "chaperone", "chaperonin", "alpha", "beta", "gamma", "delta", "epsilon",
    "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
    "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega", "chain", "subunit", "type", "class",
    "group", "family", "subfamily", "superfamily", "domain", "member", "homolog", "ortholog", "paralog",
    "inhibitor", "receptor", "transporter", "permease", "channel", "carrier", "regulator", "repressor",
    "activator", "enhancer", "suppressor", "mediator", "sensor", "adaptor", "scaffold",
    "synthase", "synthetase", "transferase", "kinase", "phosphatase", "dehydrogenase", "reductase",
    "oxidase", "oxygenase", "hydrolase", "protease", "proteinase", "peptidase", "esterase", "lipase",
    "nuclease", "endonuclease", "exonuclease", "ribonuclease", "polymerase", "helicase", "gyrase",
    "topoisomerase", "primase", "ligase", "isomerase", "mutase", "lyase", "decarboxylase", "carboxylase",
    "deaminase", "aminotransferase", "transaminase", "transketolase", "transaldolase", "aldolase",
    "enolase", "catalase", "peroxidase", "dismutase", "anhydrase", "lysozyme", "myoglobin",
    "hemoglobin", "albumin", "globulin", "prothrombin", "thrombin", "trypsin", "chymotrypsin",
    "subtilisin", "enterokinase", "enteropeptidase", "streptavidin", "avidin", "luciferase",
    "rhodopsin", "bacteriorhodopsin", "halorhodopsin", "flagellin", "pilin", "porin", "calmodulin",
    "troponin", "tropomyosin", "integrin", "cadherin", "selectin", "aquaporin", "cyclin", "sirtuin",
    "caspase", "cathepsin", "lectin", "hemagglutinin", "neuraminidase", "endolysin", "holin",
    "terminase", "integrase", "transposase", "recombinase", "methyltransferase", "acetyltransferase",
    "atp", "gtp", "ctp", "utp", "adp", "amp", "nad", "nadh", "nadp", "nadph", "fad", "fmn", "coa",
    "dna", "rna", "mrna", "trna", "rrna", "abc", "aaa", "recombinant", "native", "mature", "canonical",
    "purified", "phage", "virus", "initiation", "elongation", "release", "peptide", "polypeptide",
    "aminoacyl", "histidyl", "alanyl", "arginyl", "asparaginyl", "aspartyl", "cysteinyl", "glutaminyl",
    "glutamyl", "glycyl", "isoleucyl", "leucyl", "lysyl", "methionyl", "phenylalanyl", "prolyl",
    "seryl", "threonyl", "tryptophanyl", "tyrosyl", "valyl", "shikimate", "chorismate", "tryptophan",
    "histidine", "lysine", "arginine", "leucine", "isoleucine", "valine", "methionine", "threonine",
    "serine", "glycine", "alanine", "proline", "glutamine", "glutamate", "asparagine", "aspartate",
    "cysteine", "tyrosine", "phenylalanine", "fatty", "amino", "nucleoside", "nucleotide",
    "enzyme", "enzymes", "malolactic", "disintegrin", "metalloproteinase", "polyprotein", "replicase",
    "myc", "ribulose", "allene", "auxin", "floral", "thylakoid", "mechanosensitive", "acyclic",
    "dual", "ciliated", "neuron", "neuronal", "zygote", "zygotic", "thyroid", "nitric",
    "photoreceptor", "paired", "bone", "uveal", "malignant", "glycerophosphocholine", "meiosis",
    "meiotic", "steroid", "hematopoietic", "active", "telomere", "proto", "oncogene",
    "proto-oncogene", "bisphosphate", "oxide", "efflux", "influx", "homeotic", "homeobox",
    "lumenal", "luminal", "ion", "ionic", "specificity", "left", "right", "navigator", "arrest",
    "hormone", "cilium", "cilia", "mesoderm", "ectoderm", "endoderm", "marrow", "autoantigen",
    "fibrous", "histiocytoma", "choline", "initiator", "prostaglandin", "breakpoint", "length",
    "proteoglycan", "metallopeptidase", "phosphodiesterase", "sesquiterpene", "monoterpene",
    "diterpene", "triterpene", "cyclase", "sequence", "amplified", "organizer",
}

_NON_SPECIES_PROTEIN_WORDS = _NON_GENUS_PROTEIN_WORDS | {
    "and", "or", "the", "with", "without", "from", "into", "onto", "over", "under", "between",
    "like", "related", "binding", "bound", "associated", "interacting", "regulated", "regulating",
    "regulatory", "induced", "inducible", "activated", "activatable", "activating", "inhibited",
    "inhibiting", "inhibitory", "dependent", "independent", "containing", "rich", "deficient",
    "motif", "repeat", "repeats", "box", "finger", "zipper", "cluster", "complex", "system",
    "component", "assembly", "maturation", "biosynthesis", "synthesis", "metabolism", "catabolism",
    "degradation", "transport", "import", "export", "secretion", "translocation", "transcription",
    "translation", "replication", "recombination", "repair", "division", "septation", "sporulation",
    "chemotaxis", "motility", "virulence", "resistance", "tolerance", "response", "shock", "stress",
    "damage", "signaling", "pathway", "cycle", "apoptosis", "autophagy", "fusion", "fission",
    "splicing", "processing", "modification", "methylation", "acetylation", "phosphorylation",
    "ubiquitination", "glycosylation", "oxidation", "reduction", "transfer", "cleavage", "hydrolysis",
    "folding", "refolding", "base", "ester", "aldehyde", "ketone", "alcohol", "amine", "amide",
    "quinone", "disulfide", "thiol", "sulfide", "ammonia", "urea", "pyrophosphate", "diphosphate",
    "triphosphate", "monophosphate", "purine", "pyrimidine", "sugar", "hexose", "pentose", "lipid",
    "sterol", "heme", "flavin", "biotin", "thiamine", "riboflavin", "niacin", "folate", "cobalamin",
    "antigen", "antibody", "immunoglobulin", "toxin", "antitoxin", "defensin", "inclusion",
    "accessory", "auxiliary", "essential", "heavy", "light", "short", "long", "high", "low",
    "early", "late", "immediate", "number", "isoform", "isozyme", "precursor", "proprotein",
    "polyprotein", "glycoprotein", "lipoprotein", "phosphoprotein", "metalloprotein", "flavoprotein",
    "hemoprotein", "nucleoprotein", "tag", "tagged", "tags", "site", "terminal", "terminus",
    "coiled", "coil", "single", "double", "stranded", "water", "oxygen", "evolving", "harvesting",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xa", "xi", "xii", "xiii",
}

_VIRAL_POST_GUARD_WORDS = {
    "receptor", "receptors", "binding", "induced", "inducible", "interacting",
    "associated", "regulated", "inhibitor", "resistance", "susceptibility", "host",
}

_GST_ISOZYME_DESIGNATORS = {
    "p", "p1", "a1", "a2", "a3", "a4", "a5", "m1", "m2", "m3", "m4", "m5",
    "t1", "t2", "t2b", "k1", "o1", "o2", "z1", "s1",
    "u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8", "u9", "u10", "u19", "u20",
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "l1", "l2", "l3",
    "pi", "alpha", "mu", "theta", "kappa", "omega", "zeta", "sigma", "tau", "phi", "lambda",
}

# Biological words that follow an ambiguous prefix ("His-", "HA-", "GST-", "MBP-", "Myc-", "Strep-", "FLAG-")
# in real Swiss-Prot protein names and must NEVER be split as an N-terminal construct tag
_AMBIGUOUS_PREFIX_BIO_BLOCKLIST = {
    "like", "related", "binding", "bound", "associated", "interacting", "regulated", "regulating",
    "regulatory", "induced", "inducible", "activated", "activatable", "activating", "inhibited",
    "inhibiting", "inhibitor", "dependent", "independent", "specific", "containing", "rich", "poor",
    "deficient", "domain", "domains", "motif", "family", "superfamily", "class", "type", "subtype",
    "isoform", "isozyme", "homolog", "ortholog", "paralog", "subunit", "chain", "receptor",
    "transporter", "permease", "channel", "synthase", "synthetase", "ligase", "transferase",
    "kinase", "phosphatase", "protease", "proteinase", "peptidase", "esterase", "reductase",
    "oxidase", "dehydrogenase", "hydrolase", "isomerase", "mutase", "lyase", "atpase", "gtpase",
    "trna", "rrna", "mrna", "dna", "rna", "box", "finger", "zipper", "repeat", "repeats", "fold",
    "barrel", "helix", "sheet", "loop", "cluster", "complex", "system", "pathway", "operon",
    "regulon", "factor", "cofactor", "partner", "target", "substrate", "product", "precursor",
    "propeptide", "peptide", "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon",
    "phi", "chi", "psi", "omega",
}

# Short peptide / chemical / synthetic stems that cannot themselves be the target protein in "<STEM>-<TAG>"
_EXCLUDED_SUFFIX_STEMS = {
    "twin", "anti", "non", "6xhis", "8xhis", "10xhis", "his6", "his8", "his10",
    "hexahis", "octahis", "decahis", "polyhistidine", "his", "twin-strep",
    "strep", "flag", "ha", "tev", "3c", "hrv3c", "prescission",
    "trna", "rrna", "mrna", "phospho", "methyl", "acetyl", "formyl", "peptidyl",
    "aminoacyl", "poly", "oligo", "mono", "di", "tri", "tetra", "penta", "hexa",
    "hepta", "octa", "deca", "pre", "pro", "sub", "super", "neo", "iso", "pseudo",
    "proto", "endo", "exo", "trans", "cis", "d", "l", "n", "o", "s",
}

# Pure synthetic/epitope tag words that cannot be the target protein remainder in "<PREFIX>-<REST>"
_PURE_TAG_REST_TOKENS = {
    "tag", "tags", "tagged", "6xhis", "8xhis", "10xhis", "his6", "his8", "his10",
    "hexahis", "octahis", "decahis", "polyhistidine", "his", "his-tag",
    "twin-strep", "twin-strep-tag", "strep", "strep-tag", "flag", "flag-tag", "ha-tag",
    "myc-tag", "c-myc-tag", "gst-tag", "mbp-tag",
}


def _find_balanced_close(text: str, open_idx: int, open_ch: str = "(", close_ch: str = ")") -> int:
    """Returns index of matching closing delimiter (supporting nested delimiters), or -1 if unbalanced."""
    depth = 0
    for idx in range(open_idx, len(text)):
        ch = text[idx]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return idx
    return -1


_STRAIN_PAREN_PREFIX_RX = re.compile(
    r'^(?:strain|str\.|isolate|serotype|serovar|biovar|pathovar|subsp\.|var\.|ecotype|'
    r'group|clade|genotype|subtype|clone|breed|cultivar|cv\.|line|'
    r'ATCC|DSM|NCTC|PCC|CCAP|CBS|NBRC|JCM|NCIMB|VKM|UTEX|SAG|[ABC]/[A-Za-z0-9])',
    flags=re.IGNORECASE
)

_UNPAREN_STRAIN_TAIL_RX = re.compile(
    r'^(?:'
    r'\s+f\.\s*sp\.\s+[A-Za-z0-9_.-]+'
    r'|\s+subsp\.\s+(?-i:[A-Z][a-z]{2,}\s+[a-z]{3,})'
    r'|\s+(?:subsp\.|var\.|pv\.|bv\.|forma|f\.|serogroup|serovar|biovar|biotype|serotype|pathovar|group|subtype|genotype|clade)\s+[A-Za-z0-9_./:-]+'
    r'(?:\s*(?:/|\s+)\s*(?:serogroup|serovar|biovar|biotype|serotype|pathovar|group|subtype|genotype|clade)\s+[A-Za-z0-9_./:-]+)*'
    r'|\s+(?:brucei|gambiense|rhodesiense)\b'
    r'|\s+(?:strain|str\.|isolate)\s+[A-Za-z0-9_./:-]+'
    r'|\s+(?:PCC|ATCC|DSM|NCTC|NCIMB|ORS|JCM|NBRC|CBS|CCAP|KCTC|NRRL|VKM|GDMCC)\s*[0-9][A-Za-z0-9_./-]*'
    r'|\s+O[0-9]+[A-Za-z0-9:_-]*'
    r'|\s+mcd\s+[0-9]+'
    r'|\s+(?:K-?12|BL21(?:\(DE3\))?|H37Rv)\b'
    r'|\s+(?-i:(?:[A-Z]{2}-[0-9]+|PT[0-9]+))\b'
    r'|(?<=paratyphi)\s+(?-i:[A-C])(?![A-Za-z0-9\'-])'
    r'|(?<=albus)\s+(?-i:G)(?![A-Za-z0-9\'-])'
    r'|\s+(?-i:[A-Z])(?![A-Za-z0-9\'-])(?=\s*(?:[\)\]]|$))'
    r')+',
    flags=re.IGNORECASE
)

_SP_STRAIN_TAIL_RX = re.compile(
    r'^\s+(?:'
    r'(?i:PCC|ATCC|DSM|NCTC|NCIMB|ORS|JCM|NBRC|CBS|CCAP|KCTC|NRRL|VKM|GDMCC|CGMCC|UTEX|SAG|LMG|CIP|BCRC|IAM|IFO|FERM|MTCC)\s*[0-9][A-Za-z0-9_./-]*'
    r'|(?!(?i:16S|23S|30S|50S|40S|60S|5S)\b)(?![0-9]+-[A-Za-z]{3,}\b)(?=[A-Za-z0-9_./-]*[0-9])[A-Za-z0-9][A-Za-z0-9_./-]{0,16}(?:\s+Bresler)?(?![A-Za-z0-9_.-])'
    r'|(?!(?:ATP|GTP|CTP|UTP|AMP|DNA|RNA|MRNA|TRNA|RRNA|NAD|NADH|NADP|NADPH|FAD|FMN|COA|ABC|AAA)\b)[A-Z]{2,4}(?![A-Za-z0-9_.-])(?!\s+\((?!(?i:strain|str\.|isolate)))'
    r'|(?!(?:His|Tyr|Trp|Phe|Leu|Ile|Val|Met|Pro|Ala|Gly|Ser|Thr|Cys|Asn|Gln|Asp|Glu|Lys|Arg|Sec|Pyl)\b)[A-Z][a-z]{1,2}(?![A-Za-z0-9_.-])(?!\s+\((?!(?i:strain|str\.|isolate)))'
    r')'
)


def _is_inside_parentheses(text: str, pos: int) -> bool:
    """Returns True if index pos sits strictly inside an unclosed '(' or '[' delimiter."""
    p_depth = 0
    b_depth = 0
    for i in range(pos):
        c = text[i]
        if c == "(":
            p_depth += 1
        elif c == ")" and p_depth > 0:
            p_depth -= 1
        elif c == "[":
            b_depth += 1
        elif c == "]" and b_depth > 0:
            b_depth -= 1
    return p_depth > 0 or b_depth > 0


def _is_enclosed_whole_paren(text: str, s_idx: int, e_idx: int) -> bool:
    """Returns True if text[s_idx:e_idx] is the entire trimmed contents of its immediately enclosing (...) or [...]."""
    left = text[:s_idx].rstrip()
    right = text[e_idx:].lstrip()
    return (left.endswith("(") and right.startswith(")")) or (left.endswith("[") and right.startswith("]"))


def _consume_organism_qualifiers(text: str, end_pos: int, matched_prim_taxid: Optional[int] = None) -> Tuple[int, Optional[str]]:
    """
    Advances end_pos past any attached taxonomic subspecies/strain/isolate qualifiers
    (including unparenthesized 'Genus sp. <STRAIN>' codes, bridging short serological/biotype tokens up to
    '(strain ...)' / '(isolate ...)', and nested parens).
    Returns (new_end_pos, extracted_strain_text).
    """
    pos = end_pos
    strain_parts: List[str] = []

    if re.search(r'(?<![A-Za-z0-9-])spp?\.?$', text[:pos], flags=re.IGNORECASE):
        m_sp_str = _SP_STRAIN_TAIL_RX.match(text[pos:])
        if m_sp_str:
            strain_parts.append(m_sp_str.group(0).strip())
            pos += m_sp_str.end()

    m_unp = _UNPAREN_STRAIN_TAIL_RX.match(text[pos:])
    if m_unp:
        strain_parts.append(m_unp.group(0).strip())
        pos += m_unp.end()

    # Bridge any short remaining taxonomic gap (e.g. 'PT4', 'A', 'JF-1') directly before '(strain ...)' or '(isolate ...)'
    m_bridge = re.match(
        r'^(\s+[A-Za-z0-9_./:-]{1,35}?)\s*(\((?:strain|str\.|isolate|serotype|serovar|biovar|subsp\.)\b)',
        text[pos:],
        flags=re.IGNORECASE
    )
    if m_bridge:
        gap_txt = m_bridge.group(1).strip()
        if not re.search(r'\b(?:with|tag|tags|tagged|site|cleavage|from|in|of)\b', gap_txt, flags=re.IGNORECASE):
            strain_parts.append(gap_txt)
            pos += len(m_bridge.group(1))

    while pos < len(text):
        ws_match = re.match(r'\s*\(', text[pos:])
        if not ws_match:
            break
        open_idx = pos + ws_match.end() - 1
        close_idx = _find_balanced_close(text, open_idx, "(", ")")
        if close_idx == -1:
            break
        inner = text[open_idx + 1:close_idx].strip()
        is_tax_qual = bool(_STRAIN_PAREN_PREFIX_RX.match(inner))
        if not is_tax_qual and matched_prim_taxid is not None:
            for rx, p_id, _, _, e_ids, _ in _COMPILED_ORGANISM_CATALOG:
                if rx.fullmatch(inner) and (p_id == matched_prim_taxid or matched_prim_taxid in e_ids):
                    is_tax_qual = True
                    break
        if is_tax_qual:
            strain_parts.append(inner)
            pos = close_idx + 1
        else:
            break

    strain_str = " ; ".join(strain_parts) if strain_parts else None
    return pos, strain_str


def _is_valid_latin_binomial_tokens(genus_tok: str, species_tok: str) -> bool:
    """Strict morphological & biochemical guard validating a general 'Genus species' or 'Genus sp[.]' pair."""
    g_clean = genus_tok.strip("[]")
    if not re.fullmatch(r'[A-Z][a-z]{2,}', g_clean):
        return False
    g_low = g_clean.lower()
    if g_low in _NON_GENUS_PROTEIN_WORDS:
        return False
    if g_low.endswith(("ase", "ases", "ing", "tion", "sion", "ment", "protein")):
        return False

    if species_tok in ("sp.", "sp"):
        return True
    if not re.fullmatch(r'[a-z]{2,}(?:-[a-z]{2,})?', species_tok):
        return False
    if "-" not in species_tok and len(species_tok) < 3:
        return False
    s_low = species_tok.lower()
    if s_low in _NON_SPECIES_PROTEIN_WORDS:
        return False
    for part in s_low.split("-"):
        if part in _NON_SPECIES_PROTEIN_WORDS or part.endswith(("ase", "ases", "ing", "tion", "sion", "ment", "phage", "virus")):
            return False
    return True


def _has_raw_protein_words(rem_text: str) -> bool:
    """Checks if rem_text retains at least one non-stopword protein/gene token (including 'Protein A', single-letter & phage integer genes)."""
    cleaned = re.sub(r'[\(\)\[\],;:"\']+', ' ', rem_text).strip()
    tokens = [t.strip(".-") for t in cleaned.split() if t.strip(".-")]
    meaningful = [
        t for t in tokens
        if (t == "A" or t.lower() not in _NON_PROTEIN_STOPWORDS)
        and (not t.isdigit() or 1 <= int(t) <= 999)
    ]
    return len(meaningful) > 0


def _clean_span_removal(text: str, span_start: int, span_end: int) -> str:
    """
    Removes text[span_start:span_end] along with any directly attached organism preposition
    ('from ', 'in ', 'of ') or enclosing '(<org>)' / '[<org>]' brackets.
    """
    s, e = span_start, span_end
    before_ws = text[:s].rstrip()
    after_ws = text[e:].lstrip()
    if before_ws.endswith("(") and after_ws.startswith(")"):
        s = len(before_ws) - 1
        e = len(text) - len(after_ws) + 1
    elif before_ws.endswith("[") and after_ws.startswith("]"):
        s = len(before_ws) - 1
        e = len(text) - len(after_ws) + 1

    prefix = text[:s]
    m_prep = re.search(r'(?:,\s*)?(?<![A-Za-z0-9-])(?:from|in|of)\s+$', prefix, flags=re.IGNORECASE)
    if m_prep:
        s = m_prep.start()

    combined = (text[:s] + " " + text[e:]).strip(" ,;-")
    return " ".join(combined.split())


# Matches a viral or bacteriophage taxon starting at position 0 of a substring without eating protein/gene tokens
_VIRAL_AT_POS_RX = re.compile(
    r'^(?:'
    r'(?P<vlp>[A-Z][a-z]{2,}\s+virus-like\s+particle\s+[A-Z0-9_-]+)\b'
    r'|(?P<vname>[A-Z][A-Za-z0-9\'-]*(?:\s+[A-Za-z0-9\'-]+){0,6}\s+virus|[A-Z][A-Za-z0-9\'-]*(?:\s+[A-Za-z0-9\'-]+){0,5}[a-z]{2,}virus)(?![A-Za-z0-9-])'
    r'(?P<vtail>(?:'
    r'\s+(?:type|genotype|subtype|group|serotype)\s+(?![A-Z][a-z]{2,}\b)[A-Za-z0-9_./-]+(?![A-Za-z0-9_./-])'
    r'|\s+Newbury\s+agent-[0-9]+\b'
    r'|\s+agent-[0-9]+\b'
    r'|\s+(?![A-Z][a-z]{2,}\b)(?!RNA[0-9]+\b)(?![A-Za-z0-9_.-]+-[a-z]{3,}\b)[A-Z0-9][A-Za-z0-9_./-]{0,14}(?![A-Za-z0-9_./-])(?!\s+\((?!strain|str\.|isolate|serotype|serovar|subsp\.))(?!\s+polyprotein\b)'
    r')*)'
    r'|(?P<pname>[A-Z][a-z]{2,}\s+(?:bacterio)?phage\s+(?![A-Z][a-z]{2,}\b)(?![A-Za-z0-9_.-]+-[a-z]{2,}\b)[A-Za-z0-9_.-]+(?![A-Za-z0-9_.-])(?!\s+\((?!strain|str\.|isolate))'
    r'|[A-Z][a-z]{2,}\s+[a-z]{3,}\s+[A-Z0-9]{1,4}\s+phage\b'
    r'|bacteriophage\s+(?![A-Z][a-z]{2,}\b)[A-Za-z0-9_.-]+\b)'
    r')',
    flags=re.UNICODE
)

_NON_VIRAL_START_ACRONYMS = {
    "DNA", "RNA", "ATP", "GTP", "CTP", "UTP", "ADP", "AMP", "NAD", "NADH", "NADP", "NADPH",
    "FAD", "FMN", "UDP", "GDP", "COA", "ABC", "AAA", "II", "III", "IV", "VI", "VII", "VIII", "IX",
}

_BINOMIAL_SPAN_RX = re.compile(
    r'(?<![A-Za-z0-9-])'
    r'(?P<cand>Candidatus\s+)?'
    r'(?P<genus>\[?[A-Z][a-z]{2,}\]?)\s+'
    r'(?P<hyb_pre>x\s+)?'
    r'(?P<species>sp\.(?![A-Za-z0-9-])|sp(?![A-Za-z0-9.-])|[a-z]{2,}(?:-[a-z]{2,})?)'
    r'(?P<hybrid>\s+x\s+(?:[A-Z][a-z]{2,}\s+)?[a-z]{3,})?'
    r'(?P<taut>\s+(?P=species))?'
    r'(?P<subsp>\s+(?:familiaris|domesticus|japonica|indica|brucei|gambiense|rhodesiense|enterica|subtilis|lactis|mellifera))?'
    r'(?![A-Za-z0-9-])'
)


def _lookup_catalog_by_sci_prefix(sci_candidate: str) -> Tuple[Optional[int], Optional[int], Optional[str], Tuple[int, ...]]:
    """Checks if a general binomial/viral/common string matches any entry in _COMPILED_ORGANISM_CATALOG."""
    low = sci_candidate.lower().strip()
    for rx, prim_id, clade_id, sci_name, extra_ids, _ in _COMPILED_ORGANISM_CATALOG:
        if rx.fullmatch(low):
            return prim_id, clade_id, sci_name, extra_ids
    return None, None, None, ()


def _resolve_primary_taxid_with_strain(
    prim_id: Optional[int],
    sci_name: Optional[str],
    strain_str: Optional[str]
) -> Optional[int]:
    """
    Avoids locking organism_taxid to a single reference strain (e.g. M. tuberculosis H37Rv 83332)
    when the prompt explicitly specifies an uncatalogued strain/serotype/isolate (e.g. CDC 1551).
    """
    if prim_id is None or not strain_str:
        return prim_id
    if sci_name and strain_str.lower() in sci_name.lower():
        return prim_id
    return None


def _extract_organism_details(text: str) -> Dict[str, Any]:
    """
    Universal Tree-of-Life organism extractor:
    Pass 1: Multi-word curated scientific & specific viral/phage names (outside internal strain parens).
    Pass 2: General ICTV viral & bacteriophage species scanner (start, end, and explicit clauses).
    Pass 3: Curated common host names, bare genera, and lab shorthands (evaluated before unanchored binomial guessing).
    Pass 4: General Latin binomial / trinomial / hybrid / Candidatus / 'Genus sp[.]' parser with strict protein guard.
    """
    work = " ".join(text.strip().split())
    empty_res = {
        "remaining_text": work,
        "organism_taxid": None,
        "organism_clade_taxid": None,
        "organism_taxids": [],
        "organism_name": None,
        "organism_raw": None,
        "organism_strain": None,
    }
    if not work:
        return empty_res

    # Pass 1: Curated multi-word scientific & specific viral/phage catalog
    for rx, prim_id, clade_id, sci_name, extra_ids, ctx_guard in _COMPILED_ORGANISM_CATALOG:
        if ctx_guard:
            continue
        for m in rx.finditer(work):
            s_idx, e_idx = m.start(), m.end()
            if _is_inside_parentheses(work, s_idx):
                ext_test, _ = _consume_organism_qualifiers(work, e_idx, matched_prim_taxid=prim_id)
                if not _is_enclosed_whole_paren(work, s_idx, ext_test):
                    continue
            after_slice = work[e_idx:].lstrip()
            first_after = after_slice.split()[0].lower().strip(".,;:()") if after_slice else ""
            if "virus" in sci_name.lower() and first_after in _VIRAL_POST_GUARD_WORDS:
                continue
            ext_end, strain_str = _consume_organism_qualifiers(work, e_idx, matched_prim_taxid=prim_id)
            rem = _clean_span_removal(work, s_idx, ext_end)
            if not _has_raw_protein_words(rem) and ext_end > e_idx:
                ext_end, strain_str = e_idx, None
                rem = _clean_span_removal(work, s_idx, ext_end)
            if _has_raw_protein_words(rem) or (prim_id == 12227 and "protease" in work.lower()):
                org_raw = work[s_idx:ext_end].strip()
                eff_prim = _resolve_primary_taxid_with_strain(prim_id, sci_name, strain_str)
                tax_list = [x for x in dict.fromkeys([eff_prim, clade_id] + ([] if eff_prim is None and strain_str else list(extra_ids))) if x is not None]
                return {
                    "remaining_text": rem,
                    "organism_taxid": eff_prim,
                    "organism_clade_taxid": clade_id,
                    "organism_taxids": tax_list,
                    "organism_name": org_raw if (prim_id is None and clade_id is None) else sci_name,
                    "organism_raw": org_raw,
                    "organism_strain": strain_str,
                }

    # Pass 2: General Viral / Bacteriophage species names (evaluating candidate Capitalized word starts)
    if "virus" in work.lower() or "phage" in work.lower():
        cap_starts = [m.start() for m in re.finditer(r'(?<![A-Za-z0-9\'-])[A-Z][A-Za-z0-9\'-]*', work)]
        viral_candidates = []
        for s_idx in cap_starts:
            if _is_inside_parentheses(work, s_idx):
                if not work[:s_idx].rstrip().endswith(("(", "[")):
                    continue
            first_w = work[s_idx:].split()[0]
            if s_idx > 0 and (
                len(first_w) == 1
                or first_w.upper() in _NON_VIRAL_START_ACRONYMS
                or (re.match(r'^[A-Z]+[0-9]+[A-Za-z0-9-]*$', first_w) and not first_w.upper().startswith(("HIV", "SARS", "MERS", "TMV", "SV40", "HSV", "EBV")))
            ):
                continue
            vm = _VIRAL_AT_POS_RX.match(work[s_idx:])
            if not vm:
                continue

            # Build candidate viral end offsets, allowing fallback if greedy vtail ate a trailing viral gene symbol (e.g. 'Newcastle disease virus HN')
            candidate_v_ends = [s_idx + vm.end()]
            if vm.group("vname") and vm.group("vtail"):
                vname_end = s_idx + vm.end("vname")
                cur_tail = vm.group("vtail")
                while cur_tail.strip():
                    cur_tail = re.sub(r'\s+\S+\s*$', '', cur_tail)
                    cand_pos = vname_end + len(cur_tail)
                    if cand_pos not in candidate_v_ends:
                        candidate_v_ends.append(cand_pos)

            for v_end in candidate_v_ends:
                if s_idx == 0:
                    after_v = work[v_end:].lstrip()
                    first_after = after_v.split()[0].lower().strip(".,;:()-") if after_v else ""
                    if first_after in _VIRAL_POST_GUARD_WORDS:
                        continue
                ext_end, strain_str = _consume_organism_qualifiers(work, v_end)
                rem = _clean_span_removal(work, s_idx, ext_end)
                if not _has_raw_protein_words(rem) and ext_end > v_end:
                    ext_end, strain_str = v_end, None
                    rem = _clean_span_removal(work, s_idx, ext_end)
                if _is_inside_parentheses(work, s_idx) and not _is_enclosed_whole_paren(work, s_idx, ext_end):
                    continue
                before_sub = work[:s_idx].rstrip()
                after_sub = work[ext_end:].strip()
                at_start = (s_idx == 0)
                at_end = (after_sub == "" or after_sub in {")", "]"})
                has_prep = bool(re.search(r'(?:\b(?:from|in|of)|[\(\[])$', before_sub, flags=re.IGNORECASE))
                if not (at_start or at_end or has_prep):
                    continue
                if not _has_raw_protein_words(rem):
                    continue
                prio = 3 if has_prep else (2 if at_end else 1)
                tie = -s_idx if at_end else s_idx
                viral_candidates.append((prio, tie, s_idx, v_end, ext_end, strain_str, rem))
                break

        if viral_candidates:
            def _viral_sort_key(item):
                prio, _, s_i, _, _, _, _ = item
                w0_low = work[s_i:].split()[0].lower().strip("'-")
                is_prot_word = 1 if w0_low in _NON_GENUS_PROTEIN_WORDS else 0
                return (-prio, is_prot_word, s_i)

            viral_candidates.sort(key=_viral_sort_key)
            _, _, s_idx, v_base_end, ext_end, strain_str, rem = viral_candidates[0]
            base_sci = work[s_idx:v_base_end].strip()
            full_raw = work[s_idx:ext_end].strip()
            p_id, c_id, cat_sci, e_ids = _lookup_catalog_by_sci_prefix(base_sci)
            eff_p_id = _resolve_primary_taxid_with_strain(p_id, cat_sci or base_sci, strain_str)
            t_list = list(dict.fromkeys([x for x in ([eff_p_id, c_id] + ([] if eff_p_id is None and strain_str else list(e_ids))) if x is not None]))
            return {
                "remaining_text": rem,
                "organism_taxid": eff_p_id,
                "organism_clade_taxid": c_id,
                "organism_taxids": t_list,
                "organism_name": cat_sci or base_sci,
                "organism_raw": full_raw,
                "organism_strain": strain_str,
            }

    # Pre-scan general Latin binomials so tiny ambiguous shorthands ('cat', 'dog', 'pea', etc.) yield to real binomials
    binom_candidates = []
    for bm in _BINOMIAL_SPAN_RX.finditer(work):
        s_idx = bm.start()
        if _is_inside_parentheses(work, s_idx) and not work[:s_idx].rstrip().endswith(("(", "[")):
            continue
        genus_tok = bm.group("genus")
        species_tok = bm.group("species")
        if not _is_valid_latin_binomial_tokens(genus_tok, species_tok):
            continue
        ext_end, strain_str = _consume_organism_qualifiers(work, bm.end())
        rem = _clean_span_removal(work, s_idx, ext_end)
        if not _has_raw_protein_words(rem) and ext_end > bm.end():
            ext_end, strain_str = bm.end(), None
            rem = _clean_span_removal(work, s_idx, ext_end)
        if _is_inside_parentheses(work, s_idx) and not _is_enclosed_whole_paren(work, s_idx, ext_end):
            continue
        before_b = work[:s_idx].rstrip()
        after_b = work[ext_end:].strip()
        has_prep_or_bracket = bool(re.search(r'(?:\b(?:from|in|of)|[\(\[])$', before_b, flags=re.IGNORECASE))
        at_start = (s_idx == 0)
        at_end = (after_b == "" or after_b in {")", "]"})
        if not (has_prep_or_bracket or at_start or at_end):
            continue
        if not _has_raw_protein_words(rem):
            continue
        priority = (
            3 if (strain_str or has_prep_or_bracket) else
            2 if at_end else
            1
        )
        binom_candidates.append((priority, s_idx, bm.end(), ext_end, strain_str, rem))

    # Pass 3: Single-word common host names, bare genera, and context-guarded lab shorthands
    short_ambig_tokens = {"cat", "dog", "pea", "hen", "pig", "cow", "tb", "t4", "t7", "taq", "pfu", "kod"}
    for rx, prim_id, clade_id, sci_name, extra_ids, ctx_guard in _COMPILED_ORGANISM_CATALOG:
        if not ctx_guard:
            continue
        for m in rx.finditer(work):
            matched_tok = m.group(0)
            tok_low = matched_tok.lower()
            s_idx, e_idx = m.start(), m.end()
            if _is_inside_parentheses(work, s_idx):
                ext_test, _ = _consume_organism_qualifiers(work, e_idx, matched_prim_taxid=prim_id)
                if not _is_enclosed_whole_paren(work, s_idx, ext_test):
                    continue
            before_m = work[:s_idx].rstrip()
            # Never steal pathovar/subspecies/sp. epithets (e.g. 'Pseudomonas syringae pv. tomato')
            if re.search(r'(?:pv\.|pathovar|subsp\.|var\.|serovar|biovar|f\.\s*sp\.|sp\.)\s*$', before_m, flags=re.IGNORECASE):
                continue
            # Never match a bare genus if a valid Pass 4 binomial/sp. clause starts at this exact position
            if any(bc[1] == s_idx for bc in binom_candidates):
                continue
            has_explicit_prep = bool(re.search(r'(?:\b(?:from|in|of)|[\(\[])$', before_m, flags=re.IGNORECASE))
            if tok_low in {"cat", "dog", "pea", "hen", "pig", "tb"} and matched_tok.isupper() and matched_tok != "TB" and not has_explicit_prep:
                continue
            if tok_low in short_ambig_tokens and not has_explicit_prep and binom_candidates:
                continue
            ext_end, strain_str = _consume_organism_qualifiers(work, e_idx, matched_prim_taxid=prim_id)
            rem = _clean_span_removal(work, s_idx, ext_end)
            if not _has_raw_protein_words(rem) and ext_end > e_idx:
                ext_end, strain_str = e_idx, None
                rem = _clean_span_removal(work, s_idx, ext_end)
            after_ext = work[ext_end:].strip()
            at_eff_start = (s_idx == 0) or all(w.lower().strip(",;:-") in _NON_PROTEIN_STOPWORDS for w in before_m.split())
            at_eff_end = (after_ext == "" or after_ext in {")", "]"})
            if not (has_explicit_prep or at_eff_start or at_eff_end):
                continue
            # If a competing strain-qualified or explicit-clause Latin binomial exists, prefer it over an unanchored common word
            if not has_explicit_prep and any(bc[0] >= 3 for bc in binom_candidates):
                continue
            if _has_raw_protein_words(rem):
                org_raw = work[s_idx:ext_end].strip()
                eff_prim = _resolve_primary_taxid_with_strain(prim_id, sci_name, strain_str)
                tax_list = [x for x in dict.fromkeys([eff_prim, clade_id] + ([] if eff_prim is None and strain_str else list(extra_ids))) if x is not None]
                return {
                    "remaining_text": rem,
                    "organism_taxid": eff_prim,
                    "organism_clade_taxid": clade_id,
                    "organism_taxids": tax_list,
                    "organism_name": sci_name,
                    "organism_raw": org_raw,
                    "organism_strain": strain_str,
                }

    # Pass 4: General Latin Binomial / Trinomial / Hybrid / Candidatus / 'Genus sp[.]' Parser
    if binom_candidates:
        binom_candidates.sort(key=lambda item: (-item[0], item[1]))
        _, b_start, b_base_end, b_ext_end, strain_str, rem = binom_candidates[0]
        base_sci = work[b_start:b_base_end].strip().replace("[", "").replace("]", "")
        full_raw = work[b_start:b_ext_end].strip()
        p_id, c_id, cat_sci, e_ids = _lookup_catalog_by_sci_prefix(base_sci)
        eff_p_id = _resolve_primary_taxid_with_strain(p_id, cat_sci or base_sci, strain_str)
        t_list = list(dict.fromkeys([x for x in ([eff_p_id, c_id] + ([] if eff_p_id is None and strain_str else list(e_ids))) if x is not None]))
        return {
            "remaining_text": rem,
            "organism_taxid": eff_p_id,
            "organism_clade_taxid": c_id,
            "organism_taxids": t_list,
            "organism_name": cat_sci or base_sci,
            "organism_raw": full_raw,
            "organism_strain": strain_str,
        }

    return empty_res


def _strip_known_organisms(text: str) -> Tuple[str, Optional[int], Optional[str]]:
    """
    Backwards-compatible wrapper returning (remaining_text, primary_taxon_id, canonical_organism_name)
    across all curated and general Tree-of-Life organisms.
    """
    info = _extract_organism_details(text)
    tax_out = info["organism_taxid"] if info["organism_taxid"] is not None else info["organism_clade_taxid"]
    return info["remaining_text"], tax_out, info["organism_name"]


def _has_real_protein_tokens(candidate_text: str) -> bool:
    """Returns True iff candidate_text retains at least one real protein/gene token beyond organism, digits & stop words."""
    info = _extract_organism_details(candidate_text)
    no_org = info["remaining_text"]
    taxid = info["organism_taxid"]
    if _has_raw_protein_words(no_org):
        return True
    if taxid == 12227 and "protease" in candidate_text.lower():
        return True
    return False


def _is_pure_modifier_phrase(phrase: str) -> bool:
    """Checks if a phrase (e.g. left side of 'fused to') consists solely of tag/position/cleavage tokens."""
    low = phrase.lower()
    stripped = re.sub(
        r'\b(?:twin-strep(?:-tag)?|strep-tag(?:\s+ii)?|10xhis|8xhis|6xhis|his10|his8|his6|'
        r'hexahis|octahis|decahis|polyhistidine|his|strep|flag|ha|c-myc|myc|gst|mbp|'
        r'tev|hrv[-\s]*3c|3c|prescission|thrombin|factor\s+xa|enterokinase|protease|'
        r'n-?term(?:inal(?:ly)?|inus)?|c-?term(?:inal(?:ly)?|inus)?|'
        r'tag|tags|tagged|fusion|site|cleavage|cut|and|with|a|an|the)\b',
        ' ',
        low
    )
    stripped = re.sub(r'[-_,\s]+', '', stripped)
    return len(stripped) == 0


_UNAMBIGUOUS_MODIFIER_TOKEN_RX = re.compile(
    r'(?<![A-Za-z0-9-])(?:'
    r'10xhis|8xhis|6xhis|his10|his8|his6|hexahis|octahis|decahis|polyhistidine|'
    r'twin-?strep(?:-tag)?|strep-tag(?:\s+ii)?|prescission|hrv[-\s]*3c'
    r')(?![A-Za-z0-9-])',
    flags=re.IGNORECASE
)

_EXPLICIT_MODIFIER_GRAMMAR_RX = re.compile(
    r'\b(?:tags?|tagged|fusion|fused|cleavage|sites?|cut|cleavable|'
    r'n-?term(?:inal(?:ly)?|inus)?|c-?term(?:inal(?:ly)?|inus)?)\b',
    flags=re.IGNORECASE
)

_BIO_DOMAIN_AFTER_TERMINUS_RX = re.compile(
    r'\b(?:n|c)-?term(?:inal|inus)\s+(?:domain|domains|region|subunit|lobe|extension|tail|helix|half)\b',
    flags=re.IGNORECASE
)


def _is_valid_modifier_clause(clause_text: str) -> bool:
    """
    Verifies that a clause after 'with / carrying / having / bearing / plus / fused to'
    genuinely specifies an engineered affinity tag or protease cleavage site (and NOT a natural domain
    such as 'with PEST-domain', 'interacting with c-Myc', or 'GST C-terminal domain').
    """
    if not clause_text or not clause_text.strip():
        return False
    if _BIO_DOMAIN_AFTER_TERMINUS_RX.search(clause_text) and not re.search(r'\b(?:tags?|tagged)\b', clause_text, flags=re.IGNORECASE):
        return False
    det_t, _, det_c = _detect_tag_and_cleavage(clause_text)
    if det_t is None and det_c is None:
        return False
    if _UNAMBIGUOUS_MODIFIER_TOKEN_RX.search(clause_text):
        return True
    if _EXPLICIT_MODIFIER_GRAMMAR_RX.search(clause_text):
        return True
    return False


def _split_target_and_modifiers(prompt: str) -> Tuple[str, str, bool]:
    """
    Context-aware grammar separation between Target Protein (+ Organism) and Engineered Tag / Cleavage Modifiers.
    Handles spaced/punctuated real-world tag notations ('6x His', '6× His', 'His-6', 'Twin Strep', 'Urease + 6x His tag').
    Returns: (target_phrase, modifier_text, force_untagged)
    """
    work = " ".join(prompt.strip().split())
    force_untagged = False

    # 0. Canonicalize spaced/hyphenated affinity tag spellings BEFORE clause/inline extraction
    #    so '6x His tagged urease', 'Urease + 6x His tag', 'Urease 6x His tag', 'Urease with 6x His' never strand '6x'!
    work = re.sub(r'(?<![A-Za-z0-9])(6|8|10)\s*[x×]\s*-?\s*his\b', r'\1xHis', work, flags=re.IGNORECASE)
    work = re.sub(r'(?<![A-Za-z0-9])his\s*-\s*(6|8|10)(?![A-Za-z0-9])', r'\1xHis', work, flags=re.IGNORECASE)
    work = re.sub(r'(?<![A-Za-z0-9])(6|8|10)\s+his(?=\s*(?:tags?|tagged|fusion|fused|\b|$))', r'\1xHis', work, flags=re.IGNORECASE)
    work = re.sub(r'\btwin\s+strep(?:\s+tag)?\b', 'Twin-Strep-tag', work, flags=re.IGNORECASE)
    work = re.sub(r'\bstrep\s+tag\s+ii\b', 'Strep-tag II', work, flags=re.IGNORECASE)
    work = re.sub(r'\bstrep\s+tag\b', 'Strep-tag', work, flags=re.IGNORECASE)
    work = re.sub(r'\bflag\s+tag\b', 'FLAG-tag', work, flags=re.IGNORECASE)

    untagged_rx = re.compile(r'\b(?:untagged|tagless|tag-free|without\s+(?:any\s+)?tags?)\b', flags=re.IGNORECASE)
    if untagged_rx.search(work):
        force_untagged = True
        work = " ".join(untagged_rx.sub(" ", work).split())

    modifier_chunks: List[str] = []

    # 1. Handle "fused to" / "fused with" ("<Target> fused to <Tag>" OR "<Tag> fused to <Target>")
    for fused_match in re.finditer(r'(?<![A-Za-z0-9-])fused\s+(?:to|with)(?![A-Za-z0-9-])', work, flags=re.IGNORECASE):
        left_side = work[:fused_match.start()].strip(" ,;-+&")
        right_side = work[fused_match.end():].strip(" ,;-+&")
        if _is_pure_modifier_phrase(left_side) and _is_valid_modifier_clause(left_side) and _has_real_protein_tokens(right_side):
            modifier_chunks.append(left_side)
            work = right_side
            break
        elif _has_real_protein_tokens(left_side) and _is_valid_modifier_clause(right_side):
            left_org = _extract_organism_details(left_side)
            right_org = _extract_organism_details(right_side)
            if left_org["organism_name"] is None and right_org["organism_name"] is not None:
                orig_tc = _detect_tag_and_cleavage(right_side)
                stripped_tc = _detect_tag_and_cleavage(right_org["remaining_text"])
                if stripped_tc[0] == orig_tc[0] and stripped_tc[2] == orig_tc[2]:
                    work = f"{left_side} from {right_org['organism_raw']}"
                    modifier_chunks.append(right_org["remaining_text"])
                    break
            work = left_side
            modifier_chunks.append(right_side)
            break

    # 2. Split on explicit modifier clause introducers ("with ...", "carrying ...", "having ...", "bearing ...", "plus ...", "+ ...")
    # Iterate RIGHT-TO-LEFT so proteins whose own names contain 'with' ('Arf-GAP with ... with 6xHis tag') split at the trailing modifier!
    clause_rx = re.compile(
        r'(?:(?<![A-Za-z0-9-])(?:with|carrying|having|bearing|plus)(?![A-Za-z0-9-])|\s+\+\s+)',
        flags=re.IGNORECASE
    )
    for clause_match in reversed(list(clause_rx.finditer(work))):
        left_side = work[:clause_match.start()].strip(" ,;-+&")
        right_side = work[clause_match.end():].strip(" ,;-+&")
        if _has_real_protein_tokens(left_side) and _is_valid_modifier_clause(right_side):
            left_org = _extract_organism_details(left_side)
            right_org = _extract_organism_details(right_side)
            if left_org["organism_name"] is None and right_org["organism_name"] is not None:
                orig_tc = _detect_tag_and_cleavage(right_side)
                stripped_tc = _detect_tag_and_cleavage(right_org["remaining_text"])
                if stripped_tc[0] == orig_tc[0] and stripped_tc[2] == orig_tc[2]:
                    work = f"{left_side} from {right_org['organism_raw']}"
                    modifier_chunks.append(right_org["remaining_text"])
                    break
            work = left_side
            modifier_chunks.append(right_side)
            break

    # 3. Extract explicit inline tag/cleavage modifier spans from remaining target phrase (subject to safety guard)
    inline_patterns = [
        re.compile(
            r'(?<![A-Za-z0-9-])(?:(?:at\s+(?:the\s+)?)?(?:n|c)-?term(?:inal(?:ly)?|inus)?\s+)?'
            r'(?:a\s+|an\s+)?'
            r'(?:twin-strep(?:-tag)?|strep-tag(?:\s+ii)?|10xhis|8xhis|6xhis|his10|his8|his6|'
            r'hexahis|octahis|decahis|polyhistidine|his|strep|flag|ha|c-myc|myc|gst|mbp)'
            r'(?:[-\s]+tagged\b|[-\s]+tag\b|\s+(?-i:fusion)(?:\s+of)?\b|-fused\b)'
            r'(?:\s+at\s+(?:the\s+)?(?:n|c)-?term(?:inal|inus)?)?',
            flags=re.IGNORECASE
        ),
        re.compile(
            r'(?:(?<![A-Za-z0-9-])(?:n|c)-?term(?:inal(?:ly)?|inus)?\s+'
            r'(?:(?:twin-strep|strep-tag(?:\s+ii)?|10xhis|8xhis|6xhis|his10|his8|his6)(?![A-Za-z0-9-])'
            r'|(?:strep|his|flag|ha|c-myc|myc|gst|mbp)(?![A-Za-z0-9-])(?!\s+(?-i:domain|domains|region|subunit|chain|lobe|extension|tail|helix)\b)))'
            r'|(?:(?<![A-Za-z0-9-])(?:twin-strep|strep-tag(?:\s+ii)?|strep|10xhis|8xhis|6xhis|his10|his8|his6|his|flag|ha|c-myc|myc|gst|mbp)'
            r'\s+at\s+(?:the\s+)?(?:n|c)-?term(?:inal|inus)?\b)',
            flags=re.IGNORECASE
        ),
        re.compile(
            r'(?<![A-Za-z0-9-])(?:twin-strep(?:-tag)?|strep-tag(?:\s+ii)?|10xhis|8xhis|6xhis|hexahis|octahis|decahis)(?![A-Za-z0-9-])',
            flags=re.IGNORECASE
        ),
        re.compile(
            r'(?<![A-Za-z0-9-])(?:(?:prescission|hrv[-\s]*3c|3c|tev|thrombin|factor\s+xa|enterokinase)(?:\s+protease)?'
            r'\s+(?:cleavage\s+site|cut\s+site|recognition\s+site|cleavage|site)'
            r'|(?:prescission|hrv[-\s]*3c|3c|tev|thrombin|factor\s+xa|enterokinase)-cleavable)\b',
            flags=re.IGNORECASE
        ),
    ]

    for rx in inline_patterns:
        while True:
            m = rx.search(work)
            if not m:
                break
            candidate_remaining = (work[:m.start()] + " " + work[m.end():]).strip(" ,;-+&")
            candidate_remaining = re.sub(
                r'^\b(?:of|and)\b\s*|\s*\b(?:of|and)\b$',
                '',
                candidate_remaining,
                flags=re.IGNORECASE
            ).strip(" ,;-+&")
            if _has_real_protein_tokens(candidate_remaining):
                modifier_chunks.append(m.group(0))
                work = " ".join(candidate_remaining.split())
            else:
                break

    # 4A. Hyphenated C-terminal UNAMBIGUOUS construct suffix notation (supports beta'-6xHis, [NADP]-6xHis, Lipase A-6xHis)
    hyph_unambig_suffix_rx = re.compile(
        r'(?<=[A-Za-z0-9\)\]\'])-'
        r'(?P<csuf>'
        r'(?:(?:tev|hrv3c|3c|prescission|thrombin|enterokinase)-)+'
        r'(?:twin-strep(?:-tag)?|strep-tag|10xhis|8xhis|6xhis|his10|his8|his6|hexahis|octahis|decahis|his|strep|flag|ha|c-myc|myc)(?:-tag)?'
        r'|(?:twin-strep(?:-tag)?|strep-tag|10xhis|8xhis|6xhis|his10|his8|his6|hexahis|octahis|decahis|'
        r'flag-tag|ha-tag|c-myc-tag|myc-tag)'
        r')(?![A-Za-z0-9-])',
        flags=re.IGNORECASE
    )
    m_csuf_u = hyph_unambig_suffix_rx.search(work)
    if m_csuf_u:
        csuf_raw = m_csuf_u.group("csuf")
        candidate_remaining = (work[:m_csuf_u.start()] + work[m_csuf_u.end():]).strip()
        if _has_real_protein_tokens(candidate_remaining):
            modifier_chunks.append("C-terminal " + csuf_raw.replace("-", " "))
            work = " ".join(candidate_remaining.split())

    # 4B. Hyphenated C-terminal short/dual-meaning construct suffix notation ("GFP-His", "p53-c-Myc", "GST-FLAG human")
    if not m_csuf_u:
        hyph_ambig_suffix_rx = re.compile(
            r'(?<![A-Za-z0-9-])(?P<stem>(?:c-)?[A-Za-z][A-Za-z0-9]{1,})-'
            r'(?P<csuf>his|strep|flag|ha|c-myc|myc)(?![A-Za-z0-9-])',
            flags=re.IGNORECASE
        )
        m_csuf_a = hyph_ambig_suffix_rx.search(work)
        if m_csuf_a and m_csuf_a.group("stem").lower() not in _EXCLUDED_SUFFIX_STEMS:
            stem_tok = m_csuf_a.group("stem")
            csuf_raw = m_csuf_a.group("csuf")
            candidate_remaining = (work[:m_csuf_a.start()] + stem_tok + work[m_csuf_a.end():]).strip()
            if _has_real_protein_tokens(candidate_remaining):
                modifier_chunks.append("C-terminal " + csuf_raw.replace("-", " "))
                work = " ".join(candidate_remaining.split())

    # 5. Hyphenated N-terminal construct prefix notation ("6xHis-TEV-lysozyme", "GST-p53", "His-TEV-GFP")
    hyph_prefix_rx = re.compile(
        r'(?<![A-Za-z0-9-])'
        r'(?P<prefix>'
        r'(?:twin-strep|strep-tag|10xhis|8xhis|6xhis|his10|his8|his6|hexahis|octahis|decahis|his|strep|flag|ha|c-myc|myc|gst|mbp)-'
        r'(?:(?:tev|hrv3c|3c|prescission|thrombin|enterokinase)-)*'
        r')'
        r'(?P<rest>(?:c-)?[A-Za-z][A-Za-z0-9_]*)(?![A-Za-z0-9_])',
        flags=re.IGNORECASE
    )
    m_hyph = hyph_prefix_rx.search(work)
    if m_hyph:
        pref_raw = m_hyph.group("prefix")
        pref_clean = pref_raw.rstrip("-").lower()
        rest_tok = m_hyph.group("rest")
        rest_low = rest_tok.lower()
        is_unambig_pref = bool(
            re.match(
                r'^(?:twin-strep|strep-tag|10xhis|8xhis|6xhis|his10|his8|his6|hexahis|octahis|decahis)(?:-|$)',
                pref_clean
            )
            or "-" in pref_clean.replace("twin-strep", "twinstrep").replace("strep-tag", "streptag").replace("c-myc", "cmyc")
        )
        is_gst_isozyme = (pref_clean == "gst" and rest_low in _GST_ISOZYME_DESIGNATORS)
        is_pure_tag_rest = rest_low in _PURE_TAG_REST_TOKENS
        is_bio_compound = (not is_unambig_pref) and (
            rest_low in _AMBIGUOUS_PREFIX_BIO_BLOCKLIST or len(rest_tok) < 2
        )
        candidate_remaining = (work[:m_hyph.start()] + rest_tok + work[m_hyph.end():]).strip()
        if not is_gst_isozyme and not is_pure_tag_rest and not is_bio_compound and _has_real_protein_tokens(candidate_remaining):
            modifier_chunks.append("N-terminal " + pref_raw.replace("-", " "))
            work = " ".join(candidate_remaining.split())

    modifier_text = " ; ".join(c for c in modifier_chunks if c.strip())
    return work, modifier_text, force_untagged


def _detect_tag_and_cleavage_keys(
    modifier_text: str
) -> Tuple[Optional[str], Optional[Dict[str, str]], str, Optional[str], Optional[Dict[str, Any]]]:
    """Detects (tag_key, tag_dict, tag_position, cleavage_key, cleavage_dict) strictly within modifier_text."""
    if not modifier_text or not modifier_text.strip():
        return None, None, "N/A", None, None

    low_mod = modifier_text.lower()
    low_mod = re.sub(r'\btwin\s+strep\b', 'twin-strep', low_mod)
    low_mod = re.sub(r'\bc\s+myc\b', 'c-myc', low_mod)

    detected_tag_key: Optional[str] = None
    detected_tag: Optional[Dict[str, str]] = None
    tag_patterns = [
        (r'\btwin-?strep\b', "twin-strep"),
        (r'\b(?:10xhis|his10|decahis)\b', "10xhis"),
        (r'\b(?:8xhis|his8|octahis)\b', "8xhis"),
        (r'\b(?:6xhis|his6|hexahis|polyhistidine|his)\b', "6xhis"),
        (r'\bstrep\b', "strep"),
        (r'\bflag\b', "flag"),
        (r'(?<![A-Za-z0-9])(?:c-myc|myc)\b', "myc"),
        (r'\bha\b', "ha"),
        (r'\b(?:gst|glutathione\s+s-?transferase)\b', "gst"),
        (r'\b(?:mbp|maltose[-\s]+binding)\b', "mbp"),
    ]
    for pat, t_key in tag_patterns:
        if re.search(pat, low_mod):
            detected_tag_key = t_key
            detected_tag = STANDARD_TAGS.get(t_key)
            break

    is_c_term = bool(re.search(r'\bc-?term(?:inal(?:ly)?|inus)?\b', low_mod))
    tag_position = ("C-terminal" if is_c_term else "N-terminal") if detected_tag else "N/A"

    detected_cleavage_key: Optional[str] = None
    detected_cleavage: Optional[Dict[str, Any]] = None
    cleavage_patterns = [
        (r'\b(?:tev|tobacco\s+etch)\b', "tev"),
        (r'\b(?:prescission|hrv[-\s]*3c|3c)\b', "3c"),
        (r'\bthrombin\b', "thrombin"),
        (r'\bfactor[\s_-]*xa\b', "factor_xa"),
        (r'\b(?:enterokinase|enteropeptidase)\b', "enterokinase"),
    ]
    for pat, c_key in cleavage_patterns:
        if re.search(pat, low_mod):
            detected_cleavage_key = c_key
            detected_cleavage = STANDARD_CLEAVAGE_SITES.get(c_key)
            break

    return detected_tag_key, detected_tag, tag_position, detected_cleavage_key, detected_cleavage


def _detect_tag_and_cleavage(modifier_text: str) -> Tuple[Optional[Dict[str, str]], str, Optional[Dict[str, Any]]]:
    """Detects engineered affinity tag, terminus, and protease cleavage site strictly within modifier_text."""
    _, det_t, pos, _, det_c = _detect_tag_and_cleavage_keys(modifier_text)
    return det_t, pos, det_c


def _normalize_protein_query(raw_protein_phrase: str) -> str:
    """
    Cleans boundary construct filler descriptors while strictly preserving internal protein words
    (such as 'Lipase A', 'Protein A', 'Cytochrome c', 'Regulator of G-protein signaling 4', 'Valosin-containing protein').
    """
    q = " ".join(raw_protein_phrase.strip().split()).strip(" ,;")
    if not q:
        return ""

    while True:
        m_lead = re.match(
            r'^(?:(?-i:(?:the|a|an)\s+)|(?:from|of|in|recombinant|wild-type|wildtype|full-length|purified|canonical|native|mature)\s+)',
            q,
            flags=re.IGNORECASE
        )
        if m_lead and _has_raw_protein_words(q[m_lead.end():]):
            q = q[m_lead.end():].strip(" ,;")
        else:
            break

    q = re.sub(r'\s+\b(?:from|of|in)$', '', q, flags=re.IGNORECASE).strip(" ,;")

    m_single_prot = re.match(r'^((?:c-)?[A-Za-z0-9_]+)\s+protein$', q, flags=re.IGNORECASE)
    if m_single_prot and m_single_prot.group(1).lower() not in {
        "fluorescent", "basic", "binding", "containing", "coat", "capsid", "envelope", "matrix",
        "membrane", "ribosomal", "shock", "spike", "fusion", "carrier", "scaffold", "structural",
        "nonstructural", "inclusion", "uncharacterized", "hypothetical", "putative", "probable",
        "periplasmic", "outer", "inner", "nuclear", "nucleocapsid", "tail", "portal", "holin",
    }:
        q = m_single_prot.group(1)

    m_ase_enz = re.match(r'^([A-Za-z0-9_-]+ase)\s+enzyme$', q, flags=re.IGNORECASE)
    if m_ase_enz:
        q = m_ase_enz.group(1)

    return q


_COFACTOR_PAREN_QUALIFIERS = {
    "nad", "nad+", "nad(+)", "nadh", "nadp", "nadp+", "nadp(+)", "nadph", "nad(p)+", "nad(p)(+)",
    "fad", "fmn", "atp", "gtp", "ctp", "utp", "adp", "amp", "udp", "gdp", "coa",
    "quinone", "ubiquinone", "menaquinone", "plastoquinone", "ammonia", "glutamine-hydrolyzing",
    "isomerizing", "decarboxylating", "oxaloacetate-decarboxylating", "pyruvate", "fumarate",
    "cytochrome", "ferredoxin", "soluble", "mitochondrial", "chloroplastic", "cytoplasmic",
    "periplasmic", "extracellular", "membrane-bound", "reversible", "irreversible", "fragment",
}


def _split_gene_and_paren_protein(protein_query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Detects '<GeneSymbol> (<Recommended Protein Name>)' prompts (e.g. 'dnaK (Chaperone protein DnaK)',
    'fs(1)K10 (DNA-binding protein K10)', '2 (Terminase, large subunit)') without misclassifying
    enzyme cofactor qualifiers like 'Malate dehydrogenase (quinone)' or '1-(5-phosphoribosyl)...'.
    """
    pq = protein_query.strip()
    m = re.match(r'^(?P<gene>[A-Za-z0-9][A-Za-z0-9_.\-\(\)]{0,20})\s+\((?P<inner>.+)\)$', pq)
    if not m:
        return None, None
    gene_cand = m.group("gene")
    g_depth = 0
    for ch in gene_cand:
        if ch == "(":
            g_depth += 1
        elif ch == ")":
            g_depth -= 1
            if g_depth < 0:
                return None, None
    if g_depth != 0:
        return None, None
    open_idx = pq.index("(", len(gene_cand))
    if _find_balanced_close(pq, open_idx, "(", ")") != len(pq) - 1:
        return None, None
    inner_cand = m.group("inner").strip()
    if gene_cand.lower() in _NON_GENUS_PROTEIN_WORDS and len(gene_cand) > 4:
        return None, None
    if inner_cand.lower() in _COFACTOR_PAREN_QUALIFIERS:
        return None, None
    return gene_cand, inner_cand


_ECOLI_TAXIDS = {83333, 562, 83334, 199310, 469008, 511693}
_SAUREUS_TAXIDS = {93061, 1280, 158879, 93062, 282458, 367830, 46170}
_EUKARYOTE_CATALOG_TAXIDS = {
    # Animals / Metazoa
    9606, 10090, 10116, 9031, 9913, 9823, 9825, 9940, 9925, 9796, 9615, 9612, 9685, 9986,
    10141, 10036, 10029, 9598, 9544, 9755, 7955, 7227, 6239, 8355, 8364, 7460, 7091, 7165,
    180454, 6182, 6183, 6100, 7668,
    # Plants & Green Algae (note: plastid genes like infA/rbcL also exist in plants, handled in clauses)
    3702, 3555, 161934, 60093, 4058, 128735, 39947, 4530, 39946, 4577, 3562, 4081, 4113,
    4097, 4100, 3847, 4565, 4513, 112509, 3888, 3880, 29760, 3708, 3055, 3218,
    # Fungi
    559292, 4932, 284812, 4896, 367110, 5141, 237561, 5476, 227321, 162425, 425011, 5061,
    330879, 746128, 5085, 644223, 4922, 460519, 284591, 4952, 214684, 5207, 237631, 5270,
    # Protists
    36329, 5833, 508771, 5811, 185431, 5691, 5690, 353153, 5693, 347515, 5664, 44689, 352472,
    312017, 5911, 5888, 412030, 556484, 2850, 296543, 35128, 184922, 5741, 294381, 5759,
}


def _get_canonical_preferences(
    protein_query: str,
    organism_taxid: Optional[int],
    organism_name: Optional[str] = None
) -> Tuple[List[str], List[str], Optional[str]]:
    """Returns (preferred_accessions, preferred_genes, specific_uniprot_clause) for classic ambiguous & shorthand names."""
    pq_low = " ".join(protein_query.strip().lower().split())
    preferred_accs: List[str] = []
    preferred_genes: List[str] = []
    specific_field_clause: Optional[str] = None
    has_any_org = (organism_taxid is not None) or bool(organism_name and organism_name.strip())
    is_ecoli = organism_taxid in _ECOLI_TAXIDS
    is_euk = organism_taxid in _EUKARYOTE_CATALOG_TAXIDS

    if pq_low in {"gst", "glutathione s-transferase", "glutathione s transferase"}:
        if organism_taxid == 9606:
            preferred_accs = ["P09211", "P08263", "P09488"]
            preferred_genes = ["GSTP1", "GSTA1", "GSTM1"]
            specific_field_clause = (
                '(gene_exact:GSTP1 OR gene:GSTP1 OR gene:GSTA1 OR gene:GSTM1 '
                'OR protein_name:"Glutathione S-transferase")'
            )
        elif not has_any_org:
            preferred_accs = ["P08515", "P09211"]
            preferred_genes = ["GSTP1", "GSTA1", "GSTM1"]
            specific_field_clause = (
                '(accession:P08515 OR gene_exact:GSTP1 OR protein_name:"Glutathione S-transferase")'
            )
        else:
            preferred_genes = ["GSTP1", "GSTA1", "GSTM1", "GST"]
            specific_field_clause = (
                '(protein_name:"Glutathione S-transferase" OR gene:GSTP1 OR gene:GSTA1 OR gene:GSTM1 OR gene:gst)'
            )
    elif pq_low in {"gstp1", "gst-p1", "gst-pi"}:
        preferred_accs = ["P09211"]
        preferred_genes = ["GSTP1"]
        specific_field_clause = '(gene_exact:GSTP1 OR protein_name:"Glutathione S-transferase P")'
    elif pq_low in {"gsta1", "gst-a1", "gst-alpha"}:
        preferred_accs = ["P08263"]
        preferred_genes = ["GSTA1"]
        specific_field_clause = '(gene_exact:GSTA1 OR protein_name:"Glutathione S-transferase A1")'
    elif pq_low in {"gstm1", "gst-m1", "gst-mu"}:
        preferred_accs = ["P09488"]
        preferred_genes = ["GSTM1"]
        specific_field_clause = '(gene_exact:GSTM1 OR protein_name:"Glutathione S-transferase Mu 1")'
    elif pq_low == "mbp":
        if organism_taxid == 9606:
            preferred_accs = ["P02686"]
            preferred_genes = ["MBP"]
            specific_field_clause = '(gene_exact:MBP OR protein_name:"Myelin basic protein")'
        elif is_ecoli:
            preferred_accs = ["P0AEX9"]
            preferred_genes = ["MALE"]
            specific_field_clause = (
                '(gene_exact:malE OR gene:malE OR protein_name:"Maltose/maltodextrin-binding" '
                'OR protein_name:"Maltose-binding")'
            )
        elif not has_any_org:
            preferred_accs = ["P0AEX9", "P02686"]
            preferred_genes = ["MALE", "MBP"]
            specific_field_clause = (
                '(accession:P0AEX9 OR gene_exact:malE OR gene_exact:MBP '
                'OR protein_name:"Maltose/maltodextrin-binding" OR protein_name:"Myelin basic protein")'
            )
        else:
            preferred_genes = ["MBP", "MALE"]
            specific_field_clause = (
                '(gene:malE OR gene:MBP OR protein_name:"Maltose-binding" OR protein_name:"Myelin basic protein")'
            )
    elif pq_low in {"myc", "c-myc"}:
        if not has_any_org or organism_taxid == 9606:
            preferred_accs = ["P01106"]
        preferred_genes = ["MYC"]
        acc_hint = "accession:P01106 OR " if not has_any_org else ""
        specific_field_clause = f'({acc_hint}gene_exact:MYC OR gene:MYC OR protein_name:"Myc proto-oncogene protein")'
    elif pq_low in {"p53", "tp53"}:
        if not has_any_org or organism_taxid == 9606:
            preferred_accs = ["P04637"]
        preferred_genes = ["TP53", "TRP53"]
        if not has_any_org:
            specific_field_clause = '(accession:P04637 OR gene_exact:TP53 OR protein_name:"Cellular tumor antigen p53")'
        else:
            specific_field_clause = '(gene_exact:TP53 OR gene:TP53 OR gene:Trp53 OR protein_name:"Cellular tumor antigen p53")'
    elif pq_low in {"thrombin", "prothrombin"}:
        if not has_any_org or organism_taxid == 9606:
            preferred_accs = ["P00734"]
        preferred_genes = ["F2"]
        acc_hint = "accession:P00734 OR " if not has_any_org else ""
        specific_field_clause = f'({acc_hint}gene_exact:F2 OR protein_name:"Prothrombin" OR protein_name:"Thrombin")'
    elif pq_low in {"factor xa", "factor x", "factor 10", "f10"}:
        preferred_genes = ["F10"]
        specific_field_clause = '(gene_exact:F10 OR protein_name:"Coagulation factor X")'
    elif pq_low in {"enterokinase", "enteropeptidase"}:
        preferred_genes = ["TMPRSS15"]
        specific_field_clause = '(gene_exact:TMPRSS15 OR protein_name:"Enteropeptidase")'
    elif pq_low in {"gfp", "green fluorescent", "green fluorescent protein"}:
        if not has_any_org or organism_taxid == 6100:
            preferred_accs = ["P42212"]
        preferred_genes = ["GFP"]
        acc_hint = "accession:P42212 OR " if (not has_any_org or organism_taxid == 6100) else ""
        specific_field_clause = f'({acc_hint}gene:GFP OR protein_name:"Green fluorescent protein")'
    elif pq_low in {"bsa", "serum albumin"}:
        if pq_low == "bsa" or not has_any_org or organism_taxid == 9913:
            preferred_accs = ["P02769"]
        preferred_genes = ["ALB"]
        specific_field_clause = '(gene_exact:ALB OR protein_name:"Serum albumin" OR protein_name:"Albumin")'
    elif pq_low in {"urease", "jack bean urease", "plant urease", "urea amidohydrolase", "bacterial urease"}:
        if pq_low == "bacterial urease":
            preferred_accs = ["P18314", "P69996", "P41020"]
            preferred_genes = ["UREC", "UREB"]
            specific_field_clause = (
                '(accession:P18314 OR accession:P69996 OR accession:P41020 '
                'OR protein_name:"Urease subunit alpha" OR protein_name:"Urease subunit beta")'
            )
        else:
            if not has_any_org or organism_taxid == 3823:
                preferred_accs = ["P07374"]
            elif organism_taxid in (85962, 210):
                preferred_accs = ["P69996", "P14916"]
            elif organism_taxid == 548:
                preferred_accs = ["P18314"]
            elif organism_taxid == 1474:
                preferred_accs = ["P41020"]
            preferred_genes = ["UREA", "UREB", "UREC"]
            acc_hint = "accession:P07374 OR " if (not has_any_org or organism_taxid == 3823) else ""
            specific_field_clause = (
                f'({acc_hint}protein_name:"Urease" OR protein_name:"Urease subunit alpha" '
                'OR protein_name:"Urease subunit beta" OR ec:3.5.1.5)'
            )
    elif pq_low in {"if1", "if-1", "initiation factor 1", "initiation factor if1", "initiation factor if-1", "translation initiation factor 1"}:
        if is_ecoli or not has_any_org:
            preferred_accs = ["P69222"]
        elif organism_taxid == 9606:
            preferred_accs = ["P41567"]
        preferred_genes = ["EIF1", "SUI1", "INFA"] if is_euk else ["INFA", "EIF1"]
        acc_hint = "accession:P69222 OR " if (is_ecoli or not has_any_org) else ""
        specific_field_clause = (
            f'({acc_hint}gene_exact:infA OR gene:infA OR gene_exact:EIF1 OR gene:EIF1 OR gene_exact:SUI1 '
            'OR protein_name:"Translation initiation factor IF-1" OR protein_name:"IF-1" '
            'OR protein_name:"Eukaryotic translation initiation factor 1")'
        )
    elif pq_low in {"if2", "if-2", "initiation factor 2", "initiation factor if2", "initiation factor if-2", "translation initiation factor 2"}:
        if is_ecoli or not has_any_org:
            preferred_accs = ["P0A705"]
        elif organism_taxid == 9606:
            preferred_accs = ["P05198", "P46199"]
        preferred_genes = ["EIF2S1", "MTIF2", "SUI2", "INFB"] if is_euk else ["INFB", "EIF2S1", "MTIF2"]
        acc_hint = "accession:P0A705 OR " if (is_ecoli or not has_any_org) else ""
        specific_field_clause = (
            f'({acc_hint}gene_exact:infB OR gene:infB OR gene_exact:EIF2S1 OR gene_exact:MTIF2 OR gene_exact:SUI2 '
            'OR protein_name:"Translation initiation factor IF-2" OR protein_name:"IF-2" '
            'OR protein_name:"Eukaryotic translation initiation factor 2 subunit 1")'
        )
    elif pq_low in {"if3", "if-3", "initiation factor 3", "initiation factor if3", "initiation factor if-3", "translation initiation factor 3"}:
        if is_ecoli or not has_any_org:
            preferred_accs = ["P0A707"]
        elif organism_taxid == 9606:
            preferred_accs = ["Q14152", "Q9H2K0"]
        preferred_genes = ["EIF3A", "MTIF3", "TIF32", "INFC"] if is_euk else ["INFC", "EIF3A", "MTIF3"]
        acc_hint = "accession:P0A707 OR " if (is_ecoli or not has_any_org) else ""
        specific_field_clause = (
            f'({acc_hint}gene_exact:infC OR gene:infC OR gene_exact:EIF3A OR gene_exact:MTIF3 OR gene_exact:TIF32 '
            'OR protein_name:"Translation initiation factor IF-3" OR protein_name:"IF-3" '
            'OR protein_name:"Eukaryotic translation initiation factor 3 subunit A")'
        )
    elif pq_low in {"eftu", "ef-tu", "ef tu"}:
        if is_ecoli or not has_any_org:
            preferred_accs = ["P0CE47", "P0CE48"]
        elif organism_taxid == 9606:
            preferred_accs = ["P49411"]
        preferred_genes = ["TUFM", "TUFA", "TUFB", "TUF"] if is_euk else ["TUFA", "TUFB", "TUF", "TUFM"]
        acc_hint = "accession:P0CE47 OR " if (is_ecoli or not has_any_org) else ""
        specific_field_clause = (
            f'({acc_hint}gene_exact:tufA OR gene_exact:tuf OR gene:tufA OR gene:tufB OR gene:tuf OR gene_exact:TUFM '
            'OR protein_name:"Elongation factor Tu" OR protein_name:"EF-Tu")'
        )
    elif pq_low in {"efts", "ef-ts", "ef ts"}:
        if is_ecoli or not has_any_org:
            preferred_accs = ["P0A6P1"]
        elif organism_taxid == 9606:
            preferred_accs = ["P43897"]
        preferred_genes = ["TSFM", "TSF"] if is_euk else ["TSF", "TSFM"]
        acc_hint = "accession:P0A6P1 OR " if (is_ecoli or not has_any_org) else ""
        specific_field_clause = (
            f'({acc_hint}gene_exact:tsf OR gene:tsf OR gene_exact:TSFM '
            'OR protein_name:"Elongation factor Ts" OR protein_name:"EF-Ts")'
        )
    elif pq_low in {"efg", "ef-g", "ef g"}:
        if is_ecoli or not has_any_org:
            preferred_accs = ["P0A6M8"]
        elif organism_taxid == 9606:
            preferred_accs = ["Q96RP9"]
        preferred_genes = ["GFM1", "FUSA", "FUS"] if is_euk else ["FUSA", "FUS", "GFM1"]
        acc_hint = "accession:P0A6M8 OR " if (is_ecoli or not has_any_org) else ""
        specific_field_clause = (
            f'({acc_hint}gene_exact:fusA OR gene:fusA OR gene:fus OR gene_exact:GFM1 '
            'OR protein_name:"Elongation factor G" OR protein_name:"EF-G")'
        )
    elif pq_low in {"rf1", "rf-1"}:
        if is_ecoli or not has_any_org:
            preferred_accs = ["P0A7I0"]
        elif organism_taxid == 9606:
            preferred_accs = ["P62495"]
        preferred_genes = ["ETF1", "SUP45", "MTRF1L", "PRFA"] if is_euk else ["PRFA", "ETF1"]
        acc_hint = "accession:P0A7I0 OR " if (is_ecoli or not has_any_org) else ""
        specific_field_clause = (
            f'({acc_hint}gene_exact:prfA OR gene:prfA OR gene_exact:ETF1 OR gene_exact:SUP45 '
            'OR protein_name:"Peptide chain release factor 1" OR protein_name:"RF-1" '
            'OR protein_name:"Eukaryotic peptide chain release factor subunit 1")'
        )
    elif pq_low in {"rf2", "rf-2"}:
        if is_ecoli or not has_any_org:
            preferred_accs = ["P07012"]
        preferred_genes = ["PRFB"]
        acc_hint = "accession:P07012 OR " if (is_ecoli or not has_any_org) else ""
        specific_field_clause = (
            f'({acc_hint}gene_exact:prfB OR gene:prfB '
            'OR protein_name:"Peptide chain release factor 2" OR protein_name:"RF-2")'
        )
    elif pq_low in {"rf3", "rf-3"}:
        if is_ecoli or not has_any_org:
            preferred_accs = ["P0A7I4"]
        elif organism_taxid == 9606:
            preferred_accs = ["P15170"]
        preferred_genes = ["GSPT1", "SUP35", "PRFC"] if is_euk else ["PRFC", "GSPT1"]
        acc_hint = "accession:P0A7I4 OR " if (is_ecoli or not has_any_org) else ""
        specific_field_clause = (
            f'({acc_hint}gene_exact:prfC OR gene:prfC OR gene_exact:GSPT1 OR gene_exact:SUP35 '
            'OR protein_name:"Peptide chain release factor 3" OR protein_name:"RF-3")'
        )
    elif pq_low == "rrf":
        if is_ecoli or not has_any_org:
            preferred_accs = ["P0A805"]
        elif organism_taxid == 9606:
            preferred_accs = ["Q96E11"]
        preferred_genes = ["MRRF", "FRR"] if is_euk else ["FRR", "MRRF"]
        acc_hint = "accession:P0A805 OR " if (is_ecoli or not has_any_org) else ""
        specific_field_clause = (
            f'({acc_hint}gene_exact:frr OR gene:frr OR gene_exact:MRRF '
            'OR protein_name:"Ribosome-recycling factor" OR protein_name:"RRF")'
        )
    elif pq_low == "groel":
        if is_ecoli or not has_any_org:
            preferred_accs = ["P0A6F5"]
        elif organism_taxid == 9606:
            preferred_accs = ["P10809"]
        preferred_genes = ["HSPD1", "GROL", "GROEL"] if is_euk else ["GROL", "GROEL", "HSPD1"]
        acc_hint = "accession:P0A6F5 OR " if (is_ecoli or not has_any_org) else ""
        specific_field_clause = (
            f'({acc_hint}gene_exact:groL OR gene_exact:groEL OR gene:groL OR gene:groEL OR gene_exact:HSPD1 '
            'OR protein_name:"Chaperonin GroEL" OR protein_name:"GroEL")'
        )
    elif pq_low == "groes":
        if is_ecoli or not has_any_org:
            preferred_accs = ["P0A6F9"]
        elif organism_taxid == 9606:
            preferred_accs = ["P61604"]
        preferred_genes = ["HSPE1", "GROS", "GROES"] if is_euk else ["GROS", "GROES", "HSPE1"]
        acc_hint = "accession:P0A6F9 OR " if (is_ecoli or not has_any_org) else ""
        specific_field_clause = (
            f'({acc_hint}gene_exact:groS OR gene_exact:groES OR gene:groS OR gene:groES OR gene_exact:HSPE1 '
            'OR protein_name:"Co-chaperonin GroES" OR protein_name:"GroES")'
        )
    elif pq_low in {"rnase a", "rnasea"}:
        if not has_any_org or organism_taxid == 9913:
            preferred_accs = ["P61823"]
        elif organism_taxid == 9606:
            preferred_accs = ["P07998"]
        preferred_genes = ["RNASE1"]
        acc_hint = "accession:P61823 OR " if (not has_any_org or organism_taxid == 9913) else ""
        specific_field_clause = (
            f'({acc_hint}gene_exact:RNASE1 OR gene:RNASE1 '
            'OR protein_name:"Ribonuclease pancreatic" OR protein_name:"Ribonuclease A" OR protein_name:"RNase A")'
        )
    elif pq_low in {"rnase h", "rnaseh"}:
        if is_ecoli or not has_any_org:
            preferred_accs = ["P0A7Y4"]
        elif organism_taxid == 9606:
            preferred_accs = ["O60930"]
        preferred_genes = ["RNASEH1", "RNHA", "RNH"] if is_euk else ["RNHA", "RNH", "RNASEH1"]
        acc_hint = "accession:P0A7Y4 OR " if (is_ecoli or not has_any_org) else ""
        specific_field_clause = (
            f'({acc_hint}gene_exact:rnhA OR gene:rnhA OR gene_exact:RNASEH1 '
            'OR protein_name:"Ribonuclease H" OR protein_name:"RNase H")'
        )
    elif pq_low == "rubisco":
        if organism_taxid == 3702:
            preferred_accs = ["O03042"]
        elif not has_any_org or organism_taxid == 3562:
            preferred_accs = ["P00875"]
        preferred_genes = ["RBCL", "CBBM"]
        acc_hint = "accession:P00875 OR " if (not has_any_org or organism_taxid == 3562) else ""
        specific_field_clause = (
            f'({acc_hint}gene_exact:rbcL OR gene:rbcL OR gene:cbbM '
            'OR protein_name:"Ribulose bisphosphate carboxylase large chain" '
            'OR protein_name:"Ribulose bisphosphate carboxylase")'
        )
    elif pq_low in {"protein a", "staphylococcal protein a"} and (organism_taxid in _SAUREUS_TAXIDS or not has_any_org):
        preferred_accs = ["P02976", "P38507", "P99134"]
        preferred_genes = ["SPA"]
        acc_hint = "accession:P02976 OR " if (organism_taxid in _SAUREUS_TAXIDS or not has_any_org) else ""
        specific_field_clause = (
            f'({acc_hint}gene_exact:spa OR gene:spa '
            'OR protein_name:"Immunoglobulin G-binding protein A" OR protein_name:"Staphylococcal protein A")'
        )
    elif pq_low == "notch":
        if organism_taxid == 7227:
            preferred_accs = ["P07207"]
            preferred_genes = ["N"]
            specific_field_clause = (
                '(accession:P07207 OR gene_exact:N '
                'OR protein_name:"Neurogenic locus Notch protein" OR protein_name:"Notch")'
            )
        elif organism_taxid == 9606 or not has_any_org:
            preferred_accs = ["P46531", "P07207"]
            preferred_genes = ["NOTCH1", "N"]
            acc_hint = "accession:P46531 OR " if not has_any_org else ""
            specific_field_clause = (
                f'({acc_hint}gene_exact:NOTCH1 '
                'OR protein_name:"Neurogenic locus notch homolog protein 1" '
                'OR protein_name:"Neurogenic locus Notch protein")'
            )
        else:
            preferred_genes = ["NOTCH1", "NOTCH1A", "LIN-12", "N"]
            specific_field_clause = (
                '(gene:NOTCH1 OR gene:notch1a '
                'OR protein_name:"Neurogenic locus notch" OR protein_name:"Notch")'
            )
    elif pq_low in {"titin", "ttn", "connectin"}:
        if organism_taxid == 9606 or not has_any_org:
            preferred_accs = ["Q8WZ42"]
        elif organism_taxid == 10090:
            preferred_accs = ["A2ASS6"]
        preferred_genes = ["TTN"]
        acc_hint = "accession:Q8WZ42 OR " if (organism_taxid == 9606 or not has_any_org) else ""
        specific_field_clause = (
            f'({acc_hint}gene_exact:TTN OR gene:TTN '
            'OR protein_name:"Titin" OR protein_name:"Connectin")'
        )
    elif pq_low in {"dystrophin", "dmd"}:
        if organism_taxid == 9606 or not has_any_org:
            preferred_accs = ["P11532"]
        elif organism_taxid == 10090:
            preferred_accs = ["P11531"]
        preferred_genes = ["DMD"]
        acc_hint = "accession:P11532 OR " if (organism_taxid == 9606 or not has_any_org) else ""
        specific_field_clause = f'({acc_hint}gene_exact:DMD OR gene:DMD OR protein_name:"Dystrophin")'
    elif pq_low in {"nebulin", "neb"}:
        if organism_taxid == 9606 or not has_any_org:
            preferred_accs = ["P20929"]
        preferred_genes = ["NEB"]
        acc_hint = "accession:P20929 OR " if (organism_taxid == 9606 or not has_any_org) else ""
        specific_field_clause = f'({acc_hint}gene_exact:NEB OR gene:NEB OR protein_name:"Nebulin")'
    elif pq_low in {"apolipoprotein b-100", "apolipoprotein b 100", "apolipoprotein b", "apob-100", "apob100", "apob"}:
        if organism_taxid == 9606 or not has_any_org:
            preferred_accs = ["P04114"]
        preferred_genes = ["APOB"]
        acc_hint = "accession:P04114 OR " if (organism_taxid == 9606 or not has_any_org) else ""
        specific_field_clause = f'({acc_hint}gene_exact:APOB OR protein_name:"Apolipoprotein B-100")'
    elif pq_low in {"mucin-16", "mucin 16", "muc16", "ca-125", "ca125"}:
        if organism_taxid == 9606 or not has_any_org:
            preferred_accs = ["Q8WXI7"]
        preferred_genes = ["MUC16"]
        acc_hint = "accession:Q8WXI7 OR " if (organism_taxid == 9606 or not has_any_org) else ""
        specific_field_clause = f'({acc_hint}gene_exact:MUC16 OR protein_name:"Mucin-16")'

    return preferred_accs, preferred_genes, specific_field_clause


def _normalize_mobile_text(raw: str) -> str:
    """Normalizes soft-keyboard Unicode spaces/zero-width chars/dashes/smart quotes into ASCII equivalents."""
    if not raw:
        return ""
    s = raw.replace("\u00a0", " ")
    s = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", s)
    s = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", s)
    s = re.sub(r"[\u2018\u2019]", "'", s)
    s = re.sub(r"[\u201c\u201d]", '"', s)
    return s


def extract_nl_query_components(prompt: str) -> Dict[str, Any]:
    """
    Public deterministic parser decomposing any plain-language protein or recombinant construct prompt into:
    - Target protein / gene query (guaranteed zero-collision against tag/cleavage/organism grammar)
    - Universal Tree-of-Life organism metadata (curated taxon IDs + general Latin binomial / viral / strain clauses)
    - Affinity tag, terminus ('N-terminal' / 'C-terminal' / 'N/A'), and protease cleavage site metadata.
    """
    prompt = _normalize_mobile_text(prompt)
    if not prompt or not prompt.strip():
        raise ValueError("Please enter a protein description (e.g., 'GST human' or 'Chicken lysozyme with 6xHis tag').")

    target_phrase, modifier_text, force_untagged = _split_target_and_modifiers(prompt)
    if force_untagged:
        tag_key, detected_tag, tag_position, cleavage_key, detected_cleavage = None, None, "N/A", None, None
    else:
        tag_key, detected_tag, tag_position, cleavage_key, detected_cleavage = _detect_tag_and_cleavage_keys(modifier_text)

    if (detected_tag or detected_cleavage) and not _has_real_protein_tokens(target_phrase):
        tag_key, detected_tag, tag_position, cleavage_key, detected_cleavage = None, None, "N/A", None, None
        modifier_text = ""
        target_phrase = " ".join(prompt.strip().split())

    org_info = _extract_organism_details(target_phrase)
    organism_taxid = org_info["organism_taxid"]
    organism_clade_taxid = org_info["organism_clade_taxid"]
    organism_taxids = org_info["organism_taxids"]
    organism_name = org_info["organism_name"]
    organism_raw = org_info["organism_raw"]
    organism_strain = org_info["organism_strain"]

    protein_query = _normalize_protein_query(org_info["remaining_text"])
    if not protein_query and (organism_taxid == 12227 or organism_clade_taxid == 12227) and "protease" in target_phrase.lower():
        protein_query = "tev protease"

    gene_hint, protein_name_hint = _split_gene_and_paren_protein(protein_query)
    lookup_key = gene_hint if gene_hint and not protein_name_hint else protein_query
    preferred_accs, preferred_genes, specific_clause = _get_canonical_preferences(
        lookup_key,
        organism_taxid if organism_taxid is not None else organism_clade_taxid,
        organism_name=organism_name
    )

    return {
        "prompt": prompt,
        "target_phrase": " ".join(target_phrase.split()),
        "modifier_text": modifier_text,
        "force_untagged": force_untagged,
        "protein_query": protein_query,
        "gene_hint": gene_hint,
        "protein_name_hint": protein_name_hint,
        "organism_taxid": organism_taxid,
        "organism_clade_taxid": organism_clade_taxid,
        "organism_taxids": organism_taxids,
        "organism_name": organism_name,
        "organism_raw": organism_raw,
        "organism_strain": organism_strain,
        "tag_key": tag_key,
        "tag_dict": detected_tag,
        "tag_detected": detected_tag["name"] if detected_tag else "None",
        "tag_position": tag_position if detected_tag else "N/A",
        "cleavage_key": cleavage_key,
        "cleavage_dict": detected_cleavage,
        "cleavage_detected": detected_cleavage["name"] if detected_cleavage else "None",
        "preferred_accessions": preferred_accs,
        "preferred_genes": preferred_genes,
        "specific_field_clause": specific_clause,
    }


def _match_offline_classroom_target(
    clean_query: str,
    organism_taxid: Optional[int] = None,
    organism_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Provides reliable reference matching for textbook & dual-meaning proteins when offline or for mature chains."""
    raw = clean_query.strip()
    if organism_taxid is None and organism_name is None:
        info = _extract_organism_details(raw)
        organism_taxid = info["organism_taxid"]
        organism_name = info["organism_name"]
        pq = _normalize_protein_query(info["remaining_text"]).lower()
    else:
        pq = _normalize_protein_query(raw).lower()

    def _org_compatible(expected_taxid: int, expected_sci: str) -> bool:
        if organism_taxid is not None and organism_taxid != expected_taxid:
            return False
        if organism_name is not None and expected_sci.lower() not in organism_name.lower() and organism_name.lower() not in expected_sci.lower():
            return False
        return True

    if pq in {"lysozyme", "lysozyme c", "egg white lysozyme", "egg-white lysozyme", "hewl"}:
        if organism_taxid == 9606:
            return dict(OFFLINE_UNIPROT_REFERENCES["P61626"])
        if _org_compatible(9031, "Gallus gallus"):
            p = CLASSROOM_PRESET_PROTEINS["lysozyme_native"]
            return {
                "accession": p["uniprot_id"],
                "protein_name": "Lysozyme C (Mature Egg-White Chain)",
                "organism": "Gallus gallus",
                "sequence": p["sequence"]
            }

    if pq in {"tev", "tev protease", "nia protease", "nia-pro"} or (
        organism_taxid == 12227 and pq in {"protease", "nuclear inclusion protein a"}
    ):
        if _org_compatible(12227, "Tobacco etch virus"):
            return dict(OFFLINE_UNIPROT_REFERENCES["P04517"])

    if pq in {"gst", "glutathione s-transferase", "glutathione s transferase", "gstp1", "gst-p1", "gst-pi"}:
        if (organism_taxid == 9606 or pq in {"gstp1", "gst-p1", "gst-pi"}) and _org_compatible(9606, "Homo sapiens"):
            return dict(OFFLINE_UNIPROT_REFERENCES["P09211"])
        if _org_compatible(6182, "Schistosoma japonicum"):
            return dict(OFFLINE_UNIPROT_REFERENCES["P08515"])
    if pq in {"gsta1", "gst-a1", "gst-alpha"} and _org_compatible(9606, "Homo sapiens"):
        return dict(OFFLINE_UNIPROT_REFERENCES["P08263"])
    if pq in {"gstm1", "gst-m1", "gst-mu"} and _org_compatible(9606, "Homo sapiens"):
        return dict(OFFLINE_UNIPROT_REFERENCES["P09488"])

    if pq in {"mbp", "male", "maltose-binding protein", "maltose binding protein"}:
        if organism_taxid == 9606 and pq == "mbp":
            return dict(OFFLINE_UNIPROT_REFERENCES["P02686"])
        if organism_taxid in (None, 83333, 562) and (organism_name is None or "escherichia" in organism_name.lower()):
            return dict(OFFLINE_UNIPROT_REFERENCES["P0AEX9"])
    if pq in {
        "if3", "if-3", "infc", "initiation factor 3", "initiation factor if3",
        "initiation factor if-3", "translation initiation factor if-3",
        "translation initiation factor if3", "translation initiation factor 3"
    } and (organism_taxid in (None, *_ECOLI_TAXIDS)) and (organism_name is None or "escherichia" in organism_name.lower()):
        return dict(OFFLINE_UNIPROT_REFERENCES["P0A707"])
    if "myelin basic" in pq and _org_compatible(9606, "Homo sapiens"):
        return dict(OFFLINE_UNIPROT_REFERENCES["P02686"])

    if pq in {"myc", "c-myc", "myc proto-oncogene", "myc proto-oncogene protein"} and _org_compatible(9606, "Homo sapiens"):
        return dict(OFFLINE_UNIPROT_REFERENCES["P01106"])
    if pq in {"p53", "tp53", "cellular tumor antigen p53", "tumor protein p53"} and _org_compatible(9606, "Homo sapiens"):
        return dict(OFFLINE_UNIPROT_REFERENCES["P04637"])
    if pq in {"thrombin", "prothrombin", "coagulation factor ii", "f2"} and _org_compatible(9606, "Homo sapiens"):
        return dict(OFFLINE_UNIPROT_REFERENCES["P00734"])

    if (pq == "gfp" or pq == "green fluorescent protein" or "green fluorescent" in pq) and _org_compatible(6100, "Aequorea victoria"):
        p = CLASSROOM_PRESET_PROTEINS["gfp"]
        return {
            "accession": p["uniprot_id"],
            "protein_name": "Green fluorescent protein",
            "organism": "Aequorea victoria",
            "sequence": p["sequence"]
        }
    if pq in {"carbonic anhydrase", "carbonic anhydrase 2", "carbonic anhydrase ii", "ca2", "cah2"} and _org_compatible(9606, "Homo sapiens"):
        p = CLASSROOM_PRESET_PROTEINS["carbonic_anhydrase"]
        return {
            "accession": p["uniprot_id"],
            "protein_name": "Carbonic anhydrase 2 (Mature Chain)",
            "organism": "Homo sapiens",
            "sequence": p["sequence"]
        }
    if pq == "myoglobin" and _org_compatible(9755, "Physeter macrocephalus"):
        p = CLASSROOM_PRESET_PROTEINS["myoglobin"]
        return {
            "accession": p["uniprot_id"],
            "protein_name": "Myoglobin (Mature Chain)",
            "organism": "Physeter macrocephalus",
            "sequence": p["sequence"]
        }
    if pq in {"lipase a", "lipa"} and (organism_taxid in (None, 224308, 1423)) and (organism_name is None or "bacillus subtilis" in organism_name.lower()):
        p = CLASSROOM_PRESET_PROTEINS["lipase_a"]
        return {
            "accession": p["uniprot_id"],
            "protein_name": "Lipase A (Mature Chain)",
            "organism": "Bacillus subtilis",
            "sequence": p["sequence"]
        }
    if (pq == "bsa" or "bovine serum albumin" in raw.lower() or (pq == "serum albumin" and _org_compatible(9913, "Bos taurus"))):
        p = CLASSROOM_PRESET_PROTEINS["bsa"]
        return {
            "accession": p["uniprot_id"],
            "protein_name": "Bovine serum albumin (Mature Chain)",
            "organism": "Bos taurus",
            "sequence": p["sequence"]
        }
    if pq in {"titin", "ttn", "connectin"} and _org_compatible(9606, "Homo sapiens"):
        return dict(OFFLINE_UNIPROT_REFERENCES["Q8WZ42"])
    if pq in {"dystrophin", "dmd"} and _org_compatible(9606, "Homo sapiens"):
        return dict(OFFLINE_UNIPROT_REFERENCES["P11532"])
    if pq in {"urease", "jack bean urease", "plant urease", "urea amidohydrolase"} and _org_compatible(3823, "Canavalia ensiformis"):
        return dict(OFFLINE_UNIPROT_REFERENCES["P07374"])
    return None


def _sanitize_uniprot_phrase(phrase: str) -> str:
    """Replaces Lucene syntax metacharacters ('[', ']', '(', ')', ':', '/', '+', '--', quotes) with clean spaces."""
    cleaned = phrase.replace("--", " ")
    cleaned = re.sub(r'["\(\)\[\]\{\}:/\\+\*\?~\^\|<>]+', ' ', cleaned)
    return " ".join(cleaned.split()).strip()


def _normalize_org_prefix(org_text: str) -> str:
    """Normalizes organism nomenclature variants (e.g. optional 'type' in HIV-1/2 and 'sp.') for prefix/exact comparison."""
    norm = " ".join((org_text or "").lower().strip().split())
    norm = re.sub(r'\bhuman immunodeficiency virus type ([12])\b', r'human immunodeficiency virus \1', norm)
    norm = re.sub(r'\bsp\s+', 'sp. ', norm)
    return norm


def _has_subspecies_or_strain_spec(
    org_raw: Optional[str],
    org_name: Optional[str],
    org_strain: Optional[str]
) -> bool:
    """Returns True when organism_raw specifies a strain, serotype, pathovar, subspecies, or viral subtype beyond the bare species."""
    if org_strain and org_strain.strip():
        return True
    norm_raw = _normalize_org_prefix(org_raw or "")
    norm_name = _normalize_org_prefix(org_name or "")
    if not norm_raw or not norm_name or norm_raw == norm_name:
        return False
    if len(norm_raw.split()) >= 2 and norm_raw.startswith(norm_name + " "):
        return True
    return any(
        kw in norm_raw
        for kw in ("subsp.", "serotype", "serovar", "pathovar", "pv.", "biovar", "subtype", "group ", "strain ", "isolate ", "sp. ")
    )


_DUAL_ALPHANUM_TOK_RX = re.compile(r'^([A-Za-z]{2,8})-?([0-9]{1,3}[A-Za-z]?)$')
_DUAL_EF_TOK_MAP: Dict[str, Tuple[str, str]] = {
    "eftu": ("EFTu", "EF-Tu"),
    "ef-tu": ("EF-Tu", "EFTu"),
    "efts": ("EFTs", "EF-Ts"),
    "ef-ts": ("EF-Ts", "EFTs"),
    "efg": ("EFG", "EF-G"),
    "ef-g": ("EF-G", "EFG"),
}


def _dual_alphanum_variants(tok: str) -> List[str]:
    """
    Expands alphanumeric protein/gene shorthands into both hyphenless and hyphenated forms
    (e.g., 'IF3' -> ['IF3', 'IF-3'], 'IL-2' -> ['IL-2', 'IL2'], 'Bcl2' -> ['Bcl2', 'Bcl-2'],
    'EFTu' -> ['EFTu', 'EF-Tu'], 'EFG' -> ['EFG', 'EF-G']).
    """
    t = tok.strip()
    if not t:
        return []
    t_low = t.lower()
    if t_low in _DUAL_EF_TOK_MAP:
        return list(_DUAL_EF_TOK_MAP[t_low])
    m = _DUAL_ALPHANUM_TOK_RX.match(t)
    if m:
        p1, p2 = m.group(1), m.group(2)
        fused = f"{p1}{p2}"
        hyph = f"{p1}-{p2}"
        return [t] + [v for v in (fused, hyph) if v != t]
    return [t]


def _dual_phrase_variants(san_phrase: str) -> List[str]:
    """Generates hyphenated/de-hyphenated phrase counterparts when san_phrase contains an <ALPHA><NUM> shorthand."""
    if not san_phrase:
        return []
    words = san_phrase.split()
    has_dual = False
    fused_words: List[str] = []
    hyph_words: List[str] = []
    for w in words:
        w_low = w.lower()
        if w_low in _DUAL_EF_TOK_MAP:
            has_dual = True
            pair = _DUAL_EF_TOK_MAP[w_low]
            fused_words.append(pair[0] if "-" not in pair[0] else pair[1])
            hyph_words.append(pair[0] if "-" in pair[0] else pair[1])
        else:
            m = _DUAL_ALPHANUM_TOK_RX.match(w)
            if m:
                has_dual = True
                fused_words.append(f"{m.group(1)}{m.group(2)}")
                hyph_words.append(f"{m.group(1)}-{m.group(2)}")
            else:
                fused_words.append(w)
                hyph_words.append(w)
    if not has_dual:
        return [san_phrase]
    return list(dict.fromkeys([san_phrase, " ".join(hyph_words), " ".join(fused_words)]))


def _build_uniprot_search_plan(
    protein_query_or_components: Union[str, Dict[str, Any]],
    organism_taxid: Optional[int] = None
) -> Tuple[List[str], List[str], List[str]]:
    """
    Builds a tiered list of UniProtKB search queries supporting ALL domains of life:
    - Uses taxonomy_id:<clade_taxid> / organism_id:<taxid> AND organism_name:"<Genus species>" for arbitrary species!
    - Expands <ALPHA><NUM> <-> <ALPHA>-<NUM> across gene_exact, gene, and protein_name fields!
    - Sanitizes IUPAC enzyme brackets/parens so chemical names like hisA never trigger Lucene HTTP 400 errors.
    """
    if isinstance(protein_query_or_components, dict):
        comp = protein_query_or_components
        pq = comp.get("protein_query", "").strip()
        prim_taxid = comp.get("organism_taxid")
        clade_taxid = comp.get("organism_clade_taxid") or prim_taxid
        org_name = comp.get("organism_name")
        org_raw = comp.get("organism_raw")
        org_strain = comp.get("organism_strain")
        gene_hint = comp.get("gene_hint")
        pname_hint = comp.get("protein_name_hint")
        preferred_accs = list(comp.get("preferred_accessions") or [])
        preferred_genes = list(comp.get("preferred_genes") or [])
        specific_field_clause = comp.get("specific_field_clause")
    else:
        pq = protein_query_or_components.strip()
        prim_taxid = organism_taxid
        clade_taxid = organism_taxid
        org_name = None
        org_raw = None
        org_strain = None
        gene_hint, pname_hint = _split_gene_and_paren_protein(pq)
        preferred_accs, preferred_genes, specific_field_clause = _get_canonical_preferences(pq, prim_taxid)

    has_subspec = _has_subspecies_or_strain_spec(org_raw, org_name, org_strain)
    org_clauses: List[str] = []
    if has_subspec and org_raw:
        raw_for_query = org_raw
        if org_name and re.match(r'^[A-Za-z]{1,5}\.?\s+', org_raw) and not org_raw.lower().startswith(org_name.split()[0].lower()):
            raw_for_query = f"{org_name} {org_strain}" if org_strain else org_name
        san_raw_org = _sanitize_uniprot_phrase(raw_for_query)
        if san_raw_org:
            org_clauses.append(f' AND (organism_name:"{san_raw_org}")')

    if clade_taxid is not None and prim_taxid is not None:
        if prim_taxid == 83333 and clade_taxid == 562 and not has_subspec:
            org_clauses.append(" AND (organism_id:83333)")
            org_clauses.append(" AND (taxonomy_id:562)")
        elif prim_taxid != clade_taxid:
            org_clauses.append(f" AND (taxonomy_id:{clade_taxid} OR organism_id:{prim_taxid})")
        else:
            org_clauses.append(f" AND (taxonomy_id:{clade_taxid})")
    elif clade_taxid is not None:
        org_clauses.append(f" AND (taxonomy_id:{clade_taxid})")
    elif prim_taxid is not None:
        org_clauses.append(f" AND (taxonomy_id:{prim_taxid})")
    elif org_name:
        san_org = _sanitize_uniprot_phrase(org_name)
        if san_org:
            org_clauses.append(f' AND (organism_name:"{san_org}")')

    if not org_clauses:
        org_clauses = [""]

    primary_org = org_clauses[0]
    broad_org = org_clauses[-1]

    target_phrase_for_tokens = pname_hint if pname_hint else pq
    san_phrase = _sanitize_uniprot_phrase(target_phrase_for_tokens)
    raw_tokens = [w for w in re.split(r'[\s,;:/\(\)\[\]]+', san_phrase) if w]
    tokens = [
        w for w in raw_tokens
        if (len(raw_tokens) == 1 or w.lower() not in {"and", "or", "the", "of", "in"})
    ]
    informative_tokens = sorted(
        tokens,
        key=lambda t: (0 if (any(c.isalpha() for c in t) and len(t) >= 3) else 1)
    )[:6]

    queries: List[str] = []

    def _add_q(q_str: str) -> None:
        if q_str and q_str not in queries:
            queries.append(q_str)

    wants_fragment = "fragment" in pq.lower()
    rev_prefixes = ["(reviewed:true)"] if wants_fragment else ["(reviewed:true) AND NOT (fragment:true)", "(reviewed:true)"]

    for rev_pref in rev_prefixes:
        if specific_field_clause:
            _add_q(f"{rev_pref}{primary_org} AND {specific_field_clause}")
            if broad_org != primary_org:
                _add_q(f"{rev_pref}{broad_org} AND {specific_field_clause}")
        if gene_hint and pname_hint:
            san_gene = _sanitize_uniprot_phrase(gene_hint)
            san_pname = _sanitize_uniprot_phrase(pname_hint)
            gene_vars = _dual_alphanum_variants(san_gene)
            pname_vars = _dual_phrase_variants(san_pname)
            g_clause = " OR ".join(dict.fromkeys([f'gene_exact:"{gv}"' for gv in gene_vars] + [f'gene:"{gv}"' for gv in gene_vars]))
            p_clause = " OR ".join(f'protein_name:"{pv}"' for pv in pname_vars)
            _add_q(f"{rev_pref}{primary_org} AND ({g_clause}) AND ({p_clause})")
            _add_q(f"{rev_pref}{broad_org} AND ({g_clause} OR {p_clause})")
        elif len(tokens) == 1:
            tok = tokens[0]
            tok_vars = _dual_alphanum_variants(tok)
            tok_field_parts: List[str] = []
            for tv in tok_vars:
                tok_field_parts.append(f'gene_exact:"{tv}"')
                tok_field_parts.append(f'gene:"{tv}"')
                tok_field_parts.append(f'protein_name:"{tv}"')
            joined_tok_clause = " OR ".join(dict.fromkeys(tok_field_parts))
            _add_q(f"{rev_pref}{primary_org} AND ({joined_tok_clause})")
            if broad_org != primary_org:
                _add_q(f"{rev_pref}{broad_org} AND ({joined_tok_clause})")
        elif san_phrase and not specific_field_clause:
            phrase_vars = _dual_phrase_variants(san_phrase)
            pname_clause = " OR ".join(f'protein_name:"{pv}"' for pv in phrase_vars)
            _add_q(f"{rev_pref}{primary_org} AND ({pname_clause})")
            if broad_org != primary_org:
                _add_q(f"{rev_pref}{broad_org} AND ({pname_clause})")

        if len(informative_tokens) >= 2:
            per_tok_parts: List[str] = []
            for t in informative_tokens:
                t_vars = _dual_alphanum_variants(t)
                inner_or = " OR ".join(dict.fromkeys([f'protein_name:"{tv}"' for tv in t_vars] + [f'gene:"{tv}"' for tv in t_vars]))
                per_tok_parts.append(f"({inner_or})")
            per_tok = " AND ".join(per_tok_parts)
            _add_q(f"{rev_pref}{broad_org} AND ({per_tok})")

        if informative_tokens:
            tok_and_parts: List[str] = []
            for t in informative_tokens:
                t_vars = _dual_alphanum_variants(t)
                if len(t_vars) > 1:
                    tok_and_parts.append("(" + " OR ".join(f'"{tv}"' for tv in t_vars) + ")")
                else:
                    tok_and_parts.append(f'"{t}"')
            tok_and = " AND ".join(tok_and_parts)
            _add_q(f"{rev_pref}{broad_org} AND ({tok_and})")

    unrev_org_prefix = broad_org.replace(" AND ", "", 1) + " AND " if broad_org else ""
    if specific_field_clause:
        _add_q(f"{unrev_org_prefix}{specific_field_clause}")
    elif san_phrase:
        tok_q_parts: List[str] = []
        for t in informative_tokens[:4]:
            t_vars = _dual_alphanum_variants(t)
            tok_q_parts.append("(" + " OR ".join(f'"{tv}"' for tv in t_vars) + ")" if len(t_vars) > 1 else f'"{t}"')
        tok_q = " AND ".join(tok_q_parts)
        phrase_vars = _dual_phrase_variants(san_phrase)
        pname_unrev = " OR ".join(f'protein_name:"{pv}"' for pv in phrase_vars)
        base_unrev = f"({pname_unrev} OR ({tok_q}))" if informative_tokens else f"({pname_unrev})"
        _add_q(f"{unrev_org_prefix}{base_unrev}")

    return queries, preferred_accs, preferred_genes


_ALPHANUM_NORM_FUSE_RX = re.compile(r'\b([a-z]{2,8})\s+([0-9]{1,3}[a-z]?)\b')
_EF_NORM_FUSE_RX = re.compile(r'\bef\s+(tu|ts|g)\b')


def _normalize_comparison_name(name: str) -> str:
    """Normalizes protein name punctuation/hyphens/whitespace for exact biological identity comparison."""
    s = name.lower().strip()
    s = re.sub(r'[\s\-–—_/]+', ' ', s)
    s = re.sub(r'^[\(\[\'"]+|[\)\]\'".,;:]+$', '', s)
    return " ".join(s.split())


def _fuse_alphanum_comparison_name(norm: str) -> str:
    """Fuses de-hyphenated '<alpha> <num>' and 'ef (tu|ts|g)' pairs (e.g. 'if 3' -> 'if3', 'ef tu' -> 'eftu')."""
    if not norm:
        return norm
    fused = _ALPHANUM_NORM_FUSE_RX.sub(r'\1\2', norm)
    fused = _EF_NORM_FUSE_RX.sub(r'ef\1', fused)
    return fused


def _extract_fused_alphanum_tokens(text: str) -> Dict[str, Tuple[str, str]]:
    """
    Extracts fused alphanumeric tokens from hyphenated or adjacent '<alpha2..8>[- ]<num1..3[a-z]>' and 'ef[- ](tu|ts|g)'
    pairs in text, mapping {fused_token: (part1, part2)} (e.g., {'if3': ('if', '3'), 'eftu': ('ef', 'tu')}).
    """
    norm = _normalize_comparison_name(text)
    if not norm:
        return {}
    out: Dict[str, Tuple[str, str]] = {}
    for m in _ALPHANUM_NORM_FUSE_RX.finditer(norm):
        p1, p2 = m.group(1), m.group(2)
        out[f"{p1}{p2}"] = (p1, p2)
    for m in _EF_NORM_FUSE_RX.finditer(norm):
        p2 = m.group(1)
        out[f"ef{p2}"] = ("ef", p2)
    return out


def _tokenize_for_matching(text: str) -> List[str]:
    """Splits a protein/gene string into lowercase alphanumeric tokens (preserving subtype tokens like 'c', 'a', '2', 'h3')."""
    return [
        t for t in re.split(r'[^a-z0-9]+', text.lower())
        if t and t not in {"the", "and", "or", "of", "in", "from"}
    ]


def _extract_candidate_view(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes BOTH full UniProt REST API JSON entries AND compact corpus benchmark entries
    (from data/nl_10k_proteins.json) into a unified dictionary for deterministic scoring.
    """
    accession = str(entry.get("primaryAccession") or entry.get("accession") or "")

    if "reviewed" in entry:
        is_reviewed = bool(entry["reviewed"])
    elif "entryType" in entry:
        et_low = str(entry.get("entryType") or "").lower()
        is_reviewed = ("reviewed" in et_low) and ("unreviewed" not in et_low)
    else:
        is_reviewed = True

    annot_raw = entry.get("annotationScore", entry.get("annotation_score", 0.0))
    try:
        annotation_score = float(annot_raw or 0.0)
    except (TypeError, ValueError):
        annotation_score = 0.0

    pdesc = entry.get("proteinDescription") or {}
    seq_block = entry.get("sequence") if isinstance(entry.get("sequence"), dict) else {}
    if "is_fragment" in entry:
        is_fragment = bool(entry["is_fragment"])
    else:
        flag_val = str(pdesc.get("flag") or "").lower()
        flags_list = [str(f).lower() for f in (pdesc.get("flags") or [])]
        seq_frag = str(seq_block.get("fragment") or "").lower()
        is_fragment = (
            "fragment" in flag_val
            or any("fragment" in f for f in flags_list)
            or bool(seq_frag and seq_frag not in {"false", "0", "none"})
        )

    org_block = entry.get("organism") if isinstance(entry.get("organism"), dict) else {}
    raw_tax = org_block.get("taxonId") if org_block else entry.get("tax_id")
    tax_id: Optional[int] = int(raw_tax) if raw_tax is not None and str(raw_tax).isdigit() else None
    org_sci = str(org_block.get("scientificName") or entry.get("organism_scientific") or "")
    org_common = str(org_block.get("commonName") or entry.get("organism_common") or "")
    lineage = [str(x).lower() for x in (org_block.get("lineage") or entry.get("lineage") or [])]

    rec_name = str(
        pdesc.get("recommendedName", {}).get("fullName", {}).get("value")
        or ((pdesc.get("submissionNames") or [{}])[0].get("fullName", {}).get("value") if pdesc.get("submissionNames") else "")
        or entry.get("recommended_name")
        or entry.get("protein_name")
        or ""
    )

    short_names: List[str] = list(entry.get("short_names") or [])
    for sn in (pdesc.get("recommendedName", {}).get("shortNames") or []):
        if sn.get("value"):
            short_names.append(str(sn["value"]))

    alt_names: List[str] = list(entry.get("alt_names") or [])
    for alt in (pdesc.get("alternativeNames") or []):
        if alt.get("fullName", {}).get("value"):
            alt_names.append(str(alt["fullName"]["value"]))
        for sn in (alt.get("shortNames") or []):
            if sn.get("value"):
                short_names.append(str(sn["value"]))

    component_names: List[str] = []
    for section_key in ("contains", "includes"):
        for comp_item in (pdesc.get(section_key) or []):
            c_full = comp_item.get("recommendedName", {}).get("fullName", {}).get("value")
            if c_full:
                component_names.append(str(c_full))
            for c_alt in (comp_item.get("alternativeNames") or []):
                if c_alt.get("fullName", {}).get("value"):
                    component_names.append(str(c_alt["fullName"]["value"]))

    gene_primary = str(entry.get("gene_primary") or "")
    gene_synonyms: List[str] = [str(s) for s in (entry.get("gene_synonyms") or []) if s]
    genes_list = entry.get("genes") or []
    if genes_list:
        first_g = genes_list[0].get("geneName", {}).get("value")
        if first_g and not gene_primary:
            gene_primary = str(first_g)
        for idx, g_obj in enumerate(genes_list):
            g_val = g_obj.get("geneName", {}).get("value")
            if idx > 0 and g_val:
                gene_synonyms.append(str(g_val))
            for syn in (g_obj.get("synonyms") or []):
                if syn.get("value"):
                    gene_synonyms.append(str(syn["value"]))
            for oln in (g_obj.get("orderedLocusNames") or []):
                if oln.get("value"):
                    gene_synonyms.append(str(oln["value"]))
            for orf in (g_obj.get("orfNames") or []):
                if orf.get("value"):
                    gene_synonyms.append(str(orf["value"]))

    return {
        "accession": accession,
        "is_reviewed": is_reviewed,
        "annotation_score": annotation_score,
        "is_fragment": is_fragment,
        "tax_id": tax_id,
        "org_sci": org_sci,
        "org_common": org_common,
        "lineage": lineage,
        "rec_name": rec_name,
        "short_names": short_names,
        "alt_names": alt_names,
        "component_names": component_names,
        "gene_primary": gene_primary,
        "gene_synonyms": gene_synonyms,
    }


_DECOY_MODIFIER_PENALTY_WORDS = {
    "binding", "like", "receptor", "inhibitor", "inhibitory", "chaperone",
    "regulatory", "regulator", "regulated", "interacting", "associated",
    "homolog", "paralog", "subunit", "chain", "component", "activating",
    "activator", "induced", "inducible", "related", "domain", "containing",
    "putative", "probable", "uncharacterized", "pseudogene", "pull", "pulldown",
    "antagonist", "accessory", "auxiliary", "assembly", "maturation", "processing",
}


def score_uniprot_candidate(
    entry: Dict[str, Any],
    parsed_components: Union[str, Dict[str, Any]]
) -> float:
    """
    Deterministic multi-factor candidate scorer across all domains of life:
    (a) Organism match (exact tax_id / strain / species / common name / lineage vs mismatch penalty)
    (b) Primary & synonym gene symbol match (with case-exact & primary-over-synonym priority)
    (c) Exact normalized recommendedName / shortName / alternativeName / component chain match
    (d) Full token coverage, extra-word count penalty, and decoy-word penalty (-binding, -like, receptor, inhibitor, etc.)
    (e) Reviewed Swiss-Prot status & annotationScore
    (f) Non-fragment preference
    """
    comp = (
        extract_nl_query_components(parsed_components)
        if isinstance(parsed_components, str)
        else parsed_components
    )
    cand = _extract_candidate_view(entry)

    score = 0.0

    # (e) Reviewed Swiss-Prot & annotationScore (1..5) + (f) Non-fragment
    score += 220.0 if cand["is_reviewed"] else -220.0
    score += cand["annotation_score"] * 8.0

    pq_raw = comp.get("protein_query") or ""
    pq_low = pq_raw.strip().lower()
    if cand["is_fragment"] and "fragment" not in pq_low:
        score -= 135.0

    # (a) Organism match scoring
    req_prim_taxid = comp.get("organism_taxid")
    req_clade_taxid = comp.get("organism_clade_taxid")
    req_taxids = set(comp.get("organism_taxids") or ([] if req_prim_taxid is None else [req_prim_taxid]))
    if req_clade_taxid is not None:
        req_taxids.add(req_clade_taxid)
    req_org_name = (comp.get("organism_name") or "").strip()
    req_org_raw = (comp.get("organism_raw") or "").strip()
    req_strain = (comp.get("organism_strain") or "").strip()
    has_subspec = _has_subspecies_or_strain_spec(req_org_raw, req_org_name, req_strain)

    has_org_constraint = (req_prim_taxid is not None) or (req_clade_taxid is not None) or bool(req_org_name)
    if has_org_constraint:
        cand_tax = cand["tax_id"]
        cand_sci_low = cand["org_sci"].lower()
        cand_com_low = cand["org_common"].lower()
        req_name_low = req_org_name.lower()
        norm_cand_sci = _normalize_org_prefix(cand["org_sci"])
        norm_req_raw = _normalize_org_prefix(req_org_raw)
        norm_req_name = _normalize_org_prefix(req_org_name)

        org_matched = False
        if norm_req_raw and norm_cand_sci == norm_req_raw:
            org_matched = True
            score += 470.0 if has_subspec else 345.0
        elif (
            norm_req_raw
            and (has_subspec or (req_prim_taxid is None and req_clade_taxid is None))
            and (norm_cand_sci.startswith(norm_req_raw + " ") or norm_cand_sci.startswith(norm_req_raw + "("))
        ):
            org_matched = True
            score += 445.0 if has_subspec else 335.0
        else:
            # Check explicit strain token containment (e.g. 'Halobacterium sp. AS7092' vs 'Halobacterium sp. (strain AS7092)')
            strain_toks = [
                t for t in re.split(r'[^a-z0-9]+', req_strain.lower())
                if t and t not in {"strain", "str", "isolate", "serotype", "serovar", "subsp", "var", "pv", "biovar", "group", "subtype", "genotype", "type"}
            ]
            cand_org_toks = set(re.split(r'[^a-z0-9]+', norm_cand_sci))
            genus_ok = bool(norm_req_name) and norm_cand_sci.startswith(norm_req_name.split()[0] + " ")
            if strain_toks and genus_ok and all(st in cand_org_toks for st in strain_toks):
                org_matched = True
                score += 430.0
            elif cand_tax is not None and cand_tax in req_taxids:
                org_matched = True
                score += 315.0 if (cand_tax == req_prim_taxid and not has_subspec) else 300.0
            elif req_name_low:
                if (
                    norm_cand_sci == norm_req_name
                    or norm_cand_sci.startswith(norm_req_name + " ")
                    or norm_cand_sci.startswith(norm_req_name + "(")
                    or (norm_req_raw and (norm_cand_sci.startswith(norm_req_raw + " ") or norm_cand_sci.startswith(norm_req_raw + "(")))
                    or (cand_com_low and cand_com_low == req_name_low)
                    or any(lin == req_name_low for lin in cand["lineage"])
                ):
                    org_matched = True
                    score += 300.0
                else:
                    req_words = norm_req_name.split()
                    if len(req_words) >= 2 and req_words[1] in ("sp.", "sp") and norm_cand_sci.startswith(req_words[0] + " "):
                        org_matched = True
                        score += 290.0
                    elif len(req_words) >= 2 and re.search(r'\b' + re.escape(req_words[1]) + r'\b', cand_sci_low):
                        org_matched = True
                        score += 275.0

        if not org_matched:
            score -= 500.0

    # Preferred canonical accessions & genes
    pref_accs = comp.get("preferred_accessions") or []
    pref_genes = [g.upper() for g in (comp.get("preferred_genes") or [])]
    if cand["accession"] in pref_accs:
        score += 160.0 - pref_accs.index(cand["accession"]) * 8.0

    cand_gp = cand["gene_primary"]
    cand_gp_low = cand_gp.lower()
    cand_syns_low = [s.lower() for s in cand["gene_synonyms"]]
    all_genes_upper = ([cand_gp.upper()] if cand_gp else []) + [s.upper() for s in cand["gene_synonyms"]]
    matched_pref_primary_gene = bool(pref_genes and cand_gp and cand_gp.upper() in pref_genes)
    if matched_pref_primary_gene:
        score += 185.0
    elif pref_genes and any(g in pref_genes for g in all_genes_upper):
        score += 130.0

    gene_hint = comp.get("gene_hint")
    pname_hint = comp.get("protein_name_hint")
    effective_name_query = pname_hint if pname_hint else pq_raw
    eq_low = effective_name_query.strip().lower()
    eq_norm = _normalize_comparison_name(effective_name_query)
    eq_no_prot = _normalize_comparison_name(re.sub(r'\bprotein\b', '', eq_low).strip())

    # (b) Exact Gene Symbol Matching
    gene_targets = [gene_hint] if gene_hint else ([pq_raw.strip()] if " " not in pq_raw.strip() else [])
    matched_primary_gene = False
    for gt in gene_targets:
        gt_clean = gt.strip()
        gt_low = gt_clean.lower()
        gt_dehyph = gt_low.replace("-", "")
        if cand_gp_low and (cand_gp_low == gt_low or cand_gp_low.replace("-", "") == gt_dehyph):
            score += 240.0
            if cand_gp == gt_clean:
                score += 15.0
            matched_primary_gene = True
            break
        elif any(s == gt_low or s.replace("-", "") == gt_dehyph for s in cand_syns_low):
            score += 175.0
            break

    # (c) Exact Normalized Protein Name Matching (strict hierarchy: recommendedName > shortName > alternativeName > cleaved chain)
    rec_low = cand["rec_name"].lower().strip()
    rec_norm = _normalize_comparison_name(cand["rec_name"])
    rec_fused_norm = _fuse_alphanum_comparison_name(rec_norm)

    def _expand_norm_set(items: List[str]) -> Set[str]:
        out_set: Set[str] = set()
        for item_str in items:
            if not item_str:
                continue
            base_n = _normalize_comparison_name(item_str)
            if base_n:
                out_set.add(base_n)
                out_set.add(_fuse_alphanum_comparison_name(base_n))
        return out_set

    short_norms = _expand_norm_set(cand["short_names"])
    alt_norms = _expand_norm_set(cand["alt_names"])
    comp_norms = _expand_norm_set(cand["component_names"])

    raw_query_norms = {v for v in (eq_norm, eq_no_prot) if v}
    query_norm_variants = {
        fv
        for base_q in raw_query_norms
        for fv in (base_q, _fuse_alphanum_comparison_name(base_q))
        if fv
    }
    is_strict_rec = (rec_low == eq_low) or (rec_norm in raw_query_norms)
    is_fused_rec = (not is_strict_rec) and bool(rec_fused_norm and rec_fused_norm in query_norm_variants)
    is_exact_rec = is_strict_rec or is_fused_rec
    is_exact_short = bool(query_norm_variants & short_norms)
    is_exact_alt = bool(query_norm_variants & alt_norms)
    is_exact_comp = bool(query_norm_variants & comp_norms)
    has_any_exact_name = is_exact_rec or is_exact_short or is_exact_alt or is_exact_comp

    if rec_low == eq_low and eq_low:
        score += 340.0
    elif is_strict_rec:
        score += 325.0
    elif is_fused_rec:
        score += 320.0
    elif is_exact_short:
        score += 245.0
    elif is_exact_alt:
        score += 225.0
    elif is_exact_comp:
        score += 195.0

    # (d) Token Coverage, Extra-Word Penalty & Adversarial Decoy Penalty (only when not already matched as an exact name!)
    q_tokens = _tokenize_for_matching(effective_name_query)
    rec_tokens = _tokenize_for_matching(cand["rec_name"])
    rec_token_set = set(rec_tokens)
    rec_fused_map = _extract_fused_alphanum_tokens(cand["rec_name"])
    q_fused_map = _extract_fused_alphanum_tokens(effective_name_query)
    all_cand_tokens = set(rec_tokens) | set(rec_fused_map.keys())
    for extra_str in cand["short_names"] + cand["alt_names"] + cand["component_names"] + [cand_gp] + cand["gene_synonyms"]:
        all_cand_tokens.update(_tokenize_for_matching(extra_str))
        all_cand_tokens.update(_extract_fused_alphanum_tokens(extra_str).keys())

    if q_tokens and not has_any_exact_name:
        q_unique = list(dict.fromkeys(q_tokens))
        covered_rec_tokens: Set[str] = set()

        # Check reverse fusion when query had hyphenated tokens ('IF-3' -> 'if', '3') and candidate had fused ('IF3')
        q_reverse_matched: Set[str] = set()
        q_reverse_anywhere: Set[str] = set()
        for q_fused_tok, (qp1, qp2) in q_fused_map.items():
            if q_fused_tok in rec_token_set:
                q_reverse_matched.update((qp1, qp2))
                covered_rec_tokens.add(q_fused_tok)
            if q_fused_tok in all_cand_tokens:
                q_reverse_anywhere.update((qp1, qp2))

        matched_in_rec = 0
        matched_anywhere = 0
        for t in q_unique:
            if t in rec_token_set:
                matched_in_rec += 1
                covered_rec_tokens.add(t)
            elif t in rec_fused_map:
                matched_in_rec += 1
                covered_rec_tokens.update(rec_fused_map[t])
            elif t in q_reverse_matched:
                matched_in_rec += 1

            if t in all_cand_tokens or t in q_reverse_anywhere:
                matched_anywhere += 1

        if matched_in_rec == len(q_unique):
            score += 120.0
        elif matched_anywhere == len(q_unique):
            score += 95.0
        else:
            score += 85.0 * (matched_anywhere / len(q_unique))
            if not matched_primary_gene and not (matched_pref_primary_gene and len(q_unique) <= 2):
                score -= 45.0 * (len(q_unique) - matched_anywhere)

        if not (matched_primary_gene and not pname_hint and len(q_unique) == 1):
            extra_rec_tokens = [t for t in rec_token_set if t not in (set(q_unique) | covered_rec_tokens)]
            score -= 12.0 * len(extra_rec_tokens)
            shared_excess = sum(max(0, rec_tokens.count(t) - q_tokens.count(t)) for t in (rec_token_set & set(q_unique)))
            score -= 15.0 * shared_excess

    if not is_exact_rec and not (matched_primary_gene and not pname_hint):
        q_tok_list = _tokenize_for_matching(pname_hint if pname_hint else pq_raw)
        for p_term in _DECOY_MODIFIER_PENALTY_WORDS:
            if p_term == "binding" and pq_low in {"mbp", "protein a", "staphylococcal protein a", "protein g"}:
                continue
            if rec_tokens.count(p_term) > q_tok_list.count(p_term):
                score -= 75.0

    return score


def _rank_uniprot_hit(
    hit: Dict[str, Any],
    protein_query: str,
    preferred_accs: List[str],
    preferred_genes: List[str]
) -> float:
    """Backwards-compatible wrapper delegating to score_uniprot_candidate."""
    comp = {
        "protein_query": protein_query,
        "gene_hint": None,
        "protein_name_hint": None,
        "organism_taxid": None,
        "organism_taxids": [],
        "organism_name": None,
        "organism_raw": None,
        "preferred_accessions": preferred_accs,
        "preferred_genes": preferred_genes,
    }
    return score_uniprot_candidate(hit, comp)


def rank_uniprot_candidates(
    entries: List[Dict[str, Any]],
    parsed_components: Union[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Deterministically ranks candidate protein entries using score_uniprot_candidate with strict,
    order-independent tie-breaking: (-score, -annotation_score, is_fragment, primaryAccession).
    Guarantees 100.0% reproducible ranking regardless of input list permutation.
    """
    if not entries:
        return []
    comp = (
        extract_nl_query_components(parsed_components)
        if isinstance(parsed_components, str)
        else parsed_components
    )

    def _sort_key(entry: Dict[str, Any]) -> Tuple[float, float, bool, str]:
        sc = round(score_uniprot_candidate(entry, comp), 6)
        view = _extract_candidate_view(entry)
        return (-sc, -view["annotation_score"], view["is_fragment"], view["accession"])

    return sorted(entries, key=_sort_key)


def _candidate_matches_requested_protein(hit: Dict[str, Any], comp: Dict[str, Any]) -> bool:
    """
    Semantic guard preventing loose UniProt full-text fallbacks from silently returning a completely
    unrelated protein (e.g., returning Zn2+ Dihydropyrimidinase or Protein SIS1 when a user asks for
    'Human urease' or 'Yeast urease', which those organisms do not encode).
    """
    cand = _extract_candidate_view(hit)
    pref_accs = set(comp.get("preferred_accessions") or [])
    if cand["accession"] and cand["accession"] in pref_accs:
        return True

    pref_genes = {g.upper() for g in (comp.get("preferred_genes") or []) if g}
    cand_genes = {g.upper() for g in ([cand["gene_primary"]] + cand["gene_synonyms"]) if g}
    if pref_genes and (pref_genes & cand_genes):
        return True

    pq_raw = comp.get("protein_query") or ""
    gene_hint = comp.get("gene_hint")
    pname_hint = comp.get("protein_name_hint")
    eff_q = pname_hint if pname_hint else pq_raw

    if gene_hint:
        gh_u = gene_hint.strip().upper()
        if any(g == gh_u or g.replace("-", "") == gh_u.replace("-", "") for g in cand_genes):
            return True

    # Gather all candidate name & gene tokens (including fused alphanumeric tokens)
    cand_text_fields = (
        [cand["rec_name"]]
        + cand["short_names"]
        + cand["alt_names"]
        + cand["component_names"]
        + ([cand["gene_primary"]] if cand["gene_primary"] else [])
        + cand["gene_synonyms"]
    )
    cand_tokens: Set[str] = set()
    cand_joined_low = " ".join(s.lower() for s in cand_text_fields if s)
    for s in cand_text_fields:
        if s:
            cand_tokens.update(_tokenize_for_matching(s))
            cand_tokens.update(_extract_fused_alphanum_tokens(s).keys())

    # Require at least one substantive protein query token (or gene token) to match the candidate's names/genes
    generic_skip = {"protein", "enzyme", "subunit", "chain", "factor", "domain", "bacterial", "plant", "fungal", "viral", "alpha", "beta", "gamma"}
    q_toks = [t for t in _tokenize_for_matching(eff_q) if t not in generic_skip and len(t) >= 2]
    if not q_toks:
        q_toks = _tokenize_for_matching(eff_q)
    if not q_toks:
        return True

    for qt in q_toks:
        if qt in cand_tokens:
            return True
        if len(qt) >= 4 and qt in cand_joined_low:
            return True
    # Also check fused query tokens (e.g. 'if-3' -> 'if3')
    for qf in _extract_fused_alphanum_tokens(eff_q).keys():
        if qf in cand_tokens:
            return True
    return False


_ENGLISH_PROTEIN_SUFFIXES = (
    "TRANSFERASE", "SYNTHETASE", "HYDROLASE", "PHOSPHATASE", "REDUCTASE",
    "DEHYDROGENASE", "POLYMERASE", "ISOMERASE", "CARBOXYLASE", "ESTERASE",
    "PEPTIDASE", "PROTEASE", "KINASE", "LIGASE", "LYASE", "SYNTHASE",
    "CYCLASE", "HELICASE", "NUCLEASE", "TRANSCRIPTASE", "INTEGRASE",
    "TRANSPOSASE", "RECOMBINASE", "PERMEASE", "TRANSPORTER", "RECEPTOR",
    "INHIBITOR", "ACTIVATOR", "REGULATOR", "PROTEIN", "PEPTIDE", "ENZYME",
    "ANTIBODY", "ANTIGEN", "DOMAIN", "SUBUNIT", "PRECURSOR", "ISOFORM",
    "LYSOZYME", "ALBUMIN", "MYOGLOBIN", "HEMOGLOBIN", "INSULIN", "COLLAGEN",
    "KERATIN", "FERRITIN", "CATALASE", "UREASE", "AMYLASE", "LIPASE",
    "TRYPSIN", "PEPSIN", "THROMBIN", "UBIQUITIN", "CALMODULIN", "CYTOCHROME",
    "FERREDOXIN", "THIOREDOXIN", "GLYCERALDEHYDE", "METALLOTHIONEIN",
    "COLLAGENASE", "DYSTROPHIN",
)

_INLINE_PROSE_STOPWORDS = frozenset({
    "with", "from", "and", "for", "the", "plus", "add", "without", "carrying", "having", "bearing",
    "tag", "tags", "tagged", "site", "sites", "cleavage", "terminal", "terminus", "fused", "fusion",
    "his", "strep", "flag", "myc", "gst", "mbp", "sumo", "tev", "hrv", "prescission", "thrombin",
    "enterokinase", "human", "chicken", "mouse", "yeast", "coli", "ecoli", "rat", "bovine",
    "protein", "peptide", "enzyme", "chain", "strand", "designed", "novo", "synthetic",
    "serine", "threonine", "tyrosine", "methyl", "acetyl", "phosphoryl",
})


def _is_likely_english_word_not_aa(token_upper: str) -> bool:
    """Returns True if an alphabetic token matches common English biochemistry suffixes."""
    t = (token_upper or "").strip().upper()
    if not t:
        return False
    return any(t.endswith(sfx) for sfx in _ENGLISH_PROTEIN_SUFFIXES)


def _check_inline_dna_guard(letters_upper: str) -> None:
    """Raises a helpful ValueError if an inline sequence is purely canonical A/C/G/T DNA (>= 30 nt)."""
    if len(letters_upper) >= 30 and set(letters_upper) <= {"A", "C", "G", "T"}:
        raise ValueError(
            "This looks like a DNA sequence (only A/C/G/T bases). "
            "Switch to '🧬 Translate DNA / RNA' to translate it into protein."
        )


def _is_sequence_like_line(line: str, allow_lowercase_single_token: bool = True) -> Tuple[bool, str]:
    """Checks whether a line contains valid amino acid sequence characters rather than English prose."""
    s = (line or "").strip()
    if not s or not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy0-9 \t*-]+", s):
        return False, ""
    words = [w for w in re.split(r"\s+", s) if not re.fullmatch(r"\d+", w)]
    if not words:
        return False, ""
    for w in words:
        w_alpha = re.sub(r"[^A-Za-z]", "", w)
        if not w_alpha:
            continue
        if w_alpha.lower() in _INLINE_PROSE_STOPWORDS or _is_likely_english_word_not_aa(w_alpha):
            return False, ""
    has_lower = any(ch.islower() for ch in s)
    letters = re.sub(r"[^A-Za-z]", "", s)
    if len(letters) < 5:
        return False, ""
    if has_lower:
        if not allow_lowercase_single_token or len(words) > 1 or len(letters) < 20:
            return False, ""
    return True, letters.upper()


def _extract_inline_sequence_from_prompt(prompt: str) -> Optional[Dict[str, Any]]:
    """
    Detects raw FASTA, multi-line wrapped, or inline de novo / custom amino acid sequences
    inside the plain-English construct box (optionally paired with N-/C-terminal tag/cleavage instructions).
    Returns None for normal English protein queries.
    """
    raw = _normalize_mobile_text(prompt).strip()
    if not raw:
        return None

    lines = raw.splitlines()

    # Path A: FASTA header line ('>...')
    header_idx = next((idx for idx, ln in enumerate(lines) if ln.lstrip().startswith(">")), None)
    if header_idx is not None:
        header_raw = lines[header_idx].lstrip()[1:].strip()
        seq_parts: List[str] = []
        modifier_lines: List[str] = []
        if header_idx > 0:
            modifier_lines.extend(lines[:header_idx])
        reading_seq = True
        for ln in lines[header_idx + 1:]:
            if not ln.strip():
                if seq_parts:
                    reading_seq = False
                continue
            if ln.lstrip().startswith(">"):
                reading_seq = False
                continue
            is_seq_ln, seq_chunk = _is_sequence_like_line(ln, allow_lowercase_single_token=True)
            if reading_seq and is_seq_ln:
                seq_parts.append(seq_chunk)
            else:
                reading_seq = False
                modifier_lines.append(ln.strip())

        seq_letters = "".join(seq_parts)
        if len(seq_letters) >= 10 and re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", seq_letters):
            _check_inline_dna_guard(seq_letters)
            return {
                "sequence": seq_letters,
                "header": header_raw or None,
                "modifier_prose": " ".join(x for x in modifier_lines if x.strip()).strip(),
            }

    # Path B: Whole prompt is pure amino acid letters (single-line or wrapped)
    letters_only = re.sub(r"[\s0-9*\-]+", "", raw)
    if letters_only and re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy]+", letters_only):
        upper_seq = letters_only.upper()
        words = [re.sub(r"[^A-Za-z]", "", w) for w in re.split(r"\s+", raw) if re.sub(r"[^A-Za-z]", "", w)]
        has_lower = any(ch.islower() for ch in letters_only)
        has_prose_word = any(
            w.lower() in _INLINE_PROSE_STOPWORDS or _is_likely_english_word_not_aa(w)
            for w in words
        )
        if not has_prose_word and words:
            if not has_lower:
                uniform_blocks = (
                    len(words) >= 3
                    and len({len(w) for w in words[:-1]}) == 1
                    and len(words[0]) >= 5
                )
                multiword_ok = len(words) == 1 or max(len(w) for w in words) >= 10 or (len(upper_seq) >= 25 and uniform_blocks)
                if len(upper_seq) >= 15 and multiword_ok:
                    _check_inline_dna_guard(upper_seq)
                    return {"sequence": upper_seq, "header": None, "modifier_prose": ""}
            else:
                if len(upper_seq) >= 25 and (len(words) == 1 or max(len(w) for w in words) >= 15):
                    _check_inline_dna_guard(upper_seq)
                    return {"sequence": upper_seq, "header": None, "modifier_prose": ""}

    # Path C: Mixed prompt containing an inline sequence + English modifier prose
    # C1: Multi-line contiguous wrapped sequence block + modifier lines
    if len(lines) > 1:
        best_run: Optional[Tuple[int, int, str]] = None
        i = 0
        while i < len(lines):
            ok_ln, chunk = _is_sequence_like_line(lines[i], allow_lowercase_single_token=True)
            if ok_ln:
                j = i + 1
                chunks = [chunk]
                while j < len(lines):
                    ok_next, chunk_next = _is_sequence_like_line(lines[j], allow_lowercase_single_token=True)
                    if not ok_next:
                        break
                    chunks.append(chunk_next)
                    j += 1
                joined = "".join(chunks)
                if len(joined) >= 20 and (best_run is None or len(joined) > len(best_run[2])):
                    best_run = (i, j, joined)
                i = j
            else:
                i += 1

        if best_run is not None:
            s_idx, e_idx, seq_upper = best_run
            _check_inline_dna_guard(seq_upper)
            leftover = [lines[k].strip() for k in range(len(lines)) if not (s_idx <= k < e_idx) and lines[k].strip()]
            return {
                "sequence": seq_upper,
                "header": None,
                "modifier_prose": " ".join(leftover).strip(),
            }

    # C2: Single-line / inline uppercase AA run (>=15 aa) or unbroken lowercase AA run (>=20 aa)
    inline_candidates: List[re.Match] = []
    for m_up in re.finditer(r"(?<![A-Za-z])[ACDEFGHIKLMNPQRSTVWY]{15,}(?![A-Za-z])", raw):
        if not _is_likely_english_word_not_aa(m_up.group(0)):
            inline_candidates.append(m_up)
    if not inline_candidates:
        for m_any in re.finditer(r"(?<![A-Za-z])[ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy]{20,}(?![A-Za-z])", raw):
            tok_up = m_any.group(0).upper()
            if tok_up.lower() not in _INLINE_PROSE_STOPWORDS and not _is_likely_english_word_not_aa(tok_up):
                inline_candidates.append(m_any)

    if inline_candidates:
        best_m = max(inline_candidates, key=lambda m: len(m.group(0)))
        seq_upper = best_m.group(0).upper()
        _check_inline_dna_guard(seq_upper)
        prose = (raw[:best_m.start()] + " " + raw[best_m.end():]).strip()
        prose = " ".join(prose.split())
        return {
            "sequence": seq_upper,
            "header": None,
            "modifier_prose": prose,
        }

    return None


def _search_rcsb_pdb_for_prompt(protein_query: str) -> Optional[Dict[str, Any]]:
    """
    Searches RCSB PDB full-text index for protein structures (especially de novo / designed
    entries absent from UniProt) and validates token overlap before returning the best hit.
    """
    clean_q = (protein_query or "").strip()
    if not clean_q:
        return None

    payload = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "full_text", "parameters": {"value": clean_q}},
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": "Protein",
                    },
                },
            ],
        },
        "request_options": {"paginate": {"start": 0, "rows": 5}},
        "return_type": "entry",
    }
    search_url = "https://search.rcsb.org/rcsbsearch/v2/query?json=" + requests.utils.quote(json.dumps(payload))
    try:
        resp = _http_get(search_url, headers={"Accept": "application/json"}, timeout=8.0)
        if resp.status_code != 200:
            return None
        rset = (resp.json() or {}).get("result_set") or []
    except Exception:
        return None

    if not rset:
        return None

    pdb_stopwords = {
        "the", "of", "and", "for", "with", "from", "protein", "peptide",
        "chain", "structure", "crystal", "nmr", "cryo", "em", "de", "novo",
    }
    sig_tokens = [t for t in re.findall(r"[a-z0-9]{2,}", clean_q.lower()) if t not in pdb_stopwords]
    q_low = clean_q.lower()

    best_cand: Optional[Dict[str, Any]] = None
    best_score = -1.0
    for item in rset[:4]:
        cand_id = str(item.get("identifier") or "").strip().upper()
        if not cand_id:
            continue
        try:
            cand = fetch_pdb_by_id(cand_id)
        except Exception:
            continue

        short_desc = cand["protein_name"].split(" — ")[0].strip().lower()
        hay = f"{cand['protein_name']} {cand.get('title', '')}".lower()

        if sig_tokens:
            matched_toks = sum(1 for t in sig_tokens if re.search(rf"\b{re.escape(t)}\b", hay) or (len(t) >= 4 and t in hay))
            if matched_toks == 0:
                continue
            if len(sig_tokens) >= 2 and matched_toks < max(1, (len(sig_tokens) + 1) // 2):
                continue
        else:
            if not any(w in hay for w in ("de novo", "synthetic", "designed")):
                continue
            matched_toks = 1

        score = 20.0 * matched_toks
        if short_desc == q_low or cand["protein_name"].lower() == q_low:
            score += 100.0
        if cand.get("is_de_novo"):
            score += 10.0

        if score > best_score:
            best_score = score
            best_cand = cand

    return best_cand


def _assemble_tagged_construct(
    canonical_seq: str,
    detected_tag: Optional[Dict[str, str]],
    tag_position: str,
    detected_cleavage: Optional[Dict[str, Any]],
    target_label: str = "Target Protein",
) -> Tuple[List[Tuple[str, str]], str]:
    """Assembles ordered (label, sequence) construct parts and concatenated full sequence."""
    tag_seq = ""
    cleavage_seq = detected_cleavage["seq"] if detected_cleavage else ""
    if detected_tag:
        if tag_position == "N-terminal":
            tag_seq = detected_tag.get("n_term_short", detected_tag.get("n_term", ""))
        else:
            tag_seq = detected_tag.get("c_term_short", detected_tag.get("c_term", ""))

    construct_parts: List[Tuple[str, str]] = []
    if tag_position == "N-terminal" or not detected_tag:
        if tag_seq:
            construct_parts.append((f"Tag ({detected_tag['name']})", tag_seq))
        if cleavage_seq:
            construct_parts.append((f"Cleavage ({detected_cleavage['name']})", cleavage_seq))
        construct_parts.append((target_label, canonical_seq))
    else:
        construct_parts.append((target_label, canonical_seq))
        if cleavage_seq:
            construct_parts.append((f"Cleavage ({detected_cleavage['name']})", cleavage_seq))
        if tag_seq:
            construct_parts.append((f"Tag ({detected_tag['name']})", tag_seq))

    return construct_parts, "".join(part[1] for part in construct_parts)


def parse_natural_language_prompt(prompt: str, _retry_autocorrect: bool = True) -> Dict[str, Any]:
    """
    Parses any plain-language protein, PDB ID, or direct de novo / custom sequence prompt across
    all domains of life and synthetic biology:
    - "GST human" -> Untagged Human Glutathione S-transferase P (P09211)
    - "Chicken egg white lysozyme with N-terminal 6xHis tag and TEV site" -> 6xHis-TEV-Lysozyme (143 aa)
    - "PDB 1QYS with N-terminal 6xHis tag" or "Top7" -> De novo Top7 construct from RCSB PDB
    - Direct pasted de novo amino acid sequences (optionally + "with N-terminal 6xHis tag and TEV site")
    """
    inline = _extract_inline_sequence_from_prompt(prompt)
    if inline is not None:
        raw_seq = inline["sequence"]
        modifier_prose = inline["modifier_prose"]
        header_raw = inline["header"]

        leftover_title = ""
        detected_tag: Optional[Dict[str, str]] = None
        tag_position = "N/A"
        detected_cleavage: Optional[Dict[str, Any]] = None
        force_untagged = False

        if modifier_prose:
            t_raw, m_raw, force_untagged = _split_target_and_modifiers(f"DeNovoTarget {modifier_prose}")
            if not force_untagged and m_raw:
                _, detected_tag, tag_position, _, detected_cleavage = _detect_tag_and_cleavage_keys(m_raw)
            if not detected_tag and not detected_cleavage and not force_untagged:
                # Fallback in case tag keywords appear in modifier_prose directly
                _, detected_tag, tag_position, _, detected_cleavage = _detect_tag_and_cleavage_keys(modifier_prose)
            rem = re.sub(r"^DeNovoTarget\b\s*", "", t_raw, flags=re.IGNORECASE).strip(" ,;-+&:")
            rem_words = [w.lower() for w in re.findall(r"[A-Za-z]+", rem)]
            noise_words = {"with", "plus", "and", "or", "at", "the", "a", "an", "add", "having", "carrying", "bearing", "to"}
            if rem and not _is_pure_modifier_phrase(rem) and not all(w in noise_words for w in rem_words):
                leftover_title = rem

        if header_raw:
            hdr_meta = extract_fasta_header_metadata(header_raw)
            protein_name = hdr_meta.get("protein_name") or header_raw
        elif leftover_title:
            protein_name = leftover_title if any(ch.isupper() for ch in leftover_title) else leftover_title.title()
        else:
            protein_name = "De Novo / Custom Sequence"

        if (detected_tag or detected_cleavage) and not force_untagged:
            canonical_seq = raw_seq
            construct_parts, full_construct_seq = _assemble_tagged_construct(
                canonical_seq,
                detected_tag,
                tag_position,
                detected_cleavage,
                target_label="Target Sequence (De Novo / Custom)",
            )
            tag_name = detected_tag["name"] if detected_tag else "None"
            final_tag_pos = tag_position if detected_tag else "N/A"
            cleavage_name = detected_cleavage["name"] if detected_cleavage else "None"
        else:
            full_construct_seq = raw_seq
            term_tag = detect_sequence_terminal_tag(raw_seq) if not force_untagged else {"tag_detected": "None", "tag_position": "N/A", "core_sequence": raw_seq}
            cleavage_name = "None"
            if term_tag["tag_detected"] != "None" and len(term_tag["core_sequence"]) < len(raw_seq):
                canonical_seq = term_tag["core_sequence"]
                tag_name = term_tag["tag_detected"]
                final_tag_pos = term_tag["tag_position"]
                if final_tag_pos == "N-terminal":
                    tag_slice = raw_seq[:len(raw_seq) - len(canonical_seq)]
                    construct_parts = [
                        (f"Detected Tag ({tag_name})", tag_slice),
                        ("Target Sequence (De Novo / Custom)", canonical_seq),
                    ]
                else:
                    tag_slice = raw_seq[len(canonical_seq):]
                    construct_parts = [
                        ("Target Sequence (De Novo / Custom)", canonical_seq),
                        (f"Detected Tag ({tag_name})", tag_slice),
                    ]
            else:
                canonical_seq = raw_seq
                tag_name = "None"
                final_tag_pos = "N/A"
                construct_parts = [("Target Sequence (De Novo / Custom)", canonical_seq)]

        return {
            "prompt": prompt,
            "identified_organism": "De Novo / Synthetic Construct",
            "identified_protein": protein_name,
            "uniprot_id": "",
            "accession": "",
            "pdb_id": "",
            "chain_id": "",
            "is_de_novo": True,
            "is_direct_sequence": True,
            "gene_name": "Synthetic / De Novo",
            "biochem_annotations": {},
            "tag_detected": tag_name,
            "tag_position": final_tag_pos,
            "cleavage_detected": cleavage_name,
            "construct_parts": construct_parts,
            "canonical_sequence": canonical_seq,
            "full_construct_sequence": full_construct_seq,
            "length": len(full_construct_seq),
        }

    comp = extract_nl_query_components(prompt)
    protein_query = comp["protein_query"]
    organism_taxid = comp["organism_taxid"]
    organism_name = comp["organism_name"]
    detected_tag = comp["tag_dict"]
    tag_position = comp["tag_position"]
    detected_cleavage = comp["cleavage_dict"]
    clean_query = comp["target_phrase"]

    if not protein_query:
        raise ValueError(
            f"Could not identify a target protein name in '{prompt}'. "
            "Please include a protein/gene name (e.g., 'GST human' or 'Top7'), an RCSB PDB ID ('PDB 1QYS'), "
            "or paste your amino acid sequence directly."
        )

    target_data: Optional[Dict[str, Any]] = None

    # Direct explicit RCSB PDB ID inside plain English ('PDB 1QYS with N-terminal 6xHis tag', 'rcsb 6D0T_A')
    m_explicit_pdb = re.match(
        r"^(?:rcsb\s+)?pdb[\s:_\-]+([1-9][a-z0-9]{3})(?:[\s:_.\-]+(?:chain\s+)?([a-z0-9]{1,2}))?$",
        protein_query.strip(),
        flags=re.IGNORECASE,
    )
    if m_explicit_pdb:
        pdb_q = m_explicit_pdb.group(1).upper() + (f"_{m_explicit_pdb.group(2).upper()}" if m_explicit_pdb.group(2) else "")
        target_data = fetch_pdb_by_id(pdb_q)

    offline_match = _match_offline_classroom_target(
        protein_query,
        organism_taxid=organism_taxid,
        organism_name=organism_name
    )

    pq_low = protein_query.lower()
    is_mature_chicken_lysozyme = (
        offline_match is not None
        and pq_low in {"lysozyme", "lysozyme c", "egg white lysozyme", "egg-white lysozyme", "hewl"}
        and organism_taxid in (None, 9031)
        and (organism_name is None or organism_name.lower() == "gallus gallus")
    )
    is_mature_tev_protease = (
        offline_match is not None
        and (
            pq_low in {"tev", "tev protease", "nia protease", "nia-pro"}
            or ((organism_taxid == 12227 or comp.get("organism_clade_taxid") == 12227) and pq_low in {"protease", "nuclear inclusion protein a"})
        )
        and organism_taxid in (None, 12227)
    )
    if target_data is None and (is_mature_chicken_lysozyme or is_mature_tev_protease):
        target_data = offline_match

    live_uniprot_reachable = False
    if target_data is None:
        search_queries, _, _ = _build_uniprot_search_plan(comp)
        headers = {"Accept": "application/json"}
        try:
            for uq in search_queries:
                search_url = (
                    f"https://rest.uniprot.org/uniprotkb/search?query={requests.utils.quote(uq)}&size=15&format=json"
                )
                search_resp = _http_get(search_url, headers=headers, timeout=12)
                if search_resp.status_code != 200:
                    if search_resp.status_code in (401, 403, 502, 503, 504):
                        break
                    continue
                live_uniprot_reachable = True
                results = search_resp.json().get("results") or []
                if results:
                    ranked = rank_uniprot_candidates(results, comp)
                    valid_ranked = [h for h in ranked if _candidate_matches_requested_protein(h, comp)]
                    if not valid_ranked:
                        continue
                    best_hit = valid_ranked[0]
                    parsed_hit = _parse_uniprot_entry_dict(best_hit)
                    if parsed_hit.get("sequence"):
                        target_data = parsed_hit
                    elif parsed_hit.get("accession"):
                        target_data = fetch_uniprot_by_id(parsed_hit["accession"])
                    break
        except requests.RequestException:
            target_data = offline_match

    if target_data is None:
        target_data = offline_match

    if target_data is None and not live_uniprot_reachable:
        target_data = _search_local_corpus_offline(comp)

    # RCSB PDB fallback for bare 4-char PDB IDs or de novo / designed proteins not indexed in UniProt
    if target_data is None and re.fullmatch(r"[1-9][A-Za-z0-9]{3}(?:[_:.][A-Za-z0-9]{1,2})?", protein_query.strip()):
        try:
            target_data = fetch_pdb_by_id(protein_query.strip())
        except ValueError:
            target_data = None

    if target_data is None and (
        organism_name is None
        or any(kw in prompt.lower() for kw in ("de novo", "synthetic", "designed", "artificial", "rfdiffusion", "proteinmpnn"))
    ):
        target_data = _search_rcsb_pdb_for_prompt(protein_query)

    # Post-miss mobile keyboard autocorrect recovery (e.g. 'elf2a' -> 'eIF2a' only when initial lookup finds 0 hits;
    # requiring [1-6][a-z] ensures real ELF1..ELF5 transcription factors are never rewritten)
    if target_data is None and _retry_autocorrect:
        _elf_rx = re.compile(r"\belf([1-6][a-z][a-z0-9\-]*)\b", flags=re.IGNORECASE)
        if _elf_rx.search(prompt):
            corrected_prompt = _elf_rx.sub(r"eIF\1", _normalize_mobile_text(prompt))
            if corrected_prompt != prompt:
                return parse_natural_language_prompt(corrected_prompt, _retry_autocorrect=False)

    if target_data is None:
        if organism_name:
            raise ValueError(
                f"No matching '{protein_query}' protein was found in {organism_name} on UniProt. "
                f"(Note: {organism_name} may not encode this enzyme—try another species, omit the organism name, "
                "or paste the amino acid sequence directly!)"
            )
        raise ValueError(
            f"Could not find a matching protein sequence on UniProt for query: '{clean_query}'. "
            "Please check the spelling, enter a UniProt or RCSB PDB ID (e.g., 1QYS, 6D0T), or paste the amino acid sequence directly."
        )

    pdb_id = target_data.get("pdb_id", "")
    chain_id = target_data.get("chain_id", "")
    is_de_novo = bool(target_data.get("is_de_novo", False))
    uniprot_id = str(target_data.get("uniprot_id", target_data["accession"] if not pdb_id else "") or "")
    canonical_seq = target_data["sequence"]

    construct_parts, full_construct_seq = _assemble_tagged_construct(
        canonical_seq,
        detected_tag,
        tag_position,
        detected_cleavage,
        target_label="Target Protein",
    )

    return {
        "prompt": prompt,
        "identified_organism": target_data["organism"],
        "identified_protein": target_data["protein_name"],
        "uniprot_id": uniprot_id,
        "accession": target_data.get("accession") or uniprot_id,
        "pdb_id": pdb_id,
        "chain_id": chain_id,
        "is_de_novo": is_de_novo,
        "is_direct_sequence": False,
        "gene_name": target_data.get("gene_name") or "N/A",
        "biochem_annotations": target_data.get("biochem_annotations", {}),
        "tag_detected": detected_tag["name"] if detected_tag else "None",
        "tag_position": tag_position if detected_tag else "N/A",
        "cleavage_detected": detected_cleavage["name"] if detected_cleavage else "None",
        "construct_parts": construct_parts,
        "canonical_sequence": canonical_seq,
        "full_construct_sequence": full_construct_seq,
        "length": len(full_construct_seq),
    }
