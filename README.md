# ProPurify Studio 🧬

### Sequence-to-Bench Protein Purification, Chromatography & Biophysical Analysis Workbench

**ProPurify Studio** is an open-source, interactive Python and Streamlit web application built for biochemistry instruction, protein engineering, and structural biology laboratories. It brings together sequence retrieval, de novo / AI-designed construct support, hydrodynamic chromatography simulation, SDS-PAGE gel calibration & casting recipes, cofactor-safe purification planning, buffer stoichiometry, secondary structure prediction, interactive 3D structure visualization, and transparent empirical benchmarks in one cohesive workspace.

---

## ✨ Key Capabilities

### 1. Universal Quick Search & 6 Sequence Input Methods
* **Main-Page Universal Quick Search Bar (`Analyze Protein`):** Automatically routes plain-English queries (including mobile Unicode/autocorrect normalization), UniProt accessions/URLs, NCBI accessions/URLs, RCSB PDB IDs/chains/URLs, pasted FASTA/amino-acid sequences (with inline terminal tag/cleavage modifiers), and coding DNA/RNA sequences (≥ 30 nt).
* **6 Sidebar Input Methods:**
  1. **UniProt ID:** Supports canonical accessions (`P00698`), isoforms (`P04637-2`), entry names (`LYSC_CHICK`), or UniProt URLs, with a **1-click Signal/Propeptide Cleavage Toggle** to switch secreted precursors to their annotated mature chain.
  2. **NCBI Accession:** Supports RefSeq/GenBank protein (`NP_990612.2`, `WP_000002298.1`) and nucleotide/mRNA (`NM_205281.2`) accessions with automatic 6-frame ORF translation.
  3. **RCSB PDB ID (De Novo / 3D):** Loads 4-character PDB IDs (`1QYS`, `2KL8`, `6D0T`), chain selectors (`6D0T_A`), or RCSB URLs—tailored for experimental structures and computationally designed proteins.
  4. **Paste Amino Acid / FASTA:** Supports natural, synthetic, and AI-designed constructs (RFdiffusion, ProteinMPNN, ESMFold) with automatic N-/C-terminal affinity tag detection (`6xHis`, `Strep`, `FLAG`, etc.).
  5. **Translate DNA / RNA:** Performs 6-frame ORF detection (`+1..+3` forward and `-1..-3` reverse complement) or direct Frame 1 translation while preserving IUPAC `N` ambiguity codes.
  6. **Describe in Plain English:** Deterministic construct builder supporting organism/protein disambiguation across the Tree of Life, engineered affinity tags (`6xHis`, `Strep-tag II`, `Twin-Strep`, `GST`, `MBP`, `FLAG`), and protease cleavage sites (`TEV`, `HRV 3C`, `Thrombin`, `Factor Xa`, `SUMO`).
* **Default Startup Construct & 8 Ready-to-Load Starter Proteins:** Launches ready to explore on **`6xHis-TEV-Lysozyme`** (`143 aa`, `16.12 kDa`, reduced `pI 9.16` / disulfide-oxidized `~10.15`; **Ni-NTA IMAC ➜ TEV cleavage ➜ Superdex 75 SEC polish**), featuring an interactive **Construct N-to-C Architecture Map** showing the `6xHis` tag, `TEV` cleavage cut site (`16.12 kDa` intact → `14.37 kDa` cleaved Gly-Lysozyme), and target domain. Switch anytime via the sidebar across 8 starter proteins: `6xHis-TEV-Lysozyme`, Hen Egg-White Lysozyme C, GFP, Human Carbonic Anhydrase 2, Sperm Whale Myoglobin, *B. subtilis* Lipase A, BSA, and Jack Bean Urease.

---

### 2. At-a-Glance Bench Action Plan & 6 Physicochemical KPIs
* **4-Column Bench Action Plan:** Summarizes recommended Primary Capture at pH 7.4, Polishing SEC column and predicted elution volume ($V_e$), target SDS-PAGE percentage & buffer system (Tris-Glycine vs. Tricine), and active buffer/hazard cautions right below the construct ribbon.
* **6 ExPASy-Calibrated KPI Cards:**
  * **Molecular Weight** ($\text{kDa}$ & $\text{Da}$)
  * **Theoretical pI** (Bjellqvist / ExPASy ProtParam denatured scale, plus an oxidized-disulfide estimate when structural disulfides are present)
  * **Extinction Coefficient ($A_{280}$)** (Pace 1995 reduced vs. cystine-oxidized $\varepsilon_{280}$, plus **UV-280 blindness alerts** for zero-Trp/Tyr/cystine peptides and AI-designed scaffolds)
  * **Instability Index** (Guruprasad 1990, with caveats for Glu/Lys-rich thermostable ProteinMPNN/de novo designs)
  * **Aliphatic Index** (Ikai 1980)
  * **GRAVY Hydropathy Index** (Kyte–Doolittle 1982)

---

### 3. Size-Exclusion Chromatography (SEC), Inside-the-Column Studio & SDS-PAGE (Tab 1)
* **SEC Pore-Partitioning Schematic:** Visual bead-pore diagram showing how large HMW complexes bypass resin beads via the void volume ($V_0$), intermediate oligomers enter partial pockets for mid-range $V_e$, and compact monomers explore deep bead pores toward total volume ($V_t$).
* **8 Calibrated SEC Columns:**
  * *Preparative (120–320 mL):* `HiLoad 16/600 Superdex 75 pg`, `HiLoad 16/600 Superdex 200 pg`, `HiLoad 16/600 Superose 6 pg`, and `HiLoad 26/600 Superdex 200 pg`.
  * *Analytical (2.4–24 mL):* `Superdex 75 Increase 10/300 GL`, `Superdex 200 Increase 10/300 GL`, `Superose 6 Increase 10/300 GL`, and `Superose 6 Increase 3.2/300`.
* **Inside-the-SEC-Column Studio:** Interactive column cutaway, porous bead cross-section, fraction collector rack, live chromatogram cursor, buffer-pump slider, and 4 quick-jump run stages (*Inject*, *Void $V_0$*, *Target $V_e$*, *End $V_t$*).
* **Oligomers, Aggregates & Binding Complexes:** Simulate monomer, dimer, trimer, tetramer, soluble HMW void-volume ($V_0$) aggregates with Rayleigh light scattering, 5 conformational shapes (globular to disordered IDP), protein–protein / dsDNA / ssRNA stoichiometric complexes, dual-wavelength $A_{280}$ / $A_{260}$ traces, and calibration standards overlay.
* **Virtual Destained SDS-PAGE Gel & Migration Strip:** Includes a **4-step SDS-PAGE migration strip** (95 °C + DTT denaturation/disulfide cleavage ➜ uniform negative SDS coat ➜ 4% stacking gel packing at pH 6.8 ➜ MW-dependent resolving gel migration toward the anode), sigmoidal Ferguson migration across 4%–22% polyacrylamide, monomer-only denaturing physics, Coomassie Blue R-250 and Silver Stain modes, purification-stage lanes, and a calibrated Bio-Rad Precision Plus Dual Color ladder.
* **SDS-PAGE Gel Preparation Recipe Pop-Out:** Modal handcast recipe calculator supporting **Tris-Glycine (Laemmli 1970, 4%–22% T)** and **Tricine-SDS-PAGE (Schägger 2006, 10%–20% T, auto-selected for <10 kDa peptides)**, 30% / 40% / 49.5% acrylamide stocks, 4% / 5% stacking gels, mini/midi/custom cassette volumes (1–4 gels), glycerol toggle, running buffer formulas, EHS neurotoxin/polymerization reminders, and downloadable `.txt` recipe sheets.

---

### 4. 3-Step Purification Roadmap, 7-Method Advisor & Biochemical Guards (Tab 2)
* **Dynamic 3-Step Purification Pipeline Flowchart & Charge vs. pH Titration:** Renders a live **3-step purification pipeline flowchart** (Capture ➜ Intermediate ➜ Final Polish) that adapts automatically to active tags, pI, and cofactor/metalloprotein alerts, alongside an interactive working pH slider (pH 3.5–10.5) with a $\pm 0.5$ pH isoelectric clumping zone warning.
* **Biochemical & Cofactor Safety Guards:** Detects metalloproteins ($\text{Ni}^{2+}$, $\text{Zn}^{2+}$, $\text{Cu}^{2+}$, $\text{Mn}^{2+}$, $\text{Co}^{2+}$, $\text{Mo}$), $[\text{Fe–S}]$ clusters, heme cofactors, $\text{Ca}^{2+}/\text{Mg}^{2+}$ dependencies, structural disulfides, and integral membrane / transmembrane $\beta$-barrel detergent-micelle requirements—flagging conflicts such as Ni-NTA / imidazole on metalloproteins, high imidazole on hemoproteins, or DTT/GSH on disulfide-stabilized proteins.
* **7-Method Comparison Matrix, IEX vs. HIC Contrast Schematic & Compatibility Board:** Evaluates (1) Tag Affinity (Strep / MBP / FLAG / IMAC), (2) Heparin / Blue Pseudo-Affinity, (3) Ammonium Sulfate Salting-Out, (4) Ion-Exchange (AEX / CEX) with counterion-matched buffers, (5) Hydrophobic Interaction (HIC — Butyl / Phenyl / Octyl), (6) Hydroxyapatite / Multimodal (CHT / Capto MMC), and (7) SEC Polishing—paired with an **IEX vs. HIC salt-gradient contrast schematic** (low-salt bind / high-salt NaCl elute vs. high-salt $(\text{NH}_4)_2\text{SO}_4$ bind / low-salt elute), a 6-tile Compatibility Board, and a 1-minute bench decision quiz.

---

### 5. Customizable Laboratory Buffer & Recipe Calculator (Tab 3)
* **47-Reagent Biochemical Shelf & 14 Starter Templates:** Covers IMAC, IEX, HIC, SEC, Metalloprotein-Safe (EDTA-free & imidazole-free), Disulfide-Safe (reductant-free), Strep-Tactin, Strep-Tactin XT (`50 mM` D-Biotin), GST, and MBP workflows plus a custom blank recipe.
* **Interactive Formulation, 4-Step Prep Strip & Safety Checks:** Editable component table for arbitrary volumes (`mL` / `L`) and target pH (`2.0`–`12.0`), dual solid mass output (**`g` and `mg`**), neat liquid dispensing (**`µL` / `mL`**), live active-protein buffer auditing with 1-click safe-buffer switching, two-way `g`/`mg` ↔ `mM`/`M` converter, 4-step buffer prep visual workflow strip & temperature/safety protocol cards, and **CSV export**.

---

### 6. Interactive 3D Viewer, Secondary Structure Engine & Solubility (Tab 4)
* **2-Mode Interactive WebGL 3D Structure Viewer (3Dmol.js):**
  * **Live AlphaFold DB / RCSB PDB 3D (Default):** On-demand WebGL **3Dmol.js** fetching and rendering (`cartoon`, `surface`, `stick`, `sphere`) from AlphaFold DB or RCSB PDB, colored by secondary structure, pLDDT confidence, N→C spectrum, hydrophobicity, or side-chain charge, with optional ligand, metal, and disulfide visualization (zero outbound HTTP requests until you click Fetch; notes when engineered terminal tags are absent from native UniProt coordinates).
  * **Upload AI / Custom `.pdb` / `.cif`:** Supports `.pdb`, `.cif`, `.mmcif`, `.ent`, and `.gz` models (up to 10 MB upload; 4 MB WebGL display cap) from AlphaFold3, ColabFold, Boltz, RFdiffusion, ProteinMPNN, and ESMFold.
  * **Instant Residue Hover & Click/Tap Labels:** Hovering (`50 ms` trigger) or clicking/tapping any residue or bound cofactor/metal displays an interactive 3D callout label and live detail pill showing the **amino acid full name, 3-letter & 1-letter code, residue number, chain, secondary structure, and pLDDT confidence (or B-factor)**.
* **Cascaded Neural + GOR-IV Secondary Structure Engine:** Pure-NumPy 2-stage neural predictor with 5-fold out-of-fold GOR-IV features and Viterbi smoothing (`~18.5 ms` mean inference, `4.04 MB` model weights). Supports cosine-blended domain-tile windowing for mega-proteins (such as 34,350-aa human Titin) with a 600-aa Domain Window Explorer, 2-track Plotly topology & probability plot, segment coordinate table, residue alignment, and `.fasta` export.
* **Amino Acid Composition & Solubility:** 20-residue frequency chart, functional side-chain table, reduced/oxidized extinction card, and Wilkinson–Harrison (1991) / Davis (1999) *E. coli* cytoplasmic solubility heuristic.

---

### 7. Empirical Benchmarks & Transparent Methodological Scope (Tab 5)
* **SEC Hydrodynamics & Literature Case Studies:**
  * Uses the globular hydrodynamic scaling law $\ln(R_h) = 1.92534 + 0.38101\ln(\text{MW}) - 0.00167[\ln(\text{MW})]^2$ ($R^2 = 0.995$ across 55 Cytiva globular calibration points; `0.22 mL` MAE on 24 mL analytical columns).
  * **`0.17 mL` MAE** across **20 peaks in 7 published open-access Europe PMC case studies**, plus a **1,254-PDB** molecular-weight / hydrodynamic-radius calibration sweep (`809` with SEC-supported assembly metadata; downloadable CSV). Note: the 1,254-PDB set is a theoretical $R_h$/column-range sweep rather than 1,254 experimental chromatograms, and real elution shifts with elongated shape, glycosylation, detergent micelles, or resin interactions.
* **All-Domains-of-Life Secondary Structure Benchmark (`11,118` Held-Out PDB/DSSP Proteins):**
  * Trained strictly on `3,751` PDB/DSSP chains (`1,449,487` residues) and evaluated on `11,118` homology-filtered proteins (`4,066,153` residues; `<30%` 5-mer similarity, `0` accession/sequence overlap, `0` preset overlap): **`69.50%` mean protein $Q_3$** (`69.31%` pooled $Q_3$), **`59.78%` $\text{SOV}_{99}$**, and **`81.48%` fold-class accuracy** (`+7.74%` $Q_3$ over GOR IV, `+20.16%` over Chou–Fasman).
  * Evaluated independently on **32 experimental AI-designed / de novo PDB structures** (`3,908` residues across Rosetta, Foldit, trRosetta hallucination, ProteinMPNN, RFdiffusion+ProteinMPNN, and transmembrane de novo folds): **`80.91%` pooled $Q_3$** (`81.19%` mean $Q_3$), **`80.92%` $\text{SOV}_{99}$**, and **`93.75%` fold-class accuracy**.
* **77-Protein Stratified Cross-Reference Panel & Tab 5 Q3 Comparison Bar Chart:** Compares ProPurify SS (`68.75%` mean $Q_3$ vs. experimental DSSP) against AlphaFold DB / PDBe 3D (`86.06%` mean $Q_3$ overall; `87.68%` on 67 AlphaFold DB monomers; `70.0%–70.7%` pairwise concordance with ProPurify SS) and single-sequence baselines (`GOR IV` `54.28%`, `Chou–Fasman` `44.38%`), accompanied by a horizontal Plotly **`Single-Sequence Mean Q3 Comparison (%)` bar chart**. *(In the saved cross-reference snapshot, external `NPS@ SOPMA`, `NPS@ GOR IV`, and `JPred4` servers returned `0` live entries; because ProPurify SS is a single-sequence predictor without evolutionary MSAs, consult PDB, AlphaFold DB, or MSA servers for research-grade structural modeling.)*
* **Multi-Domain Plain-Language Search Benchmark (`13,403` Swiss-Prot Proteins):** `100.0%` offline grammar/construct/biological-identity accuracy across `13,403` proteins (`1,798` taxa) and `100.0%` biological-identity match on live UniProt tier-2 validation.
* **Scope Note:** All chromatographic, electrophoretic, pI, and solubility outputs are computational simulations and starting hypotheses—always confirm critical bench protocols against experimental measurements and resin manufacturer manuals.

---

### 8. Anonymous Error Reporting & Feedback
* Sidebar **`💬 Report Error / Feedback`** modal collects **zero personal identifiers**, attaches read-only active construct diagnostic metadata, supports an optional **screenshot / picture attachment** (`PNG`, `JPG`, or `WEBP`, up to `4 MB`, magic-byte validated) with **inline preview**, optionally forwards payloads to `FEEDBACK_WEBHOOK_URL` (via Streamlit secrets or `PROPURIFY_FEEDBACK_WEBHOOK`, using multipart upload when an image is attached), logs locally when writable, and provides both a copyable diagnostic summary and a **1-click `.json` report download** (embedding compact screenshots $\le 350\text{ KB}$ as `data_uri`).

---

## 🚀 Running Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/propurify-studio.git
   cd propurify-studio
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the application:**
   ```bash
   streamlit run app.py
   ```
   Open your browser to `http://localhost:8501`.

---

## 🌐 Cloud Deployment

ProPurify Studio uses pure-Python/NumPy offline model weights (`4.04 MB`) with no GPU or database server required, making it straightforward to deploy on community hosting platforms:

### Option 1: Streamlit Community Cloud (Recommended)
1. Push `app.py`, `modules/`, `data/`, `.streamlit/`, `.licenses/`, `requirements.txt`, and `README.md` to a GitHub repository.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) and click **Create app** (or **New app**).
3. Select your repository, branch `main`, and set the main file path to `app.py`.
4. *(Optional)* Add `FEEDBACK_WEBHOOK_URL` under **App settings ➜ Secrets** if you want anonymous feedback reports forwarded to a webhook endpoint.
5. Click **Deploy**.

### Option 2: Hugging Face Spaces
1. Sign in at [huggingface.co](https://huggingface.co) and click **New Space**.
2. Choose **Streamlit** as the Space SDK (free CPU Basic tier provides 2 vCPUs and 16 GB RAM).
3. Push or upload the repository files (if committing large binary data files, ensure Git LFS is configured for `.npz` / `.json` assets as needed by the Space).
4. Your app will be live at `https://huggingface.co/spaces/<your-username>/propurify-studio` and can be embedded in course or lab portals via `<iframe>`.

---

## 🧪 Running the Verification Test Suite

ProPurify Studio includes **15 golden regression and scientific verification suites** covering ProtParam constants, buffer stoichiometry, all 7 purification methods, SEC hydrodynamics & 1,254-PDB calibration, SDS-PAGE monotonicity, natural-language disambiguation & Tree-of-Life ranking, void-volume aggregates, secondary structure inference & mega-protein Titin (34,350 aa), Inside-the-SEC-Column studio, metalloprotein/cofactor safety guards, all input methods, Tris-Glycine/Tricine gel casting recipes, 2-mode WebGL 3Dmol.js residue inspection, screenshot-enabled anonymous feedback, AI-designed protein behavior, XML-valid SVG visual explanations, Tab 5 Q3 comparison chart, and headless `AppTest` UI & audit regression invariants:

```bash
python test_modules.py
```

---

## 📜 Scientific References & Data Notices
* **ProtParam pKa & Extinction:** Bjellqvist et al. (1993, 1994) *Electrophoresis*; Pace et al. (1995) *Protein Science*.
* **Instability & Aliphatic Indices:** Guruprasad et al. (1990) *Protein Engineering*; Ikai (1980) *J. Biochem.*.
* **Hydropathy & Solubility:** Kyte & Doolittle (1982) *J. Mol. Biol.*; Wilkinson & Harrison (1991) *Bio/Technology*; Davis et al. (1999) *Biotechnol. Bioeng.*.
* **Hydrodynamic Stokes Radii & SEC:** Uversky (1993, 2002) *Biochemistry*; Wilkins et al. (1999) *Biochemistry*; Erickson (2009) *Biol. Proced. Online*.
* **SDS-PAGE & Tricine Systems:** Laemmli (1970) *Nature*; Schägger (2006) *Nat. Protoc.*.
* **Secondary Structure:** Garnier, Gibrat & Robson (1996) *Methods Enzymol.* (GOR-IV); Chou & Fasman (1978) *Annu. Rev. Biochem.*; Rost & Sander (1993) *J. Mol. Biol.*; Zemla et al. (1999) *Proteins* ($\text{SOV}_{99}$).
* **Data Usage Notices:** See `.licenses/` for RCSB PDB and Europe PMC usage policy reminders.
