# ProPurify Studio 🧬
### Laboratory-Grade Protein Purification & Biophysics Platform for Education & Research

**ProPurify Studio** is an open-source, interactive web application built in Python and Streamlit. It is designed for high-school, undergraduate, and graduate classrooms, as well as structural biology and protein biochemistry laboratories worldwide.

ProPurify Studio enables students and researchers to simulate chromatography runs, design complete purification roadmaps, formulate laboratory buffers, predict secondary structure topology, and inspect real empirical benchmarks with zero room for error.

---

## 🌟 Key Modules & Educational Features

### 1. 📈 Preparative & Analytical Size Exclusion Chromatography (SEC) + Simulated SDS-PAGE
* **Dual-View Column Modeling:** 
  * *Preparative (120 mL):* HiLoad 16/600 Superdex 200 pg & Superdex 75 pg.
  * *Analytical (24 mL):* Superdex 200 Increase 10/300 GL, Superdex 75 Increase 10/300 GL, and Superose 6 Increase 10/300 GL (5–5,000 kDa).
* **Multi-State Oligomer Simulation:** Simulate monomer, dimer, trimer, and tetramer peaks with dual UV absorption ($A_{280}$ and $A_{260}$ based on pure amino acid composition and nucleic acid ratios).
* **Soluble High-Molecular-Weight (HMW) Aggregate Void Peak ($V_0$):** Model void volume exclusion ($K_{av} = 0$) aggregates with asymmetric tailing and Rayleigh light scattering ($A_{280} \propto \lambda^{-4}$).
* **Realistic Destained SDS-PAGE:**
  * Dynamic Ferguson sigmoidal sieving across 4% to 22% polyacrylamide gels.
  * Monomer-only denaturing physics ($R_f$ strictly independent of native quaternary oligomer state).
  * True destained Coomassie Brilliant Blue R-250 and Silver stain visualization.
  * Bio-Rad Precision Plus Dual Color protein ladder calibrated in the final well.

### 2. 🧭 3-Step Purification Strategy & 7-Method Evaluation
* **Interactive Charge vs. pH Curve:** Exact Bjellqvist ExPASy ProtParam polypeptide $\text{p}K_a$ scale with terminal residue corrections.
* **Smart Isoelectric Precipitation Warning:** Alert triggered when buffer pH is within $\pm 0.5$ units of theoretical $\text{pI}$ ($|\text{pH} - \text{pI}| \le 0.5$).
* **Comprehensive 7-Method Strategy Matrix:** Evaluates Anion Exchange (AEX), Cation Exchange (CEX), Hydrophobic Interaction Chromatography (HIC), Size Exclusion (SEC), Immobilized Metal Affinity (IMAC), Strep-Tactin / Strep-Tactin XT, and Glutathione Sepharose (GST).
* **Method A (IEX) & Method B (HIC) Comparison Cards:** Real-time salt gradient recommendations (e.g. 1.5 M to 0 M $(NH_4)_2SO_4$ for HIC; 0 to 1.0 M NaCl for IEX).

### 3. 🧪 Laboratory Buffer & Custom Recipe Calculator (`g` / `mg`)
* **Custom Volume & Target pH:** Supports any target batch volume ($1\text{ mL}$ to $50\text{ L}$) with automatic metric unit switching.
* **47-Reagent Biochemical Shelf:** Calibrated against IUPAC 2021 atomic weights.
* **Dual-Unit Solid Mass Output:** Displays mass dynamically in **grams ($g$)** and **milligrams ($mg$)** with high precision.
* **Liquid Molar Reagent Pipetting:** Computes exact microliter/milliliter volume dispensing ($\mu\text{L}$ / $\text{mL}$) using neat molarity and density (e.g. 2-Mercaptoethanol, Triton X-100, Glycerol).
* **10 Classroom Buffer Templates:** Instant 1-click loading for Ni-NTA Lysis/Wash/Elution, Strep-Tactin XT (50 mM D-Biotin), GST-tag, CEX, AEX, and SEC Polishing buffers.
* **Safety & Storage Cards:** Fume hood alerts, divalent chelator warnings, D-biotin dissolution guidelines, and 1-click CSV export.

### 4. 🔬 Amino Acid Composition, Secondary Structure & Solubility
* **Cascaded Neural + GOR-IV Secondary Structure Engine:**
  * Pure-NumPy implementation executing in $<20\text{ ms}$ for typical proteins.
  * Overlapping domain-tile windowing with cosine cross-fade for giant mega-proteins (e.g., human Titin, 34,350 aa).
  * Interactive 2-track Plotly diagram showing discrete SS topology and continuous 3-state propensity lines.
  * 600-aa domain window explorer for multi-domain assemblies.
* **Fold Classification & Extinction Coefficients:**
  * 4 fold KPI cards ($\alpha$-helix %, $\beta$-sheet %, coil %, and dominant structural class).
  * Pace (1995) extinction coefficients ($\varepsilon_{280}$) for folded and unfolded states.
  * Wilkinson-Harrison (1991) and Davis (1999) *E. coli* recombinant solubility indices.

### 5. 📚 Real Lab Examples & Rigorous Benchmarks
* **1,254-PDB Empirical Biomolecule SEC Benchmark:** Head-to-head comparison of theoretical vs. experimental Stokes radii across globular proteins, complexes, and nucleic acid assemblies.
* **7 Europe PMC Open-Access Case Studies:** Detailed laboratory purification procedures from peer-reviewed literature.
* **11,118-Protein Secondary Structure Benchmark:** Evaluated on held-out PDB/DSSP crystallographic data ($Q_3 = 69.5\%$, $\text{SOV}_{99} = 59.8\%$).
* **77-Protein External Web Server Cross-Reference:** Validated against AlphaFold DB v6, NPS@ SOPMA, NPS@ GOR IV, and JPred4.

---

## 🚀 How to Run Locally

1. **Clone or navigate to the repository:**
   ```bash
   cd e:\Prot_app
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the web application:**
   ```bash
   streamlit run app.py
   ```
   Open your browser and navigate to `http://localhost:8501`.

---

## 🌐 Free Global Deployment (No Web Server Required!)

You **do not need a personal server, virtual machine, or cloud subscription** to make ProPurify Studio available to everyone on the internet. 

Because ProPurify Studio is built on Streamlit with lightweight pure-Python dependencies and offline NumPy model weights, you can host it **100% free** on dedicated community cloud platforms:

### Option 1: Streamlit Community Cloud (Recommended ⭐⭐⭐⭐⭐)
* **Cost:** 100% Free forever.
* **Official Platform:** Maintained directly by Snowflake / Streamlit.
* **Continuous Deployment:** Every time you push an update to GitHub, the live site updates automatically in seconds.
* **Free HTTPS & Public Link:** Gives you a clean URL like `https://propurify-studio.streamlit.app`.

#### Step-by-Step Deployment:
1. **Push your code to GitHub:**
   * Create a free account at [github.com](https://github.com).
   * Create a new repository named `ProPurify-Studio`.
   * Push your project files (`app.py`, `modules/`, `data/`, `.streamlit/`, `requirements.txt`, `README.md`) to GitHub.
2. **Deploy on Streamlit Cloud:**
   * Visit [share.streamlit.io](https://share.streamlit.io).
   * Sign in with your GitHub account.
   * Click **"New app"**.
   * Select your `ProPurify-Studio` repository, branch `main`, and set Main file path to `app.py`.
   * Click **"Deploy!"**.
   * Within 2 minutes, your website is live and shareable with students, colleagues, and universities worldwide!

### Option 2: Hugging Face Spaces (Alternative ⭐⭐⭐⭐)
* **Cost:** 100% Free.
* **Hardware:** Provides 2 vCPUs and 16 GB RAM out of the box.
* **Embedding:** Allows embedding the application directly into school/college websites, Canvas, or Moodle via an HTML `<iframe>`.

#### Step-by-Step Deployment:
1. Sign up for free at [huggingface.co](https://huggingface.co).
2. Click **New Space** ➜ Choose **Streamlit** as the Space SDK.
3. Upload your files or connect your GitHub repository.
4. Your Space is immediately accessible at `https://huggingface.co/spaces/<your-username>/propurify-studio`.

---

## 🧪 Running the Verification Test Suite

ProPurify Studio includes a rigorous 10-suite regression test script validating ProtParam constants, stoichiometric SEC calculations, 13,403-protein plain language search, mega-protein secondary structure inference, and Streamlit rendering:

```bash
python test_modules.py
```

---

## 📜 Scientific References & Acknowledgments
* **ProtParam pKa & Extinction:** Bjellqvist et al. (1993, 1994) *Electrophoresis*; Pace et al. (1995) *Protein Science*.
* **Instability Index:** Guruprasad et al. (1990) *Protein Engineering*.
* **Hydrodynamic Stokes Radii & SEC:** Uversky (1993, 2002) *Biochemistry*; Wilkins et al. (1999) *Biochemistry*; Erickson (2009) *Biol. Proced. Online*.
* **Recombinant Solubility:** Wilkinson & Harrison (1991) *Bio/Technology*; Davis et al. (1999) *Biotechnol. Bioeng.*.
* **Secondary Structure:** Rost & Sander (1993) *J. Mol. Biol.*; Jones (1999) *J. Mol. Biol.*; Garnier, Osguthorpe & Robson (1978, 1996) *GOR-IV*.
