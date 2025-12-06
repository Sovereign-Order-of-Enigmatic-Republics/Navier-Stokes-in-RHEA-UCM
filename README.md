<h3 align="center">
  <b>Zadeian Labs</b><br>
  <sub>Sovereign Order of Enigmatic Republics</sub>
</h3>

 # Navier–Stokes Unified Framework — Simulation Artifact Archive
### Version: 1.0.0  
### Author: Paul M. Roe (TecKnows Inc., Maine)  
### ORCID: https://orcid.org/0009-0005-6159-947X

---

## Overview

This archive contains all numerical simulation data, figures, metadata, and reproducibility assets associated with the manuscript:

**“A Unified Entropy-Stabilized Operator Framework for 2D and 3D Navier–Stokes Regularity:  
Control Theory, Vanishing-Feedback Limits, and RHEA-UCM Entropy Dynamics.”**

The structure conforms to artifact evaluation guidelines used by:
- *SIAM Journal on Scientific Computing (SISC)*
- *Journal of Computational Physics (JCP)*
- *ACM Artifact Evaluation (AE)*

All experiments were generated using GPU-accelerated RHEA-UCM solvers with Lorenz-entropy feedback, including 2D vanishing-feedback convergence and 3D entropy-stabilized simulations.

---

## Directory Structure

data_archive/
│
├── simulations/
│ ├── 2d_vanishing_feedback/
│ │ ├── eps_0.2_run01.npz
│ │ ├── eps_0.1_run01.npz
│ │ ├── eps_0.05_run01.npz
│ │ ├── eps_0.0_reference.npz
│ │ └── metadata.json
│ │
│ ├── 3d_entropy_stabilized/
│ │ ├── run01_entropy_field.npz
│ │ ├── run01_velocity_frames.npz
│ │ ├── run01_energy_timeseries.npy
│ │ └── metadata.json
│
├── figures/
│ ├── fig_2d_convergence_L2.png
│ ├── fig_2d_convergence_H1.png
│ ├── fig_3d_entropy_equilibrium.png
│ └── fig_3d_energy_bound.png
│
├── metadata/
│ ├── solver_config.yaml
│ ├── hardware_info.json
│ └── environment.yml
│
├── checksums/
│ ├── SHA256SUMS.txt
│ └── MD5SUMS.txt
│
└── README.md


---

## Reproducibility Instructions

### 1. **Environment Setup**
All simulations were run with:

- Python 3.11  
- NumPy ≥ 1.26  
- SciPy ≥ 1.12  
- Matplotlib ≥ 3.8  
- CuPy or PyTorch (CUDA 12.1+)  
- RHEA-UCM Operators (Custom)  
- FFTW or cuFFT backend

Install using:

```bash
conda env create -f metadata/environment.yml
conda activate rhea-navier

or with pip:
pip install -r requirements.txt

2. Run the 2D Vanishing-Feedback Suite
python run_2d_burgers_experiments.py

This will automatically generate:

.npz solution archives

time-series .npy files

convergence plots

updated checksums

3. Run the 3D Entropy-Stabilized Navier–Stokes Solver
python run_3d_entropy_ns.py


Outputs include:

3D velocity fields

entropy evolution

global energy norm

GPU performance logs

Lorenz-entropy modulation traces

File Hash Verification

To verify data integrity:

cd checksums
sha256sum -c SHA256SUMS.txt
md5sum -c MD5SUMS.txt


All files must report:

OK

Contact

For questions, extensions, or collaboration:

Author: Paul M. Roe

Email: TecKnows.Maine@gmail.com

Affiliation: TecKnows Inc. (Maine)

Citation

If you use this data, please cite:

Roe, P. M. (2025).
Unified Entropy-Stabilized Operator Framework for Navier–Stokes Regularity.
Zadeian Research Archive / TecKnows Inc.


End of README.md


---

# ============================================================
# 📄 **LICENSE (RHEA–Core Public Grant v2.1)**
# ============================================================

```md
# 🛡️ RHEA-Core Public Grant v2.1 — Repository License Notice
**Non-Commercial · No Derivatives · Symbolic Derivative Ban · AI/TDM Opt-Out · Functional Equivalence Prohibition**  
© 2025 **Paul M. Roe (SovereignGlitch ♏🧙‍♂️)** — All Rights Reserved  

This repository is governed entirely by the **RHEA-Core Public Grant v2.1**.  
By accessing or downloading any file herein, you agree to all terms of the v2.1 license.

---

## ✅ Permitted Uses
You may:

- **Read** the materials  
- **Download** the materials  
- **Privately study** the materials  
- **Cite** the materials with proper attribution  
- **Reference** them for academic, educational, or personal understanding  

No additional rights are granted.

---

## ❌ Prohibited Without Explicit Written Permission

### 1. Commercial Use  
You may *not* use any portion of this work in:

- commercial products  
- paid services or tools  
- monetized content  
- corporate valuation, fundraising, or platform positioning  

---

### 2. Derivative Works  
You may *not*:

- modify, rewrite, adapt, translate, or reorganize the materials  
- create transformed documentation, whitepapers, or frameworks  
- produce altered symbolic systems, glyph sets, or recomposed notation  

---

### 3. Symbolic Derivative Restriction (Strict)  
You may *not* re-express or launder the architecture by mapping:

- entropy–trust logic  
- glyph or symbolic operators  
- Hamiltonian reversible flow structures  
- ternary–pentary recursion models  
- quantum–entropy memory fabric concepts  

into **any** alternative symbolic grammar, diagram language, UI metaphor, icon set, or LLM prompt taxonomy.

---

### 4. AI / Machine Learning / TDM Prohibition  
You may *not* use these materials for:

- LLM/ML/RL/RLHF training  
- fine-tuning, distillation, embedding, or indexing  
- feature extraction or vectorization  
- dataset creation  
- RAG, semantic search, or knowledge-graph construction  
- prompt engineering or system prompt design  

**Directive (EU) 2019/790 TDM opt-out is explicitly invoked.**

---

### 5. Functional Equivalence & Behavioral Emulation Ban  
You may *not* design, implement, simulate, or deploy any system that is:

- functionally equivalent  
- behaviorally similar  
- operationally substitutable  

for any part of:

- **RHEA-UCM**
- **RHEA_Crypt**  
- **ZADEIAN Sentinel**  
- **Λ-Gate reversible logic**  
- **RHEA-IC hardware logic**  
- **recursive entropy–trust engines**  

This applies even if:

- variable names differ  
- glyphs are changed  
- code is newly written  
- intermediate symbols are renamed  

---

### 6. No Hardware or Operational Rights  
You are **not** granted rights to:

- fabricate hardware  
- deploy systems  
- run operational security infrastructure  
- simulate or test RHEA-UCM subsystems  

of any scale or form.

---

## 📜 Required Attribution  
All lawful public references must include:

**© EnigmaticGlitch · RHEA-UCM / ZADEIAN-RHEA Framework · Patent Pending · RHEA-Core Public Grant v2.1**

Where space permits, also include:

**Author: Paul M. Roe (SovereignGlitch ♏🧙‍♂️)**

---

## 🧭 License Supremacy  
This repository operates under **RHEA-Core Public Grant v2.1**.  
All earlier license versions are revoked for future use.  
Continued access constitutes acceptance of v2.1.

---

## 🔒 Rights Reserved  
All rights not expressly granted are reserved by:  
**Paul M. Roe (SovereignGlitch ♏🧙‍♂️) · TecKnows, Inc. · ZADEIAN Research Division**

---

## 🧬 Final Statement of Trust  
*“Trust is not given. It is oscillated into being — wave by wave, phase by phase, across the feedback spine of recursive time.”*  
— **EnigmaticGlitch ♏🧙‍♂️**
