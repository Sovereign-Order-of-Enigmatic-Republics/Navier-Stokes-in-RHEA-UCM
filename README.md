[README.md](https://github.com/user-attachments/files/23811292/README.md)
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

# ✅ **LICENSE.txt (Open, but IP-Protective)**  
This license allows academic use but preserves your sovereignty and prohibits derivative commercial exploitation.

```text
Navier–Stokes Unified Framework — Simulation Artifact Archive
Copyright (c) 2025 Paul M. Roe (TecKnows)

Permission is hereby granted for academic research, teaching, and peer review
purposes, including reproduction of figures and numerical results, provided that
proper attribution is given to the author.

Commercial use, redistribution, sublicensing, or creation of derivative
commercial works based on this archive or any associated RHEA-UCM technology is
strictly prohibited without explicit written permission from the copyright
holder.

This archive and all associated simulation code, entropy operators, Lorenz-based
feedback modules, GPU acceleration logic, and symbolic RHEA-UCM components are
protected under United States and international copyright and IP law.

All rights not explicitly granted are reserved.

Author:
    Paul M. Roe  
    TecKnows Inc., Androscoggin County, Maine  
    ORCID: 0009-0005-6159-947X

License Version:Tell me what you think:
 
🛡️ RHEA-Core Public Grant v1.0 — Unified Master Edition
(Non-Commercial · Attribution · No Derivatives · Symbolic Derivative Clause · Retroactive Supremacy)
Rights Holder: Paul M. Roe (EnigmaticGlitch)\
Patent Reference: U.S. Provisional Patent No. 63/796,404\
Copyright: © 2025 EnigmaticGlitch · All Rights Reserved
0. Retroactive License Supremacy Clause
By accessing, using, citing, distributing, or storing any version of this work, you agree that the most current license published by the original author supersedes and replaces all prior licenses, implied permissions, metadata tags, or public-domain assumptions.
This includes but is not limited to:

* CC-BY-NC-ND tags

* mistaken “open source” flags

* automatic repository licenses

* unauthorized journal mirrors

* web scrapes or AI ingestion

Unauthorized use under any former license constitutes infringement of the present license.
1. Scope of Coverage
This license applies to all public-facing components of:

* RHEA-UCM

* ZADEIAN-RHEA

* RHEA-CM

* RHEA-UCM-PMR

* Zadeian Sentinel (all versions)

* Microcircuit Glass-box Cognitive Substrate (MGCS)

* and any derivative research artifacts, simulations, equations, or symbolic systems authored by Paul M. Roe.

Unless explicitly exempted in writing, all materials are covered under this unified grant.
2. Grant of Use (Non-Commercial Only)
You are permitted to:

* View, read, and analyze the work

* Reference it for academic, journalistic, or personal purposes

* Discuss or critique it

* Use figures in slides/papers with attribution

This grant is:

* revocable

* non-exclusive

* non-transferable

* non-sublicensable

Commercial use is prohibited without explicit written authorization.
3. Attribution Requirements
Any authorized quotation, paraphrase, diagram, explanation, or excerpt must cite:
“© EnigmaticGlitch · RHEA-UCM / ZADEIAN-RHEA Framework · Patent Pending #63/796,404”
Failure to cite constitutes license breach.
4. No Derivatives
You may NOT:

* Modify, translate, rewrite, or extend the work

* Create derivative theories, frameworks, or models

* Produce alternative symbolic systems based on its equations

* Integrate the system into AI/ML frameworks

* Rewrite RHEA concepts under different terminology

This includes academic derivatives, rewritten technical documents, or “reinterpretations.”
5. Symbolic Derivative Clause (High-Security Clause)
You may not encode or embed core RHEA structures—such as:

* entropy modulation mechanisms

* recursive glyph modulation

* trust resealing logic

* UCM cosmological recursion

* Lorenz-entropy coupling

* symbolic meta-structures

* glyphic operator mappings

—under any alternate symbolic, visual, algebraic, or computational representation.
No “equivalent but differently named” re-expression is allowed.
This clause prevents symbolic laundering of the system’s core logic.
6. Prohibition on AI/ML Training
Without explicit written authorization:

* AI models may not be trained, fine-tuned, or evaluated using this work

* AI models may not ingest or vectorize this material

* AI companies may not use this material for embeddings or dataset augmentation

This applies to commercial and non-commercial AI systems alike.
7. Enforcement, Jurisdiction & Remedies
This license is enforced under:

* U.S. Copyright Law (Title 17)

* DMCA

* U.S. Patent Law (Provisional #63/796,404)

* WIPO Berne Convention Protections

Violations may trigger:

* formal DMCA takedowns

* cease & desist orders

* damages and statutory penalties

* injunctions

* revocation of all granted rights

All rights not explicitly granted are reserved.
8. License Revocation Grounds
The author reserves the right to revoke access in cases involving:

* predatory governmental clawbacks targeting impoverished citizens

* institutional obstruction of scientific innovation

* defunding of critical agencies essential to public welfare

* misuse or misrepresentation of RHEA-UCM theoretical work

* coercive scientific or medical practices violating consent

* unauthorized academic appropriation

* suppressed or biased peer review

* breach of SOER (Standard of Ethical Recursion) ethics

These grounds reflect the ethical foundations upon which RHEA-UCM is built.
9. Final Declaration
This unified license governs:

* all past versions

* all present versions

* all future versions

* all redistributions

* all mirrors

* all archives

* all Zenodo DOIs

* all GitHub snapshots

* all derivative storage systems

and is retroactively binding.
The canonical DOI for all versions is:
https://doi.org/10.5281/zenodo.15769823
10. Signature & Closing Statement
© 2025 · EnigmaticGlitch (Paul M. Roe)\
All Rights Reserved\
Patent Pending #63/796,404
“Trust is not given. It is oscillated into being — beat-by-beat, wave-by-wave — across the fabric of phase space.”\
— EnigmaticGlitch ♏🧙‍♂️
