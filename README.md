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

License Version: 1.0 — January 2025