# =====================================================================
# 3D Controlled Navier–Stokes – Entropy Functional E(t) (Figure 6)
#
# Produces (relative to project root Entropy_NS_Artifact/):
#   • data_archive/simulations/3d_entropy/run01_entropy.npz
#   • data_archive/figures/fig_3D_entropy.png
#   • data_archive/checksums/run01_entropy.sha256
#   • data_archive/metadata/run01_entropy_metadata.json
# =====================================================================

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================
# TORCH / CUDA DETECTION (YOUR PATTERN)
# ============================================================
try:
    import torch
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        DEVICE_STATUS = "[GPU] CUDA Detected — Accelerated"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        DEVICE_STATUS = "[GPU] MPS Detected — Accelerated"
    else:
        DEVICE = torch.device("cpu")
        DEVICE_STATUS = "[CPU] Torch Available"
except ImportError:
    raise RuntimeError("Torch is required for this script.")

print(DEVICE_STATUS)

# ============================================================
# DIRECTORY SETUP  (YOUR EXACT BLOCK)
# ============================================================
OUTPUT_DIR = "data_archive"
SUBDIR = f"{OUTPUT_DIR}/simulations/3d_entropy"
FIGDIR = f"{OUTPUT_DIR}/figures"
METADIR = f"{OUTPUT_DIR}/metadata"
CHECKDIR = f"{OUTPUT_DIR}/checksums"

for d in [SUBDIR, FIGDIR, METADIR, CHECKDIR]:
    if not os.path.exists(d):
        os.makedirs(d)

NPZ_PATH = Path(SUBDIR) / "run01_entropy.npz"
FIG_PATH = Path(FIGDIR) / "fig_3D_entropy.png"
SHA_PATH = Path(CHECKDIR) / "run01_entropy.sha256"
META_PATH = Path(METADIR) / "run01_entropy_metadata.json"

# ============================================================
# SIMULATION PARAMETERS
# ============================================================
N = 64
nu = 0.005
eps = 0.02
Tfinal = 10.0
dt = 5e-4
Nt = int(Tfinal / dt)
L = 2 * np.pi

# ============================================================
# TORCH HELPERS
# ============================================================
DTYPE_REAL = torch.float64
DTYPE_CPLX = torch.complex128

def t_from_np(x, dtype=DTYPE_REAL):
    return torch.from_numpy(x).to(DEVICE, dtype=dtype)

def to_np(x):
    return x.detach().cpu().numpy()

# ============================================================
# GRID + SPECTRAL CONSTANTS
# ============================================================
k = np.fft.fftfreq(N, 1.0 / N) * 2 * np.pi
kx_np, ky_np, kz_np = np.meshgrid(k, k, k, indexing="ij")

kx = t_from_np(kx_np)
ky = t_from_np(ky_np)
kz = t_from_np(kz_np)

k2 = kx**2 + ky**2 + kz**2
k2[0, 0, 0] = 1.0

ikx = 1j * kx
iky = 1j * ky
ikz = 1j * kz

# Dealias mask
cut = N // 3
Kmask_np = (np.abs(kx_np) > cut) | (np.abs(ky_np) > cut) | (np.abs(kz_np) > cut)
Kmask = torch.from_numpy(Kmask_np).to(DEVICE, dtype=torch.bool)

# ============================================================
# FFT (cuFFT via torch)
# ============================================================
def FFTc(u_hat):
    return torch.fft.fftn(u_hat, dim=(-3, -2, -1))

def IFFTc(u_hat):
    return torch.fft.ifftn(u_hat, dim=(-3, -2, -1))

# ============================================================
# Dealias + projection
# ============================================================
def dealias(u_hat):
    return torch.where(Kmask, torch.zeros_like(u_hat), u_hat)

def project(u_hat):
    k_dot_u = kx*u_hat[0] + ky*u_hat[1] + kz*u_hat[2]
    return torch.stack([
        u_hat[0] - kx*k_dot_u/k2,
        u_hat[1] - ky*k_dot_u/k2,
        u_hat[2] - kz*k_dot_u/k2
    ], dim=0)

# ============================================================
# Initial condition
# ============================================================
def init_condition():
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    u0 = np.sin(X)*np.cos(Y)*np.cos(Z)
    u1 = -np.cos(X)*np.sin(Y)*np.cos(Z)
    u2 = np.zeros_like(u0)
    u_np = np.stack([u0, u1, u2])

    u = t_from_np(u_np)

    np.random.seed(42)
    noise_np = np.random.normal(scale=1e-3, size=u_np.shape)
    noise = t_from_np(noise_np)

    noise_hat = FFTc(noise.to(DTYPE_CPLX))
    noise_hat = project(dealias(noise_hat))
    noise_phys = IFFTc(noise_hat).real

    return u + noise_phys

# ============================================================
# Nonlinear term
# ============================================================
def nonlinear_term(u_hat):
    u_hat = dealias(u_hat)
    u = IFFTc(u_hat).real

    ux = IFFTc(ikx*u_hat).real
    uy = IFFTc(iky*u_hat).real
    uz = IFFTc(ikz*u_hat).real

    N = torch.stack([
        u[0]*ux[0] + u[1]*uy[0] + u[2]*uz[0],
        u[0]*ux[1] + u[1]*uy[1] + u[2]*uz[1],
        u[0]*ux[2] + u[1]*uy[2] + u[2]*uz[2],
    ], dim=0)

    return dealias(FFTc(N.to(DTYPE_CPLX)))

# ============================================================
# Feedback operator
# ============================================================
def feedback_operator(u_hat):
    return -eps * (1 + k2) * u_hat

# ============================================================
# Entropy functional
# ============================================================
def compute_entropy(u_phys):
    u_np = to_np(u_phys)
    wx = np.gradient(u_np[2], axis=1) - np.gradient(u_np[1], axis=2)
    wy = np.gradient(u_np[0], axis=2) - np.gradient(u_np[2], axis=0)
    wz = np.gradient(u_np[1], axis=0) - np.gradient(u_np[0], axis=1)
    w2 = wx**2 + wy**2 + wz**2
    return float(np.sum(w2 * np.log1p(w2)))

# ============================================================
# MAIN SIM
# ============================================================
def run_entropy_sim():
    u = init_condition()
    u_hat = FFTc(u.to(DTYPE_CPLX))
    u_hat = project(dealias(u_hat))

    denom = 1 + dt*nu*k2

    t_vals = []
    entropy_vals = []

    print(f"[+] Running 3D entropy run: N={N}, Nt={Nt}, device={DEVICE}")

    for n in tqdm(range(Nt), desc="Time-stepping"):
        N_hat = nonlinear_term(u_hat)
        F_hat = feedback_operator(u_hat)

        u_hat = (u_hat - dt*N_hat + dt*F_hat) / denom
        u_hat = project(dealias(u_hat))

        if n % 50 == 0:
            u_phys = IFFTc(u_hat).real
            entropy_vals.append(compute_entropy(u_phys))
            t_vals.append(n*dt)

    np.savez(NPZ_PATH, t=np.array(t_vals), entropy=np.array(entropy_vals))
    return np.array(t_vals), np.array(entropy_vals)

# ============================================================
# Checksum + metadata
# ============================================================
def compute_sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def write_metadata(t_arr, entropy_arr, sha_val):
    meta = {
        "simulation": "3D Controlled NS - Entropy",
        "figure": "Figure 6",
        "npz": str(NPZ_PATH),
        "figure_path": str(FIG_PATH),
        "sha256": sha_val,
        "parameters": {
            "N": N, "nu": nu, "eps": eps,
            "dt": dt, "Tfinal": Tfinal, "Nt": Nt
        },
        "device": str(DEVICE),
        "created": datetime.utcnow().isoformat()+"Z"
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

# ============================================================
# Plot
# ============================================================
def plot_entropy(t_arr, entropy_arr):
    plt.figure(figsize=(8,6))
    plt.plot(t_arr, entropy_arr)
    plt.xlabel("t")
    plt.ylabel("E(t)")
    plt.title("Entropy Functional E(t) (Figure 6)")
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=200)
    plt.close()

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    t_arr, entropy_arr = run_entropy_sim()
    sha = compute_sha256(NPZ_PATH)
    with open(SHA_PATH, "w") as f:
        f.write(f"{sha}  {NPZ_PATH.name}\n")
    write_metadata(t_arr, entropy_arr, sha)
    plot_entropy(t_arr, entropy_arr)

    print("[+] Entropy run complete.")
    print("    NPZ:", NPZ_PATH)
    print("    FIG:", FIG_PATH)
    print("    SHA:", SHA_PATH)
    print("    META:", META_PATH)