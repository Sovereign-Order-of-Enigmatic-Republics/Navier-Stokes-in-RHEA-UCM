# =====================================================================
# 3D Controlled Navier–Stokes – Spectral Shell Energies (Torch/CUDA)
# Produces:
#   • data_archive/simulations/3d_spectrum/run01_spectrum.npz (t, Em, k_shells)
#   • data_archive/figures/fig_3D_spectrum.png                (Figure 5)
#   • data_archive/checksums/run01_spectrum.sha256            (SHA-256 checksum)
#   • data_archive/metadata/run01_spectrum_metadata.json      (run metadata)
#
# Dynamics and parameters match the controlled 3D scheme used for Fig. 4.
# Run from project root: Entropy_NS_Artifact/
# =====================================================================

import os
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ====================================
# OUTPUT DIRECTORY LAYOUT
# ====================================
ROOT = os.path.abspath(".")  # assume we run from project root

DIR_SIM = os.path.join(ROOT, "data_archive", "simulations", "3d_spectrum")
DIR_FIG = os.path.join(ROOT, "data_archive", "figures")
DIR_META = os.path.join(ROOT, "data_archive", "metadata")
DIR_HASH = os.path.join(ROOT, "data_archive", "checksums")

for d in [DIR_SIM, DIR_FIG, DIR_META, DIR_HASH]:
    os.makedirs(d, exist_ok=True)

NPZ_PATH = os.path.join(DIR_SIM, "run01_spectrum.npz")
FIG_PATH = os.path.join(DIR_FIG, "fig_3D_spectrum.png")
META_PATH = os.path.join(DIR_META, "run01_spectrum_metadata.json")
HASH_PATH = os.path.join(DIR_HASH, "run01_spectrum.sha256")

print("[+] Output layout:")
print("    Sim NPZ:   ", NPZ_PATH)
print("    Figure:    ", FIG_PATH)
print("    Metadata:  ", META_PATH)
print("    Checksum:  ", HASH_PATH)

# ====================================
# TORCH / CUDA DETECTION (YOUR PATTERN)
# ====================================
try:
    import torch
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        DEVICE_STATUS = "[GPU] CUDA Detected — Accelerated"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        DEVICE_STATUS = "[GPU] MPS Detected — Accelerated"
    else:
        DEVICE = "cpu"
        DEVICE_STATUS = "[CPU] Torch Available"
except ImportError:
    raise RuntimeError("Torch is required for this script and was not found.")

print(DEVICE_STATUS)

if isinstance(DEVICE, str):
    DEVICE = torch.device("cpu")

# ==============================================================
# Simulation parameters (same as Fig. 4, reviewer-safe N=64)
# ==============================================================
N = 64          # use 128 or 160 for full-author mode
nu = 0.005
eps = 0.02
Tfinal = 10.0
dt = 5e-4
Nt = int(Tfinal / dt)
L = 2 * np.pi

# ==============================================================
# Torch helpers
# ==============================================================
DTYPE_REAL = torch.float64
DTYPE_CPLX = torch.complex128

def t_from_np(x, dtype=DTYPE_REAL):
    return torch.from_numpy(x).to(DEVICE, dtype=dtype)

def to_np(x):
    return x.detach().cpu().numpy()

# ==============================================================
# Build grid + spectral constants
# ==============================================================
k = np.fft.fftfreq(N, 1.0 / N) * 2 * np.pi
kx_np, ky_np, kz_np = np.meshgrid(k, k, k, indexing="ij")

kx = t_from_np(kx_np)
ky = t_from_np(ky_np)
kz = t_from_np(kz_np)
k2 = kx**2 + ky**2 + kz**2
k2[0, 0, 0] = 1.0  # avoid zero mode division

ikx = 1j * kx
iky = 1j * ky
ikz = 1j * kz

# 2/3 dealias mask (same as before)
cut = N // 3
Kmask_np = (np.abs(kx_np) > cut) | (np.abs(ky_np) > cut) | (np.abs(kz_np) > cut)
Kmask = torch.from_numpy(Kmask_np).to(DEVICE, dtype=torch.bool)

# ==============================================================
# Shell index for spectral energy E_m(t)
# ==============================================================
rad_np = np.sqrt(kx_np**2 + ky_np**2 + kz_np**2)
rad_nonzero = rad_np[rad_np > 0]
k0 = rad_nonzero.min()          # fundamental radial step
shell_id_np = np.floor(rad_np / k0 + 0.5).astype(int)
shell_id_np[rad_np == 0.0] = 0  # zero mode
shell_id_flat = shell_id_np.ravel()
NUM_SHELLS = int(shell_id_flat.max()) + 1
k_shells = np.arange(NUM_SHELLS)

# ==============================================================
# FFT wrappers (Torch/cuFFT)
# ==============================================================
def FFTc(u_hat):
    return torch.fft.fftn(u_hat, dim=(-3, -2, -1))

def IFFTc(u_hat):
    return torch.fft.ifftn(u_hat, dim=(-3, -2, -1))

# ==============================================================
# Dealiasing and projection
# ==============================================================
def dealias(u_hat):
    return torch.where(Kmask, torch.zeros_like(u_hat), u_hat)

def project(u_hat):
    k_dot_u = kx * u_hat[0] + ky * u_hat[1] + kz * u_hat[2]
    u_hat0 = u_hat[0] - kx * k_dot_u / k2
    u_hat1 = u_hat[1] - ky * k_dot_u / k2
    u_hat2 = u_hat[2] - kz * k_dot_u / k2
    return torch.stack([u_hat0, u_hat1, u_hat2], dim=0)

# ==============================================================
# Initial condition (div-free + noise)
# ==============================================================
def init_condition():
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    u0 = np.sin(X) * np.cos(Y) * np.cos(Z)
    u1 = -np.cos(X) * np.sin(Y) * np.cos(Z)
    u2 = np.zeros_like(u0)
    u_np = np.stack([u0, u1, u2])  # shape (3, N, N, N)

    u = t_from_np(u_np, dtype=DTYPE_REAL)

    np.random.seed(42)
    noise_np = np.random.normal(scale=1e-3, size=u_np.shape)
    noise = t_from_np(noise_np, dtype=DTYPE_REAL)

    noise_hat = FFTc(noise.to(DTYPE_CPLX))
    noise_hat = project(dealias(noise_hat))
    noise_phys = IFFTc(noise_hat).real
    u = u + noise_phys
    return u

# ==============================================================
# Nonlinear term (pseudo-spectral)
# ==============================================================
def nonlinear_term(u_hat):
    u_hat = dealias(u_hat)
    u = IFFTc(u_hat).real

    ux_hat = ikx * u_hat
    uy_hat = iky * u_hat
    uz_hat = ikz * u_hat

    ux = IFFTc(ux_hat).real
    uy = IFFTc(uy_hat).real
    uz = IFFTc(uz_hat).real

    N0 = u[0] * ux[0] + u[1] * uy[0] + u[2] * uz[0]
    N1 = u[0] * ux[1] + u[1] * uy[1] + u[2] * uz[1]
    N2 = u[0] * ux[2] + u[1] * uy[2] + u[2] * uz[2]

    N = torch.stack([N0, N1, N2], dim=0).to(DTYPE_REAL)
    N_hat = FFTc(N.to(DTYPE_CPLX))
    return dealias(N_hat)

# ==============================================================
# Feedback stabilizer Fε = -eps*(1 + k²) u_hat
# ==============================================================
def feedback_operator(u_hat):
    return -eps * (1.0 + k2) * u_hat

# ==============================================================
# MAIN SIMULATION – accumulate shell energies E_m(t)
# ==============================================================
def run_spectrum_sim():
    u = init_condition()
    u_hat = FFTc(u.to(DTYPE_CPLX))
    u_hat = project(dealias(u_hat))

    denom = 1.0 + dt * nu * k2

    t_snap = []
    Em_list = []

    print(f"[+] Running 3D spectrum run: N={N}, Nt={Nt}, device={DEVICE}")

    # choose same diagnostic cadence as Fig. 4
    diag_every = 50

    for n in tqdm(range(Nt), desc="Time-stepping"):
        N_hat = nonlinear_term(u_hat)
        F_hat = feedback_operator(u_hat)

        u_hat = (u_hat - dt * N_hat + dt * F_hat) / denom
        u_hat = project(dealias(u_hat))

        if n % diag_every == 0:
            # spectral energy density summed over components
            u_hat_np = to_np(u_hat)          # (3, N, N, N), complex
            energy_density = np.sum(np.abs(u_hat_np)**2, axis=0)  # (N, N, N)

            E_shell = np.bincount(
                shell_id_flat,
                weights=energy_density.ravel(),
                minlength=NUM_SHELLS,
            )

            Em_list.append(E_shell)
            t_snap.append(n * dt)

    Em = np.stack(Em_list, axis=1)  # shape (NUM_SHELLS, n_times)

    np.savez(
        NPZ_PATH,
        t=np.array(t_snap),
        Em=Em,
        k_shells=k_shells,
    )

# ==============================================================
# Plotting – Figure 5
# ==============================================================
def plot_spectrum():
    d = np.load(NPZ_PATH)
    t = d["t"]
    Em = d["Em"]          # (NUM_SHELLS, n_times)
    k_shells_arr = d["k_shells"]

    max_shell_to_plot = min(10, Em.shape[0] - 1)

    plt.figure(figsize=(8, 6))
    for m in range(1, max_shell_to_plot + 1):
        plt.semilogy(t, Em[m], label=f"m = {k_shells_arr[m]}")

    plt.xlabel("t")
    plt.ylabel(r"$E_m(t)$")
    plt.title("Spectral shell energies $E_m(t)$ (Figure 5)")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=200)
    plt.close()

# ==============================================================
# Checksum + Metadata
# ==============================================================
def write_checksum_and_metadata():
    # SHA-256 checksum of NPZ
    sha = hashlib.sha256()
    with open(NPZ_PATH, "rb") as f:
        sha.update(f.read())
    digest = sha.hexdigest()
    with open(HASH_PATH, "w") as f:
        f.write(digest + "\n")

    # Metadata JSON
    meta = {
        "description": "3D controlled Navier–Stokes spectral shell energies (Figure 5)",
        "npz_file": NPZ_PATH,
        "figure_file": FIG_PATH,
        "checksum_file": HASH_PATH,
        "device": str(DEVICE),
        "device_status": DEVICE_STATUS,
        "N": N,
        "nu": nu,
        "eps": eps,
        "L": L,
        "Tfinal": Tfinal,
        "dt": dt,
        "Nt": Nt,
        "diag_every": 50,
        "num_shells": int(NUM_SHELLS),
        "k_shells_min": int(k_shells.min()),
        "k_shells_max": int(k_shells.max()),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=4)

    print("[+] SHA-256:", digest)
    print("[+] Metadata written to:", META_PATH)

# ==============================================================
# Run
# ==============================================================
if __name__ == "__main__":
    run_spectrum_sim()
    plot_spectrum()
    write_checksum_and_metadata()
    print("[+] Spectrum simulation complete.")
    print("[+] Saved:")
    print("    -", NPZ_PATH)
    print("    -", FIG_PATH)
    print("    -", HASH_PATH)
    print("    -", META_PATH)