# =====================================================================
# 3D Controlled Navier–Stokes Simulation (Torch/CUDA + Spectral)
# Matches the stable spectral scheme already in the paper.
#
# Produces:
#   • run01_energy_timeseries.npz
#   • fig_3D_bounds.png
# =====================================================================

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================
# TORCH / CUDA DETECTION (YOUR PATTERN)
# =============================
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
    DEVICE = "cpu"
    DEVICE_STATUS = "[CPU] Torch Not Installed"

print(DEVICE_STATUS)

# Normalize DEVICE as a torch.device if torch is present
if DEVICE_STATUS != "[CPU] Torch Not Installed":
    # torch is importable
    if isinstance(DEVICE, str):
        DEVICE = torch.device("cpu")
else:
    raise RuntimeError("Torch is required for this script.")

# ==============================================================
# Simulation parameters (same as paper; N=64 reviewer-safe)
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
# FFT wrappers (Torch/cuFFT)
# ==============================================================

def FFTc(u_hat):
    """Full complex-to-complex FFT using torch.fft."""
    return torch.fft.fftn(u_hat, dim=(-3, -2, -1))

def IFFTc(u_hat):
    """Inverse FFT, returns complex; .real for physical."""
    return torch.fft.ifftn(u_hat, dim=(-3, -2, -1))

# ==============================================================
# Dealiasing and projection (same as stable spectral version)
# ==============================================================

def dealias(u_hat):
    # Broadcast Kmask across component axis
    return torch.where(Kmask, torch.zeros_like(u_hat), u_hat)

def project(u_hat):
    # Leray projection in Fourier space
    k_dot_u = kx * u_hat[0] + ky * u_hat[1] + kz * u_hat[2]
    u_hat0 = u_hat[0] - kx * k_dot_u / k2
    u_hat1 = u_hat[1] - ky * k_dot_u / k2
    u_hat2 = u_hat[2] - kz * k_dot_u / k2
    return torch.stack([u_hat0, u_hat1, u_hat2], dim=0)

# ==============================================================
# Initial condition (div-free + noise), same as paper version
# ==============================================================

def init_condition():
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    u0 = np.sin(X) * np.cos(Y) * np.cos(Z)
    u1 = -np.cos(X) * np.sin(Y) * np.cos(Z)
    u2 = np.zeros_like(u0)

    u_np = np.stack([u0, u1, u2])  # shape (3, N, N, N)
    u = t_from_np(u_np, dtype=DTYPE_REAL)

    # small divergence-free perturbation
    np.random.seed(42)
    noise_np = np.random.normal(scale=3e-4, size=u_np.shape)
    noise = t_from_np(noise_np, dtype=DTYPE_REAL)

    noise_hat = FFTc(noise.to(DTYPE_CPLX))
    noise_hat = project(dealias(noise_hat))
    noise_phys = IFFTc(noise_hat).real  # back to real

    u = u + noise_phys
    return u

# ==============================================================
# Nonlinear term (pseudo-spectral), same structure as stable code
# ==============================================================

def nonlinear_term(u_hat):
    # 1) dealias and go to physical
    u_hat = dealias(u_hat)
    u = IFFTc(u_hat).real  # (3, N, N, N)

    # 2) spectral derivatives of each component
    ux_hat = ikx * u_hat
    uy_hat = iky * u_hat
    uz_hat = ikz * u_hat

    ux = IFFTc(ux_hat).real
    uy = IFFTc(uy_hat).real
    uz = IFFTc(uz_hat).real

    # 3) (u · ∇)u, component-wise
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
# Entropy functional (keep as NumPy, like before)
# ==============================================================

def compute_entropy(u_phys):
    # u_phys: torch real, shape (3, N, N, N)
    u_np = to_np(u_phys)
    wx = np.gradient(u_np[2], axis=1) - np.gradient(u_np[1], axis=2)
    wy = np.gradient(u_np[0], axis=2) - np.gradient(u_np[2], axis=0)
    wz = np.gradient(u_np[1], axis=0) - np.gradient(u_np[0], axis=1)
    w2 = wx**2 + wy**2 + wz**2
    return np.sum(w2 * np.log1p(w2))

# ==============================================================
# MAIN SIMULATION (semi-implicit, paper-consistent)
# ==============================================================

def run_sim():
    u = init_condition()                                 # physical, real
    u_hat = FFTc(u.to(DTYPE_CPLX))                      # spectral
    u_hat = project(dealias(u_hat))

    t_vals = []
    E_L2 = []
    grad_norm = []
    F_norm = []
    entropy_vals = []

    denom = 1.0 + dt * nu * k2

    print(f"[+] Running 3D controlled NS: N={N}, Nt={Nt}, device={DEVICE}")
    for n in tqdm(range(Nt), desc="Time-stepping"):

        N_hat = nonlinear_term(u_hat)
        F_hat = feedback_operator(u_hat)

        # semi-implicit: (I + dt ν Δ) in Fourier is (1 + dt ν k²)
        u_hat = (u_hat - dt * N_hat + dt * F_hat) / denom

        u_hat = project(dealias(u_hat))

        if n % 50 == 0:
            u_phys = IFFTc(u_hat).real  # torch, real

            # 1. velocity L2 norm
            E_L2.append(float(torch.sqrt(torch.mean(u_phys**2)).cpu()))

            # 2. gradient norm ||∇u||_2 using NumPy gradients
            u_np = to_np(u_phys)
            gsum = 0.0
            for i in range(3):
                gx, gy, gz = np.gradient(u_np[i])
                gsum += np.sum(gx**2) + np.sum(gy**2) + np.sum(gz**2)
            grad_norm.append(np.sqrt(gsum))

            # 3. feedback norm ||Fε[u]||_2
            F_phys = IFFTc(F_hat).real
            F_norm.append(float(torch.sqrt(torch.sum(F_phys**2)).cpu()))

            # 4. entropy
            entropy_vals.append(compute_entropy(u_phys))

            t_vals.append(n * dt)

    np.savez(
        "run01_energy_timeseries.npz",
        t=np.array(t_vals),
        E_L2=np.array(E_L2),
        grad_norm=np.array(grad_norm),
        F_norm=np.array(F_norm),
        entropy=np.array(entropy_vals),
    )

# ==============================================================
# Plotting (Figure 4 style)
# ==============================================================

def plot_results():
    d = np.load("run01_energy_timeseries.npz")
    t = d["t"]

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    ax[0, 0].plot(t, d["E_L2"])
    ax[0, 0].set_title("‖u(t)‖₂")

    ax[0, 1].plot(t, d["grad_norm"])
    ax[0, 1].set_title("‖∇u(t)‖₂")

    ax[1, 0].plot(t, d["F_norm"])
    ax[1, 0].set_title("‖Fε[u(t)]‖₂")

    ax[1, 1].plot(t, d["entropy"])
    ax[1, 1].set_title("Entropy E(t)")

    fig.suptitle("3D Controlled Navier–Stokes Diagnostics (Figure 4)")
    fig.tight_layout()
    plt.savefig("fig_3D_bounds.png", dpi=200)
    plt.close()

# ==============================================================
# Run
# ==============================================================

if __name__ == "__main__":
    run_sim()
    plot_results()
    print("[+] Simulation complete. Saved fig_3D_bounds.png")