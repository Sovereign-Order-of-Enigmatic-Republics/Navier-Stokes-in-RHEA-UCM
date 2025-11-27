import numpy as np
import os, json, hashlib, time
import matplotlib.pyplot as plt

try:
    import torch
    USE_GPU = torch.cuda.is_available()
except:
    USE_GPU = False

# ============================================================
# Directory setup
# ============================================================
OUTPUT_DIR = "data_archive"
SUBDIR = f"{OUTPUT_DIR}/simulations/3d_svv"
FIGDIR = f"{OUTPUT_DIR}/figures"
METADIR = f"{OUTPUT_DIR}/metadata"
CHECKDIR = f"{OUTPUT_DIR}/checksums"

for d in [SUBDIR, FIGDIR, METADIR, CHECKDIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# ============================================================
# Spectral Vanishing Viscosity σ(k)
# ============================================================
def svv_sigma(k, k_cut, k_max):
    """Smoothly increasing filter: 0 for low modes, grows near Nyquist."""
    sigma = np.zeros_like(k)
    mask = k > k_cut
    sigma[mask] = ((k[mask] - k_cut) / (k_max - k_cut)) ** 4
    return sigma

# ============================================================
# Leray Projection: P = I - kk^T / |k|^2
# ============================================================
def project(u_hat, kx, ky, kz):
    k2 = kx**2 + ky**2 + kz**2
    k2[0,0,0] = 1.0  # avoid divide-by-zero

    div_hat = (kx*u_hat[0] + ky*u_hat[1] + kz*u_hat[2]) / k2

    u_hat[0] -= div_hat * kx
    u_hat[1] -= div_hat * ky
    u_hat[2] -= div_hat * kz
    return u_hat

# ============================================================
# 3D SVV Navier-Stokes Solver
# ============================================================
def solve_ns_3d(n=64, dt=2e-3, T=0.5, nu=0.01, eps=0.1):
    device = torch.device("cuda") if USE_GPU else torch.device("cpu")

    L = 2*np.pi
    x = np.linspace(0, L, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    # Initial condition
    u = np.zeros((3, n, n, n))
    u[0] = np.sin(X) * np.sin(Y) * np.sin(Z)
    u[1] = np.sin(X) * np.cos(Y) * np.sin(Z)
    u[2] = np.cos(X) * np.sin(Y) * np.sin(Z)

    # Frequencies
    k = np.fft.fftfreq(n, 1/n) * 2*np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    k2 = kx**2 + ky**2 + kz**2
    k_mag = np.sqrt(k2)

    # SVV filter
    k_cut = 0.65 * np.max(k_mag)
    k_max = np.max(k_mag)
    sigma = svv_sigma(k_mag, k_cut, k_max)

    steps = int(T / dt)
    energy_log = []

    for step in range(steps):

        # FFT
        u_hat = np.fft.fftn(u, axes=(1,2,3))

        # Nonlinear term (u·∇)u via spectral differentiation
        ux = np.fft.ifftn(1j*kx*u_hat[0], axes=(0,1,2)).real
        uy = np.fft.ifftn(1j*ky*u_hat[0], axes=(0,1,2)).real
        uz = np.fft.ifftn(1j*kz*u_hat[0], axes=(0,1,2)).real

        vx = np.fft.ifftn(1j*kx*u_hat[1], axes=(0,1,2)).real
        vy = np.fft.ifftn(1j*ky*u_hat[1], axes=(0,1,2)).real
        vz = np.fft.ifftn(1j*kz*u_hat[1], axes=(0,1,2)).real

        wx = np.fft.ifftn(1j*kx*u_hat[2], axes=(0,1,2)).real
        wy = np.fft.ifftn(1j*ky*u_hat[2], axes=(0,1,2)).real
        wz = np.fft.ifftn(1j*kz*u_hat[2], axes=(0,1,2)).real

        nonlinear = np.zeros_like(u)
        nonlinear[0] = u[0]*ux + u[1]*uy + u[2]*uz
        nonlinear[1] = u[0]*vx + u[1]*vy + u[2]*vz
        nonlinear[2] = u[0]*wx + u[1]*wy + u[2]*wz

        nonlinear_hat = np.fft.fftn(nonlinear, axes=(1,2,3))

        # SVV feedback: -eps * σ(k) * u_hat
        F_hat = -eps * sigma * u_hat

        # Time stepping
        denom = 1 + dt * (nu*k2 + eps*sigma)
        u_hat_new = (u_hat - dt * nonlinear_hat + dt * F_hat) / denom

        # Leray projection
        u_hat_new = project(u_hat_new, kx, ky, kz)

        # Back to physical space
        u = np.fft.ifftn(u_hat_new, axes=(1,2,3)).real

        # Energy
        energy_log.append(np.mean(u*u))

        if step % 20 == 0:
            print(f"step={step}/{steps}  Energy={energy_log[-1]:.6e}")

    return u, np.array(energy_log)


# ============================================================
# Run simulation and store data
# ============================================================

print("Running 3D entropy-stabilized Navier–Stokes simulation...")

u_final, energy = solve_ns_3d()

np.savez(f"{OUTPUT_DIR}/simulations/3d_entropy_stabilized/run01_velocity_frames.npz",
         u_final=u_final)

np.save(f"{OUTPUT_DIR}/simulations/3d_entropy_stabilized/run01_energy_timeseries.npy",
        energy)

# Metadata
metadata = {
    "simulation": "3D entropy-stabilized Navier–Stokes (RHEA-UCM)",
    "grid_resolution": 64,
    "T": 0.5,
    "dt": 1e-3,
    "viscosity": 0.01,
    "gpu_used": USE_GPU
}
with open(f"{OUTPUT_DIR}/metadata/3d_entropy_ns.json", "w") as f:
    json.dump(metadata, f, indent=4)


# Checksums
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

paths = [
    f"{OUTPUT_DIR}/simulations/3d_entropy_stabilized/run01_velocity_frames.npz",
    f"{OUTPUT_DIR}/simulations/3d_entropy_stabilized/run01_energy_timeseries.npy"
]

with open(f"{OUTPUT_DIR}/checksums/3d_SHA256SUMS.txt", "w") as f:
    for p in paths:
        f.write(f"{sha256(p)}  {os.path.basename(p)}\n")

# ============================================================
# FIGURE GENERATION (Required 3D plots)
# ============================================================

import matplotlib.pyplot as plt

print("Generating 3D diagnostic figures...")

# ---- (1) Energy Timeseries ----
plt.figure(figsize=(8,5))
plt.plot(energy, color="darkblue", linewidth=2)
plt.xlabel("Time step", fontsize=12)
plt.ylabel("Energy = ⟨u^2⟩", fontsize=12)
plt.title("3D Entropy-Stabilized Navier–Stokes: Energy Evolution")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figures/fig_3d_energy_bound.png", dpi=300)
plt.close()

# ---- (2) Entropy Equilibrium ----
# Compute entropy E = ⟨|u|^2 log(1 + |u|^2)⟩
u_final_mag2 = np.sum(u_final**2, axis=0)
entropy_field = u_final_mag2 * np.log(1 + u_final_mag2 + 1e-8)
entropy_slice = entropy_field[:, :, u_final.shape[-1]//2]  # mid Z slice

plt.figure(figsize=(6,5))
plt.imshow(entropy_slice, cmap="inferno", origin="lower")
plt.colorbar(label="Entropy Density E(x,y)")
plt.title("Mid-Plane Entropy Slice (z = L/2)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figures/fig_3d_entropy_equilibrium.png", dpi=300)
plt.close()

# ---- (3) Velocity Amplitude Slice ----
vel_slice = np.sqrt(u_final[0]**2 + u_final[1]**2 + u_final[2]**2)
vel_slice2d = vel_slice[:, :, u_final.shape[-1]//2]  # mid Z slice

plt.figure(figsize=(6,5))
plt.imshow(vel_slice2d, cmap="viridis", origin="lower")
plt.colorbar(label="|u(x,y)|")
plt.title("3D Velocity Magnitude — Mid-Plane Slice")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figures/fig_3d_velocity_slice.png", dpi=300)
plt.close()

print("3D figures saved to data_archive/figures/")


print("3D simulation complete.")
