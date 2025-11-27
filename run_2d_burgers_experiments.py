import numpy as np
import json, os, hashlib, time
import matplotlib.pyplot as plt

# ============================================================
# 2D Viscous Burgers Experiment Suite
# Generates: .npz data, convergence figures, metadata, checksums
# ============================================================

OUTPUT_DIR = "data_archive"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Create directory structure
ensure_dir(f"{OUTPUT_DIR}/simulations/2d_vanishing_feedback")
ensure_dir(f"{OUTPUT_DIR}/figures")
ensure_dir(f"{OUTPUT_DIR}/metadata")
ensure_dir(f"{OUTPUT_DIR}/checksums")

# ------------------------------------------------------------
# 2D Solver — Spectral Burgers
# ------------------------------------------------------------

def solve_burgers_2d(eps, nx=256, dt=2e-4, T=1.0):
    """
    2D viscous Burgers equation with feedback operator:
        F_eps[u] = -eps * (I - Δ) u
    on the periodic 2-torus.

    This operator satisfies (A1)-(A4).
    """

    # --- Grid setup ---
    L = 2*np.pi
    x = np.linspace(0, L, nx, endpoint=False)
    y = np.linspace(0, L, nx, endpoint=False)
    X, Y = np.meshgrid(x, y)

    # Initial condition from the paper
    u = np.sin(X) * np.sin(Y)

    # Fourier grid
    k = np.fft.fftfreq(nx, 1/nx) * 2*np.pi
    kx, ky = np.meshgrid(k, k)
    k2 = kx**2 + ky**2

    # 2/3 anti-aliasing (spectral dealiasing)
    cutoff = (2/3)*(nx//2)
    mask = (np.abs(kx) < cutoff) & (np.abs(ky) < cutoff)

    # Time-stepping
    steps = int(T/dt)
    L2_list = []
    H1_list = []

    for step in range(steps):

        # Compute derivatives
        u_hat = np.fft.fft2(u)
        ux = np.real(np.fft.ifft2(1j*kx*u_hat))
        uy = np.real(np.fft.ifft2(1j*ky*u_hat))

        # Diagnostics
        L2 = np.sqrt(np.mean(u*u))
        H1 = np.sqrt(np.mean(u*u) + np.mean(ux*ux) + np.mean(uy*uy))

        L2_list.append(L2)
        H1_list.append(H1)

        # Nonlinear term (u·∇u)
        nonlinear = u*ux + u*uy
        nonlinear_hat = np.fft.fft2(nonlinear)

        # Dealiasing
        u_hat *= mask
        nonlinear_hat *= mask

        # --- FEEDBACK OPERATOR ---
        # F_eps[u] = -eps * (I - Δ) u
        # In Fourier space:
        #   ˆF_eps[u](k) = -eps * (1 + |k|^2) * u_hat(k)
        F_hat = -eps * (1 + k2) * u_hat

        # Semi-implicit update:
        #   (I - dt*eps*(I - Δ)) û^{n+1} =
        #       û^n - dt * N̂(u)
        denom = (1 + dt*eps*(1 + k2))
        u_hat_new = (u_hat - dt * nonlinear_hat) / denom

        # Inverse FFT to update solution
        u = np.real(np.fft.ifft2(u_hat_new))

    return np.array(L2_list), np.array(H1_list), u

# ------------------------------------------------------------
# Run experiment suite
# ------------------------------------------------------------

EXPS = [0.2, 0.1, 0.05, 0.0]

metadata = {
    "equations": "2D viscous Burgers with vanishing feedback",
    "grid_resolution": 256,
    "dt": 0.002,
    "T": 1.0,
    "runs": []
}

for eps in EXPS:
    print(f"=== Running eps={eps} ===")
    L2, H1, u_final = solve_burgers_2d(eps)

    out_file = f"{OUTPUT_DIR}/simulations/2d_vanishing_feedback/eps_{eps}_run01.npz"
    np.savez(out_file, L2=L2, H1=H1, u_final=u_final)

    metadata["runs"].append({
        "eps": eps,
        "file": os.path.abspath(out_file),
        "max_H1": float(H1.max())
    })

    # ------------------ Convergence Figures ------------------
    plt.figure(figsize=(8,5))
    plt.plot(H1, label=f"eps={eps}")
    plt.xlabel("Step")
    plt.ylabel("H1 Norm")
    plt.title(f"2D Burgers Convergence (eps={eps})")
    plt.legend()
    fig_path = f"{OUTPUT_DIR}/figures/2d_H1_eps_{eps}.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

# ------------------ Write Metadata ------------------

with open(f"{OUTPUT_DIR}/metadata/2d_vanishing_feedback.json", "w") as f:
    json.dump(metadata, f, indent=4)

# ------------------ Checksums ------------------

def sha256sum(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

checksum_file = f"{OUTPUT_DIR}/checksums/2d_SHA256SUMS.txt"
with open(checksum_file, "w") as cs:
    for eps in EXPS:
        fpath = f"{OUTPUT_DIR}/simulations/2d_vanishing_feedback/eps_{eps}_run01.npz"
        cs.write(f"{sha256sum(fpath)}  {os.path.basename(fpath)}\n")

print("2D experiment suite complete.")
