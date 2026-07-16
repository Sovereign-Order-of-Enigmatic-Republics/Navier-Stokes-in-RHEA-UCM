from __future__ import annotations

import numpy as np


def periodic_wavenumbers(n: int, length: float = 2 * np.pi) -> np.ndarray:
    """Return physical Fourier wave numbers for an ``length``-periodic grid."""
    if n < 4:
        raise ValueError("n must be at least 4")
    if length <= 0:
        raise ValueError("length must be positive")
    dx = length / n
    return 2 * np.pi * np.fft.fftfreq(n, d=dx)


def dealias_mask_2d(n: int) -> np.ndarray:
    """Cartesian two-thirds dealiasing mask expressed in integer mode indices."""
    modes = np.fft.fftfreq(n) * n
    kx, ky = np.meshgrid(modes, modes, indexing="ij")
    cutoff = n / 3
    return (np.abs(kx) <= cutoff) & (np.abs(ky) <= cutoff)


def project_div_free(u_hat: np.ndarray, kx: np.ndarray, ky: np.ndarray, kz: np.ndarray) -> np.ndarray:
    """Apply the Leray projection without mutating the input array."""
    k2 = kx * kx + ky * ky + kz * kz
    safe_k2 = np.where(k2 == 0, 1.0, k2)
    dot = kx * u_hat[0] + ky * u_hat[1] + kz * u_hat[2]
    out = np.empty_like(u_hat)
    out[0] = u_hat[0] - kx * dot / safe_k2
    out[1] = u_hat[1] - ky * dot / safe_k2
    out[2] = u_hat[2] - kz * dot / safe_k2
    out[:, k2 == 0] = u_hat[:, k2 == 0]
    return out
