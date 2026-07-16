from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .spectral import dealias_mask_2d, periodic_wavenumbers


@dataclass(frozen=True)
class BurgersConfig:
    n: int = 64
    dt: float = 2e-4
    final_time: float = 0.02
    viscosity: float = 0.01
    feedback: float = 0.0
    length: float = 2 * np.pi


def solve(config: BurgersConfig):
    """Solve the two-component viscous Burgers system on a periodic square.

    The RHEA feedback operator is treated once, implicitly, as
    ``-feedback * (I - Laplacian) u``.  Physical viscosity is independent.
    """
    if config.dt <= 0 or config.final_time < 0:
        raise ValueError("dt must be positive and final_time non-negative")
    if config.viscosity < 0 or config.feedback < 0:
        raise ValueError("viscosity and feedback must be non-negative")

    n = config.n
    x = np.linspace(0, config.length, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = np.empty((2, n, n), dtype=float)
    u[0] = np.sin(X) * np.cos(Y)
    u[1] = -np.cos(X) * np.sin(Y)

    k = periodic_wavenumbers(n, config.length)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    k2 = kx * kx + ky * ky
    mask = dealias_mask_2d(n)

    steps = round(config.final_time / config.dt)
    energies: list[float] = []

    for _ in range(steps):
        u_hat = np.fft.fft2(u, axes=(-2, -1))
        u_hat *= mask
        ux = np.fft.ifft2(1j * kx * u_hat, axes=(-2, -1)).real
        uy = np.fft.ifft2(1j * ky * u_hat, axes=(-2, -1)).real
        nonlinear = np.stack(
            [
                u[0] * ux[0] + u[1] * uy[0],
                u[0] * ux[1] + u[1] * uy[1],
            ]
        )
        nonlinear_hat = np.fft.fft2(nonlinear, axes=(-2, -1)) * mask
        denominator = 1 + config.dt * (
            config.viscosity * k2 + config.feedback * (1 + k2)
        )
        u_hat = (u_hat - config.dt * nonlinear_hat) / denominator
        u = np.fft.ifft2(u_hat, axes=(-2, -1)).real
        energies.append(float(0.5 * np.mean(np.sum(u * u, axis=0))))

    metadata = asdict(config) | {
        "steps": steps,
        "equation": "2D vector viscous Burgers with optional implicit RHEA feedback",
    }
    return u, np.asarray(energies), metadata
