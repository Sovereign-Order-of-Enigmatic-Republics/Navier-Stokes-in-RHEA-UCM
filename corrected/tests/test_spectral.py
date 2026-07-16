import unittest

import numpy as np

from rhea_ns.burgers2d import BurgersConfig, solve
from rhea_ns.spectral import dealias_mask_2d, periodic_wavenumbers, project_div_free


class SpectralTests(unittest.TestCase):
    def test_periodic_modes_for_two_pi_domain(self) -> None:
        self.assertTrue(
            np.allclose(periodic_wavenumbers(8), [0, 1, 2, 3, -4, -3, -2, -1])
        )

    def test_two_thirds_mask_uses_integer_mode_indices(self) -> None:
        self.assertEqual(int(dealias_mask_2d(64).sum()), 43 * 43)

    def test_leray_projection_is_divergence_free(self) -> None:
        n = 8
        k = periodic_wavenumbers(n)
        kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
        rng = np.random.default_rng(1)
        u_hat = rng.normal(size=(3, n, n, n)) + 1j * rng.normal(size=(3, n, n, n))
        projected = project_div_free(u_hat, kx, ky, kz)
        divergence = kx * projected[0] + ky * projected[1] + kz * projected[2]
        divergence[0, 0, 0] = 0
        self.assertLess(float(np.max(np.abs(divergence))), 1e-10)

    def test_reference_solver_is_finite_and_metadata_matches_execution(self) -> None:
        config = BurgersConfig(
            n=16,
            dt=1e-3,
            final_time=5e-3,
            viscosity=0.01,
            feedback=0.02,
        )
        velocity, energy, metadata = solve(config)
        self.assertTrue(np.isfinite(velocity).all())
        self.assertTrue(np.isfinite(energy).all())
        self.assertEqual(metadata["dt"], config.dt)
        self.assertEqual(metadata["steps"], 5)


if __name__ == "__main__":
    unittest.main()
