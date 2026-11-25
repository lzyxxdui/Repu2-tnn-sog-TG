#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for global/composite Gauss--Legendre quadrature."""

import unittest

import torch

from main import build_axis_rules
from quadrature import composite_gauss_legendre, gauss_legendre_interval


class QuadratureTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float64

    def test_m1_is_bitwise_compatible_with_global(self):
        xg, wg = gauss_legendre_interval(
            32, 0.0, 1.0, device=self.device, dtype=self.dtype
        )
        xc, wc = composite_gauss_legendre(
            0.0, 1.0, 1, 32, device=self.device, dtype=self.dtype
        )
        self.assertTrue(torch.equal(xg, xc))
        self.assertTrue(torch.equal(wg, wc))

    def test_composite_count_and_weight_sum(self):
        x, w = composite_gauss_legendre(
            -0.25, 1.75, 8, 4, device=self.device, dtype=self.dtype
        )
        self.assertEqual(x.numel(), 32)
        self.assertEqual(w.numel(), 32)
        self.assertAlmostEqual(float(w.sum()), 2.0, places=14)

    def test_composite_is_not_global_total_order(self):
        xg, _ = gauss_legendre_interval(
            32, 0.0, 1.0, device=self.device, dtype=self.dtype
        )
        xc, _ = composite_gauss_legendre(
            0.0, 1.0, 8, 4, device=self.device, dtype=self.dtype
        )
        self.assertFalse(torch.equal(xg, xc))

    def test_local_polynomial_exactness(self):
        x, w = composite_gauss_legendre(
            0.0, 1.0, 8, 4, device=self.device, dtype=self.dtype
        )
        for power in range(8):  # k=4 -> exact through degree 2k-1=7
            numerical = torch.sum(w * x.pow(power)).item()
            exact = 1.0 / (power + 1)
            self.assertAlmostEqual(numerical, exact, places=14)

    def test_anisotropic_axis_settings(self):
        bounds = ((0.0, 1.0), (0.0, 2.0), (-1.0, 1.0))
        nodes, weights = build_axis_rules(
            (3, 4, 5),
            bounds,
            mode="composite",
            subintervals=(2, 3, 4),
            device=self.device,
            dtype=self.dtype,
        )
        self.assertEqual(tuple(x.numel() for x in nodes), (6, 12, 20))
        self.assertEqual(tuple(w.numel() for w in weights), (6, 12, 20))
        for w, (a, b) in zip(weights, bounds):
            self.assertAlmostEqual(float(w.sum()), b - a, places=14)


if __name__ == "__main__":
    unittest.main()
