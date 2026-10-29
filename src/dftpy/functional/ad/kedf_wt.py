from __future__ import annotations
from math import isfinite, pi
from typing import Optional, Sequence
from torch import Tensor

from .core import C_TF, TKF0, EnergyFunctionalABC, GridContext, ParameterSpec
from .kernels import lindhard_residual

class WT(EnergyFunctionalABC):
    """Full TF + vW + WT-family nonlocal kinetic energy.

    Set nelec explicitly for a fixed-N SCF/parameter-training calculation.
    """
    name = "WT-AD"

    @classmethod
    def parameter_specs(cls):
        return {
            "alpha": ParameterSpec(5/6, "Left density exponent", lower=1e-8),
            "beta": ParameterSpec(5/6, "Right density exponent", lower=1e-8),
        }

    def __init__(self, alpha=5/6, beta=5/6, *, nelec: Optional[float] = None,
                 **options):
        if nelec is not None and (not isfinite(nelec) or nelec <= 0):
            raise ValueError("nelec must be positive and finite")
        super().__init__(alpha=alpha, beta=beta, **options)
        self.nelec = None if nelec is None else float(nelec)
        self._symmetric_stamp = None
        self._equal_fixed_exponents = False

    def _kernel(self, rho, ctx):
        a, b = self.coefficients["alpha"], self.coefficients["beta"]
        if self.nelec is None:
            ne = (rho.mean() * ctx.volume).detach()
            nkey = float(ne.cpu())
        else:
            ne = self.nelec
            nkey = self.nelec
        n0 = ne / ctx.volume
        key = ("WT", ctx.key, nkey, self.parameter_versions())
        def build():
            eta = ctx.q / (2 * (3*pi*pi*n0).pow(1/3))
            return 5 / (9*a*b*n0.pow(a+b-5/3)) * lindhard_residual(eta)
        # No detached-cache path for cell or kernel-parameter differentiation.
        cacheable = ctx.key is not None and not (a.requires_grad or b.requires_grad)
        return self.kernel_cache.get_or_build(key, build, allowed=cacheable)

    def energy(self, rho: Tensor, context: GridContext) -> Tensor:
        a, b = self.coefficients["alpha"], self.coefficients["beta"]
        ctx = context
        kernel = self._kernel(rho, ctx)
        stamp = self.parameter_versions()
        if stamp != self._symmetric_stamp:
            self._equal_fixed_exponents = float(a.detach().cpu()) == float(b.detach().cpu())
            self._symmetric_stamp = stamp
        fa = ctx.fft(rho.pow(a))
        fb = fa if self._equal_fixed_exponents else ctx.fft(rho.pow(b))
        tnl = C_TF * ctx.parseval(fa, fb, kernel)
        return ctx.tf(rho) + ctx.vw(rho) + tnl
