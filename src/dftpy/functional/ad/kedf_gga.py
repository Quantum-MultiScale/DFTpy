from __future__ import annotations
from typing import Sequence
import torch
from torch import Tensor
from .core import ParameterSpec, SemilocalFunctionalABC

class LKT(SemilocalFunctionalABC):
    name = "LKT-AD"
    level = "GGA"

    @classmethod
    def parameter_specs(cls):
        return {
            "a": ParameterSpec(1.3, "LKT Pauli enhancement coefficient", lower=0.0),
            "y": ParameterSpec(1.0, "von Weizsaecker weight", lower=0.0),
        }

    def __init__(self, a=1.3, y=1.0, **kwargs):
        super().__init__(a=a, y=y)

    def enhancement(self, s2, coefficients, q=None, **kwargs):
        a = coefficients['a']
        y = coefficients['y']
        f = 1.0/torch.cosh(a * torch.sqrt(s2.clamp(max=10000))) + (5.0 / 3.0) * y * s2
        return f


class LKT_sech(LKT):
    """LKT enhancement F=sech(a*s)+(5/3)*y*s**2"""
    name = "LKTsech-AD"
    level = "GGA"

    def enhancement(self, s2, coefficients, q=None, **kwargs):
        return self.sech_sqrt(coefficients["a"].square() * s2) + (5/3) * coefficients["y"] * s2

    @staticmethod
    def sech_sqrt(z: Tensor) -> Tensor:
        """sech(sqrt(z)), finite first/second derivatives at z=0, no cosh overflow.
        The z<1e-4 branch is a fourth-order Taylor polynomial, with negligible
        truncation error near the switch. The unused branch is itself finite.
        """
        small = z < 1e-5
        poly = 1 + z * (-0.5 + z * (5/24 + z * (-61/720 + z * (1385/40320))))
        r = torch.sqrt(torch.where(small, torch.ones_like(z), z))
        t = torch.exp(-r)
        regular = 2 * t / (1 + t * t)
        return torch.where(small, poly, regular)
