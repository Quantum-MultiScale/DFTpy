from __future__ import annotations
from typing import Sequence
import torch
from torch import Tensor
from .core import ParameterSpec, SemilocalFunctionalABC


class GE2(SemilocalFunctionalABC):
    """Second-order gradient expansion: F = a + b*s2 + c*q.

    s2 is the squared reduced gradient and q = laplacian(rho)/(TKF0**2*rho**(5/3)).
    """
    name = "GE2-AD"
    level = "MGGA"

    @classmethod
    def parameter_specs(cls):
        return {
            "a": ParameterSpec(1.0, "Thomas-Fermi weight"),
            "b": ParameterSpec(5.0 / 27.0, "Squared reduced-gradient coefficient"),
            "c": ParameterSpec(20.0 / 9.0, "Reduced-Laplacian coefficient"),
        }

    def __init__(self, a=1.0, b=5.0 / 27.0, c=20.0 / 9.0, **kwargs):
        super().__init__(a=a, b=b, c=c, **kwargs)

    def enhancement(self, s2, coefficients, q=None, **kwargs):
        return coefficients["a"] + coefficients["b"] * s2 + coefficients["c"] * q


class PGSL(SemilocalFunctionalABC):
    """Pauli-Gaussian plus Laplacian: F = exp(-mu*s2) + beta*q**2 + (5/3)*y*s2.

    Uses the same reduced variables as GE2. Parameter order (beta, mu, y)
    """
    name = "PGSL-AD"
    level = "MGGA"

    @classmethod
    def parameter_specs(cls):
        return {
            "beta": ParameterSpec(0.25, "Squared reduced-Laplacian coefficient", lower=0.0),
            "mu": ParameterSpec(40.0 / 27.0, "Pauli-Gaussian exponent", lower=0.0),
            "y": ParameterSpec(1.0, "von Weizsaecker weight", lower=0.0),
        }

    def __init__(self, beta=0.25, mu=40.0 / 27.0, y=1.0, **kwargs):
        super().__init__(beta=beta, mu=mu, y=y, **kwargs)

    def enhancement(self, s2, coefficients, q=None, **kwargs):
        return (torch.exp(-coefficients["mu"] * s2)
                + coefficients["beta"] * q.square()
                + (5.0 / 3.0) * coefficients["y"] * s2)


class PGSLr(PGSL):
    """PGSL with the correction -lambd*q*s2 + sigma*s2**2.

    Parameter order (lambd, sigma, beta, mu, y).
    lambd avoids Python's reserved word 'lambda'.
    """
    name = "PGSLr-AD"

    @classmethod
    def parameter_specs(cls):
        return {
            "lambd": ParameterSpec(0.40, "Mixed reduced-Laplacian/gradient coefficient"),
            "sigma": ParameterSpec(0.20, "Fourth-order reduced-gradient coefficient"),
            **super().parameter_specs(),
        }

    def __init__(self, lambd=0.40, sigma=0.20, beta=0.25, mu=40.0 / 27.0,
                 y=1.0, **kwargs):
        super().__init__(lambd=lambd, sigma=sigma, beta=beta, mu=mu, y=y, **kwargs)

    def enhancement(self, s2, coefficients, q=None, **kwargs):
        return (super().enhancement(s2, coefficients, q=q)
                - coefficients["lambd"] * q * s2
                + coefficients["sigma"] * s2.square())


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
