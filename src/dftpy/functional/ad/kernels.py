from __future__ import annotations
from functools import lru_cache
import numpy as np
import torch
from torch import Tensor


def lindhard_residual(eta: Tensor) -> Tensor:
    """D(eta)=1/L(eta)-1-3*eta**2, stable at 0 and large eta.

    D(1)=-2. Its eta derivative has the genuine logarithmic Kohn anomaly;
    exact eta=1 is rejected when eta itself needs derivatives (e.g. stress).
    """
    small = eta < 1e-3
    large = eta > 10.0
    one = eta == 1.0
    if eta.requires_grad and bool(one.any()):
        raise ValueError("Exact eta=1 Kohn anomaly: cell derivative is singular on this mesh")
    # Both unused branches remain finite to prevent 0*NaN in backward.
    x = torch.where(small | large | one, torch.full_like(eta, 0.5), eta)
    L = 0.5 + (1 - x.square()) / (4*x) * torch.log(torch.abs((1+x)/(1-x)))
    regular = 1/L - 1 - 3*x.square()
    z = eta.square()
    near0 = z * (-8/3 + z * (8/45 + z * (104/945 + z * (1048/14175))))
    inv2 = torch.where(large, eta, torch.full_like(eta, 10.0)).reciprocal().square()
    far = -8/5 + inv2 * (-24/175 + inv2 * (-8/125 + inv2 * (-12728/336875 + inv2 * (
        -551416/21896875 + inv2 * (-41587384/2299171875 + inv2 * (-2671830232/195429609375))))))
    out = torch.where(small, near0, torch.where(large, far, regular))
    return torch.where(one, torch.full_like(out, -2.0), out)


def _residual_scalar(x: float) -> float:
    if x < 1e-3:
        z=x*x
        return z*(-8/3+z*(8/45+z*(104/945+z*1048/14175)))
    if x > 10:
        z=1/(x*x)
        return -8/5+z*(-24/175+z*(-8/125+z*(-12728/336875+z*(-551416/21896875+z*(
            -41587384/2299171875+z*(-2671830232/195429609375))))))
    if x == 1:
        return -2.0
    L=.5+(1-x*x)/(4*x)*np.log(abs((1+x)/(1-x)))
    return 1/L-1-3*x*x


@lru_cache(maxsize=8)
def build_hc_table(beta: float, eta_max: float = 50.0, n_eta: int = 10000):
    """Solve beta*eta*w'=(5-3*beta)*beta*w-(5/3)*D, w(0)=0.

    Starts from w(eta_max)=-(8/3)/[(5-3*beta)*beta]. Returns readonly numpy arrays.
    Converge eta_max and n_eta for production work. beta is NOT differentiable.
    """
    from scipy.integrate import solve_ivp
    if not (0 < beta < 5/3) or eta_max <= 1 or n_eta < 32:
        raise ValueError("Require 0<beta<5/3, eta_max>1, n_eta>=32")
    eta = np.linspace(0, eta_max, n_eta)
    w_inf = -(8/3)/((5-3*beta)*beta)
    def ode(t, y):
        return ((5-3*beta)*beta*y-(5/3)*_residual_scalar(t))/(beta*t)
    sol = solve_ivp(ode, (eta[-1], eta[1]), [w_inf], t_eval=eta[1:][::-1],
                    rtol=2e-11, atol=2e-12, method="DOP853")
    if not sol.success:
        raise RuntimeError(f"HC kernel ODE failed: {sol.message}")
    w = np.concatenate(([0.0], sol.y[0, ::-1]))
    eta.setflags(write=False)
    w.setflags(write=False)
    return eta, w


def hermite_slopes(x: Tensor, y: Tensor) -> Tensor:
    delta = (y[1:]-y[:-1])/(x[1:]-x[:-1])
    return torch.cat((delta[:1], 0.5*(delta[1:]+delta[:-1]), delta[-1:]))


def hermite_uniform(x: Tensor, y: Tensor, slopes: Tensor, query: Tensor) -> Tensor:
    """Cubic Hermite table lookup; uniform x, clamped above eta_max"""
    dx = x[1]-x[0]
    z = query.clamp(min=0.0, max=float(x[-1].detach().cpu()))
    idx = torch.floor((z-x[0])/dx).long().clamp(0, x.numel()-2)
    t = (z-x[idx])/dx
    t2=t*t; t3=t2*t
    return ((2*t3-3*t2+1)*y[idx] + (t3-2*t2+t)*dx*slopes[idx]
            + (-2*t3+3*t2)*y[idx+1] + (t3-t2)*dx*slopes[idx+1])


def local_hermite_weights(x: Tensor, query: Tensor):
    """
    Caller supplies two padding nodes on either side. Index selection is
    piecewise constant; the interpolation weights RETAIN their query graph.
    """
    i = torch.searchsorted(x, query.detach().contiguous(), right=True)-1
    if bool((i < 1).any()) or bool((i > x.numel()-3).any()):
        raise ValueError("xi lies outside the padded interpolation domain")
    dx = x[i+1]-x[i]
    left = x[i]-x[i-1]
    right = x[i+2]-x[i+1]
    t=(query-x[i])/dx
    t2=t*t; t3=t2*t
    h00=2*t3-3*t2+1; h10=t3-2*t2+t
    h01=-2*t3+3*t2; h11=t3-t2
    weights = torch.stack((
        -h10*dx/(2*left),
        h00+h10*dx/(2*left)-0.5*h10-0.5*h11,
        h01+0.5*h10+0.5*h11-h11*dx/(2*right),
        h11*dx/(2*right),
    ))
    return torch.stack((i-1, i, i+1, i+2)), weights
