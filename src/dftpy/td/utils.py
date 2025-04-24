from typing import Optional, Tuple

import numpy as np

from dftpy.constants import SPEED_OF_LIGHT
from dftpy.field import DirectField
from dftpy.td.operator import Operator


def initial_kick(
    k: float, direction: int, psi: DirectField, mask: Optional[DirectField] = None
) -> DirectField:
    x = psi.grid.r[direction]
    if mask is not None:
        x *= mask
    psi1 = psi * np.exp(1j * k * x)
    psi1.cplx = True
    return psi1


def initial_kick_vector_potential(
    k: float, direction: int, psi: DirectField
) -> Tuple[DirectField, np.ndarray]:
    psi1 = np.complex128(psi)
    psi1.cplx = True
    A = np.zeros[3]
    A[direction] = k * SPEED_OF_LIGHT
    return psi1, A


def vector_potential_energy(
    interval: float, volume: float, A: np.ndarray, A_prev: np.ndarray
) -> float:
    energy = (
        volume
        / 8.0
        / np.pi
        / SPEED_OF_LIGHT**2
        * (np.dot((A - A_prev), (A - A_prev)) / interval / interval)
    )

    return energy


class PotentialOperator(Operator):

    def __init__(self, v: DirectField):
        self.v = v
        super().__init__(v.grid)

    def __call__(self, psi):
        return self.v * psi

    def energy(self, psi):
        return (self.v * np.real(np.conj(psi) * psi)).integral()


def calc_spectra_mu(
    mu, dt, kick=1, sigma=3e-5, sigma1=None, use_fft=True, de=1e-3, emax=0.5
):
    mu = mu - mu[0]
    nmax = int(2 * np.pi / de / dt)
    if len(mu) < nmax and use_fft:
        mu = np.pad(mu, (0, nmax - len(mu)), 'constant')
    t = np.arange(len(mu)) * dt
    if sigma1 is not None:
        mu = mu * np.exp(-sigma * t)
    elif sigma is not None:
        mu = mu * np.exp(-sigma * t**2)
    if use_fft:
        omega = np.fft.rfftfreq(len(mu), d=dt) * 2 * np.pi
        spectra = np.fft.rfft(mu)
    else:
        omega = np.arange(0, emax, de)
        spectra = []
        for i, w in enumerate(omega):
            spectra.append(np.sum(mu * np.exp(-1.0j * w * t)))
        spectra = np.asarray(spectra)
    spectra = spectra.imag * omega * (-1.0 * dt / kick * 2.0 / np.pi)
    return omega, spectra
