"""
Moving projectile for electronic stopping power.

A charge Z travelling at constant velocity through the electron gas. The force
it feels back from the electrons, projected on its direction of travel, is the
stopping power:

.. math::
    \\mathbf{F}(t) = \\int \\rho(\\mathbf{r})
                     \\nabla V_{proj}(\\mathbf{r} - \\mathbf{R}(t)) d\\mathbf{r}
    S(t) = -\\mathbf{F}(t) \\cdot \\hat{v}

Design notes
------------
The projectile is *not* an ion in the Ions object: its velocity is held fixed
(that is what "stopping power at velocity v" means), so its trajectory is
advanced analytically as R(t) = R0 + v t rather than integrated.

Its potential cannot be assigned to ``hamiltonian.v`` directly, because
:class:`dftpy.td.predictor_corrector.PredictorCorrector` re-assigns
``propagator.hamiltonian.v`` from the functional on every corrector iteration.
It is instead injected through :class:`ExternalPotential`, a thin wrapper around
a TotalFunctional, so that every internal recompute includes it.

The charge is smeared into a normalised Gaussian of width ``sigma`` and its
potential is obtained with the same Coulomb route DFTpy uses for Hartree
(``fft * invgg * 4 pi``), which keeps the FFT normalisation consistent by
construction and yields the erf-screened Coulomb

.. math::
    V(r) = -Z\\, \\mathrm{erf}(r / (\\sqrt{2}\\sigma)) / r

``sigma`` is the projectile pseudopotential: it must be resolvable on the grid
(``sigma >~ grid spacing``) and the stopping power depends on it the way a
plane-wave Kohn-Sham result depends on the projectile PP. Converge it.
"""

import numpy as np

from dftpy.field import DirectField

__all__ = ['Projectile', 'ExternalPotential']

AMU_TO_AU = 1822.888486209


class Projectile:
    """
    A Gaussian-smeared point charge moving at constant velocity.

    Parameters
    ----------
    Z: float
        Charge in a.u. (+2 for a fully stripped alpha particle).
    R0: array_like, shape (3,)
        Initial position in bohr, Cartesian.
    velocity: array_like, shape (3,)
        Velocity in a.u.; its direction defines the trajectory.
    sigma: float
        Gaussian width in bohr. Acts as the projectile pseudopotential and must
        be resolved by the grid; converge results against it.
    mass_amu: float, optional
        Only used to report a momentum column in the output.

    Examples
    --------
    >>> alpha = Projectile(Z=2.0, R0=[4.0, 4.0, 0.0], velocity=[0, 0, 5.0],
    ...                    sigma=0.3, mass_amu=4.0026)
    >>> S = alpha.stopping_power(rho, t=0.5)
    """

    def __init__(self, Z, R0, velocity, sigma=0.4, mass_amu=None):
        self.Z = float(Z)
        self.R0 = np.asarray(R0, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)
        self.sigma = float(sigma)
        self.mass_amu = mass_amu
        self.speed = float(np.linalg.norm(self.velocity))
        self.vhat = self.velocity / self.speed if self.speed > 0 else np.zeros(3)

    @classmethod
    def from_config(cls, config, cell_lengths=None):
        """
        Build from the ``[PROJECTILE]`` section of a DFTpy config.

        ``position`` is given in FRACTIONAL coordinates, matching the way ion
        positions are usually specified, so ``cell_lengths`` (bohr) is needed to
        convert. Pass ``np.diag(rho.grid.lattice)``.
        """
        c = config["PROJECTILE"]
        R = np.asarray(c["position"], dtype=float)
        if cell_lengths is not None:
            R = R * np.asarray(cell_lengths, dtype=float)
        return cls(Z=c["charge"], R0=R, velocity=c["velocity"],
                   sigma=c["sigma"], mass_amu=c["mass"])

    # ------------------------------------------------------------- geometry
    def position(self, t):
        """Analytic trajectory R(t) = R0 + v t, in bohr (not wrapped)."""
        return self.R0 + self.velocity * t

    def _min_image(self, grid, R):
        """r - R under the minimum-image convention. Orthorhombic cells only."""
        lat = np.asarray(grid.lattice)
        off = np.abs(lat - np.diag(np.diag(lat))).max()
        if off > 1e-8 * max(np.abs(lat).max(), 1.0):
            raise NotImplementedError(
                "Projectile assumes an orthorhombic cell; the lattice has "
                "off-diagonal elements. Generalise _min_image before using a "
                "non-orthogonal box.")
        L = np.diag(lat)
        r = np.asarray(grid.r)
        d = np.empty_like(r)
        for i in range(3):
            di = r[i] - R[i]
            d[i] = di - L[i] * np.round(di / L[i])
        return d

    # --------------------------------------------------------------- fields
    def charge_density(self, grid, t):
        """Normalised Gaussian of total charge Z centred at R(t)."""
        d = self._min_image(grid, self.position(t))
        d2 = d[0] ** 2 + d[1] ** 2 + d[2] ** 2
        rho_p = DirectField(grid, rank=1,
                            griddata_3d=np.exp(-d2 / (2.0 * self.sigma ** 2)))
        # normalise on the grid, not analytically: exact even when the gaussian
        # is only marginally resolved, and MPI-safe (integral does the reduce)
        return rho_p * (self.Z / rho_p.integral())

    def potential(self, grid, t):
        """
        Electron-facing potential V(r) in a.u. Negative for Z > 0.

        Uses the same route as the Hartree term so the normalisation matches
        DFTpy's conventions exactly.
        """
        rho_p = self.charge_density(grid, t)
        invgg = grid.get_reciprocal().invgg
        return -(rho_p.fft() * invgg * 4.0 * np.pi).ifft(force_real=True)

    def analytic_potential(self, grid, t):
        """Isolated-ion ``-Z erf(r/(sqrt2 sigma))/r``. For verification only."""
        from scipy.special import erf
        d = self._min_image(grid, self.position(t))
        rr = np.maximum(np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2), 1e-12)
        return DirectField(grid, rank=1,
                           griddata_3d=-self.Z * erf(rr / (np.sqrt(2.0) * self.sigma)) / rr)

    # ---------------------------------------------------------- observables
    def force(self, rho, t):
        """Force on the projectile from the electrons, ``int rho grad V``."""
        V = self.potential(rho.grid, t)
        gradV = V.gradient(flag='standard', force_real=True)
        return np.asarray((rho * gradV).integral(), dtype=float)

    def stopping_power(self, rho, t):
        """``S = -F . vhat`` in Ha/bohr. Positive means energy is being lost."""
        if self.speed == 0.0:
            return 0.0
        return float(-np.dot(self.force(rho, t), self.vhat))

    def interaction_energy(self, rho, t):
        """``int rho V_proj dr`` in Ha."""
        return float((rho * self.potential(rho.grid, t)).integral())

    def momentum(self):
        """Momentum in a.u., or zeros if no mass was given."""
        if self.mass_amu is None:
            return np.zeros(3)
        return self.velocity * self.mass_amu * AMU_TO_AU

    def info(self, grid=None):
        """One-line summary, including whether sigma is resolved by the grid."""
        msg = (f"Projectile: Z = {self.Z}, |v| = {self.speed:.4f} a.u., "
               f"sigma = {self.sigma:.4f} bohr")
        if grid is not None:
            h = float(np.max(np.diag(np.asarray(grid.lattice)) / np.asarray(grid.nr)))
            msg += f", grid spacing = {h:.4f} bohr (sigma/h = {self.sigma / h:.2f})"
            if self.sigma < 1.2 * h:
                msg += "  [WARNING: sigma barely resolved, refine the grid]"
        return msg


class ExternalPotential:
    """
    Wrap a TotalFunctional and add an external potential to its output.

    Required because :class:`PredictorCorrector` overwrites ``hamiltonian.v``
    from the functional on every corrector iteration, so a potential injected at
    the Hamiltonian level would not survive. Attributes not defined here are
    delegated to the wrapped functional, so this is a drop-in replacement.

    Parameters
    ----------
    functionals: AbstractFunctional
        The functional to wrap.
    v_static: DirectField, optional
        A time-independent extra potential, e.g. frozen ions supplied outside
        the PSEUDO functional.

    Notes
    -----
    Set :attr:`v_ext` to a DirectField (or None) before each evaluation; this is
    what :class:`dftpy.td.stopping_runner.StoppingRunner` updates each step.
    """

    def __init__(self, functionals, v_static=None):
        self._tf = functionals
        self.v_ext = None
        self.v_static = v_static

    def _extra(self):
        if self.v_static is None:
            return self.v_ext
        if self.v_ext is None:
            return self.v_static
        return self.v_static + self.v_ext

    def __call__(self, rho, calcType=('V',), current=None, **kwargs):
        out = self._tf(rho, calcType=calcType, current=current, **kwargs)
        extra = self._extra()
        if extra is not None:
            if 'V' in calcType:
                out.potential = out.potential + extra
            if 'E' in calcType:
                out.energy = out.energy + float((rho * extra).integral())
        return out

    def __getattr__(self, name):
        # only reached when the attribute is not found on self
        return getattr(self._tf, name)
