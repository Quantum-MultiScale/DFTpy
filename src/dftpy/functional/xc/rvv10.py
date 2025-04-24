import numpy as np
from scipy.interpolate import CubicSpline

from dftpy.field import DirectField, ReciprocalField
from dftpy.functional.abstract_functional import AbstractFunctional, FunctionalOutput
from dftpy.utils.utils import clean_variables

# from dftpy.mpi import sprint


class RVV10NL(AbstractFunctional):
    """Nonlocal part of rVV10 functional

    note :
        This implementation is modeled after Quantum-ESPRESSO rVV10.

    refs :
        - Phys. Rev. B 87, 041108(R) (2013) DOI:10.1103/physrevb.87.041108
    """

    def __init__(
        self,
        b_value=6.3,
        epsr=1.0e-12,
        epsg=1.0e-10,
        nr_points=1024,
        r_max=100.0,
        q_min=1.0e-4,
        q_cut=0.50,
        nqs=20,
        c_value=0.0093,
        **kwargs,
    ):
        self.b_value_0 = b_value
        self.epsr = epsr
        self.epsg = epsg
        self.nr_points = nr_points
        self.r_max = r_max
        self.q_min = q_min
        self.q_cut = q_cut
        self.nqs = nqs
        self.c_value = c_value
        #
        self.initialize()

    def initialize(self):
        if self.nqs == 20:
            self.q_mesh = np.array(
                [
                    self.q_min,
                    3.0e-4,
                    5.893850845618885e-4,
                    1.008103720396345e-3,
                    1.613958359589310e-3,
                    2.490584839564653e-3,
                    3.758997979748929e-3,
                    5.594297198907115e-3,
                    8.249838297569416e-3,
                    1.209220822453922e-2,
                    1.765183095571029e-2,
                    2.569619042667097e-2,
                    3.733577865542191e-2,
                    5.417739477463518e-2,
                    7.854595729872216e-2,
                    0.1138054499321450,
                    0.1648233062188070,
                    0.2386423394972170,
                    0.3454529754349640,
                    self.q_cut,
                ]
            )
        else:
            self.q_mesh = np.geomspace(self.q_min, self.q_cut, self.nqs)
        self.dr = self.r_max / self.nr_points
        self.dk = 2.0 * np.pi / self.r_max
        #
        self.kernel = {}
        self.splines_q = {}
        self.splines_kernel = {}
        #
        self.generate_kernel()

    def compute(self, density, **kwargs):
        functional = self.xc_rvv10(density, **kwargs)
        return functional

    def xc_rvv10(
        self, density, calcType={"E", "V"}, b_value=None, clean=True, **kwargs
    ):
        #
        self.q0 = DirectField(grid=density.grid)
        self.dq0_drho = DirectField(grid=density.grid)
        self.dq0_dgradrho = DirectField(grid=density.grid)
        self.thetas = DirectField(grid=density.grid, rank=self.nqs)
        self.thetas_g = ReciprocalField(grid=density.grid, rank=self.nqs)
        #
        self.b_value = b_value or self.b_value_0
        self.beta = 0.0625 * (3.0 / (self.b_value**2.0)) ** (0.75)
        gradient_rho = density.gradient(flag="standard")
        self.get_q0_on_grid(density, gradient_rho)
        self.get_thetas_on_grid(density)
        #
        energy = self.vdw_energy()
        energy = energy + self.beta * density.sum() * density.grid.dV
        energy *= 0.5  # I don't understand why?
        self.energy = energy
        if 'V' in calcType:
            for i in range(0, self.nqs):
                self.thetas[i] = self.thetas_g[i].ifft().real
            potential = self.get_potential(density, gradient_rho)
            potential = potential + self.beta
            potential *= 0.5
        else:
            potential = None
        functional = FunctionalOutput(name="XC", potential=potential, energy=energy)

        if clean:
            clean_variables(
                self.q0, self.dq0_drho, self.dq0_dgradrho, self.thetas, self.thetas_g
            )
        return functional

    def generate_kernel(self):
        r = np.arange(0, self.nr_points + 1) * self.dr
        k = np.arange(0, self.nr_points + 1) * self.dk
        r2 = r * r
        phi = np.zeros(self.nr_points + 1)
        for q1_i in range(0, self.nqs):
            for q2_i in range(0, q1_i + 1):
                d1 = self.q_mesh[q1_i] * r2[1:]
                d2 = self.q_mesh[q2_i] * r2[1:]
                phi[1:] = -24.0 / ((d1 + 1.0) * (d2 + 1.0) * (d1 + d2 + 2.0))
                self.kernel[(q1_i, q2_i)] = self.radial_fft(phi, r, k)
                self.splines_kernel[(q1_i, q2_i)] = CubicSpline(
                    k, self.kernel[(q1_i, q2_i)], bc_type='natural'
                )
                #
                self.kernel[(q2_i, q1_i)] = self.kernel[(q1_i, q2_i)]
                self.splines_kernel[(q2_i, q1_i)] = self.splines_kernel[(q1_i, q2_i)]
        #
        y = np.zeros_like(self.q_mesh)
        for i in range(0, len(self.q_mesh)):
            y[:] = 0.0
            y[i] = 1.0
            self.splines_q[i] = CubicSpline(self.q_mesh, y, bc_type='natural')
        return

    def get_q0_on_grid(self, density, gradient_rho):
        self.q0[:] = self.q_cut
        self.dq0_drho[:] = 0.0
        self.dq0_dgradrho[:] = 0.0
        mask = density > self.epsr
        gmod2 = (gradient_rho[:, mask] ** 2).sum(axis=0)
        mod_grad = np.sqrt(gmod2)
        wp2 = 16.0 * np.pi * density[mask]
        wg2 = 4.0 * self.c_value * (mod_grad / density[mask]) ** 4
        k = (
            self.b_value
            * 3.0
            * np.pi
            * ((density[mask] / (9.0 * np.pi)) ** (1.0 / 6.0))
        )
        w0 = np.sqrt(wg2 + wp2 / 3.0)
        q = w0 / k

        exponent = 0.0
        dq0_dq = 0.0
        for index in range(1, 13):
            exponent = exponent + ((q / self.q_cut) ** index) / index
            dq0_dq = dq0_dq + ((q / self.q_cut) ** (index - 1))
        self.q0[mask] = self.q_cut * (1.0 - np.exp(-exponent))
        dq0_dq = dq0_dq * np.exp(-exponent)
        #
        self.q0[self.q0 < self.q_min] = self.q_min
        #
        dw0_dn = 1.0 / (2.0 * w0) * (16.0 / 3.0 * np.pi - 4.0 * wg2 / density[mask])
        dk_dn = k / (6.0 * density[mask])
        self.dq0_drho[mask] = dq0_dq / (k**2) * (dw0_dn * k - dk_dn * w0)
        mask2 = gmod2 > self.epsr
        #
        mask[mask] = mask2
        self.dq0_dgradrho[mask] = (
            dq0_dq[mask2]
            / (2.0 * k[mask2] * w0[mask2])
            * 4.0
            * wg2[mask2]
            / (mod_grad[mask2] ** 2)
        )

    def get_thetas_on_grid(self, density):
        for i in range(self.nqs):
            self.thetas[i] = self.splines_q[i](self.q0)
        #
        mask = density > self.epsr
        self.thetas[:, mask] *= (
            1.0 / (3.0 * np.sqrt(np.pi) * (self.b_value ** (3.0 / 2.0)))
        ) * (density[mask] / np.pi) ** (3.0 / 4.0)
        self.thetas[:, ~mask] = 0.0
        for i in range(0, self.nqs):
            self.thetas_g[i] = self.thetas[i].fft()

    def vdw_energy(self):
        grid = self.thetas.grid
        u_vdw = self.thetas_g * 0
        g = grid.get_reciprocal().q
        if grid.full:
            mask = slice(None)
        else:
            mask = grid.get_reciprocal().mask
        for q2_i in range(0, self.nqs):
            for q1_i in range(0, self.nqs):
                kernel = self.splines_kernel[(q1_i, q2_i)](g)
                u_vdw[q2_i] += kernel * self.thetas_g[q1_i]
        energy = (u_vdw[:, mask] * np.conj(self.thetas_g[:, mask])).sum()
        if not grid.full:
            energy = 2 * energy
            if grid.mp.is_root:
                e0 = (u_vdw[:, 0, 0, 0] * np.conj(self.thetas_g[:, 0, 0, 0])).sum()
                energy -= e0
        energy = 0.5 * energy.real / grid.volume
        self.thetas_g[:] = u_vdw
        return energy

    def get_potential(self, density, gradient_rho):
        u_vdw = self.thetas
        potential = density * 0
        const = 1.0 / (3.0 * self.b_value ** (3.0 / 2.0) * np.pi ** (5.0 / 4.0))
        h_prefactor = np.zeros_like(self.q0)

        mask = density > self.epsr

        for p_i in range(0, self.nqs):
            p = self.splines_q[p_i](self.q0[mask])
            dp_dq0 = self.splines_q[p_i](self.q0[mask], 1)
            dtheta_dn = (
                const * (3.0 / 4.0) / (density[mask] ** (1.0 / 4.0)) * p
                + const * density[mask] ** (3.0 / 4.0) * dp_dq0 * self.dq0_drho[mask]
            )
            dtheta_dgradn = (
                const * density[mask] ** (3.0 / 4.0) * dp_dq0 * self.dq0_dgradrho[mask]
            )
            potential[mask] = potential[mask] + u_vdw[p_i, mask] * dtheta_dn
            h_prefactor[mask] = h_prefactor[mask] + u_vdw[p_i, mask] * dtheta_dgradn
        g = density.grid.get_reciprocal().g
        gmask = density.grid.get_reciprocal().gmask
        for icar in range(0, 3):
            h = gradient_rho[icar] * h_prefactor
            h_g = h.fft()
            h_g[gmask] = 1j * g[icar][gmask] * h_g[gmask]
            h = h_g.ifft(force_real=True)
            potential[:] = potential[:] - h
        return potential

    def radial_fft(self, phi, r, k):
        n = len(phi)
        dr = r[2] - r[1]
        phi_k = np.zeros_like(phi)
        phi_k[0] = (phi * r**2).sum()
        phi_k[0] = phi_k[0] - 0.5 * r[-1] ** 2 * phi[-1]
        for i in range(1, n):
            phi_k[i] = (phi[1:] * r[1:] * np.sin(k[i] * r[1:]) / k[i]).sum()
        phi_k[1:] -= 0.5 * phi[-1] * r[-1] * np.sin(k[-1] * r[-1]) / k[-1]
        phi_k = 4.0 * np.pi * phi_k * dr
        return phi_k


class RVV10(AbstractFunctional):
    def __init__(self, **kwargs):
        self.libxc = ['gga_x_rpw86', 'gga_c_pbe']
        self.rvv10nl = RVV10NL(**kwargs)
        from dftpy.functional.xc.semilocal_xc import LibXC

        self.xcfun = LibXC

    def compute(self, density, **kwargs):
        functional = self.rvv10nl(density, **kwargs)
        kwargs.pop('libxc', None)
        fun = self.xcfun(density, libxc=self.libxc, **kwargs)
        functional += fun
        return functional
