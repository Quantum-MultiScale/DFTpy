.. _ofdft:

Introduction to DFT
====================

Density Functional Theory (DFT)
--------------------------------
DFT is a theoretical framework based on the Hohenberg and Kohn theorems, that established a one-to-one correspondence between the ground state electron density, :math:`n_0(\mathbf{r})`, the ground state wavefunction, :math:`\Psi_0(\mathbf{r})`, and the external potential, :math:`v(\mathbf{r})`. This means that the ground state properties of a system can be fully determined by the density.

.. math::
   \psi_0 \leftrightarrow n_0 \leftrightarrow v(\mathbf{r})

The ground state total energy is then given as a functional of the electron density, :math:`E[n]`, which is given by: 

.. math::
   E[n] = T[n] + E_{ee}[n] + E_{eN}[n] + E_{NN}[n]

where :math:`T[n]` is the kinetic energy, :math:`E_{ee}[n]` is the electron-electron interaction, :math:`E_{eN}[n]` the electron-nuclear interaction, and :math:`E_{NN}[n]` is the nuclear-nuclear interaction.

To determine the density of the system and evaluate the energy functional, DFT explores the mapping between the real system and fictitious systems shown in Figure 1.

.. figure:: static/maps.svg
   :align: center
   :width: 50%

   Figure1. Mapping between the real system and fictitious systems in DFT.

Kohn-Sham System
----------------

In the non-interacting fermionic system, known as the single-particle or `Kohn-Sham` system, the total energy functional is given by:

.. math::
   E[n] = T_s[n] + E_H[n] + E_{xc}[n] + \int v_{eN}(\mathbf{r}) n(\mathbf{r}) d\mathbf{r} + E_{NN}[n]

.. where :math:`T_s[n]` is the non-interacting kinetic energy functional, :math:`E_H[n]` is the Hartree energy (electron-electron interaction), :math:`E_{xc}[n]` is the exchange-correlation energy, :math:`v_{ext}(\mathbf{r})` is the external potential (electron-nuclear interaction), and :math:`E_{NN}[n]` is the nuclear-nuclear interaction functional.

where the density :math:`n(r) = \sum_i |\phi_i(r)|^2`. And the single-particle kinetic energy is

.. math::
   T_{s}[n] \equiv T_s[\{\phi_i\}]=  -\frac{1}{2}\sum_i n_i \langle \phi_i | \nabla^2 | \phi_i\rangle = -\frac{1}{2}\sum_i n_i \int \phi_i^*(r) \nabla^2 \phi_i(r) dr

The electron-electron repulsion and the total kinetic energy are related to :math:`T_s` and :math:`E_{xc}` as follows:

.. math::
   E_H[n]=\frac{1}{2}\int \frac{n(r)n(r')}{|r-r'|}drdr'

.. math::
   E_{xc} =  T[n] + E_{ee}[n] = T_s[n] + E_H[n] + E_{xc}[n] \to \textbf{Approximated}


For the Kohn-Sham system, the Lagrangian is defined as follows:

.. math::
   \mathcal{L}_{KS}[\{\phi_i\}] = E[\{\phi_i\}] - \sum_{ij} \varepsilon_{ij}\left(\langle \phi_j|\phi_i \rangle - \delta_{ij}\right)

to find the ground state KS orbitals and ground state density, the KS lagrangian is minimized :math:`\frac{\delta \mathcal{L}_{KS}[\{\phi_i\}]}{\delta \langle \phi_j|}=0`, or just :math:`\frac{\delta \mathcal{L}_{KS}[\{\phi_i\}]}{\delta \phi_j^*(r)}=0`, which yields to the KS-equation for the non-interacting fermionic system:

.. important::

   .. math::
      \left(-\frac{1}{2} \nabla^2 + v_s(\mathbf{r})\right) \phi_i(\mathbf{r}) = \epsilon_i \phi_i(\mathbf{r})

where :math:`\psi_i(\mathbf{r})` are the KS orbitals, and :math:`v_s(\mathbf{r})` is the Kohn-Sham potential given by:

.. math::
   v_s[n](r) = \frac{\delta E_{H}[n]}{\delta n(r)} + \frac{\delta E_{xc}[n]}{\delta n(r)} + v_{eN}(r)


Orbital-Free DFT
----------------

In the non-interacting bosonic system, used in Orbital-Free DFT, the total energy functionals is given by:

.. math::
   E[n] = T_{B}[n] + T_P[n] + E_{H}[n] + E_{xc}[n] + \int v_{eN}[n](r) n(r) dr  + E_{NN}

where :math:`T_P[n] = T_{s}[n] - T_{vW}[n]`, :math:`T_{s}[n]` is the non-interacting kinetic energy functional (KEDF, which is an :math:`\textbf{approximated}` functional), and :math:`T_{B}[n]` is the bosonic kinetic energy functional (von Weizsäcker kinetic energy functional), given by:

.. math::
   T_{B}[n] = T_{vW}[n] = -\frac{1}{2}\int \phi^*(r) \nabla^2 \phi(r) dr

where: :math:`\phi(r)=\sqrt{n(r)}`. The appropiate Lagrangian for OF-DFT is given by:

.. math::
   \mathcal{L}_{OF}[n] = E[n] - \mu \left( \int n(\mathbf{r}) d\mathbf{r} - N \right)

where :math:`\mu` is the Lagrange multiplier, :math:`N` is the number of valence electrons in the system. Both :math:`n_0(\mathbf{r})` and  :math:`\mu` are determined during the minimization.

To find the ground state density, the Lagrangian is minimized :math:`\frac{\delta \mathcal{L}_{OF}[n]}{\delta \langle \phi|}=0` or :math:`\frac{\delta \mathcal{L}_{OF}[n]}{\delta \phi^*(r)}=0`, which yields to the KS-equation for the non-interacting bosonic system:

.. important::

   .. math::
      -\frac{1}{2}\nabla^2 \phi(r) + v_B[n](r)\phi(r) = \mu\phi(r)

where :math:`\phi(r)` is the bosonic orbitals, and :math:`v_B[n](r)` is the bosonic potential given by:

.. math::
   v_B[n](r) = \underbrace{\frac{\delta T_{s}[n]}{\delta n(r)} - \frac{\delta T_{vW}[n]}{\delta n(r)}}_{\frac{\delta T_P[n]}{\delta n(r)}} + \frac{\delta E_{H}[n]}{\delta n(r)} + \frac{\delta E_{xc}[n]}{\delta n(r)} + v_{eN}(r)

Solvers for OF-DFT and KS-DFT
-----------------------------

In KS-DFT, the ground state electron density, :math:`n_0(\mathbf{r})`, is obtained from a self-consistent field (SCF) iteration, as shown in the table below. 

In OF-DFT, the ground state electron density, :math:`n_0(\mathbf{r})`, is obtained from the direct minimization of the ground state energy density functional, :math:`E[n]`, as shown in the table below. The ground state energy is, :math:`E_0 = E[n_0]`.


.. list-table::
   :class: solver-comparison
   :widths: 18 41 41
   :header-rows: 1
   :stub-columns: 1
   :align: center

   * - Method
     - OF-DFT
     - KS-DFT
   * - Direct Minimization
     - :math:`n_0(\mathbf{r})=\arg\underset{n}{\min}\big\{ \mathcal{L}_{OF}[n]\big\}`
     - :math:`\{\phi_i^0\} = \arg\underset{\{\phi_i\}}{\min}\big\{ \mathcal{L}_{KS}[\{\phi_i\}]\big\}`
   * - SCF
     - :math:`-\frac{1}{2}\nabla^2 \sqrt{n(r)} + v_B(r)\sqrt{n(r)} = \mu\sqrt{n(r)}`
     - :math:`-\frac{1}{2}\nabla^2 \phi_i(r) + v_s(r)\phi_i(r) = \varepsilon_i\phi_i(r)`
   
.. In practice, the above minimization can only be carried out if the ground state energy functional is known as a pure functional of the density. The energy functional is a sum of several terms: 

.. .. math::
..    E[n]=T_s[n]+E_H[n]+E_{xc}[n]+\int v_{ext}(\mathbf{r}) n(\mathbf{r}) d\mathbf{r}

.. where

..     * :math:`T_s[n]`: noninteracting kinetic energy or KEDF. 
..     * :math:`E_{xc}[n]`: exchange-correlation energy or EXC. 
..     * :math:`E_{H}[n]=\frac{1}{2}\int \frac{n(\mathbf{r})n(\mathbf{r}^\prime)}{|\mathbf{r}-\mathbf{r}^\prime|}d\mathbf{r} d\mathbf{r}^\prime`: Hartree energy.
..     * :math:`v_{ext}(\mathbf{r})`: the external potential (typically the electron-ion interaction).


.. note:: In DFTpy, :math:`T_s[n]` and :math:`E_{xc}[n]` are pure functionals of the density. Check out the tutorials for a list of available KEDF_ and EXC_ functionals, 



.. note::
   DFTpy solves the ground state problem with the so-called `direct energy minimization`. Other (faster) methods are available, such as OESCF_, which is implemented in eDFTpy_. OESCF_ may be implemented in DFTpy upon request.


Time-Dependent Density-Functional Theory (TDDFT)
------------------------------------------------

DFT can also describe a system out of equilibrium by propagating it in `real time`_ or in `frequency space`_ finding the roots of the frequency dependent polarizability (Casida). In the time regime the Runge and Gross theorem is an analog of the Hohenberg and Khon theorem. This theorem formally defines a one-to-one correspondance between densities and potentials, for any fixed initial many-body state, the it follows that the time-dependent density is a unique functional of the potentials and vice versa. This means that the many-body Hamiltonian :math:`\hat{H}(t)` and thus the many-body wave function :math:`\Psi(t)` are functionals of :math:`n(\mathbf{r},t)` as well

Following the Runge and Gross theorem, we can write time-dependent Schrodinger like equation, namely:

.. math::
   \hat{H}(t)  \Psi(\mathbf{r},t) = i \frac{d}{dt}\Psi(\mathbf{r},t)

In the KS formalism the KS orbitals are propagated with a time-dependent Hamiltonian, given by:

.. math::
   \hat{H}(t) = -\frac{1}{2} \nabla^2 + v_s[n](\mathbf{r},t)

where the time-dependent KS potential is given by:

.. math::
v_s[n](\mathbf r,t) = v_{eN}(\mathbf r,t) + v_H[n](\mathbf r,t) + v_{xc}[n](\mathbf r,t).


In the OF formalism the bosonic wavefunction :math:`\Psi(\mathbf{r},t)` is propagated with a time-dependent KS-like Hamiltonian, given by:

.. math:: 
   \hat{H}(t) = -\frac{1}{2} \nabla^2 + v_B[n](\mathbf{r},t).

The Bosonic KS-like potential is given by two major contributions

.. math::
   v_B(\mathbf{r},t) = v_s[n](\mathbf{r},t) + v_P[n](\mathbf{r},t)

where the adiabatic approximation has been invoked fro the `xc` potential :math:`v_{xc}[n(t)]`. The `Pauli` potential is given by adiabatic and nonadiabatic contributions, 

.. math:: 
   v_P(\mathbf{r},t)=v_P^{ad}(\mathbf{r},t)+v_P^{nad}[n](\mathbf{r},t).


.. note::
   The adiabatic Pauli potential can be specified according to any given KEDF_ available in DFTpy.

The nonadiabatic contribution is usually neglected in the literature. In DFTpy the JP_ functional is available,

.. math::
   v_P^{nad}(\mathbf{r},t) = -\frac{\pi^3}{12}\left(\frac{6}{k_F^2(\mathbf{r},t)}\mathcal{F}^{-1}\left\{i\mathbf{q}\cdot\mathbf{j}(\mathbf{q},t)\frac{1}{q}\right\}+\frac{1}{k_F^4(\mathbf{r},t)}\mathcal{F}^{-1}\left\{i\mathbf{q}\cdot\mathbf{j}(\mathbf{q},t)q\right\}\right)

where :math:`\mathbf{j}` and :math:`\mathbf{q}` are the electronic current density and the reciprocal space vector, respectively. The current density is determined by the standard equation :math:`\mathbf{j}(\mathbf{r})=\frac{1}{2i}\left[\Psi^*(\mathbf{r})\nabla\Psi(\mathbf{r})-\Psi(\mathbf{r})\nabla\Psi^*(\mathbf{r})\right]`.  :math:`\mathcal{F}` stands for Fourier transform and :math:`k_F(\mathbf{r},t)=[3\pi^2 n(\mathbf{r},t)]^{1/3}` is the Fermi wavevector function of the local electron density.



.. warning::
   The JP potential is numerically challenging. Refer to the original JP_ publication for details. 



.. note::
   Optical spectra and nonlinear electronic processes can be modelled by DFTpy. See tutorials for additional information. Ehrenfest dynamics is not yet available.

Linear response theory
----------------------

Linear-response TDDFT refers to the determination of excitation energies and excited state properties by solving for the linear-response function.

.. math::
   \delta n(r,\omega) = \int  \chi(r,r',\omega)\,\delta v_{appl}(r',\omega)\,\,dr'

Lehman reresentation of the response function,
.. math::
\chi(r,r',\omega) = \sum_{n}\frac{n_{0n}^*(r)n_{0n}(r')}{\omega-\Omega_{n}+i\eta}-\frac{n_{0n}^*(r)n_{0n}(r')}{\omega+\Omega_{n}+i\eta}

where :math:`n_{0n}(r) = \langle \Psi_0 | \hat n(r) | \Psi_n \rangle` is the ground-to-$n^\text{th}$ excited state transition density, and :math:`\Omega_n = E_n - E_0`.

Applying the Runge-Gross theorem,

.. math::
   \delta n(r,\omega) = \int  \chi(r,r',\omega)\,\delta v_{appl}(r',\omega)\,\,dr' =  \int  \chi_s(r,r',\omega)\,\delta v_{s}(r',\omega)\,\,dr' = \int  \chi_B(r,r',\omega)\,\delta v_{B}(r',\omega)\,\,dr'

where the variation of the KS potential can be decomposed into:

.. math::
   \delta v_{s}(r,\omega) = \delta v_{H}[n](r,\omega) + \delta v_{xc}[n](r,\omega) + \delta v_{appl}(r,\omega)

and the variation of the bosonic potential can be decomposed into crucial contributions

.. math::
   \delta v_{B}(r,\omega) = \delta v_s[n](r,\omega) + \delta v_P[n](r,\omega)

which can be derived from their dependence on the density

.. math::
   \delta v_{H}(r,\omega) = \int \frac{\delta v_{H}(r,\omega)}{\delta n(r',\omega)} \, \delta n(r',\omega) \, dr' = \int \frac{1}{|r-r'|}\, \delta n(r',\omega)\,dr'

.. math::
   \delta v_{xc}(r,\omega) = \int \frac{\delta v_{xc}(r,\omega)}{\delta n(r',\omega)} \, \delta n(r',\omega) \, dr' = \int f_{xc}(r,r',\omega)\, \delta n(r',\omega)\,dr'

.. math::
   \delta v_{P}(r,\omega) = \int \frac{\delta v_{P}(r,\omega)}{\delta n(r',\omega)} \, \delta n(r',\omega) \, dr' = \int f_{P}(r,r',\omega)\, \delta n(r',\omega)\,dr'

The Dyson equation for the response function between the real system and the KS system then given by:

.. math::
   -\chi^{-1} = -\chi_s^{-1} + \frac{1}{|r-r'|} + f_{xc}(r,r',\omega)

while the Dyson equation for the response function between the real system and the OF system then given by:

.. math::
   -\chi^{-1} = -\chi_B^{-1} + \frac{1}{|r-r'|} + f_{xc}(r,r',\omega) + f_P(r,r',\omega)

The Dyson equation turns into a matrix equation, known as the Casida equation, given by:

.. math::
   \left( \begin{array}{cc}
   A_{iajb}(\omega) & B_{iajb}(\omega) \\
   B_{iajb}(\omega) & A_{iajb}(\omega) 
   \end{array} \right) \left( \begin{array}{c}
   X_{ia}  \\
   Y_{ia} 
   \end{array} \right) = \omega \left( \begin{array}{cc}
   1 & 0 \\
   0 & -1 
   \end{array} \right) \left( \begin{array}{c}
   X_{ia}  \\
   Y_{ia} 
   \end{array} \right)

where defining :math:`\omega_{ia} = \varepsilon_a - \varepsilon_i`,

.. math::
   A_{iajb}(\omega) = \delta_{ij}\delta_{ab} \omega_{ia} + \big\langle \phi_i \phi_a \big| K(r,r',\omega) \big| \phi_j \phi_b \big\rangle

and

.. math::
   B_{iajb}(\omega) = \big\langle \phi_i \phi_a \big|  K(r,r',\omega)  \big| \phi_j \phi_b \big\rangle

where in KS :math:`K(r,r',\omega)` is the kernel given by:

.. math::
   K(r,r',\omega) = \frac{1}{|r-r'|} + f_{xc}(r,r',\omega)

while in OF :math:`K(r,r',\omega)` is the kernel given by:

.. math::
   K(r,r',\omega) = \frac{1}{|r-r'|} + f_{xc}(r,r',\omega) + f_P(r,r',\omega)

Collective modes such as **plasmons** are built from many particle–hole pairs in the fermionic KS picture but appear more directly in the bosonic reference, which can make Casida matrices **much smaller** for plasmon-dominated spectra while describing the **same** interacting response once approximations are controlled. Approximate :math:`f_P` (e.g. Thomas–Fermi–von Weizsäcker in linear response) shifts peaks slightly compared with KS-TDDFT; **Landau damping** needs **nonadiabatic** kernels—DFTpy offers the nonadiabatic Pauli (`JP`_) formulation for that regime.

In **frequency space**, DFTpy performs Casida TD-OFDFT using the modules ``Casida``, ``CasidaRunner``, and ``Hamiltonian`` on top of a converged ground-state density (see the linear-response tutorial ``lr-ofdft-tutorial.ipynb``); this parallels **real-time** propagation in ``td-ofdft-tutorial.ipynb``.

Short note on the implementation
--------------------------------

In DFTpy, the electron density is represented in a discrete set of points given by a Cartesian `grid` and contained in a simulation `cell` that is specified by 3 `lattice vectors`. The number of grid points and the cell size are regulated by the user. The Cartesian grid allows for an efficient parallelization of data and work (we use `mpi4py`), and for the exploitation of Fast Fourier Transforms for solving convolution integrals (such as the one needed to compute :math:`E_H[n]`). Either `NumPy.fft` or `PyFFT` are used depending on user input.


References
----------
* `DFTpy release paper (ground state and td-OF-DFT) <https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/wcms.1482>`_
* `DFTpy td-OF-DFT (Casida) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.103.245102>`_
* `DFTpy td-OF-DFT (JP nonadiabatic Pauli potential) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.104.235110>`_
* `OESCF solver for OF-DFT <https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.1c00716>`_


.. _KEDF: tutorials/config.html#kedf
.. _EXC: tutorials/config.html#exc
.. _JP: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.104.235110
.. _`real time`: https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/wcms.1482 
.. _`frequency space`: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.103.245102
.. _OESCF: https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.1c00716
.. _eDFTpy: http://edftpy.rutgers.edu
