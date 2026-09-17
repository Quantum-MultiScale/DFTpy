Casida matrix construction and diagonalization
==============================================

DFTpy uses the Casida equation to calculate linear-response excitation
energies and oscillator strengths from a ground-state density, orbital
energies, and orbitals.  The Casida implementation supports vectorized dense
matrix construction and an optional matrix-free iterative diagonalization.
The same interface can be used with serial and MPI-distributed DFTpy grids.

Creating a Casida object
------------------------

The ``Casida`` class requires the converged ground-state density and the
functional evaluator used to calculate the second derivative of the energy:

.. code-block:: python

   from dftpy.td.casida import Casida

   casida = Casida(rho0, energy_evaluator)

Here, ``rho0`` is a rank-one ``DirectField`` for a spin-unpolarized ground
state.  ``energy_evaluator`` must support ``calcType=['V2']`` and return the
second-order kernel in ``v2rho2``.  Spin-polarized Casida calculations are not
currently implemented.

Dense vectorized calculation
----------------------------

For a moderate number of transitions, build and diagonalize the dense Casida
matrix:

.. code-block:: python

   num_psi = len(psi_list)

   casida.build_matrix(
       num_psi=num_psi,
       eigs=eigenvalues,
       psi_list=psi_list,
       scale=1.0,
       matrix_free=False,
   )

   excitation_energies, oscillator_strengths, x_minus_y = casida()

``psi_list[0]`` and ``eigenvalues[0]`` describe the occupied reference state.
The remaining entries describe the unoccupied states used to form the
transition basis.  Orbital energies and excitation energies are returned in
Hartree atomic units.

The dense implementation forms all transition products at once and uses
matrix multiplication to assemble the interaction matrix.  In MPI mode,
local matrix contributions are combined using collective reductions.

Selecting a subset of modes
---------------------------

Use ``k_modes`` to return only the requested number of lowest excitation
modes:

.. code-block:: python

   excitation_energies, oscillator_strengths, x_minus_y = casida(
       k_modes=10,
   )

For the dense solver, ``k_modes`` may be between one and the total number of
transitions.

Matrix-free calculation
-----------------------

For a larger transition space, ``matrix_free=True`` avoids storing the dense
Casida matrix.  The lowest modes are calculated using SciPy's iterative
``eigsh`` solver:

.. code-block:: python

   casida.build_matrix(
       num_psi=len(psi_list),
       eigs=eigenvalues,
       psi_list=psi_list,
       scale=1.0,
       matrix_free=True,
   )

   excitation_energies, oscillator_strengths, x_minus_y = casida(
       k_modes=10,
   )

In matrix-free mode, ``k_modes`` must be smaller than the total number of
transitions.  Matrix-free triplet calculations are not currently supported.
The dense solver is generally faster for small and moderate matrices; the
matrix-free solver is useful when storing the dense matrix would require too
much memory.

Scaling the Hartree kernel
--------------------------

The ``scale`` argument multiplies the Hartree and exchange-correlation contributions to the Casida
interaction kernel:

.. code-block:: python

   casida.build_matrix(
       len(psi_list),
       eigenvalues,
       psi_list,
       scale=0.5,
   )

The default value is ``1.0``.  The exchange-correlation kernel is not changed
by this parameter.  A non-default scale should only be used when required by
the physical model being studied.

Transition densities and coefficients
--------------------------------------

After diagonalization, calculate the transition density and normalized
orbital weights for a selected mode:

.. code-block:: python

   mode = 0
   transition_density, transition_weights = (
       casida.calc_tdensities_and_tcoeff(
           mode,
           psi_list,
           eigenvalues,
           x_minus_y,
       )
   )

``transition_weights`` contains the squared, normalized contributions of the
individual transitions.  Its elements sum to one.  The phase or sign of a
Casida eigenvector is arbitrary, so transition densities from separate runs
may differ by an overall sign while representing the same excitation.

Tamm--Dancoff approximation
---------------------------

To use the Tamm--Dancoff approximation, request the ``A`` and ``B`` matrices
while building the dense matrix:

.. code-block:: python

   casida.build_matrix(
       len(psi_list),
       eigenvalues,
       psi_list,
       build_ab=True,
   )

   excitation_energies, oscillator_strengths = casida.tda()


