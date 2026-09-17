Hamiltonian diagonalization
===========================

DFTpy can diagonalize the Hamiltonian on either the full real-space grid or
in a reduced reciprocal-space basis.  The reduced basis contains the plane
waves that satisfy

.. math::

   \frac{1}{2}|\mathbf{G}|^2 \leq E_{\mathrm{cut}}.

Reducing the basis decreases the number of coefficients handled by the
eigensolver.  DFTpy uses the same LOBPCG algorithm for serial and MPI
reduced-basis calculations, so no separate MPI Hamiltonian class is needed.

Serial calculation
------------------

Create the grid with a serial ``MP`` context.  Supplying both
``reciprocal=True`` and ``ecut`` selects the reduced-basis solver:

.. code-block:: python

   import numpy as np

   from dftpy.field import DirectField
   from dftpy.grid import DirectGrid
   from dftpy.mpi import MP
   from dftpy.td.hamiltonian import Hamiltonian

   mp = MP(parallel=False)
   grid = DirectGrid(
       lattice=np.eye(3) * 8.0,
       nr=[8, 8, 8],
       mp=mp,
   )
   potential = DirectField(grid=grid)
   potential[:] = 0.2
   hamiltonian = Hamiltonian(potential)

   eigenvalues, wavefunctions = hamiltonian.diagonalize(
       numeig=4,
       reciprocal=True,
       ecut=20.0,
       tol=1.0e-8,
       maxiter=80,
   )

This example uses a cubic cell with a uniform potential of ``0.2`` atomic
units.  It is intentionally small so that it can be used as a quick example;
in a physical calculation, ``potential`` would normally be the effective
potential obtained from a converged ground-state calculation.

MPI calculation
---------------

For an MPI calculation, construct the grid and potential with
``MP(parallel=True)`` and use the same ``Hamiltonian`` interface:

.. code-block:: python

   import numpy as np

   from dftpy.field import DirectField
   from dftpy.grid import DirectGrid
   from dftpy.mpi import MP
   from dftpy.td.hamiltonian import Hamiltonian

   mp = MP(parallel=True)
   grid = DirectGrid(
       lattice=np.eye(3) * 8.0,
       nr=[8, 8, 8],
       mp=mp,
   )
   potential = DirectField(grid=grid)
   potential[:] = 0.2
   hamiltonian = Hamiltonian(potential)

   eigenvalues, wavefunctions = hamiltonian.diagonalize(
       numeig=4,
       reciprocal=True,
       ecut=20.0,
       tol=1.0e-8,
       maxiter=80,
   )


DFTpy determines whether the calculation is serial or parallel from the MPI
context attached to the potential's grid.  In MPI mode, the reciprocal-space
coefficients and FFT operations are distributed across ranks.  Inner
products, norms, and projected matrices use MPI collective reductions.

Full and reduced Hamiltonians
-----------------------------

If ``ecut`` is omitted, DFTpy uses the original full-grid SciPy
diagonalization.  For example, the full real-space Hamiltonian is selected
with

.. code-block:: python

   eigenvalues, wavefunctions = hamiltonian.diagonalize(
       numeig=4,
       reciprocal=False,
   )

The reduced-basis eigenvalues approach the full-grid eigenvalues as ``ecut``
is increased.  The cutoff must therefore be converged for each system.  A
useful convergence procedure is to increase ``ecut`` until the required
eigenvalues change by less than the desired accuracy.

Solver options
--------------

The main reduced-basis options are

``numeig``
   Number of lowest eigenvalues and eigenvectors to calculate.

``ecut``
   Reciprocal-basis kinetic-energy cutoff.  It must be positive and must
   retain more plane waves than the requested number of eigenvalues.

``tol``
   LOBPCG residual convergence tolerance.

``maxiter``
   Maximum number of LOBPCG iterations.

``block_size``
   Number of trial vectors advanced by LOBPCG.  The default is selected from
   ``numeig``.

``matvec_block_size``
   Maximum number of wavefunctions transformed together in one block.  This
   can be reduced when memory is limited.
