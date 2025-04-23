.. _Pseudopotentials:

Local Pseudopotentials for OF-DFT
=================================

Local Pseudopotentials (LPPs)
-----------------------------


Orbital-Free Density Functional Theory (OF-DFT) relies solely on the electron density, meaning that only local pseudopotentials can be used to compute the ion-electron potential, :math:V_{ne}(\mathbf{r}). Typically, pseudopotentials include nonlocal components (NLPPs), which require knowledge of the one-electron density matrix (DM1). However, in this context, the density matrix is expressed as a functional of the density, :math:\gamma_{1}[n(\mathbf{r})], which is not yet available.

Several types of local pseudopotentials (LPPs) have been developed and successfully applied to specific materials, including:

Bulk-derived Local Pseudopotentials (BLP <https://doi.org/10.1103/PhysRevB.69.125109>_)
High-quality Local Pseudopotentials (HQLPP <https://pubs.acs.org/doi/10.1021/acs.jctc.4c00101>_)
Optimized Effective Local Pseudopotentials (OEPP <https://doi.org/10.1063/1.4944989>_)
A new set of local pseudopotentials for transition metals is available at the following link: LPPs <https://valeriarv99.github.io/OFPP/>_. These LPPs effectively invert nonlocal pseudopotentials (NLPPs) by introducing a short-range correction while preserving the Coulomb tail of the GBRV and PSL pseudopotentials.

If you want to construct your own LPP, follow this tutorial <http://dftpy.rutgers.edu/tutorials/jupyter/lpps.html>_. The formalism behind this new class of pseudopotentials is described in the paper *Pseudononlocal Pseudopotentials for Orbital-Free DFT*.
