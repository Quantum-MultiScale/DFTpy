import os
import os.path

import numpy as np

from dftpy.formats import io
from dftpy.mpi import sprint
from dftpy.td.casida import Casida
from dftpy.td.hamiltonian import Hamiltonian


def CasidaRunner(config, rho0, E_v_Evaluator):
    numeig = config["CASIDA"]["numeig"]
    outfile = config["TD"]["outfile"]
    diagonalize = config["CASIDA"]["diagonalize"]
    tda = config["CASIDA"]["tda"]

    if diagonalize:
        potential = E_v_Evaluator(rho0, calcType={'V'}).potential
        hamiltonian = Hamiltonian(potential)
        sprint('Start diagonalizing Hamiltonian.')
        eigs, psi_list = hamiltonian.diagonalize(numeig)
        sprint('Diagonalizing Hamiltonian done.')
    else:
        raise Exception("diagonalize must be true.")

    E_v_Evaluator.UpdateFunctional(keysToRemove=['HARTREE', 'PSEUDO'])
    casida = Casida(rho0, E_v_Evaluator)

    sprint('Start building matrix.')
    casida.build_matrix(numeig, eigs, psi_list, build_ab=tda)
    sprint('Building matrix done.')

    if tda:
        omega, f = casida.tda()
    else:
        omega, f, x_minus_y_list = casida()

    with open(outfile, 'w') as fw:
        for i in range(len(omega)):
            fw.write('{0:15.8e} {1:15.8e}\n'.format(omega[i], f[i]))

    # if save_eigenvectors:
    #    if not os.isdir(ev_path):
    #        os.mkdir(ev_path)
    #    i = 0
    #    for x_minus_y in x_minus_y_list:
    #        with open('{0:s}/x_minus_y{1:d}',format(ev_path, i), 'w') as fw:


def DiagonalizeRunner(config, field, ions, E_v_Evaluator):
    numeig = config["CASIDA"]["numeig"]
    eigfile = config["TD"]["outfile"]
    direct_to_psi = './xsf'

    potential = E_v_Evaluator(field, calcType={'V'}).potential
    hamiltonian = Hamiltonian(potential)
    sprint('Start diagonalizing Hamiltonian.')
    eigs, psi_list = hamiltonian.diagonalize(numeig)
    sprint('Diagonalizing Hamiltonian done.')

    np.savetxt(eigfile, eigs, fmt='%15.8e')

    if not os.path.isdir(direct_to_psi):
        os.mkdir(direct_to_psi)
    for i in range(len(eigs)):
        io.write('{0:s}/psi{1:d}.xsf'.format(direct_to_psi, i), ions, psi_list[i])


# def SternheimerRunner(config, rho0, E_v_Evaluator):
#     outfile = config["TD"]["outfile"]
#
#     sternheimer = Sternheimer(rho0, E_v_Evaluator)
#     eigs, psi_list = sternheimer.hamiltonian.diagonalize(2)
#     sternheimer.grid.full = True
#     omega = np.linspace(0.0, 0.5, 26)
#     f = sternheimer(psi_list[1], omega, 0)
#     # f = omega
#     # sternheimer(psi_list[1], 1e-4, 0.01)
#
#     with open(outfile, 'w') as fw:
#         for i in range(len(omega)):
#             fw.write('{0:15.8e} {1:15.8e}\n'.format(omega[i], f[i]))
