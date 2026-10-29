from dftpy.optimization import Optimization
from dftpy.functional import Functional, LocalPseudo, TotalFunctional
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.mpi import MP, sprint
from dftpy.ions import Ions
from ase.build import bulk

def get_KE(name='WT-AD'):
    from dftpy.functional.kedf import KEDF
    from dftpy.functional.ad import WT, LKT, LKT_sech
    from dftpy.functional.ad import register_ad_gga, ADKEDF
    import torch
    import numpy as np

    KE_TF = KEDF(name="TF")
    KE_WT = KEDF(name="WT")
    KE_LKT = KEDF(name='GGA_LKT')

    KE_WT_AD =ADKEDF(WT())
    gname =register_ad_gga(LKT_sech(),overwrite=True)
    KE_LKT_AD=KEDF(name=gname)

    def tf_energy(box, rho):
        return 0.3 * (3 * np.pi ** 2) ** (2 / 3) * rho.pow(5 / 3).mean() * torch.abs(torch.linalg.det(box))
    KE_TF_AD = ADKEDF(tf_energy, name='TF-AD')

    try:
        from professad.functionals import WangTeter
        KE_WT_PROFESSAD = ADKEDF(WangTeter)
    except Exception as e:
        KE_WT_PROFESSAD = None
    kedf = {
            'TF': KE_TF,
            'WT': KE_WT,
            'LKT': KE_LKT,
            'WT-AD': KE_WT_AD,
            'TF-AD': KE_TF_AD,
            'LKT-AD': KE_LKT_AD,
            'WT-PROFESSAD': KE_WT_PROFESSAD,
            }
    return kedf[name]


mp = MP(parallel = False)
ions = Ions.from_ase(bulk('Al', 'fcc', a=4.05, cubic=True))
PP_list = {'Al': '../DATA/al.lda.recpot'}
grid = DirectGrid(ecut=20, lattice=ions.cell, mp=mp)
rho_ini = DirectField(grid=grid)
#
PSEUDO = LocalPseudo(grid=grid, ions=ions, PP_list=PP_list)
XC = Functional(type='XC', name='LDA')
HARTREE = Functional(type='HARTREE')
#
KE = get_KE('LKT-AD')
# KE = get_KE('LKT')
#
funcDict = {'KE' :KE, 'XC' :XC, 'HARTREE' :HARTREE, 'PSEUDO' :PSEUDO}
EnergyEvaluator = TotalFunctional(**funcDict)
#
rho_ini[:] = ions.get_ncharges() / ions.cell.volume
opt = Optimization(EnergyEvaluator=EnergyEvaluator)
new_rho = opt.optimize_rho(guess_rho=rho_ini)
#
Enew = EnergyEvaluator.Energy(rho=new_rho)
sprint('Energy New (a.u.)', Enew, comm = mp)
