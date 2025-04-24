import os

import numpy as np

from dftpy.config.config import PrintConf, ReadConf
from dftpy.constants import ENERGY_CONV, LEN_CONV, STRESS_CONV
from dftpy.density import DensityGenerator
from dftpy.field import DirectField
from dftpy.formats.io import read, read_density, write
from dftpy.functional import Functional
from dftpy.functional.functional_output import FunctionalOutput
from dftpy.functional.total_functional import TotalFunctional
from dftpy.grid import DirectGrid
from dftpy.inverter import Inverter
from dftpy.math_utils import ecut2nr
from dftpy.mixer import Mixer
from dftpy.mpi import sprint
from dftpy.optimization import OESCF, Optimization
from dftpy.properties import get_electrostatic_potential
from dftpy.utils import field2distrib


def ConfigParser(config, ions=None, rhoini=None, pseudo=None, grid=None, mp=None):
    if isinstance(config, dict):
        pass
    elif isinstance(config, str):
        # config is a file
        config = ReadConf(config)
        PrintConf(config)

    # check the input
    if grid is not None and config["MATH"]["multistep"] > 1:
        raise AttributeError("Given the 'grid', can't use 'multistep' method anymore.")

    if ions is None:
        ions, field, _ = read(
            config["PATH"]["cellpath"] + os.sep + config["CELL"]["cellfile"],
            format=config["CELL"]["format"],
            names=config["CELL"]["elename"],
            kind='all',
        )
    else:
        field = None
    lattice = ions.cell
    if config["GRID"]["nr"] is not None:
        nr = np.asarray(config["GRID"]["nr"])
    elif field is not None and field is not None:
        nr = field.grid.nr
        rhoini = field
    else:
        spacing = config["GRID"]["spacing"] * LEN_CONV["Angstrom"]["Bohr"]
        grid_options = config["GRID"].copy()
        grid_options.pop("spacing", None)
        nr = ecut2nr(lattice=lattice, spacing=spacing, **grid_options)
    sprint("The final grid size is ", nr)
    nr2 = nr.copy()
    if config["MATH"]["multistep"] > 1:
        nr = nr2 // config["MATH"]["multistep"]
        sprint("MULTI-STEP: Perform first optimization step")
        sprint("Grid size of 1  step is ", nr)

    ############################## Grid  ##############################
    if grid is None:
        grid = DirectGrid(
            lattice=lattice,
            nr=nr,
            full=config["GRID"]["gfull"],
            cplx=config["GRID"]["cplx"],
            mp=mp,
        )
    if mp is None:
        mp = grid.mp
    ############################## PSEUDO  ##############################
    PPlist = {}
    for key in config["PP"]:
        ele = key.capitalize()
        PPlist[ele] = config["PATH"]["pppath"] + os.sep + config["PP"][key]
    optional_kwargs = {}
    optional_kwargs["PP_list"] = PPlist
    optional_kwargs["ions"] = ions
    optional_kwargs["PME"] = config["MATH"]["linearie"]
    #
    linearie = config["MATH"]["linearie"]
    if pseudo is None:
        # PSEUDO = LocalPseudo(grid=grid, ions=ions, PP_list=PPlist, PME=linearie)
        PSEUDO = Functional(
            type='PSEUDO', grid=grid, ions=ions, PP_list=PPlist, PME=linearie
        )
    else:
        PSEUDO = pseudo

        PSEUDO.restart(full=False)
        PSEUDO.grid = grid
        PSEUDO.ions = ions
    kedf_config = config["KEDF"].copy()
    if kedf_config.get('temperature', None):
        kedf_config['temperature'] *= ENERGY_CONV['eV']['Hartree']
    if kedf_config.get('temperature0', None):
        kedf_config['temperature0'] *= ENERGY_CONV['eV']['Hartree']
    if config['OPT']['algorithm'] == 'OESCF':
        KE = Functional(type="KEDF", name='vW')
        if config["KEDF"]["kedf"].startswith('GGA') or config["KEDF"][
            "kedf"
        ].startswith('MGGA'):
            kedf_config['gga_remove_vw'] = True
        kedf_config['y'] = 0.0
        kedf_emb = Functional(type="KEDF", name=config["KEDF"]["kedf"], **kedf_config)
        evaluator_emb = TotalFunctional(KEDF_EMB=kedf_emb)
    else:
        KE = Functional(type="KEDF", name=config["KEDF"]["kedf"], **kedf_config)
        evaluator_emb = None
    ############################## XC and Hartree ##############################
    HARTREE = Functional(type="HARTREE")
    XC = Functional(type="XC", name=config["EXC"]["xc"], pseudo=PSEUDO, **config["EXC"])
    if config["NONADIABATIC"]["nonadiabatic"] is None:
        DYNAMIC = None
    else:
        DYNAMIC = Functional(
            type="DYNAMIC",
            name=config["NONADIABATIC"]["nonadiabatic"],
            **config["NONADIABATIC"],
        )
    E_v_Evaluator = TotalFunctional(
        KineticEnergyFunctional=KE,
        XCFunctional=XC,
        HARTREE=HARTREE,
        PSEUDO=PSEUDO,
        DYNAMIC=DYNAMIC,
    )
    ############################## Initial density ##############################
    nspin = config["DENSITY"]["nspin"]
    rho_ini = DirectField(grid=grid, rank=nspin)
    density = None
    root = 0
    if rhoini is not None:
        density = rhoini
    elif config["DENSITY"]["densityini"] == "heg":
        charge_total = ions.get_ncharges()
        rho_ini[:] = charge_total / ions.cell.volume
        if nspin > 1:
            rho_ini[:] /= nspin
    elif config["DENSITY"]["densityini"] == "read":
        density = []
        if mp.rank == root:
            density = read_density(config["DENSITY"]["densityfile"])
    elif config["DENSITY"]["densityini"] == "atomic":
        dg = DensityGenerator(pseudo=PSEUDO)
        rho_ini = dg.guess_rho(ions, grid=grid, rho=rho_ini, nspin=nspin)

    if density is not None:
        field2distrib(density, rho_ini, root=root)
    # normalization the charge (e.g. the cell changed)
    charge_total = ions.get_ncharges()
    rho_ini *= charge_total / np.sum(rho_ini.integral())
    ############################## add spin magmom ##############################
    if nspin > 1:
        magmom = config["DENSITY"]["magmom"]
        for ip in range(nspin):
            rho_ini[ip] += 1.0 / nspin * (-1) ** ip * magmom / ions.cell.volume
    # -----------------------------------------------------------------------
    # The last is a dictionary, which return some properties are used for different situations.
    others = {
        "ions": ions,
        "field": rho_ini,
        "rho": rho_ini,
        "E_v_Evaluator": E_v_Evaluator,
        "nr2": nr2,
        "evaluator_emb": evaluator_emb,
    }
    return config, others


def OptimizeDensityConf(
    config,
    ions=None,
    rho=None,
    E_v_Evaluator=None,
    nr2=None,
    evaluator_emb=None,
    **kwargs,
):
    rho_ini = rho
    grid = rho_ini.grid
    if hasattr(E_v_Evaluator, 'PSEUDO'):
        PSEUDO = E_v_Evaluator.PSEUDO
    nr = grid.nr
    charge_total = ions.get_ncharges()
    if config['OPT']['algorithm'] == 'OESCF':
        lscf = True
    else:
        lscf = False
    # -----------------------------------------------------------------------
    mix_kwargs = config["MIX"].copy()
    if mix_kwargs['predecut']:
        mix_kwargs['predecut'] *= ENERGY_CONV["eV"]["Hartree"]
    mixer = Mixer(**mix_kwargs)
    # -----------------------------------------------------------------------
    if "Optdensity" in config["JOB"]["task"]:
        optimization_options = config["OPT"].copy()
        optimization_options["econv"] *= ions.nat
        # if config['MATH']['twostep'] :
        # optimization_options["econv"] *= 10
        if lscf:
            optimization_options['algorithm'] = 'EMM'
        opt = Optimization(
            EnergyEvaluator=E_v_Evaluator,
            optimization_options=optimization_options,
            optimization_method=config["OPT"]["method"],
        )
        if lscf:
            oescf = OESCF(optimization=opt, evaluator_emb=evaluator_emb, mixer=mixer)
            opt = oescf
        rho = opt.optimize_rho(guess_rho=rho_ini)
        # perform second step, dense grid
        # -----------------------------------------------------------------------
        for istep in range(2, config["MATH"]["multistep"] + 1):
            if istep == config["MATH"]["multistep"]:
                if nr2 is None:
                    nr2 = nr * 2
                nr = nr2
            else:
                nr = nr * 2
            sprint("#" * 80)
            sprint("MULTI-STEP: Perform %d optimization step" % istep)
            sprint("Grid size of %d" % istep, " step is ", nr)
            grid2 = DirectGrid(
                lattice=grid.lattice, nr=nr, full=config["GRID"]["gfull"], mp=grid.mp
            )
            rho_ini = DirectField(grid=grid2, rank=rho.rank)
            field2distrib(rho, rho_ini, root=0)
            rho_ini *= charge_total / (np.sum(rho_ini.integral()))
            if hasattr(E_v_Evaluator, 'PSEUDO'):
                PSEUDO = E_v_Evaluator.PSEUDO
                PSEUDO.restart(full=False, ions=PSEUDO.ions, grid=grid2)
            rho = opt.optimize_rho(guess_rho=rho_ini)
        optimization_options["econv"] /= ions.nat  # reset the value
    ############################## calctype  ##############################
    if lscf:
        E_v_Evaluator.UpdateFunctional(newFuncDict=evaluator_emb.funcDict)
    sprint("-" * 80)
    calcType = set()

    if 'Both' in config["JOB"]["calctype"]:
        calcType.update({"E", "V"})
    if 'Potential' in config["JOB"]["calctype"]:
        calcType.update("V")
    if 'Energy' in config["JOB"]["calctype"]:
        calcType.update("E")
    if 'Force' in config["JOB"]["calctype"]:
        calcType.update("F")
    if 'Stress' in config["JOB"]["calctype"]:
        calcType.update("S")

    ############################## Output Density ##############################
    if config["DENSITY"]["densityoutput"]:
        sprint("Write Density...")
        outfile = config["DENSITY"]["densityoutput"]
        write(outfile, rho, ions)
    ############################## Output ##############################
    if config["OUTPUT"]["electrostatic_potential"]:
        sprint("Write electrostatic potential...")
        outfile = config["OUTPUT"]["electrostatic_potential"]
        v = get_electrostatic_potential(rho, E_v_Evaluator)
        write(outfile, v, ions)
    print_stress = config["OUTPUT"]["stress"]
    results = evaluator2results(
        E_v_Evaluator,
        rho=rho,
        calcType=calcType,
        ions=ions,
        print_stress=print_stress,
        **kwargs,
    )
    if lscf:
        E_v_Evaluator.UpdateFunctional(keysToRemove=evaluator_emb.funcDict)
    # sprint('-' * 31, 'Time information', '-' * 31)
    # -----------------------------------------------------------------------
    return results


def InvertRunner(config, ions, EnergyEvaluater):
    file_rho_in = config["INVERSION"]["rho_in"]
    file_v_out = config["INVERSION"]["v_out"]
    field = read_density(file_rho_in)

    if ions.cell != field.cell:
        raise ValueError(
            'The grid of the input density does not match the grid of the system'
        )

    inv = Inverter()
    ext = inv(field, EnergyEvaluater)
    write(file_v_out, data=ext.v, ions=ions, data_type='potential')

    return ext


def evaluator2results(
    evaluator,
    rho=None,
    calcType={'E', 'F'},
    ions=None,
    print_stress=False,
    split=True,
    **kwargs,
):
    forces = {}
    stress = {}
    energypotential = {'TOTAL': FunctionalOutput(name='TOTAL', energy=0.0)}
    if 'E' in calcType or 'V' in calcType:
        sprint("Calculate Energy/Potential...")
        energypotential = evaluator.get_energy_potential(
            rho, calcType=calcType, split=split
        )

    if 'E' in calcType:
        sprint(format("Energy information", "-^80"))
        keys = list(energypotential.keys())
        ke_energy = 0.0
        for key in sorted(keys):
            if key == "TOTAL":
                continue
            value = energypotential[key].energy
            sprint(
                "{:>10s} energy (eV): {:22.15E}".format(
                    key, value * ENERGY_CONV["Hartree"]["eV"]
                )
            )
            if key.startswith('KEDF'):
                ke_energy += value
        etot = energypotential["TOTAL"].energy
        sprint(
            "{:>10s} energy (eV): {:22.15E}".format(
                "TOTAL", etot * ENERGY_CONV["Hartree"]["eV"]
            )
        )
        sprint("-" * 80)

        etot_eV = etot * ENERGY_CONV["Hartree"]["eV"]
        fstr = "  {:<30s} : {:30.15f}"
        sprint(fstr.format("kedfs energy (a.u.)", ke_energy))
        sprint(
            fstr.format("kedfs energy (eV)", ke_energy * ENERGY_CONV["Hartree"]["eV"])
        )
        sprint(fstr.format("total energy (a.u.)", etot))
        sprint(fstr.format("total energy (eV)", etot_eV))
        if ions is not None:
            etot_eV_patom = etot * ENERGY_CONV["Hartree"]["eV"] / ions.nat
            sprint(fstr.format("total energy (eV/atom)", etot_eV_patom))
    ############################## Force ##############################
    if "F" in calcType:
        sprint("Calculate Force...")
        forces = evaluator.get_forces(rho, split=split)
        ############################## Output Force ##############################
        f = np.abs(forces["TOTAL"])
        fmax, fmin, fave = np.max(f), np.min(f), np.mean(f)
        fstr_f = " " * 8 + "{0:>22s} : {1:<22.5f}"
        sprint(fstr_f.format("Max force (a.u.)", fmax))
        sprint(fstr_f.format("Min force (a.u.)", fmin))
        sprint(fstr_f.format("Ave force (a.u.)", fave))
        sprint("-" * 80)
    ############################## Stress ##############################
    if "S" in calcType:
        sprint("Calculate Stress...")
        stress = evaluator.get_stress(rho, split=split)
        ############################## Output stress ##############################
        fstr_s = " " * 16 + "{0[0]:12.5f} {0[1]:12.5f} {0[2]:12.5f}"
        if print_stress:
            for key in sorted(stress.keys()):
                if key == "TOTAL":
                    continue
                value = stress[key]
                sprint("{:>10s} stress (a.u.): ".format(key))
                for i in range(3):
                    sprint(fstr_s.format(value[i]))
        sprint("{:>10s} stress (a.u.): ".format("TOTAL"))
        for i in range(3):
            sprint(fstr_s.format(stress["TOTAL"][i]))
        sprint("{:>10s} stress (GPa): ".format("TOTAL"))
        for i in range(3):
            sprint(fstr_s.format(stress["TOTAL"][i] * STRESS_CONV["Ha/Bohr3"]["GPa"]))
        sprint("-" * 80)

    results = {}
    results["density"] = rho
    results["energypotential"] = energypotential
    results["forces"] = forces
    results["stress"] = stress
    results["evaluator"] = evaluator
    if hasattr(evaluator, 'PSEUDO'):
        results["pseudo"] = evaluator.PSEUDO
    return results
