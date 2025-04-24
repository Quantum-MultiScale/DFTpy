def GetConf():
    import argparse

    from dftpy.mpi import pmi

    parser = argparse.ArgumentParser(description='Process task')
    parser.add_argument('confs', nargs='*')
    parser.add_argument(
        '-i',
        '--ini',
        '--input',
        dest='input',
        type=str,
        action='store',
        default='config.ini',
        help='Input file (default: config.ini)',
    )
    parser.add_argument(
        '--mpi',
        '--mpi4py',
        dest='mpi',
        action='store_true',
        default=False,
        help='Use mpi4py to be parallel',
    )

    args = parser.parse_args()
    if args.mpi or pmi.size > 0:
        from mpi4py import MPI

        from dftpy.mpi import mp

        mp.comm = MPI.COMM_WORLD
    return args


def RunJob(args):
    import time

    from dftpy import __version__
    from dftpy.interface import ConfigParser, InvertRunner, OptimizeDensityConf
    from dftpy.mpi import mp, sprint
    from dftpy.td.interface import CasidaRunner, DiagonalizeRunner
    from dftpy.td.real_time_runner import RealTimeRunner
    from dftpy.time_data import TimeData

    sprint(
        "DFTpy {} Begin on : {}".format(
            __version__, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        )
    )
    if mp.is_mpi:
        info = 'Parallel version (MPI) on {0:>8d} processors'.format(mp.comm.size)
    else:
        info = 'Serial version on {0:>8d} processors'.format(mp.comm.size)
    sprint(info)
    if len(args.confs) == 0:
        args.confs.append(args.input)
    for fname in args.confs:
        config, others = ConfigParser(fname, mp=mp)
        sprint("#" * 80)
        TimeData.Begin("TOTAL")

        if "Propagate" in config["JOB"]["task"]:
            realtimerunner = RealTimeRunner(
                others["field"], config, others["E_v_Evaluator"]
            )
            realtimerunner()
        elif "Casida" in config["JOB"]["task"]:
            CasidaRunner(config, others["field"], others["E_v_Evaluator"])
        elif "Diagonalize" in config["JOB"]["task"]:
            DiagonalizeRunner(
                config, others["field"], others["ions"], others["E_v_Evaluator"]
            )
        elif "Inversion" in config["JOB"]["task"]:
            InvertRunner(config, others["field"], others["E_v_Evaluator"])
        else:
            OptimizeDensityConf(config, **others)

        TimeData.End("TOTAL")
        TimeData.output(config, sort=1)
        sprint("-" * 80)
    sprint("#" * 80)
    sprint(
        "DFTpy {} Finished on : {}".format(
            __version__, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        )
    )


def main():
    import sys

    from dftpy.mpi.utils import sprint

    if sys.version_info < (3, 6):
        sprint('Please upgrade your python to version 3.6 or higher.')
        sys.exit()
    args = GetConf()
    RunJob(args)
