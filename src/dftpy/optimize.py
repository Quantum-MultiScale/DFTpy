"""Structure optimization. """

import collections.abc

from dftpy.mpi import mp
from dftpy.utils import IOContext


class Dynamics(IOContext):
    """Base-class for all MD and structure optimization classes."""

    def __init__(self, logfile=None):
        """Dynamics object.

        Parameters:

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.
        """

        if logfile is not None:
            self.logfile = self.openfile(logfile, mode='a', comm=mp.comm)
        self.observers_before_log = []
        self.observers_after_log = []
        self.nsteps = 0
        # maximum number of steps placeholder with maxint
        self.max_steps = 100000000
        self.stop_generator = False

    def __call__(self):
        return self.run()

    def get_number_of_steps(self):
        return self.nsteps

    def insert_observer(
        self, function, position=0, interval=1, before_log=False, *args, **kwargs
    ):
        """Insert an observer."""
        if not isinstance(function, collections.abc.Callable):
            function = function.write
        if before_log:
            self.observers_before_log.insert(
                position, (function, interval, args, kwargs)
            )
        else:
            self.observers_after_log.insert(
                position, (function, interval, args, kwargs)
            )

    def attach(self, function, interval=1, before_log=False, *args, **kwargs):
        """Attach callback function.

        If *interval > 0*, at every *interval* steps, call *function* with
        arguments *args* and keyword arguments *kwargs*.

        If *interval <= 0*, after step *interval*, call *function* with
        arguments *args* and keyword arguments *kwargs*.  This is
        currently zero indexed."""

        if hasattr(function, "set_description"):
            d = self.todict()
            d.update(interval=interval)
            function.set_description(d)
        if not hasattr(function, "__call__"):
            function = function.write
        if before_log:
            self.observers_before_log.append((function, interval, args, kwargs))
        else:
            self.observers_after_log.append((function, interval, args, kwargs))

    def call_observers(self, before_log=False):
        if before_log:
            observers = self.observers_before_log
        else:
            observers = self.observers_after_log
        for function, interval, args, kwargs in observers:
            call = False
            # Call every interval iterations
            if interval > 0:
                if (self.nsteps % interval) == 0:
                    call = True
            # Call only on iteration interval
            elif interval <= 0:
                if self.nsteps == abs(interval):
                    call = True
            if call:
                function(*args, **kwargs)

    def irun(self):
        """Run dynamics algorithm as generator. This allows, e.g.,
        to easily run two optimizers or MD thermostats at the same time.

        Examples:
        >>> opt1 = BFGS(atoms)
        >>> opt2 = BFGS(StrainFilter(atoms)).irun()
        >>> for _ in opt2:
        >>>     opt1.run()
        """

        # compute initial structure and log the first step
        self.initialize()

        # yield the first time to inspect before logging
        yield False

        if self.nsteps == 0:
            self.call_observers(before_log=True)
            self.log()
            self.call_observers()

        # run the algorithm until converged or max_steps reached
        while not self.converged() and self.nsteps < self.max_steps:

            # compute the next step
            self.step()
            self.nsteps += 1

            # let the user inspect the step and change things before logging
            # and predicting the next step
            yield False

            # log the step
            self.call_observers(before_log=True)
            self.log()
            self.call_observers()

            # stop the generator if the self.stop_generator flag is True
            if self.stop_generator:
                return

        # finally check if algorithm was converged
        yield self.converged()

    def run(self):
        """Run dynamics algorithm.

        This method will return when the forces on all individual
        atoms are less than *fmax* or when the number of steps exceeds
        *steps*."""

        for converged in Dynamics.irun(self):
            pass
        return converged

    def converged(self, *args):
        """a dummy function as placeholder for a real criterion, e.g. in
        Optimizer"""
        return False

    def log(self, *args):
        """a dummy function as placeholder for a real logger, e.g. in
        Optimizer"""
        return True

    def initialize(self, *args):
        """a dummy function as placeholder for a real initializer, e.g. in
        Optimizer"""
        pass

    def step(self):
        """this needs to be implemented by subclasses"""
        raise RuntimeError("step not implemented.")


# class Optimizer(Dynamics):
#     """Base-class for all structure optimization classes."""
#
#     # default maxstep for all optimizers
#     defaults = {'maxstep': 0.2}
#
#     def __init__(
#         self,
#         atoms,
#         restart,
#         logfile,
#         trajectory,
#         master=None,
#         append_trajectory=False,
#         force_consistent=False,
#     ):
#         """Structure optimizer object.
#
#         Parameters:
#
#         atoms: Atoms object
#             The Atoms object to relax.
#
#         restart: str
#             Filename for restart file.  Default value is *None*.
#
#         logfile: file object or str
#             If *logfile* is a string, a file with that name will be opened.
#             Use '-' for stdout.
#
#         trajectory: Trajectory object or str
#             Attach trajectory object.  If *trajectory* is a string a
#             Trajectory will be constructed.  Use *None* for no
#             trajectory.
#
#         master: boolean
#             Defaults to None, which causes only rank 0 to save files.  If
#             set to true,  this rank will save files.
#
#         append_trajectory: boolean
#             Appended to the trajectory file instead of overwriting it.
#
#         force_consistent: boolean or None
#             Use force-consistent energy calls (as opposed to the energy
#             extrapolated to 0 K).  If force_consistent=None, uses
#             force-consistent energies if available in the calculator, but
#             falls back to force_consistent=False if not.
#         """
#         Dynamics.__init__(
#             self,
#             atoms,
#             logfile,
#             trajectory,
#             append_trajectory=append_trajectory,
#             master=master,
#         )
#
#         self.force_consistent = force_consistent
#         if self.force_consistent is None:
#             self.set_force_consistent()
#
#         self.restart = restart
#
#         # initialize attribute
#         self.fmax = None
#
#         if restart is None or not isfile(restart):
#             self.initialize()
#         else:
#             self.read()
#             barrier()
#
#     def todict(self):
#         description = {
#             "type": "optimization",
#             "optimizer": self.__class__.__name__,
#         }
#         return description
#
#     def initialize(self):
#         pass
#
#     def irun(self, fmax=0.05, steps=None):
#         """ call Dynamics.irun and keep track of fmax"""
#         self.fmax = fmax
#         if steps:
#             self.max_steps = steps
#         return Dynamics.irun(self)
#
#     def run(self, fmax=0.05, steps=None):
#         """ call Dynamics.run and keep track of fmax"""
#         self.fmax = fmax
#         if steps:
#             self.max_steps = steps
#         return Dynamics.run(self)
#
#     def converged(self, forces=None):
#         """Did the optimization converge?"""
#         if forces is None:
#             forces = self.atoms.get_forces()
#         if hasattr(self.atoms, "get_curvature"):
#             return (forces ** 2).sum(
#                 axis=1
#             ).max() < self.fmax ** 2 and self.atoms.get_curvature() < 0.0
#         return (forces ** 2).sum(axis=1).max() < self.fmax ** 2
#
#     def log(self, forces=None):
#         if forces is None:
#             forces = self.atoms.get_forces()
#         fmax = sqrt((forces ** 2).sum(axis=1).max())
#         e = self.atoms.get_potential_energy(
#             force_consistent=self.force_consistent
#         )
#         T = time.localtime()
#         if self.logfile is not None:
#             name = self.__class__.__name__
#             if self.nsteps == 0:
#                 args = (" " * len(name), "Step", "Time", "Energy", "fmax")
#                 msg = "%s  %4s %8s %15s %12s\n" % args
#                 self.logfile.write(msg)
#
#                 # if self.force_consistent:
#                 #     msg = "*Force-consistent energies used in optimization.\n"
#                 #     self.logfile.write(msg)
#
#             # XXX The "force consistent" handling is really arbitrary.
#             # Let's disable the special printing for now.
#             #
#             # ast = {1: "*", 0: ""}[self.force_consistent]
#             ast = ''
#             args = (name, self.nsteps, T[3], T[4], T[5], e, ast, fmax)
#             msg = "%s:  %3d %02d:%02d:%02d %15.6f%1s %12.4f\n" % args
#             self.logfile.write(msg)
#
#             self.logfile.flush()
#
#     def dump(self, data):
#         if world.rank == 0 and self.restart is not None:
#             with open(self.restart, 'w') as fd:
#                 write_json(fd, data)
#
#     def load(self):
#         with open(self.restart) as fd:
#             try:
#                 return read_json(fd, always_array=False)
#             except Exception as ex:
#                 msg = ('Could not decode restart file as JSON.  '
#                        f'You may need to delete the restart file {self.restart}')
#                 raise RestartError(msg) from ex
#
#     def set_force_consistent(self):
#         """Automatically sets force_consistent to True if force_consistent
#         energies are supported by calculator; else False."""
#         try:
#             self.atoms.get_potential_energy(force_consistent=True)
#         except PropertyNotImplementedError:
#             self.force_consistent = False
#         else:
#             self.force_consistent = True
