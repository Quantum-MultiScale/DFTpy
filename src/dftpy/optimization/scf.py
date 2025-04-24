from dftpy.functional import ExternalPotential, Hartree
from dftpy.mpi import sprint
from dftpy.time_data import TimeData, timer


class OESCF:
    """ """

    def __init__(
        self,
        optimization=None,
        evaluator_emb=None,
        guess_rho=None,
        mixer=None,
        maxiter=0,
    ):
        self.optimization = optimization
        self.evaluator_emb = evaluator_emb
        self.rho = guess_rho
        self.mixer = mixer
        self.maxiter = maxiter
        #
        self.iter = 0
        self.residual_norm_prev = 1.0
        if self.maxiter < 1:
            self.maxiter = self.options['maxiter']
        self.extfunctional = ExternalPotential()
        self.error = 1.0

    @property
    def comm(self):
        return self.rho.mp.comm

    @property
    def mp(self):
        return self.rho.mp

    @property
    def options(self):
        return self.optimization.optimization_options

    def optimize_rho(self, **kwargs):
        return self.run(**kwargs)

    @timer('OESCF')
    def run(self, guess_rho=None, **kwargs):
        # -----------------------------------------------------------------------
        if guess_rho is None and self.rho is None:
            raise AttributeError("Must provide a guess density")
        elif guess_rho is not None:
            self.rho = guess_rho
        self.ncharges = self.rho.integral()
        self.econv = self.options['econv'] * 1.0
        self.energyhistory = []
        # -----------------------------------------------------------------------
        for i in range(self.maxiter):
            self.get_density()
            self.mix()
            costtime = TimeData.Time("OESCF")
            energy = self.energyhistory[-1]
            sprint(
                f" OESCF--> iter={self.iter:<5d} conv={self.options['econv']:<.3E} de={self.error:<.3E} energy={energy:<.6E} time={costtime:<.6E}",
                comm=self.comm,
                level=2,
            )
            if self.error < self.econv:
                sprint(
                    "##### OESCF Density Optimization Converged #####", comm=self.comm
                )
                break
        else:
            sprint("!WARN: Not converged, but reached max iterations", comm=self.comm)
        return self.rho

    def get_density(self, norm=None, **kwargs):
        self.iter += 1
        scale = 1e-2
        # -----------------------------------------------------------------------
        if norm is None:
            if self.iter == 1:
                norm = self.ncharges * 1e-1
                self.options['econv'] = 1e10
            elif self.iter == 2:
                norm = self.ncharges * 1e-3
            else:
                norm = self.dp_norm

        if self.comm.size > 1:
            norm = self.comm.bcast(norm, root=0)

        econv = max(norm * scale, 1e-7 * self.ncharges)
        if econv < self.options['econv']:
            self.options['econv'] = econv
        self.iterative(**kwargs)

    def iterative(self, **kwargs):
        #
        self.rho_prev = self.rho.copy()
        func = self.evaluator_emb(self.rho, calcType=('E', 'V'))
        self.extfunctional.v = func.potential
        self.extenergy = func.energy
        self.extenergy -= self.rho.grid.mp.vsum(
            self.extfunctional(self.rho, calcType=('E')).energy
        )
        #
        self.optimization.EnergyEvaluator.funcDict['EMB'] = self.extfunctional
        self.optimization_method = 'CG-HS'
        #
        self.optimization.optimize_rho(guess_rho=self.rho)
        self.rho = self.optimization.rho
        #
        del self.optimization.EnergyEvaluator.funcDict['EMB']
        #
        self.functional = self.optimization.functional
        self.functional.energy += self.extenergy
        #
        energy = self.functional.energy
        if len(self.energyhistory) > 0:
            self.error = abs(energy - self.energyhistory[-1])
        else:
            self.error = abs(energy)
        self.energyhistory.append(energy)

    def mix(self, **kwargs):
        r = self.rho - self.rho_prev
        self.dp_norm = Hartree.compute(r, calcType=('E')).energy
        if self.iter == 1:
            self.dp_norm_prev = self.dp_norm
        else:
            self.dp_norm, self.dp_norm_prev = self.dp_norm_prev, self.dp_norm
            self.dp_norm = (self.dp_norm + self.dp_norm_prev) / 2
        #
        if self.mixer:
            self.rho = self.mixer(self.rho_prev, self.rho, **kwargs)
