from .linear import LinearMixer
from .mixer import AbstractMixer, SpecialPrecondition
from .pulay import PulayMixer


class Mixer:
    """
    Note :
        This is a api for mixing, and also can works for spin polarization.
    """

    def __new__(cls, scheme=None, **kwargs):
        if not scheme:
            scheme = 'none'
        scheme = scheme.lower()
        if scheme.lower() == 'none':
            return None
        else:
            return super().__new__(cls)

    def __init__(self, scheme=None, nspin=2, **kwargs):
        self.mixers = [
            None,
        ] * nspin
        for i in range(nspin):
            self.mixers[i] = self.get_mixer(scheme=scheme, **kwargs)

    def get_mixer(self, scheme=None, **kwargs):
        if not scheme:
            scheme = 'none'
        scheme = scheme.lower()
        if scheme == 'pulay':
            mixer = PulayMixer(**kwargs)
        elif scheme == 'linear':
            mixer = LinearMixer(**kwargs)
        elif scheme.lower() == 'none':
            mixer = None
        else:
            raise AttributeError("!!!ERROR : NOT support ", scheme)
        return mixer

    def __call__(self, nin, nout, coef=None):
        results = self.compute(nin, nout, coef)
        return results

    def compute(self, nin, nout, coef=None):
        nspin = nin.rank
        if nspin == 1:
            results = self.mixers[0](nin, nout, coef=coef)
        else:
            results = nin * 0.0
            for i in range(nspin):
                results[i] = self.mixers[i](nin[i], nout[i], coef=coef)
        return results

    def restart(self, *args, **kwargs):
        for mixer in self.mixers:
            mixer.restart(*args, **kwargs)
