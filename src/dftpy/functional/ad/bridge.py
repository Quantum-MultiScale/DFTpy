from __future__ import annotations
from typing import Optional, Callable
import numpy as np
import torch
from torch import nn

try:
    from dftpy.field import DirectField
    from dftpy.functional.abstract_functional import AbstractFunctional
    from dftpy.functional.functional_output import FunctionalOutput
except ImportError as exc:
    raise exc

from .core import SemilocalFunctionalABC, EnergyFunctionalABC, GridContext, strain_stress

def register_ad_gga(model: SemilocalFunctionalABC, *, name: Optional[str] = None,
                       device="cpu", overwrite=False) -> str:
    """Register an ABC INSTANCE, not an arbitrary pointwise callback.

    The DFTpy registry remains {'function': callback, 'params': list}, so existing
    GGA/MGGA/gradient/divergence code and caller-provided params remain usable.
    Registered defaults are a SNAPSHOT. Re-register after changing the model's
    default coefficients, or supply KEDF(..., params=[...]) explicitly.
    """
    from dftpy.constants import TKF0
    from dftpy.functional.kedf.gga import GGA_KEDF_list, MGGA_KEDF_list
    if not isinstance(model, SemilocalFunctionalABC):
        raise TypeError("Pass an instance of a SemilocalFunctionalABC subclass, e.g. LKT()")
    model = model.to(device=device, dtype=torch.float64)
    model.validate_parameters()
    level = model.level
    if level not in {"GGA", "MGGA"}:
        raise ValueError("Model.level must be GGA or MGGA")
    registry = MGGA_KEDF_list if level == "MGGA" else GGA_KEDF_list
    label = (name or model.name).upper()
    if label in registry and not overwrite:
        raise ValueError(f"{label} is already registered")
    names = model.parameter_names
    defaults = [model.parameter_values()[n] for n in names]
    # Cache only fixed scalar parameter tensors; never a density graph.
    last_values, scalar_tensors = None, None

    def engine(s, calcType, params, q=None, **kwargs):
        nonlocal last_values, scalar_tensors
        values = tuple(float(v) for v in params)
        if len(values) != len(names):
            raise ValueError(f"Expected params in order {names}")
        if values != last_values:
            for n, v in zip(names, values):
                model.parameter_specs()[n].validate(n, v)
            scalar_tensors = {n: torch.tensor(v, dtype=torch.float64, device=device)
                              for n, v in zip(names, values)}
            last_values = values
        need = 'V' in calcType or 'S' in calcType
        u_array = (np.asarray(s, dtype=np.float64)/TKF0)**2
        u = torch.as_tensor(np.ascontiguousarray(u_array), dtype=torch.float64,
                            device=device).requires_grad_(need)
        qt = None
        if level == "MGGA":
            if q is None:
                raise ValueError("MGGA requires DFTpy's q field")
            qt = torch.as_tensor(np.array(q, dtype=np.float64, order='C', copy=True),
                                 device=device).requires_grad_(need)
        with torch.set_grad_enabled(need):
            f = model(u, qt, scalar_tensors)
            if not isinstance(f, torch.Tensor) or f.shape != u.shape:
                raise ValueError("enhancement must be pointwise and return the same shape as s2")
            if need:
                inputs = (u, qt) if qt is not None else (u,)
                grads = (torch.autograd.grad(f.sum(), inputs, allow_unused=True)
                         if f.requires_grad else (None,)*len(inputs))
        def field(x):
            return DirectField(grid=s.grid, data=x.detach().cpu().numpy())
        out = {"F": field(f)}
        if need:
            fu = grads[0] if grads[0] is not None else torch.zeros_like(u)
            out["dFds2"] = field((2/TKF0**2)*fu)
            if qt is not None:
                fq = grads[1] if grads[1] is not None else torch.zeros_like(qt)
                out["dFdq"] = field(fq)
        return out
    registry[label] = {"function": engine, "params": defaults}
    return f"{level}_{label}"


class ProfessADAdapter(EnergyFunctionalABC):
    """Wrap a legacy (cell_rows, density)->energy callable without changing it.

    No native kernel-speed claim is made for this compatibility path. Explicit
    .kernel tensors on current PROFESS-AD model objects are moved along with
    registered parameters; other custom callable state remains its author's job.
    """
    @classmethod
    def parameter_specs(cls):
        return {}

    @property
    def name(self):
        return self._name

    def __init__(self, functional, *, name=None):
        super().__init__(validate_inputs=False, cache_max_bytes=0)
        owner=getattr(functional, "__self__", None)
        if isinstance(owner, nn.Module):
            functional=owner
        if not callable(functional):
            raise TypeError("Expected a model or callable (cell, density)->energy")
        self.source=functional
        self._name=name or "PROFESS_"+getattr(functional, '__name__', type(functional).__name__)
        self._legacy_device=None

    def energy(self, rho, context):
        target=(rho.device, rho.dtype)
        if self._legacy_device != target:
            obj=self.source
            if isinstance(obj, nn.Module):
                obj.to(device=rho.device, dtype=rho.dtype)
                if hasattr(obj, 'set_device'):
                    obj.set_device(rho.device)
                if isinstance(getattr(obj, 'kernel', None), torch.Tensor):
                    obj.kernel=obj.kernel.to(device=rho.device, dtype=rho.dtype)
            self._legacy_device=target
        return self.source(context.cell, rho).reshape(())


class ADKEDF(AbstractFunctional):
    """Accept an EnergyFunctionalABC instance OR a PROFESS-AD callable.

    Serial CPU/CUDA, real rank-1 density. E/V and stress; no MPI/spin/D/V2.
    Returned DFTpy data are detached.
    """
    def __init__(self, functional: EnergyFunctionalABC | Callable, *,
                 name=None, device="cpu", copy_density=True):
        self.type="KEDF"
        self.model=(functional if isinstance(functional, EnergyFunctionalABC)
                    else ProfessADAdapter(functional, name=name))
        self.device=torch.device(device)
        self.model.to(device=self.device, dtype=torch.float64)
        self.name=name or self.model.name
        self.copy_density=bool(copy_density)

    def _tensors(self, density, need_potential=False):
        if density.mp.is_mpi:
            raise NotImplementedError("Full-energy AD requires a global serial grid; MPI is not implemented")
        arr=np.asarray(density)
        if arr.ndim != 3 or not np.issubdtype(arr.dtype, np.floating):
            raise TypeError("Only 3D real spin-unpolarized density is supported")
        if self.copy_density:
            arr=np.array(arr, dtype=np.float64, order='C', copy=True)
        else:
            arr=np.ascontiguousarray(arr, dtype=np.float64)
        n=torch.as_tensor(arr, device=self.device).requires_grad_(need_potential)
        h=torch.as_tensor(np.array(density.grid.lattice, dtype=np.float64, copy=True),
                          device=self.device)
        return h,n

    def compute(self, density, calcType={'E','V'}, split=False, **kwargs):
        if not set(calcType) or set(calcType)-{'E','V'}:
            raise NotImplementedError("Only nonempty calcType subsets of {'E','V'} are supported")
        need='V' in calcType
        h,n=self._tensors(density, need)
        with torch.set_grad_enabled(need):
            e=self.model(h,n)
            grad=torch.autograd.grad(e,n)[0] if need else None
        out=FunctionalOutput(name=self.name)
        if 'E' in calcType:
            out.energy=float(e.detach().cpu())
        if need:
            dv=torch.linalg.det(h).abs()/n.numel()
            out.potential=DirectField(grid=density.grid, data=(grad/dv).detach().cpu().numpy())
        return {self.name:out} if split else out

    def stress(self, density, split=False, **kwargs):
        h,n=self._tensors(density)
        result=strain_stress(self.model,h,n).detach().cpu().numpy()
        return {self.name:result} if split else result

    def clear_cache(self):
        self.model.clear_cache()
