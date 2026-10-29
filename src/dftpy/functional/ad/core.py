from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property
from math import pi, prod
from typing import Callable, ClassVar, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

C_TF = 0.3 * (3.0 * pi * pi) ** (2.0 / 3.0)
TKF0 = 2.0 * (3.0 * pi * pi) ** (1.0 / 3.0)
FFT_DIMS = (-3, -2, -1)


@dataclass(frozen=True)
class ParameterSpec:
    """Declared model coefficient; bounds are inclusive."""
    default: float
    description: Optional[str] = 'default'
    lower: Optional[float] = None
    upper: Optional[float] = None

    def validate(self, name: str, value: float) -> None:
        from math import isfinite
        if not isfinite(value):
            raise ValueError(f"{name} must be finite")
        if self.lower is not None and value < self.lower:
            raise ValueError(f"{name} must be >= {self.lower}")
        if self.upper is not None and value > self.upper:
            raise ValueError(f"{name} must be <= {self.upper}")


class ParametricFunctional(nn.Module, ABC):
    """Shared named-parameter contract; nn.Module retains training/state_dict support."""

    @classmethod
    def parameter_specs(cls) -> Mapping[str, ParameterSpec]:
        """Declare every coefficient, its default and domain."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def __init__(self, *, density_floor: Optional[float] = None, **values: float):
        super().__init__()
        specs = self.parameter_specs()
        unknown = set(values) - set(specs)
        if unknown:
            raise ValueError(f"Undeclared parameters: {sorted(unknown)}")
        self.density_floor = density_floor
        self.coefficients = nn.ParameterDict()
        for name, spec in specs.items():
            value = float(values.get(name, spec.default))
            spec.validate(name, value)
            self.coefficients[name] = nn.Parameter(
                torch.tensor(value, dtype=torch.float64)
            )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.parameter_specs())

    def parameter_values(self) -> dict[str, float]:
        return {name: float(self.coefficients[name].detach().cpu()) for name in self.parameter_names}

    def validate_parameters(self) -> None:
        for name, value in self.parameter_values().items():
            self.parameter_specs()[name].validate(name, value)

    def set_parameters(self, **values: float):
        """Use this, an optimizer, or no_grad().copy_(); do not mutate .data."""
        unknown = set(values) - set(self.parameter_specs())
        if unknown:
            raise ValueError(f"Undeclared parameters: {unknown}")
        for name, value in values.items():
            self.parameter_specs()[name].validate(name, float(value))
        with torch.no_grad():
            for name, value in values.items():
                self.coefficients[name].fill_(float(value))
        self.on_parameters_changed(tuple(values))
        return self

    def on_parameters_changed(self, names):
        self.clear_cache()

    def parameter_versions(self, names=None):
        names = self.parameter_names if names is None else names
        return tuple((n, id(self.coefficients[n]), self.coefficients[n]._version,
                      self.coefficients[n].requires_grad) for n in names)

    def clear_cache(self) -> None:
        pass

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self.clear_cache()
        return result


class TensorLRU:
    """Caches ONLY graph-free constants, and never grows past max_bytes."""
    def __init__(self, max_bytes: int):
        if max_bytes < 0:
            raise ValueError("cache_max_bytes cannot be negative")
        self.max_bytes = int(max_bytes)
        self.data = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = 0

    def clear(self):
        self.data.clear()
        self.bytes = 0

    def get_or_build(self, key, builder: Callable[[], Tensor], *, allowed: bool) -> Tensor:
        if not allowed or not self.max_bytes:
            return builder()
        if key in self.data:
            self.hits += 1
            self.data.move_to_end(key)
            return self.data[key]
        self.misses += 1
        value = builder()
        if value.requires_grad or value.grad_fn is not None:
            raise RuntimeError("Attempt to persist an autograd graph in a kernel cache")
        size = value.numel() * value.element_size()
        if size <= self.max_bytes:
            while self.bytes + size > self.max_bytes and self.data:
                _, old = self.data.popitem(last=False)
                self.bytes -= old.numel() * old.element_size()
            self.data[key] = value
            self.bytes += size
        return value


def safe_sqrt_nonnegative(x: Tensor) -> Tensor:
    """Safe at the constant G=0 mode, including in cell derivatives."""
    mask = x > 0
    return torch.where(mask, torch.sqrt(torch.where(mask, x, torch.ones_like(x))), torch.zeros_like(x))


def energy_potential(model: Callable, cell: Tensor, rho: Tensor, *, create_graph=False):
    """Pure-Torch path: preserves parameter gradients"""
    n = rho if rho.requires_grad else rho.detach().requires_grad_(True)
    with torch.enable_grad():
        e = model(cell, n)
        dv = torch.linalg.det(cell).abs() / n.numel()
        v = torch.autograd.grad(e, n, create_graph=create_graph)[0] / dv
    return e, v


def hessian_vector_product(model: Callable, cell: Tensor, rho: Tensor, direction: Tensor):
    """Returns D[v_T(rho)] direction, not a dense Hessian; fixed cell."""
    if rho.shape != direction.shape:
        raise ValueError("direction and rho must have equal shapes")
    n = rho.detach().requires_grad_(True)
    _, v = energy_potential(model, cell, n, create_graph=True)
    return torch.autograd.grad((v * direction).sum(), n)[0]


def strain_stress(model: Callable, cell: Tensor, rho: Tensor, *, create_graph=False):
    """sigma = dE/dstrain / volume, at fixed electron number, fractional mesh."""
    with torch.enable_grad():
        strain = torch.zeros((3, 3), dtype=cell.dtype, device=cell.device, requires_grad=True)
        h = cell @ (torch.eye(3, dtype=cell.dtype, device=cell.device) + strain).T
        v0 = torch.linalg.det(cell).abs()
        n = rho * v0 / torch.linalg.det(h).abs()
        e = model(h, n)
        return torch.autograd.grad(e, strain, create_graph=create_graph)[0] / v0

@dataclass
class GridContext:
    cell: Tensor
    shape: tuple[int, int, int]
    key: Optional[tuple] = None

    @cached_property
    def volume(self) -> Tensor:
        return torch.linalg.det(self.cell).abs()

    @property
    def size(self) -> int:
        return prod(self.shape)

    @property
    def dV(self) -> Tensor:
        return self.volume / self.size

    @cached_property
    def g(self) -> Tensor:
        b = 2.0 * pi * torch.linalg.inv(self.cell).T
        freqs = []
        for dim, size in enumerate(self.shape):
            fn = torch.fft.rfftfreq if dim == 2 else torch.fft.fftfreq
            f = fn(size, dtype=self.cell.dtype, device=self.cell.device) * size
            if dim < 2 and size % 2 == 0:
                f[size // 2] = abs(f[size // 2])
            freqs.append(f)
        indices = torch.stack(torch.meshgrid(*freqs, indexing="ij"))
        return torch.einsum("i...,ij->j...", indices, b)

    @cached_property
    def g2(self) -> Tensor:
        return self.g.square().sum(0)

    @cached_property
    def q(self) -> Tensor:
        return safe_sqrt_nonnegative(self.g2)

    @cached_property
    def rfft_weights(self) -> Tensor:
        w = torch.full((self.shape[-1] // 2 + 1,), 2.0,
                       device=self.cell.device, dtype=self.cell.dtype)
        w[0] = 1.0
        if self.shape[-1] % 2 == 0:
            w[-1] = 1.0
        return w

    def fft(self, field: Tensor) -> Tensor:
        return torch.fft.rfftn(field, dim=FFT_DIMS, norm="backward")

    def ifft(self, field: Tensor) -> Tensor:
        return torch.fft.irfftn(field, s=self.shape, dim=FFT_DIMS, norm="backward")

    def gradient(self, field: Tensor) -> Tensor:
        # One forward FFT, one batched inverse transform for all 3 components.
        return self.ifft(1j * self.g * self.fft(field).unsqueeze(0))

    def parseval(self, a: Tensor, b: Tensor, kernel: Tensor) -> Tensor:
        return self.dV / self.size * (
            self.rfft_weights * kernel * (a.conj() * b).real
        ).sum()

    def tf(self, rho: Tensor) -> Tensor:
        return C_TF * self.dV * rho.pow(5.0 / 3.0).sum()

    def vw(self, rho: Tensor) -> Tensor:
        a = self.fft(rho.sqrt())
        return 0.5 * self.parseval(a, a, self.g2)


class SemilocalFunctionalABC(ParametricFunctional):
    """F(s_red**2; coefficients; q), pointwise and dimensionless.

    Implement name, level, parameter_specs() and enhancement().
    NO spatial coupling, FFT, reductions across grid points, or detached inputs
    are allowed inside enhancement(). This pointwise contract makes grad(F.sum())
    the required local derivative. q=None for GGA and a tensor for MGGA.
    """
    @property
    @abstractmethod
    def level(self) -> str:
        """Return exactly 'GGA' or 'MGGA'."""
        raise NotImplementedError

    @abstractmethod
    def enhancement(self, s2: Tensor, coefficients: Mapping[str, Tensor],
                    q: Optional[Tensor], **kwargs) -> Tensor:
        raise NotImplementedError

    def forward(self, s2: Tensor, coefficients: Optional[Mapping[str, Tensor]] = None,
                q: Optional[Tensor] = None) -> Tensor:
        if self.level not in {"GGA", "MGGA"}:
            raise ValueError("level must be GGA or MGGA")
        if self.level == "MGGA" and q is None:
            raise ValueError("This MGGA requires q")
        return self.enhancement(s2, self.coefficients if coefficients is None else coefficients, q=q)


class EnergyFunctionalABC(ParametricFunctional, ABC):
    """
    Implement name, parameter_specs() and energy(rho, context). The base owns
    validation, geometry reuse and graph-safe caches. energy MUST return one
    scalar total energy in hartree. Never detach density-dependent quantities.
    """
    def __init__(self, *, density_floor: Optional[float] = None,
                 cache_max_bytes: int = 64 * 1024**2, validate_inputs: bool = True,
                 **values):
        if density_floor is not None and density_floor <= 0:
            raise ValueError("density_floor must be positive or None")
        super().__init__(**values)
        self.density_floor = density_floor
        self.validate_inputs = bool(validate_inputs)
        self.kernel_cache = TensorLRU(cache_max_bytes)
        self._grid = None
        self._parameter_stamp = None
        self._kernel_parameter_stamp = None

    @abstractmethod
    def energy(self, rho: Tensor, context: GridContext) -> Tensor:
        raise NotImplementedError

    def clear_cache(self):
        if hasattr(self, "kernel_cache"):
            self.kernel_cache.clear()
        self._grid = None
        self._parameter_stamp = None
        self._kernel_parameter_stamp = None

    def kernel_parameter_names(self):
        """Override when only a subset of coefficients enters cached kernels."""
        return self.parameter_names

    def on_parameters_changed(self, names):
        self._parameter_stamp = None
        if set(names) & set(self.kernel_parameter_names()):
            self.kernel_cache.clear()
            self._kernel_parameter_stamp = None

    def prepare(self, cell: Tensor, shape: Sequence[int]) -> GridContext:
        """Fixed geometry is reusable; differentiable geometry NEVER is."""
        shape = tuple(map(int, shape))
        stamp = self.parameter_versions()
        if stamp != self._parameter_stamp:
            self.validate_parameters()
            self._parameter_stamp = stamp
        kernel_stamp = self.parameter_versions(self.kernel_parameter_names())
        if kernel_stamp != self._kernel_parameter_stamp:
            self.kernel_cache.clear()
            self._kernel_parameter_stamp = kernel_stamp
        if cell.requires_grad:
            return GridContext(cell, shape)
        # Only nine cell elements cross the device boundary; not the density.
        key = (shape, str(cell.dtype), str(cell.device),
               tuple(cell.detach().cpu().reshape(-1).tolist()))
        if self._grid is None or self._grid.key != key:
            self.kernel_cache.clear()
            self._grid = GridContext(cell.detach().clone(), shape, key)
        return self._grid

    def forward(self, cell: Tensor, rho: Tensor) -> Tensor:
        if cell.shape != (3, 3) or rho.ndim != 3:
            raise ValueError("Expected cell (3,3) and scalar density (nx,ny,nz)")
        if cell.dtype != torch.float64 or rho.dtype != torch.float64:
            raise TypeError("Use float64 cell and density for this implementation")
        if cell.device != rho.device:
            raise ValueError("cell and density must be on the same device")
        if any(p.device != rho.device for p in self.parameters()):
            raise ValueError("Move model to the density device with model.to(device)")
        if self.validate_inputs:
            if not bool(torch.isfinite(cell).all()) or not bool(torch.isfinite(rho).all()):
                raise ValueError("Non-finite cell or density")
            if float(torch.linalg.det(cell).detach().abs()) == 0:
                raise ValueError("Singular cell")
            if self.density_floor is None and bool((rho <= 0).any()):
                raise ValueError("rho must be strictly positive; opt in to density_floor if appropriate")
        # This is a declared regularization, differentiated as written.
        effective = rho if self.density_floor is None else rho.clamp_min(self.density_floor)
        result = self.energy(effective, self.prepare(cell, rho.shape))
        if not isinstance(result, Tensor) or result.numel() != 1:
            raise TypeError("energy() must return one scalar torch.Tensor")
        return result.reshape(())

    def cache_info(self):
        c = self.kernel_cache
        return {"entries": len(c.data), "bytes": c.bytes,
                "max_bytes": c.max_bytes, "hits": c.hits, "misses": c.misses}
