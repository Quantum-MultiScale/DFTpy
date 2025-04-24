import warnings

import numpy as np
from scipy import ndimage, signal

from dftpy.constants import environ
from dftpy.grid import DirectGrid, ReciprocalGrid
from dftpy.math_utils import PYfft, PYifft
from dftpy.time_data import timer


class BaseField(np.ndarray):
    r"""
    Extended numpy array representing a field on a grid
    (Cell (lattice) plus discretization)

    Attributes
    ----------
    self : np.ndarray
        the values of the field

    grid : Grid
        Represent the domain of the function

    span : number of directions for which we have more than 1 point
            e.g.: for np.zeros((5,5,1)) -> ndim = 3, span = 2

    rank : rank of the field. default = 1, i.e. scalar field

    memo : optional string to label the field

    """

    def __new__(
        cls,
        grid,
        memo="",
        rank=1,
        data=None,
        order='C',
        cplx=False,
        griddata_F=None,
        griddata_C=None,
        griddata_3d=None,
    ):
        # -----------------------------------------------------------------------
        if griddata_F is not None or griddata_C is not None or griddata_3d is not None:
            # warnings.warn(FutureWarning("'griddata_*' are deprecated; please use 'data' and 'order'"))
            if griddata_F is not None:
                data = griddata_F
                order = 'F'
            elif griddata_C is not None:
                data = griddata_C
                order = 'C'
            elif griddata_3d is not None:
                data = griddata_3d
                order = 'C'
        # -----------------------------------------------------------------------
        if data is not None:
            if isinstance(data, list):
                rank = len(data)

        if rank == 1:
            nr = grid.nr
        else:
            nr = rank, *grid.nr

        if data is None:
            if cplx:
                input_values = np.zeros(nr, dtype='complex128', order=order)
            else:
                input_values = np.zeros(nr, order=order)
        else:
            input_values = np.asarray(data)
            input_values = np.reshape(input_values, nr, order=order)

        obj = np.asarray(input_values)
        if order == 'C' and not obj.flags['C_CONTIGUOUS']:
            obj = np.ascontiguousarray(obj)
        elif order == 'F' and not obj.flags['F_CONTIGUOUS']:
            obj = np.asfortranarray(obj)
        obj = obj.view(cls)
        # add the new attribute to the created instance
        obj.grid = grid
        obj.span = (grid.nr > 1).sum()
        obj.memo = str(memo)
        obj.mp = obj.grid.mp
        # Finally, we must return the newly created object:
        return obj

    @property
    def rank(self):
        if self.ndim == 4:
            return self.shape[0]
        else:
            return 1

    def __array_finalize__(self, obj):
        # Restore attributes when we are taking a slice
        if obj is None:
            return
        self.grid = getattr(obj, "grid", None)
        self.span = getattr(obj, "span", None)
        self.memo = getattr(obj, "memo", None)
        self.mp = getattr(obj, "mp", None)

    def __array_wrap__(self, obj, context=None, return_scalar=False):
        """wrap it up"""
        if obj.ndim < 3 or return_scalar:
            # This is only return numpy array not field
            return obj
        b = np.ndarray.__array_wrap__(self, obj, context, return_scalar)
        return b

    def dot(self, obj):
        """Returns the dot product of vector fields self and obj"""
        if np.shape(self) != np.shape(obj):
            raise ValueError("Shape incompatible")  # to be specified

        prod = self.mp.einsum("ijkl,ijkl->jkl", self, obj, optimize=True)
        prod = np.expand_dims(prod, axis=3)

        return type(self)(self.grid, rank=1, griddata_3d=prod)

    def project(self, obj):
        """Returns the field that self projects on obj"""
        return obj * (np.conj(obj.normalize()) * self).integral()

    def norm(self):
        return np.sqrt((np.real(np.conj(self) * self)).integral())

    def normalize(self, N=1.0):
        """Normalize the field to N"""
        return self / self.norm() * np.sqrt(N)

    def asum(self):
        return self.mp.asum(self)

    def amean(self):
        return self.mp.amean(self)

    def amax(self):
        return self.mp.amax(self)

    def amin(self):
        return self.mp.amin(self)

    @property
    def cell(self):
        return self.grid.cell


class DirectField(BaseField):
    spl_order = 3

    def __new__(cls, grid, memo="", rank=1, data=None, order='C', cplx=False, **kwargs):
        if hasattr(grid, 'get_direct'):
            grid = grid.get_direct()
        if not isinstance(grid, DirectGrid):
            raise TypeError("the grid argument is not an instance of DirectGrid")
        obj = super().__new__(
            cls, grid, memo="", rank=rank, data=data, order=order, cplx=cplx, **kwargs
        )
        #
        obj.init_options = locals()
        for k in ['__class__', 'cls', 'obj', 'kwargs', 'grid', 'data']:
            obj.init_options.pop(k, None)
        # obj.init_options.update(kwargs)
        #
        obj._N = None
        obj.spl_coeffs = None
        obj._cplx = cplx
        if obj.mp.is_mpi:
            from dftpy.mpi.mp_mpi4py import mpi_fft

            obj.fft_object = mpi_fft(obj.grid)
        elif not obj._cplx and np.all(
            obj.grid.nr == obj.grid.nrG
        ):  # Can only use numpy.fft
            obj.fft_object = np.fft.fftn
        elif environ["FFTLIB"] == "pyfftw":
            obj.fft_object = PYfft(obj.grid, obj._cplx)
        elif environ["FFTLIB"] == "numpy":
            if obj._cplx:
                obj.fft_object = np.fft.fftn
            else:
                obj.fft_object = np.fft.rfftn
        else:
            obj.fft_object = np.fft.fftn
        return obj

    def __array_finalize__(self, obj):
        # Restore attributes when we are taking a slice
        super().__array_finalize__(obj)
        if obj is None:
            return
        if isinstance(obj, (DirectField)):
            self.fft_object = getattr(obj, 'fft_object', None)
            self.cplx = getattr(obj, 'cplx', None)
        self.spl_coeffs = None
        self._N = None

    def integral(self, gather=True):
        """Returns the integral of self"""
        mp = self.mp if gather else np
        if self.rank == 1:
            return mp.einsum("ijk->", self) * self.grid.dV
        else:
            return mp.einsum("ijkl->i", self) * self.grid.dV

    def _calc_spline(self):
        padded_values = np.pad(self, ((self.spl_order,)), mode="wrap")
        self.spl_coeffs = ndimage.spline_filter(padded_values, order=self.spl_order)
        return

    def window_functions(self, window="hann"):
        """Only support for serial"""
        nr = self.nr
        Wx = signal.get_window(window, nr[0])
        Wy = signal.get_window(window, nr[1])
        Wz = signal.get_window(window, nr[2])
        Wxyz = self.mp.einsum("i, j, k -> ijk", Wx, Wy, Wz)
        return Wxyz

    def numerically_smooth_gradient(self, ipol=None):
        r"""
        DEPRECATED!!!
        """
        sq_self = np.sqrt(np.abs(self))
        grad = sq_self.standard_gradient(ipol)
        if ipol is None:
            final = np.empty(np.shape(grad), dtype=float)
            dim = np.shape(np.shape(sq_self))[0]
            if dim == 4:
                a = sq_self[0]
            if dim == 3:
                a = sq_self[:]
            for ipol in np.arange(grad.rank):
                final[ipol] = 2.0 * a * grad[ipol]
            return DirectField(grid=grad.grid, rank=grad.rank, griddata_3d=final)
        else:
            if grad.rank != 1:
                raise ValueError("Gradient rank incompatible with shape")
            return DirectField(
                grid=grad.grid,
                rank=1,
                griddata_3d=2.0 * sq_self * np.reshape(grad, np.shape(sq_self)),
            )

    def super_smooth_gradient(self, ipol=None, force_real=True, sigma=0.0):
        r"""
        Gradient of field with Gaussian convolution smoothing
         - args:
            ipol: optional, direction of gradient (1->x, 2->y, 3->z)
            force_real: default True, returns a real DirectField
            sigma: smoothing parameter in Bohrs, default=0.0
         - returns 3d gradient unless ipol is specified
        """
        reciprocal_self = self.fft()
        imag = 0 + 1j
        nr = 3, *reciprocal_self.grid.nr
        grad_g = np.empty(nr, dtype='complex128')
        if ipol is None:
            # FFT(\grad A) = i \vec(G) * FFT(A)
            grad_g = (
                reciprocal_self.grid.g
                * (reciprocal_self * imag)
                * np.exp(-reciprocal_self.grid.gg * sigma**2 / 2.0)
            )
            grad_g = ReciprocalField(
                grid=self.grid.get_reciprocal(), rank=3, griddata_3d=grad_g
            )
            grad = grad_g.ifft(force_real=force_real)
            if grad.rank != np.shape(grad)[0]:
                raise ValueError(
                    "Supersmooth Gradient: Gradient rank incompatible with shape"
                )
            return grad
        elif ipol > 3:
            raise ValueError("Supersmooth Gradient: ipol can not large than 3")
        else:
            i = ipol - 1
            grad_g = (
                reciprocal_self.grid.g[i]
                * (reciprocal_self * imag)
                * np.exp(-reciprocal_self.grid.gg * sigma**2 / 2.0)
            )
            grad_g = ReciprocalField(
                grid=self.grid.get_reciprocal(), rank=1, griddata_3d=grad_g
            )
            grad = grad_g.ifft(force_real=force_real)
            return grad

    def standard_gradient(self, ipol=None, force_real=True):
        reciprocal_self = self.fft()
        imag = 0 + 1j
        if ipol is None:
            # FFT(\grad A) = i \vec(G) * FFT(A)
            grad_g = reciprocal_self.grid.g * (reciprocal_self * imag)
            grad_g = ReciprocalField(
                grid=self.grid.get_reciprocal(), rank=3, griddata_3d=grad_g
            )
        elif ipol > 3:
            raise ValueError("Standard Gradient: ipol can not large than 3")
        else:
            i = ipol - 1
            grad_g = reciprocal_self.grid.g[i] * (reciprocal_self * imag)
            grad_g = ReciprocalField(
                grid=self.grid.get_reciprocal(), rank=1, griddata_3d=grad_g
            )
        grad = grad_g.ifft(force_real=force_real)
        return grad

    def divergence(self, flag="supersmooth", force_real=True, sigma=0.0):
        r"""
        Divergence of field
         - sigma: smoothing parameter in Bohrs, dafault 0.0.
        """
        if self.rank != 3:
            raise ValueError("Divergence: Rank incompatible ", self.rank)
        div = self[0].gradient(flag=flag, ipol=1, force_real=force_real, sigma=sigma)
        div += self[1].gradient(flag=flag, ipol=2, force_real=force_real, sigma=sigma)
        div += self[2].gradient(flag=flag, ipol=3, force_real=force_real, sigma=sigma)
        return div

    def hessian(self, flag="smooth", force_real=True):
        '''
        Calculate Hessian of Grid value.
        Return xx, yy, zz, xy, xz, yz
        '''
        if self.rank != 1:
            raise ValueError("Hessian: Rank incompatible ", self.rank)
        # 1st Order Grad
        grad_x = self.gradient(flag=flag, ipol=1, force_real=force_real)
        grad_y = self.gradient(flag=flag, ipol=2, force_real=force_real)
        grad_z = self.gradient(flag=flag, ipol=3, force_real=force_real)
        # 2nd order Hessian
        hess_xx = grad_x.gradient(flag=flag, ipol=1, force_real=force_real)
        hess_yy = grad_y.gradient(flag=flag, ipol=2, force_real=force_real)
        hess_zz = grad_z.gradient(flag=flag, ipol=3, force_real=force_real)

        hess_xy = grad_x.gradient(flag=flag, ipol=2, force_real=force_real)
        hess_xz = grad_x.gradient(flag=flag, ipol=3, force_real=force_real)
        hess_yz = grad_y.gradient(flag=flag, ipol=3, force_real=force_real)
        hess = np.stack([hess_xx, hess_yy, hess_zz, hess_xy, hess_xz, hess_yz])

        return DirectField(grid=self.grid, rank=6, griddata_3d=hess)

    def gradient(self, flag="standard", ipol=None, force_real=True, sigma=0.025):
        values = []
        for i in range(self.rank):
            if self.rank == 1:
                data = self
            else:
                data = self[i]
            val = data._gradient(
                flag=flag, ipol=ipol, force_real=force_real, sigma=sigma
            )
            values.append(val)
        if self.rank == 1:
            values = values[0]
        return values

    def _gradient(self, flag="standard", ipol=None, force_real=True, sigma=0.025):
        if flag == "standard":
            return self.standard_gradient(ipol, force_real)
        elif flag == "smooth":
            if not force_real:
                raise Exception("Smooth gradient is not implemented for complex fields")
            return self.numerically_smooth_gradient(ipol)
        elif flag == "supersmooth":
            return self.super_smooth_gradient(ipol, force_real, sigma=sigma)
        elif flag == "numerical":
            return self.numerical_gradient()
        else:
            raise Exception("Incorrect flag")

    def numerical_gradient(self):
        r"""
        Numerical gradient based on numpy.gradient with edge order=2
        """
        if self.rank == 1:
            return DirectField(
                rank=3,
                grid=self.grid,
                data=np.array(np.gradient(self, np.cbrt(self.grid.dV), edge_order=2)),
            )
        else:
            raise Exception("Numerical gradient only available for scalar fields")

    def hessian(self, flag="supersmooth", force_real=True, sigma=0.025):
        r"""
        Calculate Hessian of Grid value.
        Return field of rank=6, values: xx, yy, zz, xy, xz, yz
        """
        if np.iscomplex(self[0, 0, 0]):
            raise AttributeError("Hessian only implemented for real fields")
        if self.rank != 1:
            raise ValueError("Hessian: Rank incompatible ", self.rank)
        # 1st Order Grad
        grad_x = self.gradient(flag=flag, ipol=1, force_real=force_real, sigma=sigma)
        grad_y = self.gradient(flag=flag, ipol=2, force_real=force_real, sigma=sigma)
        grad_z = self.gradient(flag=flag, ipol=3, force_real=force_real, sigma=sigma)
        # 2nd order Hessian
        hess_xx = grad_x.gradient(flag=flag, ipol=1, force_real=force_real, sigma=sigma)
        hess_yy = grad_y.gradient(flag=flag, ipol=2, force_real=force_real, sigma=sigma)
        hess_zz = grad_z.gradient(flag=flag, ipol=3, force_real=force_real, sigma=sigma)

        hess_xy = grad_x.gradient(flag=flag, ipol=2, force_real=force_real, sigma=sigma)
        hess_xz = grad_x.gradient(flag=flag, ipol=3, force_real=force_real, sigma=sigma)
        hess_yz = grad_y.gradient(flag=flag, ipol=3, force_real=force_real, sigma=sigma)
        hess = np.stack([hess_xx, hess_yy, hess_zz, hess_xy, hess_xz, hess_yz])

        return DirectField(grid=self.grid, rank=6, griddata_3d=hess)

    def laplacian(self, check_real=False, force_real=False, sigma=0.025):
        reciprocal_self = self.fft()
        gg = reciprocal_self.grid.gg
        if sigma is None or sigma == 0:
            reciprocal_self = -reciprocal_self.grid.gg * reciprocal_self
        else:
            reciprocal_self = -gg * reciprocal_self * np.exp(-gg * sigma**2 / 2.0)
        return reciprocal_self.ifft(check_real=check_real, force_real=force_real)

    def sigma(self, flag="supersmooth", sigma_gradient=None, gradient=None):
        r"""
        \sigma(r) = |\grad rho(r)|^2
        """
        if gradient is None:
            if self.rank > 1:
                gradient = []
                for i in range(0, self.rank):
                    gradrho = self[i].gradient(flag=flag, sigma=sigma_gradient)
                    gradient.append(gradrho)
            else:
                gradient = self.gradient(flag=flag, sigma=sigma_gradient)

        if self.rank > 1:
            sigma = []
            for i in range(0, self.rank):
                for j in range(i, self.rank):
                    s = self.mp.einsum("lijk,lijk->ijk", gradient[i], gradient[j])
                    sigma.append(s)
            rank = (self.rank * (self.rank + 1)) // 2
        else:
            sigma = self.mp.einsum("lijk,lijk->ijk", gradient, gradient)
            rank = 1
        return DirectField(grid=self.grid, rank=rank, griddata_3d=sigma)

    @timer('FFT')
    def fft(self):
        r"""Implements the Discrete Fourier Transform
        Tips : If you use pyfft to perform fft, you should copy the input_array, because
        the input_array may be overwritten.
        """
        reciprocal_grid = self.grid.get_reciprocal()
        if self.rank == 1:
            griddata_3d = self.fft_object(self) * self.grid.dV
        else:
            nr = self.rank, *reciprocal_grid.nr
            griddata_3d = np.empty(nr, dtype='complex128')
            for i in range(self.rank):
                griddata_3d[i] = self.fft_object(self[i]) * self.grid.dV
        fft_data = ReciprocalField(
            grid=reciprocal_grid,
            memo=self.memo,
            rank=self.rank,
            griddata_3d=griddata_3d,
            cplx=self.cplx,
        )
        return fft_data

    def get_value_at_points(self, points):
        """
        Only support for serial.
        points is in crystal coordinates
        """
        if self.spl_coeffs is None:
            self._calc_spline()
        for ipol in range(3):
            # restrict crystal coordinates to [0,1)
            points[:, ipol] = (points[:, ipol] % 1) * self.grid.nr[
                ipol
            ] + self.spl_order
        values = ndimage.map_coordinates(
            self.spl_coeffs, [points[:, 0], points[:, 1], points[:, 2]], mode="wrap"
        )
        return values

    def get_values_flatarray(self, pad=0, order="F"):
        """
        Only support for serial.
        """
        if pad > 0:
            nr0 = self.shape
            p = []
            for i in nr0:
                if i > 1:
                    p.append([0, pad])
                else:
                    p.append([0, 0])
            vals = np.pad(self, p, mode="wrap")
        else:
            vals = np.asarray(self)
        nr = vals.shape
        nnr = 1
        for n in nr:
            nnr *= n
        return np.reshape(vals, nnr, order=order)

    def get_3dinterpolation(self, nr_new):
        """
        Only support for serial.
        Interpolates the values of the function on a cell with a different number
        of points, and returns a new Grid_Function_Base object.
        """
        if self.rank > 1:
            raise Exception("get_3dinterpolation is only implemented for scalar fields")

        if self.spl_coeffs is None:
            self._calc_spline()
        x = (
            np.linspace(0, 1, nr_new[0], endpoint=False) * self.grid.nr[0]
            + self.spl_order
        )
        y = (
            np.linspace(0, 1, nr_new[1], endpoint=False) * self.grid.nr[1]
            + self.spl_order
        )
        z = (
            np.linspace(0, 1, nr_new[2], endpoint=False) * self.grid.nr[2]
            + self.spl_order
        )
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        new_values = ndimage.map_coordinates(self.spl_coeffs, [X, Y, Z], mode="wrap")
        new_lattice = self.grid.lattice
        new_grid = DirectGrid(new_lattice, nr_new)
        return DirectField(new_grid, self.memo, griddata_3d=new_values)

    def get_3dinterpolation_map(self, nr_new):
        """
        Only support for serial.
        Interpolates the values of the function on a cell with a different number
        of points, and returns a new Grid_Function_Base object.
        """
        if self.rank > 1:
            raise Exception("get_3dinterpolation is only implemented for scalar fields")

        x = np.linspace(0, 1, nr_new[0], endpoint=False) * self.grid.nr[0]
        y = np.linspace(0, 1, nr_new[1], endpoint=False) * self.grid.nr[1]
        z = np.linspace(0, 1, nr_new[2], endpoint=False) * self.grid.nr[2]
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        new_values = ndimage.map_coordinates(self[0], (X, Y, Z), mode="wrap")
        new_lattice = self.grid.lattice
        new_grid = DirectGrid(new_lattice, nr_new)
        return DirectField(new_grid, self.memo, griddata_3d=new_values)

    def get_cut(
        self,
        r0,
        r1=None,
        r2=None,
        origin=[0, 0, 0],
        center=None,
        nr=50,
        basis='crystal',
    ):
        """
        Only support for serial.
        general routine to get the arbitrary cuts of a Grid_Function_Base object in 1,2,
        or 3 dimensions. spline interpolation will be used.
            r0 = first vector (always required)
            r1 = second vector (required for 2D and 3D cuts)
            r2 = third vector (required for 3D cuts)
            origin = origin of the cut (don't specify center)
            center = center of the cut (don't specify origin)
            nr[i] = number points to discretize each direction ; i = 0,1,2
        r0, r1, r2, origin, center are instances of Coord
        """

        if self.rank > 1:
            raise Exception("get_cut is only implemented for scalar fields")

        span = 1

        do_center = False
        if origin is None and center is None:
            raise AttributeError("Specify either origin or center")
        elif origin is not None and center is not None:
            warnings.warn(
                "Specified both origin and center, center will be ignored",
                DeprecationWarning,
            )
        elif center is not None:
            do_center = True

        if r0 is not None:
            r0 = np.asarray(r0)
        if r1 is not None:
            r1 = np.asarray(r1)
        if r2 is not None:
            r2 = np.asarray(r2)
        if origin is not None:
            origin = np.asarray(origin)
        if center is not None:
            center = np.asarray(center)

        if basis.lower() != 'crystal':
            if r0 is not None:
                r0 = self.grid.cell.scaled_positions([r0])[0]
            if r1 is not None:
                r1 = self.grid.cell.scaled_positions([r1])[0]
            if r2 is not None:
                r2 = self.grid.cell.scaled_positions([r2])[0]
            if origin is not None:
                origin = self.grid.cell.scaled_positions([origin])[0]
            if center is not None:
                center = self.grid.cell.scaled_positions([center])[0]

        if do_center:
            x0 = center
        else:
            x0 = origin

        if do_center:
            x0 = x0 - 0.5 * r0

        if r1 is not None:
            if do_center:
                x0 = x0 - 0.5 * r1
            span += 1
            if r2 is not None:
                if do_center:
                    x0 = x0 - 0.5 * r2
                span += 1
        nrx = np.ones(3, dtype=int)
        if isinstance(nr, (np.ndarray, list, tuple)):
            nrx[0:span] = np.asarray(nr, dtype=int)
        # elif isinstance(nr, (int, float)):
        else:
            nrx[0:span] = nr

        dr = np.zeros((3, 3), dtype=float)
        dr[0, :] = (r0) / nrx[0]
        if span > 1:
            dr[1, :] = (r1) / nrx[1]
            if span == 3:
                dr[2, :] = (r2) / nrx[2]
        axis = []
        for ipol in range(3):
            axis.append(np.zeros((nrx[ipol], 3)))
            for ir in range(nrx[ipol]):
                axis[ipol][ir, :] = ir * dr[ipol]

        # points = np.zeros((nrx[0], nrx[1], nrx[2], 3))
        # for i in range(nrx[0]):
        # for j in range(nrx[1]):
        # for k in range(nrx[2]):
        # points[i, j, k, :] = x0 + axis[0][i, :] + axis[1][j, :] + axis[2][k, :]
        points = (
            axis[0].reshape((nrx[0], 1, 1, 3))
            + axis[1].reshape((1, nrx[1], 1, 3))
            + axis[2].reshape((1, 1, nrx[2], 3))
        )
        points += x0[:]

        a, b, c = nrx[0], nrx[1], nrx[2]
        points = points.reshape((nrx[0] * nrx[1] * nrx[2], 3))

        values = self.get_value_at_points(points)

        # generate a new grid (possibly 1D/2D/3D)
        origin = self.grid.cell.cartesian_positions([x0])[0]
        at = np.identity(3)
        v0 = self.grid.cell.cartesian_positions([r0])[0]
        v1 = np.zeros(3)
        v2 = np.zeros(3)
        # We still need to define 3 lattice vectors even if the plot is in 1D/2D
        # Here we ensure the 'ficticious' vectors are orthonormal to the actual ones
        # so that units of length/area are correct.
        if span == 1:
            for i in range(3):
                if abs(v0[i]) > 1e-4:
                    j = i - 1
                    v1[j] = v0[i]
                    v1[i] = -v0[j]
                    v1 = v1 / np.sqrt(np.dot(v1, v1))
                    break
            v2 = np.cross(v0, v1)
            v2 = v2 / np.sqrt(np.dot(v2, v2))
        elif span == 2:
            v1 = self.grid.cell.cartesian_positions([r1])[0]
            v2 = np.cross(v0, v1)
            v2 = v2 / np.sqrt(np.dot(v2, v2))
        elif span == 3:
            v1 = self.grid.cell.cartesian_positions([r1])[0]
            v2 = self.grid.cell.cartesian_positions([r2])[0]
        at[:, 0] = v0
        at[:, 1] = v1
        at[:, 2] = v2

        cut_grid = DirectGrid(lattice=at, nr=nrx, origin=origin)

        if span == 1:
            values = values.reshape((a,))
        elif span == 2:
            values = values.reshape((a, b))
        elif span == 3:
            values = values.reshape((a, b, c))

        return DirectField(grid=cut_grid, memo=self.memo, griddata_3d=values)

    def para_current(self, sigma=0.025):
        r"""
        Calculate <\psi|i\nabla|psi>
        """
        reciprocal_self = self.fft()
        reciprocal_self_conj = np.conj(self).fft()
        j_p = (
            reciprocal_self.grid.g
            * (-reciprocal_self * reciprocal_self_conj)
            * np.exp(-reciprocal_self.grid.gg * (sigma / 2.0) ** 2)
        )
        return j_p.integral()

    @property
    def N(self):
        if self._N is None:
            self._N = self.integral()
        return self._N

    @property
    def cplx(self):
        return self._cplx

    @cplx.setter
    def cplx(self, value):
        self._cplx = value
        if self._cplx:
            self.grid.full = True
            self.grid.cplx = True

    def tile(self, reps=1, **kwargs):
        # Overwrite the numpy.tile
        try:
            tup = tuple(reps)
        except TypeError:
            tup = (reps,)
        reps = np.asarray(tup)
        data = np.tile(np.asarray(self), reps)
        shape = data.shape
        rank = 1 if len(shape) == 3 else shape[0]
        if len(reps) > 3:
            reps = reps[-3:]
        if np.all(reps == 1):
            grid = self.grid
        else:
            self.grid.tile(reps)
        results = self.__class__(grid=grid, rank=rank, griddata_3d=data, cplx=self.cplx)
        return results

    def repeat(self, rep=1, **kwargs):
        # Overwrite the numpy.repeat, the different is it only repeat last three dimensions with same rep
        if not isinstance(rep, int):
            raise AttributeError(
                "Field repeat only support one integer, Please use 'tile'."
            )
        if self.rank == 1:
            reps = np.ones(3, dtype='int') * rep
        else:
            reps = np.ones(4, dtype='int') * rep
            reps[0] = 1
        return self.tile(reps, **kwargs)

    def gather(self, grid=None, out=None, root=0):
        if out is None:
            value = self.grid.gather(self, root=root)
            if self.grid.mp.rank == root:
                if grid is None:
                    grid = DirectGrid(
                        self.grid.lattice, self.grid.nrR, full=self.grid.full
                    )
                value = self.__class__(
                    grid=grid, rank=self.rank, griddata_3d=value, cplx=self.cplx
                )
        else:
            value = self.grid.gather(self, out=out, root=root)
        return value

    def __scatter(self, grid, data=None):
        pass
        if data is None:
            if self.grid.mp.is_mpi:
                data = self.grid.gather(self)
            else:
                data = self
        value = grid.scatter(data)
        value = self.__class__(
            grid=grid, rank=self.rank, griddata_3d=value, cplx=self.cplx
        )
        return value

    def write(self, filename, ions=None, format=None, **kwargs):
        from dftpy.formats import io as dftpy_io

        dftpy_io.write(filename, data=self, ions=ions, format=format, **kwargs)

    def read(self, filename, format=None, **kwargs):
        from dftpy.formats import io as dftpy_io

        self[:] = dftpy_io.read_density(filename, format=format, **kwargs)

    def cut_highg(self, g2max=None):
        recip = self.fft().cut_highg(g2max)
        return recip.ifft()


class ReciprocalField(BaseField):
    def __new__(cls, grid, memo="", rank=1, data=None, order='C', cplx=False, **kwargs):
        if hasattr(grid, 'get_reciprocal'):
            grid = grid.get_reciprocal()
        if not isinstance(grid, ReciprocalGrid):
            raise TypeError("the grid argument is not an instance of ReciprocalGrid")
        obj = super().__new__(
            cls, grid, memo="", rank=rank, data=data, order=order, cplx=True, **kwargs
        )
        #
        obj.init_options = locals()
        for k in ['__class__', 'cls', 'obj', 'kwargs', 'grid', 'data']:
            obj.init_options.pop(k, None)
        # obj.init_options.update(kwargs)
        #
        obj.spl_coeffs = None
        obj._cplx = cplx
        if obj.mp.is_mpi:
            from dftpy.mpi.mp_mpi4py import mpi_ifft

            obj.ifft_object = mpi_ifft(obj.grid)
        elif not obj._cplx and np.all(
            obj.grid.nrG == obj.grid.nrR
        ):  # Can only use numpy.fft
            obj.ifft_object = np.fft.ifftn
            # if environ["FFTLIB"] != 'numpy' :
            # print('!WARN : For full G-space, you can only use numpy.fft.\n So here we reset the FFT as numpy.fft.')
        elif environ["FFTLIB"] == "pyfftw":
            obj.ifft_object = PYifft(obj.grid, obj._cplx)
        elif environ["FFTLIB"] == "numpy":
            if obj._cplx:
                obj.ifft_object = np.fft.ifftn
            else:
                obj.ifft_object = np.fft.irfftn
        else:
            obj.ifft_object = np.fft.ifftn
        return obj

    def __array_finalize__(self, obj):
        r"""
        Restores attributes when we are taking a slice
        """
        if obj is None:
            return
        super().__array_finalize__(obj)
        if isinstance(obj, (ReciprocalField)):
            self.ifft_object = obj.ifft_object
            self.cplx = obj.cplx
        self.spl_coeffs = None

    def integral(self):
        """Returns the integral of self"""
        if self.rank == 1:
            return (
                self.mp.einsum("ijk->", self)
                * self.grid.dV
                * self.grid.nnrG
                / (2.0 * np.pi) ** 3
            )
        else:
            return (
                self.mp.einsum("lijk->l", self)
                * self.grid.dV
                * self.grid.nnrG
                / (2.0 * np.pi) ** 3
            )

    @timer('InvFFT')
    def ifft(self, check_real=False, force_real=False):
        """
        Implements the Inverse Discrete Fourier Transform
        """
        direct_grid = self.grid.get_direct()
        nr = self.rank, *direct_grid.nr
        fft_kwargs = {}
        if environ["FFTLIB"] == "numpy":
            fft_kwargs['s'] = self.grid.nrR
        if self.rank == 1:
            griddata_3d = self.ifft_object(self, **fft_kwargs) / direct_grid.dV
        else:
            if self.cplx or np.all(
                self.grid.nrG == self.grid.nrR
            ):  # Can only use numpy.fft
                griddata_3d = np.empty(nr, dtype="complex128")
            else:
                griddata_3d = np.empty(nr)
            for i in range(self.rank):
                griddata_3d[i] = (
                    self.ifft_object(self[i], **fft_kwargs) / direct_grid.dV
                )
        if check_real:
            if np.isclose(np.imag(griddata_3d), 0.0, atol=1.0e-16).all():
                griddata_3d = np.real(griddata_3d)
        if force_real:
            griddata_3d = np.real(griddata_3d)
        fft_data = DirectField(
            grid=direct_grid,
            memo=self.memo,
            rank=self.rank,
            griddata_3d=griddata_3d,
            cplx=self.cplx,
        )
        return fft_data

    @property
    def cplx(self):
        return self._cplx

    @cplx.setter
    def cplx(self, value):
        self._cplx = value
        if self._cplx and not self.grid.full:
            self.grid.full = True

    def cut_highg(self, g2max=None):
        if self.rank == 1:
            self[~self.grid.get_gmask(g2max)] = 0.0
        else:
            for i in range(self.rank):
                self[i][~self.grid.get_gmask(g2max)] = 0.0
        return self
