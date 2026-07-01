#!/usr/bin/env python3
import unittest
import numpy as np

from dftpy.grid import DirectGrid, ReciprocalGrid
from dftpy.field import DirectField, ReciprocalField
from dftpy.constants import Units

from common import run_test_orthorombic, run_test_triclinic, make_orthorombic_cell, make_triclinic_cell

class TestField(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
          This setUp is in common for all the test cases below, and it's only execuded once
        """
        # Test a constant scalar field
        N = 8
        A, B, C = 5, 11, 7
        nr = np.array([A, B, C])
        A, B, C = A/Units.Bohr, B/Units.Bohr, C/Units.Bohr
        grid = make_orthorombic_cell(A=A,B=B,C=C,CellClass=DirectGrid, nr=nr)
        d = N/grid.volume
        initial_vals = np.ones(nr)*d
        cls.constant_field = DirectField(grid=grid, griddata_3d=initial_vals)
        cls.three_field = DirectField(grid=grid, griddata_3d=np.stack([initial_vals,initial_vals,initial_vals]),rank=3)
        cls.N = N

    @classmethod
    def Gaussian_3d(self, grid_coor,sigma=0.8):
        r"""
        This function calculates analytical 3D Gaussian function.
        Parameters:
            grid_coor (grid.r):  three coordinates (X, Y, Z) at which to compute the gradient.
            sigma (float): The standard deviation of the Gaussian function. Default is 0.3.

        Returns:
            numpy.ndarray: A three-dimensional numpy array representing the gradient vector.

        The Gaussian function is defined as:
            G(x, y, z) = exp(-((x - 0.9)^2 + (y - 0.9)^2 + (z - 0.9)^2) / (2 * sigma^2))

        """
        X = grid_coor[0]
        Y = grid_coor[1]
        Z = grid_coor[2]

        d = np.sqrt((X-5)**2 + (Y-5)**2 + (Z-5)**2)
        g = 1.0/(np.sqrt(2*np.pi)*sigma ) *  np.exp(-d**2 / ( 2.0 * sigma**2 ) )
               
        return g

    @classmethod
    def Gaussian_3d_grad(self, grid_coor,sigma=0.8):
        r"""
        This function calculates the analytical gradient of a three-dimensional Gaussian function.
        Parameters:
            grid_coor (grid.r):  three coordinates (X, Y, Z) at which to compute the gradient.
            sigma (float): The standard deviation of the Gaussian function. Default is 0.3.

        Returns:
            numpy.ndarray: A three-dimensional numpy array representing the gradient vector.

        Note: The Gaussian_3d function is referenced here but should be defined elsewhere in the code.
        """
        X = grid_coor[0]
        Y = grid_coor[1]
        Z = grid_coor[2]
        
        g = self.Gaussian_3d(grid_coor,sigma)
        
        gradX = (-(X-5)/sigma**2)*g
        gradY = (-(Y-5)/sigma**2)*g
        gradZ = (-(Z-5)/sigma**2)*g
    
        return np.stack([gradX, gradY, gradZ])

    @classmethod
    def Gaussian_3d_hess(self,grid_coor,sigma=0.8):
        r"""
        This function calculates the analytical Hessian matrix of a 3D Gaussian function.
        Parameters:
            grid_coor (grid.r): A list containing the three coordinates (X, Y, Z).
            sigma (float): The standard deviation of the Gaussian function. Default is 0.3.

        Returns:
            numpy.ndarray: Asymmetric Hessian matrix represented as a flattened array [XX, YY, ZZ, XY, XZ, YZ].
        """
        X = grid_coor[0]
        Y = grid_coor[1]
        Z = grid_coor[2]
        
        g = self.Gaussian_3d(grid_coor,sigma)
        
        hessXX=  (((X-5)**2-sigma**2)/sigma**4)*g
        hessYY = (((X-5)**2-sigma**2)/sigma**4)*g
        hessZZ = (((X-5)**2-sigma**2)/sigma**4)*g    

        hessXY = (-(X-5)/sigma**2)*(-(Y-5)/sigma**2)*g
        hessYZ = (-(Y-5)/sigma**2)*(-(Z-5)/sigma**2)*g
        hessXZ = (-(Z-5)/sigma**2)*(-(X-5)/sigma**2)*g

        return np.stack([hessXX, hessYY, hessZZ, hessXY, hessXZ, hessYZ])


    def test_direct_field(self):
        #
        # TO DO: add tests for values using Gaussian fields
        #
        print()
        print("*"*50)
        print("Testing DirectField")
        #print(initial_vals[0,0,:])
        #print(field[0,0,:])
        field = self.constant_field
        three_field = self.three_field
        N = self.N

        self.assertTrue(type(field) is DirectField)
        N1 = field.integral()
        self.assertAlmostEqual(N,N1)

        # fft
        reciprocal_field = field.fft()
        self.assertAlmostEqual(N, reciprocal_field[0,0,0])

        # ifft
        field1 = reciprocal_field.ifft(check_real=True)
        N1 = field1.integral()
        self.assertAlmostEqual(N,N1)
        
        # Gradient, Hessian, Divergence, Laplacian type and value test
        #
        # 0. Generate grid for the test (coarse enough for CI; was 19× base → 209³)
        A, B, C = 11, 11, 11
        gaussian_mult = 7
        nr = np.array([A * gaussian_mult, B * gaussian_mult, C * gaussian_mult])
        grid = make_orthorombic_cell(A=A,B=B,C=C, CellClass=DirectGrid,\
                                                 nr=nr)
        DataGaussian = self.Gaussian_3d(grid_coor=grid.r,sigma=0.7) # generate 3D Gaussian Grid
        # Map 3D gaussian on grids
        fieldGaussian = DirectField(grid=grid, griddata_3d=DataGaussian)
        # Analytical gradient
        ana_grad = self.Gaussian_3d_grad(grid_coor=grid.r,sigma=0.7)

        # 1. gradient of a Gaussian function
        gradient = fieldGaussian.gradient() # default standard gradient
        gradient_n = fieldGaussian.gradient(flag="numerical") # default standard gradient
        # Type Check
        self.assertTrue(isinstance(gradient, DirectField))
        self.assertEqual(gradient.rank, 3)
        self.assertTrue(isinstance(gradient_n, DirectField))
        self.assertEqual(gradient_n.rank, 3)
        # Value Check for standard gradient
        D1 = np.max(gradient)-np.max(ana_grad)  # Compare the maximum value
        self.assertTrue(np.isclose(D1, 0.0, atol=0.01))

        D2 = np.sqrt(((gradient - ana_grad) ** 2).mean()) # RMSE
        self.assertTrue(np.isclose(D2, 0.0, atol=0.01))

        # Value Check for numerical standard gradient
        D1 = np.max(gradient_n)-np.max(ana_grad)  # Compare the maximum value
        self.assertTrue(np.isclose(D1, 0.0, atol=0.01))

        D2 = np.sqrt(((gradient_n - ana_grad) ** 2).mean()) # RMSE
        self.assertTrue(np.isclose(D2, 0.0, atol=0.01))
         
        # Value Check for smooth gradient
        gradient = fieldGaussian.gradient(flag="supersmooth", sigma=0.03) # supersmooth gradient
        D1 = np.max(gradient)-np.max(ana_grad)  # Compare the maximum value
        self.assertTrue(np.isclose(D1, 0.0, atol=0.01))

        D2 = np.sqrt(((gradient - ana_grad) ** 2).mean()) # RMSE
        self.assertTrue(np.isclose(D2, 0.0, atol=0.01))



        # 2. Hessian of a Gaussian function
        hess = fieldGaussian.hessian(flag="supersmooth",sigma=0.03)
        # Analytical hessian
        ana_hess = self.Gaussian_3d_hess(grid_coor=grid.r,sigma=0.7)

        self.assertTrue(isinstance(hess, DirectField))
        self.assertEqual(hess.rank, 6)

        # Value Check for smooth Hessian
        D1 = np.max(hess)-np.max(ana_hess)  # Compare the maximum value
        self.assertTrue(np.isclose(D1, 0.0, atol=0.01))

        D2 = np.sqrt(((hess - ana_hess) ** 2).mean()) # RMSE
        self.assertTrue(np.isclose(D2, 0.0, atol=0.03))
        # Todo: Standard and Smooth hessian


        # 3. Divergence: value check TBD.
        div = three_field.divergence()
        self.assertTrue(isinstance(div, DirectField))
        self.assertEqual(div.rank, 1)

        # 4. Laplacian of a Gaussian function
        lap = fieldGaussian.laplacian(force_real=True, sigma=0.03)  # Standard method
        ana_lap = hess[0] + hess[1] + hess[2]
        # Type Check
        self.assertTrue(isinstance(lap, DirectField))
        self.assertEqual(lap.rank, 1)

        # Value Check for standard hessian 
        D1 = np.max(lap)-np.max(ana_lap)  # Compare the maximum value
        self.assertTrue(np.isclose(D1, 0.0, atol=0.01))

        D2 = np.sqrt(((lap - ana_lap) ** 2).mean()) # RMSE
        self.assertTrue(np.isclose(D2, 0.0, atol=0.01))

        # 5. sigma of a Gaussian function rho(r)**2
        sig = fieldGaussian.sigma(flag="standard") # default standard gradient
        # Analytical gradient
        ana_sig = ana_grad[0]**2 + ana_grad[1]**2 + ana_grad[2]**2
        # Type Check
        self.assertTrue(isinstance(sig, DirectField))
        self.assertEqual(sig.rank, 1)
        # Value Check for standard sigma
        D1 = np.max(sig)-np.max(ana_sig)  # Compare the maximum value
        self.assertTrue(np.isclose(D1, 0.0, atol=0.01))

        D2 = np.sqrt(((sig - ana_sig) ** 2).mean()) # RMSE
        self.assertTrue(np.isclose(D2, 0.0, atol=0.01))

        # hessian
        hessian = field.hessian()
        self.assertTrue(isinstance(hessian, DirectField))
        self.assertEqual(hessian.rank, 6)

    def test_direct_field_interpolation(self):
        field = self.constant_field
        nr = field.grid.nr
        # interpolate up
        field1 = field.get_3dinterpolation(np.array(nr*1.5,dtype=int))
        N1 = field1.integral()
        self.assertAlmostEqual(self.N,N1)

        # interpolate down
        field2 = field.get_3dinterpolation(nr//2)
        N1 = field2.integral()
        self.assertAlmostEqual(self.N,N1)

    def test_direct_field_cut(self):
        field = self.constant_field
        nr = field.grid.nr
        x0 = [0,0,0]
        r0 = [1,0,0]
        field_cut = field.get_cut(origin=x0, r0=r0, nr=nr[0])
        self.assertTrue(np.isclose(field_cut[0,0,:], field[0,0,:]).all())

    def test_fft_ifft(self):
        nr=(51,51,51)
        A=10.0
        B=10.0
        C=10.0

        dgrid = make_orthorombic_cell(A=A, B=B, C=C, CellClass=DirectGrid, nr=nr)
        rgrid=dgrid.get_reciprocal()

        def ReciprocalSpaceGaussian(sigma,mu,grid):
            if not isinstance(grid,(ReciprocalGrid)):
                raise Exception()
            a = np.einsum('lijk,l->ijk',grid.g,mu)
            b = np.exp(-sigma**2*grid.gg/2.0)
            c = np.exp(-1j*a)
            d=np.einsum('ijk,ijk->ijk',b,c)
            return ReciprocalField(grid=grid,rank=1,griddata_3d=d)

        def DirectSpaceGaussian(sigma,mu,grid):
            if not isinstance(grid,(DirectGrid)):
                raise Exception()
            a = grid.r-mu[:, None, None, None]
            b = (sigma*np.sqrt(2.0*np.pi))**(-3.0)*np.exp( - 0.5 * np.einsum('lijk,lijk->ijk', a, a) / sigma**2.0 )
            return DirectField(grid=grid,rank=1,griddata_3d=b)

        center = dgrid.r[:, 25,25,25]
        sigma = 0.5

        rf = ReciprocalSpaceGaussian(sigma,center,rgrid)
        df = DirectSpaceGaussian(sigma,center,dgrid)

        df_dftpy=rf.ifft()
        rf_dftpy=df.fft()

        self.assertTrue(np.isclose(df_dftpy.integral(),1.0))
        self.assertTrue(np.isclose(df.integral(),1.0))

        self.assertTrue(np.isclose(rf,rf_dftpy).all())
        self.assertTrue(np.isclose(df,df_dftpy).all())


if __name__ == "__main__":
    unittest.main()
