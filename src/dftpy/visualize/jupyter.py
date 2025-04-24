import numpy as np


def view_density(density, level=None, viewer='ipyvolume', **kwargs):
    '''
    Visualize a density on jupyter notebooks.
    '''
    rho = np.pad(density, [[0, 1], [0, 1], [0, 1]], mode="wrap")

    if viewer == 'ipyvolume':
        import dftpy.visualize.ipv_viewer as ipvv

        ipvv.view_density(rho, level)
    else:
        import dftpy.visualize.mpl_viewer as mplv

        mplv.plot_isosurface(rho, level)


def view_ions(ions, viewer='ipyvolume', **kwargs):
    '''
    Visualize a density on jupyter notebooks.
    '''
    pos = ions.get_scaled_positions()
    tol = 1e-6
    tol2 = 2.0 * tol
    ixyzA = np.mgrid[-1:2, -1:2, -1:2].reshape((3, -1)).T
    pbcpos = []
    for p in ixyzA:
        pbcpos.append(pos + p)
    pbcpos = np.asarray(pbcpos).reshape((-1, 3))
    val = pbcpos
    for i in range(3):
        val = val[np.logical_and(val[:, i] < 1 + tol, val[:, i] > -tol)]

    if viewer == 'ipyvolume':
        import dftpy.visualize.ipv_viewer as ipvv

        ipvv.view_ions(val, tol2)
    else:
        import dftpy.visualize.mpl_viewer as mplv

        mplv.plot_scatter(val, tol2)
