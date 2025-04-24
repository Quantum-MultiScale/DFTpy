from itertools import combinations, product

import matplotlib.pyplot as plt
import numpy as np

try:
    from skimage import measure
except Exception:
    raise ModuleNotFoundError("Please install 'scikit-image'")


class BasePlot(object):

    def __init__(self, fig=None, ax=None, projection=None):
        if fig is None:
            fig = plt.figure()
        self.fig = fig

        if ax is None:
            ax = fig.add_subplot(111, projection=projection)
        self.ax = ax

    def plot_cube(self, ax=None, r=[0, 1], color='k'):
        if ax is None:
            ax = self.ax
        for s, e in combinations(np.array(list(product(r, r, r))), 2):
            if np.sum(np.abs(s - e)) == r[1] - r[0]:
                ax.plot3D(*zip(s, e), color=color)

    def plot_remove_backgroud(self, ax=None):
        if ax is None:
            ax = self.ax
        # plt.axis('off')
        # ax.set_facecolor('white')
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        # removing the tick marks
        # ax.tick_params(bottom="off", left="off")
        # ax.tick_params(axis='both', which='both',length=0)
        ax.tick_params(length=0)
        ax.grid(False)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))


def plot_isosurface(
    density, level=None, spacing=(0.1, 0.1, 0.1), color='red', **kwargs
):
    dftpyplot = BasePlot(projection='3d')
    dftpyplot.plot_cube()
    dftpyplot.plot_remove_backgroud()
    nr = density.shape
    if level is None:
        level = 0.5 * (np.max(density) + np.min(density))
    # rho = np.pad(density, [[0,1],[0,1],[0,1]], mode="wrap")
    rho = density
    x, y, z = np.mgrid[0 : 1 : nr[0] + 1, 0 : 1 : nr[1] + 1, 0 : 1 : nr[2] + 1]
    verts, faces, normals, values = measure.marching_cubes(
        rho, level=level, spacing=spacing, **kwargs
    )
    dftpyplot.ax.plot_trisurf(
        verts[:, 0] / 2, verts[:, 1] / 2, faces, verts[:, 2] / 2, color=color, lw=1
    )


def plot_scatter(val, tol2=2e-6, marker='o', s=200, color='blue', **kwargs):
    dftpyplot = BasePlot(projection='3d')
    dftpyplot.plot_cube()
    dftpyplot.plot_remove_backgroud()

    dftpyplot.ax.scatter(
        val[:, 0], val[:, 1], val[:, 2], marker=marker, s=s, color=color
    )
    dftpyplot.ax.set_xlim(0 - tol2, 1 + tol2)
    dftpyplot.ax.set_ylim(0 - tol2, 1 + tol2)
    dftpyplot.ax.set_zlim(0 - tol2, 1 + tol2)
    dftpyplot.ax.set_xticks([])
    dftpyplot.ax.set_yticks([])
    dftpyplot.ax.set_zticks([])
