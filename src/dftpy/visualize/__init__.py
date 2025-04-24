from dftpy.visualize.ase_viewer import ase_view
from dftpy.visualize.jupyter import view_density, view_ions
from dftpy.visualize.vesta_viewer import view_on_vesta


def view(ions=None, data=None, viewer='mpl', **kwargs):
    """visualize the ions and field

    Parameters
    ----------
    ions
        ions
    data
        the field which can be density of potential
    viewer
        the visualization tool
    **kwargs
        kwargs

    Notes
    -----
    viewer = 'ipyvolume' need install ipyvolume (==0.6.0a10 for python == 3.10) and scikit-image
    viewer = 'mpl' need install the scikit-image
    """
    if viewer in ['ipyvolume', 'mpl']:
        if ions is None:
            view_density(data, viewer=viewer, **kwargs)
        elif data is None:
            view_ions(ions, viewer=viewer, **kwargs)
    elif viewer == 'vesta':
        view_on_vesta(ions=ions, data=data, **kwargs)
    else:
        ase_view(ions, data=data, viewer=viewer, **kwargs)
