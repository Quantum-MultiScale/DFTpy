from ase.visualize import view


def ase_view(ions, data=None, viewer='ase', repeat=None, block=False, **kwargs):
    atoms = ions.to_ase()
    view(atoms, data=data, viewer=viewer, repeat=repeat, block=block)
