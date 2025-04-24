def view_on_vesta(ions=None, data=None, saveas=None, **kwargs):
    """View ions and data on VESTA. Must have VESTA installed.

    Parameters
    ----------
    ions : Ions
        DFTpy Ions
    data : DirectField
        DFTpy DirectField
    saveas : string
        File name to save the data
    """
    import os
    import sys

    from dftpy.formats import io

    if saveas is None:
        if data is not None:
            saveas = './.dftpy.xsf'
        else:
            saveas = './.dftpy.vasp'

    if saveas in os.listdir('./'):
        os.remove(saveas)

    io.write(saveas, data=data, ions=ions)

    print(f'save {saveas} in {sys.platform} platform')

    if sys.platform == 'linux':
        os.system('VESTA ' + saveas)
    elif sys.platform == 'darwin':
        os.system('open -a VESTA ' + saveas)
    else:
        raise Exception('Unknown OS')
