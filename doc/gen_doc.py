#!/usr/bin/env python3

from dftpy.config.config import default_json

header = r"""
.. _config:

====================
Script mode of DFTpy
====================

DFTpy is a set of python modules. However, it can be executed in the old way by using the `dftpy` script which is generated at installation time. Here's a quick guide to the script's configuration dictionary, or `config`.


.. list-table::

     * - `JOB`_
       - `PATH`_
       - `MATH`_
       - `PP`_
     * - `CELL`_
       - `GRID`_
       - `DENSITY`_
       - `EXC`_
     * - `KEDF`_
       - `OUTPUT`_
       - `OPT`_
       - `PROPAGATOR`_
     * - `TD`_
       - `CASIDA`_
       - `INVERSION`_
       -

.. warning::
    `PP`_ is a mandatory input (i.e., no default is avaliable for it).

.. note::
    Defaults work well for most arguments.

    When *Options* is empty, it can accept any value.

.. _pylibxc: https://tddft.org/programs/libxc/
"""

def gen_list_table(dicts, parent = None, ncol = 4):
    fstr = '\n'
    keys = list(dicts.keys())
    try:
        keys.remove('comment')
    except :
        pass
    if len(keys) > 0 :
        fstr += ".. list-table::\n\n"
        for i, key in enumerate(keys):
            if i%ncol == 0 :
                shift = '\t\t*'
            else :
                shift = '\t\t '

            if parent :
                fstr += "{0} - :ref:`{1}<{2}-{1}>`\n".format(shift, key, parent)
            else :
                fstr += "{0} - `{1}`_\n".format(shift, key)
        if len(keys) > ncol :
            for j in range(ncol - i%ncol - 1):
                fstr += '\t\t  -\n'
    return fstr + '\n'

def gen_config_rst():
    configentries = default_json()
    with open('./source/tutorials/config.rst', 'w') as f:
        f.write(header)
        for section in configentries:
            #-----------------------------------------------------------------------
            fstr = '\n'
            fstr += '{0}\n'.format(section)
            fstr += '-----------------\n\n'.format(section)
            if not isinstance(configentries[section], dict):
                configentries[section] = {'comment': configentries[section]}
            if 'comment' in configentries[section] :
                item = configentries[section]['comment']
                lines = str(item.default)
                lines = lines.replace('\\\\n', '\n')
                lines = lines.replace('\\\\t', '\t')
                # fstr += "\t{0}\n\n".format(item.default)
                fstr += "\t{0}\n\n".format(lines)
                if item.example :
                    fstr += "\t- *e.g.* : \n\n\t\t{0}\n".format(item.example)
                if item.note:
                    fstr += ".. note::\n {0}\n".format(item.note)
                if item.warning:
                    fstr += ".. warning::\n {0}\n".format(item.warning)
            f.write(fstr)
            f.write(gen_list_table(configentries[section], section))
            #-----------------------------------------------------------------------
            parent = section.lower()
            for key, item in configentries[section].items() :
                if key == 'comment' : continue
                fstr = "\n"
                fstr += ".. _{0}-{1}:\n\n".format(parent, key)
                fstr += "**{0}**\n".format(key)
                # fstr += ".. option:: {0}\n\n".format(key)
                if item.comment:
                    lines = str(item.comment)
                    lines = lines.replace('\\\\n', '\n')
                    lines = lines.replace('\\\\t', '\t')
                    # fstr += "\t{0}\n".format(item.comment)
                    fstr += "\t{0}\n".format(lines)
                fstr += '\n'
                fstr += "\t- *Options* : {0}\n\n".format(item.options)
                fstr += "\t- *Default* : {0}\n\n".format(item.default)
                if item.unit :
                    fstr += "\t- *Unit* : {0}\n\n".format(item.unit)
                if item.example :
                    fstr += "\t- *e.g.* : \n\n\t\t\t{0}\n".format(item.example)
                if item.note:
                    fstr += ".. note::\n {0}\n".format(item.note)
                if item.warning:
                    fstr += ".. warning::\n {0}\n".format(item.warning)
                f.write(fstr)


if __name__ == "__main__":
    gen_config_rst()
