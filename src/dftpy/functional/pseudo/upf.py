import importlib.util
import re
import xml.etree.ElementTree as ET

import numpy as np

from dftpy.functional.pseudo.abstract_pseudo import BasePseudo

"""
Ref :
    http://pseudopotentials.quantum-espresso.org/home/unified-pseudopotential-format
    https://esl.cecam.org/data/upf/
"""


class UPFDICT(BasePseudo):
    def __init__(self, fname, direct=True, **kwargs):
        super().__init__(fname, direct=direct, **kwargs)

    def upf_v1(self, info):
        """
        Note :
            The PP_NONLOCAL is different from UPF2.0
        """
        pp_header = dict.fromkeys(
            [
                'generated',
                'author',
                'date',
                'comment',
                'element',
                'pseudo_type',
                'relativistic',
                'is_ultrasoft',
                'is_paw',
                'is_coulomb',
                'has_so',
                'has_wfc',
                'has_gipaw',
                'core_correction',
                'functional',
                'z_valence',
                'total_psenergy',
                'wfc_cutoff',
                'rho_cutoff',
                'l_max',
                'l_local',
                'mesh_size',
                'number_of_wfc',
                'number_of_proj',
            ]
        )
        lines = info["PP_HEADER"].splitlines()
        data = [x.split() for x in lines]
        data[4] = lines[4]
        pp_header['element'] = data[1][0]
        pp_header['pseudo_type'] = data[2][0]
        pp_header['core_correction'] = data[3][0]
        pp_header['functional'] = data[4][:20].strip()
        pp_header['z_valence'] = data[5][0]
        pp_header['total_psenergy'] = data[6][0]
        pp_header["rho_cutoff"], pp_header["wfc_cutoff"] = data[7][0:2]
        pp_header['l_max'] = data[8][0]
        pp_header['mesh_size'] = data[9][0]
        pp_header['number_of_wfc'], pp_header['number_of_proj'] = data[10][0:2]
        #
        relativistic = 'scalar'
        l_local = '0'
        has_wfc = 'F'
        #
        is_ultrasoft = 'F'
        is_paw = 'F'
        is_coulomb = 'F'
        has_so = 'F'
        has_gipaw = 'F'
        if pp_header['pseudo_type'] == 'US':
            is_ultrasoft = 'T'
        elif pp_header['pseudo_type'] == 'PAW':
            is_ultrasoft = 'T'
            is_paw = 'T'
        elif pp_header['pseudo_type'] == 'NC':
            pass
        elif pp_header['pseudo_type'] == '1/r':
            is_coulomb = 'T'
        else:
            raise ValueError('Unknown pseudo_type', pp_header['pseudo_type'])

        if 'PP_ADDINFO' in info:
            has_so = 'T'
        if 'PP_GIPAW_RECONSTRUCTION_DATA' in info:
            has_gipaw = 'T'
        #
        pp_header['relativistic'] = relativistic
        pp_header['has_so'] = has_so
        pp_header['l_local'] = l_local
        pp_header['is_ultrasoft'] = is_ultrasoft
        pp_header['is_paw'] = is_paw
        pp_header['is_coulomb'] = is_coulomb
        pp_header['has_wfc'] = has_wfc
        pp_header['has_gipaw'] = has_gipaw
        #
        generated = " "
        author = " "
        date = " "
        comment = " "
        lines = info["PP_INFO"].splitlines()
        if len(lines) > 0:
            generated = lines[0]
        if len(lines) > 1:
            l = lines[1].lower()
            i0 = l.find(':')
            i1 = l.find('generat')
            author = lines[1][i0 + 1 : i1].strip()
            l = l[i1:]
            i2 = l.find(':')
            date = l[i2 + 1 :].strip()
        if len(lines) > 2:
            comment = lines[2].strip()
        pp_header['generated'] = generated
        pp_header['author'] = author
        pp_header['date'] = date
        pp_header['comment'] = comment

        info["PP_HEADER"] = pp_header
        return info

    def upf2dict(self, string):
        string = '<UPF>\n' + string + '</UPF>\n'
        if not importlib.util.find_spec("xmltodict"):
            raise ModuleNotFoundError("Please install xmltodict to read 'UPF' file")
        import xmltodict

        doc = xmltodict.parse(string, attr_prefix='')
        info = doc[next(iter(doc.keys()))]
        if len(info) < 2:
            info = info[next(iter(info.keys()))]
        else:
            info = self.upf_v1(info)
        return info

    def read(self, fname):
        """Reads QE UPF type PP"""
        self.pattern = re.compile(r'\s+')
        with open(fname) as fd:
            string = fd.read()
        info = self.upf2dict(string)
        r = self.get_array(info['PP_MESH']['PP_R'])
        v = self.get_array(info['PP_LOCAL']) * 0.5  # Ry to a.u.
        self.r = r
        self.v = v
        self.info = info
        self._zval = float(self.info["PP_HEADER"]["z_valence"])
        if 'PP_NLCC' in self.info:
            self._core_density = self.get_array(self.info['PP_NLCC'])
        if 'PP_RHOATOM' in self.info:
            rho = self.get_array(self.info['PP_RHOATOM'])[: self.r.size]
            if self.r[0] > 1e-10:
                rho[:] /= 4 * np.pi * self.r[:] ** 2
            else:
                rho[1:] /= 4 * np.pi * self.r[1:] ** 2
            self._atomic_density = rho

    def get_array(self, attr):
        if isinstance(attr, dict):
            attr = attr['#text']
        if attr is None or len(attr) == 0:
            return None
        value = self.pattern.split(attr)
        return np.array(value, dtype=np.float64)


class UPFJSON_alternative(BasePseudo):
    def __init__(self, fname, direct=True, **kwargs):
        super().__init__(fname, direct=direct, **kwargs)

    def read(self, fname):
        """Reads QE UPF type PP"""
        import importlib.util

        info, header, _ = parse_upf_file(fname)

        r = np.array(info["PP_R"], dtype=np.float64)
        v = np.array(info["PP_LOCAL"], dtype=np.float64) * 0.5  # Ry to Ha
        self.r = r
        self.v = v
        self.info = header
        self._zval = float(self.info["z_valence"])


class UPFJSON(BasePseudo):
    def __init__(self, fname, direct=True, **kwargs):
        super().__init__(fname, direct=direct, **kwargs)

    def read(self, fname):
        """Reads QE UPF type PP"""
        import importlib.util

        upf2json = importlib.util.find_spec("upf_to_json")
        found = upf2json is not None
        if found:
            from upf_to_json import upf_to_json
        else:
            raise ModuleNotFoundError("Please pip install upf_to_json")
        with open(fname, "r") as outfil:
            info = upf_to_json(upf_str=outfil.read(), fname=fname)["pseudo_potential"]
        r = np.array(info["radial_grid"], dtype=np.float64)
        v = np.array(info["local_potential"], dtype=np.float64)
        self.r = r
        self.v = v
        self.info = info
        self._zval = self.info["header"]["z_valence"]
        if 'core_charge_density' in self.info:
            self._core_density = np.array(
                self.info["core_charge_density"], dtype=np.float64
            )
        if 'total_charge_density' in self.info:
            rho = np.array(self.info["total_charge_density"], dtype=np.float64)[
                : self.r.size
            ]
            if self.r[0] > 1e-10:
                rho[:] /= 4 * np.pi * self.r[:] ** 2
            else:
                rho[1:] /= 4 * np.pi * self.r[1:] ** 2
            self._atomic_density = rho


class UPF:
    def __new__(cls, fname):
        """
        Note :
            Prefer xmltodict which is more robust
        """

        # obj = UPFJSON_alternative(fname)
        try:
            obj = UPFDICT(fname)
        except Exception:
            try:
                obj = UPFJSON_alternative(fname)
            except Exception:
                if importlib.util.find_spec("upf_to_json"):
                    try:
                        obj = UPFJSON(fname)
                    except Exception:
                        raise ModuleNotFoundError("Please use a standard 'UPF' file.")
                else:
                    raise ModuleNotFoundError(
                        "Please install 'upf_to_json' or use a standard 'UPF' file."
                    )
        return obj


import re
import xml.etree.ElementTree as ET

import numpy as np


def parse_upf_file(file_path):
    """
    Reads a UPF file and extracts array_dict, header_info, and array_metadata.

    Parameters:
    - file_path (str): Path to the UPF file.

    Returns:
    - array_dict (dict): Dictionary where keys are XML tags and values are NumPy arrays.
    - header_info (dict): Dictionary with header information.
    - array_metadata (dict): Dictionary containing metadata for each array tag (e.g., type, size, columns).
    """
    array_dict = {}
    header_info = {}
    array_metadata = {}

    # Parse the XML structure in the UPF file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Extract PP_HEADER information directly from attributes
    header_elem = root.find('PP_HEADER')
    if header_elem is not None:
        # Extract each attribute and add to header_info
        for attr, value in header_elem.attrib.items():
            header_info[attr] = value

    # Recursive function to extract arrays and metadata from elements
    def extract_arrays(elem):
        # Skip PP_INFO and PP_HEADER sections, which have been handled
        if elem.tag in ['PP_INFO', 'PP_HEADER']:
            return

        # Check if element contains numerical array data
        if elem.text and any(char.isdigit() for char in elem.text):
            # Extract metadata from element attributes (type, size, columns)
            array_type = elem.attrib.get('type', 'real')
            array_size = int(elem.attrib.get('size', 0))
            array_columns = int(elem.attrib.get('columns', 1))
            array_metadata[elem.tag] = {
                'type': array_type,
                'size': array_size,
                'columns': array_columns,
            }

            # Extract and parse the numerical data
            # Remove extra whitespace, split by lines, then convert to floats
            data_lines = elem.text.strip().splitlines()
            data_values = []
            for line in data_lines:
                # Extract individual numbers in scientific notation
                numbers = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", line)
                data_values.extend([float(num) for num in numbers])

            # Store as a NumPy array
            array_dict[elem.tag] = np.array(data_values)

        # Recursively handle nested elements
        for child in elem:
            extract_arrays(child)

    # Start the recursive extraction from the root element
    for elem in root:
        extract_arrays(elem)

    return array_dict, header_info, array_metadata
