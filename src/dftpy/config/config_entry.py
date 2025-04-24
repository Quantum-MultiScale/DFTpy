import ast
import os
import re
from collections import OrderedDict
from json import JSONEncoder

import numpy as np

try:
    from numexpr import evaluate

    is_numexpr = True
except Exception:
    is_numexpr = False


def format_bool(expression):
    s = expression.lower()[0]
    if s in ["n", "f", "0"]:
        return False
    else:
        return True


def format_float(expression):
    if is_numexpr:
        return evaluate(expression).item()
    else:
        return eval(expression)


def format_str(expression):
    return expression


def format_lstr(expression):
    return expression.lower()


def format_ustr(expression):
    return expression.upper()


def format_cstr(expression):
    return expression.capitalize()


def format_path(expression):
    def func(m):
        return os.environ.get(m.group(1))

    return re.subn(r'\$\{(.*?)\}', func, expression)[0]


def format_slice(expression):
    if ':' in expression:
        ls = expression.split(':')
        l = [
            None,
        ] * 3
        for i, item in enumerate(ls):
            if item.lstrip('-+').isdigit():
                l[i] = int(item)
        return slice(*l)
    else:
        return int(expression)


def format_intlist(expression):
    if ':' in expression:
        items = expression.split()
        if len(items) == 1:
            return format_slice(items[0])
        ints = []
        for item in items:
            s = format_slice(item)
            if ':' in item:
                a = np.arange(0, s.stop)[s].tolist()
                ints.extend(a)
            else:
                ints.append(s)
        return ints
    else:
        return list(map(int, expression.split()))


def format_floatlist(expression):
    return list(map(format_float, expression.split()))


def format_strlist(expression):
    return expression.split()


def format_cstrlist(expression):
    return expression.title().split()


def format_lstrlist(expression):
    return expression.lower().split()


def format_direction(expression):
    direct_dict = {"x": 0, "y": 1, "z": 2}
    if expression in direct_dict:
        return direct_dict[expression]
    else:
        return int(expression)


def format_cdict(expression):
    vk = ast.literal_eval(expression)
    return OrderedDict((v.capitalize(), k) for v, k in vk.items())


def format_cfdict(expression):
    vk = ast.literal_eval(expression)
    return OrderedDict((v.capitalize(), float(k)) for v, k in vk.items())


def format_cidict(expression):
    vk = ast.literal_eval(expression)
    return OrderedDict((v.capitalize(), int(k)) for v, k in vk.items())


def format_odict(expression):
    vk = ast.literal_eval(expression)
    return OrderedDict((v, k) for v, k in vk.items())


def format_ofdict(expression):
    vk = ast.literal_eval(expression)
    return OrderedDict((v, float(k)) for v, k in vk.items())


def format_oidict(expression):
    vk = ast.literal_eval(expression)
    return OrderedDict((v, int(k)) for v, k in vk.items())


class ConfigEntry(object):

    def __init__(
        self,
        type='str',
        default=None,
        comment='',
        options='',
        example=None,
        note=None,
        warning=None,
        unit=None,
        level=None,
        **kwargs,
    ):
        self.type = type
        self.default = default
        self.comment = comment
        self.options = options
        self.note = note
        self.example = example
        self.warning = warning
        self.unit = unit
        self.level = level
        if self.type == 'bool' and self.options == '':
            self.options = 'True, False'

    def format(self, string):
        format_dict = {
            "bool": format_bool,
            "int": int,
            "float": format_float,
            "str": format_str,
            "lstr": format_lstr,
            "ustr": format_ustr,
            "cstr": format_cstr,
            "path": format_path,
            "intlist": format_intlist,
            "floatlist": format_floatlist,
            "strlist": format_strlist,
            "cstrlist": format_cstrlist,
            "lstrlist": format_lstrlist,
            "direction": format_direction,
            "cdict": format_cdict,
            "cfdict": format_cfdict,
            "cidict": format_cidict,
            "odict": format_odict,
            "ofdict": format_ofdict,
            "oidict": format_oidict,
        }
        expression = re.split('#|!', string)[0]
        return format_dict[self.type](expression)

    def gen_doc(self, key):
        output = [
            ".. option:: {0:}".format(key),
            "",
            "    {0:}".format(self.comment),
            "        *Options* : {0:}".format(self.options),
            "",
            "        *Default* : {0:}".format(self.default),
            "",
            "",
        ]
        return '\n'.join(output)


class ConfigEntryEncoder(JSONEncoder):

    def default(self, o):
        return o.__dict__
