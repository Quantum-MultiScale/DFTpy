import time
from functools import wraps

from dftpy.mpi import sprint


class TimeObj(object):
    """ """

    def __init__(self, **kwargs):
        self.reset(**kwargs)

    def reset(self, **kwargs):
        self.labels = []
        self.tic = {}
        self.toc = {}
        self.cost = {}
        self.number = {}

    def Begin(self, label):
        if label in self.tic:
            self.number[label] += 1
        else:
            self.labels.append(label)
            self.number[label] = 1
            self.cost[label] = 0.0

        self.tic[label] = time.time()

    def Time(self, label):
        if label not in self.tic:
            print(
                '!WARN : You should call "Begin" before this label : {}'.format(label)
            )
            t = 0
        else:
            t = time.time() - self.tic[label]
        return t

    def End(self, label):
        if label not in self.tic:
            print(
                '!WARN : You should call "Begin" before this label : {}'.format(label)
            )
            t = 0
        else:
            self.toc[label] = time.time()
            t = time.time() - self.tic[label]
            self.cost[label] += t
        return t

    def output(self, config=None, sort=0, lprint=False, comm=None, **kwargs):
        """
        sort : Label(0), Cost(1), Number(2), Avg(3)
        """
        column = {
            'label': 0,
            'cost': 1,
            'number': 2,
            'avg': 3,
        }
        if isinstance(sort, str):
            sort = sort.lower()
        if sort in column:
            idx = column[sort]
        elif isinstance(sort, (int, float)):
            idx = int(sort)
            if idx < 0 or idx > 3:
                idx = 0
        else:
            idx = 0
        sprint(format("Time information", "-^80"), comm=comm)
        lenk = max(max([len(x) for x in self.cost]), 28)
        fmth = "{:" + str(lenk) + "s}{:24s}{:16s}{:24s}"
        sprint(fmth.format("Label", "Cost(s)", "Number", "Avg. Cost(s)"), comm=comm)
        fmt = "{:" + str(lenk) + "s}{:<24.4f}{:<16d}{:<24.4f}"
        if config:
            if isinstance(config, dict) and not config["OUTPUT"]["time"]:
                lprint = False
            else:
                lprint = True
        if lprint:
            info = []
            for key, cost in self.cost.items():
                if key == 'TOTAL':
                    continue
                item = [key, cost, self.number[key], cost / self.number[key]]
                info.append(item)
            for item in sorted(info, key=lambda d: d[idx]):
                sprint(fmt.format(*item), comm=comm)
        key = "TOTAL"
        if key in self.cost:
            sprint(
                fmt.format(
                    key,
                    self.cost[key],
                    self.number[key],
                    self.cost[key] / self.number[key],
                ),
                comm=comm,
            )


TimeData = TimeObj()


def timer(label: str = None):
    """
    A decorator times the function
    Parameters
    ----------
    label

    Returns
    -------


    """

    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            tag = label
            if tag is None:
                if hasattr(function, '__qualname__'):
                    tag = function.__qualname__
                else:
                    tag = function.__class__.__name__
            TimeData.Begin(tag)
            results = function(*args, **kwargs)
            TimeData.End(tag)
            return results

        return wrapper

    return decorator
