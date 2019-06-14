import os, numpy as np
from . import utils
from abc import ABCMeta, abstractmethod, abstractproperty

class Measurement(metaclass=ABCMeta):
    @abstractproperty
    def mode(self): pass

    @abstractproperty
    def scan_num(self): pass

    @abstractproperty
    def prefix(self): pass

    @property
    def path(self):
        return os.path.join(os.path.join(utils.raw_path, utils.prefixes[self.prefix], utils.measpath[self.mode].format(self.scan_num)))

    @property
    def nxsfilepath(self):
        return self.path + '.nxs'

    @property
    def command(self):
        command = utils.scan_command(self.nxsfilepath)
        if type(command) == np.ndarray:
            command = str(command)[2:-2]
        return command

    @property
    def masterfilepath(self):
        return os.path.join(self.path, utils.masterfilepath[self.mode].format(self.scan_num))

    @property
    def energy(self):
        return utils.energy(self.nxsfilepath)

    @property
    def exposure(self):
        return float(self.command.split(" ")[-1])

class Frame(Measurement):
    mode = 'frame'
    prefix, scan_num = None, None

    def __init__(self, prefix, scan_num):
        self.prefix, self.scan_num = prefix, scan_num
        self.data = utils.data(self.masterfilepath, 1)
  
class ScanFactory(Measurement):
    mode = 'scan'
    prefix, scan_num = None, None

    def __init__(self, prefix, scan_num):
        self.prefix, self.scan_num = prefix, scan_num

    def open(self):
        if self.command.startswith(utils.commands['stepscan1d']):
            return StepScan1D(self.prefix, self.scan_num)
        elif self.command.startswith(utils.commands['stepscan2d']):
            return StepScan2D(self.prefix, self.scan_num)
        elif self.command.startswith(utils.commands['flyscan2d']):
            return FlyScan2D(self.prefix, self.scan_num)
        else:
            raise ValueError('Unknown scan type')

class Scan(Measurement, metaclass=ABCMeta):
    mode = 'scan'
    prefix, scan_num = None, None

    @abstractproperty
    def fast_size(self): pass

    @abstractproperty
    def fast_crds(self): pass

    def data(self):
        return utils.data(self.masterfilepath, self.fast_size)

    def save_average(self):
        return 

class StepScan1D(Scan):
    mode = 'scan'
    prefix, scan_num, fast_size, fast_crds = None, None, None, None

    def __init__(self, prefix, scan_num):
        self.prefix, self.scan_num = prefix, scan_num
        self.fast_crds, self.fast_size = utils.coordinates(self.command)

class StepScan2D(Scan):
    mode = 'scan'
    prefix, scan_num, fast_size, fast_crds = None, None, None, None

    def __init__(self, prefix, scan_num):
        self.prefix, self.scan_num = prefix, scan_num
        self.fast_crds, self.fast_size, self.slow_crds, self.slow_size = utils.coordinates2d(self.command)

class FlyScan2D(Scan):
    mode = 'scan'
    prefix, scan_num, fast_size, fast_crds = None, None, None, None

    def __init__(self, prefix, scan_num):
        self.prefix, self.scan_num = prefix, scan_num
        self.fast_crds, self.fast_size, self.slow_crds, self.slow_size = utils.coordinates2d(self.command)