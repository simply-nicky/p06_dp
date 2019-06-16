import os, numpy as np, h5py
from . import utils
from abc import ABCMeta, abstractmethod, abstractproperty

class Measurement(metaclass=ABCMeta):
    @abstractproperty
    def mode(self): pass

    @abstractproperty
    def scan_num(self): pass

    @abstractproperty
    def prefix(self): pass

    @abstractmethod
    def _save_data(self, outfile): pass

    @abstractmethod
    def size(self): pass

    @property
    def path(self):
        return os.path.join(os.path.join(utils.raw_path, utils.prefixes[self.prefix], utils.measpath[self.mode].format(self.scan_num)))

    @property
    def nxsfilepath(self):
        return self.path + '.nxs'

    @property
    def command(self):
        return utils.scan_command(self.nxsfilepath)

    @property
    def datapath(self):
        return os.path.join(self.path, utils.datafilepath)

    @property
    def energy(self):
        return utils.energy(self.nxsfilepath)

    @property
    def exposure(self):
        return float(self.command.split(" ")[-1])

    def data(self):
        return utils.data(self.datapath, self.size[-1])

    def _create_outfile(self):
        self.outpath = os.path.join(os.path.dirname(__file__), utils.outpath[self.mode].format(self.scan_num))
        self.filename = utils.filename[self.mode].format(self.scan_num)
        utils.make_output_dir(self.outpath)
        return h5py.File(os.path.join(self.outpath, self.filename), 'w')

    def _save_parameters(self, outfile):
        arggroup = outfile.create_group('arguments')
        arggroup.create_dataset('experiment',data=self.prefix)
        arggroup.create_dataset('scan mode', data=self.__class__.__name__)
        arggroup.create_dataset('scan number', data=self.scan_num)
        arggroup.create_dataset('raw path', data=self.path)
        arggroup.create_dataset('command', data=self.command)
        arggroup.create_dataset('energy', data=self.energy)
        arggroup.create_dataset('exposure', data=self.exposure)

    def save(self):
        outfile = self._create_outfile()
        self._save_parameters(outfile)
        self._save_data(outfile)
        outfile.close()

class Frame(Measurement):
    mode = 'frame'
    prefix, scan_num = None, None

    def size(self): return (1,)

    def __init__(self, prefix, scan_num):
        self.prefix, self.scan_num = prefix, scan_num
    
    def _save_data(self, outfile):
        datagroup = outfile.create_group('data')
        datagroup.create_dataset('data', data=self.data)
  
class ScanFactory(object):
    def __init__(self, prefix, scan_num):
        self.prefix, self.scan_num = prefix, scan_num
        self.path = os.path.join(os.path.join(utils.raw_path, utils.prefixes[self.prefix], utils.measpath['scan'].format(self.scan_num)))
        self.command = utils.scan_command(self.path + '.nxs')

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

    @abstractproperty
    def fast_size(self): pass

    @abstractproperty
    def fast_crds(self): pass

    @abstractproperty
    def data(self): pass

    @property
    def size(self): return (self.fast_size,)

    def flatfield_correct(self, bg_num):
        bg_scan = ScanFactory(self.prefix, bg_num).open()
        flatfield = np.mean(bg_scan.data, axis=0)
        return CorrectedScan(self, flatfield)

    def _save_data(self, outfile):
        datagroup = outfile.create_group('data')
        datagroup.create_dataset('fast_coordinates', data=self.fast_crds)
        datagroup.create_dataset('data', data=self.data, compression='gzip')

class CorrectedScan(object):
    def __init__(self, scan, flatfield):
        self.scan, self.flatfield = scan, flatfield

    def subtract_data(self):
        return np.subtract(self.scan.data, self.flatfield[np.newaxis, :])

    def divide_data(self):
        return np.divide(self.scan.data, self.flatfield[np.newaxis, :] + 1)

    def save(self):
        outfile = self.scan._create_outfile()
        self.scan._save_parameters(outfile)
        self.scan._save_data(outfile)
        correct_group = outfile.create_group('corrected_data')
        correct_group.create_dataset('flatfield', data=self.flatfield)
        correct_group.create_dataset('divided_data', data=self.divide_data(), compression='gzip')
        correct_group.create_dataset('sibtract_data', data=self.subtract_data(), compression='gzip')
        outfile.close()

class StepScan1D(Scan):
    prefix, scan_num, fast_size, fast_crds, data = None, None, None, None, None

    def __init__(self, prefix, scan_num):
        self.prefix, self.scan_num = prefix, scan_num
        self.fast_crds, self.fast_size = utils.coordinates(self.command)

class Scan2D(Scan, metaclass=ABCMeta):
    @abstractproperty
    def slow_size(self): pass

    @abstractproperty
    def slow_crds(self): pass

    @property
    def size(self): return (self.slow_size, self.fast_size)

    def _save_data(self, outfile):
        datagroup = outfile.create_group('data')
        datagroup.create_dataset('fast_coordinates', data=self.fast_crds)
        datagroup.create_dataset('slow_coordinates', data=self.slow_crds)
        datagroup.create_dataset('data', data=self.data, compression='gzip')

class StepScan2D(Scan2D):
    prefix, scan_num, fast_size, fast_crds, slow_size, slow_crds, data = None, None, None, None, None, None, None

    def __init__(self, prefix, scan_num):
        self.prefix, self.scan_num = prefix, scan_num
        self.fast_crds, self.fast_size, self.slow_crds, self.slow_size = utils.coordinates2d(self.command)

class FlyScan2D(Scan2D):
    prefix, scan_num, fast_size, fast_crds, slow_size, slow_crds, data = None, None, None, None, None, None, None

    def __init__(self, prefix, scan_num):
        self.prefix, self.scan_num = prefix, scan_num
        self.fast_crds, self.fast_size, self.slow_crds, self.slow_size = utils.coordinates2d(self.command)