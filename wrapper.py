import os, numpy as np, h5py, sys, concurrent.futures
from . import utils
from abc import ABCMeta, abstractmethod, abstractproperty
from multiprocessing import cpu_count

try:
    from PyQt5 import QtCore, QtGui
except ImportError:
    from PyQt4 import QtCore, QtGui

class Measurement(metaclass=ABCMeta):
    @abstractproperty
    def mode(self): pass

    @abstractproperty
    def scan_num(self): pass

    @abstractproperty
    def prefix(self): pass

    @abstractmethod
    def _save_data(self, outfile, data=None): pass

    @abstractmethod
    def size(self): pass

    @abstractmethod
    def data_chunk(self, paths): pass

    @abstractmethod
    def _save_parameters(self, outfile): pass

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
        parts = self.command.split(" ")
        try: exposure = float(parts[-1])
        except: exposure = float(parts[-2])
        return exposure

    def data(self):
        _paths = np.sort(np.array([os.path.join(self.datapath, filename) for filename in os.listdir(self.datapath) if not filename.endswith('master.h5')], dtype=object))
        _thread_num = min(_paths.size, cpu_count())
        _data_list = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for _data_chunk in executor.map(self.data_chunk, np.array_split(_paths, _thread_num)):
                if not _data_chunk is None:
                    _data_list.append(_data_chunk)
        return np.concatenate(_data_list, axis=0)

    def show(self, data=None, levels=(0, 100)):
        _app = QtGui.QApplication([])
        _viewer = utils.Viewer(data=self.data() if data is None else data, label=self.path, levels=levels)
        if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
            _app.exec_()

    def _create_outfile(self):
        self.outpath = os.path.join(os.path.dirname(__file__), utils.outpath[self.mode].format(self.scan_num))
        self.filename = utils.filename[self.mode].format(self.scan_num)
        utils.make_output_dir(self.outpath)
        return h5py.File(os.path.join(self.outpath, self.filename), 'w')
    
    def _save_data(self, outfile, data=None):
        datagroup = outfile.create_group('data')
        datagroup.create_dataset('data', data=self.data() if data is None else data)

    def save(self):
        outfile = self._create_outfile()
        self._save_parameters(outfile)
        self._save_data(outfile)
        outfile.close()

class FullMeasurement(Measurement, metaclass=ABCMeta):
    def data_chunk(self, paths):
        data_list = []
        for path in paths:
            with h5py.File(path, 'r') as datafile:
                try: data_list.append(np.multiply(utils.mask, np.mean(datafile[utils.datapath][:], axis=0)))
                except KeyError: continue
        return None if not data_list else np.stack(data_list, axis=0)

    def _save_parameters(self, outfile):
        arggroup = outfile.create_group('arguments')
        arggroup.create_dataset('experiment',data=self.prefix)
        arggroup.create_dataset('scan mode', data=self.__class__.__name__)
        arggroup.create_dataset('scan number', data=self.scan_num)
        arggroup.create_dataset('raw path', data=self.path)
        arggroup.create_dataset('command', data=self.command)
        arggroup.create_dataset('energy', data=self.energy)
        arggroup.create_dataset('exposure', data=self.exposure)

class CropMeasurement(Measurement, metaclass=ABCMeta):
    @abstractproperty
    def roi(self): pass

    @property
    def roislice(self): return (slice(self.roi[0], self.roi[1]), slice(self.roi[2], self.roi[3]))

    def data_chunk(self, paths):
        data_list = []
        for path in paths:
            with h5py.File(path, 'r') as datafile:
                try: data_list.append(np.multiply(utils.mask[self.roislice], np.mean(datafile[utils.datapath][(slice(None),) + self.roislice], axis=0)))
                except KeyError: continue
        return None if not data_list else np.stack(data_list, axis=0)

    def _save_parameters(self, outfile):
        arggroup = outfile.create_group('arguments')
        arggroup.create_dataset('experiment',data=self.prefix)
        arggroup.create_dataset('scan mode', data=self.__class__.__name__)
        arggroup.create_dataset('scan number', data=self.scan_num)
        arggroup.create_dataset('roi', data=self.roi)
        arggroup.create_dataset('raw path', data=self.path)
        arggroup.create_dataset('command', data=self.command)
        arggroup.create_dataset('energy', data=self.energy)
        arggroup.create_dataset('exposure', data=self.exposure)

class FullFrame(FullMeasurement):
    mode = 'frame'
    prefix, scan_num = None, None

    def size(self): return (1,)

    def __init__(self, prefix, scan_num):
        self.prefix, self.scan_num = prefix, scan_num

class CropFrame(CropMeasurement):
    mode = 'frame'
    prefix, scan_num, roi = None, None, None

    def size(self): return (1,)

    def __init__(self, prefix, scan_num, roi):
        self.prefix, self.scan_num, self.roi = prefix, scan_num, roi
  
class ScanFactory(object):
    def __init__(self, prefix, scan_num):
        self.prefix, self.scan_num = prefix, scan_num
        self.path = os.path.join(os.path.join(utils.raw_path, utils.prefixes[self.prefix], utils.measpath['scan'].format(self.scan_num)))
        self.command = utils.scan_command(self.path + '.nxs')

    def OpenFull(self):
        if self.command.startswith(utils.commands['scan1d']):
            return FullScan1D(self.prefix, self.scan_num)
        elif self.command.startswith(utils.commands['scan2d']):
            return FullScan2D(self.prefix, self.scan_num)
        else:
            raise ValueError('Unknown scan type')

    def OpenCrop(self, roi):
        if self.command.startswith(utils.commands['scan1d']):
            return CropScan1D(self.prefix, self.scan_num, roi)
        elif self.command.startswith(utils.commands['scan2d']):
            return CropScan2D(self.prefix, self.scan_num, roi)
        else:
            raise ValueError('Unknown scan type')

class FullScan(FullMeasurement, metaclass=ABCMeta):
    mode = 'scan'

    @abstractproperty
    def fast_size(self): pass

    @abstractproperty
    def fast_crds(self): pass

    @property
    def size(self): return (self.fast_size,)

    def correct(self, bg_num):
        bg_scan = ScanFactory(self.prefix, bg_num).OpenFull()
        flatfield = np.mean(bg_scan.data(), axis=0)
        return CorrectedScan(self, flatfield)

    def _save_data(self, outfile, data=None):
        datagroup = outfile.create_group('data')
        datagroup.create_dataset('fast_coordinates', data=self.fast_crds)
        datagroup.create_dataset('data', data=self.data() if data is None else data, compression='gzip')

class CropScan(CropMeasurement, metaclass=ABCMeta):
    mode = 'scan'

    @abstractproperty
    def fast_size(self): pass

    @abstractproperty
    def fast_crds(self): pass

    @property
    def size(self): return (self.fast_size,)

    def correct(self, bg_num):
        bg_scan = ScanFactory(self.prefix, bg_num).OpenCrop(self.roi)
        flatfield = np.mean(bg_scan.data(), axis=0)
        return CorrectedScan(self, flatfield)

    def _save_data(self, outfile, data=None):
        datagroup = outfile.create_group('data')
        datagroup.create_dataset('fast_coordinates', data=self.fast_crds)
        datagroup.create_dataset('data', data=self.data() if data is None else data, compression='gzip')

class CorrectedScan(object):
    def __init__(self, scan, flatfield):
        self.scan, self.flatfield = scan, flatfield

    def _create_outfile(self):
        self.outpath = os.path.join(os.path.dirname(__file__), utils.outpath[self.scan.mode].format(self.scan.scan_num))
        self.filename = utils.filename_corrected[self.scan.mode].format(self.scan.scan_num)
        utils.make_output_dir(self.outpath)
        return h5py.File(os.path.join(self.outpath, self.filename), 'w')

    def subtracted_data(self, data=None):
        return np.subtract(self.scan.data() if data is None else data, self.flatfield[np.newaxis, :])

    def divided_data(self, data=None):
        return np.divide(self.scan.data() if data is None else data, self.flatfield[np.newaxis, :] + 1)

    def show_divided(self, data=None, levels=(0, 100)):
        self.scan.show(data=self.divided_data(data=data), levels=levels)

    def show_subtracted(self, data=None, levels=(0, 100)):
        self.scan.show(data=self.subtracted_data(data=data), levels=levels)

    def save(self):
        data = self.scan.data()
        outfile = self._create_outfile()
        self.scan._save_parameters(outfile)
        self.scan._save_data(outfile, data=data)
        correct_group = outfile.create_group('corrected_data')
        correct_group.create_dataset('flatfield', data=self.flatfield)
        correct_group.create_dataset('divided_data', data=self.divided_data(data=data), compression='gzip')
        correct_group.create_dataset('subtracted_data', data=self.subtracted_data(data=data), compression='gzip')
        outfile.close()

class FullScan1D(FullScan):
    prefix, scan_num, fast_size, fast_crds = None, None, None, None

    def __init__(self, prefix, scan_num):
        self.prefix, self.scan_num = prefix, scan_num
        self.fast_crds, self.fast_size = utils.coordinates(self.command)

class FullScan2D(FullScan):
    prefix, scan_num, fast_size, fast_crds = None, None, None, None

    def __init__(self, prefix, scan_num):
        self.prefix, self.scan_num = prefix, scan_num
        self.fast_crds, self.fast_size, self.slow_crds, self.slow_size = utils.coordinates2d(self.command)

    @property
    def size(self): return (self.slow_size, self.fast_size)

    def _save_data(self, outfile, data=None):
        datagroup = outfile.create_group('data')
        datagroup.create_dataset('fast_coordinates', data=self.fast_crds)
        datagroup.create_dataset('slow_coordinates', data=self.slow_crds)
        datagroup.create_dataset('data', data=self.data() if data is None else data, compression='gzip')

class CropScan1D(CropScan):
    prefix, scan_num, fast_size, fast_crds, roi = None, None, None, None, None

    def __init__(self, prefix, scan_num, roi):
        self.prefix, self.scan_num, self.roi = prefix, scan_num, roi
        self.fast_crds, self.fast_size = utils.coordinates(self.command)

class CropScan2D(CropScan):
    prefix, scan_num, fast_size, fast_crds, roi = None, None, None, None, None

    def __init__(self, prefix, scan_num, roi):
        self.prefix, self.scan_num, self.roi = prefix, scan_num, roi
        self.fast_crds, self.fast_size, self.slow_crds, self.slow_size = utils.coordinates2d(self.command)

    @property
    def size(self): return (self.slow_size, self.fast_size)

    def _save_data(self, outfile, data=None):
        datagroup = outfile.create_group('data')
        datagroup.create_dataset('fast_coordinates', data=self.fast_crds)
        datagroup.create_dataset('slow_coordinates', data=self.slow_crds)
        datagroup.create_dataset('data', data=self.data() if data is None else data, compression='gzip')