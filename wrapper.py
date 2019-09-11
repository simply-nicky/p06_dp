import os, numpy as np, h5py, concurrent.futures
from . import utils
from .data import LineSegmentDetector
from abc import ABCMeta, abstractmethod, abstractproperty
from multiprocessing import cpu_count
from functools import partial
from scipy.ndimage.filters import median_filter
from scipy import constants

class Measurement(metaclass=ABCMeta):
    mask = utils.hotmask

    @abstractproperty
    def mode(self): pass

    @abstractproperty
    def scan_num(self): pass

    @abstractproperty
    def prefix(self): pass

    @abstractproperty
    def rawdata(self): pass

    @abstractproperty
    def size(self): pass

    @abstractmethod
    def _save_data(self, outfile): pass

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
        return os.path.join(self.path, utils.datafolder)

    @property
    def energy(self):
        return utils.energy(self.nxsfilepath)[0] * constants.e

    @property
    def exposure(self):
        parts = self.command.split(" ")
        try: exposure = float(parts[-1])
        except: exposure = float(parts[-2])
        return exposure
    
    @property
    def data(self):
        return self.mask * self.rawdata

    def filename(self, tag, ext): return utils.filename[self.mode].format(tag, self.scan_num, ext)

    def _create_outfile(self, tag, ext='h5'):
        self.outpath = os.path.join(os.path.dirname(__file__), utils.outpath[self.mode].format(self.scan_num))
        utils.make_output_dir(self.outpath)
        return h5py.File(os.path.join(self.outpath, self.filename(tag, ext)), 'w')
    
    def _save_parameters(self, outfile):
        arggroup = outfile.create_group('arguments')
        arggroup.create_dataset('experiment',data=self.prefix)
        arggroup.create_dataset('scan mode', data=self.__class__.__name__)
        arggroup.create_dataset('scan number', data=self.scan_num)
        arggroup.create_dataset('raw path', data=self.path)
        arggroup.create_dataset('command', data=self.command)
        arggroup.create_dataset('energy', data=self.energy)
        arggroup.create_dataset('exposure', data=self.exposure)

    def save_raw(self):
        outfile = self._create_outfile(tag='raw')
        self._save_parameters(outfile)
        self._save_data(outfile)
        outfile.close()

def OpenScan(prefix, scan_num, good_frames=None):
    path = os.path.join(os.path.join(utils.raw_path, utils.prefixes[prefix], utils.measpath['scan'].format(scan_num)))
    command = utils.scan_command(path + '.nxs')
    if command.startswith(utils.commands['scan1d']):
        return Scan1D(prefix, scan_num, good_frames)
    elif command.startswith(utils.commands['scan2d']):
        return Scan2D(prefix, scan_num, good_frames)
    else:
        raise ValueError('Unknown scan type')

class Frame(Measurement):
    prefix, scan_num, mode, framenum = None, None, None, 1

    def __init__(self, prefix, scan_num, mode='frame'):
        self.prefix, self.scan_num, self.mode = prefix, scan_num, mode

    @property
    def datafilename(self):
        return utils.datafilename[self.mode].format(self.scan_num, self.framenum)

    @property
    def rawdata(self):
        return h5py.File(os.path.join(self.datapath, self.datafilename), 'r')[utils.datapath][:].sum(axis=0, dtype=np.uint64)

    @property
    def size(self): return (1,)

    def _save_data(self, outfile):
        datagroup = outfile.create_group('data')
        datagroup.create_dataset('data', data=self.data, compression='gzip')
        datagroup.create_dataset('mask', data=self.mask, compression='gzip')

class ABCScan(Measurement, metaclass=ABCMeta):
    mode, _rawdata, _good_frames = 'scan', None, None

    @abstractmethod
    def data_chunk(self, paths): pass

    @property
    def good_frames(self):
        if np.any(self._good_frames): return self._good_frames
        else: return np.arange(0, self.rawdata.shape[0])

    @property
    def rawdata(self):
        if np.any(self._rawdata): return self._rawdata
        else:
            _paths = np.sort(np.array([os.path.join(self.datapath, filename) for filename in os.listdir(self.datapath) if not filename.endswith('master.h5')], dtype=object))
            _thread_num = min(_paths.size, cpu_count())
            _data_list = []
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for _data_chunk in executor.map(self.data_chunk, np.array_split(_paths, _thread_num)):
                    if np.any(_data_chunk): _data_list.append(_data_chunk)
            self._rawdata = np.concatenate(_data_list, axis=0)
            return self.rawdata

class Scan(ABCScan, metaclass=ABCMeta):
    @abstractproperty
    def fast_size(self): pass

    @abstractproperty
    def fast_crds(self): pass

    @property
    def size(self): return (self.fast_size,)

    def data_chunk(self, paths):
        data_list = []
        for path in paths:
            with h5py.File(path, 'r') as datafile:
                try: data_list.append(datafile[utils.datapath][:].sum(axis=0, dtype=np.uint64))
                except KeyError: continue
        return None if not data_list else np.stack(data_list, axis=0)

    def corrected_data(self, ffnum):
        flatfield = Frame(self.prefix, ffnum, 'scan').data
        return CorrectedData(self.data, flatfield, self.scan_num, self.good_frames)

    def _save_data(self, outfile):
        datagroup = outfile.create_group('data')
        datagroup.create_dataset('data', data=self.rawdata, compression='gzip')
        datagroup.create_dataset('mask', data=self.mask, compression='gzip')
        datagroup.create_dataset('fs_coordinates', data=self.fast_crds)

    def save_corrected(self, ffnum):
        outfile = self._create_outfile(tag='corrected')
        self._save_parameters(outfile)
        self._save_data(outfile)
        cordata = self.corrected_data(ffnum)
        cordata.save(outfile)
        outfile.close()

    def save_streaks(self, ffnum, zero, drtau, drn):
        outfile = self._create_outfile(tag='corrected')
        self._save_parameters(outfile)
        self._save_data(outfile)
        cordata = self.corrected_data(ffnum)
        cordata.save(outfile)
        streaks = LineSegmentDetector().detectScan(cordata.streaksdata, zero, drtau, drn)
        streaks.save(self.rawdata, outfile)
        outfile.close()

class Scan1D(Scan):
    prefix, scan_num, fast_size, fast_crds = None, None, None, None

    def __init__(self, prefix, scan_num, good_frames=None):
        self.prefix, self.scan_num, self._good_frames = prefix, scan_num, good_frames
        self.fast_crds, self.fast_size = utils.coordinates(self.command)

class Scan2D(Scan):
    prefix, scan_num, fast_size, fast_crds = None, None, None, None

    def __init__(self, prefix, scan_num, good_frames=None):
        self.prefix, self.scan_num, self._good_frames = prefix, scan_num, good_frames
        self.fast_crds, self.fast_size, self.slow_crds, self.slow_size = utils.coordinates2d(self.command)

    @property
    def size(self): return (self.slow_size, self.fast_size)

    def _save_data(self, outfile):
        datagroup = outfile.create_group('data')
        datagroup.create_dataset('data', data=self.rawdata, compression='gzip')
        datagroup.create_dataset('mask', data=self.mask, compression='gzip')
        datagroup.create_dataset('fs_coordinates', data=self.fast_crds)
        datagroup.create_dataset('ss_coordinates', data=self.slow_crds)

class ScanST(ABCScan):
    prefix, scan_num = None, None
    pixel_vector = np.array([7.5e-5, 7.5e-5, 0])
    unit_vector_fs = np.array([0, -1, 0])
    unit_vector_ss = np.array([-1, 0, 0])
    
    @property
    def detector_distance(self): return utils.det_dist[self.prefix]

    @property
    def x_pixel_size(self): return self.pixel_vector[0]

    @property
    def y_pixel_size(self): return self.pixel_vector[1]

    @property
    def size(self): return self.slow_size * self.fast_size

    @property
    def wavelength(self): return constants.c * constants.h / self.energy

    def __init__(self, prefix, scan_num, ff_num, good_frames=None, flip_axes=False):
        self.prefix, self.scan_num, self.good_frames, self.flip = prefix, scan_num, good_frames, flip_axes
        self.fast_crds, self.fast_size, self.slow_crds, self.slow_size = utils.coordinates2d(self.command)
        if self.good_frames is None: self.good_frames = np.arange(0, self.size)
        self.flatfield = Frame(self.prefix, ff_num, 'scan').data

    def basis_vectors(self):
        _vec_fs = np.tile(self.pixel_vector * self.unit_vector_fs, (self.size, 1))
        _vec_ss = np.tile(self.pixel_vector * self.unit_vector_ss, (self.size, 1))
        return np.stack((_vec_ss, _vec_fs), axis=1) if self.flip else np.stack((_vec_fs, _vec_ss), axis=1)

    def data_chunk(self, paths):
        data_list = []
        for path in paths:
            with h5py.File(path, 'r') as datafile:
                try: data_list.append(datafile[utils.datapath][:])
                except KeyError: continue
        return None if not data_list else np.concatenate(data_list, axis=0)

    def translation(self):
        _x_pos = np.tile(self.fast_crds * 1e-6, self.slow_size)
        _y_pos = np.repeat(self.slow_crds * 1e-6, self.fast_size)
        _z_pos = np.zeros(self.size)
        return np.stack((_x_pos, _y_pos, _z_pos), axis=1)

    def _save_data(self, outfile):
        outfile.create_dataset('frame_selector/good_frames', data=self.good_frames)
        outfile.create_dataset('mask_maker/mask', data=self.mask)
        outfile.create_dataset('make_whitefield/whitefield', data=self.flatfield)
        outfile.create_dataset('entry_1/data_1/data', data=self.data)

    def save_st(self):
        outfile = self._create_outfile(tag='st', ext='cxi')
        detector_1 = outfile.create_group('entry_1/instrument_1/detector_1')
        detector_1.create_dataset('basis_vectors', data=self.basis_vectors())
        detector_1.create_dataset('distance', data=self.detector_distance)
        detector_1.create_dataset('x_pixel_size', data=self.x_pixel_size)
        detector_1.create_dataset('y_pixel_size', data=self.y_pixel_size)
        source_1 = outfile.create_group('entry_1/instrument_1/source_1')
        source_1.create_dataset('energy', data=self.energy)
        source_1.create_dataset('wavelength', data=self.wavelength)
        outfile.create_dataset('entry_1/sample_3/geometry/translation', data=self.translation())
        self._save_data(outfile)

class CorrectedData(object):
    bgd_worker = partial(median_filter, size=(30, 1))
    bgd_filter = partial(median_filter, size=(1, 3, 3))
    feature_threshold = 10
    _subdata, _bgd, _strksdata = None, None, None

    def __init__(self, data, flatfield, scan_num, good_frames):
        self.data, self.flatfield = data[good_frames], flatfield
        self.mask = utils.mask.get(scan_num, np.ones(self.flatfield.shape))

    @property
    def subdata(self):
        if np.any(self._subdata): return self._subdata
        else:
            self._subdata = (self.data - self.flatfield[np.newaxis, :]).astype(np.int64)
            self._subdata[self.subdata < 0] = 0
            return self.subdata

    @property
    def background(self):
        if np.any(self._bgd): return self._bgd
        else:
            idx = np.where(self.mask == 1)
            filtdata = self.subdata[:, idx[0], idx[1]]
            datalist = []
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for chunk in executor.map(self.bgd_worker, np.array_split(filtdata, cpu_count(), axis=1)):
                    datalist.append(chunk)
            self._bgd = np.copy(self.subdata)
            self._bgd[:, idx[0], idx[1]] = np.concatenate(datalist, axis=1)
            return self.background

    @property
    def streaksdata(self):
        if np.any(self._strksdata): return self._strksdata
        else:
            sub = (self.subdata - self.background).astype(np.int64)
            self._strksdata = np.where(sub - self.background > self.feature_threshold, self.subdata, 0)
            with concurrent.futures.ProcessPoolExecutor() as executor:
                self._strksdata = np.concatenate([chunk for chunk in executor.map(self.bgd_filter, np.array_split(self._strksdata, cpu_count()))])
            return self.streaksdata

    def save(self, outfile):
        correct_group = outfile.create_group('corrected_data')
        correct_group.create_dataset('flatfield', data=self.flatfield, compression='gzip')
        correct_group.create_dataset('corrected_data', data=self.subdata, compression='gzip')
        correct_group.create_dataset('background', data=self.background, compression='gzip')
        correct_group.create_dataset('streaks_data', data=self.streaksdata, compression='gzip')