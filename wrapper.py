import os, numpy as np, h5py, concurrent.futures
from . import utils
from abc import ABCMeta, abstractmethod, abstractproperty
from multiprocessing import cpu_count
from scipy import constants
from scipy.ndimage.filters import median_filter
from skimage.transform import probabilistic_hough_line

class Measurement(metaclass=ABCMeta):
    @abstractproperty
    def mode(self): pass

    @abstractproperty
    def scan_num(self): pass

    @abstractproperty
    def prefix(self): pass

    @abstractmethod
    def size(self): pass

    @abstractmethod
    def data(self): pass

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
        return utils.energy(self.nxsfilepath)

    @property
    def exposure(self):
        parts = self.command.split(" ")
        try: exposure = float(parts[-1])
        except: exposure = float(parts[-2])
        return exposure
    
    @property
    def mask(self): return utils.hotmask

    def filename(self, tag): return utils.filename[self.mode].format(tag, self.scan_num)

    def masked_data(self, data=None):
        if data is None: data = self.data()
        return self.mask * data

    def _create_outfile(self, tag):
        self.outpath = os.path.join(os.path.dirname(__file__), utils.outpath[self.mode].format(self.scan_num))
        utils.make_output_dir(self.outpath)
        return h5py.File(os.path.join(self.outpath, self.filename(tag)), 'w')
    
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

def OpenScan(prefix, scan_num):
    path = os.path.join(os.path.join(utils.raw_path, utils.prefixes[prefix], utils.measpath['scan'].format(scan_num)))
    command = utils.scan_command(path + '.nxs')
    if command.startswith(utils.commands['scan1d']):
        return Scan1D(prefix, scan_num)
    elif command.startswith(utils.commands['scan2d']):
        return Scan2D(prefix, scan_num)
    else:
        raise ValueError('Unknown scan type')

class Frame(Measurement):
    prefix, scan_num, mode = None, None, None

    def datafilename(self, framenum):
        return utils.datafilename[self.mode].format(self.scan_num, framenum)

    def size(self): return (1,)

    def __init__(self, prefix, scan_num, mode='frame'):
        self.prefix, self.scan_num, self.mode = prefix, scan_num, mode

    def data(self, framenum=1):
        return h5py.File(os.path.join(self.datapath, self.datafilename(framenum)), 'r')[utils.datapath][:].sum(axis=0, dtype=np.uint64)

    def _save_data(self, outfile):
        datagroup = outfile.create_group('data')
        datagroup.create_dataset('data', data=self.data(), compression='gzip')
        datagroup.create_dataset('mask', data=self.mask, compression='gzip')

class ABCScan(Measurement, metaclass=ABCMeta):
    mode = 'scan'

    @abstractmethod
    def data_chunk(self, paths): pass

    def data(self):
        _paths = np.sort(np.array([os.path.join(self.datapath, filename) for filename in os.listdir(self.datapath) if not filename.endswith('master.h5')], dtype=object))
        _thread_num = min(_paths.size, cpu_count())
        _data_list = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for _data_chunk in executor.map(self.data_chunk, np.array_split(_paths, _thread_num)):
                if not _data_chunk is None:
                    _data_list.append(_data_chunk)
        return np.concatenate(_data_list, axis=0)

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

    def corrected_data(self, ffnum, data=None):
        if data is None: data = self.data()
        ffscan = Frame(self.prefix, ffnum, 'scan')
        flatfield = ffscan.masked_data()
        return CorrectedData(self.masked_data(data), flatfield)

    def peaks(self, ffnum, data=None, good_frames=None):
        if data is None: data = self.data()
        if good_frames is None: good_frames = np.arange(0, data.shape[0])
        ffscan = Frame(self.prefix, ffnum, 'scan')
        flatfield = ffscan.masked_data()
        return Peaks(self.masked_data(data), flatfield, self.scan_num, good_frames)

    def _save_data(self, outfile, data=None):
        if data is None: data = self.data()
        datagroup = outfile.create_group('data')
        datagroup.create_dataset('data', data=data, compression='gzip')
        datagroup.create_dataset('mask', data=self.mask, compression='gzip')
        datagroup.create_dataset('fs_coordinates', data=self.fast_crds)

    def save_corrected(self, ffnum):
        outfile = self._create_outfile(tag='corrected')
        self._save_parameters(outfile)
        data = self.data()
        self._save_data(outfile, data)
        cordata = self.corrected_data(ffnum, data)
        cordata.save(outfile)
        outfile.close()

    def save_peaks(self, ffnum, good_frames=None):
        outfile = self._create_outfile(tag='peaks')
        self._save_parameters(outfile)
        data = self.data()
        self._save_data(outfile, data)
        peaks = self.peaks(ffnum, data, good_frames)
        peaks.save(outfile)
        outfile.close()

class CorrectedData(object):
    def __init__(self, data, flatfield):
        self.data, self.flatfield = data, flatfield

    @property
    def corrected_data(self):
        cordata = (self.data - self.flatfield[np.newaxis, :]).astype(np.int64)
        cordata[cordata < 0] = 0
        return cordata.astype(np.uint64)

    def save(self, outfile):
        correct_group = outfile.create_group('corrected_data')
        correct_group.create_dataset('flatfield', data=self.flatfield, compression='gzip')
        correct_group.create_dataset('corrected_data', data=self.corrected_data, compression='gzip')

class Peaks(object):
    def __init__(self, data, flatfield, scan_num, good_frames):
        self.data, self.flatfield = data[good_frames], flatfield
        self.mask = utils.mask.get(scan_num, np.ones(self.flatfield.shape))
        self.zero = utils.zero.get(scan_num, np.array(np.unravel_index(self.data.sum(axis=0).argmax(), self.flatfield.shape)))
        self.linelength = utils.linelens.get(scan_num, 20)

    def subtracted_data(self):
        subdata = (self.data - self.flatfield[np.newaxis, :]).astype(np.int64)
        subdata[subdata < 0] = 0
        return subdata.astype(np.uint64)

    def peaks(self, kernel_size=30, threshold=25, line_gap=5, drtau=30, drn=10):
        _subdata = self.subtracted_data()
        _background = utils.background(_subdata, self.mask, kernel_size)
        _diffdata = utils.subtract_bgd(_subdata, _background)
        _lineslist, _intslist = [], []
        for _frame, _rawframe in zip(_diffdata, _subdata):
            _lines, _ints = np.array([[[x0, y0], [x1, y1]] for (x0, y0), (x1, y1)
                                    in probabilistic_hough_line(_frame, threshold=threshold, line_length=self.linelength, line_gap=line_gap)]), []
            if _lines.any():
                _lines = utils.findlines(_lines, self.zero, drtau, drn)
                _ints = utils.peakintensity(_rawframe, _lines)
            _lineslist.append(_lines); _intslist.append(_ints)
        return _lineslist, _intslist

    def save(self, outfile, kernel_size=30, threshold=25, line_gap=5, drtau=30, drn=10):
        _lineslist, _intslist = self.peaks(kernel_size, threshold, line_gap, drtau, drn)
        _peakXPos = np.zeros((len(_lineslist), 1024), dtype=np.float32)
        _peakYPos = np.zeros((len(_lineslist), 1024), dtype=np.float32)
        _peakTotalIntensity = np.zeros((len(_lineslist), 1024), dtype=np.float32)
        _nPeaks = np.zeros((len(_lineslist),), dtype=np.int32)
        for idx, (lines, ints) in enumerate(zip(_lineslist, _intslist)):
            if lines.any():
                _peakXPos[idx, :lines.shape[0]] = lines.mean(axis=1)[:, 0]
                _peakYPos[idx, :lines.shape[0]] = lines.mean(axis=1)[:, 1]
                _peakTotalIntensity[idx, :lines.shape[0]] = ints
                _nPeaks[idx] = lines.shape[0]
        resgroup = outfile.create_group('entry_1/result_1')
        resgroup.create_dataset('peakXPosRaw', data=_peakXPos)
        resgroup.create_dataset('peakYPosRaw', data=_peakYPos)
        resgroup.create_dataset('peakTotalIntensity', data=_peakTotalIntensity)
        datagroup = outfile.create_group('peaks_data')
        datagroup.create_dataset('data', data=self.subtracted_data(), compression='gzip')
        datagroup.create_dataset('mask', data=self.mask, compression='gzip')
        datagroup.create_dataset('framesum', data=self.subtracted_data().sum(axis=0), compression='gzip')
        datagroup.create_dataset('center_coordinate', data=self.zero)
        linesgroup = datagroup.create_group('bragg_lines')
        intsgroup = datagroup.create_group('bragg_intensities')
        for idx, (lines, ints) in enumerate(zip(_lineslist, _intslist)):
            linesgroup.create_dataset(str(idx), data=lines)
            intsgroup.create_dataset(str(idx), data=ints)


class Scan1D(Scan):
    prefix, scan_num, fast_size, fast_crds = None, None, None, None

    def __init__(self, prefix, scan_num):
        self.prefix, self.scan_num = prefix, scan_num
        self.fast_crds, self.fast_size = utils.coordinates(self.command)

class Scan2D(Scan):
    prefix, scan_num, fast_size, fast_crds = None, None, None, None

    def __init__(self, prefix, scan_num):
        self.prefix, self.scan_num = prefix, scan_num
        self.fast_crds, self.fast_size, self.slow_crds, self.slow_size = utils.coordinates2d(self.command)

    @property
    def size(self): return (self.slow_size, self.fast_size)

    def _save_data(self, outfile, data=None):
        data = self.data() if data is None else data
        datagroup = outfile.create_group('data')
        datagroup.create_dataset('data', data=data, compression='gzip')
        datagroup.create_dataset('mask', data=self.mask, compression='gzip')
        datagroup.create_dataset('fs_coordinates', data=self.fast_crds)
        datagroup.create_dataset('ss_coordinates', data=self.slow_crds)

class ScanST(ABCScan):
    prefix, scan_num = None, None
    pixel_vector = np.array([7.5e-5, 7.5e-5, 0])
    unit_vector_fs = np.array([0, -1, 0])
    unit_vector_ss = np.array([-1, 0, 0])
    detector_distance = 1.46

    @property
    def x_pixel_size(self): return self.pixel_vector[0]

    @property
    def y_pixel_size(self): return self.pixel_vector[1]

    @property
    def size(self): return self.slow_size * self.fast_size

    @property
    def wavelength(self): return constants.c * constants.h / self.energy

    def __init__(self, prefix, scan_num, ff_num, good_frames=None):
        self.prefix, self.scan_num, self.good_frames = prefix, scan_num, good_frames
        self.fast_crds, self.fast_size, self.slow_crds, self.slow_size = utils.coordinates2d(self.command)
        if self.good_frames is None: self.good_frames = np.arange(0, self.size)
        self.flatfield = Frame(self.prefix, ff_num).data()

    def basis_vectors(self):
        _vec_fs = np.tile(self.pixel_vector * self.unit_vector_fs, (self.size, 1))
        _vec_ss = np.tile(self.pixel_vector * self.unit_vector_ss, (self.size, 1))
        return np.stack((_vec_fs, _vec_ss), axis=1)

    def data_chunk(self, paths):
        data_list = []
        for path in paths:
            with h5py.File(path, 'r') as datafile:
                try: data_list.append(datafile[utils.datapath][:])
                except KeyError: continue
        return None if not data_list else np.stack(data_list, axis=0)

    def translation(self):
        _x_pos = np.tile(self.fast_crds * 1e-6, self.slow_size)
        _y_pos = np.repeat(self.slow_crds * 1e-6, self.fast_size)
        _z_pos = np.zeros(self.size)
        return np.stack((_x_pos, _y_pos, _z_pos), axis=1)

    def save(self, data=None):
        if data is None: data = self.data()
        outfile = self._create_outfile(tag='st')
        detector_1 = outfile.create_group('instrument_1/detector_1')
        detector_1.create_dataset('basis_vectors', data=self.basis_vectors())
        detector_1.create_dataset('distance', data=self.detector_distance)
        detector_1.create_dataset('x_pixel_size', data=self.x_pixel_size)
        detector_1.create_dataset('y_pixel_size', data=self.y_pixel_size)
        source_1 = outfile.create_group('instrument_1/source_1')
        source_1.create_dataset('energy', data=self.energy)
        source_1.create_dataset('wavelength', data=self.wavelength)
        outfile.create_dataset('sample_3/geometry/translation', data=self.translation())
        outfile.create_dataset('frame_selector/good_frames', data=self.good_frames)
        outfile.create_dataset('mask_maker/mask', data=self.mask, compression='gzip')
        outfile.create_dataset('make_whitefield/whitefield', data=self.flatfield, compression='gzip')
        outfile.create_dataset('entry_1/data_1/data', data=data, compression='gzip')