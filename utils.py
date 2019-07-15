import numpy as np, h5py, os, errno,  pyqtgraph as pg, sys, numba as nb, concurrent.futures
from math import sqrt, atan2
from functools import partial
from multiprocessing import cpu_count
from scipy.ndimage.filters import median_filter
from skimage.draw import line_aa

try:
    from PyQt5 import QtCore, QtGui
except ImportError:
    from PyQt4 import QtCore, QtGui

raw_path = "/asap3/petra3/gpfs/p06/2019/data/11006252/raw"
prefixes = {'alignment': '0001_alignment', 'opal': '0001_opal', 'b12_1': '0002_b12_1', 'b12_2': '0002_b12_2'}
mask = np.load(os.path.join(os.path.dirname(__file__), "P06_mask.npy")).astype(np.int)
measpath = {'scan': "scan_{0:05d}", "frame": "count_{0:05d}"}
datafilepath = "eiger4m_01"
nxspath = "/scan/program_name"
commandpath = "scan_command"
datapath = "entry/data/data"
energypath = "scan/data/energy"
outpath = {'scan': "../results/scan_{0:05d}", 'frame': "../results/count_{0:05d}"}
filename = {'scan': "scan_{0:s}_{1:05d}.h5", 'frame': "count_{0:s}_{1:05d}.h5"}
commands = {'single_frame': ('cnt', 'ct'), 'scan1d': ('dscan', 'ascan'), 'scan2d': ('dmesh', 'cmesh')}
lysroi = np.array([500, 1767, 800, 2070])
b12roi = np.array([700, 2167, 600, 2070])

def make_output_dir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST: raise

def scan_command(nxsfilepath):
    command = h5py.File(nxsfilepath, 'r')[nxspath].attrs[commandpath]
    if type(command) == np.ndarray:
        command = str(command)[2:-2]
    return command

def energy(nxsfilepath):
    nxsfile = h5py.File(nxsfilepath, 'r')
    return nxsfile[energypath]

def get_attributes(command):
    nums = []
    for part in command.split(" "):
        try: nums.append(float(part))
        except: continue
    return tuple(nums[:-1])

def coordinates(command):
    start, stop, steps = get_attributes(command)
    return np.linspace(start, stop, int(steps) + 1), int(steps) + 1

def coordinates2d(command):
    start0, stop0, steps0, start1, stop1, steps1 = get_attributes(command)
    fast_crds = np.linspace(start0, stop0, int(steps0) + 1, endpoint=True)
    slow_crds = np.linspace(start1, stop1, int(steps1) + 1, endpoint=True)
    return fast_crds, int(steps0) + 1, slow_crds, int(steps1) + 1

class Viewer(QtGui.QMainWindow):
    def __init__(self, data, label, levels, parent=None, size=(640, 480)):
        super(Viewer, self).__init__(parent=parent, size=QtCore.QSize(size[0], size[1]))
        self.setWindowTitle('CBC Viewer')
        self.update_ui(data, label, levels)

        self.central_widget = QtGui.QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        self.show()

    def update_ui(self, data, label, levels):
        self.layout = QtGui.QVBoxLayout()
        _label_widget = QtGui.QLabel(label)
        _label_widget.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(_label_widget)
        _image_view = pg.ImageView()
        _image_view.setPredefinedGradient('thermal')
        _image_view.setImage(img=data, levels=levels)
        self.layout.addWidget(_image_view)

def show_data(self, data, label, levels=(0, 100)):
    _app = QtGui.QApplication([])
    _viewer = Viewer(data=data, label=label, levels=levels)
    if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
        _app.exec_()

def medfilt(data, kernel_size=30):
    bgdworker = partial(median_filter, size=(kernel_size, 1, 1))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        bgd = np.concatenate([chunk for chunk in executor.map(bgdworker, np.array_split(data, cpu_count(), axis=0))])
    return bgd

@nb.njit(nb.int64[:, :, :](nb.int64[:, :, :], nb.float64, nb.float64), fastmath=True)
def findlines(lines, dalpha, dr, zero):
    newlines = np.empty(lines.shape, dtype=np.int64)
    angles = np.empty((lines.shape[0],), dtype=np.float64)
    rs = np.empty((lines.shape[0],), dtype=np.float64)
    taus = np.empty((lines.shape[0], 2), dtype=np.float64)
    for idx in range(lines.shape[0]):
        x = (lines[idx, 0, 0] + lines[idx, 1, 0]) / 2 - zero[1]
        y = (lines[idx, 0, 1] + lines[idx, 1, 1]) / 2 - zero[0]
        tau = (lines[idx, 1] - lines[idx, 0]).astype(np.float64)
        taus[idx] = tau / sqrt(tau[0]**2 + tau[1]**2)
        angles[idx] = atan2(y, x)
        rs[idx] = sqrt(x**2 + y**2)
    for idx in range(lines.shape[0]):
        newline = np.empty((2, 2), dtype=np.float64)
        proj0 = lines[idx, 0, 0] * taus[idx, 0] + lines[idx, 0, 1] * taus[idx, 1]
        proj1 = lines[idx, 1, 0] * taus[idx, 0] + lines[idx, 1, 1] * taus[idx, 1]
        if proj0 < proj1: newline[0] = lines[idx, 0]; newline[1] = lines[idx, 1]
        else: newline[0] = lines[idx, 1]; newline[1] = lines[idx, 0]
        for idx2 in range(lines.shape[0]):
            if idx == idx2: continue
            elif abs(angles[idx] - angles[idx2]) < dalpha and abs(rs[idx] - rs[idx2]) < dr:
                proj20 = lines[idx2, 0, 0] * taus[idx, 0] + lines[idx2, 0, 1] * taus[idx, 1]
                proj21 = lines[idx2, 1, 0] * taus[idx, 0] + lines[idx2, 1, 1] * taus[idx, 1]
                if proj20 < proj0: newline[0] = lines[idx2, 0]
                elif proj20 > proj1: newline[1] = lines[idx2, 0]
                if proj21 < proj0: newline[0] = lines[idx2, 1]
                elif proj21 > proj1: newline[1] = lines[idx2, 1]           
        newlines[idx] = newline
    return newlines

def findlinesrec(lines, zero, dalpha=0.1, dr=5, order=5):
    if order > 0:
        newlines = np.unique(findlines(lines, dalpha, dr, zero), axis=0)
        return findlinesrec(newlines, dalpha, dr, order - 1)
    else:
        return lines

def peakintensity(frame, lines):
    ints = []
    for line in lines:
        rr, cc, val = line_aa(line[0, 1], line[0, 0], line[1, 1], line[1, 0])
        ints.append((frame[rr, cc] * val).sum())
    return np.array(ints)