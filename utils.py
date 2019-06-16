import numpy as np, h5py, os, errno, concurrent.futures, pyqtgraph as pg
from functools import partial
from multiprocessing import cpu_count

try:
    from PyQt5 import QtCore, QtGui
except ImportError:
    from PyQt4 import QtCore, QtGui

raw_path = "/asap3/petra3/gpfs/p06/2019/data/11006252/raw"
prefixes = {'alignment': '0001_alignment', 'opal': '0001_opal', 'b12_1': '0002_b12_1', 'b12_2': '0002_b12_2'}
mask = np.load("/gpfs/cfel/cxi/scratch/user/murrayke/Processed_Data/Petra/Jun_2019/mask/P06_mask.npy").astype(np.int)
measpath = {'scan': "scan_{0:05}", "frame": "count_{0:05}"}
datafilepath = "eiger4m_01"
nxspath = "/scan/program_name"
commandpath = "scan_command"
datapath = "entry/data/data"
energypath = "scan/data/energy"
outpath = {'scan': "../results/scan_{0:05}", 'frame': "../results/count_{0:05}"}
filename = {'scan': "scan_{0:05}.h5", 'frame': "count_{0:05}.h5"}
filename_corrected = {'scan': "scan_corrected_{0:05}.h5", 'frame': "count_corrected_{0:05}.h5"}
commands = {'single_frame': ('cnt', 'ct'), 'stepscan1d': ('dscan', 'ascan'), 'stepscan2d': 'dmesh', 'flyscan2d': 'cmesh'}

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

def data_chunk(paths):
    data_list = []
    for path in paths:
        with h5py.File(path, 'r') as datafile:
            try: data_list.append(np.multiply(mask, np.mean(datafile[datapath][:], axis=0)))
            except KeyError: continue
    return None if not data_list else np.concatenate(data_list, axis=0) 

def data(path, fast_size):
    paths = np.sort(np.array([os.path.join(path, filename) for filename in os.listdir(path) if not filename.endswith('master.h5')], dtype=object))
    thread_num = min(paths.size, cpu_count())
    print('thread_num: {}'.format(thread_num))
    data_list = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for _data_chunk in executor.map(data_chunk, np.array_split(paths, thread_num)):
            if not _data_chunk is None:
                data_list.append(_data_chunk)
    return np.concatenate(data_list, axis=0)

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