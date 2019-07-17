import numpy as np, h5py, os, errno,  pyqtgraph as pg, sys, numba as nb, concurrent.futures
from math import sqrt, atan2
from functools import partial
from multiprocessing import cpu_count
from scipy.ndimage.filters import median_filter
from skimage.draw import line_aa

raw_path = "/asap3/petra3/gpfs/p06/2019/data/11006252/raw"
prefixes = {'alignment': '0001_alignment', 'opal': '0001_opal', 'b12_1': '0002_b12_1', 'b12_2': '0002_b12_2'}
hotmask = np.load(os.path.join(os.path.dirname(__file__), "P06_mask.npy"))
measpath = {'scan': "scan_{0:05d}", "frame": "count_{0:05d}"}
datafolder = "eiger4m_01"
nxspath = "/scan/program_name"
commandpath = "scan_command"
datapath = "entry/data/data"
energypath = "scan/data/energy"
outpath = {'scan': "../results/scan_{0:05d}", 'frame': "../results/count_{0:05d}"}
filename = {'scan': "scan_{0:s}_{1:05d}.h5", 'frame': "count_{0:s}_{1:05d}.h5"}
datafilename = {'scan': 'scan_{0:05d}_data_{1:06d}.h5', 'frame': 'count_{0:05d}_data_{1:06d}.h5'}
commands = {'single_frame': ('cnt', 'ct'), 'scan1d': ('dscan', 'ascan'), 'scan2d': ('dmesh', 'cmesh')}
mask = {'lys': np.load(os.path.join(os.path.dirname(__file__), 'lys_mask.npy')), 'b12': np.load(os.path.join(os.path.dirname(__file__), 'b12_mask.npy'))}
zero = {'lys': np.array([1010, 925]), 'b12': np.array([665, 680])}
thresholds = {'lys': np.load(os.path.join(os.path.dirname(__file__), 'lysthresholds.npy')), 'b12': np.load(os.path.join(os.path.dirname(__file__), 'b12thresholds.npy'))}
linelens = {'lys': 25, 'b12': 15}
fullroi = np.array([0, 2167, 0, 2070])

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
    nums = get_attributes(command)
    return np.linspace(nums[0], nums[1], int(nums[2]) + 1), int(nums[2]) + 1

def coordinates2d(command):
    nums = get_attributes(command)
    fast_crds = np.linspace(nums[0], nums[1], int(nums[2]) + 1, endpoint=True)
    slow_crds = np.linspace(nums[3], nums[4], int(nums[5]) + 1, endpoint=True)
    return fast_crds, fast_crds.size, slow_crds, slow_crds.size

def background(data, mask, kernel_size=30):
    idx = np.where(mask == 1)
    filtdata = data[:, idx[0], idx[1]]
    bgdworker, datalist = partial(median_filter, size=(kernel_size, 1)), []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for chunk in executor.map(bgdworker, np.array_split(filtdata, cpu_count(), axis=1)):
            datalist.append(chunk)
    resdata = np.copy(data)
    resdata[:, idx[0], idx[1]] = np.concatenate(datalist, axis=1)
    return resdata

def subtract_bgd(data, bgd, thresholds):
    filt = partial(median_filter, size=(1, 3, 3))
    sub = (data - bgd).astype(np.int32)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        res = np.where(sub - bgd > thresholds[:, np.newaxis, np.newaxis], data, 0)
        res = np.concatenate([chunk for chunk in executor.map(filt, np.array_split(res, cpu_count()))])
    return res

@nb.njit(nb.int64[:, :, :](nb.int64[:, :, :], nb.int64[:], nb.float64, nb.float64), fastmath=True)
def findlines(lines, zero, drtau, drn):
    newlines = np.empty(lines.shape, dtype=np.int64)
    angles = np.empty((lines.shape[0],), dtype=np.float64)
    rs = np.empty((lines.shape[0],), dtype=np.float64)
    taus = np.empty((lines.shape[0], 2), dtype=np.float64)
    for idx in range(lines.shape[0]):
        x = (lines[idx, 0, 0] + lines[idx, 1, 0]) / 2 - zero[0]
        y = (lines[idx, 0, 1] + lines[idx, 1, 1]) / 2 - zero[1]
        tau = (lines[idx, 1] - lines[idx, 0]).astype(np.float64)
        taus[idx] = tau / sqrt(tau[0]**2 + tau[1]**2)
        angles[idx] = atan2(y, x)
        rs[idx] = sqrt(x**2 + y**2)
    idxs = []
    count = 0
    for idx in range(lines.shape[0]):
        if idx not in idxs:
            newline = np.empty((2, 2), dtype=np.float64)
            proj0 = lines[idx, 0, 0] * taus[idx, 0] + lines[idx, 0, 1] * taus[idx, 1]
            proj1 = lines[idx, 1, 0] * taus[idx, 0] + lines[idx, 1, 1] * taus[idx, 1]
            if proj0 < proj1: newline[0] = lines[idx, 0]; newline[1] = lines[idx, 1]
            else: newline[0] = lines[idx, 1]; newline[1] = lines[idx, 0]
            for idx2 in range(lines.shape[0]):
                if idx == idx2: continue
                elif abs((angles[idx] - angles[idx2]) * rs[idx]) < drtau and abs(rs[idx] - rs[idx2]) < drn:
                    idxs.append(idx2)
                    proj20 = lines[idx2, 0, 0] * taus[idx, 0] + lines[idx2, 0, 1] * taus[idx, 1]
                    proj21 = lines[idx2, 1, 0] * taus[idx, 0] + lines[idx2, 1, 1] * taus[idx, 1]
                    if proj20 < proj0: newline[0] = lines[idx2, 0]
                    elif proj20 > proj1: newline[1] = lines[idx2, 0]
                    if proj21 < proj0: newline[0] = lines[idx2, 1]
                    elif proj21 > proj1: newline[1] = lines[idx2, 1]           
            newlines[count] = newline
            count += 1
    return newlines[:count]

def peakintensity(frame, lines):
    ints = []
    for line in lines:
        rr, cc, val = line_aa(line[0, 1], line[0, 0], line[1, 1], line[1, 0])
        ints.append((frame[rr, cc] * val).sum())
    return np.array(ints)