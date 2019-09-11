import numpy as np, numba as nb, h5py, os, errno
from cv2 import cvtColor, COLOR_BGR2GRAY
from math import sqrt , cos, sin

raw_path = "/asap3/petra3/gpfs/p06/2019/data/11006252/raw"
prefixes = {'alignment': '0001_alignment', 'opal': '0001_opal', 'b12_1': '0002_b12_1', 'b12_2': '0002_b12_2', 'imaging': '0003_imaging1'}
hotmask = np.load(os.path.join(os.path.dirname(__file__), "P06_mask.npy"))
measpath = {'scan': "scan_{0:05d}", "frame": "count_{0:05d}"}
datafolder = "eiger4m_01"
nxspath = "/scan/program_name"
commandpath = "scan_command"
datapath = "entry/data/data"
energypath = "scan/data/energy"
outpath = {'scan': "../results/scan_{0:05d}", 'frame': "../results/count_{0:05d}"}
filename = {'scan': "scan_{0:s}_{1:05d}.{2:s}", 'frame': "count_{0:s}_{1:05d}.{2:s}"}
datafilename = {'scan': 'scan_{0:05d}_data_{1:06d}.h5', 'frame': 'count_{0:05d}_data_{1:06d}.h5'}
commands = {'single_frame': ('cnt', 'ct'), 'scan1d': ('dscan', 'ascan'), 'scan2d': ('dmesh', 'cmesh')}
mask = {107: np.load(os.path.join(os.path.dirname(__file__), '107_mask.npy')), 135: np.load(os.path.join(os.path.dirname(__file__), '135_mask.npy'))}
zero = {107: np.array([1480, 1155]), 135: np.array([1470, 1710])}
linelens = {107: 25, 135: 15, 133: 20}
det_dist = {'alignment': 0.9, 'imaging': 1.46}

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
    fast_crds = np.linspace(nums[3], nums[4], int(nums[5]) + 1, endpoint=True)
    slow_crds = np.linspace(nums[0], nums[1], int(nums[2]) + 1, endpoint=True)
    return fast_crds, fast_crds.size, slow_crds, slow_crds.size

def arraytoimg(array):
    img = np.tile((array / array.max() * 255).astype(np.uint8)[..., np.newaxis], (1, 1, 3))
    return cvtColor(img, COLOR_BGR2GRAY)

def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis / sqrt(np.dot(axis, axis))
    a = cos(theta / 2.0)
    b, c, d = -axis * sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

@nb.njit(nb.types.UniTuple(nb.float64[:], 3)(nb.float64[:, :], nb.float64[:], nb.float64[:], nb.float64[:]), fastmath=True)
def rotate(m, xs, ys, zs):
    XS = np.empty(xs.shape, dtype=np.float64)
    YS = np.empty(xs.shape, dtype=np.float64)
    ZS = np.empty(xs.shape, dtype=np.float64)
    for i in range(xs.size):
            XS[i] = m[0,0] * xs[i] + m[0,1] * ys[i] + m[0,2] * zs[i]
            YS[i] = m[1,0] * xs[i] + m[1,1] * ys[i] + m[1,2] * zs[i]
            ZS[i] = m[2,0] * xs[i] + m[2,1] * ys[i] + m[2,2] * zs[i]
    return XS, YS, ZS