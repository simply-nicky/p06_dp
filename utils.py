import numpy as np, h5py, os

raw_path = "/asap3/petra3/gpfs/p06/2019/data/11006252/raw"
prefixes = {'alignment': '0001_alignment', 'opal': '0001_opal', 'b12': '0002_b12_1'}
mask = np.load("/gpfs/cfel/cxi/scratch/user/murrayke/Processed_Data/Petra/Jun_2019/mask/P06_mask.npy").astype(np.int)
measpath = {'scan': "scan_{0:05}", "frame": "count_{0:05}"}
masterfilepath = {'scan': "eiger4m_01/scan_{0:05}_master.h5", 'frame': "eiger4m_01/count_{0:05}_master.h5"}
nxspath = "/scan/program_name"
commandpath = "scan_command"
datapath = "/entry/data"
energypath = "scan/data/energy"
commands = {'single_frame': 'cnt', 'stepscan1d': 'dscan', 'stepscan2d': 'dmesh', 'flyscan2d': 'cmesh'}

def scan_command(nxsfilepath):
    nxsfile = h5py.File(nxsfilepath, 'r')
    return nxsfile[nxspath].attrs[commandpath]

def energy(nxsfilepath):
    nxsfile = h5py.File(nxsfilepath, 'r')
    return nxsfile[energypath]

def scan2d_attrs(command):
    nums = []
    for part in command.split(" "):
        try: nums.append(float(part))
        except: continue
    *coord_attrs, exp = nums
    return coord_attrs, exp

def get_attributes(command):
    nums = []
    for part in command.split(" ")[:-1]:
        try: nums.append(float(part))
        except: continue
    return tuple(nums)

def coordinates(command):
    start, stop, steps = get_attributes(command)
    return np.linspace(start, stop, int(steps) + 1), int(steps) + 1

def coordinates2d(command):
    start0, stop0, steps0, start1, stop1, steps1 = get_attributes(command)
    fast_crds = np.linspace(start0, stop0, int(steps0) + 1, endpoint=True)
    slow_crds = np.linspace(start1, stop1, int(steps1) + 1, endpoint=True)
    return fast_crds, int(steps0) + 1, slow_crds, int(steps1) + 1

def data(masterfilepath, fast_size):
    masterfile = h5py.File(masterfilepath, 'r')
    dataset = masterfile[datapath]
    full_mask = np.tile(mask, (fast_size, 1, 1))
    data = []
    for key in dataset:
        try: data.append(np.multiply(full_mask, dataset[key][:]))
        except KeyError: continue
    return np.concatenate(data)