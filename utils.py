import numpy as np, h5py, os, errno, concurrent.futures
from functools import partial
from multiprocessing import cpu_count

raw_path = "/asap3/petra3/gpfs/p06/2019/data/11006252/raw"
prefixes = {'alignment': '0001_alignment', 'opal': '0001_opal', 'b12_1': '0002_b12_1', 'b12_2': '0002_b12_2'}
mask = np.load("/gpfs/cfel/cxi/scratch/user/murrayke/Processed_Data/Petra/Jun_2019/mask/P06_mask.npy").astype(np.int)
measpath = {'scan': "scan_{0:05}", "frame": "count_{0:05}"}
masterfilepath = {'scan': "eiger4m_01/scan_{0:05}_master.h5", 'frame': "eiger4m_01/count_{0:05}_master.h5"}
nxspath = "/scan/program_name"
commandpath = "scan_command"
datapath = "/entry/data"
energypath = "scan/data/energy"
outpath = {'scan': "../results/scan_{0:05}", 'frame': "../results/count_{0:05}"}
filename = {'scan': "scan_{0:05}.h5", 'frame': "count_{0:05}.h5"}
commands = {'single_frame': ('cnt', 'ct'), 'stepscan1d': ('dscan', 'ascan'), 'stepscan2d': 'dmesh', 'flyscan2d': 'cmesh'}

def make_output_dir(path):
    try:
        os.makedirs(os.path.dirname(path))
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

def scan2d_attrs(command):
    nums = []
    for part in command.split(" "):
        try: nums.append(float(part))
        except: continue
    *coord_attrs, exp = nums
    return coord_attrs, exp

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

def data_chunk(keys, masterfilepath, full_mask):
    masterfile = h5py.File(masterfilepath, 'r')
    dataset = masterfile[datapath]
    data_list = []
    for key in keys:
        try: data_list.append(np.multiply(full_mask, dataset[key][:]))
        except KeyError: continue
    return data_list

def data(masterfilepath, fast_size):
    keys = np.sort(np.array(list(h5py.File(masterfilepath, 'r')[datapath].keys()), dtype=object))
    full_mask = np.tile(mask, (fast_size, 1, 1))
    thread_num = min(keys.size, cpu_count())
    max_workers = min(thread_num, cpu_count())
    worker = partial(data_chunk, masterfilepath=masterfilepath, full_mask=full_mask)
    data_list = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _data_chunk in executor.map(worker, np.array_split(keys, thread_num)):
            data_list.extend(_data_chunk)
    return np.concatenate(data_list, axis=0)