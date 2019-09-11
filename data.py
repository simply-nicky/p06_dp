import numpy as np, numba as nb, concurrent.futures
from . import utils
from math import sqrt
from itertools import accumulate
from multiprocessing import cpu_count
from skimage.transform import probabilistic_hough_line
from skimage.draw import line_aa
from cv2 import createLineSegmentDetector
from abc import ABCMeta, abstractmethod

class LineDetector(object, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def _refiner(lines, angles, rs, taus, drtau, drn): pass
    
    @abstractmethod
    def _detector(self, frame): pass

    def detectFrameRaw(self, frame):
        return np.array([[[x0, y0], [x1, y1]] for (x0, y0), (x1, y1) in self._detector(frame)])

    def detectFrame(self, frame, zero, drtau, drn):
        lines = FrameStreaks(self.detectFrameRaw(frame), zero)
        return self._refiner(lines.lines, lines.angles, lines.radii, lines.taus, drtau, drn)

    def detectScanRaw(self, data): return [self.detectFrameRaw(frame) for frame in data]

    def detectScan(self, data, zero, drtau, drn): return ScanStreaks([self.detectFrame(frame, zero, drtau, drn) for frame in data])

class HoughLineDetector(LineDetector):
    def __init__(self, threshold, line_length, line_gap, dth):
        self.trhd, self.ll, self.lg = threshold, line_length, line_gap
        self.thetas = np.linspace(-np.pi / 2, np.pi / 2, int(np.pi / dth), endpoint=True)

    @staticmethod
    @nb.njit(nb.int64[:, :, :](nb.int64[:, :, :], nb.float64[:], nb.float64[:], nb.float64[:, :], nb.float64, nb.float64))
    def _refiner(lines, angles, rs, taus, drtau, drn):
        newlines = np.empty(lines.shape, dtype=np.int64)
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

    def _detector(self, frame):
        return probabilistic_hough_line(frame, threshold=self.trhd, line_length=self.ll, line_gap=self.lg, theta=self.thetas)

class LineSegmentDetector(LineDetector):
    def __init__(self, scale=0.8, sigma_scale=0.6, log_eps=0):
        self.detector = createLineSegmentDetector(_scale=scale, _sigma_scale=sigma_scale, _log_eps=log_eps)
    
    @staticmethod
    @nb.njit(nb.float32[:, :, :](nb.float32[:, :, :], nb.float64[:], nb.float64[:], nb.float64[:, :], nb.float64, nb.float64))
    def _refiner(lines, angles, rs, taus, drtau, drn):
        lsdlines = np.empty(lines.shape, dtype=np.float32)
        idxs = []
        count = 0
        for idx in range(lines.shape[0]):
            if idx not in idxs:
                newline = np.empty((2, 2), dtype=np.float32)
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
                        if proj20 < proj21:
                            newline[0] = (lines[idx2, 0] + newline[0]) / 2
                            newline[1] = (lines[idx2, 1] + newline[1]) / 2
                        else:
                            newline[0] = (lines[idx2, 1] + newline[0]) / 2
                            newline[1] = (lines[idx2, 0] + newline[1]) / 2
                lsdlines[count] = newline
                count += 1
        return lsdlines[:count]

    def _detector(self, frame):
        cap = np.mean(frame[frame != 0]) + np.std(frame[frame != 0])
        img = utils.arraytoimg(np.clip(frame, 0, cap))
        return self.detector.detect(img)[0][:, 0].reshape((-1, 2, 2))

class FrameStreaks(object):
    def __init__(self, lines, zero):
        self.lines, self.zero = lines, zero
        self.dlines = lines - zero
        self.pts = self.dlines.mean(axis=1)

    @property
    def size(self): return self.lines.shape[0]

    @property
    def xs(self): return self.pts[:, 0]

    @property
    def ys(self): return self.pts[:, 1]

    @property
    def radii(self): return np.sqrt(self.xs**2 + self.ys**2)

    @property
    def angles(self): return np.arctan2(self.ys, self.xs)

    @property
    def taus(self):
        taus = (self.lines[:, 1] - self.lines[:, 0])
        return taus / np.sqrt(taus[:,0]**2 + taus[:,1]**2)[:, np.newaxis]

    def __iter__(self):
        for line in self.lines: yield line

    def indexpoints(self):
        ts = self.dlines[:, 0, 1] * self.taus[:, 0] - self.dlines[:, 0, 0] * self.taus[:, 1]
        return ts * np.stack((-self.taus[:, 1], self.taus[:, 0]), axis=1)

    def intensities(self, frame):
        ints = []
        for line in iter(self):
            rr, cc, val = line_aa(line[0, 1], line[0, 0], line[1, 1], line[1, 0])
            ints.append((frame[rr, cc] * val).sum())
        return np.array(ints)

class ScanStreaks(object):
    def __init__(self, streakslist):
        self.strkslist = streakslist

    @property
    def shapes(self): return np.array(list(accumulate([self.strkslist.size for strks in self.strkslist], lambda x, y: x + y)))

    @property
    def zero(self): return self.__getitem__(0).zero

    @staticmethod
    @nb.njit(nb.float64[:,:](nb.float64[:,:],  nb.int64[:], nb.float64))
    def _refiner(qs, shapes, dk):
        b = len(shapes)
        out = np.empty(qs.shape, dtype=np.float64)
        idxs = []; jj = 0; count = 0
        for i in range(shapes[b - 2]):
            if i == shapes[jj]: jj += 1
            if i in idxs: continue
            qslist = []
            for j in range(shapes[jj], shapes[jj + 1]):
                if sqrt((qs[i,0] - qs[j,0])**2 + (qs[i,1] - qs[j,1])**2 + (qs[i,2] - qs[j,2])**2) < dk:
                    qslist.append(qs[i]); idxs.append(i)
                    break
            else:
                out[count] = qs[i]; count += 1
                continue
            for k in range(jj, b - 1):
                skip = True; q = qslist[-1]
                for l in range(shapes[k], shapes[k + 1]):
                    if sqrt((q[0] - qs[l,0])**2 + (q[1] - qs[l,1])**2 + (q[2] - qs[l,2])**2) < dk:
                        skip = False; qslist.append(qs[l]); idxs.append(l)
                if skip: break
            qsum = np.copy(qslist[0])
            for q in qslist[1:]:
                qsum += q
            out[count] = qsum / len(qslist); count += 1
        return out[:count]

    def __getitem__(self, index): return self.strkslist[index]

    def __iter__(self):
        for strks in self.strkslist: yield strks

    def qs(self, axis, thetas, pixsize, detdist):
        qslist = []
        for strks, theta in zip(iter(self), thetas):
            kxs = np.arctan(strks.radii / detdist) * np.cos(strks.angles)
            kys = np.arctan(strks.radii / detdist) * np.sin(strks.angles)
            rotm = utils.rotation_matrix(axis, theta)
            qxs, qys, qzs = utils.rotate(rotm, kxs, kys, np.sqrt(1 - kxs**2 - kys**2) - 1)
            qslist.append(np.stack((qxs, qys, qzs), axis=1))
        return ReciprocalPeaks(np.concatenate(qslist))

    def refined_qs(self, axis, thetas, pixsize, detdist, dk):
        qs = self.qs(axis, thetas, pixsize, detdist)
        return ReciprocalPeaks(self._refiner(qs, self.shapes, dk))

    def save(self, data, outfile):
        linesgroup = outfile.create_group('bragg_lines')
        intsgroup = outfile.create_group('bragg_intensities')
        for idx, (streaks, frame) in enumerate(zip(iter(self), data)):
            linesgroup.create_dataset(str(idx), data=streaks.lines)
            intsgroup.create_dataset(str(idx), data=streaks.intensities(frame))

class ReciprocalPeaks(object):
    def __init__(self, qs):
        self.qs = qs

    @staticmethod
    @nb.njit(nb.uint64[:, :, :](nb.float64[:,:], nb.float64, nb.int64), parallel=True)
    def __corgrid_func(qs, qmax, size):
        a = qs.shape[0]
        corgrid = np.zeros((size, size, size), dtype=np.uint64)
        ks = np.linspace(-qmax, qmax, size)
        for i in nb.prange(a):
            for j in range(i + 1, a):
                dk = qs[i] - qs[j]
                if abs(dk[0]) < qmax and abs(dk[1]) < qmax and abs(dk[2]) < qmax:
                    ii = np.searchsorted(ks, dk[0])
                    jj = np.searchsorted(ks, dk[1])
                    kk = np.searchsorted(ks, dk[2])
                    corgrid[ii, jj, kk] += 1
        return corgrid

    @staticmethod
    @nb.njit(nb.float64[:, :](nb.float64[:,:], nb.float64), parallel=True)
    def __cor_func(qs, qmax):
        a = qs.shape[0]
        cor = np.empty((int(a * (a - 1) / 2), 3), dtype=np.float64)
        count = 0
        for i in nb.prange(a):
            for j in range(i + 1, a):
                dk = qs[i] - qs[j]
                if abs(dk[0]) < qmax and abs(dk[1]) < qmax and abs(dk[2]) < qmax:
                    cor[count] = dk
                    count += 1
        return cor[:count]

    def correlation_grid(self, qmax, size):
        return self.__corgrid_func(self.qs, qmax, size)

    def correlation(self, qmax):
        return self.__cor_func(self.qs, qmax)
