import numpy as np


def calc_bmids(bin_edges):
    bmids = (bin_edges[1:] + bin_edges[:-1]) * 0.5
    underflow = 2 * bin_edges[0] - bin_edges[1]
    overflow =  2 * bin_edges[-1] - bin_edges[-2]
    bmids = np.r_[underflow, bmids, overflow]
    return bmids


def calc_bdiff(bin_edges):
    bdiff = np.diff(bin_edges)
    return np.r_[bdiff[0], bdiff, bdiff[-1]]