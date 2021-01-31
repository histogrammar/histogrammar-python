# Copyright (c) 2020 ING Wholesale Banking Advanced Analytics
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import warnings
import numpy as np


def prepare_2dgrid(hist):
    """Get lists of all unique x and y keys

    Used as input by get_2dgrid(hist).

    :param hist: input histogrammar histogram
    :return: two comma-separated lists of unique x and y keys
    """
    if hist.n_dim < 2:
        warnings.warn(
            "Input histogram only has {n} dimensions (<2). Returning empty lists.".format(
                n=hist.n_dim
            )
        )
        return [], []

    xkeys = set()
    ykeys = set()
    # SparselyBin, Categorize, IrregularlyBin, CentrallyBin
    if hasattr(hist, "bins"):
        hist_bins = dict(hist.bins)
        xkeys = xkeys.union(hist_bins.keys())
        for h in hist_bins.values():
            if hasattr(h, "bins"):
                h_bins = dict(h.bins)
                ykeys = ykeys.union(h_bins.keys())
            elif hasattr(h, "values"):
                ykeys = ykeys.union(range(len(h.values)))
    # Bin
    elif hasattr(hist, "values"):
        xkeys = xkeys.union(range(len(hist.values)))
        for h in hist.values:
            if hasattr(h, "bins"):
                h_bins = dict(h.bins)
                ykeys = ykeys.union(h_bins.keys())
            elif hasattr(h, "values"):
                ykeys = ykeys.union(range(len(h.values)))
    return sorted(xkeys), sorted(ykeys)


def set_2dgrid(hist, xkeys, ykeys):
    """Set 2d grid of first two dimenstions of input histogram

    Used as input by get_2dgrid(hist).

    :param hist: input histogrammar histogram
    :param list xkeys: list with unique x keys
    :param list ykeys: list with unique y keys
    :return: filled 2d numpy grid
    """
    grid = np.zeros((len(ykeys), len(xkeys)))

    if hist.n_dim < 2:
        warnings.warn(
            "Input histogram only has {n} dimensions (<2). Returning original grid.".format(
                n=hist.n_dim
            )
        )
        return grid

    # SparselyBin, Categorize, IrregularlyBin, CentrallyBin
    if hasattr(hist, "bins"):
        hist_bins = dict(hist.bins)
        for k, h in hist_bins.items():
            if k not in xkeys:
                continue
            i = xkeys.index(k)
            if hasattr(h, "bins"):
                h_bins = dict(h.bins)
                for l, g in h_bins.items():
                    if l not in ykeys:
                        continue
                    j = ykeys.index(l)
                    grid[j, i] = g.entries
            elif hasattr(h, "values"):
                for j, g in enumerate(h.values):
                    grid[j, i] = g.entries
    # Bin
    elif hasattr(hist, "values"):
        for i, h in enumerate(hist.values):
            if hasattr(h, "bins"):
                h_bins = dict(h.bins)
                for l, g in h_bins.items():
                    if l not in ykeys:
                        continue
                    j = ykeys.index(l)
                    grid[j, i] = g.entries
            elif hasattr(h, "values"):
                for j, g in enumerate(h.values):
                    grid[j, i] = g.entries
    return grid


def get_2dgrid(hist):
    """Get filled x,y grid of first two dimensions of input histogram

    :param hist: input histogrammar histogram
    :return: x,y,grid of first two dimenstions of input histogram
    """
    import numpy as np

    if hist.n_dim < 2:
        warnings.warn(
            "Input histogram only has {n} dimensions (<2). Returning empty grid.".format(
                n=hist.n_dim
            )
        )
        return np.zeros((0, 0))

    xkeys, ykeys = prepare_2dgrid(hist)
    grid = set_2dgrid(hist, xkeys, ykeys)

    x_labels = get_x_labels(hist, xkeys)
    y_labels = get_y_labels(hist, ykeys)

    return x_labels, y_labels, grid


def get_x_labels(hist, xkeys):
    xlabels = [str(hist._center_from_key(key)) for key in xkeys]
    return xlabels


def get_y_labels(hist, ykeys):
    # SparselyBin, Categorize, IrregularlyBin, CentrallyBin
    if hasattr(hist, "bins"):
        hist_bins = dict(hist.bins)
        h = list(hist_bins.values())[0]
    # Bin
    elif hasattr(hist, "values"):
        h = hist.values[0]
    ylabels = [str(h._center_from_key(key)) for key in ykeys]
    return ylabels


def prepare2Dsparse(sparse):
    yminBins = [v.minBin for v in sparse.bins.values() if v.minBin is not None]
    ymaxBins = [v.maxBin for v in sparse.bins.values() if v.maxBin is not None]
    if len(yminBins) > 0 and len(ymaxBins) > 0:
        yminBin = min(yminBins)
        ymaxBin = max(ymaxBins)
    else:
        yminBin = 0
        ymaxBin = 0
    sample = list(sparse.bins.values())[0]
    ynum = 1 + ymaxBin - yminBin
    ylow = yminBin * sample.binWidth + sample.origin
    yhigh = (ymaxBin + 1.0) * sample.binWidth + sample.origin
    return yminBin, ymaxBin, ynum, ylow, yhigh


def set2Dsparse(sparse, yminBin, ymaxBin, grid):
    for i, iindex in enumerate(range(sparse.minBin, sparse.maxBin + 1)):
        for j, jindex in enumerate(range(yminBin, ymaxBin + 1)):
            if iindex in sparse.bins and jindex in sparse.bins[iindex].bins:
                grid[j, i] = sparse.bins[iindex].bins[jindex].entries
    return grid
