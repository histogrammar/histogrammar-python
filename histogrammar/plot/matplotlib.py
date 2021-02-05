#!/usr/bin/env python

# Copyright 2016 Jim Pivarski
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import

# python 2/3 compatibility fixes
from histogrammar.util import xrange
from histogrammar.plot.hist_numpy import get_2dgrid, prepare2Dsparse, set2Dsparse


# 1d plotting of counts + generic 2d plotting of counts

class HistogramMethods(object):
    def plotmatplotlib(self, name=None, **kwargs):
        """
            name : title of the plot.
            kwargs :  `matplotlib.patches.Rectangle` properties.

            Returns a matplotlib.axes instance
        """
        # catch generic 2d plotting of counts
        if self.n_dim >= 2:
            return plot2dmatplotlib(self, name, **kwargs)

        # specialized 1d plotting of counts from here on
        import matplotlib.pyplot as plt
        ax = plt.gca()

        edges = self.bin_edges()
        entries = self.bin_entries()
        width = self.bin_width()

        ax.bar(edges[:-1], entries, width=width, align='edge', **kwargs)

        if name is not None:
            ax.set_title(name)
        else:
            ax.set_title(self.name)

        return ax


class SparselyHistogramMethods(object):
    def plotmatplotlib(self, name=None, **kwargs):
        """
            name : title of the plot.
            kwargs :  `matplotlib.patches.Rectangle` properties.

            Returns a matplotlib.axes instance
        """
        # catch generic 2d plotting of counts
        if self.n_dim >= 2:
            return plot2dmatplotlib(self, name, **kwargs)

        # specialized 1d plotting of counts from here on
        import matplotlib.pyplot as plt
        ax = plt.gca()

        edges = self.bin_edges()
        entries = self.bin_entries()
        width = self.bin_width()

        ax.bar(edges[:-1], entries, width=width, align='edge', **kwargs)
        ax.set_xlim(self.low, self.high)

        if name is not None:
            ax.set_title(name)
        else:
            ax.set_title(self.name)

        return ax


class IrregularlyHistogramMethods(object):
    def plotmatplotlib(self, name=None, **kwargs):
        """
            name : title of the plot.
            kwargs :  `matplotlib.patches.Rectangle` properties.

            Returns a matplotlib.axes instance
        """
        # catch generic 2d plotting of counts
        if self.n_dim >= 2:
            return plot2dmatplotlib(self, name, **kwargs)

        # specialized 1d plotting of counts from here on
        import matplotlib.pyplot as plt
        ax = plt.gca()

        edges = self.edges[1:-1]
        entries = self.bin_entries()[1:-1]
        width = self.bin_width()

        ax.bar(edges, entries, width=width, align='edge', **kwargs)

        if name is not None:
            ax.set_title(name)
        else:
            ax.set_title(self.name)

        return ax


class CentrallyHistogramMethods(object):
    def plotmatplotlib(self, name=None, **kwargs):
        """
            name : title of the plot.
            kwargs :  `matplotlib.patches.Rectangle` properties.

            Returns a matplotlib.axes instance
        """
        # catch generic 2d plotting of counts
        if self.n_dim >= 2:
            return plot2dmatplotlib(self, name, **kwargs)

        # specialized 1d plotting of counts from here on
        import matplotlib.pyplot as plt
        import numpy as np
        ax = plt.gca()

        width = kwargs.pop('width', 0.8)

        labels = self.bin_centers()
        values = self.bin_entries()
        assert len(labels) == len(values), \
            'labels and values have different array lengths: %d vs %d.' % \
            (len(labels), len(values))

        # plot histogram
        tick_pos = np.arange(len(labels)) + 0.5
        ax.bar(tick_pos, values, width=width, **kwargs)

        # set x-axis properties
        def xtick(lab):
            lab = str(lab)
            if len(lab) > 20:
                lab = lab[:17] + '...'
            return lab
        ax.set_xlim((0., float(len(labels))))
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([xtick(lab) for lab in labels], fontsize=12, rotation=90)

        # set title
        if name is not None:
            ax.set_title(name)
        else:
            ax.set_title(self.name)

        return ax


class CategorizeHistogramMethods(object):
    def plotmatplotlib(self, name=None, **kwargs):
        """
            name : title of the plot.
            kwargs :  `matplotlib.patches.Rectangle` properties.

            Returns a matplotlib.axes instance
        """
        # catch generic 2d plotting of counts
        if self.n_dim >= 2:
            return plot2dmatplotlib(self, name, **kwargs)

        # specialized 1d plotting of counts from here on
        import matplotlib.pyplot as plt
        import numpy as np
        ax = plt.gca()

        width = kwargs.pop('width', 0.8)

        labels = self.bin_labels()
        values = self.bin_entries()
        assert len(labels) == len(values), \
            'labels and values have different array lengths: %d vs %d.' % \
            (len(labels), len(values))

        # sort labels alphabetically
        idx = np.argsort(labels)
        labels = labels[idx]
        values = values[idx]

        # plot histogram
        tick_pos = np.arange(len(labels)) + 0.5
        ax.bar(tick_pos, values, width=width, **kwargs)

        # set x-axis properties
        def xtick(lab):
            lab = str(lab)
            if len(lab) > 20:
                lab = lab[:17] + '...'
            return lab
        ax.set_xlim((0., float(len(labels))))
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([xtick(lab) for lab in labels], fontsize=12, rotation=90)

        # set title
        if name is not None:
            ax.set_title(name)
        else:
            ax.set_title(self.name)

        return ax


# 1d plotting of profiles

class ProfileMethods(object):
    def plotmatplotlib(self, name=None, **kwargs):
        """ Plotting method for Bin of Average
              name : title of the plot.
              kwargs : matplotlib.collections.LineCollection properties.

            Returns a matplotlib.axes instance
        """
        import matplotlib.pyplot as plt
        ax = plt.gca()

        xranges = [self.range(x) for x in self.indexes]

        xmins = [x[0] for x in xranges]
        xmaxs = [x[1] for x in xranges]
        ax.hlines(self.meanValues, xmins, xmaxs, **kwargs)

        if name is not None:
            ax.set_title(name)
        else:
            ax.set_title(self.name)
        return ax


class SparselyProfileMethods(object):
    def plotmatplotlib(self, name=None, **kwargs):
        """ Plotting method for SparselyBin of Average
              name : title of the plot.
              kwargs : matplotlib.collections.LineCollection properties.

            Returns a matplotlib.axes instance
        """
        import matplotlib.pyplot as plt
        import numpy as np
        ax = plt.gca()

        xmins = np.arange(self.low, self.high, self.binWidth)
        xmaxs = np.arange(self.low + self.binWidth, self.high + self.binWidth, self.binWidth)

        means = np.nan*np.ones(xmaxs.shape)

        for i in xrange(self.minBin, self.maxBin + 1):
            if i in self.bins:
                means[i - self.minBin] = self.bins[i].mean
        idx = np.isfinite(means)

        ax.hlines(means[idx], xmins[idx], xmaxs[idx], **kwargs)

        if name is not None:
            ax.set_title(name)
        else:
            ax.set_title(self.name)
        return ax


class ProfileErrMethods(object):
    def plotmatplotlib(self, name=None, aspect=True, **kwargs):
        """ Plotting method for Bin of Deviate
              name : title of the plot.
              aspect :
              kwargs :  `matplotlib.collections.LineCollection` properties.

            Returns a matplotlib.axes instance
        """
        import matplotlib.pyplot as plt
        import numpy as np
        ax = plt.gca()

        bin_centers = [sum(self.range(x))/2.0 for x in self.indexes]
        xranges = [self.range(x) for x in self.indexes]
        means = self.meanValues
        variances = self.varianceValues
        num_bins = len(variances)

        xmins = [x[0] for x in xranges]
        xmaxs = [x[1] for x in xranges]
        ax.hlines(self.meanValues, xmins, xmaxs, **kwargs)

        if aspect is True:
            counts = [p.entries for p in self.values]
            ymins = [means[i] - np.sqrt(variances[i])/np.sqrt(counts[i]) for i in range(num_bins)]
            ymaxs = [means[i] + np.sqrt(variances[i])/np.sqrt(counts[i]) for i in range(num_bins)]
        else:
            ymins = [means[i] - np.sqrt(variances[i]) for i in range(num_bins)]
            ymaxs = [means[i] + np.sqrt(variances[i]) for i in range(num_bins)]
        ax.vlines(bin_centers, ymins, ymaxs, **kwargs)

        if name is not None:
            ax.set_title(name)
        else:
            ax.set_title(self.name)

        return ax


class SparselyProfileErrMethods(object):
    def plotmatplotlib(self, name=None, aspect=True, **kwargs):
        """ Plotting method for
        """
        import matplotlib.pyplot as plt
        import numpy as np
        ax = plt.gca()

        xmins = np.arange(self.low, self.high, self.binWidth)
        xmaxs = np.arange(self.low + self.binWidth, self.high + self.binWidth, self.binWidth)

        means = np.nan*np.ones(xmaxs.shape)
        variances = np.nan*np.ones(xmaxs.shape)
        counts = np.nan*np.ones(xmaxs.shape)

        for i in xrange(self.minBin, self.maxBin + 1):
            if i in self.bins:
                means[i - self.minBin] = self.bins[i].mean
                variances[i - self.minBin] = self.bins[i].variance
                counts[i - self.minBin] = self.bins[i].entries
        idx = np.isfinite(means)

        # pull out non nans
        means = means[idx]
        xmins = xmins[idx]
        xmaxs = xmaxs[idx]
        variances = variances[idx]

        ax.hlines(means, xmins, xmaxs, **kwargs)

        bin_centers = (self.binWidth/2.0) + xmins
        if aspect is True:
            ymins = [means[i] - np.sqrt(variances[i])/np.sqrt(counts[i]) for i in xrange(len(means))]
            ymaxs = [means[i] + np.sqrt(variances[i])/np.sqrt(counts[i]) for i in xrange(len(means))]
        else:
            ymins = [means[i] - np.sqrt(variances[i]) for i in xrange(len(means))]
            ymaxs = [means[i] + np.sqrt(variances[i]) for i in xrange(len(means))]

        ax.vlines(bin_centers, ymins, ymaxs, **kwargs)

        if name is not None:
            ax.set_title(name)
        else:
            ax.set_title(self.name)
        return ax


# other 1d/2d plotting

class StackedHistogramMethods(object):
    def plotmatplotlib(self, name=None, **kwargs):
        """ Plotting method for
        """
        import matplotlib.pyplot as plt
        ax = plt.gca()
        color_cycle = plt.rcParams['axes.color_cycle']
        if "color" in kwargs:
            kwargs.pop("color")

        for i, hist in enumerate(self.values):
            color = color_cycle[i]
            hist.plotmatplotlib(color=color, label=hist.name, **kwargs)
            color_cycle.append(color)

        if name is not None:
            ax.set_title(name)
        else:
            ax.set_title(self.name)
        return ax


class PartitionedHistogramMethods(object):
    def plotmatplotlib(self, name=None, **kwargs):
        """ Plotting method for
        """
        import matplotlib.pyplot as plt
        ax = plt.gca()
        color_cycle = plt.rcParams['axes.color_cycle']
        if "color" in kwargs:
            kwargs.pop("color")

        for i, hist in enumerate(self.values):
            color = color_cycle[i]
            hist.plotmatplotlib(color=color, label=hist.name, **kwargs)
            color_cycle.append(color)

        if name is not None:
            ax.set_title(name)
        else:
            ax.set_title(self.name)
        return ax


class FractionedHistogramMethods(object):
    def plotmatplotlib(self, name=None, **kwargs):
        """ Plotting method for
        """
        import matplotlib.pyplot as plt
        import numpy as np
        ax = plt.gca()

        if isinstance(self.numerator, HistogramMethods):
            fracs = [x[0].entries / float(x[1].entries) for x in zip(self.numerator.values, self.denominator.values)]
            xranges = [self.numerator.range(x) for x in self.numerator.indexes]

            xmins = [x[0] for x in xranges]
            xmaxs = [x[1] for x in xranges]
            ax.hlines(fracs, xmins, xmaxs, **kwargs)

        elif isinstance(self.numerator, SparselyHistogramMethods):
            assert self.numerator.binWidth == self.denominator.binWidth,\
                   "Fraction numerator and denominator histograms must have same binWidth."
            numerator = self.numerator
            denominator = self.denominator
            xmins = np.arange(numerator.low, numerator.high, numerator.binWidth)
            xmaxs = np.arange(
                numerator.low +
                numerator.binWidth,
                numerator.high +
                numerator.binWidth,
                numerator.binWidth)

            fracs = np.nan*np.zeros(xmaxs.shape)

            for i in xrange(denominator.minBin, denominator.maxBin + 1):
                if i in self.numerator.bins and i in self.denominator.bins:
                    fracs[i - denominator.minBin] = numerator.bins[i].entries / denominator.bins[i].entries
            idx = np.isfinite(fracs)
            ax.hlines(fracs[idx], xmins[idx], xmaxs[idx], **kwargs)

        ax.set_ylim((0.0, 1.0))
        if name is not None:
            ax.set_title(name)
        else:
            ax.set_title(self.name)
        return ax


# specialized 2d plotting of counts

class TwoDimensionallyHistogramMethods(object):
    def plotmatplotlib(self, name=None, **kwargs):
        """ Plotting method for Bin of Bin of Count
              name : title of the plot.
              kwargs: matplotlib.collections.QuadMesh properties.

            Returns a matplotlib.axes instance
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=1)

        x_ranges, y_ranges, grid = self.xy_ranges_grid()

        im = ax.pcolormesh(x_ranges, y_ranges, grid, **kwargs)
        fig.colorbar(im, ax=ax)

        ax.set_ylim(self.y_lim())
        ax.set_xlim(self.x_lim())

        if name is not None:
            ax.set_title(name)
        else:
            ax.set_title(self.name)
        return ax

    def xy_ranges_grid(self):
        """ Return x and y ranges and x,y grid
        """
        import numpy as np

        samp = self.values[0]
        x_ranges = np.unique(np.array([self.range(i) for i in self.indexes]).flatten())
        y_ranges = np.unique(np.array([samp.range(i) for i in samp.indexes]).flatten())

        grid = np.zeros((samp.num, self.num))

        for j in range(self.num):
            for i in range(samp.num):
                grid[i, j] = self.values[j].values[i].entries

        return x_ranges, y_ranges, grid

    def x_lim(self):
        """ return x low high tuble
        """
        return (self.low, self.high)

    def y_lim(self):
        """ return y low high tuble
        """
        samp = self.values[0]
        return (samp.low, samp.high)

    def project_on_x(self):
        """ project 2d histogram onto x-axis

        :returns: on x-axis projected histogram (1d)
        :rtype: histogrammar.Bin
        """
        from histogrammar import Bin, Count

        h_x = Bin(num=self.num, low=self.low, high=self.high, quantity=self.quantity, value=Count())
        # loop over all counters and integrate over y (=j)
        for i, bi in enumerate(self.values):
            h_x.values[i].entries += sum(bj.entries for bj in bi.values)
        return h_x

    def project_on_y(self):
        """ project 2d histogram onto y-axis

        :returns: on y-axis projected histogram (1d)
        :rtype: histogrammar.Bin
        """
        from histogrammar import Bin, Count

        ybin = self.values[0]
        h_y = Bin(num=ybin.num, low=ybin.low, high=ybin.high, quantity=ybin.quantity, value=Count())
        # loop over all counters and integrate over x (=i)
        for bi in self.values:
            for j, bj in enumerate(bi.values):
                h_y.values[j].entries += bj.entries
        return h_y


class SparselyTwoDimensionallyHistogramMethods(object):
    def plotmatplotlib(self, name=None, **kwargs):
        """ Plotting method for SparselyBin of SparselyBin of Count
              name : title of the plot.
              kwargs: matplotlib.collections.QuadMesh properties.

            Returns a matplotlib.axes instance
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=1)

        x_ranges, y_ranges, grid = self.xy_ranges_grid()

        im = ax.pcolormesh(x_ranges, y_ranges, grid, **kwargs)
        fig.colorbar(im, ax=ax)

        ax.set_ylim(self.y_lim())
        ax.set_xlim(self.x_lim())

        if name is not None:
            ax.set_title(name)
        else:
            ax.set_title(self.name)
        return ax

    def xy_ranges_grid(self):
        """ Return x and y ranges and x,y grid
        """
        import numpy as np

        yminBin, ymaxBin, ynum, ylow, yhigh = prepare2Dsparse(self)

        xbinWidth = self.binWidth
        try:
            ykey = list(self.bins.keys())[0]
        except BaseException:
            raise KeyError('SparselyBin 2d hist is not filled.')
        ybinWidth = self.bins[ykey].binWidth

        xmaxBin = max(self.bins.keys())
        xminBin = min(self.bins.keys())
        xnum = 1 + xmaxBin - xminBin
        xlow = xminBin * self.binWidth + self.origin
        xhigh = (xmaxBin + 1) * self.binWidth + self.origin

        grid = set2Dsparse(self, yminBin, ymaxBin, np.zeros((ynum, xnum)))

        x_ranges = np.arange(xlow, xhigh + xbinWidth, xbinWidth)
        y_ranges = np.arange(ylow, yhigh + ybinWidth, ybinWidth)

        return x_ranges, y_ranges, grid

    def x_lim(self):
        """ return x low high tuble
        """
        xmaxBin = max(self.bins.keys())
        xminBin = min(self.bins.keys())
        xlow = xminBin * self.binWidth + self.origin
        xhigh = (xmaxBin + 1) * self.binWidth + self.origin
        return (xlow, xhigh)

    def y_lim(self):
        """ return y low high tuble
        """
        yminBin, ymaxBin, ynum, ylow, yhigh = prepare2Dsparse(self)
        return (ylow, yhigh)

    def project_on_x(self):
        """ project 2d sparselybin histogram onto x-axis

        :returns: on x-axis projected histogram (1d)
        :rtype: histogrammar.SparselyBin
        """
        from histogrammar import SparselyBin, Count

        h_x = SparselyBin(binWidth=self.binWidth, origin=self.origin,
                          quantity=self.quantity, value=Count())
        # loop over all counters and integrate over y (=j)
        for i in self.bins:
            bi = self.bins[i]
            h_x.bins[i] = Count.ed(sum(bi.bins[j].entries for j in bi.bins))
        return h_x

    def project_on_y(self):
        """ project 2d sparselybin histogram onto y-axis

        :returns: on y-axis projected histogram (1d)
        :rtype: histogrammar.SparselyBin
        """
        from histogrammar import SparselyBin, Count

        try:
            ykey = list(self.bins.keys())[0]
        except BaseException:
            raise KeyError('SparselyBin 2d hist is not filled. Cannot project on y-axis.')
        ybin = self.bins[ykey]
        h_y = SparselyBin(binWidth=ybin.binWidth, origin=ybin.origin,
                          quantity=ybin.quantity, value=Count())
        # loop over all counters and integrate over x (=i)
        for i in self.bins:
            bi = self.bins[i]
            for j in bi.bins:
                if j not in h_y.bins:
                    h_y.bins[j] = Count()
                h_y.bins[j].entries += bi.bins[j].entries
        return h_y


class IrregularlyTwoDimensionallyHistogramMethods(object):
    def plotmatplotlib(self, name=None, **kwargs):
        """ Plotting method for Bin of Bin of Count
              name : title of the plot.
              kwargs: matplotlib.collections.QuadMesh properties.

            Returns a matplotlib.axes instance
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=1)

        x_ranges, y_ranges, grid = self.xy_ranges_grid()

        im = ax.pcolormesh(x_ranges, y_ranges, grid, **kwargs)
        fig.colorbar(im, ax=ax)

        ax.set_ylim(self.y_lim())
        ax.set_xlim(self.x_lim())

        if name is not None:
            ax.set_title(name)
        else:
            ax.set_title(self.name)
        return ax

    def xy_ranges_grid(self):
        """ Return x and y ranges and x,y grid
        """
        import numpy as np

        samp = self.bins[0][1]
        x_ranges = self.bin_edges()[1:-1]  # cut underflow and overflow bins
        y_ranges = samp.bin_edges()[1:-1]

        grid = np.zeros((samp.n_bins - 2, self.n_bins - 2))

        for j in range(1, self.n_bins - 1):
            for i in range(1, samp.n_bins - 1):
                grid[i-1, j-1] = (self.bins[j][1]).bins[i][1].entries

        return x_ranges, y_ranges, grid

    def x_lim(self):
        """ return x low high tuble
        """
        # cut underflow and overflow bins
        x_ranges = self.bin_edges()[1:-1]
        return (x_ranges[0], x_ranges[-1])

    def y_lim(self):
        """ return y low high tuble
        """
        # cut underflow and overflow bins
        samp = self.bins[0][1]
        y_ranges = samp.bin_edges()[1:-1]
        return (y_ranges[0], y_ranges[-1])

    def project_on_x(self):
        """ project 2d histogram onto x-axis

        :returns: on x-axis projected histogram (1d)
        :rtype: histogrammar.Bin
        """
        from histogrammar import IrregularlyBin, Count

        h_x = IrregularlyBin(edges=self.edges[1:], quantity=self.quantity, value=Count())
        # loop over all counters and integrate over y (=j)
        for i, bi in enumerate(self.bins):
            h_x.bins[i][1].entries += sum(bj.entries for _, bj in bi[1].bins)
        return h_x

    def project_on_y(self):
        """ project 2d histogram onto y-axis

        :returns: on y-axis projected histogram (1d)
        :rtype: histogrammar.Bin
        """
        from histogrammar import IrregularlyBin, Count

        ybin = self.bins[0][1]
        h_y = IrregularlyBin(edges=ybin.edges[1:], quantity=ybin.quantity, value=Count())
        # loop over all counters and integrate over x (=i)
        for _, bi in self.bins:
            for j, bj in enumerate(bi.bins):
                h_y.bins[j][1].entries += bj[1].entries
        return h_y


# generic 2d plotting function of counts

def plot2dmatplotlib(self, name=None, **kwargs):
    """ General plotting method for 2d Bin/SparselyBin/Categorize/CentrallyBin/IrregularlyBin of Count
          name : title of the plot.
          kwargs: matplotlib.collections.QuadMesh properties.

        Returns a matplotlib.axes instance
    """
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(nrows=1)

    x_labels, y_labels, grid = get_2dgrid(self)

    # set label properties
    def tick(lab):
        lab = str(lab)
        if len(lab) > 20:
            lab = lab[:17] + '...'
        return lab
    x_labels = [tick(lab) for lab in x_labels]
    y_labels = [tick(lab) for lab in y_labels]

    # plot 2d grid
    xtick_pos = np.arange(len(x_labels)) + 0.5
    ytick_pos = np.arange(len(y_labels)) + 0.5

    im = ax.pcolormesh(xtick_pos, ytick_pos, grid, shading='auto', **kwargs)
    fig.colorbar(im, ax=ax)

    ax.set_xlim((0., float(len(x_labels))))
    ax.set_ylim((0., float(len(y_labels))))
    ax.set_xticks(xtick_pos)
    ax.set_yticks(ytick_pos)
    ax.set_xticklabels(x_labels, rotation=90)
    ax.set_yticklabels(y_labels)

    if name is not None:
        ax.set_title(name)
    else:
        ax.set_title(self.name)
    return ax
