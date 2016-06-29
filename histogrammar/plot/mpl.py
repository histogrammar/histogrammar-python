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
import types

class HistogramMethods(object):
    def matplotlib(self, name=None, **kwargs):
        """ Plotting method for Bin.
              kwargs :  `matplotlib.patches.Rectangle` properties.

            Returns a matplotlib.axes instance
        """
        import matplotlib.pyplot as plt
        import numpy as np
        fig = plt.gcf()
        ax = fig.gca()

        entries = [x.entries for x in self.values]

        num_bins = len(self.values)
        width = (self.high - self.low)/num_bins
        edges = np.linspace(self.low, self.high, num_bins + 1)[:-1]

        ax.bar(edges, entries, width=width, **kwargs)
        if name is not None:
            ax.set_title(name)
        else:
            ax.set_title(self.name)

        return ax

class ProfileErrMethods(object):
    def matplotlib(self, name=None, **kwargs):
        """ Plotting method for Bin of Deviate
              kwargs :  `matplotlib.collections.LineCollection` properties.

            Returns a matplotlib.axes instance
        """
        import matplotlib.pyplot as plt
        import numpy as np
        fig = plt.gcf()
        ax = fig.gca()

        bin_centers = [sum(self.range(x))/2.0 for x in self.indexes]
        xranges = [self.range(x) for x in self.indexes]
        means = self.meanValues
        variances = self.varianceValues
        num_bins = len(variances)

        xmins = [x[0] for x in xranges]
        xmaxs = [x[1] for x in xranges]
        ax.hlines(self.meanValues, xmins, xmaxs, **kwargs)

        ymins = [means[i] - np.sqrt(variances[i]) for i in range(num_bins)]
        ymaxs = [means[i] + np.sqrt(variances[i]) for i in range(num_bins)]
        ax.vlines(bin_centers, ymins, ymaxs, **kwargs)

        if name is not None:
            ax.set_title(name)
        else:
            ax.set_title(self.name)

        return ax


class TwoDimensionallyHistogramMethods(object):
    def matplotlib(self, name=None, **kwargs):
        """ Plotting method for Bin of Bin of Count
              kwargs :  `matplotlib.collections.QuadMesh` properties.

            Returns a matplotlib.axes instance
        """
        import matplotlib.pyplot as plt
        import numpy as np
        fig = plt.gcf()
        ax = fig.gca()

        samp = self.values[0]
        x_ranges = np.unique(np.array([self.range(i) for i in self.indexes]).flatten())
        y_ranges = np.unique(np.array([samp.range(i) for i in samp.indexes]).flatten())

        grid = np.zeros((samp.num, self.num))

        for j in xrange(self.num):
            for i in xrange(samp.num):
                grid[i,j] = self.values[j].values[i].entries

        ax.pcolormesh(x_ranges, y_ranges, grid, **kwargs)

        if name is not None:
            ax.set_title(name)
        else:
            ax.set_title(self.name)

        return ax






