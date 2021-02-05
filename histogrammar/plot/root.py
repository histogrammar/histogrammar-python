#!/usr/bin/env python

# Copyright 2016 DIANA-HEP
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

# "Private" methods; not attached to the histogram because not a member of the class,
# but within scope because it's a closure.

import math
import types

# python 2/3 compatibility fixes
from histogrammar.util import xrange

try:
    from collections import OrderedDict
except ImportError:
    class OrderedDict(dict):
        def __init__(self):
            self._data = []

        def __setattr__(self, key, value):
            self._data.append((key, value))

        def __getattr__(self, key):
            return dict(self._data)[key]

        def items(self):
            return self._data

        def keys(self):
            return [k for k, v in self._data]

        def values(self):
            return [v for k, v in self._data]


def setTH1(entries, values, underflow, overflow, th1):
    th1.SetBinContent(0, underflow)
    for i, v in enumerate(values):
        th1.SetBinContent(i + 1, v)
    th1.SetBinContent(len(values) + 1, overflow)
    th1.SetEntries(entries)


def prepareTH2sparse(sparse):
    sample = list(sparse.bins.values())[0]
    yminBins = [v.minBin for v in sparse.bins.values() if v.minBin is not None]
    ymaxBins = [v.maxBin for v in sparse.bins.values() if v.maxBin is not None]
    if len(yminBins) > 0 and len(ymaxBins) > 0:
        yminBin = min(yminBins)
        ymaxBin = max(ymaxBins)
    else:
        yminBin = 0
        ymaxBin = 0
    ynum = 1 + ymaxBin - yminBin
    ylow = yminBin * sample.binWidth + sample.origin
    yhigh = (ymaxBin + 1) * sample.binWidth + sample.origin
    return yminBin, ymaxBin, ynum, ylow, yhigh


def setTH2sparse(sparse, yminBin, ymaxBin, th2):
    for i, iindex in enumerate(xrange(sparse.minBin, sparse.maxBin + 1)):
        for j, jindex in enumerate(xrange(yminBin, ymaxBin + 1)):
            if iindex in sparse.bins and jindex in sparse.bins[iindex].bins:
                th2.SetBinContent(i + 1, j + 1, sparse.bins[iindex].bins[jindex].entries)

# "Public" methods; what we want to attach to the Histogram as a mix-in.


class HistogramMethods(object):
    def plotroot(self, name, title="", binType="D"):
        import ROOT
        constructor = getattr(ROOT, "TH1" + binType)
        th1 = constructor(name, title, len(self.values), self.low, self.high)
        setTH1(self.entries, [x.entries for x in self.values], self.underflow.entries, self.overflow.entries, th1)
        return th1


class SparselyHistogramMethods(object):
    def plotroot(self, name, title="", binType="D"):
        import ROOT
        constructor = getattr(ROOT, "TH1" + binType)
        if self.minBin is None or self.maxBin is None:
            th1 = constructor(name, title, 1, self.origin, self.origin + 1.0)
        else:
            size = 1 + self.maxBin - self.minBin
            th1 = constructor(name, title, size, self.low, self.high)
            setTH1(
                self.entries, [
                    self.bins[i].entries if i in self.bins else 0.0 for i in xrange(
                        self.minBin, self.maxBin + 1)], 0.0, 0.0, th1)
        return th1


class IrregularlyHistogramMethods(object):
    pass


class CentrallyHistogramMethods(object):
    pass


class CategorizeHistogramMethods(object):
    def plotroot(self, name, title="", binType="C"):
        """ Construct a ROOT histogram

        :param str name: name of the histogram
        :param str title: title of the histogram (optional)
        :param str binType: histogram bin type. Default is "C" (char).
        :returns: ROOT histgram
        """
        import ROOT
        constructor = getattr(ROOT, "TH1" + binType)
        th1 = constructor(name, title, len(self.bins), 0, 1)
        th1.SetMinimum(0)
        for i, key in enumerate(self.bins.keys()):
            b = self.bins[key]
            try:
                label = str(key)
            except BaseException:
                label = 'bin_%d' % i
            th1.Fill(label, b.entries)
        return th1


class ProfileMethods(object):
    def plotroot(self, name, title=""):
        import ROOT
        tprofile = ROOT.TProfile(name, title, len(self.values), self.low, self.high)
        tprofile.SetBinContent(0, self.underflow.entries*self.underflow.entries)
        tprofile.SetBinEntries(0, self.underflow.entries)
        for i, v in enumerate(self.values):
            if not math.isnan(v.mean):
                tprofile.SetBinError(i + 1, math.sqrt(v.entries) * v.mean)
                tprofile.SetBinContent(i + 1, v.entries * v.mean)
                tprofile.SetBinEntries(i + 1, v.entries)
        tprofile.SetBinContent(len(self.values) + 1, self.overflow.entries*self.overflow.entries)
        tprofile.SetBinEntries(len(self.values) + 1, self.overflow.entries)
        tprofile.SetEntries(self.entries)
        return tprofile


class SparselyProfileMethods(object):
    def plotroot(self, name, title=""):
        import ROOT
        if self.minBin is None or self.maxBin is None:
            tprofile = ROOT.TProfile(name, title, 1, self.origin, self.origin + 1.0)
        else:
            tprofile = ROOT.TProfile(name, title, 1 + self.maxBin - self.minBin, self.low, self.high)
            for i, index in enumerate(xrange(self.minBin, self.maxBin + 1)):
                if index in self.bins:
                    v = self.bins[index]
                    if not math.isnan(v.mean):
                        tprofile.SetBinError(i + 1, math.sqrt(v.entries) * v.mean)
                        tprofile.SetBinContent(i + 1, v.entries * v.mean)
                        tprofile.SetBinEntries(i + 1, v.entries)
            tprofile.SetBinContent(0, 0.0)
            tprofile.SetBinEntries(0, 0.0)
            tprofile.SetBinContent(1 + self.maxBin - self.minBin, 0.0)
            tprofile.SetBinEntries(1 + self.maxBin - self.minBin, 0.0)
            tprofile.SetEntries(self.entries)
        return tprofile


class ProfileErrMethods(object):
    def plotroot(self, name, title=""):
        import ROOT
        tprofile = ROOT.TProfile(name, title, len(self.values), self.low, self.high)
        tprofile.SetBinContent(0, self.underflow.entries*self.underflow.entries)
        tprofile.SetBinEntries(0, self.underflow.entries)
        for i, v in enumerate(self.values):
            if not math.isnan(v.mean):
                tprofile.SetBinError(i + 1, math.sqrt(v.entries*(v.variance + v.mean*v.mean)))
                tprofile.SetBinContent(i + 1, v.entries * v.mean)
                tprofile.SetBinEntries(i + 1, v.entries)
        tprofile.SetBinContent(len(self.values) + 1, self.overflow.entries*self.overflow.entries)
        tprofile.SetBinEntries(len(self.values) + 1, self.overflow.entries)
        tprofile.SetEntries(self.entries)
        return tprofile


class SparselyProfileErrMethods(object):
    def plotroot(self, name, title=""):
        import ROOT
        if self.minBin is None or self.maxBin is None:
            tprofile = ROOT.TProfile(name, title, 1, self.origin, self.origin + 1.0)
        else:
            tprofile = ROOT.TProfile(name, title, 1 + self.maxBin - self.minBin, self.low, self.high)
            for i, index in enumerate(xrange(self.minBin, self.maxBin + 1)):
                if index in self.bins:
                    v = self.bins[index]
                    if not math.isnan(v.mean):
                        tprofile.SetBinError(i + 1, math.sqrt(v.entries*(v.variance + v.mean*v.mean)))
                        tprofile.SetBinContent(i + 1, v.entries * v.mean)
                        tprofile.SetBinEntries(i + 1, v.entries)
            tprofile.SetBinContent(0, 0.0)
            tprofile.SetBinEntries(0, 0.0)
            tprofile.SetBinContent(1 + self.maxBin - self.minBin, 0.0)
            tprofile.SetBinEntries(1 + self.maxBin - self.minBin, 0.0)
            tprofile.SetEntries(self.entries)
        return tprofile


class StackedHistogramMethods(object):
    def plotroot(self, *names):
        out = OrderedDict()
        for n, (c, v) in zip(names, self.thresholds):
            if isinstance(n, (list, tuple)) and len(n) == 2:
                name, title = n
            else:
                name, title = n, ""
            out[c] = v.plotroot(name, title)

        def Draw(self, options=""):
            first = True
            for v in self.values():
                v.Draw(options)
                if first:
                    options = options + "same"
                    first = False

        out.Draw = types.MethodType(Draw, out)
        return out


class PartitionedHistogramMethods(object):
    def plotroot(self, *names):
        out = OrderedDict()
        for n, (c, v) in zip(names, self.thresholds):
            if isinstance(n, (list, tuple)) and len(n) == 2:
                name, title = n
            else:
                name, title = n, ""
            out[c] = v.plotroot(name, title)

        def Draw(self, options=""):
            first = True
            for v in self.values():
                v.Draw(options)
                if first:
                    options = options + "same"
                    first = False

        out.Draw = types.MethodType(Draw, out)
        return out


class FractionedHistogramMethods(object):
    def plotroot(self, numeratorName, denominatorName):
        import ROOT
        denominator = self.denominator.plotroot(denominatorName)
        num = denominator.GetNbinsX()
        low = denominator.GetBinLowEdge(1)
        high = denominator.GetBinLowEdge(num) + denominator.GetBinWidth(num)

        numerator = ROOT.TH1D(numeratorName, "", num, low, high)
        if isinstance(self.numerator, HistogramMethods):
            setTH1(self.numerator.entries, [x.entries for x in self.numerator.values],
                   self.numerator.underflow.entries, self.numerator.overflow.entries, numerator)
        elif isinstance(self.numerator, SparselyHistogramMethods):
            setTH1(self.numerator.entries,
                   [self.numerator.bins[i].entries if i in self.numerator.bins else 0.0 for i in xrange(
                       self.denominator.minBin, self.denominator.maxBin + 1)],
                   0.0,
                   0.0,
                   numerator)

        return ROOT.TEfficiency(numerator, denominator)


class TwoDimensionallyHistogramMethods(object):
    def plotroot(self, name, title="", binType="D"):
        import ROOT
        constructor = getattr(ROOT, "TH2" + binType)
        sample = self.values[0]
        th2 = constructor(name, title, int(self.num), float(self.low), float(self.high),
                          int(sample.num), float(sample.low), float(sample.high))
        for i in xrange(self.num):
            for j in xrange(sample.num):
                th2.SetBinContent(i + 1, j + 1, self.values[i].values[j].entries)
        return th2


class SparselyTwoDimensionallyHistogramMethods(object):
    def plotroot(self, name, title="", binType="D"):
        import ROOT
        constructor = getattr(ROOT, "TH2" + binType)
        yminBin, ymaxBin, ynum, ylow, yhigh = prepareTH2sparse(self)
        th2 = constructor(name, title, int(self.num), float(self.low), float(self.high),
                          int(ynum), float(ylow), float(yhigh))
        setTH2sparse(self, yminBin, ymaxBin, th2)
        return th2


class IrregularlyTwoDimensionallyHistogramMethods(object):
    pass


class CentrallyTwoDimensionallyHistogramMethods(object):
    pass
