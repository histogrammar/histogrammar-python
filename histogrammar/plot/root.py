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

# "Private" methods; not attached to the histogram because not a member of the class, but within scope because it's a closure.

import math

def setTH1(entries, values, underflow, overflow, th1):
    th1.SetBinContent(0, underflow)
    for i, v in enumerate(values):
        th1.SetBinContent(i + 1, v)
    th1.SetBinContent(len(values), overflow)
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
    def root(self, name, title="", binType="D"):
        import ROOT
        constructor = getattr(ROOT, "TH1" + binType)
        th1 = constructor(name, title, len(self.values), self.low, self.high)
        setTH1(self.entries, [x.entries for x in self.values], self.underflow.entries, self.overflow.entries, th1)
        return th1

class SparselyHistogramMethods(object):
    def root(self, name, title="", binType="D"):
        import ROOT
        constructor = getattr(ROOT, "TH1" + binType)
        if self.minBin is None or self.maxBin is None:
            th1 = constructor(name, title, 1, self.origin, self.origin + 1.0)
        else:
            size = 1 + self.maxBin - self.minBin
            th1 = constructor(name, title, size, self.low, self.high)
            setTH1(self.entries, [self.bins[i].entries if i in self.bins else 0.0 for i in xrange(size)], 0.0, 0.0, th1)
        return th1

class ProfileMethods(object):
    def root(self, name, title=""):
        import ROOT
        tprofile = ROOT.TProfile(name, title, len(self.values), self.low, self.high)
        tprofile.SetBinContent(0, self.underflow.entries**2)
        tprofile.SetBinEntries(0, self.underflow.entries)
        for i, v in enumerate(self.values):
            tprofile.SetBinError(i + 1, math.sqrt(v.entries) * v.mean)
            tprofile.SetBinContent(i + 1, v.entries * v.mean)
            tprofile.SetBinEntries(i + 1, v.entries)
        tprofile.SetBinContent(len(self.values), self.overflow.entries**2)
        tprofile.SetBinEntries(len(self.values), self.overflow.entries)
        tprofile.SetEntries(self.entries)
        return tprofile

class SparselyProfileMethods(object):
    def root(self, name, title=""):
        import ROOT
        if self.minBin is None or self.maxBin is None:
            tprofile = ROOT.TProfile(name, title, 1, self.origin, self.origin + 1.0)
        else:
            tprofile = ROOT.TProfile(name, title, 1 + self.maxBin - self.minBin, self.low, self.high)
            for i, index in enumerate(xrange(self.minBin, self.maxBin + 1)):
                if index in self.bins:
                    v = self.bins[index]
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
    def root(self, name, title=""):
        import ROOT
        tprofile = ROOT.TProfile(name, title, len(self.values), self.low, self.high)
        tprofile.SetBinContent(0, self.underflow.entries**2)
        tprofile.SetBinEntries(0, self.underflow.entries)
        for i, v in enumerate(self.values):
            tprofile.SetBinError(i + 1, math.sqrt(v.entries*(v.variance + v.mean**2)))
            tprofile.SetBinContent(i + 1, v.entries * v.mean)
            tprofile.SetBinEntries(i + 1, v.entries)
        tprofile.SetBinContent(len(self.values), self.overflow.entries**2)
        tprofile.SetBinEntries(len(self.values), self.overflow.entries)
        tprofile.SetEntries(self.entries)
        return tprofile

class SparselyProfileErrMethods(object):
    def root(self, name, title=""):
        import ROOT
        if self.minBin is None or self.maxBin is None:
            tprofile = ROOT.TProfile(name, title, 1, self.origin, self.origin + 1.0)
        else:
            tprofile = ROOT.TProfile(name, title, 1 + self.maxBin - self.minBin, self.low, self.high)
            for i, index in enumerate(xrange(self.minBin, self.maxBin + 1)):
                if index in self.bins:
                    v = self.bins[index]
                    tprofile.SetBinError(i + 1, math.sqrt(v.entries*(v.variance + v.mean**2)))
                    tprofile.SetBinContent(i + 1, v.entries * v.mean)
                    tprofile.SetBinEntries(i + 1, v.entries)
            tprofile.SetBinContent(0, 0.0)
            tprofile.SetBinEntries(0, 0.0)
            tprofile.SetBinContent(1 + self.maxBin - self.minBin, 0.0)
            tprofile.SetBinEntries(1 + self.maxBin - self.minBin, 0.0)
            tprofile.SetEntries(self.entries)
        return tprofile

class StackedHistogramMethods(object):
    pass

class PartitionedHistogramMethods(object):
    pass

class FractionedHistogramMethods(object):
    def root(self, numeratorName, denominatorName):
        import ROOT
        numerator = self.numerator.root(numeratorName)
        denominator = self.denominator.root(denominatorName)
        return ROOT.TEfficiency(numerator, denominator)

class TwoDimensionallyHistogramMethods(object):
    def root(self, name, title="", binType="D"):
        import ROOT
        constructor = getattr(ROOT, "TH2" + binType)
        sample = self.values[0]
        th2 = constructor(name, title, self.num, self.low, self.high, sample.num, sample.low, sample.high)
        for i in xrange(self.num):
            for j in xrange(sample.num):
                th2.SetBinContent(i + 1, j + 1, self.values[i].values[j].entries)
        return th2

class SparselyTwoDimensionallyHistogramMethods(object):
    def root(self, name, title="", binType="D"):
        import ROOT
        constructor = getattr(ROOT, "TH2" + binType)
        yminBin, ymaxBin, ynum, ylow, yhigh = prepareTH2sparse(self)
        th2 = constructor(name, title, self.num, self.low, self.high, ynum, ylow, yhigh)
        setTH2sparse(self, yminBin, ymaxBin, th2)
        return th2
