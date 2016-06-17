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
    def TH1C(self, name, title=""):
        import ROOT
        th1 = ROOT.TH1C(name, title, len(self.values), self.low, self.high)
        setTH1(self.entries, [x.entries for x in self.values], self.underflow.entries, self.overflow.entries, th1)
        return th1

    def TH1S(self, name, title=""):
        import ROOT
        th1 = ROOT.TH1S(name, title, len(self.values), self.low, self.high)
        setTH1(self.entries, [x.entries for x in self.values], self.underflow.entries, self.overflow.entries, th1)
        return th1

    def TH1I(self, name, title=""):
        import ROOT
        th1 = ROOT.TH1I(name, title, len(self.values), self.low, self.high)
        setTH1(self.entries, [x.entries for x in self.values], self.underflow.entries, self.overflow.entries, th1)
        return th1

    def TH1F(self, name, title=""):
        import ROOT
        th1 = ROOT.TH1F(name, title, len(self.values), self.low, self.high)
        setTH1(self.entries, [x.entries for x in self.values], self.underflow.entries, self.overflow.entries, th1)
        return th1

    def TH1D(self, name, title=""):
        import ROOT
        th1 = ROOT.TH1D(name, title, len(self.values), self.low, self.high)
        setTH1(self.entries, [x.entries for x in self.values], self.underflow.entries, self.overflow.entries, th1)
        return th1

class SparselyHistogramMethods(object):
    def TH1C(self, name, title=""):
        import ROOT
        if self.minBin is None or self.maxBin is None:
            th1 = ROOT.TH1C(name, title, 1, self.origin, self.origin + 1.0)
        else:
            size = 1 + self.maxBin - self.minBin
            th1 = ROOT.TH1C(name, title, size, self.low, self.high)
            setTH1(self.entries, [self.bins[i].entries if i in self.bins else 0.0 for i in xrange(size)], 0.0, 0.0, th1)
        return th1

    def TH1S(self, name, title=""):
        import ROOT
        if self.minBin is None or self.maxBin is None:
            th1 = ROOT.TH1S(name, title, 1, self.origin, self.origin + 1.0)
        else:
            size = 1 + self.maxBin - self.minBin
            th1 = ROOT.TH1S(name, title, size, self.low, self.high)
            setTH1(self.entries, [self.bins[i].entries if i in self.bins else 0.0 for i in xrange(size)], 0.0, 0.0, th1)
        return th1

    def TH1I(self, name, title=""):
        import ROOT
        if self.minBin is None or self.maxBin is None:
            th1 = ROOT.TH1I(name, title, 1, self.origin, self.origin + 1.0)
        else:
            size = 1 + self.maxBin - self.minBin
            th1 = ROOT.TH1I(name, title, size, self.low, self.high)
            setTH1(self.entries, [self.bins[i].entries if i in self.bins else 0.0 for i in xrange(size)], 0.0, 0.0, th1)
        return th1

    def TH1F(self, name, title=""):
        import ROOT
        if self.minBin is None or self.maxBin is None:
            th1 = ROOT.TH1F(name, title, 1, self.origin, self.origin + 1.0)
        else:
            size = 1 + self.maxBin - self.minBin
            th1 = ROOT.TH1F(name, title, size, self.low, self.high)
            setTH1(self.entries, [self.bins[i].entries if i in self.bins else 0.0 for i in xrange(size)], 0.0, 0.0, th1)
        return th1

    def TH1D(self, name, title=""):
        import ROOT
        if self.minBin is None or self.maxBin is None:
            th1 = ROOT.TH1D(name, title, 1, self.origin, self.origin + 1.0)
        else:
            size = 1 + self.maxBin - self.minBin
            th1 = ROOT.TH1D(name, title, size, self.low, self.high)
            setTH1(self.entries, [self.bins[i].entries if i in self.bins else 0.0 for i in xrange(size)], 0.0, 0.0, th1)
        return th1

class ProfileMethods(object):
    def TProfile(self, name, title=""):
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
    def TProfile(self, name, title=""):
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
    def TProfile(self, name, title=""):
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
    def TProfile(self, name, title=""):
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
    def TProfile(self, name, title="", confidenceInterval=None):
        intervals = []
        for n, d in zip(self.numerator.values, self.denominator.values):
            if d.entries > 0.0:
                if confidenceInterval is None:
                    frac = n.entries / d.entries
                    intervals.append((frac, frac, frac))
                else:
                    intervals.append((confidenceInterval(n.entries, d.entries, -1.0),
                                      confidenceInterval(n.entries, d.entries,  0.0),
                                      confidenceInterval(n.entries, d.entries,  1.0)))
            else:
                intervals.append(None)

        tprofile = ROOT.TProfile(name, title, len(self.denominator.values), self.denominator.low, self.denominator.high)
        for i, interval in enumerate(intervals):
            if interval is not None:
                low, mid, high = interval
                tprofile.SetBinError(???)
                tprofile.SetBinContent(???)
                tprofile.SetBinEntries(self.denominator.values[i].entries)

        return tprofile

        # for i, v in enumerate(self.values):
        #     tprofile.SetBinError(i + 1, math.sqrt(v.entries*(v.variance + v.mean**2)))
        #     tprofile.SetBinContent(i + 1, v.entries * v.mean)
        #     tprofile.SetBinEntries(i + 1, v.entries)



class TwoDimensionallyHistogramMethods(object):
    def TH2C(self, name, title=""):
        import ROOT
        sample = self.values[0]
        th2 = ROOT.TH2C(name, title, self.num, self.low, self.high, sample.num, sample.low, sample.high)
        for i in xrange(self.num):
            for j in xrange(sample.num):
                th2.SetBinContent(i + 1, j + 1, self.values[i].values[j].entries)
        return th2

    def TH2S(self, name, title=""):
        import ROOT
        sample = self.values[0]
        th2 = ROOT.TH2S(name, title, self.num, self.low, self.high, sample.num, sample.low, sample.high)
        for i in xrange(self.num):
            for j in xrange(sample.num):
                th2.SetBinContent(i + 1, j + 1, self.values[i].values[j].entries)
        return th2

    def TH2I(self, name, title=""):
        import ROOT
        sample = self.values[0]
        th2 = ROOT.TH2I(name, title, self.num, self.low, self.high, sample.num, sample.low, sample.high)
        for i in xrange(self.num):
            for j in xrange(sample.num):
                th2.SetBinContent(i + 1, j + 1, self.values[i].values[j].entries)
        return th2

    def TH2F(self, name, title=""):
        import ROOT
        sample = self.values[0]
        th2 = ROOT.TH2F(name, title, self.num, self.low, self.high, sample.num, sample.low, sample.high)
        for i in xrange(self.num):
            for j in xrange(sample.num):
                th2.SetBinContent(i + 1, j + 1, self.values[i].values[j].entries)
        return th2

    def TH2D(self, name, title=""):
        import ROOT
        sample = self.values[0]
        th2 = ROOT.TH2D(name, title, self.num, self.low, self.high, sample.num, sample.low, sample.high)
        for i in xrange(self.num):
            for j in xrange(sample.num):
                th2.SetBinContent(i + 1, j + 1, self.values[i].values[j].entries)
        return th2

class SparselyTwoDimensionallyHistogramMethods(object):
    def TH2C(self, name, title=""):
        import ROOT
        yminBin, ymaxBin, ynum, ylow, yhigh = prepareTH2sparse(self)
        th2 = ROOT.TH2C(name, title, self.num, self.low, self.high, ynum, ylow, yhigh)
        setTH2sparse(self, yminBin, ymaxBin, th2)
        return th2

    def TH2S(self, name, title=""):
        import ROOT
        yminBin, ymaxBin, ynum, ylow, yhigh = prepareTH2sparse(self)
        th2 = ROOT.TH2S(name, title, self.num, self.low, self.high, ynum, ylow, yhigh)
        setTH2sparse(self, yminBin, ymaxBin, th2)
        return th2

    def TH2I(self, name, title=""):
        import ROOT
        yminBin, ymaxBin, ynum, ylow, yhigh = prepareTH2sparse(self)
        th2 = ROOT.TH2I(name, title, self.num, self.low, self.high, ynum, ylow, yhigh)
        setTH2sparse(self, yminBin, ymaxBin, th2)
        return th2

    def TH2F(self, name, title=""):
        import ROOT
        yminBin, ymaxBin, ynum, ylow, yhigh = prepareTH2sparse(self)
        th2 = ROOT.TH2F(name, title, self.num, self.low, self.high, ynum, ylow, yhigh)
        setTH2sparse(self, yminBin, ymaxBin, th2)
        return th2

    def TH2D(self, name, title=""):
        import ROOT
        yminBin, ymaxBin, ynum, ylow, yhigh = prepareTH2sparse(self)
        th2 = ROOT.TH2D(name, title, self.num, self.low, self.high, ynum, ylow, yhigh)
        setTH2sparse(self, yminBin, ymaxBin, th2)
        return th2
