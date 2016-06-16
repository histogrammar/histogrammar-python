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
    def TH1F(self, name, title=""):
        import ROOT
        if self.minBin is None or self.maxBin is None:
            th1 = ROOT.TH1F(name, title, 1, self.origin, self.origin + 1.0)
            setTH1(self.entries, [0.0], 0.0, 0.0, th1)
        else:
            size = 1 + self.maxBin - self.minBin
            th1 = ROOT.TH1F(name, title, size, self.low, self.high)
            setTH1(self.entries, [self.bins[x].entries if x in self.bins else 0.0 for x in xrange(size)], 0.0, 0.0, th1)
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
        tprofile.SetBinContent(len(self.values), self.underflow.entries**2)
        tprofile.SetBinEntries(len(self.values), self.underflow.entries)
        tprofile.SetEntries(self.entries)
        return tprofile

class SparselyProfileMethods(object):
    pass

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
        tprofile.SetBinContent(len(self.values), self.underflow.entries**2)
        tprofile.SetBinEntries(len(self.values), self.underflow.entries)
        tprofile.SetEntries(self.entries)
        return tprofile

class SparselyProfileErrMethods(object):
    pass

class StackedHistogramMethods(object):
    pass

class PartitionedHistogramMethods(object):
    pass

class FractionedHistogramMethods(object):
    pass

class TwoDimensionallyHistogramMethods(object):
    pass

class SparselyTwoDimensionallyHistogramMethods(object):
    pass
