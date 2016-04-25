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

from histogrammar.defs import unweighted
from histogrammar.primitives.bin import Bin
from histogrammar.primitives.count import Count

def Histogram(num, low, high, quantity, selection=unweighted):
    return Bin(num, low, high, quantity, selection, Count(), Count(), Count(), Count())

## Histogram.ed = lambda low, high, entries, values, underflow, overflow, nanflow: \
##     Bin(len(values), low, high, None, None, None, underflow, overflow, nanflow)

class HistogramMethods(Bin):
    @property
    def name(self):
        return super(Bin, self).name

    @property
    def factory(self):
        return super(Bin, self).factory

    @property
    def numericalValues(self):
        return [v.entries for v in self.values]

    @property
    def numericalOverflow(self):
        return self.overflow.entries

    @property
    def numericalUnderflow(self):
        return self.underflow.entries

    @property
    def numericalNanflow(self):
        return self.nanflow.entries

    def __setTH1(self, th1):
        th1.SetBinContent(0, self.underflow.entries)
        for i, v in enumerate(self.values):
            th1.SetBinContent(i + 1, v.entries)
        th1.SetBinContent(len(self.values), self.overflow.entries)
        th1.SetEntries(self.entries)

    def TH1C(self, name, title):
        import ROOT
        th1 = ROOT.TH1C(name, title, len(self.values), self.low, self.high)
        self.__setTH1(th1)
        return th1

    def TH1S(self, name, title):
        import ROOT
        th1 = ROOT.TH1S(name, title, len(self.values), self.low, self.high)
        self.__setTH1(th1)
        return th1

    def TH1I(self, name, title):
        import ROOT
        th1 = ROOT.TH1I(name, title, len(self.values), self.low, self.high)
        self.__setTH1(th1)
        return th1

    def TH1F(self, name, title):
        import ROOT
        th1 = ROOT.TH1F(name, title, len(self.values), self.low, self.high)
        self.__setTH1(th1)
        return th1

    def TH1D(self, name, title):
        import ROOT
        th1 = ROOT.TH1D(name, title, len(self.values), self.low, self.high)
        self.__setTH1(th1)
        return th1

def addImplicitMethods(container):
    if isinstance(container, Bin) and \
       all(isinstance(v, Count) for v in container.values) and \
       isinstance(container.underflow, Count) and \
       isinstance(container.overflow, Count) and \
       isinstance(container.nanflow, Count):
        container.__class__ = HistogramMethods
