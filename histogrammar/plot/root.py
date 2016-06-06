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

def setTH1(self, th1):
    th1.SetBinContent(0, self.underflow.entries)
    for i, v in enumerate(self.values):
        th1.SetBinContent(i + 1, v.entries)
    th1.SetBinContent(len(self.values), self.overflow.entries)
    th1.SetEntries(self.entries)

# "Public" methods; what we want to attach to the Histogram as a mix-in.

class HistogramMethods(object):
    def TH1C(self, name, title):
        import ROOT
        th1 = ROOT.TH1C(name, title, len(self.values), self.low, self.high)
        setTH1(self, th1)
        return th1

    def TH1S(self, name, title):
        import ROOT
        th1 = ROOT.TH1S(name, title, len(self.values), self.low, self.high)
        setTH1(self, th1)
        return th1

    def TH1I(self, name, title):
        import ROOT
        th1 = ROOT.TH1I(name, title, len(self.values), self.low, self.high)
        setTH1(self, th1)
        return th1

    def TH1F(self, name, title):
        import ROOT
        th1 = ROOT.TH1F(name, title, len(self.values), self.low, self.high)
        setTH1(self, th1)
        return th1

    def TH1D(self, name, title):
        import ROOT
        th1 = ROOT.TH1D(name, title, len(self.values), self.low, self.high)
        setTH1(self, th1)
        return th1
