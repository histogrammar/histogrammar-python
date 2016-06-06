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
from histogrammar.util import serializable
from histogrammar.primitives.select import Select
from histogrammar.primitives.bin import Bin
from histogrammar.primitives.count import Count

import histogrammar.plot.root
import histogrammar.plot.bokeh

def Histogram(num, low, high, quantity, selection=unweighted):
    return Select(selection, Bin(num, low, high, quantity, Count(), Count(), Count(), Count()))

class HistogramMethods(Select,
                       histogrammar.plot.root.HistogramMethods,
                       histogrammar.plot.bokeh.HistogramMethods):
    @property
    def name(self):
        return "Select"

    @property
    def factory(self):
        return Select

    @property
    def numericalValues(self):
        return [v.entries for v in self.cut.values]

    @property
    def numericalOverflow(self):
        return self.cut.overflow.entries

    @property
    def numericalUnderflow(self):
        return self.cut.underflow.entries

    @property
    def numericalNanflow(self):
        return self.cut.nanflow.entries

def addImplicitMethods(container):
    if isinstance(container, Select) and \
           isinstance(container.cut, Bin) and \
           all(isinstance(v, Count) for v in container.cut.values) and \
           isinstance(container.cut.underflow, Count) and \
           isinstance(container.cut.overflow, Count) and \
           isinstance(container.cut.nanflow, Count):
        container.__class__ = HistogramMethods
