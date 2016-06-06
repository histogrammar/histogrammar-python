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
from histogrammar.primitives.deviate import Deviate

import histogrammar.plot.root
import histogrammar.plot.bokeh

def Histogram(num, low, high, quantity, selection=unweighted):
    return Select(selection, Bin(num, low, high, quantity, Count(), Count(), Count(), Count()))

def Profile(num, low, high, fillx, filly, selection=unweighted):
    return Select(selection, Bin(num, low, high, fillx, Deviate(filly), Count(), Count(), Count()))

class SelectedHistogramMethods(Select):
    @property
    def name(self):
        return "Select"

    @property
    def factory(self):
        return Select

    def __getattr__(self, attr):
        if attr.startswith("__") and attr.endswith("__"):
            return getattr(Select, attr)
        elif attr not in self.__dict__ and hasattr(self.__dict__["cut"], attr):
            return getattr(self.__dict__["cut"], attr)
        else:
            return self.__dict__[attr]

class HistogramMethods(Bin,
                       histogrammar.plot.root.HistogramMethods,
                       histogrammar.plot.bokeh.HistogramMethods):
    @property
    def name(self):
        return "Bin"

    @property
    def factory(self):
        return Bin

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

class SelectedProfileMethods(Select):
    @property
    def name(self):
        return "Select"

    @property
    def factory(self):
        return Select

    def __getattr__(self, attr):
        if attr.startswith("__") and attr.endswith("__"):
            return getattr(Select, attr)
        elif attr not in self.__dict__ and hasattr(self.__dict__["cut"], attr):
            return getattr(self.__dict__["cut"], attr)
        else:
            return self.__dict__[attr]

class ProfileMethods(Bin,
                     histogrammar.plot.root.ProfileMethods,
                     histogrammar.plot.bokeh.ProfileMethods):
    @property
    def name(self):
        return "Bin"

    @property
    def factory(self):
        return Bin

def addImplicitMethods(container):
    if isinstance(container, Bin) and \
       all(isinstance(v, Count) for v in container.values) and \
       isinstance(container.underflow, Count) and \
       isinstance(container.overflow, Count) and \
       isinstance(container.nanflow, Count):
        container.__class__ = HistogramMethods

    elif isinstance(container, Select) and \
           isinstance(container.cut, Bin) and \
           all(isinstance(v, Count) for v in container.cut.values) and \
           isinstance(container.cut.underflow, Count) and \
           isinstance(container.cut.overflow, Count) and \
           isinstance(container.cut.nanflow, Count):
        container.__class__ = SelectedHistogramMethods

    elif isinstance(container, Bin) and \
       all(isinstance(v, Deviate) for v in container.values) and \
       isinstance(container.underflow, Count) and \
       isinstance(container.overflow, Count) and \
       isinstance(container.nanflow, Count):
        container.__class__ = ProfileMethods

    elif isinstance(container, Select) and \
           isinstance(container.cut, Bin) and \
           all(isinstance(v, Deviate) for v in container.cut.values) and \
           isinstance(container.cut.underflow, Count) and \
           isinstance(container.cut.overflow, Count) and \
           isinstance(container.cut.nanflow, Count):
        container.__class__ = SelectedProfileMethods
