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
from histogrammar.primitives.average import Average
from histogrammar.primitives.bin import Bin
from histogrammar.primitives.count import Count
from histogrammar.primitives.deviate import Deviate
from histogrammar.primitives.fraction import Fraction
from histogrammar.primitives.partition import Partition
from histogrammar.primitives.select import Select
from histogrammar.primitives.sparsebin import SparselyBin
from histogrammar.primitives.stack import Stack
from histogrammar.util import serializable

import histogrammar.plot.root
import histogrammar.plot.bokeh
import histogrammar.plot.mpl

def Histogram(num, low, high, quantity, selection=unweighted):
    return Select.ing(selection, Bin.ing(num, low, high, quantity,
        Count.ing(), Count.ing(), Count.ing(), Count.ing()))

def SparselyHistogram(binWidth, quantity, selection=unweighted, origin=0.0):
    return Select.ing(selection,
        SparselyBin.ing(binWidth, quantity, Count.ing(), Count.ing(), origin))

def Profile(num, low, high, binnedQuantity, averagedQuantity, selection=unweighted):
    return Select.ing(selection,
        Bin.ing(num, low, high, binnedQuantity,
            Average.ing(averagedQuantity)))

def SparselyProfile(binWidth, binnedQuantity, averagedQuantity, selection=unweighted, origin=0.0):
    return Select.ing(selection,
        SparselyBin.ing(binWidth, binnedQuantity,
            Average.ing(averagedQuantity), Count.ing(), origin))

def ProfileErr(num, low, high, binnedQuantity, averagedQuantity, selection=unweighted):
    return Select.ing(selection,
        Bin.ing(num, low, high, binnedQuantity,
            Deviate.ing(averagedQuantity)))

def SparselyProfileErr(binWidth, binnedQuantity, averagedQuantity, selection=unweighted, origin=0.0):
    return Select.ing(selection,
        SparselyBin.ing(binWidth, binnedQuantity,
            Deviate.ing(averagedQuantity), Count.ing(), origin))

def TwoDimensionallyHistogram(xnum, xlow, xhigh, xquantity,
                              ynum, ylow, yhigh, yquantity,
                              selection=unweighted):
    return Select.ing(selection,
        Bin.ing(xnum, xlow, xhigh, xquantity,
            Bin.ing(ynum, ylow, yhigh, yquantity)))

def TwoDimensionallySparselyHistogram(xbinWidth, xquantity,
                                      ybinWidth, yquantity,
                                      selection=unweighted,
                                      xorigin=0.0, yorigin=0.0):
    return Select.ing(selection,
        SparselyBin.ing(xbinWidth, xquantity,
            SparselyBin.ing(ybinWidth, yquantity,
                Count.ing(), Count.ing(), yorigin), Count.ing(), xorigin))

class HistogramMethods(Bin,
        histogrammar.plot.root.HistogramMethods,
        histogrammar.plot.bokeh.HistogramMethods,
        histogrammar.plot.mpl.HistogramMethods):

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

class SparselyHistogramMethods(SparselyBin,
        histogrammar.plot.root.SparselyHistogramMethods,
        histogrammar.plot.bokeh.SparselyHistogramMethods,
        histogrammar.plot.mpl.SparselyHistogramMethods):

    @property
    def name(self):
        return "SparselyBin"

    @property
    def factory(self):
        return SparselyBin

class ProfileMethods(Bin,
        histogrammar.plot.root.ProfileMethods,
        histogrammar.plot.bokeh.ProfileMethods,
        histogrammar.plot.mpl.ProfileMethods):

    @property
    def name(self):
        return "Bin"

    @property
    def factory(self):
        return Bin

    @property
    def meanValues(self):
        return [v.mean for v in self.values]

    @property
    def numericalOverflow(self):
        return self.overflow.entries

    @property
    def numericalUnderflow(self):
        return self.underflow.entries

    @property
    def numericalNanflow(self):
        return self.nanflow.entries

class SparselyProfileMethods(SparselyBin,
        histogrammar.plot.root.SparselyProfileMethods,
        histogrammar.plot.bokeh.SparselyProfileMethods,
        histogrammar.plot.mpl.SparselyProfileMethods):

    @property
    def name(self):
        return "SparselyBin"

    @property
    def factory(self):
        return SparselyBin

class ProfileErrMethods(Bin,
        histogrammar.plot.root.ProfileErrMethods,
        histogrammar.plot.bokeh.ProfileErrMethods,
        histogrammar.plot.mpl.ProfileErrMethods):

    @property
    def name(self):
        return "Bin"

    @property
    def factory(self):
        return Bin

    @property
    def meanValues(self):
        return [v.mean for v in self.values]

    @property
    def varianceValues(self):
        return [v.variance for v in self.values]

    @property
    def numericalOverflow(self):
        return self.overflow.entries

    @property
    def numericalUnderflow(self):
        return self.underflow.entries

    @property
    def numericalNanflow(self):
        return self.nanflow.entries

class SparselyProfileErrMethods(SparselyBin,
        histogrammar.plot.root.SparselyProfileErrMethods,
        histogrammar.plot.bokeh.SparselyProfileErrMethods,
        histogrammar.plot.mpl.SparselyProfileErrMethods):

    @property
    def name(self):
        return "SparselyBin"

    @property
    def factory(self):
        return SparselyBin

class StackedHistogramMethods(Stack,
        histogrammar.plot.root.StackedHistogramMethods,
        histogrammar.plot.bokeh.StackedHistogramMethods,
        histogrammar.plot.mpl.StackedHistogramMethods):

    @property
    def name(self):
        return "Stack"

    @property
    def factory(self):
        return Stack

class PartitionedHistogramMethods(Partition,
        histogrammar.plot.root.PartitionedHistogramMethods,
        histogrammar.plot.bokeh.PartitionedHistogramMethods,
        histogrammar.plot.mpl.PartitionedHistogramMethods):

    @property
    def name(self):
        return "Partition"

    @property
    def factory(self):
        return Partition

class FractionedHistogramMethods(Fraction,
        histogrammar.plot.root.FractionedHistogramMethods,
        histogrammar.plot.bokeh.FractionedHistogramMethods,
        histogrammar.plot.mpl.FractionedHistogramMethods):

    @property
    def name(self):
        return "Fraction"

    @property
    def factory(self):
        return Fraction

class TwoDimensionallyHistogramMethods(Bin,
        histogrammar.plot.root.TwoDimensionallyHistogramMethods,
        histogrammar.plot.bokeh.TwoDimensionallyHistogramMethods,
        histogrammar.plot.mpl.TwoDimensionallyHistogramMethods):

    @property
    def name(self):
        return "Bin"

    @property
    def factory(self):
        return Bin

class SparselyTwoDimensionallyHistogramMethods(SparselyBin,
        histogrammar.plot.root.SparselyTwoDimensionallyHistogramMethods,
        histogrammar.plot.bokeh.SparselyTwoDimensionallyHistogramMethods,
        histogrammar.plot.mpl.SparselyTwoDimensionallyHistogramMethods):

    @property
    def name(self):
        return "SparselyBin"

    @property
    def factory(self):
        return SparselyBin

def addImplicitMethods(container):
    if isinstance(container, Bin) and all(isinstance(v, Count) for v in container.values):
        container.__class__ = HistogramMethods

    elif isinstance(container, SparselyBin) and container.contentType == "Count" and all(isinstance(v, Count) for v in container.bins.values()):
        container.__class__ = SparselyHistogramMethods

    elif isinstance(container, Bin) and all(isinstance(v, Average) for v in container.values):
        container.__class__ = ProfileMethods

    elif isinstance(container, SparselyBin) and container.contentType == "Average" and all(isinstance(v, Average) for v in container.bins.values()):
        container.__class__ = SparselyProfileMethods

    elif isinstance(container, Bin) and all(isinstance(v, Deviate) for v in container.values):
        container.__class__ = ProfileErrMethods

    elif isinstance(container, SparselyBin) and container.contentType == "Deviate" and all(isinstance(v, Deviate) for v in container.bins.values()):
        container.__class__ = SparselyProfileErrMethods

    elif isinstance(container, Stack) and (
        all(isinstance(v, Bin) and all(isinstance(vv, Count) for vv in v.values) for c, v in container.cuts) or
        all(isinstance(v, Select) and isinstance(v.cut, Bin) and all(isinstance(vv, Count) for vv in v.cut.values) for c, v in container.cuts) or
        all(isinstance(v, SparselyBin) and v.contentType == "Count" and all(isinstance(vv, Count) for vv in v.bins.values()) for c, v in container.cuts) or
        all(isinstance(v, Select) and isinstance(v.cut, SparselyBin) and v.cut.contentType == "Count" and all(isinstance(vv, Count) for vv in v.cut.bins.values()) for c, v in container.cuts)):
        container.__class__ = StackedHistogramMethods

    elif isinstance(container, Partition) and (
        all(isinstance(v, Bin) and all(isinstance(vv, Count) for vv in v.values) for c, v in container.cuts) or
        all(isinstance(v, Select) and isinstance(v.cut, Bin) and all(isinstance(vv, Count) for vv in v.cut.values) for c, v in container.cuts) or
        all(isinstance(v, SparselyBin) and v.contentType == "Count" and all(isinstance(vv, Count) for vv in v.bins.values()) for c, v in container.cuts) or
        all(isinstance(v, Select) and isinstance(v.cut, SparselyBin) and v.cut.contentType == "Count" and all(isinstance(vv, Count) for vv in v.cut.bins.values()) for c, v in container.cuts)):
        container.__class__ = PartitionedHistogramMethods

    elif isinstance(container, Fraction) and (
        (isinstance(container.denominator, Bin) and all(isinstance(v, Count) for v in container.denominator.values)) or
        (isinstance(container.denominator, Select) and isinstance(container.denominator.cut, Bin) and all(isinstance(v, Count) for v in container.denominator.cut.values)) or
        (isinstance(container.denominator, SparselyBin) and container.denominator.contentType == "Count" and all(isinstance(v, Count) for v in container.denominator.bins.values())) or
        (isinstance(container.denominator, Select) and isinstance(container.denominator.cut, SparselyBin) and container.denominator.cut.contentType == "Count" and all(isinstance(v, Count) for v in container.denominator.cut.bins.values()))):
        container.__class__ = FractionedHistogramMethods

    elif isinstance(container, Bin) and all(isinstance(v, Bin) and all(isinstance(vv, Count) for vv in v.values) for v in container.values):
        container.__class__ = TwoDimensionallyHistogramMethods

    elif isinstance(container, SparselyBin) and container.contentType == "SparselyBin" and all(isinstance(v, SparselyBin) and v.contentType == "Count" and all(isinstance(vv, Count) for vv in v.bins.values()) for v in container.bins.values()):
        container.__class__ = SparselyTwoDimensionallyHistogramMethods
