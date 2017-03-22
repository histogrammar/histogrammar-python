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

from histogrammar.defs import unweighted
from histogrammar.primitives.average import Average
from histogrammar.primitives.bin import Bin
from histogrammar.primitives.count import Count
from histogrammar.primitives.deviate import Deviate
from histogrammar.primitives.fraction import Fraction
from histogrammar.primitives.irregularlybin import IrregularlyBin
from histogrammar.primitives.select import Select
from histogrammar.primitives.sparselybin import SparselyBin
from histogrammar.primitives.categorize import Categorize
from histogrammar.primitives.stack import Stack
from histogrammar.util import serializable

import histogrammar.plot.root
import histogrammar.plot.bokeh
import histogrammar.plot.matplotlib

def Histogram(num, low, high, quantity, selection=unweighted):
    """Convenience function for creating a conventional histogram."""
    return Select.ing(selection, Bin.ing(num, low, high, quantity,
        Count.ing(), Count.ing(), Count.ing(), Count.ing()))

def SparselyHistogram(binWidth, quantity, selection=unweighted, origin=0.0):
    """Convenience function for creating a sparsely binned histogram."""
    return Select.ing(selection,
        SparselyBin.ing(binWidth, quantity, Count.ing(), Count.ing(), origin))

def CategorizeHistogram(quantity, selection=unweighted):
    """Convenience function for creating a categorize histogram."""
    return Select.ing(selection, Categorize.ing(quantity, Count.ing()))

def Profile(num, low, high, binnedQuantity, averagedQuantity, selection=unweighted):
    """Convenience function for creating binwise averages."""
    return Select.ing(selection,
        Bin.ing(num, low, high, binnedQuantity,
            Average.ing(averagedQuantity)))

def SparselyProfile(binWidth, binnedQuantity, averagedQuantity, selection=unweighted, origin=0.0):
    """Convenience function for creating sparsely binned binwise averages."""
    return Select.ing(selection,
        SparselyBin.ing(binWidth, binnedQuantity,
            Average.ing(averagedQuantity), Count.ing(), origin))

def ProfileErr(num, low, high, binnedQuantity, averagedQuantity, selection=unweighted):
    """Convenience function for creating a physicist's "profile plot," which is a Profile with variances."""
    return Select.ing(selection,
        Bin.ing(num, low, high, binnedQuantity,
            Deviate.ing(averagedQuantity)))

def SparselyProfileErr(binWidth, binnedQuantity, averagedQuantity, selection=unweighted, origin=0.0):
    """Convenience function for creating a physicist's sparsely binned "profile plot," which is a Profile with variances."""
    return Select.ing(selection,
        SparselyBin.ing(binWidth, binnedQuantity,
            Deviate.ing(averagedQuantity), Count.ing(), origin))

def TwoDimensionallyHistogram(xnum, xlow, xhigh, xquantity,
                              ynum, ylow, yhigh, yquantity,
                              selection=unweighted):
    """Convenience function for creating a conventional, two-dimensional histogram."""
    return Select.ing(selection,
        Bin.ing(xnum, xlow, xhigh, xquantity,
            Bin.ing(ynum, ylow, yhigh, yquantity)))

def TwoDimensionallySparselyHistogram(xbinWidth, xquantity,
                                      ybinWidth, yquantity,
                                      selection=unweighted,
                                      xorigin=0.0, yorigin=0.0):
    """Convenience function for creating a sparsely binned, two-dimensional histogram."""
    return Select.ing(selection,
        SparselyBin.ing(xbinWidth, xquantity,
            SparselyBin.ing(ybinWidth, yquantity,
                Count.ing(), Count.ing(), yorigin), Count.ing(), xorigin))

class HistogramMethods(Bin,
        histogrammar.plot.root.HistogramMethods,
        histogrammar.plot.bokeh.HistogramMethods,
        histogrammar.plot.matplotlib.HistogramMethods):
    """Methods that are implicitly added to container combinations that look like histograms."""

    @property
    def name(self):
        return "Bin"

    @property
    def factory(self):
        return Bin

    @property
    def numericalValues(self):
        """Bin values as numbers, rather than histogrammar.primitives.count.Count."""
        return [v.entries for v in self.values]

    @property
    def numericalOverflow(self):
        """Overflow as a number, rather than histogrammar.primitives.count.Count."""
        return self.overflow.entries

    @property
    def numericalUnderflow(self):
        """Underflow as a number, rather than histogrammar.primitives.count.Count."""
        return self.underflow.entries

    @property
    def numericalNanflow(self):
        """Nanflow as a number, rather than histogrammar.primitives.count.Count."""
        return self.nanflow.entries

    def confidenceIntervalValues(self,absz=1.0): 
        from math import sqrt
        return map(lambda v: absz*sqrt(v), self.numericalValues)

class SparselyHistogramMethods(SparselyBin,
        histogrammar.plot.root.SparselyHistogramMethods,
        histogrammar.plot.bokeh.SparselyHistogramMethods,
        histogrammar.plot.matplotlib.SparselyHistogramMethods):

    """Methods that are implicitly added to container combinations that look like sparsely binned histograms."""

    @property
    def name(self):
        return "SparselyBin"

    @property
    def factory(self):
        return SparselyBin

    def confidenceIntervalValues(self,absz=1.0):
        from math import sqrt
        return map(lambda v: absz*sqrt(v), [v.entries for _, v in sorted(self.bins.items())])

class CategorizeHistogramMethods(Categorize,
                                 histogrammar.plot.root.CategorizeHistogramMethods,
                                 histogrammar.plot.bokeh.CategorizeHistogramMethods,
                                 histogrammar.plot.matplotlib.CategorizeHistogramMethods):

    """Methods that are implicitly added to container combinations that look like categorical histograms."""

    @property
    def name(self):
        return "Categorize"

    @property
    def factory(self):
        return Categorize

class ProfileMethods(Bin,
        histogrammar.plot.root.ProfileMethods,
        histogrammar.plot.bokeh.ProfileMethods,
        histogrammar.plot.matplotlib.ProfileMethods):
    '''Methods that are implicitly added to container combinations that look like a physicist's "profile plot."'''

    @property
    def name(self):
        return "Bin"

    @property
    def factory(self):
        return Bin

    @property
    def meanValues(self):
        """Bin means as numbers, rather than histogrammar.primitives.average.Average."""
        return [v.mean for v in self.values]

    @property
    def numericalOverflow(self):
        """Overflow as a number, rather than histogrammar.primitives.count.Count."""
        return self.overflow.entries

    @property
    def numericalUnderflow(self):
        """Underflow as a number, rather than histogrammar.primitives.count.Count."""
        return self.underflow.entries

    @property
    def numericalNanflow(self):
        """Nanflow as a number, rather than histogrammar.primitives.count.Count."""
        return self.nanflow.entries

class SparselyProfileMethods(SparselyBin,
        histogrammar.plot.root.SparselyProfileMethods,
        histogrammar.plot.bokeh.SparselyProfileMethods,
        histogrammar.plot.matplotlib.SparselyProfileMethods):
    '''Methods that are implicitly added to container combinations that look like a sparsely binned physicist's "profile plot."'''

    @property
    def name(self):
        return "SparselyBin"

    @property
    def factory(self):
        return SparselyBin

class ProfileErrMethods(Bin,
        histogrammar.plot.root.ProfileErrMethods,
        histogrammar.plot.bokeh.ProfileErrMethods,
        histogrammar.plot.matplotlib.ProfileErrMethods):

    '''Methods that are implicitly added to container combinations that look like a physicist's "profile plot."'''

    @property
    def name(self):
        return "Bin"

    @property
    def factory(self):
        return Bin

    @property
    def meanValues(self):
        """Bin means as numbers, rather than [[org.dianahep.histogrammar.Deviated]]/[[org.dianahep.histogrammar.Deviating]]."""
        return [v.mean for v in self.values]

    @property
    def varianceValues(self):
        """Bin variances as numbers, rather than [[org.dianahep.histogrammar.Deviated]]/[[org.dianahep.histogrammar.Deviating]]."""
        return [v.variance for v in self.values]

    @property
    def numericalOverflow(self):
        """Overflow as a number, rather than histogrammar.primitives.count.Count."""
        return self.overflow.entries

    @property
    def numericalUnderflow(self):
        """Underflow as a number, rather than histogrammar.primitives.count.Count."""
        return self.underflow.entries

    @property
    def numericalNanflow(self):
        """Nanflow as a number, rather than histogrammar.primitives.count.Count."""
        return self.nanflow.entries

class SparselyProfileErrMethods(SparselyBin,
        histogrammar.plot.root.SparselyProfileErrMethods,
        histogrammar.plot.bokeh.SparselyProfileErrMethods,
        histogrammar.plot.matplotlib.SparselyProfileErrMethods):

    '''Methods that are implicitly added to container combinations that look like a sparsely binned physicist's "profile plot."'''

    @property
    def name(self):
        return "SparselyBin"

    @property
    def factory(self):
        return SparselyBin

class StackedHistogramMethods(Stack,
        histogrammar.plot.root.StackedHistogramMethods,
        histogrammar.plot.bokeh.StackedHistogramMethods,
        histogrammar.plot.matplotlib.StackedHistogramMethods):
    """Methods that are implicitly added to container combinations that look like stacked histograms."""

    @property
    def name(self):
        return "Stack"

    @property
    def factory(self):
        return Stack

class PartitionedHistogramMethods(IrregularlyBin,
        histogrammar.plot.root.PartitionedHistogramMethods,
        histogrammar.plot.bokeh.PartitionedHistogramMethods,
        histogrammar.plot.matplotlib.PartitionedHistogramMethods):
    """Methods that are implicitly added to container combinations that look like partitioned histograms."""

    @property
    def name(self):
        return "IrregularlyBin"

    @property
    def factory(self):
        return IrregularlyBin

class FractionedHistogramMethods(Fraction,
        histogrammar.plot.root.FractionedHistogramMethods,
        histogrammar.plot.bokeh.FractionedHistogramMethods,
        histogrammar.plot.matplotlib.FractionedHistogramMethods):
    """Methods that are implicitly added to container combinations that look like fractioned histograms."""

    @property
    def name(self):
        return "Fraction"

    @property
    def factory(self):
        return Fraction

class TwoDimensionallyHistogramMethods(Bin,
        histogrammar.plot.root.TwoDimensionallyHistogramMethods,
        histogrammar.plot.bokeh.TwoDimensionallyHistogramMethods,
        histogrammar.plot.matplotlib.TwoDimensionallyHistogramMethods):
    """Convenience function for creating a conventional, two-dimensional histogram."""

    @property
    def name(self):
        return "Bin"

    @property
    def factory(self):
        return Bin

class SparselyTwoDimensionallyHistogramMethods(SparselyBin,
        histogrammar.plot.root.SparselyTwoDimensionallyHistogramMethods,
        histogrammar.plot.bokeh.SparselyTwoDimensionallyHistogramMethods,
        histogrammar.plot.matplotlib.SparselyTwoDimensionallyHistogramMethods):
    """Convenience function for creating a sparsely binned, two-dimensional histogram."""

    @property
    def name(self):
        return "SparselyBin"

    @property
    def factory(self):
        return SparselyBin

def addImplicitMethods(container):
    """Adds methods for each of the plotting front-ends on recognized combinations of primitives.

    Every histogrammar.defs.Container's constructor invokes these soon after it is constructed (in its ``specialize`` method), except for early code that can't resolve dependencies. (histogrammar.primitives.count.Count objects created as default parameter values for containers like histogrammar.primitives.bin.Bin are created before the histogrammar.specialized module can be created. These don't get checked by ``addImplicitMethods``, but they don't have any implicit methods to add, either.

    This function emulates Scala's "pimp my library" pattern, though ``addImplicitMethods`` has to be explicitly invoked and binds early, rather than late.
    """

    if isinstance(container, Bin) and all(isinstance(v, Count) for v in container.values):
        container.__class__ = HistogramMethods

    elif isinstance(container, SparselyBin) and container.contentType == "Count" and all(isinstance(v, Count) for v in container.bins.values()):
        container.__class__ = SparselyHistogramMethods

    elif isinstance(container, Categorize) and container.contentType == "Count" and all(isinstance(v, Count) for v in container.bins.values()):
        container.__class__ = CategorizeHistogramMethods

    elif isinstance(container, Bin) and all(isinstance(v, Average) for v in container.values):
        container.__class__ = ProfileMethods

    elif isinstance(container, SparselyBin) and container.contentType == "Average" and all(isinstance(v, Average) for v in container.bins.values()):
        container.__class__ = SparselyProfileMethods

    elif isinstance(container, Bin) and all(isinstance(v, Deviate) for v in container.values):
        container.__class__ = ProfileErrMethods

    elif isinstance(container, SparselyBin) and container.contentType == "Deviate" and all(isinstance(v, Deviate) for v in container.bins.values()):
        container.__class__ = SparselyProfileErrMethods

    elif isinstance(container, Stack) and (
        all(isinstance(v, Bin) and all(isinstance(vv, Count) for vv in v.values) for c, v in container.bins) or
        all(isinstance(v, Select) and isinstance(v.cut, Bin) and all(isinstance(vv, Count) for vv in v.cut.values) for c, v in container.bins) or
        all(isinstance(v, SparselyBin) and v.contentType == "Count" and all(isinstance(vv, Count) for vv in v.bins.values()) for c, v in container.bins) or
        all(isinstance(v, Select) and isinstance(v.cut, SparselyBin) and v.cut.contentType == "Count" and all(isinstance(vv, Count) for vv in v.cut.bins.values()) for c, v in container.bins)):
        container.__class__ = StackedHistogramMethods

    elif isinstance(container, IrregularlyBin) and (
        all(isinstance(v, Bin) and all(isinstance(vv, Count) for vv in v.values) for c, v in container.bins) or
        all(isinstance(v, Select) and isinstance(v.cut, Bin) and all(isinstance(vv, Count) for vv in v.cut.values) for c, v in container.bins) or
        all(isinstance(v, SparselyBin) and v.contentType == "Count" and all(isinstance(vv, Count) for vv in v.bins.values()) for c, v in container.bins) or
        all(isinstance(v, Select) and isinstance(v.cut, SparselyBin) and v.cut.contentType == "Count" and all(isinstance(vv, Count) for vv in v.cut.bins.values()) for c, v in container.bins)):
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
