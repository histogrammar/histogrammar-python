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

import histogrammar.plot.root as plotroot
import histogrammar.plot.bokeh as plotbokeh
import histogrammar.plot.matplotlib as plotmpl

from histogrammar.primitives.average import Average
from histogrammar.primitives.bin import Bin
from histogrammar.primitives.count import Count
from histogrammar.primitives.deviate import Deviate
from histogrammar.primitives.fraction import Fraction
from histogrammar.primitives.irregularlybin import IrregularlyBin
from histogrammar.primitives.centrallybin import CentrallyBin
from histogrammar.primitives.select import Select
from histogrammar.primitives.sparselybin import SparselyBin
from histogrammar.primitives.categorize import Categorize
from histogrammar.primitives.stack import Stack

# moved to convenience.py, but imported for backward compatibility
from histogrammar.convenience import Histogram, HistogramCut  # noqa: F401
from histogrammar.convenience import SparselyHistogram  # noqa: F401
from histogrammar.convenience import CategorizeHistogram  # noqa: F401
from histogrammar.convenience import Profile, SparselyProfile  # noqa: F401
from histogrammar.convenience import ProfileErr, SparselyProfileErr  # noqa: F401
from histogrammar.convenience import TwoDimensionallyHistogram  # noqa: F401
from histogrammar.convenience import TwoDimensionallySparselyHistogram  # noqa: F401

COMMON_PLOT_TYPES = (Count, Bin, SparselyBin, Categorize, IrregularlyBin, CentrallyBin)


# 1d plotting of counts + generic 2d plotting of counts

class HistogramMethods(Bin,
                       plotroot.HistogramMethods,
                       plotbokeh.HistogramMethods,
                       plotmpl.HistogramMethods):
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

    def confidenceIntervalValues(self, absz=1.0):
        from math import sqrt
        return map(lambda v: absz*sqrt(v), self.numericalValues)


class SparselyHistogramMethods(SparselyBin,
                               plotroot.SparselyHistogramMethods,
                               plotbokeh.SparselyHistogramMethods,
                               plotmpl.SparselyHistogramMethods):

    """Methods that are implicitly added to container combinations that look like sparsely binned histograms."""

    @property
    def name(self):
        return "SparselyBin"

    @property
    def factory(self):
        return SparselyBin

    def confidenceIntervalValues(self, absz=1.0):
        from math import sqrt
        return map(lambda v: absz * sqrt(v), [v.entries for _, v in sorted(self.bins.items())])


class CategorizeHistogramMethods(Categorize,
                                 plotroot.CategorizeHistogramMethods,
                                 plotbokeh.CategorizeHistogramMethods,
                                 plotmpl.CategorizeHistogramMethods):

    """Methods that are implicitly added to container combinations that look like categorical histograms."""

    @property
    def name(self):
        return "Categorize"

    @property
    def factory(self):
        return Categorize


class IrregularlyHistogramMethods(IrregularlyBin,
                                  plotmpl.IrregularlyHistogramMethods):
    """Methods that are implicitly added to container combinations that look like partitioned histograms."""

    @property
    def name(self):
        return "IrregularlyBin"

    @property
    def factory(self):
        return IrregularlyBin


class CentrallyHistogramMethods(CentrallyBin,
                                plotmpl.CentrallyHistogramMethods):
    """Methods that are implicitly added to containers that look like centrally histograms."""

    @property
    def name(self):
        return "CentrallyBin"

    @property
    def factory(self):
        return CentrallyBin


# specialized 2d plotting of counts

class TwoDimensionallyHistogramMethods(Bin,
                                       plotroot.TwoDimensionallyHistogramMethods,
                                       plotbokeh.TwoDimensionallyHistogramMethods,
                                       plotmpl.TwoDimensionallyHistogramMethods):
    """Convenience function for creating a conventional, two-dimensional histogram."""

    @property
    def name(self):
        return "Bin"

    @property
    def factory(self):
        return Bin


class SparselyTwoDimensionallyHistogramMethods(SparselyBin,
                                               plotroot.SparselyTwoDimensionallyHistogramMethods,
                                               plotbokeh.SparselyTwoDimensionallyHistogramMethods,
                                               plotmpl.SparselyTwoDimensionallyHistogramMethods):
    """Convenience function for creating a sparsely binned, two-dimensional histogram."""

    @property
    def name(self):
        return "SparselyBin"

    @property
    def factory(self):
        return SparselyBin


class IrregularlyTwoDimensionallyHistogramMethods(IrregularlyBin,
                                                  plotmpl.IrregularlyTwoDimensionallyHistogramMethods):
    """Convenience function for creating a sparsely binned, two-dimensional histogram."""

    @property
    def name(self):
        return "IrregularlyBin"

    @property
    def factory(self):
        return IrregularlyBin


# 1d plotting of profiles

class ProfileMethods(Bin,
                     plotroot.ProfileMethods,
                     plotbokeh.ProfileMethods,
                     plotmpl.ProfileMethods):
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
                             plotroot.SparselyProfileMethods,
                             plotbokeh.SparselyProfileMethods,
                             plotmpl.SparselyProfileMethods):
    '''Methods that are implicitly added to container combinations that look like a sparsely
    binned physicist's "profile plot."'''

    @property
    def name(self):
        return "SparselyBin"

    @property
    def factory(self):
        return SparselyBin


class ProfileErrMethods(Bin,
                        plotroot.ProfileErrMethods,
                        plotbokeh.ProfileErrMethods,
                        plotmpl.ProfileErrMethods):

    '''Methods that are implicitly added to container combinations that look like a physicist's "profile plot."'''

    @property
    def name(self):
        return "Bin"

    @property
    def factory(self):
        return Bin

    @property
    def meanValues(self):
        """Bin means as numbers"""
        return [v.mean for v in self.values]

    @property
    def varianceValues(self):
        """Bin variances as numbers"""
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
                                plotroot.SparselyProfileErrMethods,
                                plotbokeh.SparselyProfileErrMethods,
                                plotmpl.SparselyProfileErrMethods):
    '''Methods that are implicitly added to container combinations that look like a sparsely binned profile plot."'''

    @property
    def name(self):
        return "SparselyBin"

    @property
    def factory(self):
        return SparselyBin


# other 1d/2d plotting

class StackedHistogramMethods(Stack,
                              plotroot.StackedHistogramMethods,
                              plotbokeh.StackedHistogramMethods,
                              plotmpl.StackedHistogramMethods):
    """Methods that are implicitly added to container combinations that look like stacked histograms."""

    @property
    def name(self):
        return "Stack"

    @property
    def factory(self):
        return Stack


class PartitionedHistogramMethods(IrregularlyBin,
                                  plotroot.PartitionedHistogramMethods,
                                  plotbokeh.PartitionedHistogramMethods,
                                  plotmpl.PartitionedHistogramMethods):
    """Methods that are implicitly added to container combinations that look like partitioned histograms."""

    @property
    def name(self):
        return "IrregularlyBin"

    @property
    def factory(self):
        return IrregularlyBin


class FractionedHistogramMethods(Fraction,
                                 plotroot.FractionedHistogramMethods,
                                 plotbokeh.FractionedHistogramMethods,
                                 plotmpl.FractionedHistogramMethods):
    """Methods that are implicitly added to container combinations that look like fractioned histograms."""

    @property
    def name(self):
        return "Fraction"

    @property
    def factory(self):
        return Fraction


def addImplicitMethods(container):
    """Adds methods for each of the plotting front-ends on recognized combinations of primitives.

    Every histogrammar.defs.Container's constructor invokes these soon after it is constructed
    (in its ``specialize`` method), except for early code that can't resolve dependencies.
    (histogrammar.primitives.count.Count objects created as default parameter values for containers
    like histogrammar.primitives.bin.Bin are created before the histogrammar.specialized module can be created.
    These don't get checked by ``addImplicitMethods``, but they don't have any implicit methods to add, either.

    This function emulates Scala's "pimp my library" pattern, though ``addImplicitMethods`` has to be explicitly
    invoked and binds early, rather than late.
    """
    # specialized 2d plotting of counts
    if isinstance(container, Bin) and all(isinstance(v, Bin) and all(isinstance(vv, Count)
                                                                     for vv in v.values) for v in container.values):
        container.__class__ = TwoDimensionallyHistogramMethods

    elif isinstance(container, SparselyBin) and container.contentType == "SparselyBin" and \
            all(isinstance(v, SparselyBin) and v.contentType == "Count" and
                all(isinstance(vv, Count) for vv in v.bins.values()) for v in container.bins.values()):
        container.__class__ = SparselyTwoDimensionallyHistogramMethods

    elif isinstance(container, IrregularlyBin) and \
            all(isinstance(v, IrregularlyBin) and all(isinstance(vv, Count)
                                                      for _j, vv in v.bins) for _i, v in container.bins):
        container.__class__ = IrregularlyTwoDimensionallyHistogramMethods

    # 1d plotting of profiles
    elif isinstance(container, Bin) and all(isinstance(v, Average) for v in container.values):
        container.__class__ = ProfileMethods

    elif isinstance(container, SparselyBin) and \
            container.contentType == "Average" and \
            all(isinstance(v, Average) for v in container.bins.values()):
        container.__class__ = SparselyProfileMethods

    elif isinstance(container, Bin) and all(isinstance(v, Deviate) for v in container.values):
        container.__class__ = ProfileErrMethods

    elif isinstance(container, SparselyBin) and \
            container.contentType == "Deviate" and \
            all(isinstance(v, Deviate) for v in container.bins.values()):
        container.__class__ = SparselyProfileErrMethods

    # other 1d/2d plotting
    elif isinstance(container, Stack) and (
            all(isinstance(v, Bin) and all(isinstance(vv, Count) for vv in v.values) for c, v in container.bins) or
            all(isinstance(v, Select) and
                isinstance(v.cut, Bin) and
                all(isinstance(vv, Count) for vv in v.cut.values) for c, v in container.bins) or
            all(isinstance(v, SparselyBin) and
                v.contentType == "Count" and
                all(isinstance(vv, Count) for vv in v.bins.values()) for c, v in container.bins) or
            all(isinstance(v, Select) and
                isinstance(v.cut, SparselyBin) and
                v.cut.contentType == "Count" and
                all(isinstance(vv, Count) for vv in v.cut.bins.values()) for c, v in container.bins)):
        container.__class__ = StackedHistogramMethods

    elif isinstance(container, IrregularlyBin) and (
            all(isinstance(v, Bin) and
                all(isinstance(vv, Count) for vv in v.values) for c, v in container.bins) or
            all(isinstance(v, Select) and isinstance(v.cut, Bin) and
                all(isinstance(vv, Count) for vv in v.cut.values) for c, v in container.bins) or
            all(isinstance(v, SparselyBin) and
                v.contentType == "Count" and
                all(isinstance(vv, Count) for vv in v.bins.values()) for c, v in container.bins) or
            all(isinstance(v, Select) and
                isinstance(v.cut, SparselyBin) and
                v.cut.contentType == "Count" and
                all(isinstance(vv, Count) for vv in v.cut.bins.values()) for c, v in container.bins)):
        container.__class__ = PartitionedHistogramMethods

    elif isinstance(container, Fraction) and (
        (isinstance(container.denominator, Bin) and
         all(isinstance(v, Count) for v in container.denominator.values)) or
        (isinstance(container.denominator, Select) and
         isinstance(container.denominator.cut, Bin) and
         all(isinstance(v, Count) for v in container.denominator.cut.values)) or
        (isinstance(container.denominator, SparselyBin) and
         container.denominator.contentType == "Count" and
         all(isinstance(v, Count) for v in container.denominator.bins.values())) or
            (isinstance(container.denominator, Select) and
             isinstance(container.denominator.cut, SparselyBin) and
             container.denominator.cut.contentType == "Count" and
             all(isinstance(v, Count) for v in container.denominator.cut.bins.values()))):
        container.__class__ = FractionedHistogramMethods

    # 1d plotting of counts + generic 2d plotting of counts
    elif isinstance(container, Bin) and all(isinstance(v, COMMON_PLOT_TYPES) for v in container.values):
        container.__class__ = HistogramMethods

    elif isinstance(container, SparselyBin) and all(isinstance(v, COMMON_PLOT_TYPES) for v in container.bins.values()):
        container.__class__ = SparselyHistogramMethods

    elif isinstance(container, Categorize) and all(isinstance(v, COMMON_PLOT_TYPES) for v in container.bins.values()):
        container.__class__ = CategorizeHistogramMethods

    elif isinstance(container, IrregularlyBin) and all(isinstance(v, COMMON_PLOT_TYPES) for _, v in container.bins):
        container.__class__ = IrregularlyHistogramMethods

    elif isinstance(container, CentrallyBin) and container.bins is not None and \
            all(isinstance(v, COMMON_PLOT_TYPES) for _, v in container.bins):
        container.__class__ = CentrallyHistogramMethods
