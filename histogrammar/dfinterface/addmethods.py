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

import json
import types
import inspect

from ..primitives.average import Average as hg_Average
from ..primitives.bag import Bag as hg_Bag
from ..primitives.bin import Bin as hg_Bin
from ..primitives.categorize import Categorize as hg_Categorize
from ..primitives.centrallybin import CentrallyBin as hg_CentrallyBin
from ..primitives.collection import Label as hg_Label
from ..primitives.collection import UntypedLabel as hg_UntypedLabel
from ..primitives.collection import Branch as hg_Branch
from ..primitives.collection import Index as hg_Index
from ..primitives.count import Count as hg_Count
from ..primitives.deviate import Deviate as hg_Deviate
from ..primitives.fraction import Fraction as hg_Fraction
from ..primitives.irregularlybin import IrregularlyBin as hg_IrregularlyBin
from ..primitives.minmax import Maximize as hg_Maximize, Minimize as hg_Minimize
from ..primitives.select import Select as hg_Select
from ..primitives.sparselybin import SparselyBin as hg_SparselyBin
from ..primitives.stack import Stack as hg_Stack
from ..primitives.sum import Sum as hg_Sum

from ..convenience import Histogram as hg_Histogram
from ..convenience import SparselyHistogram as hg_SparselyHistogram
from ..convenience import CategorizeHistogram as hg_CategorizeHistogram
from ..convenience import Profile as hg_Profile
from ..convenience import SparselyProfile as hg_SparselyProfile
from ..convenience import ProfileErr as hg_ProfileErr
from ..convenience import SparselyProfileErr as hg_SparselyProfileErr
from ..convenience import TwoDimensionallyHistogram as hg_TwoDimensionallyHistogram
from ..convenience import TwoDimensionallySparselyHistogram as hg_TwoDimensionallySparselyHistogram


from ..defs import Factory
from .make_histograms import make_histograms


def add_sparksql_methods(cls, prefix=''):
    add_methods(cls=cls, hg=hg_fill_sparksql, prefix=prefix)


def add_pandas_methods(cls, prefix=''):
    add_methods(cls=cls, hg=hg_fill_numpy, prefix=prefix)


def add_methods(cls, hg, prefix='hg_'):
    def Average(self, quantity):
        return self.histogrammar(hg_Average(quantity))

    def Bag(self, quantity, range):
        return self.histogrammar(hg_Bag(quantity, range))

    def Bin(self, num, low, high, quantity, value=hg_Count(),
            underflow=hg_Count(), overflow=hg_Count(), nanflow=hg_Count()):
        return self.histogrammar(hg_Bin(num, low, high, quantity, value, underflow, overflow, nanflow))

    def Categorize(self, quantity, value=hg_Count()):
        return self.histogrammar(hg_Categorize(quantity, value))

    def CentrallyBin(self, centers, quantity, value=hg_Count(), nanflow=hg_Count()):
        return self.histogrammar(hg_CentrallyBin(centers, quantity, value, nanflow))

    def Label(self, **pairs):
        return self.histogrammar(hg_Label(**pairs))

    def UntypedLabel(self, **pairs):
        return self.histogrammar(hg_UntypedLabel(**pairs))

    def Index(self, *values):
        return self.histogrammar(hg_Index(*values))

    def Branch(self, *values):
        return self.histogrammar(hg_Branch(*values))

    def Count(self):    # TODO: handle transform
        return self.histogrammar(hg_Count())

    def Deviate(self, quantity):
        return self.histogrammar(hg_Deviate(quantity))

    def Fraction(self, quantity, value=hg_Count()):
        return self.histogrammar(hg_Fraction(quantity, value))

    def IrregularlyBin(self, edges, quantity, value=hg_Count(), nanflow=hg_Count()):
        return self.histogrammar(hg_IrregularlyBin(edges, quantity, value, nanflow))

    def Minimize(self, quantity):
        return self.histogrammar(hg_Minimize(quantity))

    def Maximize(self, quantity):
        return self.histogrammar(hg_Maximize(quantity))

    def Select(self, quantity, cut=hg_Count()):
        return self.histogrammar(hg_Select(quantity, cut))

    def SparselyBin(self, binWidth, quantity, value=hg_Count(), nanflow=hg_Count(), origin=0.0):
        return self.histogrammar(hg_SparselyBin(binWidth, quantity, value, nanflow, origin))

    def Stack(self, thresholds, quantity, value=hg_Count(), nanflow=hg_Count()):
        return self.histogrammar(hg_Stack(thresholds, quantity, value, nanflow))

    def Sum(self, quantity):
        return self.histogrammar(hg_Sum(quantity))

    # convenience functions
    def Histogram(self, num, low, high, quantity):
        return self.histogrammar(hg_Histogram(num, low, high, quantity))

    def SparselyHistogram(self, binWidth, quantity, origin=0.0):
        return self.histogrammar(hg_SparselyHistogram(binWidth, quantity, origin))

    def CategorizeHistogram(self, quantity):
        return self.histogrammar(hg_CategorizeHistogram(quantity))

    def Profile(self, num, low, high, binnedQuantity, averagedQuantity):
        return self.histogrammar(hg_Profile(num, low, high, binnedQuantity, averagedQuantity))

    def SparselyProfile(self, binWidth, binnedQuantity, averagedQuantity, origin=0.0):
        return self.histogrammar(hg_SparselyProfile(binWidth, binnedQuantity, averagedQuantity, origin))

    def ProfileErr(self, num, low, high, binnedQuantity, averagedQuantity):
        return self.histogrammar(hg_ProfileErr(num, low, high, binnedQuantity, averagedQuantity))

    def SparselyProfileErr(self, binWidth, binnedQuantity, averagedQuantity, origin=0.0):
        return self.histogrammar(hg_SparselyProfileErr(binWidth, binnedQuantity, averagedQuantity, origin))

    def TwoDimensionallyHistogram(self, xnum, xlow, xhigh, xquantity, ynum, ylow, yhigh, yquantity):
        return self.histogrammar(hg_TwoDimensionallyHistogram(xnum, xlow, xhigh, xquantity, ynum, ylow, yhigh,
                                                              yquantity))

    def TwoDimensionallySparselyHistogram(self, xbinWidth, xquantity, ybinWidth, yquantity, xorigin=0.0, yorigin=0.0):
        return self.histogrammar(hg_TwoDimensionallySparselyHistogram(xbinWidth, xquantity, ybinWidth, yquantity,
                                                                      xorigin, yorigin))

    if inspect.isclass(cls):
        # pure class, not instantiated
        setattr(cls, 'histogrammar', hg)
        setattr(cls, prefix + 'Average', Average)
        setattr(cls, prefix + 'Bag', Bag)
        setattr(cls, prefix + 'Bin', Bin)
        setattr(cls, prefix + 'Categorize', Categorize)
        setattr(cls, prefix + 'CentrallyBin', CentrallyBin)
        setattr(cls, prefix + 'Label', Label)
        setattr(cls, prefix + 'UntypedLabel', UntypedLabel)
        setattr(cls, prefix + 'Index', Index)
        setattr(cls, prefix + 'Branch', Branch)
        setattr(cls, prefix + 'Count', Count)
        setattr(cls, prefix + 'Deviate', Deviate)
        setattr(cls, prefix + 'Fraction', Fraction)
        setattr(cls, prefix + 'IrregularlyBin', IrregularlyBin)
        setattr(cls, prefix + 'Minimize', Minimize)
        setattr(cls, prefix + 'Maximize', Maximize)
        setattr(cls, prefix + 'Select', Select)
        setattr(cls, prefix + 'SparselyBin', SparselyBin)
        setattr(cls, prefix + 'Stack', Stack)
        setattr(cls, prefix + 'Sum', Sum)
        setattr(cls, prefix + 'make_histograms', make_histograms)
        setattr(cls, prefix + 'Histogram', Histogram)
        setattr(cls, prefix + 'SparselyHistogram', SparselyHistogram)
        setattr(cls, prefix + 'CategorizeHistogram', CategorizeHistogram)
        setattr(cls, prefix + 'Profile', Profile)
        setattr(cls, prefix + 'SparselyProfile', SparselyProfile)
        setattr(cls, prefix + 'ProfileErr', ProfileErr)
        setattr(cls, prefix + 'SparselyProfileErr', SparselyProfileErr)
        setattr(cls, prefix + 'TwoDimensionallyHistogram', TwoDimensionallyHistogram)
        setattr(cls, prefix + 'TwoDimensionallySparselyHistogram', TwoDimensionallySparselyHistogram)
    else:
        # instantiated class
        setattr(cls, 'histogrammar', types.MethodType(hg, cls))
        setattr(cls, prefix + 'Average', types.MethodType(Average, cls))
        setattr(cls, prefix + 'Bag', types.MethodType(Bag, cls))
        setattr(cls, prefix + 'Bin', types.MethodType(Bin, cls))
        setattr(cls, prefix + 'Categorize', types.MethodType(Categorize, cls))
        setattr(cls, prefix + 'CentrallyBin', types.MethodType(CentrallyBin, cls))
        setattr(cls, prefix + 'Label', types.MethodType(Label, cls))
        setattr(cls, prefix + 'UntypedLabel', types.MethodType(UntypedLabel, cls))
        setattr(cls, prefix + 'Index', types.MethodType(Index, cls))
        setattr(cls, prefix + 'Branch', types.MethodType(Branch, cls))
        setattr(cls, prefix + 'Count', types.MethodType(Count, cls))
        setattr(cls, prefix + 'Deviate', types.MethodType(Deviate, cls))
        setattr(cls, prefix + 'Fraction', types.MethodType(Fraction, cls))
        setattr(cls, prefix + 'IrregularlyBin', types.MethodType(IrregularlyBin, cls))
        setattr(cls, prefix + 'Minimize', types.MethodType(Minimize, cls))
        setattr(cls, prefix + 'Maximize', types.MethodType(Maximize, cls))
        setattr(cls, prefix + 'Select', types.MethodType(Select, cls))
        setattr(cls, prefix + 'SparselyBin', types.MethodType(SparselyBin, cls))
        setattr(cls, prefix + 'Stack', types.MethodType(Stack, cls))
        setattr(cls, prefix + 'Sum', types.MethodType(Sum, cls))
        setattr(cls, prefix + 'make_histograms', types.MethodType(make_histograms, cls))
        setattr(cls, prefix + 'Histogram', types.MethodType(Histogram, cls))
        setattr(cls, prefix + 'SparselyHistogram', types.MethodType(SparselyHistogram, cls))
        setattr(cls, prefix + 'CategorizeHistogram', types.MethodType(CategorizeHistogram, cls))
        setattr(cls, prefix + 'Profile', types.MethodType(Profile, cls))
        setattr(cls, prefix + 'SparselyProfile', types.MethodType(SparselyProfile, cls))
        setattr(cls, prefix + 'ProfileErr', types.MethodType(ProfileErr, cls))
        setattr(cls, prefix + 'SparselyProfileErr', types.MethodType(SparselyProfileErr, cls))
        setattr(cls, prefix + 'TwoDimensionallyHistogram', types.MethodType(TwoDimensionallyHistogram, cls))
        setattr(cls, prefix + 'TwoDimensionallySparselyHistogram',
                types.MethodType(TwoDimensionallySparselyHistogram, cls))


def hg_fill_sparksql(self, hist):
    hist.fill.sparksql(self)
    return hist


def hg_fill_numpy(self, hist):
    hist.fill.numpy(self)
    return hist


def hg(self, h):
    # alternative for spark
    converter = self._sc._jvm.org.dianahep.histogrammar.sparksql.pyspark.AggregatorConverter()
    agg = h._sparksql(self._sc._jvm, converter)
    result = converter.histogrammar(self._jdf, agg)
    return Factory.fromJson(json.loads(result.toJsonString()))
