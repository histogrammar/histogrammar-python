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

import histogrammar.primitives.average
import histogrammar.primitives.bag
import histogrammar.primitives.bin
import histogrammar.primitives.categorize
import histogrammar.primitives.centrallybin
import histogrammar.primitives.collection
import histogrammar.primitives.count
import histogrammar.primitives.deviate
import histogrammar.primitives.fraction
import histogrammar.primitives.irregularlybin
import histogrammar.primitives.minmax
import histogrammar.primitives.select
import histogrammar.primitives.sparselybin
import histogrammar.primitives.stack
import histogrammar.primitives.sum
from histogrammar.defs import Factory

def addMethods(df):
    def hg(self, h):
        converter = self._sc._jvm.org.dianahep.histogrammar.sparksql.pyspark.AggregatorConverter()
        agg = h._sparksql(self._sc._jvm, converter)
        result = converter.histogrammar(self._jdf, agg)
        return Factory.fromJson(json.loads(result.toJsonString()))

    def Average(self, quantity):
        return self.histogrammar(histogrammar.primitives.average.Average(quantity))

    def Bag(self, quantity, range):
        return self.histogrammar(histogrammar.primitives.bag.Bag(quantity, range))

    def Bin(self, num, low, high, quantity, value=histogrammar.primitives.count.Count(), underflow=histogrammar.primitives.count.Count(), overflow=histogrammar.primitives.count.Count(), nanflow=histogrammar.primitives.count.Count()):
        return self.histogrammar(histogrammar.primitives.bin.Bin(num, low, high, quantity, value, underflow, overflow, nanflow))

    def Categorize(self, quantity, value=histogrammar.primitives.count.Count()):
        return self.histogrammar(histogrammar.primitives.categorize.Categorize(quantity, value))

    def CentrallyBin(self, bins, quantity, value=histogrammar.primitives.count.Count(), nanflow=histogrammar.primitives.count.Count()):
        return self.histogrammar(histogrammar.primitives.centrallybin.CentrallyBin(bins, quantity, value, nanflow))

    def Label(self, **pairs):
        return self.histogrammar(histogrammar.primitives.collection.Label(**pairs))

    def UntypedLabel(self, **pairs):
        return self.histogrammar(histogrammar.primitives.collection.UntypedLabel(**pairs))

    def Index(self, *values):
        return self.histogrammar(histogrammar.primitives.collection.Index(*values))

    def Branch(self, *values):
        return self.histogrammar(histogrammar.primitives.collection.Branch(*values))

    def Count(self):    # TODO: handle transform
        return self.histogrammar(histogrammar.primitives.count.Count())

    def Deviate(self, quantity):
        return self.histogrammar(histogrammar.primitives.deviate.Deviate(quantity))

    def Fraction(self, quantity, value=histogrammar.primitives.count.Count()):
        return self.histogrammar(histogrammar.primitives.fraction.Fraction(quantity, value))

    def IrregularlyBin(self, thresholds, quantity, value=histogrammar.primitives.count.Count(), nanflow=histogrammar.primitives.count.Count()):
        return self.histogrammar(histogrammar.primitives.irregularlybin.IrregularlyBin(thresholds, quantity, value=histogrammar.primitives.count.Count(), nanflow=histogrammar.primitives.count.Count()))

    def Minimize(self, quantity):
        return self.histogrammar(histogrammar.primitives.minmax.Minimize(quantity))

    def Maximize(self, quantity):
        return self.histogrammar(histogrammar.primitives.minmax.Maximize(quantity))

    def Select(self, quantity, cut=histogrammar.primitives.count.Count()):
        return self.histogrammar(histogrammar.primitives.select.Select(quantity, cut))

    def SparselyBin(self, binWidth, quantity, value=histogrammar.primitives.count.Count(), nanflow=histogrammar.primitives.count.Count(), origin=0.0):
        return self.histogrammar(histogrammar.primitives.sparselybin.SparselyBin(binWidth, quantity, value, nanflow, origin))

    def Stack(self, bins, quantity, value=histogrammar.primitives.count.Count(), nanflow=histogrammar.primitives.count.Count()):
        return self.histogrammar(histogrammar.primitives.stack.Stack(bins, quantity, value, nanflow))

    def Sum(self, quantity):
        return self.histogrammar(histogrammar.primitives.sum.Sum(quantity))

    df.histogrammar = types.MethodType(hg, df)

    df.Average         = types.MethodType(Average        , df)
    df.Bag             = types.MethodType(Bag            , df)
    df.Bin             = types.MethodType(Bin            , df)
    df.Categorize      = types.MethodType(Categorize     , df)
    df.CentrallyBin    = types.MethodType(CentrallyBin   , df)
    df.Label           = types.MethodType(Label          , df)
    df.UntypedLabel    = types.MethodType(UntypedLabel   , df)
    df.Index           = types.MethodType(Index          , df)
    df.Branch          = types.MethodType(Branch         , df)
    df.Count           = types.MethodType(Count          , df)
    df.Deviate         = types.MethodType(Deviate        , df)
    df.Fraction        = types.MethodType(Fraction       , df)
    df.IrregularlyBin  = types.MethodType(IrregularlyBin , df)
    df.Minimize        = types.MethodType(Minimize       , df)
    df.Maximize        = types.MethodType(Maximize       , df)
    df.Select          = types.MethodType(Select         , df)
    df.SparselyBin     = types.MethodType(SparselyBin    , df)
    df.Stack           = types.MethodType(Stack          , df)
    df.Sum             = types.MethodType(Sum            , df)
