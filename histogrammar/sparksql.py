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

def addMethods(df):
    def histogrammar(self, h):
        converter = self.df._sc._jvm.org.dianahep.histogrammar.sparksql.pyspark.AggregatorConverter()
        agg = h._sparksql(self.df._sc._jvm, converter)
        result = converter.histogrammar(self.df._jdf, agg)
        return Factory.fromJson(jsonlib.loads(result.toJsonString()))

    def Average(self, quantity):
        return self.histogrammar(histogrammar.primitives.average.Average(quantity))

    def Bag(self, quantity, range):
        return self.histogrammar(histogrammar.primitives.bag.Bag(quantity, range))

    # def Bin(self, num, low, high, quantity, value=histogrammar.primitives.count.Count(), underflow=histogrammar.primitives.count.Count(), overflow=histogrammar.primitives.count.Count(), nanflow=histogrammar.primitives.count.Count()):
    #     return self.histogrammar()




    # df.histogrammar    = types.MethodType(histogrammar, df)

    # hg.Average         = types.MethodType(Average        , df)
    # hg.Bag             = types.MethodType(Bag            , df)
    # hg.Bin             = types.MethodType(Bin            , df)
    # hg.Categorize      = types.MethodType(Categorize     , df)
    # hg.CentrallyBin    = types.MethodType(CentrallyBin   , df)
    # hg.Label           = types.MethodType(Label          , df)
    # hg.UntypedLabel    = types.MethodType(UntypedLabel   , df)
    # hg.Index           = types.MethodType(Index          , df)
    # hg.Branch          = types.MethodType(Branch         , df)
    # hg.Count           = types.MethodType(Count          , df)
    # hg.Deviate         = types.MethodType(Deviate        , df)
    # hg.Fraction        = types.MethodType(Fraction       , df)
    # hg.IrregularlyBin  = types.MethodType(IrregularlyBin , df)
    # hg.Minimize        = types.MethodType(Minimize       , df)
    # hg.Maximize        = types.MethodType(Maximize       , df)
    # hg.Select          = types.MethodType(Select         , df)
    # hg.SparselyBin     = types.MethodType(SparselyBin    , df)
    # hg.Stack           = types.MethodType(Stack          , df)
    # hg.Sum             = types.MethodType(Sum            , df)
