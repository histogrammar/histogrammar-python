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

import math
import json
import random
import time
import unittest

from histogrammar import *

tolerance = 1e-12
util.relativeTolerance = tolerance
util.absoluteTolerance = tolerance

class Numpy(object):
    def __enter__(self):
        import numpy
        self.errstate = numpy.geterr()
        numpy.seterr(invalid="ignore")
        return numpy

    def __exit__(self, exc_type, exc_value, traceback):
        import numpy
        numpy.seterr(**self.errstate)

class TestEverything(unittest.TestCase):
    def runTest(self):
        pass
        
    with Numpy() as numpy:
        empty = numpy.array([], dtype=float)

        SIZE = 10000
        HOLES = 100
        if numpy is not None:
            rand = random.Random(12345)

            positive = numpy.array([abs(rand.gauss(0, 1)) + 1e-12 for i in xrange(SIZE)])
            assert all(x > 0.0 for x in positive)

            noholes = numpy.array([rand.gauss(0, 1) for i in xrange(SIZE)])

            withholes = numpy.array([rand.gauss(0, 1) for i in xrange(SIZE)])
            for i in xrange(HOLES):
                withholes[rand.randint(0, SIZE)] = float("nan")
            for i in xrange(HOLES):
                withholes[rand.randint(0, SIZE)] = float("inf")
            for i in xrange(HOLES):
                withholes[rand.randint(0, SIZE)] = float("-inf")

    def twosigfigs(self, number):
        return round(number, 1 - int(math.floor(math.log10(number))))

    def compare(self, name, h, data, weight=1.0):
        import numpy

        hnp = h.copy()
        hpy = h.copy()

        startTime = time.time()
        hnp.fillnp(data, weight)
        numpyTime = time.time() - startTime

        if isinstance(weight, numpy.ndarray):
            startTime = time.time()
            for d, w in zip(data, weight):
                hpy.fill(d, w)
            pyTime = time.time() - startTime
        else:
            startTime = time.time()
            for d in data:
                hpy.fill(d, weight)
            pyTime = time.time() - startTime

        if hnp != hpy:
            raise AssertionError("\n numpy: {0}\npython: {1}".format(json.dumps(hnp.toJson()), json.dumps(hpy.toJson())))
        else:
            print("{0:40s} | numpy: {1:.3f}ms python: {2:.3f}ms = {3:g}X speedup".format(name, numpyTime*1000, pyTime*1000, self.twosigfigs(pyTime/numpyTime)))
        
    # Warmup: apparently, Numpy does some dynamic optimization that needs to warm up...
    Count().fillnp(withholes, positive)
    Count().fillnp(withholes, positive)
    Count().fillnp(withholes, positive)
    Count().fillnp(withholes, positive)
    Count().fillnp(withholes, positive)

    ################################################################ Count
    
    def testCount(self):
        with Numpy() as numpy:
            print("")
            self.compare("Count no data", Count(), self.empty)
            self.compare("Count noholes w/o weights", Count(), self.noholes)
            self.compare("Count noholes const weight", Count(), self.noholes, 1.5)
            self.compare("Count noholes positive weights", Count(), self.noholes, self.positive)
            self.compare("Count noholes with weights", Count(), self.noholes, self.noholes)
            self.compare("Count noholes with holes", Count(), self.noholes, self.withholes)
            self.compare("Count holes w/o weights", Count(), self.withholes)
            self.compare("Count holes const weight", Count(), self.withholes, 1.5)
            self.compare("Count holes positive weights", Count(), self.withholes, self.positive)
            self.compare("Count holes with weights", Count(), self.withholes, self.noholes)
            self.compare("Count holes with holes", Count(), self.withholes, self.withholes)

    def testSum(self):
        with Numpy() as numpy:
            good = lambda x: x**3
            print("")
            self.compare("Sum no data", Sum(good), self.empty)
            self.compare("Sum noholes w/o weights", Sum(good), self.noholes)
            self.compare("Sum noholes const weight", Sum(good), self.noholes, 1.5)
            self.compare("Sum noholes positive weights", Sum(good), self.noholes, self.positive)
            self.compare("Sum noholes with weights", Sum(good), self.noholes, self.noholes)
            self.compare("Sum noholes with holes", Sum(good), self.noholes, self.withholes)
            self.compare("Sum holes w/o weights", Sum(good), self.withholes)
            self.compare("Sum holes const weight", Sum(good), self.withholes, 1.5)
            self.compare("Sum holes positive weights", Sum(good), self.withholes, self.positive)
            self.compare("Sum holes with weights", Sum(good), self.withholes, self.noholes)
            self.compare("Sum holes with holes", Sum(good), self.withholes, self.withholes)
            self.assertRaises(AssertionError, lambda: Sum(lambda x: x[:self.SIZE/2]).fillnp(self.noholes))
            self.assertRaises(AssertionError, lambda: Sum(good).fillnp(self.noholes, self.noholes[:self.SIZE/2]))

    def testAverage(self):
        with Numpy() as numpy:
            good = lambda x: x**3
            print("")
            self.compare("Average no data", Average(good), self.empty)
            self.compare("Average noholes w/o weights", Average(good), self.noholes)
            self.compare("Average noholes const weight", Average(good), self.noholes, 1.5)
            self.compare("Average noholes positive weights", Average(good), self.noholes, self.positive)
            self.compare("Average noholes with weights", Average(good), self.noholes, self.noholes)
            self.compare("Average noholes with holes", Average(good), self.noholes, self.withholes)
            self.compare("Average holes w/o weights", Average(good), self.withholes)
            self.compare("Average holes const weight", Average(good), self.withholes, 1.5)
            self.compare("Average holes positive weights", Average(good), self.withholes, self.positive)
            self.compare("Average holes with weights", Average(good), self.withholes, self.noholes)
            self.compare("Average holes with holes", Average(good), self.withholes, self.withholes)
            self.assertRaises(AssertionError, lambda: Average(lambda x: x[:self.SIZE/2]).fillnp(self.noholes))
            self.assertRaises(AssertionError, lambda: Average(good).fillnp(self.noholes, self.noholes[:self.SIZE/2]))

    def testDeviate(self):
        with Numpy() as numpy:
            good = lambda x: x**3
            print("")
            self.compare("Deviate no data", Deviate(good), self.empty)
            self.compare("Deviate noholes w/o weights", Deviate(good), self.noholes)
            self.compare("Deviate noholes const weight", Deviate(good), self.noholes, 1.5)
            self.compare("Deviate noholes positive weights", Deviate(good), self.noholes, self.positive)
            self.compare("Deviate noholes with weights", Deviate(good), self.noholes, self.noholes)
            self.compare("Deviate noholes with holes", Deviate(good), self.noholes, self.withholes)
            self.compare("Deviate holes w/o weights", Deviate(good), self.withholes)
            self.compare("Deviate holes const weight", Deviate(good), self.withholes, 1.5)
            self.compare("Deviate holes positive weights", Deviate(good), self.withholes, self.positive)
            self.compare("Deviate holes with weights", Deviate(good), self.withholes, self.noholes)
            self.compare("Deviate holes with holes", Deviate(good), self.withholes, self.withholes)
            self.assertRaises(AssertionError, lambda: Deviate(lambda x: x[:self.SIZE/2]).fillnp(self.noholes))
            self.assertRaises(AssertionError, lambda: Deviate(good).fillnp(self.noholes, self.noholes[:self.SIZE/2]))

    def testAbsoluteErr(self):
        with Numpy() as numpy:
            good = lambda x: x**3
            print("")
            self.compare("AbsoluteErr no data", AbsoluteErr(good), self.empty)
            self.compare("AbsoluteErr noholes w/o weights", AbsoluteErr(good), self.noholes)
            self.compare("AbsoluteErr noholes const weight", AbsoluteErr(good), self.noholes, 1.5)
            self.compare("AbsoluteErr noholes positive weights", AbsoluteErr(good), self.noholes, self.positive)
            self.compare("AbsoluteErr noholes with weights", AbsoluteErr(good), self.noholes, self.noholes)
            self.compare("AbsoluteErr noholes with holes", AbsoluteErr(good), self.noholes, self.withholes)
            self.compare("AbsoluteErr holes w/o weights", AbsoluteErr(good), self.withholes)
            self.compare("AbsoluteErr holes const weight", AbsoluteErr(good), self.withholes, 1.5)
            self.compare("AbsoluteErr holes positive weights", AbsoluteErr(good), self.withholes, self.positive)
            self.compare("AbsoluteErr holes with weights", AbsoluteErr(good), self.withholes, self.noholes)
            self.compare("AbsoluteErr holes with holes", AbsoluteErr(good), self.withholes, self.withholes)
            self.assertRaises(AssertionError, lambda: AbsoluteErr(lambda x: x[:self.SIZE/2]).fillnp(self.noholes))
            self.assertRaises(AssertionError, lambda: AbsoluteErr(good).fillnp(self.noholes, self.noholes[:self.SIZE/2]))

    def testMinimize(self):
        with Numpy() as numpy:
            good = lambda x: x**3
            print("")
            self.compare("Minimize no data", Minimize(good), self.empty)
            self.compare("Minimize noholes w/o weights", Minimize(good), self.noholes)
            self.compare("Minimize noholes const weight", Minimize(good), self.noholes, 1.5)
            self.compare("Minimize noholes positive weights", Minimize(good), self.noholes, self.positive)
            self.compare("Minimize noholes with weights", Minimize(good), self.noholes, self.noholes)
            self.compare("Minimize noholes with holes", Minimize(good), self.noholes, self.withholes)
            self.compare("Minimize holes w/o weights", Minimize(good), self.withholes)
            self.compare("Minimize holes const weight", Minimize(good), self.withholes, 1.5)
            self.compare("Minimize holes positive weights", Minimize(good), self.withholes, self.positive)
            self.compare("Minimize holes with weights", Minimize(good), self.withholes, self.noholes)
            self.compare("Minimize holes with holes", Minimize(good), self.withholes, self.withholes)
            self.assertRaises(AssertionError, lambda: Minimize(lambda x: x[:self.SIZE/2]).fillnp(self.noholes))
            self.assertRaises(AssertionError, lambda: Minimize(good).fillnp(self.noholes, self.noholes[:self.SIZE/2]))

    def testMaximize(self):
        with Numpy() as numpy:
            good = lambda x: x**3
            print("")
            self.compare("Maximize no data", Maximize(good), self.empty)
            self.compare("Maximize noholes w/o weights", Maximize(good), self.noholes)
            self.compare("Maximize noholes const weight", Maximize(good), self.noholes, 1.5)
            self.compare("Maximize noholes positive weights", Maximize(good), self.noholes, self.positive)
            self.compare("Maximize noholes with weights", Maximize(good), self.noholes, self.noholes)
            self.compare("Maximize noholes with holes", Maximize(good), self.noholes, self.withholes)
            self.compare("Maximize holes w/o weights", Maximize(good), self.withholes)
            self.compare("Maximize holes const weight", Maximize(good), self.withholes, 1.5)
            self.compare("Maximize holes positive weights", Maximize(good), self.withholes, self.positive)
            self.compare("Maximize holes with weights", Maximize(good), self.withholes, self.noholes)
            self.compare("Maximize holes with holes", Maximize(good), self.withholes, self.withholes)
            self.assertRaises(AssertionError, lambda: Maximize(lambda x: x[:self.SIZE/2]).fillnp(self.noholes))
            self.assertRaises(AssertionError, lambda: Maximize(good).fillnp(self.noholes, self.noholes[:self.SIZE/2]))

    def testBin(self):
        with Numpy() as numpy:
            good = lambda x: x**3
            print("")
            self.compare("Bin no data", Bin(100, -3.0, 3.0, good), self.empty)
            self.compare("Bin noholes w/o weights", Bin(100, -3.0, 3.0, good), self.noholes)
            self.compare("Bin noholes const weight", Bin(100, -3.0, 3.0, good), self.noholes, 1.5)
            self.compare("Bin noholes positive weights", Bin(100, -3.0, 3.0, good), self.noholes, self.positive)
            self.compare("Bin noholes with weights", Bin(100, -3.0, 3.0, good), self.noholes, self.noholes)
            self.compare("Bin noholes with holes", Bin(100, -3.0, 3.0, good), self.noholes, self.withholes)
            self.compare("Bin holes w/o weights", Bin(100, -3.0, 3.0, good), self.withholes)
            self.compare("Bin holes const weight", Bin(100, -3.0, 3.0, good), self.withholes, 1.5)
            self.compare("Bin holes positive weights", Bin(100, -3.0, 3.0, good), self.withholes, self.positive)
            self.compare("Bin holes with weights", Bin(100, -3.0, 3.0, good), self.withholes, self.noholes)
            self.compare("Bin holes with holes", Bin(100, -3.0, 3.0, good), self.withholes, self.withholes)
            self.assertRaises(AssertionError, lambda: Bin(100, -3.0, 3.0, lambda x: x[:self.SIZE/2]).fillnp(self.noholes))
            self.assertRaises(AssertionError, lambda: Bin(100, -3.0, 3.0, good).fillnp(self.noholes, self.noholes[:self.SIZE/2]))

    def testSparselyBin(self):
        with Numpy() as numpy:
            good = lambda x: x**3
            print("")
            self.compare("SparselyBin no data", SparselyBin(0.1, good), self.empty)
            self.compare("SparselyBin noholes w/o weights", SparselyBin(0.1, good), self.noholes)
            self.compare("SparselyBin noholes const weight", SparselyBin(0.1, good), self.noholes, 1.5)
            self.compare("SparselyBin noholes positive weights", SparselyBin(0.1, good), self.noholes, self.positive)
            self.compare("SparselyBin noholes with weights", SparselyBin(0.1, good), self.noholes, self.noholes)
            self.compare("SparselyBin noholes with holes", SparselyBin(0.1, good), self.noholes, self.withholes)
            self.compare("SparselyBin holes w/o weights", SparselyBin(0.1, good), self.withholes)
            self.compare("SparselyBin holes const weight", SparselyBin(0.1, good), self.withholes, 1.5)
            self.compare("SparselyBin holes positive weights", SparselyBin(0.1, good), self.withholes, self.positive)
            self.compare("SparselyBin holes with weights", SparselyBin(0.1, good), self.withholes, self.noholes)
            self.compare("SparselyBin holes with holes", SparselyBin(0.1, good), self.withholes, self.withholes)
            self.assertRaises(AssertionError, lambda: SparselyBin(0.1, lambda x: x[:self.SIZE/2]).fillnp(self.noholes))
            self.assertRaises(AssertionError, lambda: SparselyBin(0.1, good).fillnp(self.noholes, self.noholes[:self.SIZE/2]))





# | SparselySparselyBin            | untested
# | CentrallySparselyBin           | untested
# | Fraction               | untested
# | Stack                  | untested
# | Partition              | untested
# | Select                 | untested
# | Limit                  | untested
# | Label                  | untested
# | UntypedLabel           | untested
# | Index                  | untested
# | Branch                 | untested
