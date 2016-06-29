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
import math
import random
import sys
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

def makeSamples(SIZE, HOLES):
    try:
        with Numpy() as numpy:
            empty = numpy.array([], dtype=float)

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

                withholes2 = numpy.array([rand.gauss(0, 1) for i in xrange(SIZE)])
                for i in xrange(HOLES):
                    withholes2[rand.randint(0, SIZE)] = float("nan")
                for i in xrange(HOLES):
                    withholes2[rand.randint(0, SIZE)] = float("inf")
                for i in xrange(HOLES):
                    withholes2[rand.randint(0, SIZE)] = float("-inf")

            return {"empty": empty, "positive": positive, "noholes": noholes, "withholes": withholes, "withholes2": withholes2}

    except ImportError:
        return {}

class TestEverything(unittest.TestCase):
    def runTest(self):
        pass
        
    SIZE = 10000
    HOLES = 100
    data = makeSamples(SIZE, HOLES)
    empty = data["empty"]
    positive = data["positive"]
    noholes = data["noholes"]
    withholes = data["withholes"]
    withholes2 = data["withholes2"]

    def twosigfigs(self, number):
        return round(number, 1 - int(math.floor(math.log10(number))))

    scorecard = []

    def compare(self, name, hnp, npdata, hpy, pydata, weight):
        import numpy

        startTime = time.time()
        hnp.numpy(npdata, weight)
        numpyTime = time.time() - startTime
        
        if isinstance(weight, numpy.ndarray):
            startTime = time.time()
            for d, w in zip(pydata, weight):
                hpy.fill(d, w)
            pyTime = time.time() - startTime
        else:
            startTime = time.time()
            for d in pydata:
                hpy.fill(d, weight)
            pyTime = time.time() - startTime

        hnpj = json.dumps(hnp.toJson())
        hpyj = json.dumps(hpy.toJson())

        if Factory.fromJson(hnp.toJson()) != Factory.fromJson(hpy.toJson()):
            raise AssertionError("\n numpy: {0}\npython: {1}".format(hnpj, hpyj))
        else:
            sys.stderr.write("{0:45s} | numpy: {1:.3f}ms python: {2:.3f}ms = {3:g}X speedup\n".format(name, numpyTime*1000, pyTime*1000, self.twosigfigs(pyTime/numpyTime)))

        self.scorecard.append((pyTime/numpyTime, name))

    # # Warmup: apparently, Numpy does some dynamic optimization that needs to warm up...
    # Sum(lambda x: x["empty"]**3).fill(data)
    # Sum(lambda x: x["empty"]**3).fill(data)
    # Sum(lambda x: x["empty"]**3).fill(data)
    # Sum(lambda x: x["empty"]**3).fill(data)
    # Sum(lambda x: x["empty"]**3).fill(data)

    def testSum(self):
        with Numpy() as numpy:
            sys.stderr.write("\n")
            self.compare("Sum no data", Sum(lambda x: x["empty"]**3), self.data, Sum(lambda x: x**3), self.empty, 1.0)
            self.compare("Sum noholes w/o weights", Sum(lambda x: x["noholes"]**3), self.data, Sum(lambda x: x**3), self.noholes, 1.0)
            self.compare("Sum noholes const weights", Sum(lambda x: x["noholes"]**3), self.data, Sum(lambda x: x**3), self.noholes, 0.5)
            self.compare("Sum noholes positive weights", Sum(lambda x: x["noholes"]**3), self.data, Sum(lambda x: x**3), self.noholes, self.positive)
            self.compare("Sum noholes with weights", Sum(lambda x: x["noholes"]**3), self.data, Sum(lambda x: x**3), self.noholes, self.noholes)
            self.compare("Sum noholes with holes", Sum(lambda x: x["noholes"]**3), self.data, Sum(lambda x: x**3), self.noholes, self.withholes)
            self.compare("Sum holes w/o weights", Sum(lambda x: x["withholes"]**3), self.data, Sum(lambda x: x**3), self.withholes, 1.0)
            self.compare("Sum holes const weights", Sum(lambda x: x["withholes"]**3), self.data, Sum(lambda x: x**3), self.withholes, 0.5)
            self.compare("Sum holes positive weights", Sum(lambda x: x["withholes"]**3), self.data, Sum(lambda x: x**3), self.withholes, self.positive)
            self.compare("Sum holes with weights", Sum(lambda x: x["withholes"]**3), self.data, Sum(lambda x: x**3), self.withholes, self.noholes)
            self.compare("Sum holes with holes", Sum(lambda x: x["withholes"]**3), self.data, Sum(lambda x: x**3), self.withholes, self.withholes)
            self.compare("Sum holes with holes2", Sum(lambda x: x["withholes"]**3), self.data, Sum(lambda x: x**3), self.withholes, self.withholes2)

    def testAverage(self):
        with Numpy() as numpy:
            sys.stderr.write("\n")
            self.compare("Average no data", Average(lambda x: x["empty"]**3), self.data, Average(lambda x: x**3), self.empty, 1.0)
            self.compare("Average noholes w/o weights", Average(lambda x: x["noholes"]**3), self.data, Average(lambda x: x**3), self.noholes, 1.0)
            self.compare("Average noholes const weights", Average(lambda x: x["noholes"]**3), self.data, Average(lambda x: x**3), self.noholes, 0.5)
            self.compare("Average noholes positive weights", Average(lambda x: x["noholes"]**3), self.data, Average(lambda x: x**3), self.noholes, self.positive)
            self.compare("Average noholes with weights", Average(lambda x: x["noholes"]**3), self.data, Average(lambda x: x**3), self.noholes, self.noholes)
            self.compare("Average noholes with holes", Average(lambda x: x["noholes"]**3), self.data, Average(lambda x: x**3), self.noholes, self.withholes)
            self.compare("Average holes w/o weights", Average(lambda x: x["withholes"]**3), self.data, Average(lambda x: x**3), self.withholes, 1.0)
            self.compare("Average holes const weights", Average(lambda x: x["withholes"]**3), self.data, Average(lambda x: x**3), self.withholes, 0.5)
            self.compare("Average holes positive weights", Average(lambda x: x["withholes"]**3), self.data, Average(lambda x: x**3), self.withholes, self.positive)
            self.compare("Average holes with weights", Average(lambda x: x["withholes"]**3), self.data, Average(lambda x: x**3), self.withholes, self.noholes)
            self.compare("Average holes with holes", Average(lambda x: x["withholes"]**3), self.data, Average(lambda x: x**3), self.withholes, self.withholes)
            self.compare("Average holes with holes2", Average(lambda x: x["withholes"]**3), self.data, Average(lambda x: x**3), self.withholes, self.withholes2)

    def testDeviate(self):
        with Numpy() as numpy:
            sys.stderr.write("\n")
            self.compare("Deviate no data", Deviate(lambda x: x["empty"]**3), self.data, Deviate(lambda x: x**3), self.empty, 1.0)
            self.compare("Deviate noholes w/o weights", Deviate(lambda x: x["noholes"]**3), self.data, Deviate(lambda x: x**3), self.noholes, 1.0)
            self.compare("Deviate noholes const weights", Deviate(lambda x: x["noholes"]**3), self.data, Deviate(lambda x: x**3), self.noholes, 0.5)
            self.compare("Deviate noholes positive weights", Deviate(lambda x: x["noholes"]**3), self.data, Deviate(lambda x: x**3), self.noholes, self.positive)
            self.compare("Deviate noholes with weights", Deviate(lambda x: x["noholes"]**3), self.data, Deviate(lambda x: x**3), self.noholes, self.noholes)
            self.compare("Deviate noholes with holes", Deviate(lambda x: x["noholes"]**3), self.data, Deviate(lambda x: x**3), self.noholes, self.withholes)
            self.compare("Deviate holes w/o weights", Deviate(lambda x: x["withholes"]**3), self.data, Deviate(lambda x: x**3), self.withholes, 1.0)
            self.compare("Deviate holes const weights", Deviate(lambda x: x["withholes"]**3), self.data, Deviate(lambda x: x**3), self.withholes, 0.5)
            self.compare("Deviate holes positive weights", Deviate(lambda x: x["withholes"]**3), self.data, Deviate(lambda x: x**3), self.withholes, self.positive)
            self.compare("Deviate holes with weights", Deviate(lambda x: x["withholes"]**3), self.data, Deviate(lambda x: x**3), self.withholes, self.noholes)
            self.compare("Deviate holes with holes", Deviate(lambda x: x["withholes"]**3), self.data, Deviate(lambda x: x**3), self.withholes, self.withholes)
            self.compare("Deviate holes with holes2", Deviate(lambda x: x["withholes"]**3), self.data, Deviate(lambda x: x**3), self.withholes, self.withholes2)

    def testAbsoluteErr(self):
        with Numpy() as numpy:
            sys.stderr.write("\n")
            self.compare("AbsoluteErr no data", AbsoluteErr(lambda x: x["empty"]**3), self.data, AbsoluteErr(lambda x: x**3), self.empty, 1.0)
            self.compare("AbsoluteErr noholes w/o weights", AbsoluteErr(lambda x: x["noholes"]**3), self.data, AbsoluteErr(lambda x: x**3), self.noholes, 1.0)
            self.compare("AbsoluteErr noholes const weights", AbsoluteErr(lambda x: x["noholes"]**3), self.data, AbsoluteErr(lambda x: x**3), self.noholes, 0.5)
            self.compare("AbsoluteErr noholes positive weights", AbsoluteErr(lambda x: x["noholes"]**3), self.data, AbsoluteErr(lambda x: x**3), self.noholes, self.positive)
            self.compare("AbsoluteErr noholes with weights", AbsoluteErr(lambda x: x["noholes"]**3), self.data, AbsoluteErr(lambda x: x**3), self.noholes, self.noholes)
            self.compare("AbsoluteErr noholes with holes", AbsoluteErr(lambda x: x["noholes"]**3), self.data, AbsoluteErr(lambda x: x**3), self.noholes, self.withholes)
            self.compare("AbsoluteErr holes w/o weights", AbsoluteErr(lambda x: x["withholes"]**3), self.data, AbsoluteErr(lambda x: x**3), self.withholes, 1.0)
            self.compare("AbsoluteErr holes const weights", AbsoluteErr(lambda x: x["withholes"]**3), self.data, AbsoluteErr(lambda x: x**3), self.withholes, 0.5)
            self.compare("AbsoluteErr holes positive weights", AbsoluteErr(lambda x: x["withholes"]**3), self.data, AbsoluteErr(lambda x: x**3), self.withholes, self.positive)
            self.compare("AbsoluteErr holes with weights", AbsoluteErr(lambda x: x["withholes"]**3), self.data, AbsoluteErr(lambda x: x**3), self.withholes, self.noholes)
            self.compare("AbsoluteErr holes with holes", AbsoluteErr(lambda x: x["withholes"]**3), self.data, AbsoluteErr(lambda x: x**3), self.withholes, self.withholes)
            self.compare("AbsoluteErr holes with holes2", AbsoluteErr(lambda x: x["withholes"]**3), self.data, AbsoluteErr(lambda x: x**3), self.withholes, self.withholes2)

    def testMinimize(self):
        with Numpy() as numpy:
            sys.stderr.write("\n")
            self.compare("Minimize no data", Minimize(lambda x: x["empty"]**3), self.data, Minimize(lambda x: x**3), self.empty, 1.0)
            self.compare("Minimize noholes w/o weights", Minimize(lambda x: x["noholes"]**3), self.data, Minimize(lambda x: x**3), self.noholes, 1.0)
            self.compare("Minimize noholes const weights", Minimize(lambda x: x["noholes"]**3), self.data, Minimize(lambda x: x**3), self.noholes, 0.5)
            self.compare("Minimize noholes positive weights", Minimize(lambda x: x["noholes"]**3), self.data, Minimize(lambda x: x**3), self.noholes, self.positive)
            self.compare("Minimize noholes with weights", Minimize(lambda x: x["noholes"]**3), self.data, Minimize(lambda x: x**3), self.noholes, self.noholes)
            self.compare("Minimize noholes with holes", Minimize(lambda x: x["noholes"]**3), self.data, Minimize(lambda x: x**3), self.noholes, self.withholes)
            self.compare("Minimize holes w/o weights", Minimize(lambda x: x["withholes"]**3), self.data, Minimize(lambda x: x**3), self.withholes, 1.0)
            self.compare("Minimize holes const weights", Minimize(lambda x: x["withholes"]**3), self.data, Minimize(lambda x: x**3), self.withholes, 0.5)
            self.compare("Minimize holes positive weights", Minimize(lambda x: x["withholes"]**3), self.data, Minimize(lambda x: x**3), self.withholes, self.positive)
            self.compare("Minimize holes with weights", Minimize(lambda x: x["withholes"]**3), self.data, Minimize(lambda x: x**3), self.withholes, self.noholes)
            self.compare("Minimize holes with holes", Minimize(lambda x: x["withholes"]**3), self.data, Minimize(lambda x: x**3), self.withholes, self.withholes)
            self.compare("Minimize holes with holes2", Minimize(lambda x: x["withholes"]**3), self.data, Minimize(lambda x: x**3), self.withholes, self.withholes2)

    def testMaximize(self):
        with Numpy() as numpy:
            sys.stderr.write("\n")
            self.compare("Maximize no data", Maximize(lambda x: x["empty"]**3), self.data, Maximize(lambda x: x**3), self.empty, 1.0)
            self.compare("Maximize noholes w/o weights", Maximize(lambda x: x["noholes"]**3), self.data, Maximize(lambda x: x**3), self.noholes, 1.0)
            self.compare("Maximize noholes const weights", Maximize(lambda x: x["noholes"]**3), self.data, Maximize(lambda x: x**3), self.noholes, 0.5)
            self.compare("Maximize noholes positive weights", Maximize(lambda x: x["noholes"]**3), self.data, Maximize(lambda x: x**3), self.noholes, self.positive)
            self.compare("Maximize noholes with weights", Maximize(lambda x: x["noholes"]**3), self.data, Maximize(lambda x: x**3), self.noholes, self.noholes)
            self.compare("Maximize noholes with holes", Maximize(lambda x: x["noholes"]**3), self.data, Maximize(lambda x: x**3), self.noholes, self.withholes)
            self.compare("Maximize holes w/o weights", Maximize(lambda x: x["withholes"]**3), self.data, Maximize(lambda x: x**3), self.withholes, 1.0)
            self.compare("Maximize holes const weights", Maximize(lambda x: x["withholes"]**3), self.data, Maximize(lambda x: x**3), self.withholes, 0.5)
            self.compare("Maximize holes positive weights", Maximize(lambda x: x["withholes"]**3), self.data, Maximize(lambda x: x**3), self.withholes, self.positive)
            self.compare("Maximize holes with weights", Maximize(lambda x: x["withholes"]**3), self.data, Maximize(lambda x: x**3), self.withholes, self.noholes)
            self.compare("Maximize holes with holes", Maximize(lambda x: x["withholes"]**3), self.data, Maximize(lambda x: x**3), self.withholes, self.withholes)
            self.compare("Maximize holes with holes2", Maximize(lambda x: x["withholes"]**3), self.data, Maximize(lambda x: x**3), self.withholes, self.withholes2)


    # def testBin(self):
    #     with Numpy() as numpy:
    #         sys.stderr.write("\n")
    #         for bins in [10, 100, 1000]:
    #             self.compare("Bin ({0} bins) no data".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["empty"]**3), self.data, 1.0, Bin(bins, -3.0, 3.0, lambda x: x**3), self.empty)
    #             self.compare("Bin ({0} bins) noholes w/o weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["noholes"]**3), self.data, 1.0, Bin(bins, -3.0, 3.0, lambda x: x**3), self.noholes)
    #             self.compare("Bin ({0} bins) noholes positive weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["noholes"]**3), self.data, self.positive, Bin(bins, -3.0, 3.0, lambda x: x**3), self.noholes)
    #             self.compare("Bin ({0} bins) noholes with weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["noholes"]**3), self.data, self.noholes, Bin(bins, -3.0, 3.0, lambda x: x**3), self.noholes)
    #             self.compare("Bin ({0} bins) noholes with holes".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["noholes"]**3), self.data, self.withholes, Bin(bins, -3.0, 3.0, lambda x: x**3), self.noholes)
    #             self.compare("Bin ({0} bins) holes w/o weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"]**3), self.data, 1.0, Bin(bins, -3.0, 3.0, lambda x: x**3), self.withholes)
    #             self.compare("Bin ({0} bins) holes positive weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"]**3), self.data, self.positive, Bin(bins, -3.0, 3.0, lambda x: x**3), self.withholes)
    #             self.compare("Bin ({0} bins) holes with weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"]**3), self.data, self.noholes, Bin(bins, -3.0, 3.0, lambda x: x**3), self.withholes)
    #             self.compare("Bin ({0} bins) holes with holes".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"]**3), self.data, self.withholes, Bin(bins, -3.0, 3.0, lambda x: x**3), self.withholes)








    # def testBin(self):
    #     with Numpy() as numpy:
    #         sys.stderr.write("\n")
    #         for bins in [10, 100, 1000]:
    #             self.compare("Bin no data", Bin(bins, -3.0, 3.0, lambda x: x["empty"]**3), self.data, 1.0, Bin(bins, -3.0, 3.0, lambda x: x**3), self.empty)
    #             self.compare("Bin noholes w/o weights", Bin(bins, -3.0, 3.0, lambda x: x["noholes"]**3), self.data, 1.0, Bin(bins, -3.0, 3.0, lambda x: x**3), self.noholes)
    #             self.compare("Bin noholes const weight", Bin(bins, -3.0, 3.0, lambda x: x["noholes"]**3), self.data, 1.5, Bin(bins, -3.0, 3.0, lambda x: x**3), self.noholes)
    #             self.compare("Bin noholes positive weights", Bin(bins, -3.0, 3.0, lambda x: x["noholes"]**3), self.data, self.positive, Bin(bins, -3.0, 3.0, lambda x: x**3), self.noholes)
    #             self.compare("Bin noholes with weights", Bin(bins, -3.0, 3.0, lambda x: x["noholes"]**3), self.data, self.noholes, Bin(bins, -3.0, 3.0, lambda x: x**3), self.noholes)
    #             self.compare("Bin noholes with holes", Bin(bins, -3.0, 3.0, lambda x: x["noholes"]**3), self.data, self.withholes, Bin(bins, -3.0, 3.0, lambda x: x**3), self.noholes)
    #             self.compare("Bin holes w/o weights", Bin(bins, -3.0, 3.0, lambda x: x["withholes"]**3), self.data, 1.0, Bin(bins, -3.0, 3.0, lambda x: x**3), self.withholes)
    #             self.compare("Bin holes const weight", Bin(bins, -3.0, 3.0, lambda x: x["withholes"]**3), self.data, 1.5, Bin(bins, -3.0, 3.0, lambda x: x**3), self.withholes)
    #             self.compare("Bin holes positive weights", Bin(bins, -3.0, 3.0, lambda x: x["withholes"]**3), self.data, self.positive, Bin(bins, -3.0, 3.0, lambda x: x**3), self.withholes)
    #             self.compare("Bin holes with weights", Bin(bins, -3.0, 3.0, lambda x: x["withholes"]**3), self.data, self.noholes, Bin(bins, -3.0, 3.0, lambda x: x**3), self.withholes)
    #             self.compare("Bin holes with holes", Bin(bins, -3.0, 3.0, lambda x: x["withholes"]**3), self.data, self.withholes, Bin(bins, -3.0, 3.0, lambda x: x**3), self.withholes)
    #             self.assertRaises(AssertionError, lambda: Bin(bins, -3.0, 3.0, lambda x: x[:self.SIZE//2]).fillnp(self.noholes, lengthAssertion=self.SIZE))
    #             self.assertRaises(AssertionError, lambda: Bin(bins, -3.0, 3.0, lambda x: x["noholes"]).fillnp(self.data, self.noholes[:self.SIZE//2], lengthAssertion=self.SIZE))

    # def testBin(self):
    #     with Numpy() as numpy:
    #         good = lambda x: x**3
    #         sys.stderr.write("\n")
    #         for bins in [10, 100, 1000]:
    #             self.compare("Bin ({0} bins) no data".format(bins), Bin(bins, -3.0, 3.0, good), self.empty)
    #             self.compare("Bin ({0} bins) noholes w/o weights".format(bins), Bin(bins, -3.0, 3.0, good), self.noholes)
    #             self.compare("Bin ({0} bins) noholes const weight".format(bins), Bin(bins, -3.0, 3.0, good), self.noholes, 1.5)
    #             self.compare("Bin ({0} bins) noholes positive weights".format(bins), Bin(bins, -3.0, 3.0, good), self.noholes, self.positive)
    #             self.compare("Bin ({0} bins) noholes with weights".format(bins), Bin(bins, -3.0, 3.0, good), self.noholes, self.noholes)
    #             self.compare("Bin ({0} bins) noholes with holes".format(bins), Bin(bins, -3.0, 3.0, good), self.noholes, self.withholes)
    #             self.compare("Bin ({0} bins) holes w/o weights".format(bins), Bin(bins, -3.0, 3.0, good), self.withholes)
    #             self.compare("Bin ({0} bins) holes const weight".format(bins), Bin(bins, -3.0, 3.0, good), self.withholes, 1.5)
    #             self.compare("Bin ({0} bins) holes positive weights".format(bins), Bin(bins, -3.0, 3.0, good), self.withholes, self.positive)
    #             self.compare("Bin ({0} bins) holes with weights".format(bins), Bin(bins, -3.0, 3.0, good), self.withholes, self.noholes)
    #             self.compare("Bin ({0} bins) holes with holes".format(bins), Bin(bins, -3.0, 3.0, good), self.withholes, self.withholes)
    #             self.assertRaises(AssertionError, lambda: Bin(bins, -3.0, 3.0, lambda x: x[:self.SIZE//2]).fillnp(self.noholes))
    #             self.assertRaises(AssertionError, lambda: Bin(bins, -3.0, 3.0, good).fillnp(self.noholes, self.noholes[:self.SIZE//2]))

    # def testSparselyBin(self):
    #     with Numpy() as numpy:
    #         good = lambda x: x**3
    #         sys.stderr.write("\n")
    #         self.compare("SparselyBin no data", SparselyBin(0.1, good), self.empty)
    #         self.compare("SparselyBin noholes w/o weights", SparselyBin(0.1, good), self.noholes)
    #         self.compare("SparselyBin noholes const weight", SparselyBin(0.1, good), self.noholes, 1.5)
    #         self.compare("SparselyBin noholes positive weights", SparselyBin(0.1, good), self.noholes, self.positive)
    #         self.compare("SparselyBin noholes with weights", SparselyBin(0.1, good), self.noholes, self.noholes)
    #         self.compare("SparselyBin noholes with holes", SparselyBin(0.1, good), self.noholes, self.withholes)
    #         self.compare("SparselyBin holes w/o weights", SparselyBin(0.1, good), self.withholes)
    #         self.compare("SparselyBin holes const weight", SparselyBin(0.1, good), self.withholes, 1.5)
    #         self.compare("SparselyBin holes positive weights", SparselyBin(0.1, good), self.withholes, self.positive)
    #         self.compare("SparselyBin holes with weights", SparselyBin(0.1, good), self.withholes, self.noholes)
    #         self.compare("SparselyBin holes with holes", SparselyBin(0.1, good), self.withholes, self.withholes)
    #         self.assertRaises(AssertionError, lambda: SparselyBin(0.1, lambda x: x[:self.SIZE//2]).fillnp(self.noholes))
    #         self.assertRaises(AssertionError, lambda: SparselyBin(0.1, good).fillnp(self.noholes, self.noholes[:self.SIZE//2]))

    # def testCentrallyBin(self):
    #     with Numpy() as numpy:
    #         good = lambda x: x**3
    #         centers = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
    #         sys.stderr.write("\n")
    #         self.compare("CentrallyBin no data", CentrallyBin(centers, good), self.empty)
    #         self.compare("CentrallyBin noholes w/o weights", CentrallyBin(centers, good), self.noholes)
    #         self.compare("CentrallyBin noholes const weight", CentrallyBin(centers, good), self.noholes, 1.5)
    #         self.compare("CentrallyBin noholes positive weights", CentrallyBin(centers, good), self.noholes, self.positive)
    #         self.compare("CentrallyBin noholes with weights", CentrallyBin(centers, good), self.noholes, self.noholes)
    #         self.compare("CentrallyBin noholes with holes", CentrallyBin(centers, good), self.noholes, self.withholes)
    #         self.compare("CentrallyBin holes w/o weights", CentrallyBin(centers, good), self.withholes)
    #         self.compare("CentrallyBin holes const weight", CentrallyBin(centers, good), self.withholes, 1.5)
    #         self.compare("CentrallyBin holes positive weights", CentrallyBin(centers, good), self.withholes, self.positive)
    #         self.compare("CentrallyBin holes with weights", CentrallyBin(centers, good), self.withholes, self.noholes)
    #         self.compare("CentrallyBin holes with holes", CentrallyBin(centers, good), self.withholes, self.withholes)
    #         self.assertRaises(AssertionError, lambda: CentrallyBin(centers, lambda x: x[:self.SIZE//2]).fillnp(self.noholes))
    #         self.assertRaises(AssertionError, lambda: CentrallyBin(centers, good).fillnp(self.noholes, self.noholes[:self.SIZE//2]))

    # def testFraction(self):
    #     with Numpy() as numpy:
    #         boolean = lambda x: x**2 > 1.5
    #         positive = lambda x: x**2
    #         good = lambda x: x**3
    #         sys.stderr.write("\n")

    #         self.compare("Fraction boolean no data", Fraction(boolean, Bin(100, -3.0, 3.0, good)), self.empty)
    #         self.compare("Fraction boolean noholes w/o weights", Fraction(boolean, Bin(100, -3.0, 3.0, good)), self.noholes)
    #         self.compare("Fraction boolean noholes const weight", Fraction(boolean, Bin(100, -3.0, 3.0, good)), self.noholes, 1.5)
    #         self.compare("Fraction boolean noholes positive weights", Fraction(boolean, Bin(100, -3.0, 3.0, good)), self.noholes, self.positive)
    #         self.compare("Fraction boolean noholes with weights", Fraction(boolean, Bin(100, -3.0, 3.0, good)), self.noholes, self.noholes)
    #         self.compare("Fraction boolean noholes with holes", Fraction(boolean, Bin(100, -3.0, 3.0, good)), self.noholes, self.withholes)
    #         self.compare("Fraction boolean holes w/o weights", Fraction(boolean, Bin(100, -3.0, 3.0, good)), self.withholes)
    #         self.compare("Fraction boolean holes const weight", Fraction(boolean, Bin(100, -3.0, 3.0, good)), self.withholes, 1.5)
    #         self.compare("Fraction boolean holes positive weights", Fraction(boolean, Bin(100, -3.0, 3.0, good)), self.withholes, self.positive)
    #         self.compare("Fraction boolean holes with weights", Fraction(boolean, Bin(100, -3.0, 3.0, good)), self.withholes, self.noholes)
    #         self.compare("Fraction boolean holes with holes", Fraction(boolean, Bin(100, -3.0, 3.0, good)), self.withholes, self.withholes)
    #         self.assertRaises(AssertionError, lambda: Fraction(lambda x: (x**2 > 1.5)[:self.SIZE//2], Bin(100, -3.0, 3.0, good)).fillnp(self.noholes))

    #         self.compare("Fraction positive no data", Fraction(positive, Bin(100, -3.0, 3.0, good)), self.empty)
    #         self.compare("Fraction positive noholes w/o weights", Fraction(positive, Bin(100, -3.0, 3.0, good)), self.noholes)
    #         self.compare("Fraction positive noholes const weight", Fraction(positive, Bin(100, -3.0, 3.0, good)), self.noholes, 1.5)
    #         self.compare("Fraction positive noholes positive weights", Fraction(positive, Bin(100, -3.0, 3.0, good)), self.noholes, self.positive)
    #         self.compare("Fraction positive noholes with weights", Fraction(positive, Bin(100, -3.0, 3.0, good)), self.noholes, self.noholes)
    #         self.compare("Fraction positive noholes with holes", Fraction(positive, Bin(100, -3.0, 3.0, good)), self.noholes, self.withholes)
    #         self.compare("Fraction positive holes w/o weights", Fraction(positive, Bin(100, -3.0, 3.0, good)), self.withholes)
    #         self.compare("Fraction positive holes const weight", Fraction(positive, Bin(100, -3.0, 3.0, good)), self.withholes, 1.5)
    #         self.compare("Fraction positive holes positive weights", Fraction(positive, Bin(100, -3.0, 3.0, good)), self.withholes, self.positive)
    #         self.compare("Fraction positive holes with weights", Fraction(positive, Bin(100, -3.0, 3.0, good)), self.withholes, self.noholes)
    #         self.compare("Fraction positive holes with holes", Fraction(positive, Bin(100, -3.0, 3.0, good)), self.withholes, self.withholes)
    #         self.assertRaises(AssertionError, lambda: Fraction(lambda x: (x**2)[:self.SIZE//2], Bin(100, -3.0, 3.0, good)).fillnp(self.noholes))

    #         self.compare("Fraction good no data", Fraction(good, Bin(100, -3.0, 3.0, good)), self.empty)
    #         self.compare("Fraction good noholes w/o weights", Fraction(good, Bin(100, -3.0, 3.0, good)), self.noholes)
    #         self.compare("Fraction good noholes const weight", Fraction(good, Bin(100, -3.0, 3.0, good)), self.noholes, 1.5)
    #         self.compare("Fraction good noholes positive weights", Fraction(good, Bin(100, -3.0, 3.0, good)), self.noholes, self.positive)
    #         self.compare("Fraction good noholes with weights", Fraction(good, Bin(100, -3.0, 3.0, good)), self.noholes, self.noholes)
    #         self.compare("Fraction good noholes with holes", Fraction(good, Bin(100, -3.0, 3.0, good)), self.noholes, self.withholes)
    #         self.compare("Fraction good holes w/o weights", Fraction(good, Bin(100, -3.0, 3.0, good)), self.withholes)
    #         self.compare("Fraction good holes const weight", Fraction(good, Bin(100, -3.0, 3.0, good)), self.withholes, 1.5)
    #         self.compare("Fraction good holes positive weights", Fraction(good, Bin(100, -3.0, 3.0, good)), self.withholes, self.positive)
    #         self.compare("Fraction good holes with weights", Fraction(good, Bin(100, -3.0, 3.0, good)), self.withholes, self.noholes)
    #         self.compare("Fraction good holes with holes", Fraction(good, Bin(100, -3.0, 3.0, good)), self.withholes, self.withholes)
    #         self.assertRaises(AssertionError, lambda: Fraction(lambda x: (x**3)[:self.SIZE//2], Bin(100, -3.0, 3.0, good)).fillnp(self.noholes))

    # def testStack(self):
    #     with Numpy() as numpy:
    #         good = lambda x: x**3
    #         cuts = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
    #         sys.stderr.write("\n")
    #         self.compare("Stack good no data", Stack(cuts, good, Bin(100, -3.0, 3.0, good)), self.empty)
    #         self.compare("Stack good noholes w/o weights", Stack(cuts, good, Bin(100, -3.0, 3.0, good)), self.noholes)
    #         self.compare("Stack good noholes const weight", Stack(cuts, good, Bin(100, -3.0, 3.0, good)), self.noholes, 1.5)
    #         self.compare("Stack good noholes positive weights", Stack(cuts, good, Bin(100, -3.0, 3.0, good)), self.noholes, self.positive)
    #         self.compare("Stack good noholes with weights", Stack(cuts, good, Bin(100, -3.0, 3.0, good)), self.noholes, self.noholes)
    #         self.compare("Stack good noholes with holes", Stack(cuts, good, Bin(100, -3.0, 3.0, good)), self.noholes, self.withholes)
    #         self.compare("Stack good holes w/o weights", Stack(cuts, good, Bin(100, -3.0, 3.0, good)), self.withholes)
    #         self.compare("Stack good holes const weight", Stack(cuts, good, Bin(100, -3.0, 3.0, good)), self.withholes, 1.5)
    #         self.compare("Stack good holes positive weights", Stack(cuts, good, Bin(100, -3.0, 3.0, good)), self.withholes, self.positive)
    #         self.compare("Stack good holes with weights", Stack(cuts, good, Bin(100, -3.0, 3.0, good)), self.withholes, self.noholes)
    #         self.compare("Stack good holes with holes", Stack(cuts, good, Bin(100, -3.0, 3.0, good)), self.withholes, self.withholes)
    #         self.assertRaises(AssertionError, lambda: Stack(cuts, lambda x: (x**3)[:self.SIZE//2], Bin(100, -3.0, 3.0, good)).fillnp(self.noholes))

    # def testPartition(self):
    #     with Numpy() as numpy:
    #         good = lambda x: x**3
    #         cuts = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
    #         sys.stderr.write("\n")
    #         self.compare("Partition good no data", Partition(cuts, good, Bin(100, -3.0, 3.0, good)), self.empty)
    #         self.compare("Partition good noholes w/o weights", Partition(cuts, good, Bin(100, -3.0, 3.0, good)), self.noholes)
    #         self.compare("Partition good noholes const weight", Partition(cuts, good, Bin(100, -3.0, 3.0, good)), self.noholes, 1.5)
    #         self.compare("Partition good noholes positive weights", Partition(cuts, good, Bin(100, -3.0, 3.0, good)), self.noholes, self.positive)
    #         self.compare("Partition good noholes with weights", Partition(cuts, good, Bin(100, -3.0, 3.0, good)), self.noholes, self.noholes)
    #         self.compare("Partition good noholes with holes", Partition(cuts, good, Bin(100, -3.0, 3.0, good)), self.noholes, self.withholes)
    #         self.compare("Partition good holes w/o weights", Partition(cuts, good, Bin(100, -3.0, 3.0, good)), self.withholes)
    #         self.compare("Partition good holes const weight", Partition(cuts, good, Bin(100, -3.0, 3.0, good)), self.withholes, 1.5)
    #         self.compare("Partition good holes positive weights", Partition(cuts, good, Bin(100, -3.0, 3.0, good)), self.withholes, self.positive)
    #         self.compare("Partition good holes with weights", Partition(cuts, good, Bin(100, -3.0, 3.0, good)), self.withholes, self.noholes)
    #         self.compare("Partition good holes with holes", Partition(cuts, good, Bin(100, -3.0, 3.0, good)), self.withholes, self.withholes)
    #         self.assertRaises(AssertionError, lambda: Stack(cuts, lambda x: (x**3)[:self.SIZE//2], Bin(100, -3.0, 3.0, good)).fillnp(self.noholes))

    # def testSelect(self):
    #     with Numpy() as numpy:
    #         boolean = lambda x: x**2 > 1.5
    #         positive = lambda x: x**2
    #         good = lambda x: x**3
    #         sys.stderr.write("\n")

    #         self.compare("Select boolean no data", Select(boolean, Bin(100, -3.0, 3.0, good)), self.empty)
    #         self.compare("Select boolean noholes w/o weights", Select(boolean, Bin(100, -3.0, 3.0, good)), self.noholes)
    #         self.compare("Select boolean noholes const weight", Select(boolean, Bin(100, -3.0, 3.0, good)), self.noholes, 1.5)
    #         self.compare("Select boolean noholes positive weights", Select(boolean, Bin(100, -3.0, 3.0, good)), self.noholes, self.positive)
    #         self.compare("Select boolean noholes with weights", Select(boolean, Bin(100, -3.0, 3.0, good)), self.noholes, self.noholes)
    #         self.compare("Select boolean noholes with holes", Select(boolean, Bin(100, -3.0, 3.0, good)), self.noholes, self.withholes)
    #         self.compare("Select boolean holes w/o weights", Select(boolean, Bin(100, -3.0, 3.0, good)), self.withholes)
    #         self.compare("Select boolean holes const weight", Select(boolean, Bin(100, -3.0, 3.0, good)), self.withholes, 1.5)
    #         self.compare("Select boolean holes positive weights", Select(boolean, Bin(100, -3.0, 3.0, good)), self.withholes, self.positive)
    #         self.compare("Select boolean holes with weights", Select(boolean, Bin(100, -3.0, 3.0, good)), self.withholes, self.noholes)
    #         self.compare("Select boolean holes with holes", Select(boolean, Bin(100, -3.0, 3.0, good)), self.withholes, self.withholes)
    #         self.assertRaises(AssertionError, lambda: Select(lambda x: (x**2 > 1.5)[:self.SIZE//2], Bin(100, -3.0, 3.0, good)).fillnp(self.noholes))

    #         self.compare("Select positive no data", Select(positive, Bin(100, -3.0, 3.0, good)), self.empty)
    #         self.compare("Select positive noholes w/o weights", Select(positive, Bin(100, -3.0, 3.0, good)), self.noholes)
    #         self.compare("Select positive noholes const weight", Select(positive, Bin(100, -3.0, 3.0, good)), self.noholes, 1.5)
    #         self.compare("Select positive noholes positive weights", Select(positive, Bin(100, -3.0, 3.0, good)), self.noholes, self.positive)
    #         self.compare("Select positive noholes with weights", Select(positive, Bin(100, -3.0, 3.0, good)), self.noholes, self.noholes)
    #         self.compare("Select positive noholes with holes", Select(positive, Bin(100, -3.0, 3.0, good)), self.noholes, self.withholes)
    #         self.compare("Select positive holes w/o weights", Select(positive, Bin(100, -3.0, 3.0, good)), self.withholes)
    #         self.compare("Select positive holes const weight", Select(positive, Bin(100, -3.0, 3.0, good)), self.withholes, 1.5)
    #         self.compare("Select positive holes positive weights", Select(positive, Bin(100, -3.0, 3.0, good)), self.withholes, self.positive)
    #         self.compare("Select positive holes with weights", Select(positive, Bin(100, -3.0, 3.0, good)), self.withholes, self.noholes)
    #         self.compare("Select positive holes with holes", Select(positive, Bin(100, -3.0, 3.0, good)), self.withholes, self.withholes)
    #         self.assertRaises(AssertionError, lambda: Select(lambda x: (x**2)[:self.SIZE//2], Bin(100, -3.0, 3.0, good)).fillnp(self.noholes))

    #         self.compare("Select good no data", Select(good, Bin(100, -3.0, 3.0, good)), self.empty)
    #         self.compare("Select good noholes w/o weights", Select(good, Bin(100, -3.0, 3.0, good)), self.noholes)
    #         self.compare("Select good noholes const weight", Select(good, Bin(100, -3.0, 3.0, good)), self.noholes, 1.5)
    #         self.compare("Select good noholes positive weights", Select(good, Bin(100, -3.0, 3.0, good)), self.noholes, self.positive)
    #         self.compare("Select good noholes with weights", Select(good, Bin(100, -3.0, 3.0, good)), self.noholes, self.noholes)
    #         self.compare("Select good noholes with holes", Select(good, Bin(100, -3.0, 3.0, good)), self.noholes, self.withholes)
    #         self.compare("Select good holes w/o weights", Select(good, Bin(100, -3.0, 3.0, good)), self.withholes)
    #         self.compare("Select good holes const weight", Select(good, Bin(100, -3.0, 3.0, good)), self.withholes, 1.5)
    #         self.compare("Select good holes positive weights", Select(good, Bin(100, -3.0, 3.0, good)), self.withholes, self.positive)
    #         self.compare("Select good holes with weights", Select(good, Bin(100, -3.0, 3.0, good)), self.withholes, self.noholes)
    #         self.compare("Select good holes with holes", Select(good, Bin(100, -3.0, 3.0, good)), self.withholes, self.withholes)
    #         self.assertRaises(AssertionError, lambda: Select(lambda x: (x**3)[:self.SIZE//2], Bin(100, -3.0, 3.0, good)).fillnp(self.noholes))

    # def testLimit(self):
    #     with Numpy() as numpy:
    #         good = lambda x: x**3
    #         sys.stderr.write("\n")

    #         self.compare("Limit SIZE - 1 no data", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, good)), self.empty)
    #         self.compare("Limit SIZE - 1 noholes w/o weights", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, good)), self.noholes)
    #         self.compare("Limit SIZE - 1 noholes const weight", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, good)), self.noholes, 1.5)
    #         self.compare("Limit SIZE - 1 noholes positive weights", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, good)), self.noholes, self.positive)
    #         self.compare("Limit SIZE - 1 noholes with weights", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, good)), self.noholes, self.noholes)
    #         self.compare("Limit SIZE - 1 noholes with holes", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, good)), self.noholes, self.withholes)
    #         self.compare("Limit SIZE - 1 holes w/o weights", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, good)), self.withholes)
    #         self.compare("Limit SIZE - 1 holes const weight", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, good)), self.withholes, 1.5)
    #         self.compare("Limit SIZE - 1 holes positive weights", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, good)), self.withholes, self.positive)
    #         self.compare("Limit SIZE - 1 holes with weights", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, good)), self.withholes, self.noholes)
    #         self.compare("Limit SIZE - 1 holes with holes", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, good)), self.withholes, self.withholes)

    #         self.compare("Limit SIZE no data", Limit(self.SIZE, Bin(100, -3.0, 3.0, good)), self.empty)
    #         self.compare("Limit SIZE noholes w/o weights", Limit(self.SIZE, Bin(100, -3.0, 3.0, good)), self.noholes)
    #         self.compare("Limit SIZE noholes const weight", Limit(self.SIZE, Bin(100, -3.0, 3.0, good)), self.noholes, 1.5)
    #         self.compare("Limit SIZE noholes positive weights", Limit(self.SIZE, Bin(100, -3.0, 3.0, good)), self.noholes, self.positive)
    #         self.compare("Limit SIZE noholes with weights", Limit(self.SIZE, Bin(100, -3.0, 3.0, good)), self.noholes, self.noholes)
    #         self.compare("Limit SIZE noholes with holes", Limit(self.SIZE, Bin(100, -3.0, 3.0, good)), self.noholes, self.withholes)
    #         self.compare("Limit SIZE holes w/o weights", Limit(self.SIZE, Bin(100, -3.0, 3.0, good)), self.withholes)
    #         self.compare("Limit SIZE holes const weight", Limit(self.SIZE, Bin(100, -3.0, 3.0, good)), self.withholes, 1.5)
    #         self.compare("Limit SIZE holes positive weights", Limit(self.SIZE, Bin(100, -3.0, 3.0, good)), self.withholes, self.positive)
    #         self.compare("Limit SIZE holes with weights", Limit(self.SIZE, Bin(100, -3.0, 3.0, good)), self.withholes, self.noholes)
    #         self.compare("Limit SIZE holes with holes", Limit(self.SIZE, Bin(100, -3.0, 3.0, good)), self.withholes, self.withholes)

    #         self.compare("Limit SIZE*1.5 no data", Limit(self.SIZE*1.5, Bin(100, -3.0, 3.0, good)), self.empty)
    #         self.compare("Limit SIZE*1.5 noholes w/o weights", Limit(self.SIZE*1.5, Bin(100, -3.0, 3.0, good)), self.noholes)
    #         self.compare("Limit SIZE*1.5 noholes const weight", Limit(self.SIZE*1.5, Bin(100, -3.0, 3.0, good)), self.noholes, 1.5)
    #         self.compare("Limit SIZE*1.5 noholes positive weights", Limit(self.SIZE*1.5, Bin(100, -3.0, 3.0, good)), self.noholes, self.positive)
    #         self.compare("Limit SIZE*1.5 noholes with weights", Limit(self.SIZE*1.5, Bin(100, -3.0, 3.0, good)), self.noholes, self.noholes)
    #         self.compare("Limit SIZE*1.5 noholes with holes", Limit(self.SIZE*1.5, Bin(100, -3.0, 3.0, good)), self.noholes, self.withholes)
    #         self.compare("Limit SIZE*1.5 holes w/o weights", Limit(self.SIZE*1.5, Bin(100, -3.0, 3.0, good)), self.withholes)
    #         self.compare("Limit SIZE*1.5 holes const weight", Limit(self.SIZE*1.5, Bin(100, -3.0, 3.0, good)), self.withholes, 1.5)
    #         self.compare("Limit SIZE*1.5 holes positive weights", Limit(self.SIZE*1.5, Bin(100, -3.0, 3.0, good)), self.withholes, self.positive)
    #         self.compare("Limit SIZE*1.5 holes with weights", Limit(self.SIZE*1.5, Bin(100, -3.0, 3.0, good)), self.withholes, self.noholes)
    #         self.compare("Limit SIZE*1.5 holes with holes", Limit(self.SIZE*1.5, Bin(100, -3.0, 3.0, good)), self.withholes, self.withholes)

    #         self.compare("Limit SIZE*1.5 - 1 no data", Limit(self.SIZE*1.5 - 1, Bin(100, -3.0, 3.0, good)), self.empty)
    #         self.compare("Limit SIZE*1.5 - 1 noholes w/o weights", Limit(self.SIZE*1.5 - 1, Bin(100, -3.0, 3.0, good)), self.noholes)
    #         self.compare("Limit SIZE*1.5 - 1 noholes const weight", Limit(self.SIZE*1.5 - 1, Bin(100, -3.0, 3.0, good)), self.noholes, 1.5)
    #         self.compare("Limit SIZE*1.5 - 1 noholes positive weights", Limit(self.SIZE*1.5 - 1, Bin(100, -3.0, 3.0, good)), self.noholes, self.positive)
    #         self.compare("Limit SIZE*1.5 - 1 noholes with weights", Limit(self.SIZE*1.5 - 1, Bin(100, -3.0, 3.0, good)), self.noholes, self.noholes)
    #         self.compare("Limit SIZE*1.5 - 1 noholes with holes", Limit(self.SIZE*1.5 - 1, Bin(100, -3.0, 3.0, good)), self.noholes, self.withholes)
    #         self.compare("Limit SIZE*1.5 - 1 holes w/o weights", Limit(self.SIZE*1.5 - 1, Bin(100, -3.0, 3.0, good)), self.withholes)
    #         self.compare("Limit SIZE*1.5 - 1 holes const weight", Limit(self.SIZE*1.5 - 1, Bin(100, -3.0, 3.0, good)), self.withholes, 1.5)
    #         self.compare("Limit SIZE*1.5 - 1 holes positive weights", Limit(self.SIZE*1.5 - 1, Bin(100, -3.0, 3.0, good)), self.withholes, self.positive)
    #         self.compare("Limit SIZE*1.5 - 1 holes with weights", Limit(self.SIZE*1.5 - 1, Bin(100, -3.0, 3.0, good)), self.withholes, self.noholes)
    #         self.compare("Limit SIZE*1.5 - 1 holes with holes", Limit(self.SIZE*1.5 - 1, Bin(100, -3.0, 3.0, good)), self.withholes, self.withholes)

    # def testLabel(self):
    #     with Numpy() as numpy:
    #         good = lambda x: x**3
    #         sys.stderr.write("\n")
    #         self.compare("Label no data", Label(x=Bin(100, -3.0, 3.0, good)), self.empty)
    #         self.compare("Label noholes w/o weights", Label(x=Bin(100, -3.0, 3.0, good)), self.noholes)
    #         self.compare("Label noholes const weight", Label(x=Bin(100, -3.0, 3.0, good)), self.noholes, 1.5)
    #         self.compare("Label noholes positive weights", Label(x=Bin(100, -3.0, 3.0, good)), self.noholes, self.positive)
    #         self.compare("Label noholes with weights", Label(x=Bin(100, -3.0, 3.0, good)), self.noholes, self.noholes)
    #         self.compare("Label noholes with holes", Label(x=Bin(100, -3.0, 3.0, good)), self.noholes, self.withholes)
    #         self.compare("Label holes w/o weights", Label(x=Bin(100, -3.0, 3.0, good)), self.withholes)
    #         self.compare("Label holes const weight", Label(x=Bin(100, -3.0, 3.0, good)), self.withholes, 1.5)
    #         self.compare("Label holes positive weights", Label(x=Bin(100, -3.0, 3.0, good)), self.withholes, self.positive)
    #         self.compare("Label holes with weights", Label(x=Bin(100, -3.0, 3.0, good)), self.withholes, self.noholes)
    #         self.compare("Label holes with holes", Label(x=Bin(100, -3.0, 3.0, good)), self.withholes, self.withholes)
    #         self.assertRaises(AssertionError, lambda: Label(x=Bin(100, -3.0, 3.0, lambda x: x[:self.SIZE//2])).fillnp(self.noholes))
    #         self.assertRaises(AssertionError, lambda: Label(x=Bin(100, -3.0, 3.0, good)).fillnp(self.noholes, self.noholes[:self.SIZE//2]))

    # def testUntypedLabel(self):
    #     with Numpy() as numpy:
    #         good = lambda x: x**3
    #         sys.stderr.write("\n")
    #         self.compare("UntypedLabel no data", UntypedLabel(x=Bin(100, -3.0, 3.0, good)), self.empty)
    #         self.compare("UntypedLabel noholes w/o weights", UntypedLabel(x=Bin(100, -3.0, 3.0, good)), self.noholes)
    #         self.compare("UntypedLabel noholes const weight", UntypedLabel(x=Bin(100, -3.0, 3.0, good)), self.noholes, 1.5)
    #         self.compare("UntypedLabel noholes positive weights", UntypedLabel(x=Bin(100, -3.0, 3.0, good)), self.noholes, self.positive)
    #         self.compare("UntypedLabel noholes with weights", UntypedLabel(x=Bin(100, -3.0, 3.0, good)), self.noholes, self.noholes)
    #         self.compare("UntypedLabel noholes with holes", UntypedLabel(x=Bin(100, -3.0, 3.0, good)), self.noholes, self.withholes)
    #         self.compare("UntypedLabel holes w/o weights", UntypedLabel(x=Bin(100, -3.0, 3.0, good)), self.withholes)
    #         self.compare("UntypedLabel holes const weight", UntypedLabel(x=Bin(100, -3.0, 3.0, good)), self.withholes, 1.5)
    #         self.compare("UntypedLabel holes positive weights", UntypedLabel(x=Bin(100, -3.0, 3.0, good)), self.withholes, self.positive)
    #         self.compare("UntypedLabel holes with weights", UntypedLabel(x=Bin(100, -3.0, 3.0, good)), self.withholes, self.noholes)
    #         self.compare("UntypedLabel holes with holes", UntypedLabel(x=Bin(100, -3.0, 3.0, good)), self.withholes, self.withholes)
    #         self.assertRaises(AssertionError, lambda: UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x[:self.SIZE//2])).fillnp(self.noholes))
    #         self.assertRaises(AssertionError, lambda: UntypedLabel(x=Bin(100, -3.0, 3.0, good)).fillnp(self.noholes, self.noholes[:self.SIZE//2]))

    # def testIndex(self):
    #     with Numpy() as numpy:
    #         good = lambda x: x**3
    #         sys.stderr.write("\n")
    #         self.compare("Index no data", Index(Bin(100, -3.0, 3.0, good)), self.empty)
    #         self.compare("Index noholes w/o weights", Index(Bin(100, -3.0, 3.0, good)), self.noholes)
    #         self.compare("Index noholes const weight", Index(Bin(100, -3.0, 3.0, good)), self.noholes, 1.5)
    #         self.compare("Index noholes positive weights", Index(Bin(100, -3.0, 3.0, good)), self.noholes, self.positive)
    #         self.compare("Index noholes with weights", Index(Bin(100, -3.0, 3.0, good)), self.noholes, self.noholes)
    #         self.compare("Index noholes with holes", Index(Bin(100, -3.0, 3.0, good)), self.noholes, self.withholes)
    #         self.compare("Index holes w/o weights", Index(Bin(100, -3.0, 3.0, good)), self.withholes)
    #         self.compare("Index holes const weight", Index(Bin(100, -3.0, 3.0, good)), self.withholes, 1.5)
    #         self.compare("Index holes positive weights", Index(Bin(100, -3.0, 3.0, good)), self.withholes, self.positive)
    #         self.compare("Index holes with weights", Index(Bin(100, -3.0, 3.0, good)), self.withholes, self.noholes)
    #         self.compare("Index holes with holes", Index(Bin(100, -3.0, 3.0, good)), self.withholes, self.withholes)
    #         self.assertRaises(AssertionError, lambda: Index(Bin(100, -3.0, 3.0, lambda x: x[:self.SIZE//2])).fillnp(self.noholes))
    #         self.assertRaises(AssertionError, lambda: Index(Bin(100, -3.0, 3.0, good)).fillnp(self.noholes, self.noholes[:self.SIZE//2]))

    # def testBranch(self):
    #     with Numpy() as numpy:
    #         good = lambda x: x**3
    #         sys.stderr.write("\n")
    #         self.compare("Branch no data", Branch(Bin(100, -3.0, 3.0, good)), self.empty)
    #         self.compare("Branch noholes w/o weights", Branch(Bin(100, -3.0, 3.0, good)), self.noholes)
    #         self.compare("Branch noholes const weight", Branch(Bin(100, -3.0, 3.0, good)), self.noholes, 1.5)
    #         self.compare("Branch noholes positive weights", Branch(Bin(100, -3.0, 3.0, good)), self.noholes, self.positive)
    #         self.compare("Branch noholes with weights", Branch(Bin(100, -3.0, 3.0, good)), self.noholes, self.noholes)
    #         self.compare("Branch noholes with holes", Branch(Bin(100, -3.0, 3.0, good)), self.noholes, self.withholes)
    #         self.compare("Branch holes w/o weights", Branch(Bin(100, -3.0, 3.0, good)), self.withholes)
    #         self.compare("Branch holes const weight", Branch(Bin(100, -3.0, 3.0, good)), self.withholes, 1.5)
    #         self.compare("Branch holes positive weights", Branch(Bin(100, -3.0, 3.0, good)), self.withholes, self.positive)
    #         self.compare("Branch holes with weights", Branch(Bin(100, -3.0, 3.0, good)), self.withholes, self.noholes)
    #         self.compare("Branch holes with holes", Branch(Bin(100, -3.0, 3.0, good)), self.withholes, self.withholes)
    #         self.assertRaises(AssertionError, lambda: Branch(Bin(100, -3.0, 3.0, lambda x: x[:self.SIZE//2])).fillnp(self.noholes))
    #         self.assertRaises(AssertionError, lambda: Branch(Bin(100, -3.0, 3.0, good)).fillnp(self.noholes, self.noholes[:self.SIZE//2]))

    # def testZZZ(self):
    #     self.scorecard.sort()
    #     sys.stderr.write("\n----------------------------------------------+----------------------------\n")
    #     sys.stderr.write("Numpy/PurePython comparison                   | Speedup factor\n")
    #     sys.stderr.write("----------------------------------------------+----------------------------\n")
    #     for score, name in self.scorecard:
    #         sys.stderr.write("{0:45s} | {1:g}\n".format(name, score))
