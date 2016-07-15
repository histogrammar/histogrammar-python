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

import test.specification

tolerance = 1e-12
util.relativeTolerance = tolerance
util.absoluteTolerance = tolerance

class Numpy(object):
    def __enter__(self):
        try:
            import numpy
        except ImportError:
            return None
        self.errstate = numpy.geterr()
        numpy.seterr(invalid="ignore")
        return numpy

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            import numpy
            numpy.seterr(**self.errstate)
        except ImportError:
            pass

def makeSamples(SIZE, HOLES):
    with Numpy() as numpy:
        if numpy is None:
            return {"empty": None, "positive": None, "boolean": None, "noholes": None, "withholes": None, "withholes2": None}

        empty = numpy.array([], dtype=float)

        if numpy is not None:
            rand = random.Random(12345)

            positive = numpy.array([abs(rand.gauss(0, 1)) + 1e-12 for i in xrange(SIZE)])
            assert all(x > 0.0 for x in positive)

            boolean = positive > 1.5

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

        return {"empty": empty, "positive": positive, "boolean": boolean, "noholes": noholes, "withholes": withholes, "withholes2": withholes2}

class TestNumpy(unittest.TestCase):
    def runTest(self):
        pass
        
    SIZE = 10000
    HOLES = 100
    data = makeSamples(SIZE, HOLES)
    empty = data["empty"]
    positive = data["positive"]
    boolean = data["boolean"]
    noholes = data["noholes"]
    withholes = data["withholes"]
    withholes2 = data["withholes2"]

    def twosigfigs(self, number):
        return round(number, 1 - int(math.floor(math.log10(number))))

    def compare(self, name, hnp, npdata, hpy, pydata, weight):
        import numpy

        npdata2 = npdata.copy()
        if isinstance(weight, numpy.ndarray):
            weight2 = weight.copy()

        hnp2 = hnp.copy()
        hnp3 = hnp.copy()
        hpy2 = hpy.copy()
        hpy3 = hpy.copy()
        hpy4 = hpy.copy()
        hpy5 = hpy.copy()

        startTime = time.time()
        hnp.numpy(npdata, weight)
        numpyTime = time.time() - startTime

        if pydata.dtype != numpy.unicode_:
            for key in npdata:
                diff = (npdata[key] != npdata2[key]) & numpy.bitwise_not(numpy.isnan(npdata[key])) & numpy.bitwise_not(numpy.isnan(npdata2[key]))
                if numpy.any(diff):
                    raise AssertionError("npdata has been modified:\n{0}\n{1}\n{2}\n{3} vs {4}".format(npdata[key], npdata2[key], numpy.nonzero(diff), npdata[key][numpy.nonzero(diff)[0][0]], npdata2[key][numpy.nonzero(diff)[0][0]]))

        if isinstance(weight, numpy.ndarray):
            diff = (weight != weight2) & numpy.bitwise_not(numpy.isnan(weight)) & numpy.bitwise_not(numpy.isnan(weight2))
            if numpy.any(diff):
                raise AssertionError("weight has been modified:\n{0}\n{1}\n{2}\n{3} vs {4}".format(weight, weight2, numpy.nonzero(diff), weight[numpy.nonzero(diff)[0][0]], weight2[numpy.nonzero(diff)[0][0]]))

        hnp2.numpy(npdata, weight)
        hnp3.numpy(npdata, weight)
        hnp3.numpy(npdata, weight)
        assert (hnp + hnp2) == hnp3
        assert (hnp2 + hnp) == hnp3
        assert (hnp + hnp.zero()) == hnp2
        assert (hnp.zero() + hnp) == hnp2

        if isinstance(weight, numpy.ndarray):
            startTime = time.time()
            for d, w in zip(pydata, weight):
                if isinstance(d, numpy.unicode_):
                    d = str(d)
                else:
                    d = float(d)
                hpy.fill(d, float(w))
            pyTime = time.time() - startTime

            for h in [hpy2, hpy3, hpy3]:
                for d, w in zip(pydata, weight):
                    if isinstance(d, numpy.unicode_):
                        d = str(d)
                    else:
                        d = float(d)
                    h.fill(d, float(w))

            for h in [hpy4, hpy5, hpy5]:
                for d, w in zip(pydata, weight):
                    if isinstance(d, numpy.unicode_):
                        d = str(d)
                    else:
                        d = float(d)
                    test.specification.fill(h, d, float(w))

        else:
            startTime = time.time()
            for d in pydata:
                if isinstance(d, numpy.unicode_):
                    d = str(d)
                else:
                    d = float(d)
                hpy.fill(d, float(weight))
            pyTime = time.time() - startTime

            for h in [hpy2, hpy3, hpy3]:
                for d in pydata:
                    if isinstance(d, numpy.unicode_):
                        d = str(d)
                    else:
                        d = float(d)
                    h.fill(d, float(weight))

            for h in [hpy4, hpy5, hpy5]:
                for d in pydata:
                    if isinstance(d, numpy.unicode_):
                        d = str(d)
                    else:
                        d = float(d)
                    test.specification.fill(h, d, float(weight))

        assert (hpy + hpy) == hpy3
        assert (hpy + hpy2) == hpy3
        assert (hpy2 + hpy) == hpy3
        assert (hpy + hpy.zero()) == hpy2
        assert (hpy.zero() + hpy) == hpy2

        hnpj = json.dumps(hnp.toJson())
        hpyj = json.dumps(hpy.toJson())

        if Factory.fromJson(hnp.toJson()) != Factory.fromJson(hpy.toJson()):
            raise AssertionError("\n numpy: {0}\npython: {1}".format(hnpj, hpyj))
        else:
            sys.stderr.write("{0:45s} | numpy: {1:.3f}ms python: {2:.3f}ms = {3:g}X speedup\n".format(name, numpyTime*1000, pyTime*1000, self.twosigfigs(pyTime/numpyTime)))

        assert Factory.fromJson((hnp + hnp2).toJson()) == Factory.fromJson((hpy + hpy2).toJson())
        assert Factory.fromJson(hnp3.toJson()) == Factory.fromJson(hpy3.toJson())

        assert hpy2 == hpy4
        assert hpy3 == hpy5
        assert test.specification.combine(hpy2, hpy2) == test.specification.combine(hpy3, hpy3.zero())
        assert test.specification.combine(hpy2, hpy2) == test.specification.combine(hpy3.zero(), hpy3)

    # Warmup: apparently, Numpy does some dynamic optimization that needs to warm up...
    if empty is not None:
        Sum(lambda x: x["empty"]).numpy(data)
        Sum(lambda x: x["empty"]).numpy(data)
        Sum(lambda x: x["empty"]).numpy(data)
        Sum(lambda x: x["empty"]).numpy(data)
        Sum(lambda x: x["empty"]).numpy(data)

    def testSum(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            self.compare("Sum no data", Sum(lambda x: x["empty"]), self.data, Sum(lambda x: x), self.empty, 1.0)
            self.compare("Sum noholes w/o weights", Sum(lambda x: x["noholes"]), self.data, Sum(lambda x: x), self.noholes, 1.0)
            self.compare("Sum noholes const weights", Sum(lambda x: x["noholes"]), self.data, Sum(lambda x: x), self.noholes, 0.5)
            self.compare("Sum noholes positive weights", Sum(lambda x: x["noholes"]), self.data, Sum(lambda x: x), self.noholes, self.positive)
            self.compare("Sum noholes with weights", Sum(lambda x: x["noholes"]), self.data, Sum(lambda x: x), self.noholes, self.noholes)
            self.compare("Sum noholes with holes", Sum(lambda x: x["noholes"]), self.data, Sum(lambda x: x), self.noholes, self.withholes)
            self.compare("Sum holes w/o weights", Sum(lambda x: x["withholes"]), self.data, Sum(lambda x: x), self.withholes, 1.0)
            self.compare("Sum holes const weights", Sum(lambda x: x["withholes"]), self.data, Sum(lambda x: x), self.withholes, 0.5)
            self.compare("Sum holes positive weights", Sum(lambda x: x["withholes"]), self.data, Sum(lambda x: x), self.withholes, self.positive)
            self.compare("Sum holes with weights", Sum(lambda x: x["withholes"]), self.data, Sum(lambda x: x), self.withholes, self.noholes)
            self.compare("Sum holes with holes", Sum(lambda x: x["withholes"]), self.data, Sum(lambda x: x), self.withholes, self.withholes)
            self.compare("Sum holes with holes2", Sum(lambda x: x["withholes"]), self.data, Sum(lambda x: x), self.withholes, self.withholes2)

    def testAverage(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            self.compare("Average no data", Average(lambda x: x["empty"]), self.data, Average(lambda x: x), self.empty, 1.0)
            self.compare("Average noholes w/o weights", Average(lambda x: x["noholes"]), self.data, Average(lambda x: x), self.noholes, 1.0)
            self.compare("Average noholes const weights", Average(lambda x: x["noholes"]), self.data, Average(lambda x: x), self.noholes, 0.5)
            self.compare("Average noholes positive weights", Average(lambda x: x["noholes"]), self.data, Average(lambda x: x), self.noholes, self.positive)
            self.compare("Average noholes with weights", Average(lambda x: x["noholes"]), self.data, Average(lambda x: x), self.noholes, self.noholes)
            self.compare("Average noholes with holes", Average(lambda x: x["noholes"]), self.data, Average(lambda x: x), self.noholes, self.withholes)
            self.compare("Average holes w/o weights", Average(lambda x: x["withholes"]), self.data, Average(lambda x: x), self.withholes, 1.0)
            self.compare("Average holes const weights", Average(lambda x: x["withholes"]), self.data, Average(lambda x: x), self.withholes, 0.5)
            self.compare("Average holes positive weights", Average(lambda x: x["withholes"]), self.data, Average(lambda x: x), self.withholes, self.positive)
            self.compare("Average holes with weights", Average(lambda x: x["withholes"]), self.data, Average(lambda x: x), self.withholes, self.noholes)
            self.compare("Average holes with holes", Average(lambda x: x["withholes"]), self.data, Average(lambda x: x), self.withholes, self.withholes)
            self.compare("Average holes with holes2", Average(lambda x: x["withholes"]), self.data, Average(lambda x: x), self.withholes, self.withholes2)

    def testDeviate(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            self.compare("Deviate no data", Deviate(lambda x: x["empty"]), self.data, Deviate(lambda x: x), self.empty, 1.0)
            self.compare("Deviate noholes w/o weights", Deviate(lambda x: x["noholes"]), self.data, Deviate(lambda x: x), self.noholes, 1.0)
            self.compare("Deviate noholes const weights", Deviate(lambda x: x["noholes"]), self.data, Deviate(lambda x: x), self.noholes, 0.5)
            self.compare("Deviate noholes positive weights", Deviate(lambda x: x["noholes"]), self.data, Deviate(lambda x: x), self.noholes, self.positive)
            self.compare("Deviate noholes with weights", Deviate(lambda x: x["noholes"]), self.data, Deviate(lambda x: x), self.noholes, self.noholes)
            self.compare("Deviate noholes with holes", Deviate(lambda x: x["noholes"]), self.data, Deviate(lambda x: x), self.noholes, self.withholes)
            self.compare("Deviate holes w/o weights", Deviate(lambda x: x["withholes"]), self.data, Deviate(lambda x: x), self.withholes, 1.0)
            self.compare("Deviate holes const weights", Deviate(lambda x: x["withholes"]), self.data, Deviate(lambda x: x), self.withholes, 0.5)
            self.compare("Deviate holes positive weights", Deviate(lambda x: x["withholes"]), self.data, Deviate(lambda x: x), self.withholes, self.positive)
            self.compare("Deviate holes with weights", Deviate(lambda x: x["withholes"]), self.data, Deviate(lambda x: x), self.withholes, self.noholes)
            self.compare("Deviate holes with holes", Deviate(lambda x: x["withholes"]), self.data, Deviate(lambda x: x), self.withholes, self.withholes)
            self.compare("Deviate holes with holes2", Deviate(lambda x: x["withholes"]), self.data, Deviate(lambda x: x), self.withholes, self.withholes2)

    def testMinimize(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            self.compare("Minimize no data", Minimize(lambda x: x["empty"]), self.data, Minimize(lambda x: x), self.empty, 1.0)
            self.compare("Minimize noholes w/o weights", Minimize(lambda x: x["noholes"]), self.data, Minimize(lambda x: x), self.noholes, 1.0)
            self.compare("Minimize noholes const weights", Minimize(lambda x: x["noholes"]), self.data, Minimize(lambda x: x), self.noholes, 0.5)
            self.compare("Minimize noholes positive weights", Minimize(lambda x: x["noholes"]), self.data, Minimize(lambda x: x), self.noholes, self.positive)
            self.compare("Minimize noholes with weights", Minimize(lambda x: x["noholes"]), self.data, Minimize(lambda x: x), self.noholes, self.noholes)
            self.compare("Minimize noholes with holes", Minimize(lambda x: x["noholes"]), self.data, Minimize(lambda x: x), self.noholes, self.withholes)
            self.compare("Minimize holes w/o weights", Minimize(lambda x: x["withholes"]), self.data, Minimize(lambda x: x), self.withholes, 1.0)
            self.compare("Minimize holes const weights", Minimize(lambda x: x["withholes"]), self.data, Minimize(lambda x: x), self.withholes, 0.5)
            self.compare("Minimize holes positive weights", Minimize(lambda x: x["withholes"]), self.data, Minimize(lambda x: x), self.withholes, self.positive)
            self.compare("Minimize holes with weights", Minimize(lambda x: x["withholes"]), self.data, Minimize(lambda x: x), self.withholes, self.noholes)
            self.compare("Minimize holes with holes", Minimize(lambda x: x["withholes"]), self.data, Minimize(lambda x: x), self.withholes, self.withholes)
            self.compare("Minimize holes with holes2", Minimize(lambda x: x["withholes"]), self.data, Minimize(lambda x: x), self.withholes, self.withholes2)

    def testMaximize(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            self.compare("Maximize no data", Maximize(lambda x: x["empty"]), self.data, Maximize(lambda x: x), self.empty, 1.0)
            self.compare("Maximize noholes w/o weights", Maximize(lambda x: x["noholes"]), self.data, Maximize(lambda x: x), self.noholes, 1.0)
            self.compare("Maximize noholes const weights", Maximize(lambda x: x["noholes"]), self.data, Maximize(lambda x: x), self.noholes, 0.5)
            self.compare("Maximize noholes positive weights", Maximize(lambda x: x["noholes"]), self.data, Maximize(lambda x: x), self.noholes, self.positive)
            self.compare("Maximize noholes with weights", Maximize(lambda x: x["noholes"]), self.data, Maximize(lambda x: x), self.noholes, self.noholes)
            self.compare("Maximize noholes with holes", Maximize(lambda x: x["noholes"]), self.data, Maximize(lambda x: x), self.noholes, self.withholes)
            self.compare("Maximize holes w/o weights", Maximize(lambda x: x["withholes"]), self.data, Maximize(lambda x: x), self.withholes, 1.0)
            self.compare("Maximize holes const weights", Maximize(lambda x: x["withholes"]), self.data, Maximize(lambda x: x), self.withholes, 0.5)
            self.compare("Maximize holes positive weights", Maximize(lambda x: x["withholes"]), self.data, Maximize(lambda x: x), self.withholes, self.positive)
            self.compare("Maximize holes with weights", Maximize(lambda x: x["withholes"]), self.data, Maximize(lambda x: x), self.withholes, self.noholes)
            self.compare("Maximize holes with holes", Maximize(lambda x: x["withholes"]), self.data, Maximize(lambda x: x), self.withholes, self.withholes)
            self.compare("Maximize holes with holes2", Maximize(lambda x: x["withholes"]), self.data, Maximize(lambda x: x), self.withholes, self.withholes2)

    def testBin(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            for bins in [10, 100]:
                self.compare("Bin ({0} bins) no data".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["empty"]), self.data, Bin(bins, -3.0, 3.0, lambda x: x), self.empty, 1.0)
                self.compare("Bin ({0} bins) noholes w/o weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["noholes"]), self.data, Bin(bins, -3.0, 3.0, lambda x: x), self.noholes, 1.0)
                self.compare("Bin ({0} bins) noholes const weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["noholes"]), self.data, Bin(bins, -3.0, 3.0, lambda x: x), self.noholes, 0.5)
                self.compare("Bin ({0} bins) noholes positive weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["noholes"]), self.data, Bin(bins, -3.0, 3.0, lambda x: x), self.noholes, self.positive)
                self.compare("Bin ({0} bins) noholes with weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["noholes"]), self.data, Bin(bins, -3.0, 3.0, lambda x: x), self.noholes, self.noholes)
                self.compare("Bin ({0} bins) noholes with holes".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["noholes"]), self.data, Bin(bins, -3.0, 3.0, lambda x: x), self.noholes, self.withholes)
                self.compare("Bin ({0} bins) holes w/o weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"]), self.data, Bin(bins, -3.0, 3.0, lambda x: x), self.withholes, 1.0)
                self.compare("Bin ({0} bins) holes const weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"]), self.data, Bin(bins, -3.0, 3.0, lambda x: x), self.withholes, 0.5)
                self.compare("Bin ({0} bins) holes positive weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"]), self.data, Bin(bins, -3.0, 3.0, lambda x: x), self.withholes, self.positive)
                self.compare("Bin ({0} bins) holes with weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"]), self.data, Bin(bins, -3.0, 3.0, lambda x: x), self.withholes, self.noholes)
                self.compare("Bin ({0} bins) holes with holes".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"]), self.data, Bin(bins, -3.0, 3.0, lambda x: x), self.withholes, self.withholes)
                self.compare("Bin ({0} bins) holes with holes2".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"]), self.data, Bin(bins, -3.0, 3.0, lambda x: x), self.withholes, self.withholes2)

    def testBinTrans(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            for bins in [10, 100]:
                self.compare("BinTrans ({0} bins) no data".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["empty"], Count(lambda x: 0.5*x)), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Count(lambda x: 0.5*x)), self.empty, 1.0)
                self.compare("BinTrans ({0} bins) noholes w/o weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["noholes"], Count(lambda x: 0.5*x)), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Count(lambda x: 0.5*x)), self.noholes, 1.0)
                self.compare("BinTrans ({0} bins) noholes const weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["noholes"], Count(lambda x: 0.5*x)), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Count(lambda x: 0.5*x)), self.noholes, 0.5)
                self.compare("BinTrans ({0} bins) noholes positive weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["noholes"], Count(lambda x: 0.5*x)), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Count(lambda x: 0.5*x)), self.noholes, self.positive)
                self.compare("BinTrans ({0} bins) noholes with weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["noholes"], Count(lambda x: 0.5*x)), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Count(lambda x: 0.5*x)), self.noholes, self.noholes)
                self.compare("BinTrans ({0} bins) noholes with holes".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["noholes"], Count(lambda x: 0.5*x)), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Count(lambda x: 0.5*x)), self.noholes, self.withholes)
                self.compare("BinTrans ({0} bins) holes w/o weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"], Count(lambda x: 0.5*x)), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Count(lambda x: 0.5*x)), self.withholes, 1.0)
                self.compare("BinTrans ({0} bins) holes const weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"], Count(lambda x: 0.5*x)), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Count(lambda x: 0.5*x)), self.withholes, 0.5)
                self.compare("BinTrans ({0} bins) holes positive weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"], Count(lambda x: 0.5*x)), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Count(lambda x: 0.5*x)), self.withholes, self.positive)
                self.compare("BinTrans ({0} bins) holes with weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"], Count(lambda x: 0.5*x)), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Count(lambda x: 0.5*x)), self.withholes, self.noholes)
                self.compare("BinTrans ({0} bins) holes with holes".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"], Count(lambda x: 0.5*x)), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Count(lambda x: 0.5*x)), self.withholes, self.withholes)
                self.compare("BinTrans ({0} bins) holes with holes2".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"], Count(lambda x: 0.5*x)), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Count(lambda x: 0.5*x)), self.withholes, self.withholes2)

    def testBinAverage(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            for bins in [10, 100]:
                self.compare("BinAverage ({0} bins) no data".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["empty"], Average(lambda x: x["empty"])), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Average(lambda x: x)), self.empty, 1.0)
                self.compare("BinAverage ({0} bins) noholes w/o weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["noholes"], Average(lambda x: x["noholes"])), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Average(lambda x: x)), self.noholes, 1.0)
                self.compare("BinAverage ({0} bins) noholes const weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["noholes"], Average(lambda x: x["noholes"])), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Average(lambda x: x)), self.noholes, 0.5)
                self.compare("BinAverage ({0} bins) noholes positive weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["noholes"], Average(lambda x: x["noholes"])), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Average(lambda x: x)), self.noholes, self.positive)
                self.compare("BinAverage ({0} bins) noholes with weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["noholes"], Average(lambda x: x["noholes"])), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Average(lambda x: x)), self.noholes, self.noholes)
                self.compare("BinAverage ({0} bins) noholes with holes".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["noholes"], Average(lambda x: x["noholes"])), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Average(lambda x: x)), self.noholes, self.withholes)
                self.compare("BinAverage ({0} bins) holes w/o weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"], Average(lambda x: x["withholes"])), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Average(lambda x: x)), self.withholes, 1.0)
                self.compare("BinAverage ({0} bins) holes const weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"], Average(lambda x: x["withholes"])), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Average(lambda x: x)), self.withholes, 0.5)
                self.compare("BinAverage ({0} bins) holes positive weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"], Average(lambda x: x["withholes"])), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Average(lambda x: x)), self.withholes, self.positive)
                self.compare("BinAverage ({0} bins) holes with weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"], Average(lambda x: x["withholes"])), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Average(lambda x: x)), self.withholes, self.noholes)
                self.compare("BinAverage ({0} bins) holes with holes".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"], Average(lambda x: x["withholes"])), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Average(lambda x: x)), self.withholes, self.withholes)
                self.compare("BinAverage ({0} bins) holes with holes2".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"], Average(lambda x: x["withholes"])), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Average(lambda x: x)), self.withholes, self.withholes2)

    def testBinDeviate(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            for bins in [10, 100]:
                self.compare("BinDeviate ({0} bins) no data".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["empty"], Deviate(lambda x: x["empty"])), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Deviate(lambda x: x)), self.empty, 1.0)
                self.compare("BinDeviate ({0} bins) noholes w/o weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["noholes"], Deviate(lambda x: x["noholes"])), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Deviate(lambda x: x)), self.noholes, 1.0)
                self.compare("BinDeviate ({0} bins) noholes const weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["noholes"], Deviate(lambda x: x["noholes"])), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Deviate(lambda x: x)), self.noholes, 0.5)
                self.compare("BinDeviate ({0} bins) noholes positive weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["noholes"], Deviate(lambda x: x["noholes"])), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Deviate(lambda x: x)), self.noholes, self.positive)
                self.compare("BinDeviate ({0} bins) noholes with weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["noholes"], Deviate(lambda x: x["noholes"])), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Deviate(lambda x: x)), self.noholes, self.noholes)
                self.compare("BinDeviate ({0} bins) noholes with holes".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["noholes"], Deviate(lambda x: x["noholes"])), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Deviate(lambda x: x)), self.noholes, self.withholes)
                self.compare("BinDeviate ({0} bins) holes w/o weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"], Deviate(lambda x: x["withholes"])), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Deviate(lambda x: x)), self.withholes, 1.0)
                self.compare("BinDeviate ({0} bins) holes const weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"], Deviate(lambda x: x["withholes"])), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Deviate(lambda x: x)), self.withholes, 0.5)
                self.compare("BinDeviate ({0} bins) holes positive weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"], Deviate(lambda x: x["withholes"])), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Deviate(lambda x: x)), self.withholes, self.positive)
                self.compare("BinDeviate ({0} bins) holes with weights".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"], Deviate(lambda x: x["withholes"])), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Deviate(lambda x: x)), self.withholes, self.noholes)
                self.compare("BinDeviate ({0} bins) holes with holes".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"], Deviate(lambda x: x["withholes"])), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Deviate(lambda x: x)), self.withholes, self.withholes)
                self.compare("BinDeviate ({0} bins) holes with holes2".format(bins), Bin(bins, -3.0, 3.0, lambda x: x["withholes"], Deviate(lambda x: x["withholes"])), self.data, Bin(bins, -3.0, 3.0, lambda x: x, Deviate(lambda x: x)), self.withholes, self.withholes2)

    def testSparselyBin(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            self.compare("SparselyBin no data", SparselyBin(0.1, lambda x: x["empty"]), self.data, SparselyBin(0.1, lambda x: x), self.empty, 1.0)
            self.compare("SparselyBin noholes w/o weights", SparselyBin(0.1, lambda x: x["noholes"]), self.data, SparselyBin(0.1, lambda x: x), self.noholes, 1.0)
            self.compare("SparselyBin noholes const weights", SparselyBin(0.1, lambda x: x["noholes"]), self.data, SparselyBin(0.1, lambda x: x), self.noholes, 0.5)
            self.compare("SparselyBin noholes positive weights", SparselyBin(0.1, lambda x: x["noholes"]), self.data, SparselyBin(0.1, lambda x: x), self.noholes, self.positive)
            self.compare("SparselyBin noholes with weights", SparselyBin(0.1, lambda x: x["noholes"]), self.data, SparselyBin(0.1, lambda x: x), self.noholes, self.noholes)
            self.compare("SparselyBin noholes with holes", SparselyBin(0.1, lambda x: x["noholes"]), self.data, SparselyBin(0.1, lambda x: x), self.noholes, self.withholes)
            self.compare("SparselyBin holes w/o weights", SparselyBin(0.1, lambda x: x["withholes"]), self.data, SparselyBin(0.1, lambda x: x), self.withholes, 1.0)
            self.compare("SparselyBin holes const weights", SparselyBin(0.1, lambda x: x["withholes"]), self.data, SparselyBin(0.1, lambda x: x), self.withholes, 0.5)
            self.compare("SparselyBin holes positive weights", SparselyBin(0.1, lambda x: x["withholes"]), self.data, SparselyBin(0.1, lambda x: x), self.withholes, self.positive)
            self.compare("SparselyBin holes with weights", SparselyBin(0.1, lambda x: x["withholes"]), self.data, SparselyBin(0.1, lambda x: x), self.withholes, self.noholes)
            self.compare("SparselyBin holes with holes", SparselyBin(0.1, lambda x: x["withholes"]), self.data, SparselyBin(0.1, lambda x: x), self.withholes, self.withholes)
            self.compare("SparselyBin holes with holes2", SparselyBin(0.1, lambda x: x["withholes"]), self.data, SparselyBin(0.1, lambda x: x), self.withholes, self.withholes2)

    def testSparselyBinTrans(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            self.compare("SparselyBinTrans no data", SparselyBin(0.1, lambda x: x["empty"], Count(lambda x: 0.5*x)), self.data, SparselyBin(0.1, lambda x: x, Count(lambda x: 0.5*x)), self.empty, 1.0)
            self.compare("SparselyBinTrans noholes w/o weights", SparselyBin(0.1, lambda x: x["noholes"], Count(lambda x: 0.5*x)), self.data, SparselyBin(0.1, lambda x: x, Count(lambda x: 0.5*x)), self.noholes, 1.0)
            self.compare("SparselyBinTrans noholes const weights", SparselyBin(0.1, lambda x: x["noholes"], Count(lambda x: 0.5*x)), self.data, SparselyBin(0.1, lambda x: x, Count(lambda x: 0.5*x)), self.noholes, 0.5)
            self.compare("SparselyBinTrans noholes positive weights", SparselyBin(0.1, lambda x: x["noholes"], Count(lambda x: 0.5*x)), self.data, SparselyBin(0.1, lambda x: x, Count(lambda x: 0.5*x)), self.noholes, self.positive)
            self.compare("SparselyBinTrans noholes with weights", SparselyBin(0.1, lambda x: x["noholes"], Count(lambda x: 0.5*x)), self.data, SparselyBin(0.1, lambda x: x, Count(lambda x: 0.5*x)), self.noholes, self.noholes)
            self.compare("SparselyBinTrans noholes with holes", SparselyBin(0.1, lambda x: x["noholes"], Count(lambda x: 0.5*x)), self.data, SparselyBin(0.1, lambda x: x, Count(lambda x: 0.5*x)), self.noholes, self.withholes)
            self.compare("SparselyBinTrans holes w/o weights", SparselyBin(0.1, lambda x: x["withholes"], Count(lambda x: 0.5*x)), self.data, SparselyBin(0.1, lambda x: x, Count(lambda x: 0.5*x)), self.withholes, 1.0)
            self.compare("SparselyBinTrans holes const weights", SparselyBin(0.1, lambda x: x["withholes"], Count(lambda x: 0.5*x)), self.data, SparselyBin(0.1, lambda x: x, Count(lambda x: 0.5*x)), self.withholes, 0.5)
            self.compare("SparselyBinTrans holes positive weights", SparselyBin(0.1, lambda x: x["withholes"], Count(lambda x: 0.5*x)), self.data, SparselyBin(0.1, lambda x: x, Count(lambda x: 0.5*x)), self.withholes, self.positive)
            self.compare("SparselyBinTrans holes with weights", SparselyBin(0.1, lambda x: x["withholes"], Count(lambda x: 0.5*x)), self.data, SparselyBin(0.1, lambda x: x, Count(lambda x: 0.5*x)), self.withholes, self.noholes)
            self.compare("SparselyBinTrans holes with holes", SparselyBin(0.1, lambda x: x["withholes"], Count(lambda x: 0.5*x)), self.data, SparselyBin(0.1, lambda x: x, Count(lambda x: 0.5*x)), self.withholes, self.withholes)
            self.compare("SparselyBinTrans holes with holes2", SparselyBin(0.1, lambda x: x["withholes"], Count(lambda x: 0.5*x)), self.data, SparselyBin(0.1, lambda x: x, Count(lambda x: 0.5*x)), self.withholes, self.withholes2)

    def testSparselyBinAverage(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            self.compare("SparselyBinAverage no data", SparselyBin(0.1, lambda x: x["empty"], Average(lambda x: x["empty"])), self.data, SparselyBin(0.1, lambda x: x, Average(lambda x: x)), self.empty, 1.0)
            self.compare("SparselyBinAverage noholes w/o weights", SparselyBin(0.1, lambda x: x["noholes"], Average(lambda x: x["noholes"])), self.data, SparselyBin(0.1, lambda x: x, Average(lambda x: x)), self.noholes, 1.0)
            self.compare("SparselyBinAverage noholes const weights", SparselyBin(0.1, lambda x: x["noholes"], Average(lambda x: x["noholes"])), self.data, SparselyBin(0.1, lambda x: x, Average(lambda x: x)), self.noholes, 0.5)
            self.compare("SparselyBinAverage noholes positive weights", SparselyBin(0.1, lambda x: x["noholes"], Average(lambda x: x["noholes"])), self.data, SparselyBin(0.1, lambda x: x, Average(lambda x: x)), self.noholes, self.positive)
            self.compare("SparselyBinAverage noholes with weights", SparselyBin(0.1, lambda x: x["noholes"], Average(lambda x: x["noholes"])), self.data, SparselyBin(0.1, lambda x: x, Average(lambda x: x)), self.noholes, self.noholes)
            self.compare("SparselyBinAverage noholes with holes", SparselyBin(0.1, lambda x: x["noholes"], Average(lambda x: x["noholes"])), self.data, SparselyBin(0.1, lambda x: x, Average(lambda x: x)), self.noholes, self.withholes)
            self.compare("SparselyBinAverage holes w/o weights", SparselyBin(0.1, lambda x: x["withholes"], Average(lambda x: x["withholes"])), self.data, SparselyBin(0.1, lambda x: x, Average(lambda x: x)), self.withholes, 1.0)
            self.compare("SparselyBinAverage holes const weights", SparselyBin(0.1, lambda x: x["withholes"], Average(lambda x: x["withholes"])), self.data, SparselyBin(0.1, lambda x: x, Average(lambda x: x)), self.withholes, 0.5)
            self.compare("SparselyBinAverage holes positive weights", SparselyBin(0.1, lambda x: x["withholes"], Average(lambda x: x["withholes"])), self.data, SparselyBin(0.1, lambda x: x, Average(lambda x: x)), self.withholes, self.positive)
            self.compare("SparselyBinAverage holes with weights", SparselyBin(0.1, lambda x: x["withholes"], Average(lambda x: x["withholes"])), self.data, SparselyBin(0.1, lambda x: x, Average(lambda x: x)), self.withholes, self.noholes)
            self.compare("SparselyBinAverage holes with holes", SparselyBin(0.1, lambda x: x["withholes"], Average(lambda x: x["withholes"])), self.data, SparselyBin(0.1, lambda x: x, Average(lambda x: x)), self.withholes, self.withholes)
            self.compare("SparselyBinAverage holes with holes2", SparselyBin(0.1, lambda x: x["withholes"], Average(lambda x: x["withholes"])), self.data, SparselyBin(0.1, lambda x: x, Average(lambda x: x)), self.withholes, self.withholes2)

    def testSparselyBinDeviate(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            self.compare("SparselyBinDeviate no data", SparselyBin(0.1, lambda x: x["empty"], Deviate(lambda x: x["empty"])), self.data, SparselyBin(0.1, lambda x: x, Deviate(lambda x: x)), self.empty, 1.0)
            self.compare("SparselyBinDeviate noholes w/o weights", SparselyBin(0.1, lambda x: x["noholes"], Deviate(lambda x: x["noholes"])), self.data, SparselyBin(0.1, lambda x: x, Deviate(lambda x: x)), self.noholes, 1.0)
            self.compare("SparselyBinDeviate noholes const weights", SparselyBin(0.1, lambda x: x["noholes"], Deviate(lambda x: x["noholes"])), self.data, SparselyBin(0.1, lambda x: x, Deviate(lambda x: x)), self.noholes, 0.5)
            self.compare("SparselyBinDeviate noholes positive weights", SparselyBin(0.1, lambda x: x["noholes"], Deviate(lambda x: x["noholes"])), self.data, SparselyBin(0.1, lambda x: x, Deviate(lambda x: x)), self.noholes, self.positive)
            self.compare("SparselyBinDeviate noholes with weights", SparselyBin(0.1, lambda x: x["noholes"], Deviate(lambda x: x["noholes"])), self.data, SparselyBin(0.1, lambda x: x, Deviate(lambda x: x)), self.noholes, self.noholes)
            self.compare("SparselyBinDeviate noholes with holes", SparselyBin(0.1, lambda x: x["noholes"], Deviate(lambda x: x["noholes"])), self.data, SparselyBin(0.1, lambda x: x, Deviate(lambda x: x)), self.noholes, self.withholes)
            self.compare("SparselyBinDeviate holes w/o weights", SparselyBin(0.1, lambda x: x["withholes"], Deviate(lambda x: x["withholes"])), self.data, SparselyBin(0.1, lambda x: x, Deviate(lambda x: x)), self.withholes, 1.0)
            self.compare("SparselyBinDeviate holes const weights", SparselyBin(0.1, lambda x: x["withholes"], Deviate(lambda x: x["withholes"])), self.data, SparselyBin(0.1, lambda x: x, Deviate(lambda x: x)), self.withholes, 0.5)
            self.compare("SparselyBinDeviate holes positive weights", SparselyBin(0.1, lambda x: x["withholes"], Deviate(lambda x: x["withholes"])), self.data, SparselyBin(0.1, lambda x: x, Deviate(lambda x: x)), self.withholes, self.positive)
            self.compare("SparselyBinDeviate holes with weights", SparselyBin(0.1, lambda x: x["withholes"], Deviate(lambda x: x["withholes"])), self.data, SparselyBin(0.1, lambda x: x, Deviate(lambda x: x)), self.withholes, self.noholes)
            self.compare("SparselyBinDeviate holes with holes", SparselyBin(0.1, lambda x: x["withholes"], Deviate(lambda x: x["withholes"])), self.data, SparselyBin(0.1, lambda x: x, Deviate(lambda x: x)), self.withholes, self.withholes)
            self.compare("SparselyBinDeviate holes with holes2", SparselyBin(0.1, lambda x: x["withholes"], Deviate(lambda x: x["withholes"])), self.data, SparselyBin(0.1, lambda x: x, Deviate(lambda x: x)), self.withholes, self.withholes2)

    def testCentrallyBin(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            centers = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
            self.compare("CentrallyBin no data", CentrallyBin(centers, lambda x: x["empty"]), self.data, CentrallyBin(centers, lambda x: x), self.empty, 1.0)
            self.compare("CentrallyBin noholes w/o weights", CentrallyBin(centers, lambda x: x["noholes"]), self.data, CentrallyBin(centers, lambda x: x), self.noholes, 1.0)
            self.compare("CentrallyBin noholes const weights", CentrallyBin(centers, lambda x: x["noholes"]), self.data, CentrallyBin(centers, lambda x: x), self.noholes, 0.5)
            self.compare("CentrallyBin noholes positive weights", CentrallyBin(centers, lambda x: x["noholes"]), self.data, CentrallyBin(centers, lambda x: x), self.noholes, self.positive)
            self.compare("CentrallyBin noholes with weights", CentrallyBin(centers, lambda x: x["noholes"]), self.data, CentrallyBin(centers, lambda x: x), self.noholes, self.noholes)
            self.compare("CentrallyBin noholes with holes", CentrallyBin(centers, lambda x: x["noholes"]), self.data, CentrallyBin(centers, lambda x: x), self.noholes, self.withholes)
            self.compare("CentrallyBin holes w/o weights", CentrallyBin(centers, lambda x: x["withholes"]), self.data, CentrallyBin(centers, lambda x: x), self.withholes, 1.0)
            self.compare("CentrallyBin holes const weights", CentrallyBin(centers, lambda x: x["withholes"]), self.data, CentrallyBin(centers, lambda x: x), self.withholes, 0.5)
            self.compare("CentrallyBin holes positive weights", CentrallyBin(centers, lambda x: x["withholes"]), self.data, CentrallyBin(centers, lambda x: x), self.withholes, self.positive)
            self.compare("CentrallyBin holes with weights", CentrallyBin(centers, lambda x: x["withholes"]), self.data, CentrallyBin(centers, lambda x: x), self.withholes, self.noholes)
            self.compare("CentrallyBin holes with holes", CentrallyBin(centers, lambda x: x["withholes"]), self.data, CentrallyBin(centers, lambda x: x), self.withholes, self.withholes)
            self.compare("CentrallyBin holes with holes2", CentrallyBin(centers, lambda x: x["withholes"]), self.data, CentrallyBin(centers, lambda x: x), self.withholes, self.withholes2)

    def testCentrallyBinTrans(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            centers = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
            self.compare("CentrallyBinTrans no data", CentrallyBin(centers, lambda x: x["empty"], Count(lambda x: 0.5*x)), self.data, CentrallyBin(centers, lambda x: x, Count(lambda x: 0.5*x)), self.empty, 1.0)
            self.compare("CentrallyBinTrans noholes w/o weights", CentrallyBin(centers, lambda x: x["noholes"], Count(lambda x: 0.5*x)), self.data, CentrallyBin(centers, lambda x: x, Count(lambda x: 0.5*x)), self.noholes, 1.0)
            self.compare("CentrallyBinTrans noholes const weights", CentrallyBin(centers, lambda x: x["noholes"], Count(lambda x: 0.5*x)), self.data, CentrallyBin(centers, lambda x: x, Count(lambda x: 0.5*x)), self.noholes, 0.5)
            self.compare("CentrallyBinTrans noholes positive weights", CentrallyBin(centers, lambda x: x["noholes"], Count(lambda x: 0.5*x)), self.data, CentrallyBin(centers, lambda x: x, Count(lambda x: 0.5*x)), self.noholes, self.positive)
            self.compare("CentrallyBinTrans noholes with weights", CentrallyBin(centers, lambda x: x["noholes"], Count(lambda x: 0.5*x)), self.data, CentrallyBin(centers, lambda x: x, Count(lambda x: 0.5*x)), self.noholes, self.noholes)
            self.compare("CentrallyBinTrans noholes with holes", CentrallyBin(centers, lambda x: x["noholes"], Count(lambda x: 0.5*x)), self.data, CentrallyBin(centers, lambda x: x, Count(lambda x: 0.5*x)), self.noholes, self.withholes)
            self.compare("CentrallyBinTrans holes w/o weights", CentrallyBin(centers, lambda x: x["withholes"], Count(lambda x: 0.5*x)), self.data, CentrallyBin(centers, lambda x: x, Count(lambda x: 0.5*x)), self.withholes, 1.0)
            self.compare("CentrallyBinTrans holes const weights", CentrallyBin(centers, lambda x: x["withholes"], Count(lambda x: 0.5*x)), self.data, CentrallyBin(centers, lambda x: x, Count(lambda x: 0.5*x)), self.withholes, 0.5)
            self.compare("CentrallyBinTrans holes positive weights", CentrallyBin(centers, lambda x: x["withholes"], Count(lambda x: 0.5*x)), self.data, CentrallyBin(centers, lambda x: x, Count(lambda x: 0.5*x)), self.withholes, self.positive)
            self.compare("CentrallyBinTrans holes with weights", CentrallyBin(centers, lambda x: x["withholes"], Count(lambda x: 0.5*x)), self.data, CentrallyBin(centers, lambda x: x, Count(lambda x: 0.5*x)), self.withholes, self.noholes)
            self.compare("CentrallyBinTrans holes with holes", CentrallyBin(centers, lambda x: x["withholes"], Count(lambda x: 0.5*x)), self.data, CentrallyBin(centers, lambda x: x, Count(lambda x: 0.5*x)), self.withholes, self.withholes)
            self.compare("CentrallyBinTrans holes with holes2", CentrallyBin(centers, lambda x: x["withholes"], Count(lambda x: 0.5*x)), self.data, CentrallyBin(centers, lambda x: x, Count(lambda x: 0.5*x)), self.withholes, self.withholes2)

    def testCentrallyBinAverage(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            centers = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
            self.compare("CentrallyBinAverage no data", CentrallyBin(centers, lambda x: x["empty"], Average(lambda x: x["empty"])), self.data, CentrallyBin(centers, lambda x: x, Average(lambda x: x)), self.empty, 1.0)
            self.compare("CentrallyBinAverage noholes w/o weights", CentrallyBin(centers, lambda x: x["noholes"], Average(lambda x: x["noholes"])), self.data, CentrallyBin(centers, lambda x: x, Average(lambda x: x)), self.noholes, 1.0)
            self.compare("CentrallyBinAverage noholes const weights", CentrallyBin(centers, lambda x: x["noholes"], Average(lambda x: x["noholes"])), self.data, CentrallyBin(centers, lambda x: x, Average(lambda x: x)), self.noholes, 0.5)
            self.compare("CentrallyBinAverage noholes positive weights", CentrallyBin(centers, lambda x: x["noholes"], Average(lambda x: x["noholes"])), self.data, CentrallyBin(centers, lambda x: x, Average(lambda x: x)), self.noholes, self.positive)
            self.compare("CentrallyBinAverage noholes with weights", CentrallyBin(centers, lambda x: x["noholes"], Average(lambda x: x["noholes"])), self.data, CentrallyBin(centers, lambda x: x, Average(lambda x: x)), self.noholes, self.noholes)
            self.compare("CentrallyBinAverage noholes with holes", CentrallyBin(centers, lambda x: x["noholes"], Average(lambda x: x["noholes"])), self.data, CentrallyBin(centers, lambda x: x, Average(lambda x: x)), self.noholes, self.withholes)
            self.compare("CentrallyBinAverage holes w/o weights", CentrallyBin(centers, lambda x: x["withholes"], Average(lambda x: x["withholes"])), self.data, CentrallyBin(centers, lambda x: x, Average(lambda x: x)), self.withholes, 1.0)
            self.compare("CentrallyBinAverage holes const weights", CentrallyBin(centers, lambda x: x["withholes"], Average(lambda x: x["withholes"])), self.data, CentrallyBin(centers, lambda x: x, Average(lambda x: x)), self.withholes, 0.5)
            self.compare("CentrallyBinAverage holes positive weights", CentrallyBin(centers, lambda x: x["withholes"], Average(lambda x: x["withholes"])), self.data, CentrallyBin(centers, lambda x: x, Average(lambda x: x)), self.withholes, self.positive)
            self.compare("CentrallyBinAverage holes with weights", CentrallyBin(centers, lambda x: x["withholes"], Average(lambda x: x["withholes"])), self.data, CentrallyBin(centers, lambda x: x, Average(lambda x: x)), self.withholes, self.noholes)
            self.compare("CentrallyBinAverage holes with holes", CentrallyBin(centers, lambda x: x["withholes"], Average(lambda x: x["withholes"])), self.data, CentrallyBin(centers, lambda x: x, Average(lambda x: x)), self.withholes, self.withholes)
            self.compare("CentrallyBinAverage holes with holes2", CentrallyBin(centers, lambda x: x["withholes"], Average(lambda x: x["withholes"])), self.data, CentrallyBin(centers, lambda x: x, Average(lambda x: x)), self.withholes, self.withholes2)

    def testCentrallyBinDeviate(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            centers = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
            self.compare("CentrallyBinDeviate no data", CentrallyBin(centers, lambda x: x["empty"], Deviate(lambda x: x["empty"])), self.data, CentrallyBin(centers, lambda x: x, Deviate(lambda x: x)), self.empty, 1.0)
            self.compare("CentrallyBinDeviate noholes w/o weights", CentrallyBin(centers, lambda x: x["noholes"], Deviate(lambda x: x["noholes"])), self.data, CentrallyBin(centers, lambda x: x, Deviate(lambda x: x)), self.noholes, 1.0)
            self.compare("CentrallyBinDeviate noholes const weights", CentrallyBin(centers, lambda x: x["noholes"], Deviate(lambda x: x["noholes"])), self.data, CentrallyBin(centers, lambda x: x, Deviate(lambda x: x)), self.noholes, 0.5)
            self.compare("CentrallyBinDeviate noholes positive weights", CentrallyBin(centers, lambda x: x["noholes"], Deviate(lambda x: x["noholes"])), self.data, CentrallyBin(centers, lambda x: x, Deviate(lambda x: x)), self.noholes, self.positive)
            self.compare("CentrallyBinDeviate noholes with weights", CentrallyBin(centers, lambda x: x["noholes"], Deviate(lambda x: x["noholes"])), self.data, CentrallyBin(centers, lambda x: x, Deviate(lambda x: x)), self.noholes, self.noholes)
            self.compare("CentrallyBinDeviate noholes with holes", CentrallyBin(centers, lambda x: x["noholes"], Deviate(lambda x: x["noholes"])), self.data, CentrallyBin(centers, lambda x: x, Deviate(lambda x: x)), self.noholes, self.withholes)
            self.compare("CentrallyBinDeviate holes w/o weights", CentrallyBin(centers, lambda x: x["withholes"], Deviate(lambda x: x["withholes"])), self.data, CentrallyBin(centers, lambda x: x, Deviate(lambda x: x)), self.withholes, 1.0)
            self.compare("CentrallyBinDeviate holes const weights", CentrallyBin(centers, lambda x: x["withholes"], Deviate(lambda x: x["withholes"])), self.data, CentrallyBin(centers, lambda x: x, Deviate(lambda x: x)), self.withholes, 0.5)
            self.compare("CentrallyBinDeviate holes positive weights", CentrallyBin(centers, lambda x: x["withholes"], Deviate(lambda x: x["withholes"])), self.data, CentrallyBin(centers, lambda x: x, Deviate(lambda x: x)), self.withholes, self.positive)
            self.compare("CentrallyBinDeviate holes with weights", CentrallyBin(centers, lambda x: x["withholes"], Deviate(lambda x: x["withholes"])), self.data, CentrallyBin(centers, lambda x: x, Deviate(lambda x: x)), self.withholes, self.noholes)
            self.compare("CentrallyBinDeviate holes with holes", CentrallyBin(centers, lambda x: x["withholes"], Deviate(lambda x: x["withholes"])), self.data, CentrallyBin(centers, lambda x: x, Deviate(lambda x: x)), self.withholes, self.withholes)
            self.compare("CentrallyBinDeviate holes with holes2", CentrallyBin(centers, lambda x: x["withholes"], Deviate(lambda x: x["withholes"])), self.data, CentrallyBin(centers, lambda x: x, Deviate(lambda x: x)), self.withholes, self.withholes2)

    def testCategorize(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            self.compare("Categorize no data", Categorize(lambda x: numpy.array(numpy.floor(x["empty"]), dtype="<U5")), self.data, Categorize(lambda x: x), numpy.array(numpy.floor(self.empty), dtype="<U5"), 1.0)
            self.compare("Categorize noholes w/o weights", Categorize(lambda x: numpy.array(numpy.floor(x["noholes"]), dtype="<U5")), self.data, Categorize(lambda x: x), numpy.array(numpy.floor(self.noholes), dtype="<U5"), 1.0)
            self.compare("Categorize noholes const weights", Categorize(lambda x: numpy.array(numpy.floor(x["noholes"]), dtype="<U5")), self.data, Categorize(lambda x: x), numpy.array(numpy.floor(self.noholes), dtype="<U5"), 0.5)
            self.compare("Categorize noholes positive weights", Categorize(lambda x: numpy.array(numpy.floor(x["noholes"]), dtype="<U5")), self.data, Categorize(lambda x: x), numpy.array(numpy.floor(self.noholes), dtype="<U5"), self.positive)
            self.compare("Categorize noholes with weights", Categorize(lambda x: numpy.array(numpy.floor(x["noholes"]), dtype="<U5")), self.data, Categorize(lambda x: x), numpy.array(numpy.floor(self.noholes), dtype="<U5"), self.noholes)
            self.compare("Categorize noholes with holes", Categorize(lambda x: numpy.array(numpy.floor(x["noholes"]), dtype="<U5")), self.data, Categorize(lambda x: x), numpy.array(numpy.floor(self.noholes), dtype="<U5"), self.withholes)
            self.compare("Categorize holes w/o weights", Categorize(lambda x: numpy.array(numpy.floor(x["withholes"]), dtype="<U5")), self.data, Categorize(lambda x: x), numpy.array(numpy.floor(self.withholes), dtype="<U5"), 1.0)
            self.compare("Categorize holes const weights", Categorize(lambda x: numpy.array(numpy.floor(x["withholes"]), dtype="<U5")), self.data, Categorize(lambda x: x), numpy.array(numpy.floor(self.withholes), dtype="<U5"), 0.5)
            self.compare("Categorize holes positive weights", Categorize(lambda x: numpy.array(numpy.floor(x["withholes"]), dtype="<U5")), self.data, Categorize(lambda x: x), numpy.array(numpy.floor(self.withholes), dtype="<U5"), self.positive)
            self.compare("Categorize holes with weights", Categorize(lambda x: numpy.array(numpy.floor(x["withholes"]), dtype="<U5")), self.data, Categorize(lambda x: x), numpy.array(numpy.floor(self.withholes), dtype="<U5"), self.noholes)
            self.compare("Categorize holes with holes", Categorize(lambda x: numpy.array(numpy.floor(x["withholes"]), dtype="<U5")), self.data, Categorize(lambda x: x), numpy.array(numpy.floor(self.withholes), dtype="<U5"), self.withholes)
            self.compare("Categorize holes with holes2", Categorize(lambda x: numpy.array(numpy.floor(x["withholes"]), dtype="<U5")), self.data, Categorize(lambda x: x), numpy.array(numpy.floor(self.withholes), dtype="<U5"), self.withholes)

    def testCategorizeTrans(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            self.compare("CategorizeTrans no data", Categorize(lambda x: numpy.array(numpy.floor(x["empty"]), dtype="<U5"), Count(lambda x: 0.5*x)), self.data, Categorize(lambda x: x, Count(lambda x: 0.5*x)), numpy.array(numpy.floor(self.empty), dtype="<U5"), 1.0)
            self.compare("CategorizeTrans noholes w/o weights", Categorize(lambda x: numpy.array(numpy.floor(x["noholes"]), dtype="<U5"), Count(lambda x: 0.5*x)), self.data, Categorize(lambda x: x, Count(lambda x: 0.5*x)), numpy.array(numpy.floor(self.noholes), dtype="<U5"), 1.0)
            self.compare("CategorizeTrans noholes const weights", Categorize(lambda x: numpy.array(numpy.floor(x["noholes"]), dtype="<U5"), Count(lambda x: 0.5*x)), self.data, Categorize(lambda x: x, Count(lambda x: 0.5*x)), numpy.array(numpy.floor(self.noholes), dtype="<U5"), 0.5)
            self.compare("CategorizeTrans noholes positive weights", Categorize(lambda x: numpy.array(numpy.floor(x["noholes"]), dtype="<U5"), Count(lambda x: 0.5*x)), self.data, Categorize(lambda x: x, Count(lambda x: 0.5*x)), numpy.array(numpy.floor(self.noholes), dtype="<U5"), self.positive)
            self.compare("CategorizeTrans noholes with weights", Categorize(lambda x: numpy.array(numpy.floor(x["noholes"]), dtype="<U5"), Count(lambda x: 0.5*x)), self.data, Categorize(lambda x: x, Count(lambda x: 0.5*x)), numpy.array(numpy.floor(self.noholes), dtype="<U5"), self.noholes)
            self.compare("CategorizeTrans noholes with holes", Categorize(lambda x: numpy.array(numpy.floor(x["noholes"]), dtype="<U5"), Count(lambda x: 0.5*x)), self.data, Categorize(lambda x: x, Count(lambda x: 0.5*x)), numpy.array(numpy.floor(self.noholes), dtype="<U5"), self.withholes)
            self.compare("CategorizeTrans holes w/o weights", Categorize(lambda x: numpy.array(numpy.floor(x["withholes"]), dtype="<U5"), Count(lambda x: 0.5*x)), self.data, Categorize(lambda x: x, Count(lambda x: 0.5*x)), numpy.array(numpy.floor(self.withholes), dtype="<U5"), 1.0)
            self.compare("CategorizeTrans holes const weights", Categorize(lambda x: numpy.array(numpy.floor(x["withholes"]), dtype="<U5"), Count(lambda x: 0.5*x)), self.data, Categorize(lambda x: x, Count(lambda x: 0.5*x)), numpy.array(numpy.floor(self.withholes), dtype="<U5"), 0.5)
            self.compare("CategorizeTrans holes positive weights", Categorize(lambda x: numpy.array(numpy.floor(x["withholes"]), dtype="<U5"), Count(lambda x: 0.5*x)), self.data, Categorize(lambda x: x, Count(lambda x: 0.5*x)), numpy.array(numpy.floor(self.withholes), dtype="<U5"), self.positive)
            self.compare("CategorizeTrans holes with weights", Categorize(lambda x: numpy.array(numpy.floor(x["withholes"]), dtype="<U5"), Count(lambda x: 0.5*x)), self.data, Categorize(lambda x: x, Count(lambda x: 0.5*x)), numpy.array(numpy.floor(self.withholes), dtype="<U5"), self.noholes)
            self.compare("CategorizeTrans holes with holes", Categorize(lambda x: numpy.array(numpy.floor(x["withholes"]), dtype="<U5"), Count(lambda x: 0.5*x)), self.data, Categorize(lambda x: x, Count(lambda x: 0.5*x)), numpy.array(numpy.floor(self.withholes), dtype="<U5"), self.withholes)
            self.compare("CategorizeTrans holes with holes2", Categorize(lambda x: numpy.array(numpy.floor(x["withholes"]), dtype="<U5"), Count(lambda x: 0.5*x)), self.data, Categorize(lambda x: x, Count(lambda x: 0.5*x)), numpy.array(numpy.floor(self.withholes), dtype="<U5"), self.withholes)

    def testFractionBin(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            self.compare("FractionBin no data", Fraction(lambda x: x["empty"], Bin(100, -3.0, 3.0, lambda x: x["empty"])), self.data, Fraction(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.empty, 1.0)
            self.compare("FractionBin noholes w/o weights", Fraction(lambda x: x["noholes"], Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Fraction(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, 1.0)
            self.compare("FractionBin noholes const weights", Fraction(lambda x: x["noholes"], Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Fraction(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, 0.5)
            self.compare("FractionBin noholes positive weights", Fraction(lambda x: x["noholes"], Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Fraction(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.positive)
            self.compare("FractionBin noholes with weights", Fraction(lambda x: x["noholes"], Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Fraction(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.noholes)
            self.compare("FractionBin noholes with holes", Fraction(lambda x: x["noholes"], Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Fraction(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.withholes)
            self.compare("FractionBin holes w/o weights", Fraction(lambda x: x["withholes"], Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Fraction(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, 1.0)
            self.compare("FractionBin holes const weights", Fraction(lambda x: x["withholes"], Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Fraction(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, 0.5)
            self.compare("FractionBin holes positive weights", Fraction(lambda x: x["withholes"], Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Fraction(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.positive)
            self.compare("FractionBin holes with weights", Fraction(lambda x: x["withholes"], Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Fraction(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.noholes)
            self.compare("FractionBin holes with holes", Fraction(lambda x: x["withholes"], Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Fraction(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.withholes)
            self.compare("FractionBin holes with holes2", Fraction(lambda x: x["withholes"], Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Fraction(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.withholes2)

    def testStackBin(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            cuts = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
            self.compare("StackBin no data", Stack(cuts, lambda x: x["empty"], Bin(100, -3.0, 3.0, lambda x: x["empty"])), self.data, Stack(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.empty, 1.0)
            self.compare("StackBin noholes w/o weights", Stack(cuts, lambda x: x["noholes"], Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Stack(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, 1.0)
            self.compare("StackBin noholes const weights", Stack(cuts, lambda x: x["noholes"], Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Stack(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, 0.5)
            self.compare("StackBin noholes positive weights", Stack(cuts, lambda x: x["noholes"], Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Stack(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.positive)
            self.compare("StackBin noholes with weights", Stack(cuts, lambda x: x["noholes"], Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Stack(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.noholes)
            self.compare("StackBin noholes with holes", Stack(cuts, lambda x: x["noholes"], Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Stack(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.withholes)
            self.compare("StackBin holes w/o weights", Stack(cuts, lambda x: x["withholes"], Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Stack(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, 1.0)
            self.compare("StackBin holes const weights", Stack(cuts, lambda x: x["withholes"], Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Stack(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, 0.5)
            self.compare("StackBin holes positive weights", Stack(cuts, lambda x: x["withholes"], Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Stack(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.positive)
            self.compare("StackBin holes with weights", Stack(cuts, lambda x: x["withholes"], Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Stack(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.noholes)
            self.compare("StackBin holes with holes", Stack(cuts, lambda x: x["withholes"], Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Stack(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.withholes)
            self.compare("StackBin holes with holes2", Stack(cuts, lambda x: x["withholes"], Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Stack(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.withholes2)

    def testIrregularlyBinBin(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            cuts = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
            self.compare("IrregularlyBinBin no data", IrregularlyBin(cuts, lambda x: x["empty"], Bin(100, -3.0, 3.0, lambda x: x["empty"])), self.data, IrregularlyBin(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.empty, 1.0)
            self.compare("IrregularlyBinBin noholes w/o weights", IrregularlyBin(cuts, lambda x: x["noholes"], Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, IrregularlyBin(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, 1.0)
            self.compare("IrregularlyBinBin noholes const weights", IrregularlyBin(cuts, lambda x: x["noholes"], Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, IrregularlyBin(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, 0.5)
            self.compare("IrregularlyBinBin noholes positive weights", IrregularlyBin(cuts, lambda x: x["noholes"], Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, IrregularlyBin(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.positive)
            self.compare("IrregularlyBinBin noholes with weights", IrregularlyBin(cuts, lambda x: x["noholes"], Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, IrregularlyBin(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.noholes)
            self.compare("IrregularlyBinBin noholes with holes", IrregularlyBin(cuts, lambda x: x["noholes"], Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, IrregularlyBin(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.withholes)
            self.compare("IrregularlyBinBin holes w/o weights", IrregularlyBin(cuts, lambda x: x["withholes"], Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, IrregularlyBin(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, 1.0)
            self.compare("IrregularlyBinBin holes const weights", IrregularlyBin(cuts, lambda x: x["withholes"], Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, IrregularlyBin(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, 0.5)
            self.compare("IrregularlyBinBin holes positive weights", IrregularlyBin(cuts, lambda x: x["withholes"], Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, IrregularlyBin(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.positive)
            self.compare("IrregularlyBinBin holes with weights", IrregularlyBin(cuts, lambda x: x["withholes"], Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, IrregularlyBin(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.noholes)
            self.compare("IrregularlyBinBin holes with holes", IrregularlyBin(cuts, lambda x: x["withholes"], Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, IrregularlyBin(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.withholes)
            self.compare("IrregularlyBinBin holes with holes2", IrregularlyBin(cuts, lambda x: x["withholes"], Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, IrregularlyBin(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.withholes2)

    def testSelectBin(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            self.compare("SelectBin no data", Select(lambda x: x["empty"], Bin(100, -3.0, 3.0, lambda x: x["empty"])), self.data, Select(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.empty, 1.0)
            self.compare("SelectBin noholes w/o weights", Select(lambda x: x["noholes"], Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Select(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, 1.0)
            self.compare("SelectBin noholes const weights", Select(lambda x: x["noholes"], Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Select(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, 0.5)
            self.compare("SelectBin noholes positive weights", Select(lambda x: x["noholes"], Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Select(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.positive)
            self.compare("SelectBin noholes with weights", Select(lambda x: x["noholes"], Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Select(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.noholes)
            self.compare("SelectBin noholes with holes", Select(lambda x: x["noholes"], Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Select(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.withholes)
            self.compare("SelectBin holes w/o weights", Select(lambda x: x["withholes"], Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Select(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, 1.0)
            self.compare("SelectBin holes const weights", Select(lambda x: x["withholes"], Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Select(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, 0.5)
            self.compare("SelectBin holes positive weights", Select(lambda x: x["withholes"], Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Select(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.positive)
            self.compare("SelectBin holes with weights", Select(lambda x: x["withholes"], Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Select(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.noholes)
            self.compare("SelectBin holes with holes", Select(lambda x: x["withholes"], Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Select(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.withholes)
            self.compare("SelectBin holes with holes2", Select(lambda x: x["withholes"], Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Select(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.withholes2)

    def testLimitBin(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            self.compare("LimitBin SIZE - 1 no data", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, lambda x: x["empty"])), self.data, Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, lambda x: x)), self.empty, 1.0)
            self.compare("LimitBin SIZE - 1 noholes w/o weights", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, 1.0)
            self.compare("LimitBin SIZE - 1 noholes const weights", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, 0.5)
            self.compare("LimitBin SIZE - 1 noholes positive weights", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.positive)
            self.compare("LimitBin SIZE - 1 noholes with weights", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.noholes)
            self.compare("LimitBin SIZE - 1 noholes with holes", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.withholes)
            self.compare("LimitBin SIZE - 1 holes w/o weights", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, 1.0)
            self.compare("LimitBin SIZE - 1 holes const weights", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, 0.5)
            self.compare("LimitBin SIZE - 1 holes positive weights", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.positive)
            self.compare("LimitBin SIZE - 1 holes with weights", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.noholes)
            self.compare("LimitBin SIZE - 1 holes with holes", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.withholes)
            self.compare("LimitBin SIZE - 1 holes with holes2", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.withholes2)
            self.compare("LimitBin SIZE no data", Limit(self.SIZE, Bin(100, -3.0, 3.0, lambda x: x["empty"])), self.data, Limit(self.SIZE, Bin(100, -3.0, 3.0, lambda x: x)), self.empty, 1.0)
            self.compare("LimitBin SIZE noholes w/o weights", Limit(self.SIZE, Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Limit(self.SIZE, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, 1.0)
            self.compare("LimitBin SIZE noholes const weights", Limit(self.SIZE, Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Limit(self.SIZE, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, 0.5)
            self.compare("LimitBin SIZE noholes positive weights", Limit(self.SIZE, Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Limit(self.SIZE, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.positive)
            self.compare("LimitBin SIZE noholes with weights", Limit(self.SIZE, Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Limit(self.SIZE, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.noholes)
            self.compare("LimitBin SIZE noholes with holes", Limit(self.SIZE, Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Limit(self.SIZE, Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.withholes)
            self.compare("LimitBin SIZE holes w/o weights", Limit(self.SIZE, Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Limit(self.SIZE, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, 1.0)
            self.compare("LimitBin SIZE holes const weights", Limit(self.SIZE, Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Limit(self.SIZE, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, 0.5)
            self.compare("LimitBin SIZE holes positive weights", Limit(self.SIZE, Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Limit(self.SIZE, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.positive)
            self.compare("LimitBin SIZE holes with weights", Limit(self.SIZE, Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Limit(self.SIZE, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.noholes)
            self.compare("LimitBin SIZE holes with holes", Limit(self.SIZE, Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Limit(self.SIZE, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.withholes)
            self.compare("LimitBin SIZE holes with holes2", Limit(self.SIZE, Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Limit(self.SIZE, Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.withholes2)

    def testLabelBin(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            self.compare("LabelBin no data", Label(x=Bin(100, -3.0, 3.0, lambda x: x["empty"])), self.data, Label(x=Bin(100, -3.0, 3.0, lambda x: x)), self.empty, 1.0)
            self.compare("LabelBin noholes w/o weights", Label(x=Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Label(x=Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, 1.0)
            self.compare("LabelBin noholes const weights", Label(x=Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Label(x=Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, 0.5)
            self.compare("LabelBin noholes positive weights", Label(x=Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Label(x=Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.positive)
            self.compare("LabelBin noholes with weights", Label(x=Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Label(x=Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.noholes)
            self.compare("LabelBin noholes with holes", Label(x=Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Label(x=Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.withholes)
            self.compare("LabelBin holes w/o weights", Label(x=Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Label(x=Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, 1.0)
            self.compare("LabelBin holes const weights", Label(x=Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Label(x=Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, 0.5)
            self.compare("LabelBin holes positive weights", Label(x=Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Label(x=Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.positive)
            self.compare("LabelBin holes with weights", Label(x=Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Label(x=Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.noholes)
            self.compare("LabelBin holes with holes", Label(x=Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Label(x=Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.withholes)
            self.compare("LabelBin holes with holes2", Label(x=Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Label(x=Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.withholes2)

    def testUntypedLabelBin(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            self.compare("UntypedLabelBin no data", UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x["empty"])), self.data, UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x)), self.empty, 1.0)
            self.compare("UntypedLabelBin noholes w/o weights", UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, 1.0)
            self.compare("UntypedLabelBin noholes const weights", UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, 0.5)
            self.compare("UntypedLabelBin noholes positive weights", UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.positive)
            self.compare("UntypedLabelBin noholes with weights", UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.noholes)
            self.compare("UntypedLabelBin noholes with holes", UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.withholes)
            self.compare("UntypedLabelBin holes w/o weights", UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, 1.0)
            self.compare("UntypedLabelBin holes const weights", UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, 0.5)
            self.compare("UntypedLabelBin holes positive weights", UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.positive)
            self.compare("UntypedLabelBin holes with weights", UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.noholes)
            self.compare("UntypedLabelBin holes with holes", UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.withholes)
            self.compare("UntypedLabelBin holes with holes2", UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.withholes2)

    def testIndexBin(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            self.compare("IndexBin no data", Index(Bin(100, -3.0, 3.0, lambda x: x["empty"])), self.data, Index(Bin(100, -3.0, 3.0, lambda x: x)), self.empty, 1.0)
            self.compare("IndexBin noholes w/o weights", Index(Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Index(Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, 1.0)
            self.compare("IndexBin noholes const weights", Index(Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Index(Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, 0.5)
            self.compare("IndexBin noholes positive weights", Index(Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Index(Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.positive)
            self.compare("IndexBin noholes with weights", Index(Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Index(Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.noholes)
            self.compare("IndexBin noholes with holes", Index(Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Index(Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.withholes)
            self.compare("IndexBin holes w/o weights", Index(Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Index(Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, 1.0)
            self.compare("IndexBin holes const weights", Index(Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Index(Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, 0.5)
            self.compare("IndexBin holes positive weights", Index(Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Index(Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.positive)
            self.compare("IndexBin holes with weights", Index(Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Index(Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.noholes)
            self.compare("IndexBin holes with holes", Index(Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Index(Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.withholes)
            self.compare("IndexBin holes with holes2", Index(Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Index(Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.withholes2)

    def testBranchBin(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            self.compare("BranchBin no data", Branch(Bin(100, -3.0, 3.0, lambda x: x["empty"])), self.data, Branch(Bin(100, -3.0, 3.0, lambda x: x)), self.empty, 1.0)
            self.compare("BranchBin noholes w/o weights", Branch(Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Branch(Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, 1.0)
            self.compare("BranchBin noholes const weights", Branch(Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Branch(Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, 0.5)
            self.compare("BranchBin noholes positive weights", Branch(Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Branch(Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.positive)
            self.compare("BranchBin noholes with weights", Branch(Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Branch(Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.noholes)
            self.compare("BranchBin noholes with holes", Branch(Bin(100, -3.0, 3.0, lambda x: x["noholes"])), self.data, Branch(Bin(100, -3.0, 3.0, lambda x: x)), self.noholes, self.withholes)
            self.compare("BranchBin holes w/o weights", Branch(Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Branch(Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, 1.0)
            self.compare("BranchBin holes const weights", Branch(Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Branch(Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, 0.5)
            self.compare("BranchBin holes positive weights", Branch(Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Branch(Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.positive)
            self.compare("BranchBin holes with weights", Branch(Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Branch(Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.noholes)
            self.compare("BranchBin holes with holes", Branch(Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Branch(Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.withholes)
            self.compare("BranchBin holes with holes2", Branch(Bin(100, -3.0, 3.0, lambda x: x["withholes"])), self.data, Branch(Bin(100, -3.0, 3.0, lambda x: x)), self.withholes, self.withholes2)

    def testBag(self):
        with Numpy() as numpy:
            if numpy is None: return
            sys.stderr.write("\n")
            self.compare("Bag no data", Bag(lambda x: x["empty"]), self.data, Bag(lambda x: x), self.empty, 1.0)
            self.compare("Bag noholes w/o weights", Bag(lambda x: x["noholes"]), self.data, Bag(lambda x: x), self.noholes, 1.0)
            self.compare("Bag noholes const weights", Bag(lambda x: x["noholes"]), self.data, Bag(lambda x: x), self.noholes, 0.5)
            self.compare("Bag noholes positive weights", Bag(lambda x: x["noholes"]), self.data, Bag(lambda x: x), self.noholes, self.positive)
            self.compare("Bag noholes with weights", Bag(lambda x: x["noholes"]), self.data, Bag(lambda x: x), self.noholes, self.noholes)
            self.compare("Bag noholes with holes", Bag(lambda x: x["noholes"]), self.data, Bag(lambda x: x), self.noholes, self.withholes)
            self.compare("Bag holes w/o weights", Bag(lambda x: x["withholes"]), self.data, Bag(lambda x: x), self.withholes, 1.0)
            self.compare("Bag holes const weights", Bag(lambda x: x["withholes"]), self.data, Bag(lambda x: x), self.withholes, 0.5)
            self.compare("Bag holes positive weights", Bag(lambda x: x["withholes"]), self.data, Bag(lambda x: x), self.withholes, self.positive)
            self.compare("Bag holes with weights", Bag(lambda x: x["withholes"]), self.data, Bag(lambda x: x), self.withholes, self.noholes)
            self.compare("Bag holes with holes", Bag(lambda x: x["withholes"]), self.data, Bag(lambda x: x), self.withholes, self.withholes)
            self.compare("Bag holes with holes2", Bag(lambda x: x["withholes"]), self.data, Bag(lambda x: x), self.withholes, self.withholes2)
