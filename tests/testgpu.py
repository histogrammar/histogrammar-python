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
import subprocess
import sys
import unittest
from distutils import spawn

from histogrammar.defs import Factory
from histogrammar.primitives.average import Average
from histogrammar.primitives.bin import Bin
from histogrammar.primitives.centrallybin import CentrallyBin
from histogrammar.primitives.collection import Branch, Index, Label, UntypedLabel
from histogrammar.primitives.count import Count
from histogrammar.primitives.deviate import Deviate
from histogrammar.primitives.fraction import Fraction
from histogrammar.primitives.irregularlybin import IrregularlyBin
from histogrammar.primitives.minmax import Minimize, Maximize
from histogrammar.primitives.select import Select
from histogrammar.primitives.stack import Stack
from histogrammar.primitives.sum import Sum
from histogrammar.util import xrange, named
from histogrammar import util
from tests.testnumpy import makeSamples

tolerance = 1e-5
util.relativeTolerance = tolerance
util.absoluteTolerance = tolerance


class TestGPU(unittest.TestCase):
    try:
        import numpy as np
        import pycuda.autoinit
        import pycuda.driver
        import pycuda.compiler
        import pycuda.gpuarray
        numpy = np
    except ImportError:
        numpy = None

    nvcc = spawn.find_executable("nvcc")

    def runTest(self):
        self.testCount()
        self.testSum()
        self.testAverage()
        self.testDeviate()
        self.testMinimize()
        self.testMaximize()
        self.testBin()
        self.testCentrallyBin()
        self.testIrregularlyBin()
        self.testStack()
        self.testFraction()
        self.testSelect()
        self.testLabel()
        self.testUntypedLabel()
        self.testIndex()
        self.testBranch()

    SIZE = 10000
    HOLES = 100
    data = makeSamples(SIZE, HOLES)
    empty = data["empty"]
    positive = data["positive"]
    boolean = data["boolean"]
    noholes = data["noholes"]
    withholes = data["withholes"]
    withholes2 = data["withholes2"]

    def runStandalone(self, aggregator, expected):
        duplicate = aggregator.copy()
        for x in xrange(10):
            duplicate.fill(x)
        code = aggregator.cuda(commentMain=False, testData=range(10))
        if self.nvcc is not None:
            open("compileme.cu", "w").write(code)
            compilation = subprocess.Popen([self.nvcc, "-o", "runme", "compileme.cu"])
            if compilation.wait() == 0:
                execution = subprocess.Popen(["./runme"], stdout=subprocess.PIPE)
                if execution.wait() == 0:
                    result = execution.stdout.read()
                    self.assertEqual(Factory.fromJson(result), Factory.fromJson(expected))
                    self.assertEqual(Factory.fromJson(result), duplicate.toImmutable())

    def runNumpy(self, aggregator, expected):
        duplicate = aggregator.copy()
        for x in xrange(10):
            duplicate.fill(x)
        if self.numpy is not None:
            aggregator.fill.pycuda(x=self.numpy.array(range(10)))
            self.assertEqual(aggregator.toImmutable(), Factory.fromJson(expected))
            self.assertEqual(aggregator.toImmutable(), duplicate.toImmutable())
            self.assertEqual(aggregator, duplicate)

    def twosigfigs(self, number):
        return round(number, 1 - int(math.floor(math.log10(number))))

    def compare(self, name, hgpu, hpy, pydata, debug=False):
        sys.stderr.write(name + "\n")

        hgpu.fill.pycuda(empty=self.empty, positive=self.positive, boolean=self.boolean,
                         noholes=self.noholes, withholes=self.withholes)
        for x in pydata:
            hpy.fill(x)

        if Factory.fromJson(hgpu.toJson()) != Factory.fromJson(hpy.toJson()):
            sys.stderr.write("FAILED ON FIRST FILL\n")
            left = json.dumps(hgpu.toJson(), sort_keys=True, indent=2)
            right = json.dumps(hpy.toJson(), sort_keys=True, indent=2)
            for leftline, rightline in zip(left.split("\n"), right.split("\n")):
                if leftline != rightline:
                    sys.stderr.write("{0:50s} > {1}\n".format(leftline, rightline))
                else:
                    sys.stderr.write("{0:50s} | {1}\n".format(leftline, rightline))
            self.assertEqual(Factory.fromJson(hgpu.toJson()), Factory.fromJson(hpy.toJson()))

        hgpu.fill.pycuda(empty=self.empty, positive=self.positive, boolean=self.boolean,
                         noholes=self.noholes, withholes=self.withholes)
        for x in pydata:
            hpy.fill(x)

        if Factory.fromJson(hgpu.toJson()) != Factory.fromJson(hpy.toJson()):
            sys.stderr.write("FAILED ON SECOND FILL\n")
            left = json.dumps(hgpu.toJson(), sort_keys=True, indent=2)
            right = json.dumps(hpy.toJson(), sort_keys=True, indent=2)
            for leftline, rightline in zip(left.split("\n"), right.split("\n")):
                if leftline != rightline:
                    sys.stderr.write("{0:50s} > {1}\n".format(leftline, rightline))
                else:
                    sys.stderr.write("{0:50s} | {1}\n".format(leftline, rightline))
            self.assertEqual(Factory.fromJson(hgpu.toJson()), Factory.fromJson(hpy.toJson()))

    def testCount(self):
        self.runStandalone(Count(), {"version": "0.9", "type": "Count", "data": 10.0})
        self.runStandalone(Count("2*weight"), {"version": "0.9", "type": "Count", "data": 20.0})

    def testSum(self):
        self.runStandalone(Sum("x"), {"version": "0.9", "type": "Sum", "data": {
                           "entries": 10.0, "sum": 45.0, "name": "x"}})
        self.runNumpy(Sum("x"), {"version": "0.9", "type": "Sum", "data": {"entries": 10.0, "sum": 45.0, "name": "x"}})
        if self.numpy is not None:
            sys.stderr.write("\n")
            self.compare("Sum noholes", Sum("noholes"), Sum(named("noholes", lambda x: x)), self.noholes)
            self.compare("Sum holes", Sum("withholes"), Sum(named("withholes", lambda x: x)), self.withholes)

    def testAverage(self):
        self.runStandalone(Average("x"), {"version": "0.9", "type": "Average",
                                          "data": {"entries": 10.0, "mean": 4.5, "name": "x"}})
        self.runNumpy(Average("x"), {"version": "0.9", "type": "Average",
                                     "data": {"entries": 10.0, "mean": 4.5, "name": "x"}})
        if self.numpy is not None:
            sys.stderr.write("\n")
            self.compare("Average noholes", Average("noholes"), Average(named("noholes", lambda x: x)), self.noholes)
            self.compare("Average holes", Average("withholes"), Average(
                named("withholes", lambda x: x)), self.withholes)

    def testDeviate(self):
        self.runStandalone(Deviate("x"), {"version": "0.9", "type": "Deviate", "data": {
                           "entries": 10.0, "mean": 4.5, "variance": 8.25, "name": "x"}})
        self.runNumpy(Deviate("x"), {"version": "0.9", "type": "Deviate", "data": {
                      "entries": 10.0, "mean": 4.5, "variance": 8.25, "name": "x"}})
        if self.numpy is not None:
            sys.stderr.write("\n")
            self.compare("Deviate noholes", Deviate("noholes"), Deviate(named("noholes", lambda x: x)), self.noholes)
            self.compare("Deviate holes", Deviate("withholes"), Deviate(
                named("withholes", lambda x: x)), self.withholes)

    def testMinimize(self):
        self.runStandalone(Minimize("x"), {"version": "0.9", "type": "Minimize",
                                           "data": {"entries": 10.0, "min": 0.0, "name": "x"}})
        self.runNumpy(Minimize("x"), {"version": "0.9", "type": "Minimize",
                                      "data": {"entries": 10.0, "min": 0.0, "name": "x"}})
        if self.numpy is not None:
            sys.stderr.write("\n")
            self.compare("Minimize noholes", Minimize("noholes"), Minimize(named("noholes", lambda x: x)), self.noholes)
            self.compare("Minimize holes", Minimize("withholes"), Minimize(
                named("withholes", lambda x: x)), self.withholes)

    def testMaximize(self):
        self.runStandalone(Maximize("x"), {"version": "0.9", "type": "Maximize",
                                           "data": {"entries": 10.0, "max": 9.0, "name": "x"}})
        self.runNumpy(Maximize("x"), {"version": "0.9", "type": "Maximize",
                                      "data": {"entries": 10.0, "max": 9.0, "name": "x"}})
        if self.numpy is not None:
            sys.stderr.write("\n")
            self.compare("Maximize noholes", Maximize("noholes"), Maximize(named("noholes", lambda x: x)), self.noholes)
            self.compare("Maximize holes", Maximize("withholes"), Maximize(
                named("withholes", lambda x: x)), self.withholes)

    def testBin(self):
        self.runStandalone(Bin(10, -10, 10, "x"), {"version": "0.9", "type": "Bin", "data": {"low": -10, "high": 10, "entries": 10, "nanflow:type": "Count", "nanflow": 0,
                                                                                             "underflow:type": "Count", "underflow": 0, "overflow:type": "Count", "overflow": 0, "values:type": "Count", "values": [0, 0, 0, 0, 0, 2, 2, 2, 2, 2], "name": "x"}})
        self.runNumpy(Bin(10, -10, 10, "x"), {"version": "0.9", "type": "Bin", "data": {"low": -10, "high": 10, "entries": 10, "nanflow:type": "Count", "nanflow": 0,
                                                                                        "underflow:type": "Count", "underflow": 0, "overflow:type": "Count", "overflow": 0, "values:type": "Count", "values": [0, 0, 0, 0, 0, 2, 2, 2, 2, 2], "name": "x"}})
        if self.numpy is not None:
            sys.stderr.write("\n")
            for bins in [10, 100]:
                self.compare("Bin ({0} bins) noholes".format(bins), Bin(bins, -3.0, 3.0, "noholes"),
                             Bin(bins, -3.0, 3.0, named("noholes", lambda x: x)), self.noholes)
                self.compare("Bin ({0} bins) holes".format(bins), Bin(bins, -3.0, 3.0, "withholes"),
                             Bin(bins, -3.0, 3.0, named("withholes", lambda x: x)), self.withholes)
            sys.stderr.write("\n")
            for bins in [10, 100]:
                self.compare("BinTrans ({0} bins) noholes".format(bins), Bin(bins, -3.0, 3.0, "noholes", Count("0.5*weight")), Bin(
                    bins, -3.0, 3.0, named("noholes", lambda x: x), Count("0.5*weight")), self.noholes)
                self.compare("BinTrans ({0} bins) holes".format(bins), Bin(bins, -3.0, 3.0, "withholes", Count("0.5*weight")), Bin(
                    bins, -3.0, 3.0, named("withholes", lambda x: x), Count("0.5*weight")), self.withholes)
            sys.stderr.write("\n")
            for bins in [10, 100]:
                self.compare("BinAverage ({0} bins) noholes".format(bins), Bin(bins, -3.0, 3.0, "noholes", Average("noholes")), Bin(
                    bins, -3.0, 3.0, named("noholes", lambda x: x), Average(named("noholes", lambda x: x))), self.noholes)
                self.compare("BinAverage ({0} bins) holes".format(bins), Bin(bins, -3.0, 3.0, "withholes", Average("withholes")), Bin(
                    bins, -3.0, 3.0, named("withholes", lambda x: x), Average(named("withholes", lambda x: x))), self.withholes)
            sys.stderr.write("\n")
            for bins in [10, 100]:
                self.compare("BinDeviate ({0} bins) noholes".format(bins), Bin(bins, -3.0, 3.0, "noholes", Deviate("noholes")), Bin(
                    bins, -3.0, 3.0, named("noholes", lambda x: x), Deviate(named("noholes", lambda x: x))), self.noholes)
                self.compare("BinDeviate ({0} bins) holes".format(bins), Bin(bins, -3.0, 3.0, "withholes", Deviate("withholes")), Bin(
                    bins, -3.0, 3.0, named("withholes", lambda x: x), Deviate(named("withholes", lambda x: x))), self.withholes)

    def testCentrallyBin(self):
        self.runStandalone(CentrallyBin([1, 2, 3, 4, 5], "x"), {"version": "0.9", "type": "CentrallyBin", "data": {"entries": 10, "nanflow:type": "Count", "nanflow": 0, "bins:type": "Count", "bins": [
                           {"center": 1, "data": 2}, {"center": 2, "data": 1}, {"center": 3, "data": 1}, {"center": 4, "data": 1}, {"center": 5, "data": 5}], "name": "x"}})
        self.runNumpy(CentrallyBin([1, 2, 3, 4, 5], "x"), {"version": "0.9", "type": "CentrallyBin", "data": {"entries": 10, "nanflow:type": "Count", "nanflow": 0, "bins:type": "Count", "bins": [
                      {"center": 1, "data": 2}, {"center": 2, "data": 1}, {"center": 3, "data": 1}, {"center": 4, "data": 1}, {"center": 5, "data": 5}], "name": "x"}})
        if self.numpy is not None:
            sys.stderr.write("\n")
            centers = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
            self.compare("CentrallyBin noholes", CentrallyBin(centers, "noholes"),
                         CentrallyBin(centers, named("noholes", lambda x: x)), self.noholes)
            self.compare("CentrallyBin holes", CentrallyBin(centers, "withholes"),
                         CentrallyBin(centers, named("withholes", lambda x: x)), self.withholes)
            sys.stderr.write("\n")
            centers = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
            self.compare("CentrallyBinTrans noholes", CentrallyBin(centers, "noholes", Count("0.5*weight")),
                         CentrallyBin(centers, named("noholes", lambda x: x), Count("0.5*weight")), self.noholes)
            self.compare("CentrallyBinTrans holes", CentrallyBin(centers, "withholes", Count("0.5*weight")),
                         CentrallyBin(centers, named("withholes", lambda x: x), Count("0.5*weight")), self.withholes)
            sys.stderr.write("\n")
            centers = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
            self.compare("CentrallyBinAverage noholes", CentrallyBin(centers, "noholes", Average("noholes")), CentrallyBin(
                centers, named("noholes", lambda x: x), Average(named("noholes", lambda x: x))), self.noholes)
            self.compare("CentrallyBinAverage holes", CentrallyBin(centers, "withholes", Average("withholes")), CentrallyBin(
                centers, named("withholes", lambda x: x), Average(named("withholes", lambda x: x))), self.withholes)
            sys.stderr.write("\n")
            centers = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
            self.compare("CentrallyBinDeviate noholes", CentrallyBin(centers, "noholes", Deviate("noholes")), CentrallyBin(
                centers, named("noholes", lambda x: x), Deviate(named("noholes", lambda x: x))), self.noholes)
            self.compare("CentrallyBinDeviate holes", CentrallyBin(centers, "withholes", Deviate("withholes")), CentrallyBin(
                centers, named("withholes", lambda x: x), Deviate(named("withholes", lambda x: x))), self.withholes)

    def testIrregularlyBin(self):
        self.runStandalone(IrregularlyBin([1.5, 2.5, 3.5, 4.5, 5.5], "x"), {"data": {"nanflow:type": "Count", "name": "x", "nanflow": 0.0, "bins:type": "Count", "entries": 10.0, "bins": [{"data": 2.0, "atleast": "-inf"}, {
                           "data": 1.0, "atleast": 1.5}, {"data": 1.0, "atleast": 2.5}, {"data": 1.0, "atleast": 3.5}, {"data": 1.0, "atleast": 4.5}, {"data": 4.0, "atleast": 5.5}]}, "version": "0.9", "type": "IrregularlyBin"})
        self.runNumpy(IrregularlyBin([1.5, 2.5, 3.5, 4.5, 5.5], "x"), {"data": {"nanflow:type": "Count", "name": "x", "nanflow": 0.0, "bins:type": "Count", "entries": 10.0, "bins": [{"data": 2.0, "atleast": "-inf"}, {
                      "data": 1.0, "atleast": 1.5}, {"data": 1.0, "atleast": 2.5}, {"data": 1.0, "atleast": 3.5}, {"data": 1.0, "atleast": 4.5}, {"data": 4.0, "atleast": 5.5}]}, "version": "0.9", "type": "IrregularlyBin"})
        if self.numpy is not None:
            sys.stderr.write("\n")
            cuts = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
            self.compare("IrregularlyBinBin noholes", IrregularlyBin(cuts, "noholes", Bin(100, -3.0, 3.0, "noholes")),
                         IrregularlyBin(cuts, named("noholes", lambda x: x), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("IrregularlyBinBin holes", IrregularlyBin(cuts, "withholes", Bin(100, -3.0, 3.0, "withholes")), IrregularlyBin(
                cuts, named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)

    def testStack(self):
        self.runStandalone(Stack([1.5, 2.5, 3.5, 4.5, 5.5], "x"), {"data": {"nanflow:type": "Count", "name": "x", "nanflow": 0.0, "bins:type": "Count", "entries": 10.0, "bins": [{"data": 10.0, "atleast": "-inf"}, {
                           "data": 8.0, "atleast": 1.5}, {"data": 7.0, "atleast": 2.5}, {"data": 6.0, "atleast": 3.5}, {"data": 5.0, "atleast": 4.5}, {"data": 4.0, "atleast": 5.5}]}, "version": "0.9", "type": "Stack"})
        self.runNumpy(Stack([1.5, 2.5, 3.5, 4.5, 5.5], "x"), {"data": {"nanflow:type": "Count", "name": "x", "nanflow": 0.0, "bins:type": "Count", "entries": 10.0, "bins": [{"data": 10.0, "atleast": "-inf"}, {
                      "data": 8.0, "atleast": 1.5}, {"data": 7.0, "atleast": 2.5}, {"data": 6.0, "atleast": 3.5}, {"data": 5.0, "atleast": 4.5}, {"data": 4.0, "atleast": 5.5}]}, "version": "0.9", "type": "Stack"})
        if self.numpy is not None:
            sys.stderr.write("\n")
            cuts = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
            self.compare("StackBin noholes", Stack(cuts, "noholes", Bin(100, -3.0, 3.0, "noholes")), Stack(cuts,
                                                                                                           named("noholes", lambda x: x), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("StackBin holes", Stack(cuts, "withholes", Bin(100, -3.0, 3.0, "withholes")), Stack(cuts,
                                                                                                             named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)

    def testFraction(self):
        self.runStandalone(Fraction("x > 5"), {"data": {"sub:type": "Count", "denominator": 10.0,
                                                        "numerator": 4.0, "name": "x > 5", "entries": 10.0}, "version": "0.9", "type": "Fraction"})
        self.runNumpy(Fraction("x > 5"), {"data": {"sub:type": "Count", "denominator": 10.0,
                                                   "numerator": 4.0, "name": "x > 5", "entries": 10.0}, "version": "0.9", "type": "Fraction"})
        if self.numpy is not None:
            sys.stderr.write("\n")
            self.compare("FractionBin noholes", Fraction("noholes", Bin(100, -3.0, 3.0, "noholes")),
                         Fraction(named("noholes", lambda x: x), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("FractionBin holes", Fraction("withholes", Bin(100, -3.0, 3.0, "withholes")),
                         Fraction(named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)

    def testSelect(self):
        self.runStandalone(Select("x > 5"), {"data": {"sub:type": "Count", "data": 4.0,
                                                      "name": "x > 5", "entries": 10.0}, "version": "0.9", "type": "Select"})
        self.runNumpy(Select("x > 5"), {"data": {"sub:type": "Count", "data": 4.0,
                                                 "name": "x > 5", "entries": 10.0}, "version": "0.9", "type": "Select"})
        if self.numpy is not None:
            sys.stderr.write("\n")
            self.compare("SelectBin noholes weights", Select("noholes", Bin(100, -3.0, 3.0, "noholes")),
                         Select(named("noholes", lambda x: x), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("SelectBin holes", Select("withholes", Bin(100, -3.0, 3.0, "withholes")),
                         Select(named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)

    def testLabel(self):
        self.runStandalone(Label(one=Sum("x"), two=Sum("2*x"), three=Sum("3*x")), {"data": {"sub:type": "Sum", "data": {"one": {"sum": 45.0, "name": "x", "entries": 10.0}, "three": {
                           "sum": 135.0, "name": "3*x", "entries": 10.0}, "two": {"sum": 90.0, "name": "2*x", "entries": 10.0}}, "entries": 10.0}, "version": "0.9", "type": "Label"})
        self.runNumpy(Label(one=Sum("x"), two=Sum("2*x"), three=Sum("3*x")), {"data": {"sub:type": "Sum", "data": {"one": {"sum": 45.0, "name": "x", "entries": 10.0}, "three": {
                      "sum": 135.0, "name": "3*x", "entries": 10.0}, "two": {"sum": 90.0, "name": "2*x", "entries": 10.0}}, "entries": 10.0}, "version": "0.9", "type": "Label"})
        if self.numpy is not None:
            sys.stderr.write("\n")
            self.compare("LabelBin single noholes", Label(x=Bin(100, -3.0, 3.0, "noholes")),
                         Label(x=Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("LabelBin identical noholes", Label(x=Bin(100, -3.0, 3.0, "noholes"), y=Bin(100, -3.0, 3.0, "noholes")), Label(
                x=Bin(100, -3.0, 3.0, named("noholes", lambda x: x)), y=Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("LabelBin same struct noholes", Label(x=Bin(100, -3.0, 3.0, "noholes"), y=Bin(100, -2.0, 2.0, "noholes")), Label(
                x=Bin(100, -3.0, 3.0, named("noholes", lambda x: x)), y=Bin(100, -2.0, 2.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("LabelBin trans noholes", Label(x=Bin(100, -3.0, 3.0, "noholes"), y=Bin(100, -3.0, 3.0, "noholes", Count("0.5*weight"))), Label(
                x=Bin(100, -3.0, 3.0, named("noholes", lambda x: x)), y=Bin(100, -3.0, 3.0, named("noholes", lambda x: x), Count(lambda x: 0.5*x))), self.noholes)
            self.compare("LabelBin different structs noholes", Label(x=Bin(100, -3.0, 3.0, "noholes"), y=Bin(50, -3.0, 3.0, "noholes")),
                         Label(x=Bin(100, -3.0, 3.0, named("noholes", lambda x: x)), y=Bin(50, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("LabelBin single holes", Label(x=Bin(100, -3.0, 3.0, "withholes")),
                         Label(x=Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("LabelBin identical holes", Label(x=Bin(100, -3.0, 3.0, "withholes"), y=Bin(100, -3.0, 3.0, "withholes")), Label(
                x=Bin(100, -3.0, 3.0, named("withholes", lambda x: x)), y=Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("LabelBin same struct holes", Label(x=Bin(100, -3.0, 3.0, "withholes"), y=Bin(100, -2.0, 2.0, "withholes")), Label(
                x=Bin(100, -3.0, 3.0, named("withholes", lambda x: x)), y=Bin(100, -2.0, 2.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("LabelBin trans withholes", Label(x=Bin(100, -3.0, 3.0, "withholes"), y=Bin(100, -3.0, 3.0, "withholes", Count("0.5*weight"))), Label(
                x=Bin(100, -3.0, 3.0, named("withholes", lambda x: x)), y=Bin(100, -3.0, 3.0, named("withholes", lambda x: x), Count(lambda x: 0.5*x))), self.withholes)
            self.compare("LabelBin different structs holes", Label(x=Bin(100, -3.0, 3.0, "withholes"), y=Bin(50, -3.0, 3.0, "withholes")), Label(
                x=Bin(100, -3.0, 3.0, named("withholes", lambda x: x)), y=Bin(50, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)

    def testUntypedLabel(self):
        self.runStandalone(UntypedLabel(one=Sum("x"), two=Average("2*x"), three=Deviate("3*x")), {"data": {"data": {"one": {"data": {"sum": 45.0, "name": "x", "entries": 10.0}, "type": "Sum"}, "three": {"data": {
                           "variance": 74.25, "entries": 10.0, "name": "3*x", "mean": 13.5}, "type": "Deviate"}, "two": {"data": {"entries": 10.0, "name": "2*x", "mean": 9.0}, "type": "Average"}}, "entries": 10.0}, "version": "0.9", "type": "UntypedLabel"})
        self.runNumpy(UntypedLabel(one=Sum("x"), two=Average("2*x"), three=Deviate("3*x")), {"data": {"data": {"one": {"data": {"sum": 45.0, "name": "x", "entries": 10.0}, "type": "Sum"}, "three": {"data": {
                      "variance": 74.25, "entries": 10.0, "name": "3*x", "mean": 13.5}, "type": "Deviate"}, "two": {"data": {"entries": 10.0, "name": "2*x", "mean": 9.0}, "type": "Average"}}, "entries": 10.0}, "version": "0.9", "type": "UntypedLabel"})
        if self.numpy is not None:
            sys.stderr.write("\n")
            self.compare("UntypedLabelBin single noholes", UntypedLabel(x=Bin(100, -3.0, 3.0, "noholes")),
                         UntypedLabel(x=Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("UntypedLabelBin identical noholes", UntypedLabel(x=Bin(100, -3.0, 3.0, "noholes"), y=Bin(100, -3.0, 3.0, "noholes")),
                         UntypedLabel(x=Bin(100, -3.0, 3.0, named("noholes", lambda x: x)), y=Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("UntypedLabelBin same struct noholes", UntypedLabel(x=Bin(100, -3.0, 3.0, "noholes"), y=Bin(100, -2.0, 2.0, "noholes")),
                         UntypedLabel(x=Bin(100, -3.0, 3.0, named("noholes", lambda x: x)), y=Bin(100, -2.0, 2.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("UntypedLabelBin trans noholes", UntypedLabel(x=Bin(100, -3.0, 3.0, "noholes"), y=Bin(100, -3.0, 3.0, "noholes", Count("0.5*weight"))),
                         UntypedLabel(x=Bin(100, -3.0, 3.0, named("noholes", lambda x: x)), y=Bin(100, -3.0, 3.0, named("noholes", lambda x: x), Count(lambda x: 0.5*x))), self.noholes)
            self.compare("UntypedLabelBin different structs noholes", UntypedLabel(x=Bin(100, -3.0, 3.0, "noholes"), y=Bin(50, -3.0, 3.0, "noholes")),
                         UntypedLabel(x=Bin(100, -3.0, 3.0, named("noholes", lambda x: x)), y=Bin(50, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("UntypedLabelBin deeply different structs noholes", UntypedLabel(x=Bin(100, -3.0, 3.0, "noholes"), y=Bin(100, -3.0, 3.0, "noholes", Sum("noholes"))),
                         UntypedLabel(x=Bin(100, -3.0, 3.0, named("noholes", lambda x: x)), y=Bin(100, -3.0, 3.0, named("noholes", lambda x: x), Sum(named("noholes", lambda x: x)))), self.noholes)
            self.compare("UntypedLabelBin single holes", UntypedLabel(x=Bin(100, -3.0, 3.0, "withholes")),
                         UntypedLabel(x=Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("UntypedLabelBin identical holes", UntypedLabel(x=Bin(100, -3.0, 3.0, "withholes"), y=Bin(100, -3.0, 3.0, "withholes")),
                         UntypedLabel(x=Bin(100, -3.0, 3.0, named("withholes", lambda x: x)), y=Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("UntypedLabelBin same struct holes", UntypedLabel(x=Bin(100, -3.0, 3.0, "withholes"), y=Bin(100, -2.0, 2.0, "withholes")),
                         UntypedLabel(x=Bin(100, -3.0, 3.0, named("withholes", lambda x: x)), y=Bin(100, -2.0, 2.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("UntypedLabelBin trans withholes", UntypedLabel(x=Bin(100, -3.0, 3.0, "withholes"), y=Bin(100, -3.0, 3.0, "withholes", Count("0.5*weight"))),
                         UntypedLabel(x=Bin(100, -3.0, 3.0, named("withholes", lambda x: x)), y=Bin(100, -3.0, 3.0, named("withholes", lambda x: x), Count(lambda x: 0.5*x))), self.withholes)
            self.compare("UntypedLabelBin different structs holes", UntypedLabel(x=Bin(100, -3.0, 3.0, "withholes"), y=Bin(50, -3.0, 3.0, "withholes")),
                         UntypedLabel(x=Bin(100, -3.0, 3.0, named("withholes", lambda x: x)), y=Bin(50, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("UntypedLabelBin deeply different structs holes", UntypedLabel(x=Bin(100, -3.0, 3.0, "withholes"), y=Bin(100, -3.0, 3.0, "withholes", Sum("withholes"))), UntypedLabel(
                x=Bin(100, -3.0, 3.0, named("withholes", lambda x: x)), y=Bin(100, -3.0, 3.0, named("withholes", lambda x: x), Sum(named("withholes", lambda x: x)))), self.withholes)

    def testIndex(self):
        self.runStandalone(Index(Sum("x"), Sum("2*x"), Sum("3*x")), {"data": {"sub:type": "Sum", "data": [{"sum": 45.0, "name": "x", "entries": 10.0}, {
                           "sum": 90.0, "name": "2*x", "entries": 10.0}, {"sum": 135.0, "name": "3*x", "entries": 10.0}], "entries": 10.0}, "version": "0.9", "type": "Index"})
        self.runNumpy(Index(Sum("x"), Sum("2*x"), Sum("3*x")), {"data": {"sub:type": "Sum", "data": [{"sum": 45.0, "name": "x", "entries": 10.0}, {
                      "sum": 90.0, "name": "2*x", "entries": 10.0}, {"sum": 135.0, "name": "3*x", "entries": 10.0}], "entries": 10.0}, "version": "0.9", "type": "Index"})
        if self.numpy is not None:
            sys.stderr.write("\n")
            self.compare("IndexBin single noholes", Index(Bin(100, -3.0, 3.0, "noholes")),
                         Index(Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("IndexBin identical noholes", Index(Bin(100, -3.0, 3.0, "noholes"), Bin(100, -3.0, 3.0, "noholes")), Index(
                Bin(100, -3.0, 3.0, named("noholes", lambda x: x)), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("IndexBin same struct noholes", Index(Bin(100, -3.0, 3.0, "noholes"), Bin(100, -2.0, 2.0, "noholes")), Index(
                Bin(100, -3.0, 3.0, named("noholes", lambda x: x)), Bin(100, -2.0, 2.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("IndexBin trans noholes", Index(Bin(100, -3.0, 3.0, "noholes"), Bin(100, -3.0, 3.0, "noholes", Count("0.5*weight"))), Index(
                Bin(100, -3.0, 3.0, named("noholes", lambda x: x)), Bin(100, -3.0, 3.0, named("noholes", lambda x: x), Count(lambda x: 0.5*x))), self.noholes)
            self.compare("IndexBin different structs noholes", Index(Bin(100, -3.0, 3.0, "noholes"), Bin(50, -3.0, 3.0, "noholes")),
                         Index(Bin(100, -3.0, 3.0, named("noholes", lambda x: x)), Bin(50, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("IndexBin single holes", Index(Bin(100, -3.0, 3.0, "withholes")),
                         Index(Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("IndexBin identical holes", Index(Bin(100, -3.0, 3.0, "withholes"), Bin(100, -3.0, 3.0, "withholes")), Index(
                Bin(100, -3.0, 3.0, named("withholes", lambda x: x)), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("IndexBin same struct holes", Index(Bin(100, -3.0, 3.0, "withholes"), Bin(100, -2.0, 2.0, "withholes")), Index(
                Bin(100, -3.0, 3.0, named("withholes", lambda x: x)), Bin(100, -2.0, 2.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("IndexBin trans withholes", Index(Bin(100, -3.0, 3.0, "withholes"), Bin(100, -3.0, 3.0, "withholes", Count("0.5*weight"))), Index(
                Bin(100, -3.0, 3.0, named("withholes", lambda x: x)), Bin(100, -3.0, 3.0, named("withholes", lambda x: x), Count(lambda x: 0.5*x))), self.withholes)
            self.compare("IndexBin different structs holes", Index(Bin(100, -3.0, 3.0, "withholes"), Bin(50, -3.0, 3.0, "withholes")), Index(
                Bin(100, -3.0, 3.0, named("withholes", lambda x: x)), Bin(50, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)

    def testBranch(self):
        self.runStandalone(Branch(Sum("x"), Average("2*x"), Deviate("3*x")), {"data": {"data": [{"data": {"sum": 45.0, "name": "x", "entries": 10.0}, "type": "Sum"}, {"data": {
                           "entries": 10.0, "name": "2*x", "mean": 9.0}, "type": "Average"}, {"data": {"variance": 74.25, "entries": 10.0, "name": "3*x", "mean": 13.5}, "type": "Deviate"}], "entries": 10.0}, "version": "0.9", "type": "Branch"})
        self.runNumpy(Branch(Sum("x"), Average("2*x"), Deviate("3*x")), {"data": {"data": [{"data": {"sum": 45.0, "name": "x", "entries": 10.0}, "type": "Sum"}, {"data": {
                      "entries": 10.0, "name": "2*x", "mean": 9.0}, "type": "Average"}, {"data": {"variance": 74.25, "entries": 10.0, "name": "3*x", "mean": 13.5}, "type": "Deviate"}], "entries": 10.0}, "version": "0.9", "type": "Branch"})
        if self.numpy is not None:
            sys.stderr.write("\n")
            self.compare("BranchBin single noholes", Branch(Bin(100, -3.0, 3.0, "noholes")),
                         Branch(Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("BranchBin identical noholes", Branch(Bin(100, -3.0, 3.0, "noholes"), Bin(100, -3.0, 3.0, "noholes")), Branch(
                Bin(100, -3.0, 3.0, named("noholes", lambda x: x)), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("BranchBin same struct noholes", Branch(Bin(100, -3.0, 3.0, "noholes"), Bin(100, -2.0, 2.0, "noholes")), Branch(
                Bin(100, -3.0, 3.0, named("noholes", lambda x: x)), Bin(100, -2.0, 2.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("BranchBin trans noholes", Branch(Bin(100, -3.0, 3.0, "noholes"), Bin(100, -3.0, 3.0, "noholes", Count("0.5*weight"))), Branch(
                Bin(100, -3.0, 3.0, named("noholes", lambda x: x)), Bin(100, -3.0, 3.0, named("noholes", lambda x: x), Count(lambda x: 0.5*x))), self.noholes)
            self.compare("BranchBin different structs noholes", Branch(Bin(100, -3.0, 3.0, "noholes"), Bin(50, -3.0, 3.0, "noholes")),
                         Branch(Bin(100, -3.0, 3.0, named("noholes", lambda x: x)), Bin(50, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("BranchBin deeply different structs noholes", Branch(Bin(100, -3.0, 3.0, "noholes"), Bin(100, -3.0, 3.0, "noholes", Sum("noholes"))), Branch(
                Bin(100, -3.0, 3.0, named("noholes", lambda x: x)), Bin(100, -3.0, 3.0, named("noholes", lambda x: x), Sum(named("noholes", lambda x: x)))), self.noholes)
            self.compare("BranchBin single holes", Branch(Bin(100, -3.0, 3.0, "withholes")),
                         Branch(Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("BranchBin identical holes", Branch(Bin(100, -3.0, 3.0, "withholes"), Bin(100, -3.0, 3.0, "withholes")), Branch(
                Bin(100, -3.0, 3.0, named("withholes", lambda x: x)), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("BranchBin same struct holes", Branch(Bin(100, -3.0, 3.0, "withholes"), Bin(100, -2.0, 2.0, "withholes")), Branch(
                Bin(100, -3.0, 3.0, named("withholes", lambda x: x)), Bin(100, -2.0, 2.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("BranchBin trans withholes", Branch(Bin(100, -3.0, 3.0, "withholes"), Bin(100, -3.0, 3.0, "withholes", Count("0.5*weight"))), Branch(
                Bin(100, -3.0, 3.0, named("withholes", lambda x: x)), Bin(100, -3.0, 3.0, named("withholes", lambda x: x), Count(lambda x: 0.5*x))), self.withholes)
            self.compare("BranchBin different structs holes", Branch(Bin(100, -3.0, 3.0, "withholes"), Bin(50, -3.0, 3.0, "withholes")), Branch(
                Bin(100, -3.0, 3.0, named("withholes", lambda x: x)), Bin(50, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("BranchBin deeply different structs holes", Branch(Bin(100, -3.0, 3.0, "withholes"), Bin(100, -3.0, 3.0, "withholes", Sum("withholes"))), Branch(
                Bin(100, -3.0, 3.0, named("withholes", lambda x: x)), Bin(100, -3.0, 3.0, named("withholes", lambda x: x), Sum(named("withholes", lambda x: x)))), self.withholes)
