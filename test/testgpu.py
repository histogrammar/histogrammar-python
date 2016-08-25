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
import subprocess
import sys
import time
import unittest
from distutils import spawn

from histogrammar import *

tolerance = 1e-6
util.relativeTolerance = tolerance
util.absoluteTolerance = tolerance

class TestGPU(unittest.TestCase):
    try:
        import numpy as np
        numpy = np
    except ImportError:
        numpy = None

    nvcc = spawn.find_executable("nvcc")

    def runStandalone(self, code, expected):
        if self.nvcc is not None:
            open("compileme.cu", "w").write(code)
            compilation = subprocess.Popen([self.nvcc, "-o", "runme", "compileme.cu"])
            if compilation.wait() == 0:
                execution = subprocess.Popen(["./runme"], stdout=subprocess.PIPE)
                if execution.wait() == 0:
                    result = execution.stdout.read()
                    self.assertEqual(Factory.fromJson(result), Factory.fromJson(expected))

    def runTest(self):
        pass

    # def testCount(self):
    #     self.runStandalone(Count().cuda(commentMain=False, testData=range(10)), {"version": "0.9", "type": "Count", "data": 10.0})
    #     self.runStandalone(Count("2*weight").cuda(commentMain=False, testData=range(10)), {"version": "0.9", "type": "Count", "data": 20.0})

    # def testSum(self):
    #     self.runStandalone(Sum("x").cuda(commentMain=False, testData=range(10)), {"version": "0.9", "type": "Sum", "data": {"entries": 10.0, "sum": 45.0, "name": "x"}})

    # def testSumNumpy(self):
    #     h = Sum("x")
    #     if self.numpy is not None:
    #         h.pycuda(x = self.numpy.array(range(10)))
    #         self.assertEqual(h.toImmutable(), Factory.fromJson({"version": "0.9", "type": "Sum", "data": {"entries": 10.0, "sum": 45.0, "name": "x"}}))

    # def testAverage(self):
    #     self.runStandalone(Average("x").cuda(commentMain=False, testData=range(10)), {"version": "0.9", "type": "Average", "data": {"entries": 10.0, "mean": 4.5, "name": "x"}})

    # def testAverageNumpy(self):
    #     h = Average("x")
    #     if self.numpy is not None:
    #         h.pycuda(x = self.numpy.array(range(10)))
    #         self.assertEqual(h.toImmutable(), Factory.fromJson({"version": "0.9", "type": "Average", "data": {"entries": 10.0, "mean": 4.5, "name": "x"}}))

    # def testDeviate(self):
    #     self.runStandalone(Deviate("x").cuda(commentMain=False, testData=range(10)), {"version": "0.9", "type": "Deviate", "data": {"entries": 10.0, "mean": 4.5, "variance": 8.25, "name": "x"}})

    # def testDeviateNumpy(self):
    #     h = Deviate("x")
    #     if self.numpy is not None:
    #         h.pycuda(x = self.numpy.array(range(10)))
    #         self.assertEqual(h.toImmutable(), Factory.fromJson({"version": "0.9", "type": "Deviate", "data": {"entries": 10.0, "mean": 4.5, "variance": 8.25, "name": "x"}}))

    # def testMinimize(self):
    #     self.runStandalone(Minimize("x").cuda(commentMain=False, testData=range(10)), {"version": "0.9", "type": "Minimize", "data": {"entries": 10.0, "min": 0.0, "name": "x"}})

    # def testMinimizeNumpy(self):
    #     h = Minimize("x")
    #     if self.numpy is not None:
    #         h.pycuda(x = self.numpy.array(range(10)))
    #         self.assertEqual(h.toImmutable(), Factory.fromJson({"version": "0.9", "type": "Minimize", "data": {"entries": 10.0, "min": 0.0, "name": "x"}}))

    # def testMaximize(self):
    #     self.runStandalone(Maximize("x").cuda(commentMain=False, testData=range(10)), {"version": "0.9", "type": "Maximize", "data": {"entries": 10.0, "max": 9.0, "name": "x"}})

    # def testMaximizeNumpy(self):
    #     h = Maximize("x")
    #     if self.numpy is not None:
    #         h.pycuda(x = self.numpy.array(range(10)))
    #         self.assertEqual(h.toImmutable(), Factory.fromJson({"version": "0.9", "type": "Maximize", "data": {"entries": 10.0, "max": 9.0, "name": "x"}}))

    # def testBin(self):
    #     self.runStandalone(Bin(10, -10, 10, "x").cuda(commentMain=False, testData=range(10)), {"version": "0.9", "type": "Bin", "data": {"low": -10, "high": 10, "entries": 10, "nanflow:type": "Count", "nanflow": 0, "underflow:type": "Count", "underflow": 0, "overflow:type": "Count", "overflow": 0, "values:type": "Count", "values": [0, 0, 0, 0, 0, 2, 2, 2, 2, 2], "name": "x"}})

    # def testBinNumpy(self):
    #     h = Bin(10, -10, 10, "x")
    #     if self.numpy is not None:
    #         h.pycuda(x = self.numpy.array(range(10)))
    #         self.assertEqual(h.toImmutable(), Factory.fromJson({"version": "0.9", "type": "Bin", "data": {"low": -10, "high": 10, "entries": 10, "nanflow:type": "Count", "nanflow": 0, "underflow:type": "Count", "underflow": 0, "overflow:type": "Count", "overflow": 0, "values:type": "Count", "values": [0, 0, 0, 0, 0, 2, 2, 2, 2, 2], "name": "x"}}))

    # def testCentrallyBin(self):
    #     self.runStandalone(CentrallyBin([1, 2, 3, 4, 5], "x").cuda(commentMain=False, testData=range(10)), {"version": "0.9", "type": "CentrallyBin", "data": {"entries": 10, "nanflow:type": "Count", "nanflow": 0, "bins:type": "Count", "bins": [{"center": 1, "data": 2}, {"center": 2, "data": 1}, {"center": 3, "data": 1}, {"center": 4, "data": 1}, {"center": 5, "data": 5}], "name": "x"}})

    # def testCentrallyBinNumpy(self):
    #     h = CentrallyBin([1, 2, 3, 4, 5], "x")
    #     if self.numpy is not None:
    #         h.pycuda(x = self.numpy.array(range(10)))
    #         self.assertEqual(h.toImmutable(), Factory.fromJson({"version": "0.9", "type": "CentrallyBin", "data": {"entries": 10, "nanflow:type": "Count", "nanflow": 0, "bins:type": "Count", "bins": [{"center": 1, "data": 2}, {"center": 2, "data": 1}, {"center": 3, "data": 1}, {"center": 4, "data": 1}, {"center": 5, "data": 5}], "name": "x"}}))

    # def testIrregularlyBin(self):
    #     self.runStandalone(IrregularlyBin([1.5, 2.5, 3.5, 4.5, 5.5], "x").cuda(commentMain=False, testData=range(10)), {"data": {"nanflow:type": "Count", "name": "x", "nanflow": 0.0, "bins:type": "Count", "entries": 10.0, "bins": [{"data": 2.0, "atleast": "-inf"}, {"data": 1.0, "atleast": 1.5}, {"data": 1.0, "atleast": 2.5}, {"data": 1.0, "atleast": 3.5}, {"data": 1.0, "atleast": 4.5}, {"data": 4.0, "atleast": 5.5}]}, "version": "0.9", "type": "IrregularlyBin"})

    # def testIrregularlyBinNumpy(self):
    #     h = IrregularlyBin([1.5, 2.5, 3.5, 4.5, 5.5], "x")
    #     if self.numpy is not None:
    #         h.pycuda(x = self.numpy.array(range(10)))
    #         self.assertEqual(h.toImmutable(), Factory.fromJson({"data": {"nanflow:type": "Count", "name": "x", "nanflow": 0.0, "bins:type": "Count", "entries": 10.0, "bins": [{"data": 2.0, "atleast": "-inf"}, {"data": 1.0, "atleast": 1.5}, {"data": 1.0, "atleast": 2.5}, {"data": 1.0, "atleast": 3.5}, {"data": 1.0, "atleast": 4.5}, {"data": 4.0, "atleast": 5.5}]}, "version": "0.9", "type": "IrregularlyBin"}))

    # def testStack(self):
    #     self.runStandalone(Stack([1.5, 2.5, 3.5, 4.5, 5.5], "x").cuda(commentMain=False, testData=range(10)), {"data": {"nanflow:type": "Count", "name": "x", "nanflow": 0.0, "bins:type": "Count", "entries": 10.0, "bins": [{"data": 10.0, "atleast": "-inf"}, {"data": 8.0, "atleast": 1.5}, {"data": 7.0, "atleast": 2.5}, {"data": 6.0, "atleast": 3.5}, {"data": 5.0, "atleast": 4.5}, {"data": 4.0, "atleast": 5.5}]}, "version": "0.9", "type": "Stack"})

    # def testStackNumpy(self):
    #     h = Stack([1.5, 2.5, 3.5, 4.5, 5.5], "x")
    #     if self.numpy is not None:
    #         h.pycuda(x = self.numpy.array(range(10)))
    #         self.assertEqual(h.toImmutable(), Factory.fromJson({"data": {"nanflow:type": "Count", "name": "x", "nanflow": 0.0, "bins:type": "Count", "entries": 10.0, "bins": [{"data": 10.0, "atleast": "-inf"}, {"data": 8.0, "atleast": 1.5}, {"data": 7.0, "atleast": 2.5}, {"data": 6.0, "atleast": 3.5}, {"data": 5.0, "atleast": 4.5}, {"data": 4.0, "atleast": 5.5}]}, "version": "0.9", "type": "Stack"}))

    # def testFraction(self):
    #     self.runStandalone(Fraction("x > 5").cuda(commentMain=False, testData=range(10)), {"data": {"sub:type": "Count", "denominator": 10.0, "numerator": 4.0, "name": "x > 5", "entries": 10.0}, "version": "0.9", "type": "Fraction"})

    # def testFractionNumpy(self):
    #     h = Fraction("x > 5")
    #     if self.numpy is not None:
    #         h.pycuda(x = self.numpy.array(range(10)))
    #         self.assertEqual(h.toImmutable(), Factory.fromJson({"data": {"sub:type": "Count", "denominator": 10.0, "numerator": 4.0, "name": "x > 5", "entries": 10.0}, "version": "0.9", "type": "Fraction"}))

    # def testSelect(self):
    #     self.runStandalone(Select("x > 5").cuda(commentMain=False, testData=range(10)), {"data": {"sub:type": "Count", "data": 4.0, "name": "x > 5", "entries": 10.0}, "version": "0.9", "type": "Select"})

    # def testSelectNumpy(self):
    #     h = Select("x > 5")
    #     if self.numpy is not None:
    #         h.pycuda(x = self.numpy.array(range(10)))
    #         self.assertEqual(h.toImmutable(), Factory.fromJson({"data": {"sub:type": "Count", "data": 4.0, "name": "x > 5", "entries": 10.0}, "version": "0.9", "type": "Select"}))

    # def testLabel(self):
    #     self.runStandalone(Label(one=Sum("x"), two=Sum("2*x"), three=Sum("3*x")).cuda(commentMain=False, testData=range(10)), {"data": {"sub:type": "Sum", "data": {"one": {"sum": 45.0, "name": "x", "entries": 10.0}, "three": {"sum": 135.0, "name": "3*x", "entries": 10.0}, "two": {"sum": 90.0, "name": "2*x", "entries": 10.0}}, "entries": 10.0}, "version": "0.9", "type": "Label"})

    # def testLabelNumpy(self):
    #     h = Label(one=Sum("x"), two=Sum("2*x"), three=Sum("3*x"))
    #     if self.numpy is not None:
    #         h.pycuda(x = self.numpy.array(range(10)))
    #         self.assertEqual(h.toImmutable(), Factory.fromJson({"data": {"sub:type": "Sum", "data": {"one": {"sum": 45.0, "name": "x", "entries": 10.0}, "three": {"sum": 135.0, "name": "3*x", "entries": 10.0}, "two": {"sum": 90.0, "name": "2*x", "entries": 10.0}}, "entries": 10.0}, "version": "0.9", "type": "Label"}))

    def testUntypedLabel(self):
        self.runStandalone(UntypedLabel(one=Sum("x"), two=Average("2*x"), three=Deviate("3*x")).cuda(commentMain=False, testData=range(10)), {"data": {"data": {"one": {"data": {"sum": 45.0, "name": "x", "entries": 10.0}, "type": "Sum"}, "three": {"data": {"variance": 74.25, "entries": 10.0, "name": "3*x", "mean": 13.5}, "type": "Deviate"}, "two": {"data": {"entries": 10.0, "name": "2*x", "mean": 9.0}, "type": "Average"}}, "entries": 10.0}, "version": "0.9", "type": "UntypedLabel"})

    def testUntypedLabelNumpy(self):
        h = UntypedLabel(one=Sum("x"), two=Average("2*x"), three=Deviate("3*x"))
        if self.numpy is not None:
            h.pycuda(x = self.numpy.array(range(10)))
            self.assertEqual(h.toImmutable(), Factory.fromJson({"data": {"data": {"one": {"data": {"sum": 45.0, "name": "x", "entries": 10.0}, "type": "Sum"}, "three": {"data": {"variance": 74.25, "entries": 10.0, "name": "3*x", "mean": 13.5}, "type": "Deviate"}, "two": {"data": {"entries": 10.0, "name": "2*x", "mean": 9.0}, "type": "Average"}}, "entries": 10.0}, "version": "0.9", "type": "UntypedLabel"}))
