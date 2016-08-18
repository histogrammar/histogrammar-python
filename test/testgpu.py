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
    nvcc = spawn.find_executable("nvcc")

    def runStandalone(self, code, expected):
        if self.nvcc is not None:
            open("compileme.cu", "w").write(code)
            compilation = subprocess.Popen([self.nvcc, "-o", "runme", "compileme.cu"])
            if compilation.wait() == 0:
                execution = subprocess.Popen(["./runme"], stdout=subprocess.PIPE)
                if execution.wait() == 0:
                    result = Factory.fromJson(execution.stdout.read())
                    self.assertEqual(result, expected)

    def runTest(self):
        pass

    def testCount(self):
        self.runStandalone(Count().cuda(commentMain=False, testData=range(10)), Count.ed(10.0))
        self.runStandalone(Count("2*weight").cuda(commentMain=False, testData=range(10)), Count.ed(20.0))

    def testSum(self):
        self.runStandalone(Sum("x").cuda(commentMain=False, testData=range(10)), Sum.ed(10, 45.0))

    def testAverage(self):
        self.runStandalone(Average("x").cuda(commentMain=False, testData=range(10)), Average.ed(10, 4.5))

    def testDeviate(self):
        self.runStandalone(Deviate("x").cuda(commentMain=False, testData=range(10)), Deviate.ed(10, 4.5, 8.25))

    def testMinimize(self):
        self.runStandalone(Minimize("x").cuda(commentMain=False, testData=range(10)), Minimize.ed(10, 0.0))

    def testMaximize(self):
        self.runStandalone(Maximize("x").cuda(commentMain=False, testData=range(10)), Maximize.ed(10, 9.0))
