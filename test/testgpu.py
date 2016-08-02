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

class TestGPU(unittest.TestCase):
    def runTest(self):
        pass

    def testAnything(self):
        try:
            import pycuda.autoinit
            import pycuda.driver as drv
            import numpy
            from pycuda.compiler import SourceModule
        except ImportError:
            pass
        else:
            module = SourceModule("""
typedef struct {
  float values[100];
  float overflow;
  float underflow;
  float nanflow;
} Histogram;

__global__ void doThings(Histogram* histogram) {

}
""")

