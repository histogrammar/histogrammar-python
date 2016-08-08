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

# from histogrammar import *

tolerance = 1e-12
util.relativeTolerance = tolerance
util.absoluteTolerance = tolerance

class TestGPU(unittest.TestCase):
    def runTest(self):
        pass

    def testAnything(self):
        try:
            import pycuda.autoinit
            import pycuda.driver
            import numpy
            from pycuda.compiler import SourceModule
        except ImportError:
            pass
        else:

if True:
            module = SourceModule("""
#include <stdio.h>

typedef struct {
  float test;
} Histogram;

namespace histogrammar {
  __global__ void init(Histogram* histogram) {
    const int index = threadIdx.x;
    histogram[index].test = 5;
  }

  void fillall() {
    Histogram *working;
    Histogram output[100];

    cudaMalloc((void**)&working, 100 * sizeof(Histogram));

    histogrammar::init<<<1, 100>>>(working);

    cudaMemcpy(output, working, 100 * sizeof(Histogram), cudaMemcpyDeviceToHost);

    cudaFree(working);

    for (int i = 0;  i < 100;  i++)
      printf("%g\\n", output[i].test);
  }
}
""")
            output = numpy.empty(100, dtype=numpy.float32)
            doinit = module.get_function("doinit")
            dofill = module.get_function("dofill")
            fill(pycuda.driver.Out(output), block=(100, 1, 1))



            
#             module = SourceModule("""
  # float values[100];
  # float overflow;
  # float underflow;
  # float nanflow;

# __global__ void fill(Histogram* histogram, float* data, float* weights) {

# }
# """)

