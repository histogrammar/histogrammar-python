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

import sys
from setuptools import setup, find_packages

import histogrammar.version

setup(name = "histogrammar",
      version = histogrammar.version.__version__,
      packages = find_packages(),
      scripts = ["scripts/hgwatch"],
      description = "Composable histogram primitives for distributed data reduction.",
      long_description = """Histogrammar is a suite of data aggregation primitives designed for use in parallel processing. In the simplest case, you can use this to compute histograms, but the generality of the primitives allows much more.

See http://histogrammar.org for a complete introduction.

This Python implementation of Histogrammar adheres to version 1.0 of the specification and has been tested to guarantee compatibility with the Scala implementation. The test suite includes empty datasets, NaN/infinity handling, associativity tests, and numerical agreement at the level of one part in a trillion (double precision). Several common histogram types can be plotted in Matplotlib, PyROOT, and Bokeh with a single method call.

If Numpy or Pandas is available, histograms and other aggregators can be filled from arrays ten to a hundred times more quickly via Numpy commands, rather than Python for loops.

If PyROOT is available, histograms and other aggregators can be filled from ROOT TTrees hundreds of times more quickly by JIT-compiling a specialized C++ filler.

Histograms and other aggregators may also be converted into CUDA code for inclusion in a GPU workflow. And if PyCUDA is available, they can also be filled from Numpy arrays by JIT-compiling the CUDA.""",
      author = "Jim Pivarski (DIANA-HEP)",
      author_email = "pivarski@fnal.gov",
      maintainer = "Jim Pivarski (DIANA-HEP)",
      maintainer_email = "pivarski@fnal.gov",
      url = "http://histogrammar.org",
      download_url = "https://github.com/histogrammar/histogrammar-python",
      license = "Apache Software License v2",
      test_suite = "tests",
      install_requires = [],
      tests_require = [],
      classifiers = ["Development Status :: 5 - Production/Stable",
                     "Environment :: Console",
                     "Intended Audience :: Science/Research",
                     "License :: OSI Approved :: Apache Software License",
                     "Topic :: Scientific/Engineering :: Information Analysis",
                     "Topic :: Scientific/Engineering :: Mathematics",
                     "Topic :: Scientific/Engineering :: Physics",
                     ],
      platforms = "Any",
      )
