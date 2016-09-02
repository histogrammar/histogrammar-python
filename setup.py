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

setup(name = "Histogrammar",
      version = histogrammar.version.__version__,
      packages = find_packages(),
      scripts = ["scripts/hgwatch"],
      description = "Composable histogram primitives for distributed data reduction.",
      author = "Jim Pivarski (DIANA-HEP)",
      author_email = "pivarski@fnal.gov",
      url = "http://histogrammar.org",
      license = "Apache Software License v2",
      test_suite = "test",
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
      )
