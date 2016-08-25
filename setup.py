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
from distutils.core import setup
from distutils.cmd import Command

import histogrammar.version

class TestCommand(Command):
    user_options = []
    def initialize_options(self): pass
    def finalize_options(self): pass
    def run(self):
        import sys, subprocess
        if sys.version_info[0] == 2 and sys.version_info[1] == 6:
            args = ["-c", "import unittest; from test.testbasic import *; from test.testnumpy import *; from test.testrootcling import *; from test.testgpu import *; unittest.main()"]
        else:
            args = ["-m", "unittest", "-v", "test.testbasic", "test.testnumpy", "test.testrootcling", "test.testgpu"]
        raise SystemExit(subprocess.call([sys.executable] + args))

setup(name="Histogrammar",
      version=histogrammar.version.__version__,
      scripts=["scripts/hgwatch"],
      description="Composable histogram primitives for distributed data reduction.",
      author="Jim Pivarski (DIANA-HEP)",
      author_email="pivarski@fnal.gov",
      url="https://github.com/diana-hep/histogrammar",
      cmdclass={"test": TestCommand},
      packages=["histogrammar", "histogrammar.primitives", "histogrammar.plot", "histogrammar.pycparser", "histogrammar.pycparser.ply"],
      )
