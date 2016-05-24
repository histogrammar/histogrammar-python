#!/usr/bin/env python

# Copyright 2016 Jim Pivarski
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

import math
import pickle
import unittest

from histogrammar import *
from histogrammar.histogram import Histogram

def ed(x):
    return Factory.fromJson(x.toJson())

class TestEverything(unittest.TestCase):
    class Struct(object):
        def __init__(self, x, y, z, w):
            self.bool = x
            self.int = y
            self.double = z
            self.string = w
        def __repr__(self):
            return "Struct({}, {}, {}, {})".format(self.bool, self.int, self.double, self.string)

    def test_count(self):
        x = Count()
        self.assertEqual(x, x)
        self.assertEqual(ed(x), ed(x))
        self.assertEqual(hash(x), hash(x))
        self.assertEqual(hash(ed(x)), hash(ed(x)))
        self.assertEqual(x, x + x.zero())
        self.assertEqual(ed(x), ed(x) + ed(x).zero())
        self.assertEqual(ed(x + x), ed(x) + ed(x))
        x.fill(3.4)
        self.assertEqual(ed(x + x), ed(x) + ed(x))
        self.assertEqual(x, x)
        self.assertEqual(ed(x), ed(x))
        self.assertEqual(hash(x), hash(x))
        self.assertEqual(hash(ed(x)), hash(ed(x)))
        self.assertEqual(x, x + x.zero())
        self.assertEqual(ed(x), ed(x) + ed(x).zero())
        self.assertEqual(ed(x + x), ed(x) + ed(x))
        x.fill(2.2)
        self.assertEqual(ed(x + x), ed(x) + ed(x))
        self.assertEqual(x, x)
        self.assertEqual(ed(x), ed(x))
        self.assertEqual(hash(x), hash(x))
        self.assertEqual(hash(ed(x)), hash(ed(x)))
        self.assertEqual(x, x + x.zero())
        self.assertEqual(ed(x), ed(x) + ed(x).zero())
        self.assertEqual(ed(x + x), ed(x) + ed(x))
        x.fill(-1.8)
        self.assertEqual(ed(x + x), ed(x) + ed(x))
