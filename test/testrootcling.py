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

import unittest

from histogrammar import *

def ed(x): return Factory.fromJson(x.toJson())

class TestRootCling(unittest.TestCase):
    ttreeFlat = None
    ttreeEvent = None

    try:
        import ROOT
        ROOT.gInterpreter.AddIncludePath("test/Event.h")
        ROOT.gInterpreter.ProcessLine(".L test/Event.cxx")
        tfileEvent = ROOT.TFile("test/Event.root")
        ttreeEvent = tfileEvent.Get("T")
    except ImportError:
        pass

    ################################################################ Count

    def testCount(self):
        if TestRootCling.ttreeEvent is not None:
            hg = Count()
            hg.cling(TestRootCling.ttreeEvent, debug=False)
            self.assertEqual(ed(hg), Count.ed(1000))
            hg.cling(TestRootCling.ttreeEvent, debug=False)
            self.assertEqual(ed(hg), Count.ed(2000))

            hg = Count("0.5 * weight")
            hg.cling(TestRootCling.ttreeEvent, debug=False)
            self.assertEqual(ed(hg), Count.ed(500))
            hg.cling(TestRootCling.ttreeEvent, debug=False)
            self.assertEqual(ed(hg), Count.ed(1000))

    ################################################################ Sum

    def testSum(self):
        if TestRootCling.ttreeEvent is not None:
            hg = Sum("event.GetNtrack()")
            hg.cling(TestRootCling.ttreeEvent, debug=False)
            self.assertEqual(ed(hg), Factory.fromJson({"data": {"sum": 599640.0, "name": "event.GetNtrack()", "entries": 1000.0}, "type": "Sum"}))
            hg.cling(TestRootCling.ttreeEvent, debug=False)
            self.assertEqual(ed(hg), Factory.fromJson({"data": {"sum": 2*599640.0, "name": "event.GetNtrack()", "entries": 2*1000.0}, "type": "Sum"}))
