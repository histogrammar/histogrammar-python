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
        tfileFlat = ROOT.TFile("test/flat.root")
        ttreeFlat = tfileFlat.Get("simple")
        tfileEvent = ROOT.TFile("test/Event.root")
        ttreeEvent = tfileEvent.Get("T")
    except ImportError:
        pass

    ################################################################ Count

    def testCount(self):
        if TestRootCling.ttreeFlat is not None:
            hg = Count()
            hg.cling(TestRootCling.ttreeFlat, debug=False)
            self.assertEqual(hg.toJson(), {"type": "Count", "data": 10000})
            hg.cling(TestRootCling.ttreeFlat, debug=False)
            self.assertEqual(hg.toJson(), {"type": "Count", "data": 20000})

            hg = Count("0.5 * weight")
            hg.cling(TestRootCling.ttreeFlat, debug=False)
            self.assertEqual(hg.toJson(), {"type": "Count", "data": 5000})
            hg.cling(TestRootCling.ttreeFlat, debug=False)
            self.assertEqual(hg.toJson(), {"type": "Count", "data": 10000})

    #     if TestRootCling.ttreeEvent is not None:
    #         hg = Count()
    #         hg.cling(TestRootCling.ttreeEvent, debug=False)
    #         self.assertEqual(hg.toJson(), {"type": "Count", "data": 1000})
    #         hg.cling(TestRootCling.ttreeEvent, debug=False)
    #         self.assertEqual(hg.toJson(), {"type": "Count", "data": 2000})

    #         hg = Count("0.5 * weight")
    #         hg.cling(TestRootCling.ttreeEvent, debug=False)
    #         self.assertEqual(hg.toJson(), {"type": "Count", "data": 500})
    #         hg.cling(TestRootCling.ttreeEvent, debug=False)
    #         self.assertEqual(hg.toJson(), {"type": "Count", "data": 1000})

    ################################################################ Sum

    def testSum(self):
        if TestRootCling.ttreeFlat is not None:
            hg = Sum("positive")
            hg.cling(TestRootCling.ttreeFlat, debug=False)
            self.assertEqual(hg.toJson(), {"data": {"sum": 7970.933535083706, "name": "positive", "entries": 10000}, "type": "Sum"})
            hg.cling(TestRootCling.ttreeFlat, debug=False)
            self.assertEqual(hg.toJson(), {"data": {"sum": 2*7970.933535083706, "name": "positive", "entries": 20000}, "type": "Sum"})

            hg = Sum("2 * noholes")
            hg.cling(TestRootCling.ttreeFlat, debug=False)
            self.assertEqual(hg.toJson(), {"data": {"sum": 137.62044119255137, "name": "2 * noholes", "entries": 10000}, "type": "Sum"})
            hg.cling(TestRootCling.ttreeFlat, debug=True)
            self.assertEqual(hg.toJson(), {"data": {"sum": 2*137.62044119255137, "name": "2 * noholes", "entries": 20000}, "type": "Sum"})

    #     if TestRootCling.ttreeEvent is not None:
    #         hg = Sum("event.GetNtrack()")
    #         hg.cling(TestRootCling.ttreeEvent, debug=False)
    #         self.assertEqual(hg.toJson(), {"data": {"sum": 599640, "name": "event.GetNtrack()", "entries": 1000}, "type": "Sum"})
    #         hg.cling(TestRootCling.ttreeEvent, debug=False)
    #         self.assertEqual(hg.toJson(), {"data": {"sum": 2*599640, "name": "event.GetNtrack()", "entries": 2*1000}, "type": "Sum"})
