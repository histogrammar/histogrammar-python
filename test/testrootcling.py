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

    ################################################################ Bin

    def testBin(self):
        if TestRootCling.ttreeFlat is not None:
            hg = Bin(10, 0, 1, "positive")
            hg.cling(TestRootCling.ttreeFlat, debug=True)
            self.assertEqual(hg.toJson(), {"data": {
    "nanflow:type": "Count",
    "name": "positive",
    "nanflow": 0.0,
    "overflow:type": "Count",
    "values:type": "Count",
    "high": 1.0,
    "values": [6853.0, 2699.0, 426.0, 21.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "low": 0.0,
    "entries": 10000.0,
    "overflow": 0.0,
    "underflow": 0.0,
    "underflow:type": "Count"
  },
  "type": "Bin"})

            hg = Bin(10, 0, 1, "positive", Sum("noholes"))
            hg.cling(TestRootCling.ttreeFlat, debug=True)
            self.assertEqual(hg.toJson(), {"data": {
    "values:name": "noholes",
    "nanflow:type": "Count",
    "name": "positive",
    "nanflow": 0.0,
    "overflow:type": "Count",
    "values:type": "Sum",
    "high": 1.0,
    "values": [
      {"sum": 66.24699453630666, "entries": 6853.0},
      {"sum": -4.076107526864597, "entries": 2699.0},
      {"sum": 14.08908939411051, "entries": 426.0},
      {"sum": -9.21929781695716, "entries": 21.0},
      {"sum": 1.769542009679859, "entries": 1.0},
      {"sum": 0.0, "entries": 0.0},
      {"sum": 0.0, "entries": 0.0},
      {"sum": 0.0, "entries": 0.0},
      {"sum": 0.0, "entries": 0.0},
      {"sum": 0.0, "entries": 0.0}
    ],
    "low": 0.0,
    "entries": 10000.0,
    "overflow": 0.0,
    "underflow": 0.0,
    "underflow:type": "Count"
  },
  "type": "Bin"})

