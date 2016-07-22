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
import time
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
        tfileBig = ROOT.TFile("test/big.root")
        ttreeBig = tfileBig.Get("big")
        tfileEvent = ROOT.TFile("test/Event.root")
        ttreeEvent = tfileEvent.Get("T")
    except ImportError:
        pass

    ################################################################ Timing

    def testTiming(self):
        if TestRootCling.ttreeBig is not None:
            TestRootCling.ttreeBig.AddBranchToCache("*", True)
            for row in TestRootCling.ttreeBig:
                pass
            print TestRootCling.ttreeBig.GetCurrentFile().GetBytesRead(), TestRootCling.ttreeBig.GetCurrentFile().GetReadCalls()
            print TestRootCling.ttreeBig.PrintCacheStats()
            for row in TestRootCling.ttreeBig:
                pass
            print TestRootCling.ttreeBig.GetCurrentFile().GetBytesRead(), TestRootCling.ttreeBig.GetCurrentFile().GetReadCalls()
            print TestRootCling.ttreeBig.PrintCacheStats()

            hg = Select("!boolean", Bin(100, -10, 10, "2 * noholes"))

            print
            print "Histogrammar JIT-compilation"

            startTime = time.time()
            hg.cling(TestRootCling.ttreeBig, 0, 1, debug=False)
            print time.time() - startTime, TestRootCling.ttreeBig.GetCurrentFile().GetBytesRead(), TestRootCling.ttreeBig.GetCurrentFile().GetReadCalls()

            print
            print "Histogrammar running"

            startTime = time.time()
            hg.cling(TestRootCling.ttreeBig)
            print time.time() - startTime, TestRootCling.ttreeBig.GetCurrentFile().GetBytesRead(), TestRootCling.ttreeBig.GetCurrentFile().GetReadCalls()

            startTime = time.time()
            hg.cling(TestRootCling.ttreeBig)
            print time.time() - startTime, TestRootCling.ttreeBig.GetCurrentFile().GetBytesRead(), TestRootCling.ttreeBig.GetCurrentFile().GetReadCalls()

            startTime = time.time()
            hg.cling(TestRootCling.ttreeBig)
            print time.time() - startTime, TestRootCling.ttreeBig.GetCurrentFile().GetBytesRead(), TestRootCling.ttreeBig.GetCurrentFile().GetReadCalls()

            controlTestCode = """class ControlTest {
public:
  TH1D histogram;

  ControlTest() : histogram("control1", "", 20, -10, 10) {}

  Int_t input_boolean;
  double input_noholes;
 
  void fillall(TTree* ttree, Long64_t start, Long64_t end) {
    if (start < 0) start = 0;
    if (end < 0) end = ttree->GetEntries();

    ttree->SetBranchAddress("noholes", &input_noholes);
    ttree->SetBranchAddress("boolean", &input_boolean);

    for (;  start < end;  ++start) {
      ttree->GetEntry(start);

      if (!input_boolean)
        histogram.Fill(2 * input_noholes);
    }

    ttree->ResetBranchAddresses();
  }
};"""

            print
            print "ROOT C++ compilation"

            import ROOT
            startTime = time.time()
            ROOT.gInterpreter.Declare(controlTestCode)
            controlTest = ROOT.ControlTest()
            print time.time() - startTime, TestRootCling.ttreeBig.GetCurrentFile().GetBytesRead(), TestRootCling.ttreeBig.GetCurrentFile().GetReadCalls()

            print
            print "ROOT C++ first event"

            startTime = time.time()
            controlTest.fillall(TestRootCling.ttreeBig, 0, 1)
            print time.time() - startTime, TestRootCling.ttreeBig.GetCurrentFile().GetBytesRead(), TestRootCling.ttreeBig.GetCurrentFile().GetReadCalls()

            print
            print "ROOT C++ subsequent"

            startTime = time.time()
            controlTest.fillall(TestRootCling.ttreeBig, -1, -1)
            print time.time() - startTime, TestRootCling.ttreeBig.GetCurrentFile().GetBytesRead(), TestRootCling.ttreeBig.GetCurrentFile().GetReadCalls()

            startTime = time.time()
            controlTest.fillall(TestRootCling.ttreeBig, -1, -1)
            print time.time() - startTime, TestRootCling.ttreeBig.GetCurrentFile().GetBytesRead(), TestRootCling.ttreeBig.GetCurrentFile().GetReadCalls()

            startTime = time.time()
            controlTest.fillall(TestRootCling.ttreeBig, -1, -1)
            print time.time() - startTime, TestRootCling.ttreeBig.GetCurrentFile().GetBytesRead(), TestRootCling.ttreeBig.GetCurrentFile().GetReadCalls()

            print
            print "PyROOT first event"

            histogram = ROOT.TH1D("control2", "", 20, -10, 10)

            startTime = time.time()
            for row in TestRootCling.ttreeBig:
                if not row.boolean:
                    histogram.Fill(2 * row.noholes)
                break
            print time.time() - startTime, TestRootCling.ttreeBig.GetCurrentFile().GetBytesRead(), TestRootCling.ttreeBig.GetCurrentFile().GetReadCalls()

            print
            print "PyROOT subsequent"

            startTime = time.time()
            for row in TestRootCling.ttreeBig:
                if not row.boolean:
                    histogram.Fill(2 * row.noholes)
            print time.time() - startTime, TestRootCling.ttreeBig.GetCurrentFile().GetBytesRead(), TestRootCling.ttreeBig.GetCurrentFile().GetReadCalls()

            startTime = time.time()
            for row in TestRootCling.ttreeBig:
                if not row.boolean:
                    histogram.Fill(2 * row.noholes)
            print time.time() - startTime, TestRootCling.ttreeBig.GetCurrentFile().GetBytesRead(), TestRootCling.ttreeBig.GetCurrentFile().GetReadCalls()

            startTime = time.time()
            for row in TestRootCling.ttreeBig:
                if not row.boolean:
                    histogram.Fill(2 * row.noholes)
            print time.time() - startTime, TestRootCling.ttreeBig.GetCurrentFile().GetBytesRead(), TestRootCling.ttreeBig.GetCurrentFile().GetReadCalls()

            print
            print "TFormula first pass"

            histogram3 = ROOT.TH1D("control3", "", 20, -10, 10)

            startTime = time.time()
            TestRootCling.ttreeBig.Draw("2 * noholes >>+ control3", "!boolean", "goff")
            print time.time() - startTime, TestRootCling.ttreeBig.GetCurrentFile().GetBytesRead(), TestRootCling.ttreeBig.GetCurrentFile().GetReadCalls()

            print
            print "TFormula subsequent"

            startTime = time.time()
            TestRootCling.ttreeBig.Draw("2 * noholes >>+ control3", "!boolean", "goff")
            print time.time() - startTime, TestRootCling.ttreeBig.GetCurrentFile().GetBytesRead(), TestRootCling.ttreeBig.GetCurrentFile().GetReadCalls()

            startTime = time.time()
            TestRootCling.ttreeBig.Draw("2 * noholes >>+ control3", "!boolean", "goff")
            print time.time() - startTime, TestRootCling.ttreeBig.GetCurrentFile().GetBytesRead(), TestRootCling.ttreeBig.GetCurrentFile().GetReadCalls()

            startTime = time.time()
            TestRootCling.ttreeBig.Draw("2 * noholes >>+ control3", "!boolean", "goff")
            print time.time() - startTime, TestRootCling.ttreeBig.GetCurrentFile().GetBytesRead(), TestRootCling.ttreeBig.GetCurrentFile().GetReadCalls()

            import numpy
            table = {"boolean": numpy.empty(TestRootCling.ttreeBig.GetEntries(), dtype=numpy.int32),
                     "noholes": numpy.empty(TestRootCling.ttreeBig.GetEntries(), dtype=numpy.double)}

            for i, row in enumerate(TestRootCling.ttreeBig):
                table["boolean"][i] = row.boolean
                table["noholes"][i] = row.noholes

            hg = Select("logical_not(boolean)", Bin(100, -10, 10, "2 * noholes"))

            print
            print "Numpy first"

            startTime = time.time()
            hg.numpy(table)
            print time.time() - startTime

            print
            print "Numpy subsequent"

            startTime = time.time()
            hg.numpy(table)
            print time.time() - startTime

            startTime = time.time()
            hg.numpy(table)
            print time.time() - startTime

            startTime = time.time()
            hg.numpy(table)
            print time.time() - startTime
                
  #   ################################################################ Count

  #   def testCount(self):
  #       if TestRootCling.ttreeFlat is not None:
  #           hg = Count()
  #           hg.cling(TestRootCling.ttreeFlat, debug=False)
  #           self.assertEqual(hg.toJson(), {"type": "Count", "data": 10000})
  #           hg.cling(TestRootCling.ttreeFlat, debug=False)
  #           self.assertEqual(hg.toJson(), {"type": "Count", "data": 20000})

  #           hg = Count("0.5 * weight")
  #           hg.cling(TestRootCling.ttreeFlat, debug=False)
  #           self.assertEqual(hg.toJson(), {"type": "Count", "data": 5000})
  #           hg.cling(TestRootCling.ttreeFlat, debug=False)
  #           self.assertEqual(hg.toJson(), {"type": "Count", "data": 10000})

  #   #     if TestRootCling.ttreeEvent is not None:
  #   #         hg = Count()
  #   #         hg.cling(TestRootCling.ttreeEvent, debug=False)
  #   #         self.assertEqual(hg.toJson(), {"type": "Count", "data": 1000})
  #   #         hg.cling(TestRootCling.ttreeEvent, debug=False)
  #   #         self.assertEqual(hg.toJson(), {"type": "Count", "data": 2000})

  #   #         hg = Count("0.5 * weight")
  #   #         hg.cling(TestRootCling.ttreeEvent, debug=False)
  #   #         self.assertEqual(hg.toJson(), {"type": "Count", "data": 500})
  #   #         hg.cling(TestRootCling.ttreeEvent, debug=False)
  #   #         self.assertEqual(hg.toJson(), {"type": "Count", "data": 1000})

  #   ################################################################ Sum

  #   def testSum(self):
  #       if TestRootCling.ttreeFlat is not None:
  #           hg = Sum("positive")
  #           hg.cling(TestRootCling.ttreeFlat, debug=False)
  #           self.assertEqual(hg.toJson(), {"data": {"sum": 7970.933535083706, "name": "positive", "entries": 10000}, "type": "Sum"})
  #           hg.cling(TestRootCling.ttreeFlat, debug=False)
  #           self.assertEqual(hg.toJson(), {"data": {"sum": 2*7970.933535083706, "name": "positive", "entries": 20000}, "type": "Sum"})

  #           hg = Sum("2 * noholes")
  #           hg.cling(TestRootCling.ttreeFlat, debug=False)
  #           self.assertEqual(hg.toJson(), {"data": {"sum": 137.62044119255137, "name": "2 * noholes", "entries": 10000}, "type": "Sum"})
  #           hg.cling(TestRootCling.ttreeFlat, debug=True)
  #           self.assertEqual(hg.toJson(), {"data": {"sum": 2*137.62044119255137, "name": "2 * noholes", "entries": 20000}, "type": "Sum"})

  #   #     if TestRootCling.ttreeEvent is not None:
  #   #         hg = Sum("event.GetNtrack()")
  #   #         hg.cling(TestRootCling.ttreeEvent, debug=False)
  #   #         self.assertEqual(hg.toJson(), {"data": {"sum": 599640, "name": "event.GetNtrack()", "entries": 1000}, "type": "Sum"})
  #   #         hg.cling(TestRootCling.ttreeEvent, debug=False)
  #   #         self.assertEqual(hg.toJson(), {"data": {"sum": 2*599640, "name": "event.GetNtrack()", "entries": 2*1000}, "type": "Sum"})

  #   ################################################################ Bin

  #   def testBin(self):
  #       if TestRootCling.ttreeFlat is not None:
  #           hg = Bin(20, -10, 10, "withholes")
  #           hg.cling(TestRootCling.ttreeFlat, debug=False)
  #           self.assertEqual(hg.toJson(), {"data": {
  #   "nanflow:type": "Count", 
  #   "name": "withholes", 
  #   "nanflow": 96.0, 
  #   "overflow:type": "Count", 
  #   "values:type": "Count", 
  #   "high": 10.0, 
  #   "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 201.0, 1346.0, 3385.0, 3182.0, 1358.0, 211.0, 15.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #   "low": -10.0, 
  #   "entries": 10000.0, 
  #   "overflow": 99.0, 
  #   "underflow": 96.0, 
  #   "underflow:type": "Count"
  # }, 
  # "type": "Bin"})
  #           hg.cling(TestRootCling.ttreeFlat, debug=False)
  #           self.assertEqual(hg.toJson(), {"data": {
  #   "nanflow:type": "Count", 
  #   "name": "withholes", 
  #   "nanflow": 2*96.0, 
  #   "overflow:type": "Count", 
  #   "values:type": "Count", 
  #   "high": 10.0, 
  #   "values": [2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*10.0, 2*201.0, 2*1346.0, 2*3385.0, 2*3182.0, 2*1358.0, 2*211.0, 2*15.0, 2*1.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0],
  #   "low": -10.0, 
  #   "entries": 2*10000.0, 
  #   "overflow": 2*99.0, 
  #   "underflow": 2*96.0, 
  #   "underflow:type": "Count"
  # }, 
  # "type": "Bin"})

  #           hg = Bin(20, -10, 10, "2 * withholes", Sum("positive"))
  #           hg.cling(TestRootCling.ttreeFlat, debug=False)
  #           self.assertEqual(hg.toJson(), {"data": {
  #   "values:name": "positive",
  #   "nanflow:type": "Count",
  #   "name": "2 * withholes",
  #   "nanflow": 96.0,
  #   "overflow:type": "Count",
  #   "values:type": "Sum",
  #   "high": 10.0,
  #   "values": [
  #     {"sum": 0.0, "entries": 0.0},
  #     {"sum": 0.0, "entries": 0.0},
  #     {"sum": 0.48081424832344055, "entries": 1.0},
  #     {"sum": 10.879940822720528, "entries": 9.0},
  #     {"sum": 43.35080977156758, "entries": 54.0},
  #     {"sum": 113.69398449920118, "entries": 147.0},
  #     {"sum": 349.6867558255326, "entries": 449.0},
  #     {"sum": 729.5858678516815, "entries": 897.0},
  #     {"sum": 1155.193773361767, "entries": 1451.0},
  #     {"sum": 1520.5854493912775, "entries": 1934.0},
  #     {"sum": 1436.6912576352042, "entries": 1796.0},
  #     {"sum": 1116.2790022112895, "entries": 1386.0},
  #     {"sum": 728.2537153647281, "entries": 922.0},
  #     {"sum": 353.9190010114107, "entries": 436.0},
  #     {"sum": 121.04832566762343, "entries": 158.0},
  #     {"sum": 42.87702897598501, "entries": 53.0},
  #     {"sum": 8.222344039008021, "entries": 13.0},
  #     {"sum": 2.8457946181297302, "entries": 2.0},
  #     {"sum": 0.36020421981811523, "entries": 1.0},
  #     {"sum": 0.0, "entries": 0.0}
  #   ],
  #   "low": -10.0,
  #   "entries": 10000.0,
  #   "overflow": 99.0,
  #   "underflow": 96.0,
  #   "underflow:type": "Count"
  # },
  # "type": "Bin"})
  #           hg.cling(TestRootCling.ttreeFlat, debug=False)
  #           self.assertEqual(hg.toJson(), {"data": {
  #   "values:name": "positive",
  #   "nanflow:type": "Count",
  #   "name": "2 * withholes",
  #   "nanflow": 2*96.0,
  #   "overflow:type": "Count",
  #   "values:type": "Sum",
  #   "high": 10.0,
  #   "values": [
  #     {"sum": 2*0.0, "entries": 2*0.0},
  #     {"sum": 2*0.0, "entries": 2*0.0},
  #     {"sum": 2*0.48081424832344055, "entries": 2*1.0},
  #     {"sum": 2*10.879940822720528, "entries": 2*9.0},
  #     {"sum": 2*43.35080977156758, "entries": 2*54.0},
  #     {"sum": 2*113.69398449920118, "entries": 2*147.0},
  #     {"sum": 2*349.6867558255326, "entries": 2*449.0},
  #     {"sum": 2*729.5858678516815, "entries": 2*897.0},
  #     {"sum": 2*1155.193773361767, "entries": 2*1451.0},
  #     {"sum": 2*1520.5854493912775, "entries": 2*1934.0},
  #     {"sum": 2*1436.6912576352042, "entries": 2*1796.0},
  #     {"sum": 2*1116.2790022112895, "entries": 2*1386.0},
  #     {"sum": 2*728.2537153647281, "entries": 2*922.0},
  #     {"sum": 2*353.9190010114107, "entries": 2*436.0},
  #     {"sum": 2*121.04832566762343, "entries": 2*158.0},
  #     {"sum": 2*42.87702897598501, "entries": 2*53.0},
  #     {"sum": 2*8.222344039008021, "entries": 2*13.0},
  #     {"sum": 2*2.8457946181297302, "entries": 2*2.0},
  #     {"sum": 2*0.36020421981811523, "entries": 2*1.0},
  #     {"sum": 2*0.0, "entries": 2*0.0}
  #   ],
  #   "low": -10.0,
  #   "entries": 2*10000.0,
  #   "overflow": 2*99.0,
  #   "underflow": 2*96.0,
  #   "underflow:type": "Count"
  # },
  # "type": "Bin"})

  #   def testSelect(self):
  #       if TestRootCling.ttreeFlat is not None:
  #           hg = Select("boolean", Bin(20, -10, 10, "noholes"))
  #           hg.cling(TestRootCling.ttreeFlat, debug=False)
  #           self.assertEqual(hg.toJson(), {"data": {
  #   "type": "Bin", 
  #   "data": {
  #     "nanflow:type": "Count", 
  #     "name": "noholes", 
  #     "nanflow": 0.0, 
  #     "overflow:type": "Count", 
  #     "values:type": "Count", 
  #     "high": 10.0, 
  #     "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 22.0, 183.0, 425.0, 472.0, 181.0, 29.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
  #     "low": -10.0, 
  #     "entries": 1317.0, 
  #     "overflow": 0.0, 
  #     "underflow": 0.0, 
  #     "underflow:type": "Count"
  #   }, 
  #   "name": "boolean", 
  #   "entries": 10000.0
  # }, 
  # "type": "Select"}) 
  #           hg.cling(TestRootCling.ttreeFlat, debug=False)
  #           self.assertEqual(hg.toJson(), {"data": {
  #   "type": "Bin", 
  #   "data": {
  #     "nanflow:type": "Count", 
  #     "name": "noholes", 
  #     "nanflow": 2*0.0, 
  #     "overflow:type": "Count", 
  #     "values:type": "Count", 
  #     "high": 10.0, 
  #     "values": [2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*3.0, 2*22.0, 2*183.0, 2*425.0, 2*472.0, 2*181.0, 2*29.0, 2*2.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0], 
  #     "low": -10.0, 
  #     "entries": 2*1317.0, 
  #     "overflow": 2*0.0, 
  #     "underflow": 2*0.0, 
  #     "underflow:type": "Count"
  #   }, 
  #   "name": "boolean", 
  #   "entries": 2*10000.0
  # }, 
  # "type": "Select"}) 

  #           hg = Select("withholes / 2", Bin(20, -10, 10, "noholes"))
  #           hg.cling(TestRootCling.ttreeFlat, debug=False)
  #           self.assertEqual(hg.toJson(), {"data": {
  #   "type": "Bin", 
  #   "data": {
  #     "nanflow:type": "Count", 
  #     "name": "noholes", 
  #     "nanflow": 0.0, 
  #     "overflow:type": "Count", 
  #     "values:type": "Count", 
  #     "high": 10.0, 
  #     "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.7656523132161417, "inf", "inf", "inf", "inf", "inf", 40.84895628585768, 2.824571537630074, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
  #     "low": -10.0, 
  #     "entries": "inf", 
  #     "overflow": 0.0, 
  #     "underflow": 0.0, 
  #     "underflow:type": "Count"
  #   }, 
  #   "name": "withholes / 2", 
  #   "entries": 10000.0
  # }, 
  # "type": "Select"})
  #           hg.cling(TestRootCling.ttreeFlat, debug=False)
  #           self.assertEqual(hg.toJson(), {"data": {
  #   "type": "Bin", 
  #   "data": {
  #     "nanflow:type": "Count", 
  #     "name": "noholes", 
  #     "nanflow": 2*0.0, 
  #     "overflow:type": "Count", 
  #     "values:type": "Count", 
  #     "high": 10.0, 
  #     "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2*3.7656523132161417, "inf", "inf", "inf", "inf", "inf", 2*40.84895628585768, 2*2.824571537630074, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0], 
  #     "low": -10.0, 
  #     "entries": "inf", 
  #     "overflow": 2*0.0, 
  #     "underflow": 2*0.0, 
  #     "underflow:type": "Count"
  #   }, 
  #   "name": "withholes / 2", 
  #   "entries": 2*10000.0
  # }, 
  # "type": "Select"})
