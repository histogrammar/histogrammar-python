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
import sys
import time
import unittest

from histogrammar import *

tolerance = 1e-12
util.relativeTolerance = tolerance
util.absoluteTolerance = tolerance

class TestRootCling(unittest.TestCase):
    ttreeFlat = None
    ttreeEvent = None
    positive = []
    boolean = []
    noholes = []
    withholes = []
    withholes2 = []

    try:
        import ROOT
        ROOT.gInterpreter.AddIncludePath("test/Event.h")
        ROOT.gInterpreter.ProcessLine(".L test/Event.cxx")
        tfileFlat = ROOT.TFile("test/flat.root")
        ttreeFlat = tfileFlat.Get("simple")
        tfileBig = ROOT.TFile("test/big.root")
        ttreeBig = tfileBig.Get("big")
        # tfileEvent = ROOT.TFile("test/Event.root")
        # ttreeEvent = tfileEvent.Get("T")

        for row in ttreeFlat:
            positive.append(row.positive)
            boolean.append(row.boolean)
            noholes.append(row.noholes)
            withholes.append(row.withholes)
            withholes2.append(row.withholes2)

    except ImportError:
        pass

#     ################################################################ Timing

#     def testTiming(self):
#         which = TestRootCling.ttreeFlat

#         if which is not None:
#             which.AddBranchToCache("*", True)
#             for row in which:
#                 pass
#             # print which.GetCurrentFile().GetBytesRead(), which.GetCurrentFile().GetReadCalls()
#             # print which.PrintCacheStats()
#             for row in which:
#                 pass
#             # print which.GetCurrentFile().GetBytesRead(), which.GetCurrentFile().GetReadCalls()
#             # print which.PrintCacheStats()

#             hg = Select("!boolean", Bin(100, -10, 10, "2 * noholes"))

#             # print
#             # print "Histogrammar JIT-compilation"

#             startTime = time.time()
#             hg.cling(which, 0, 1, debug=False)
#             # print time.time() - startTime, which.GetCurrentFile().GetBytesRead(), which.GetCurrentFile().GetReadCalls()

#             # print
#             # print "Histogrammar running"

#             startTime = time.time()
#             hg.cling(which)
#             # print time.time() - startTime, which.GetCurrentFile().GetBytesRead(), which.GetCurrentFile().GetReadCalls()

#             startTime = time.time()
#             hg.cling(which)
#             # print time.time() - startTime, which.GetCurrentFile().GetBytesRead(), which.GetCurrentFile().GetReadCalls()

#             startTime = time.time()
#             hg.cling(which)
#             # print time.time() - startTime, which.GetCurrentFile().GetBytesRead(), which.GetCurrentFile().GetReadCalls()

#             controlTestCode = """class ControlTest {
# public:
#   TH1D histogram;

#   ControlTest() : histogram("control1", "", 100, -10, 10) {}

#   Int_t input_boolean;
#   double input_noholes;
 
#   void fillall(TTree* ttree, Long64_t start, Long64_t end) {
#     if (start < 0) start = 0;
#     if (end < 0) end = ttree->GetEntries();

#     ttree->SetBranchAddress("noholes", &input_noholes);
#     ttree->SetBranchAddress("boolean", &input_boolean);

#     for (;  start < end;  ++start) {
#       ttree->GetEntry(start);
#       if (!input_boolean)
#         histogram.Fill(2 * input_noholes);
#     }

#     ttree->ResetBranchAddresses();
#   }
# };"""

#             # print
#             # print "ROOT C++ compilation"

#             import ROOT
#             startTime = time.time()
#             ROOT.gInterpreter.Declare(controlTestCode)
#             controlTest = ROOT.ControlTest()
#             # print time.time() - startTime, which.GetCurrentFile().GetBytesRead(), which.GetCurrentFile().GetReadCalls()

#             # print
#             # print "ROOT C++ first event"

#             startTime = time.time()
#             controlTest.fillall(which, 0, 1)
#             # print time.time() - startTime, which.GetCurrentFile().GetBytesRead(), which.GetCurrentFile().GetReadCalls()

#             # print
#             # print "ROOT C++ subsequent"

#             startTime = time.time()
#             controlTest.fillall(which, -1, -1)
#             # print time.time() - startTime, which.GetCurrentFile().GetBytesRead(), which.GetCurrentFile().GetReadCalls()

#             startTime = time.time()
#             controlTest.fillall(which, -1, -1)
#             # print time.time() - startTime, which.GetCurrentFile().GetBytesRead(), which.GetCurrentFile().GetReadCalls()

#             startTime = time.time()
#             controlTest.fillall(which, -1, -1)
#             # print time.time() - startTime, which.GetCurrentFile().GetBytesRead(), which.GetCurrentFile().GetReadCalls()

#             # print
#             # print "PyROOT first event"

#             histogram = ROOT.TH1D("control2", "", 100, -10, 10)

#             startTime = time.time()
#             for row in which:
#                 if not row.boolean:
#                     histogram.Fill(2 * row.noholes)
#                 break
#             # print time.time() - startTime, which.GetCurrentFile().GetBytesRead(), which.GetCurrentFile().GetReadCalls()

#             # print
#             # print "PyROOT subsequent"

#             startTime = time.time()
#             for row in which:
#                 if not row.boolean:
#                     histogram.Fill(2 * row.noholes)
#             # print time.time() - startTime, which.GetCurrentFile().GetBytesRead(), which.GetCurrentFile().GetReadCalls()

#             startTime = time.time()
#             for row in which:
#                 if not row.boolean:
#                     histogram.Fill(2 * row.noholes)
#             # print time.time() - startTime, which.GetCurrentFile().GetBytesRead(), which.GetCurrentFile().GetReadCalls()

#             startTime = time.time()
#             for row in which:
#                 if not row.boolean:
#                     histogram.Fill(2 * row.noholes)
#             # print time.time() - startTime, which.GetCurrentFile().GetBytesRead(), which.GetCurrentFile().GetReadCalls()

#             # print
#             # print "TFormula first pass"

#             histogram3 = ROOT.TH1D("control3", "", 100, -10, 10)

#             startTime = time.time()
#             which.Draw("2 * noholes >>+ control3", "!boolean", "goff")
#             # print time.time() - startTime, which.GetCurrentFile().GetBytesRead(), which.GetCurrentFile().GetReadCalls()

#             # print
#             # print "TFormula subsequent"

#             startTime = time.time()
#             which.Draw("2 * noholes >>+ control3", "!boolean", "goff")
#             # print time.time() - startTime, which.GetCurrentFile().GetBytesRead(), which.GetCurrentFile().GetReadCalls()

#             startTime = time.time()
#             which.Draw("2 * noholes >>+ control3", "!boolean", "goff")
#             # print time.time() - startTime, which.GetCurrentFile().GetBytesRead(), which.GetCurrentFile().GetReadCalls()

#             startTime = time.time()
#             which.Draw("2 * noholes >>+ control3", "!boolean", "goff")
#             # print time.time() - startTime, which.GetCurrentFile().GetBytesRead(), which.GetCurrentFile().GetReadCalls()

#             import numpy
#             table = {"boolean": numpy.empty(which.GetEntries(), dtype=numpy.int32),
#                      "noholes": numpy.empty(which.GetEntries(), dtype=numpy.double)}

#             for i, row in enumerate(which):
#                 table["boolean"][i] = row.boolean
#                 table["noholes"][i] = row.noholes

#             hg = Select("logical_not(boolean)", Bin(100, -10, 10, "2 * noholes"))

#             # print
#             # print "Numpy first"

#             startTime = time.time()
#             hg.numpy(table)
#             # print time.time() - startTime

#             # print
#             # print "Numpy subsequent"

#             startTime = time.time()
#             hg.numpy(table)
#             # print time.time() - startTime

#             startTime = time.time()
#             hg.numpy(table)
#             # print time.time() - startTime

#             startTime = time.time()
#             hg.numpy(table)
#             # print time.time() - startTime

#             # print
#             # print "Native Histogrammar first pass"

#             class Row(object):
#                 __slots__ = ["boolean", "noholes"]
#                 def __init__(self, boolean, noholes):
#                     self.boolean = boolean
#                     self.noholes = noholes

#             table2 = []
#             for i in xrange(len(table["boolean"])):
#                 table2.append(Row(table["boolean"][i], table["noholes"][i]))

#             hg = Select(lambda t: not t.boolean, Bin(100, -10, 10, lambda t: 2 * t.noholes))

#             startTime = time.time()
#             for t in table2: hg.fill(t)
#             # print time.time() - startTime

#             # print
#             # print "Native Histogrammar subsequent"

#             startTime = time.time()
#             for t in table2: hg.fill(t)
#             # print time.time() - startTime

#             startTime = time.time()
#             for t in table2: hg.fill(t)
#             # print time.time() - startTime

#             startTime = time.time()
#             for t in table2: hg.fill(t)
#             # print time.time() - startTime

#     ################################################################ OriginalCount

#     def testOriginalCount(self):
#         if TestRootCling.ttreeFlat is not None:
#             hg = Count()
#             hg.cling(TestRootCling.ttreeFlat, debug=False)
#             self.assertEqual(hg.toJson(), {"type": "Count", "data": 10000})
#             hg.cling(TestRootCling.ttreeFlat, debug=False)
#             self.assertEqual(hg.toJson(), {"type": "Count", "data": 20000})

#             hg = Count("0.5 * weight")
#             hg.cling(TestRootCling.ttreeFlat, debug=False)
#             self.assertEqual(hg.toJson(), {"type": "Count", "data": 5000})
#             hg.cling(TestRootCling.ttreeFlat, debug=False)
#             self.assertEqual(hg.toJson(), {"type": "Count", "data": 10000})

#             hg = Count("double twice = weight * 2; twice")
#             hg.cling(TestRootCling.ttreeFlat, debug=False)
#             self.assertEqual(hg.toJson(), {"type": "Count", "data": 20000})
#             hg.cling(TestRootCling.ttreeFlat, debug=False)
#             self.assertEqual(hg.toJson(), {"type": "Count", "data": 40000})

#         if TestRootCling.ttreeEvent is not None:
#             hg = Count()
#             hg.cling(TestRootCling.ttreeEvent, debug=False)
#             self.assertEqual(hg.toJson(), {"type": "Count", "data": 1000})
#             hg.cling(TestRootCling.ttreeEvent, debug=False)
#             self.assertEqual(hg.toJson(), {"type": "Count", "data": 2000})

#             hg = Count("0.5 * weight")
#             hg.cling(TestRootCling.ttreeEvent, debug=False)
#             self.assertEqual(hg.toJson(), {"type": "Count", "data": 500})
#             hg.cling(TestRootCling.ttreeEvent, debug=False)
#             self.assertEqual(hg.toJson(), {"type": "Count", "data": 1000})

#     ################################################################ OriginalSum

#     def testOriginalSum(self):
#         if TestRootCling.ttreeFlat is not None:
#             hg = Sum("positive")
#             hg.cling(TestRootCling.ttreeFlat, debug=False)
#             self.assertEqual(hg.toJson(), {"data": {"sum": 7970.933535083706, "name": "positive", "entries": 10000}, "type": "Sum"})
#             hg.cling(TestRootCling.ttreeFlat, debug=False)
#             self.assertEqual(hg.toJson(), {"data": {"sum": 2*7970.933535083706, "name": "positive", "entries": 20000}, "type": "Sum"})

#             hg = Sum("""2 * t("positive")""")
#             hg.cling(TestRootCling.ttreeFlat, debug=False)
#             self.assertEqual(hg.toJson(), {"data": {"sum": 2*7970.933535083706, "name": """2 * t("positive")""", "entries": 10000}, "type": "Sum"})
#             hg.cling(TestRootCling.ttreeFlat, debug=False)
#             self.assertEqual(hg.toJson(), {"data": {"sum": 4*7970.933535083706, "name": """2 * t("positive")""", "entries": 20000}, "type": "Sum"})

#             hg = Sum("2 * noholes")
#             hg.cling(TestRootCling.ttreeFlat, debug=False)
#             self.assertEqual(hg.toJson(), {"data": {"sum": 137.62044119255137, "name": "2 * noholes", "entries": 10000}, "type": "Sum"})
#             hg.cling(TestRootCling.ttreeFlat, debug=False)
#             self.assertEqual(hg.toJson(), {"data": {"sum": 2*137.62044119255137, "name": "2 * noholes", "entries": 20000}, "type": "Sum"})

#             hg = Sum("double twice = 2 * noholes; twice;")
#             hg.cling(TestRootCling.ttreeFlat, debug=False)
#             self.assertEqual(hg.toJson(), {"data": {"sum": 137.62044119255137, "name": "double twice = 2 * noholes; twice;", "entries": 10000}, "type": "Sum"})
#             hg.cling(TestRootCling.ttreeFlat, debug=False)
#             self.assertEqual(hg.toJson(), {"data": {"sum": 2*137.62044119255137, "name": "double twice = 2 * noholes; twice;", "entries": 20000}, "type": "Sum"})

#             hg = Sum("twice")
#             hg.cling(TestRootCling.ttreeFlat, debug=False, twice="2 * noholes")
#             self.assertEqual(hg.toJson(), {"data": {"sum": 137.62044119255137, "name": "twice", "entries": 10000}, "type": "Sum"})
#             hg.cling(TestRootCling.ttreeFlat, debug=False, twice="2 * noholes")
#             self.assertEqual(hg.toJson(), {"data": {"sum": 2*137.62044119255137, "name": "twice", "entries": 20000}, "type": "Sum"})

#             hg = Sum("quadruple")
#             hg.cling(TestRootCling.ttreeFlat, debug=False, quadruple="double x = 2 * noholes; x*2")
#             self.assertEqual(hg.toJson(), {"data": {"sum": 2*137.62044119255137, "name": "quadruple", "entries": 10000}, "type": "Sum"})
#             hg.cling(TestRootCling.ttreeFlat, debug=False, quadruple="double x = 2 * noholes; x*2")
#             self.assertEqual(hg.toJson(), {"data": {"sum": 4*137.62044119255137, "name": "quadruple", "entries": 20000}, "type": "Sum"})

#         if TestRootCling.ttreeEvent is not None:
#             hg = Sum("event.GetNtrack()")
#             hg.cling(TestRootCling.ttreeEvent, debug=False)
#             self.assertEqual(hg.toJson(), {"data": {"sum": 599640, "name": "event.GetNtrack()", "entries": 1000}, "type": "Sum"})
#             hg.cling(TestRootCling.ttreeEvent, debug=False)
#             self.assertEqual(hg.toJson(), {"data": {"sum": 2*599640, "name": "event.GetNtrack()", "entries": 2*1000}, "type": "Sum"})

#     ################################################################ OriginalBin

#     def testOriginalBin(self):
#         if TestRootCling.ttreeFlat is not None:
#             hg = Bin(20, -10, 10, "withholes")
#             hg.cling(TestRootCling.ttreeFlat, debug=False)

#             self.assertEqual(hg.toJson(), {"data": {
#     "nanflow:type": "Count", 
#     "name": "withholes", 
#     "nanflow": 96.0, 
#     "overflow:type": "Count", 
#     "values:type": "Count", 
#     "high": 10.0, 
#     "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 201.0, 1346.0, 3385.0, 3182.0, 1358.0, 211.0, 15.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     "low": -10.0, 
#     "entries": 10000.0, 
#     "overflow": 99.0, 
#     "underflow": 96.0, 
#     "underflow:type": "Count"
#   }, 
#   "type": "Bin"})
#             hg.cling(TestRootCling.ttreeFlat, debug=False)
#             self.assertEqual(hg.toJson(), {"data": {
#     "nanflow:type": "Count", 
#     "name": "withholes", 
#     "nanflow": 2*96.0, 
#     "overflow:type": "Count", 
#     "values:type": "Count", 
#     "high": 10.0, 
#     "values": [2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*10.0, 2*201.0, 2*1346.0, 2*3385.0, 2*3182.0, 2*1358.0, 2*211.0, 2*15.0, 2*1.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0],
#     "low": -10.0, 
#     "entries": 2*10000.0, 
#     "overflow": 2*99.0, 
#     "underflow": 2*96.0, 
#     "underflow:type": "Count"
#   }, 
#   "type": "Bin"})

#             hg = Bin(20, -10, 10, "2 * withholes", Sum("positive"))
#             hg.cling(TestRootCling.ttreeFlat, debug=False)
#             self.assertEqual(hg.toJson(), {"data": {
#     "values:name": "positive",
#     "nanflow:type": "Count",
#     "name": "2 * withholes",
#     "nanflow": 96.0,
#     "overflow:type": "Count",
#     "values:type": "Sum",
#     "high": 10.0,
#     "values": [
#       {"sum": 0.0, "entries": 0.0},
#       {"sum": 0.0, "entries": 0.0},
#       {"sum": 0.48081424832344055, "entries": 1.0},
#       {"sum": 10.879940822720528, "entries": 9.0},
#       {"sum": 43.35080977156758, "entries": 54.0},
#       {"sum": 113.69398449920118, "entries": 147.0},
#       {"sum": 349.6867558255326, "entries": 449.0},
#       {"sum": 729.5858678516815, "entries": 897.0},
#       {"sum": 1155.193773361767, "entries": 1451.0},
#       {"sum": 1520.5854493912775, "entries": 1934.0},
#       {"sum": 1436.6912576352042, "entries": 1796.0},
#       {"sum": 1116.2790022112895, "entries": 1386.0},
#       {"sum": 728.2537153647281, "entries": 922.0},
#       {"sum": 353.9190010114107, "entries": 436.0},
#       {"sum": 121.04832566762343, "entries": 158.0},
#       {"sum": 42.87702897598501, "entries": 53.0},
#       {"sum": 8.222344039008021, "entries": 13.0},
#       {"sum": 2.8457946181297302, "entries": 2.0},
#       {"sum": 0.36020421981811523, "entries": 1.0},
#       {"sum": 0.0, "entries": 0.0}
#     ],
#     "low": -10.0,
#     "entries": 10000.0,
#     "overflow": 99.0,
#     "underflow": 96.0,
#     "underflow:type": "Count"
#   },
#   "type": "Bin"})
#             hg.cling(TestRootCling.ttreeFlat, debug=False)
#             self.assertEqual(hg.toJson(), {"data": {
#     "values:name": "positive",
#     "nanflow:type": "Count",
#     "name": "2 * withholes",
#     "nanflow": 2*96.0,
#     "overflow:type": "Count",
#     "values:type": "Sum",
#     "high": 10.0,
#     "values": [
#       {"sum": 2*0.0, "entries": 2*0.0},
#       {"sum": 2*0.0, "entries": 2*0.0},
#       {"sum": 2*0.48081424832344055, "entries": 2*1.0},
#       {"sum": 2*10.879940822720528, "entries": 2*9.0},
#       {"sum": 2*43.35080977156758, "entries": 2*54.0},
#       {"sum": 2*113.69398449920118, "entries": 2*147.0},
#       {"sum": 2*349.6867558255326, "entries": 2*449.0},
#       {"sum": 2*729.5858678516815, "entries": 2*897.0},
#       {"sum": 2*1155.193773361767, "entries": 2*1451.0},
#       {"sum": 2*1520.5854493912775, "entries": 2*1934.0},
#       {"sum": 2*1436.6912576352042, "entries": 2*1796.0},
#       {"sum": 2*1116.2790022112895, "entries": 2*1386.0},
#       {"sum": 2*728.2537153647281, "entries": 2*922.0},
#       {"sum": 2*353.9190010114107, "entries": 2*436.0},
#       {"sum": 2*121.04832566762343, "entries": 2*158.0},
#       {"sum": 2*42.87702897598501, "entries": 2*53.0},
#       {"sum": 2*8.222344039008021, "entries": 2*13.0},
#       {"sum": 2*2.8457946181297302, "entries": 2*2.0},
#       {"sum": 2*0.36020421981811523, "entries": 2*1.0},
#       {"sum": 2*0.0, "entries": 2*0.0}
#     ],
#     "low": -10.0,
#     "entries": 2*10000.0,
#     "overflow": 2*99.0,
#     "underflow": 2*96.0,
#     "underflow:type": "Count"
#   },
#   "type": "Bin"})

#     def testSelect(self):
#         if TestRootCling.ttreeFlat is not None:
#             hg = Select("boolean", Bin(20, -10, 10, "noholes"))
#             hg.cling(TestRootCling.ttreeFlat, debug=False)
#             self.assertEqual(hg.toJson(), {"data": {
#     "type": "Bin", 
#     "data": {
#       "nanflow:type": "Count", 
#       "name": "noholes", 
#       "nanflow": 0.0, 
#       "overflow:type": "Count", 
#       "values:type": "Count", 
#       "high": 10.0, 
#       "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 22.0, 183.0, 425.0, 472.0, 181.0, 29.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
#       "low": -10.0, 
#       "entries": 1317.0, 
#       "overflow": 0.0, 
#       "underflow": 0.0, 
#       "underflow:type": "Count"
#     }, 
#     "name": "boolean", 
#     "entries": 10000.0
#   }, 
#   "type": "Select"}) 
#             hg.cling(TestRootCling.ttreeFlat, debug=False)
#             self.assertEqual(hg.toJson(), {"data": {
#     "type": "Bin", 
#     "data": {
#       "nanflow:type": "Count", 
#       "name": "noholes", 
#       "nanflow": 2*0.0, 
#       "overflow:type": "Count", 
#       "values:type": "Count", 
#       "high": 10.0, 
#       "values": [2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*3.0, 2*22.0, 2*183.0, 2*425.0, 2*472.0, 2*181.0, 2*29.0, 2*2.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0], 
#       "low": -10.0, 
#       "entries": 2*1317.0, 
#       "overflow": 2*0.0, 
#       "underflow": 2*0.0, 
#       "underflow:type": "Count"
#     }, 
#     "name": "boolean", 
#     "entries": 2*10000.0
#   }, 
#   "type": "Select"}) 

#             hg = Select("withholes / 2", Bin(20, -10, 10, "noholes"))
#             hg.cling(TestRootCling.ttreeFlat, debug=False)
#             self.assertEqual(hg.toJson(), {"data": {
#     "type": "Bin", 
#     "data": {
#       "nanflow:type": "Count", 
#       "name": "noholes", 
#       "nanflow": 0.0, 
#       "overflow:type": "Count", 
#       "values:type": "Count", 
#       "high": 10.0, 
#       "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.7656523132161417, "inf", "inf", "inf", "inf", "inf", 40.84895628585768, 2.824571537630074, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
#       "low": -10.0, 
#       "entries": "inf", 
#       "overflow": 0.0, 
#       "underflow": 0.0, 
#       "underflow:type": "Count"
#     }, 
#     "name": "withholes / 2", 
#     "entries": 10000.0
#   }, 
#   "type": "Select"})
#             hg.cling(TestRootCling.ttreeFlat, debug=False)
#             self.assertEqual(hg.toJson(), {"data": {
#     "type": "Bin", 
#     "data": {
#       "nanflow:type": "Count", 
#       "name": "noholes", 
#       "nanflow": 2*0.0, 
#       "overflow:type": "Count", 
#       "values:type": "Count", 
#       "high": 10.0, 
#       "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2*3.7656523132161417, "inf", "inf", "inf", "inf", "inf", 2*40.84895628585768, 2*2.824571537630074, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0], 
#       "low": -10.0, 
#       "entries": "inf", 
#       "overflow": 2*0.0, 
#       "underflow": 2*0.0, 
#       "underflow:type": "Count"
#     }, 
#     "name": "withholes / 2", 
#     "entries": 2*10000.0
#   }, 
#   "type": "Select"})

    ################################################################ Tests copied from Numpy

    def twosigfigs(self, number):
        return round(number, 1 - int(math.floor(math.log10(number))))

    def compare(self, name, hrc, hpy, pydata, debug=False):
        sys.stderr.write(name + "\n")

        hrc.cling(TestRootCling.ttreeFlat, debug=debug)
        for x in pydata:
            hpy.fill(x)
        
        if Factory.fromJson(hrc.toJson()) != Factory.fromJson(hpy.toJson()):
            sys.stderr.write("FAILED ON FIRST FILL\n")
            left = json.dumps(hrc.toJson(), sort_keys=True, indent=2)
            right = json.dumps(hpy.toJson(), sort_keys=True, indent=2)
            for leftline, rightline in zip(left.split("\n"), right.split("\n")):
                if leftline != rightline:
                    sys.stderr.write("{0:50s} > {1}\n".format(leftline, rightline))
                else:
                    sys.stderr.write("{0:50s} | {1}\n".format(leftline, rightline))
            self.assertEqual(Factory.fromJson(hrc.toJson()), Factory.fromJson(hpy.toJson()))

        hrc.cling(TestRootCling.ttreeFlat, debug=debug)
        for x in pydata:
            hpy.fill(x)
        
        if Factory.fromJson(hrc.toJson()) != Factory.fromJson(hpy.toJson()):
            sys.stderr.write("FAILED ON SECOND FILL\n")
            left = json.dumps(hrc.toJson(), sort_keys=True, indent=2)
            right = json.dumps(hpy.toJson(), sort_keys=True, indent=2)
            for leftline, rightline in zip(left.split("\n"), right.split("\n")):
                if leftline != rightline:
                    sys.stderr.write("{0:50s} > {1}\n".format(leftline, rightline))
                else:
                    sys.stderr.write("{0:50s} | {1}\n".format(leftline, rightline))
            self.assertEqual(Factory.fromJson(hrc.toJson()), Factory.fromJson(hpy.toJson()))

    # def testSum(self):
    #     if TestRootCling.ttreeFlat is not None:
    #         sys.stderr.write("\n")
    #         self.compare("Sum noholes w/o weights", Sum("noholes"), Sum(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Sum noholes const weights", Sum("noholes"), Sum(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Sum noholes positive weights", Sum("noholes"), Sum(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Sum noholes with weights", Sum("noholes"), Sum(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Sum noholes with holes", Sum("noholes"), Sum(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Sum holes w/o weights", Sum("withholes"), Sum(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Sum holes const weights", Sum("withholes"), Sum(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Sum holes positive weights", Sum("withholes"), Sum(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Sum holes with weights", Sum("withholes"), Sum(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Sum holes with holes", Sum("withholes"), Sum(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Sum holes with holes2", Sum("withholes"), Sum(named("withholes", lambda x: x)), self.withholes)

    # def testAverage(self):
    #     if TestRootCling.ttreeFlat is not None:
    #         sys.stderr.write("\n")
    #         self.compare("Average noholes w/o weights", Average("noholes"), Average(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Average noholes const weights", Average("noholes"), Average(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Average noholes positive weights", Average("noholes"), Average(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Average noholes with weights", Average("noholes"), Average(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Average noholes with holes", Average("noholes"), Average(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Average holes w/o weights", Average("withholes"), Average(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Average holes const weights", Average("withholes"), Average(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Average holes positive weights", Average("withholes"), Average(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Average holes with weights", Average("withholes"), Average(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Average holes with holes", Average("withholes"), Average(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Average holes with holes2", Average("withholes"), Average(named("withholes", lambda x: x)), self.withholes)

    # def testDeviate(self):
    #     if TestRootCling.ttreeFlat is not None:
    #         sys.stderr.write("\n")
    #         self.compare("Deviate noholes w/o weights", Deviate("noholes"), Deviate(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Deviate noholes const weights", Deviate("noholes"), Deviate(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Deviate noholes positive weights", Deviate("noholes"), Deviate(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Deviate noholes with weights", Deviate("noholes"), Deviate(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Deviate noholes with holes", Deviate("noholes"), Deviate(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Deviate holes w/o weights", Deviate("withholes"), Deviate(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Deviate holes const weights", Deviate("withholes"), Deviate(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Deviate holes positive weights", Deviate("withholes"), Deviate(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Deviate holes with weights", Deviate("withholes"), Deviate(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Deviate holes with holes", Deviate("withholes"), Deviate(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Deviate holes with holes2", Deviate("withholes"), Deviate(named("withholes", lambda x: x)), self.withholes)

    # def testMinimize(self):
    #     if TestRootCling.ttreeFlat is not None:
    #         sys.stderr.write("\n")
    #         self.compare("Minimize noholes w/o weights", Minimize("noholes"), Minimize(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Minimize noholes const weights", Minimize("noholes"), Minimize(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Minimize noholes positive weights", Minimize("noholes"), Minimize(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Minimize noholes with weights", Minimize("noholes"), Minimize(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Minimize noholes with holes", Minimize("noholes"), Minimize(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Minimize holes w/o weights", Minimize("withholes"), Minimize(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Minimize holes const weights", Minimize("withholes"), Minimize(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Minimize holes positive weights", Minimize("withholes"), Minimize(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Minimize holes with weights", Minimize("withholes"), Minimize(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Minimize holes with holes", Minimize("withholes"), Minimize(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Minimize holes with holes2", Minimize("withholes"), Minimize(named("withholes", lambda x: x)), self.withholes)

    # def testMaximize(self):
    #     if TestRootCling.ttreeFlat is not None:
    #         sys.stderr.write("\n")
    #         self.compare("Maximize noholes w/o weights", Maximize("noholes"), Maximize(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Maximize noholes const weights", Maximize("noholes"), Maximize(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Maximize noholes positive weights", Maximize("noholes"), Maximize(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Maximize noholes with weights", Maximize("noholes"), Maximize(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Maximize noholes with holes", Maximize("noholes"), Maximize(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Maximize holes w/o weights", Maximize("withholes"), Maximize(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Maximize holes const weights", Maximize("withholes"), Maximize(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Maximize holes positive weights", Maximize("withholes"), Maximize(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Maximize holes with weights", Maximize("withholes"), Maximize(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Maximize holes with holes", Maximize("withholes"), Maximize(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Maximize holes with holes2", Maximize("withholes"), Maximize(named("withholes", lambda x: x)), self.withholes)

    # def testBin(self):
    #     if TestRootCling.ttreeFlat is not None:
    #         sys.stderr.write("\n")
    #         for bins in [10, 100]:
    #             self.compare("Bin ({0} bins) noholes w/o weights".format(bins), Bin(bins, -3.0, 3.0, "noholes"), Bin(bins, -3.0, 3.0, named("noholes", lambda x: x)), self.noholes)
    #             self.compare("Bin ({0} bins) noholes const weights".format(bins), Bin(bins, -3.0, 3.0, "noholes"), Bin(bins, -3.0, 3.0, named("noholes", lambda x: x)), self.noholes)
    #             self.compare("Bin ({0} bins) noholes positive weights".format(bins), Bin(bins, -3.0, 3.0, "noholes"), Bin(bins, -3.0, 3.0, named("noholes", lambda x: x)), self.noholes)
    #             self.compare("Bin ({0} bins) noholes with weights".format(bins), Bin(bins, -3.0, 3.0, "noholes"), Bin(bins, -3.0, 3.0, named("noholes", lambda x: x)), self.noholes)
    #             self.compare("Bin ({0} bins) noholes with holes".format(bins), Bin(bins, -3.0, 3.0, "noholes"), Bin(bins, -3.0, 3.0, named("noholes", lambda x: x)), self.noholes)
    #             self.compare("Bin ({0} bins) holes w/o weights".format(bins), Bin(bins, -3.0, 3.0, "withholes"), Bin(bins, -3.0, 3.0, named("withholes", lambda x: x)), self.withholes)
    #             self.compare("Bin ({0} bins) holes const weights".format(bins), Bin(bins, -3.0, 3.0, "withholes"), Bin(bins, -3.0, 3.0, named("withholes", lambda x: x)), self.withholes)
    #             self.compare("Bin ({0} bins) holes positive weights".format(bins), Bin(bins, -3.0, 3.0, "withholes"), Bin(bins, -3.0, 3.0, named("withholes", lambda x: x)), self.withholes)
    #             self.compare("Bin ({0} bins) holes with weights".format(bins), Bin(bins, -3.0, 3.0, "withholes"), Bin(bins, -3.0, 3.0, named("withholes", lambda x: x)), self.withholes)
    #             self.compare("Bin ({0} bins) holes with holes".format(bins), Bin(bins, -3.0, 3.0, "withholes"), Bin(bins, -3.0, 3.0, named("withholes", lambda x: x)), self.withholes)
    #             self.compare("Bin ({0} bins) holes with holes2".format(bins), Bin(bins, -3.0, 3.0, "withholes"), Bin(bins, -3.0, 3.0, named("withholes", lambda x: x)), self.withholes)

    # def testBinTrans(self):
    #     if TestRootCling.ttreeFlat is not None:
    #         sys.stderr.write("\n")
    #         for bins in [10, 100]:
    #             self.compare("BinTrans ({0} bins) noholes w/o weights".format(bins), Bin(bins, -3.0, 3.0, "noholes", Count("0.5*weight")), Bin(bins, -3.0, 3.0, named("noholes", lambda x: x), Count("0.5*weight")), self.noholes)
    #             self.compare("BinTrans ({0} bins) noholes const weights".format(bins), Bin(bins, -3.0, 3.0, "noholes", Count("0.5*weight")), Bin(bins, -3.0, 3.0, named("noholes", lambda x: x), Count("0.5*weight")), self.noholes)
    #             self.compare("BinTrans ({0} bins) noholes positive weights".format(bins), Bin(bins, -3.0, 3.0, "noholes", Count("0.5*weight")), Bin(bins, -3.0, 3.0, named("noholes", lambda x: x), Count("0.5*weight")), self.noholes)
    #             self.compare("BinTrans ({0} bins) noholes with weights".format(bins), Bin(bins, -3.0, 3.0, "noholes", Count("0.5*weight")), Bin(bins, -3.0, 3.0, named("noholes", lambda x: x), Count("0.5*weight")), self.noholes)
    #             self.compare("BinTrans ({0} bins) noholes with holes".format(bins), Bin(bins, -3.0, 3.0, "noholes", Count("0.5*weight")), Bin(bins, -3.0, 3.0, named("noholes", lambda x: x), Count("0.5*weight")), self.noholes)
    #             self.compare("BinTrans ({0} bins) holes w/o weights".format(bins), Bin(bins, -3.0, 3.0, "withholes", Count("0.5*weight")), Bin(bins, -3.0, 3.0, named("withholes", lambda x: x), Count("0.5*weight")), self.withholes)
    #             self.compare("BinTrans ({0} bins) holes const weights".format(bins), Bin(bins, -3.0, 3.0, "withholes", Count("0.5*weight")), Bin(bins, -3.0, 3.0, named("withholes", lambda x: x), Count("0.5*weight")), self.withholes)
    #             self.compare("BinTrans ({0} bins) holes positive weights".format(bins), Bin(bins, -3.0, 3.0, "withholes", Count("0.5*weight")), Bin(bins, -3.0, 3.0, named("withholes", lambda x: x), Count("0.5*weight")), self.withholes)
    #             self.compare("BinTrans ({0} bins) holes with weights".format(bins), Bin(bins, -3.0, 3.0, "withholes", Count("0.5*weight")), Bin(bins, -3.0, 3.0, named("withholes", lambda x: x), Count("0.5*weight")), self.withholes)
    #             self.compare("BinTrans ({0} bins) holes with holes".format(bins), Bin(bins, -3.0, 3.0, "withholes", Count("0.5*weight")), Bin(bins, -3.0, 3.0, named("withholes", lambda x: x), Count("0.5*weight")), self.withholes)
    #             self.compare("BinTrans ({0} bins) holes with holes2".format(bins), Bin(bins, -3.0, 3.0, "withholes", Count("0.5*weight")), Bin(bins, -3.0, 3.0, named("withholes", lambda x: x), Count("0.5*weight")), self.withholes)

    # def testBinAverage(self):
    #     if TestRootCling.ttreeFlat is not None:
    #         sys.stderr.write("\n")
    #         for bins in [10, 100]:
    #             self.compare("BinAverage ({0} bins) noholes w/o weights".format(bins), Bin(bins, -3.0, 3.0, "noholes", Average("noholes")), Bin(bins, -3.0, 3.0, named("noholes", lambda x: x), Average(named("noholes", lambda x: x))), self.noholes)
    #             self.compare("BinAverage ({0} bins) noholes const weights".format(bins), Bin(bins, -3.0, 3.0, "noholes", Average("noholes")), Bin(bins, -3.0, 3.0, named("noholes", lambda x: x), Average(named("noholes", lambda x: x))), self.noholes)
    #             self.compare("BinAverage ({0} bins) noholes positive weights".format(bins), Bin(bins, -3.0, 3.0, "noholes", Average("noholes")), Bin(bins, -3.0, 3.0, named("noholes", lambda x: x), Average(named("noholes", lambda x: x))), self.noholes)
    #             self.compare("BinAverage ({0} bins) noholes with weights".format(bins), Bin(bins, -3.0, 3.0, "noholes", Average("noholes")), Bin(bins, -3.0, 3.0, named("noholes", lambda x: x), Average(named("noholes", lambda x: x))), self.noholes)
    #             self.compare("BinAverage ({0} bins) noholes with holes".format(bins), Bin(bins, -3.0, 3.0, "noholes", Average("noholes")), Bin(bins, -3.0, 3.0, named("noholes", lambda x: x), Average(named("noholes", lambda x: x))), self.noholes)
    #             self.compare("BinAverage ({0} bins) holes w/o weights".format(bins), Bin(bins, -3.0, 3.0, "withholes", Average("withholes")), Bin(bins, -3.0, 3.0, named("withholes", lambda x: x), Average(named("withholes", lambda x: x))), self.withholes)
    #             self.compare("BinAverage ({0} bins) holes const weights".format(bins), Bin(bins, -3.0, 3.0, "withholes", Average("withholes")), Bin(bins, -3.0, 3.0, named("withholes", lambda x: x), Average(named("withholes", lambda x: x))), self.withholes)
    #             self.compare("BinAverage ({0} bins) holes positive weights".format(bins), Bin(bins, -3.0, 3.0, "withholes", Average("withholes")), Bin(bins, -3.0, 3.0, named("withholes", lambda x: x), Average(named("withholes", lambda x: x))), self.withholes)
    #             self.compare("BinAverage ({0} bins) holes with weights".format(bins), Bin(bins, -3.0, 3.0, "withholes", Average("withholes")), Bin(bins, -3.0, 3.0, named("withholes", lambda x: x), Average(named("withholes", lambda x: x))), self.withholes)
    #             self.compare("BinAverage ({0} bins) holes with holes".format(bins), Bin(bins, -3.0, 3.0, "withholes", Average("withholes")), Bin(bins, -3.0, 3.0, named("withholes", lambda x: x), Average(named("withholes", lambda x: x))), self.withholes)
    #             self.compare("BinAverage ({0} bins) holes with holes2".format(bins), Bin(bins, -3.0, 3.0, "withholes", Average("withholes")), Bin(bins, -3.0, 3.0, named("withholes", lambda x: x), Average(named("withholes", lambda x: x))), self.withholes)

    # def testBinDeviate(self):
    #     if TestRootCling.ttreeFlat is not None:
    #         sys.stderr.write("\n")
    #         for bins in [10, 100]:
    #             self.compare("BinDeviate ({0} bins) noholes w/o weights".format(bins), Bin(bins, -3.0, 3.0, "noholes", Deviate("noholes")), Bin(bins, -3.0, 3.0, named("noholes", lambda x: x), Deviate(named("noholes", lambda x: x))), self.noholes)
    #             self.compare("BinDeviate ({0} bins) noholes const weights".format(bins), Bin(bins, -3.0, 3.0, "noholes", Deviate("noholes")), Bin(bins, -3.0, 3.0, named("noholes", lambda x: x), Deviate(named("noholes", lambda x: x))), self.noholes)
    #             self.compare("BinDeviate ({0} bins) noholes positive weights".format(bins), Bin(bins, -3.0, 3.0, "noholes", Deviate("noholes")), Bin(bins, -3.0, 3.0, named("noholes", lambda x: x), Deviate(named("noholes", lambda x: x))), self.noholes)
    #             self.compare("BinDeviate ({0} bins) noholes with weights".format(bins), Bin(bins, -3.0, 3.0, "noholes", Deviate("noholes")), Bin(bins, -3.0, 3.0, named("noholes", lambda x: x), Deviate(named("noholes", lambda x: x))), self.noholes)
    #             self.compare("BinDeviate ({0} bins) noholes with holes".format(bins), Bin(bins, -3.0, 3.0, "noholes", Deviate("noholes")), Bin(bins, -3.0, 3.0, named("noholes", lambda x: x), Deviate(named("noholes", lambda x: x))), self.noholes)
    #             self.compare("BinDeviate ({0} bins) holes w/o weights".format(bins), Bin(bins, -3.0, 3.0, "withholes", Deviate("withholes")), Bin(bins, -3.0, 3.0, named("withholes", lambda x: x), Deviate(named("withholes", lambda x: x))), self.withholes)
    #             self.compare("BinDeviate ({0} bins) holes const weights".format(bins), Bin(bins, -3.0, 3.0, "withholes", Deviate("withholes")), Bin(bins, -3.0, 3.0, named("withholes", lambda x: x), Deviate(named("withholes", lambda x: x))), self.withholes)
    #             self.compare("BinDeviate ({0} bins) holes positive weights".format(bins), Bin(bins, -3.0, 3.0, "withholes", Deviate("withholes")), Bin(bins, -3.0, 3.0, named("withholes", lambda x: x), Deviate(named("withholes", lambda x: x))), self.withholes)
    #             self.compare("BinDeviate ({0} bins) holes with weights".format(bins), Bin(bins, -3.0, 3.0, "withholes", Deviate("withholes")), Bin(bins, -3.0, 3.0, named("withholes", lambda x: x), Deviate(named("withholes", lambda x: x))), self.withholes)
    #             self.compare("BinDeviate ({0} bins) holes with holes".format(bins), Bin(bins, -3.0, 3.0, "withholes", Deviate("withholes")), Bin(bins, -3.0, 3.0, named("withholes", lambda x: x), Deviate(named("withholes", lambda x: x))), self.withholes)
    #             self.compare("BinDeviate ({0} bins) holes with holes2".format(bins), Bin(bins, -3.0, 3.0, "withholes", Deviate("withholes")), Bin(bins, -3.0, 3.0, named("withholes", lambda x: x), Deviate(named("withholes", lambda x: x))), self.withholes)

    # def testSparselyBin(self):
    #     if TestRootCling.ttreeFlat is not None:
    #         sys.stderr.write("\n")
    #         self.compare("SparselyBin noholes w/o weights", SparselyBin(0.1, "noholes"), SparselyBin(0.1, named("noholes", lambda x: x)), self.noholes)
    #         self.compare("SparselyBin noholes const weights", SparselyBin(0.1, "noholes"), SparselyBin(0.1, named("noholes", lambda x: x)), self.noholes)
    #         self.compare("SparselyBin noholes positive weights", SparselyBin(0.1, "noholes"), SparselyBin(0.1, named("noholes", lambda x: x)), self.noholes)
    #         self.compare("SparselyBin noholes with weights", SparselyBin(0.1, "noholes"), SparselyBin(0.1, named("noholes", lambda x: x)), self.noholes)
    #         self.compare("SparselyBin noholes with holes", SparselyBin(0.1, "noholes"), SparselyBin(0.1, named("noholes", lambda x: x)), self.noholes)
    #         self.compare("SparselyBin holes w/o weights", SparselyBin(0.1, "withholes"), SparselyBin(0.1, named("withholes", lambda x: x)), self.withholes)
    #         self.compare("SparselyBin holes const weights", SparselyBin(0.1, "withholes"), SparselyBin(0.1, named("withholes", lambda x: x)), self.withholes)
    #         self.compare("SparselyBin holes positive weights", SparselyBin(0.1, "withholes"), SparselyBin(0.1, named("withholes", lambda x: x)), self.withholes)
    #         self.compare("SparselyBin holes with weights", SparselyBin(0.1, "withholes"), SparselyBin(0.1, named("withholes", lambda x: x)), self.withholes)
    #         self.compare("SparselyBin holes with holes", SparselyBin(0.1, "withholes"), SparselyBin(0.1, named("withholes", lambda x: x)), self.withholes)
    #         self.compare("SparselyBin holes with holes2", SparselyBin(0.1, "withholes"), SparselyBin(0.1, named("withholes", lambda x: x)), self.withholes)

    # def testSparselyBinTrans(self):
    #     if TestRootCling.ttreeFlat is not None:
    #         sys.stderr.write("\n")
    #         self.compare("SparselyBinTrans noholes w/o weights", SparselyBin(0.1, "noholes", Count("0.5*weight")), SparselyBin(0.1, named("noholes", lambda x: x), Count("0.5*weight")), self.noholes)
    #         self.compare("SparselyBinTrans noholes const weights", SparselyBin(0.1, "noholes", Count("0.5*weight")), SparselyBin(0.1, named("noholes", lambda x: x), Count("0.5*weight")), self.noholes)
    #         self.compare("SparselyBinTrans noholes positive weights", SparselyBin(0.1, "noholes", Count("0.5*weight")), SparselyBin(0.1, named("noholes", lambda x: x), Count("0.5*weight")), self.noholes)
    #         self.compare("SparselyBinTrans noholes with weights", SparselyBin(0.1, "noholes", Count("0.5*weight")), SparselyBin(0.1, named("noholes", lambda x: x), Count("0.5*weight")), self.noholes)
    #         self.compare("SparselyBinTrans noholes with holes", SparselyBin(0.1, "noholes", Count("0.5*weight")), SparselyBin(0.1, named("noholes", lambda x: x), Count("0.5*weight")), self.noholes)
    #         self.compare("SparselyBinTrans holes w/o weights", SparselyBin(0.1, "withholes", Count("0.5*weight")), SparselyBin(0.1, named("withholes", lambda x: x), Count("0.5*weight")), self.withholes)
    #         self.compare("SparselyBinTrans holes const weights", SparselyBin(0.1, "withholes", Count("0.5*weight")), SparselyBin(0.1, named("withholes", lambda x: x), Count("0.5*weight")), self.withholes)
    #         self.compare("SparselyBinTrans holes positive weights", SparselyBin(0.1, "withholes", Count("0.5*weight")), SparselyBin(0.1, named("withholes", lambda x: x), Count("0.5*weight")), self.withholes)
    #         self.compare("SparselyBinTrans holes with weights", SparselyBin(0.1, "withholes", Count("0.5*weight")), SparselyBin(0.1, named("withholes", lambda x: x), Count("0.5*weight")), self.withholes)
    #         self.compare("SparselyBinTrans holes with holes", SparselyBin(0.1, "withholes", Count("0.5*weight")), SparselyBin(0.1, named("withholes", lambda x: x), Count("0.5*weight")), self.withholes)
    #         self.compare("SparselyBinTrans holes with holes2", SparselyBin(0.1, "withholes", Count("0.5*weight")), SparselyBin(0.1, named("withholes", lambda x: x), Count("0.5*weight")), self.withholes)

    # def testSparselyBinAverage(self):
    #     if TestRootCling.ttreeFlat is not None:
    #         sys.stderr.write("\n")
    #         self.compare("SparselyBinAverage noholes w/o weights", SparselyBin(0.1, "noholes", Average("noholes")), SparselyBin(0.1, named("noholes", lambda x: x), Average(named("noholes", lambda x: x))), self.noholes)
    #         self.compare("SparselyBinAverage noholes const weights", SparselyBin(0.1, "noholes", Average("noholes")), SparselyBin(0.1, named("noholes", lambda x: x), Average(named("noholes", lambda x: x))), self.noholes)
    #         self.compare("SparselyBinAverage noholes positive weights", SparselyBin(0.1, "noholes", Average("noholes")), SparselyBin(0.1, named("noholes", lambda x: x), Average(named("noholes", lambda x: x))), self.noholes)
    #         self.compare("SparselyBinAverage noholes with weights", SparselyBin(0.1, "noholes", Average("noholes")), SparselyBin(0.1, named("noholes", lambda x: x), Average(named("noholes", lambda x: x))), self.noholes)
    #         self.compare("SparselyBinAverage noholes with holes", SparselyBin(0.1, "noholes", Average("noholes")), SparselyBin(0.1, named("noholes", lambda x: x), Average(named("noholes", lambda x: x))), self.noholes)
    #         self.compare("SparselyBinAverage holes w/o weights", SparselyBin(0.1, "withholes", Average("withholes")), SparselyBin(0.1, named("withholes", lambda x: x), Average(named("withholes", lambda x: x))), self.withholes)
    #         self.compare("SparselyBinAverage holes const weights", SparselyBin(0.1, "withholes", Average("withholes")), SparselyBin(0.1, named("withholes", lambda x: x), Average(named("withholes", lambda x: x))), self.withholes)
    #         self.compare("SparselyBinAverage holes positive weights", SparselyBin(0.1, "withholes", Average("withholes")), SparselyBin(0.1, named("withholes", lambda x: x), Average(named("withholes", lambda x: x))), self.withholes)
    #         self.compare("SparselyBinAverage holes with weights", SparselyBin(0.1, "withholes", Average("withholes")), SparselyBin(0.1, named("withholes", lambda x: x), Average(named("withholes", lambda x: x))), self.withholes)
    #         self.compare("SparselyBinAverage holes with holes", SparselyBin(0.1, "withholes", Average("withholes")), SparselyBin(0.1, named("withholes", lambda x: x), Average(named("withholes", lambda x: x))), self.withholes)
    #         self.compare("SparselyBinAverage holes with holes2", SparselyBin(0.1, "withholes", Average("withholes")), SparselyBin(0.1, named("withholes", lambda x: x), Average(named("withholes", lambda x: x))), self.withholes)

    # def testSparselyBinDeviate(self):
    #     if TestRootCling.ttreeFlat is not None:
    #         sys.stderr.write("\n")
    #         self.compare("SparselyBinDeviate noholes w/o weights", SparselyBin(0.1, "noholes", Deviate("noholes")), SparselyBin(0.1, named("noholes", lambda x: x), Deviate(named("noholes", lambda x: x))), self.noholes)
    #         self.compare("SparselyBinDeviate noholes const weights", SparselyBin(0.1, "noholes", Deviate("noholes")), SparselyBin(0.1, named("noholes", lambda x: x), Deviate(named("noholes", lambda x: x))), self.noholes)
    #         self.compare("SparselyBinDeviate noholes positive weights", SparselyBin(0.1, "noholes", Deviate("noholes")), SparselyBin(0.1, named("noholes", lambda x: x), Deviate(named("noholes", lambda x: x))), self.noholes)
    #         self.compare("SparselyBinDeviate noholes with weights", SparselyBin(0.1, "noholes", Deviate("noholes")), SparselyBin(0.1, named("noholes", lambda x: x), Deviate(named("noholes", lambda x: x))), self.noholes)
    #         self.compare("SparselyBinDeviate noholes with holes", SparselyBin(0.1, "noholes", Deviate("noholes")), SparselyBin(0.1, named("noholes", lambda x: x), Deviate(named("noholes", lambda x: x))), self.noholes)
    #         self.compare("SparselyBinDeviate holes w/o weights", SparselyBin(0.1, "withholes", Deviate("withholes")), SparselyBin(0.1, named("withholes", lambda x: x), Deviate(named("withholes", lambda x: x))), self.withholes)
    #         self.compare("SparselyBinDeviate holes const weights", SparselyBin(0.1, "withholes", Deviate("withholes")), SparselyBin(0.1, named("withholes", lambda x: x), Deviate(named("withholes", lambda x: x))), self.withholes)
    #         self.compare("SparselyBinDeviate holes positive weights", SparselyBin(0.1, "withholes", Deviate("withholes")), SparselyBin(0.1, named("withholes", lambda x: x), Deviate(named("withholes", lambda x: x))), self.withholes)
    #         self.compare("SparselyBinDeviate holes with weights", SparselyBin(0.1, "withholes", Deviate("withholes")), SparselyBin(0.1, named("withholes", lambda x: x), Deviate(named("withholes", lambda x: x))), self.withholes)
    #         self.compare("SparselyBinDeviate holes with holes", SparselyBin(0.1, "withholes", Deviate("withholes")), SparselyBin(0.1, named("withholes", lambda x: x), Deviate(named("withholes", lambda x: x))), self.withholes)
    #         self.compare("SparselyBinDeviate holes with holes2", SparselyBin(0.1, "withholes", Deviate("withholes")), SparselyBin(0.1, named("withholes", lambda x: x), Deviate(named("withholes", lambda x: x))), self.withholes)

    def testCentrallyBin(self):
        if TestRootCling.ttreeFlat is not None:
            sys.stderr.write("\n")
            centers = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
            self.compare("CentrallyBin noholes w/o weights", CentrallyBin(centers, "noholes"), CentrallyBin(centers, named("noholes", lambda x: x)), self.noholes)
            self.compare("CentrallyBin noholes const weights", CentrallyBin(centers, "noholes"), CentrallyBin(centers, named("noholes", lambda x: x)), self.noholes)
            self.compare("CentrallyBin noholes positive weights", CentrallyBin(centers, "noholes"), CentrallyBin(centers, named("noholes", lambda x: x)), self.noholes)
            self.compare("CentrallyBin noholes with weights", CentrallyBin(centers, "noholes"), CentrallyBin(centers, named("noholes", lambda x: x)), self.noholes)
            self.compare("CentrallyBin noholes with holes", CentrallyBin(centers, "noholes"), CentrallyBin(centers, named("noholes", lambda x: x)), self.noholes)
            self.compare("CentrallyBin holes w/o weights", CentrallyBin(centers, "withholes"), CentrallyBin(centers, named("withholes", lambda x: x)), self.withholes)
            self.compare("CentrallyBin holes const weights", CentrallyBin(centers, "withholes"), CentrallyBin(centers, named("withholes", lambda x: x)), self.withholes)
            self.compare("CentrallyBin holes positive weights", CentrallyBin(centers, "withholes"), CentrallyBin(centers, named("withholes", lambda x: x)), self.withholes)
            self.compare("CentrallyBin holes with weights", CentrallyBin(centers, "withholes"), CentrallyBin(centers, named("withholes", lambda x: x)), self.withholes)
            self.compare("CentrallyBin holes with holes", CentrallyBin(centers, "withholes"), CentrallyBin(centers, named("withholes", lambda x: x)), self.withholes)
            self.compare("CentrallyBin holes with holes2", CentrallyBin(centers, "withholes"), CentrallyBin(centers, named("withholes", lambda x: x)), self.withholes)

    def testCentrallyBinTrans(self):
        if TestRootCling.ttreeFlat is not None:
            sys.stderr.write("\n")
            centers = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
            self.compare("CentrallyBinTrans noholes w/o weights", CentrallyBin(centers, "noholes", Count("0.5*weight")), CentrallyBin(centers, named("noholes", lambda x: x), Count("0.5*weight")), self.noholes)
            self.compare("CentrallyBinTrans noholes const weights", CentrallyBin(centers, "noholes", Count("0.5*weight")), CentrallyBin(centers, named("noholes", lambda x: x), Count("0.5*weight")), self.noholes)
            self.compare("CentrallyBinTrans noholes positive weights", CentrallyBin(centers, "noholes", Count("0.5*weight")), CentrallyBin(centers, named("noholes", lambda x: x), Count("0.5*weight")), self.noholes)
            self.compare("CentrallyBinTrans noholes with weights", CentrallyBin(centers, "noholes", Count("0.5*weight")), CentrallyBin(centers, named("noholes", lambda x: x), Count("0.5*weight")), self.noholes)
            self.compare("CentrallyBinTrans noholes with holes", CentrallyBin(centers, "noholes", Count("0.5*weight")), CentrallyBin(centers, named("noholes", lambda x: x), Count("0.5*weight")), self.noholes)
            self.compare("CentrallyBinTrans holes w/o weights", CentrallyBin(centers, "withholes", Count("0.5*weight")), CentrallyBin(centers, named("withholes", lambda x: x), Count("0.5*weight")), self.withholes)
            self.compare("CentrallyBinTrans holes const weights", CentrallyBin(centers, "withholes", Count("0.5*weight")), CentrallyBin(centers, named("withholes", lambda x: x), Count("0.5*weight")), self.withholes)
            self.compare("CentrallyBinTrans holes positive weights", CentrallyBin(centers, "withholes", Count("0.5*weight")), CentrallyBin(centers, named("withholes", lambda x: x), Count("0.5*weight")), self.withholes)
            self.compare("CentrallyBinTrans holes with weights", CentrallyBin(centers, "withholes", Count("0.5*weight")), CentrallyBin(centers, named("withholes", lambda x: x), Count("0.5*weight")), self.withholes)
            self.compare("CentrallyBinTrans holes with holes", CentrallyBin(centers, "withholes", Count("0.5*weight")), CentrallyBin(centers, named("withholes", lambda x: x), Count("0.5*weight")), self.withholes)
            self.compare("CentrallyBinTrans holes with holes2", CentrallyBin(centers, "withholes", Count("0.5*weight")), CentrallyBin(centers, named("withholes", lambda x: x), Count("0.5*weight")), self.withholes)

    def testCentrallyBinAverage(self):
        if TestRootCling.ttreeFlat is not None:
            sys.stderr.write("\n")
            centers = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
            self.compare("CentrallyBinAverage noholes w/o weights", CentrallyBin(centers, "noholes", Average("noholes")), CentrallyBin(centers, named("noholes", lambda x: x), Average(named("noholes", lambda x: x))), self.noholes)
            self.compare("CentrallyBinAverage noholes const weights", CentrallyBin(centers, "noholes", Average("noholes")), CentrallyBin(centers, named("noholes", lambda x: x), Average(named("noholes", lambda x: x))), self.noholes)
            self.compare("CentrallyBinAverage noholes positive weights", CentrallyBin(centers, "noholes", Average("noholes")), CentrallyBin(centers, named("noholes", lambda x: x), Average(named("noholes", lambda x: x))), self.noholes)
            self.compare("CentrallyBinAverage noholes with weights", CentrallyBin(centers, "noholes", Average("noholes")), CentrallyBin(centers, named("noholes", lambda x: x), Average(named("noholes", lambda x: x))), self.noholes)
            self.compare("CentrallyBinAverage noholes with holes", CentrallyBin(centers, "noholes", Average("noholes")), CentrallyBin(centers, named("noholes", lambda x: x), Average(named("noholes", lambda x: x))), self.noholes)
            self.compare("CentrallyBinAverage holes w/o weights", CentrallyBin(centers, "withholes", Average("withholes")), CentrallyBin(centers, named("withholes", lambda x: x), Average(named("withholes", lambda x: x))), self.withholes)
            self.compare("CentrallyBinAverage holes const weights", CentrallyBin(centers, "withholes", Average("withholes")), CentrallyBin(centers, named("withholes", lambda x: x), Average(named("withholes", lambda x: x))), self.withholes)
            self.compare("CentrallyBinAverage holes positive weights", CentrallyBin(centers, "withholes", Average("withholes")), CentrallyBin(centers, named("withholes", lambda x: x), Average(named("withholes", lambda x: x))), self.withholes)
            self.compare("CentrallyBinAverage holes with weights", CentrallyBin(centers, "withholes", Average("withholes")), CentrallyBin(centers, named("withholes", lambda x: x), Average(named("withholes", lambda x: x))), self.withholes)
            self.compare("CentrallyBinAverage holes with holes", CentrallyBin(centers, "withholes", Average("withholes")), CentrallyBin(centers, named("withholes", lambda x: x), Average(named("withholes", lambda x: x))), self.withholes)
            self.compare("CentrallyBinAverage holes with holes2", CentrallyBin(centers, "withholes", Average("withholes")), CentrallyBin(centers, named("withholes", lambda x: x), Average(named("withholes", lambda x: x))), self.withholes)

    def testCentrallyBinDeviate(self):
        if TestRootCling.ttreeFlat is not None:
            sys.stderr.write("\n")
            centers = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
            self.compare("CentrallyBinDeviate noholes w/o weights", CentrallyBin(centers, "noholes", Deviate("noholes")), CentrallyBin(centers, named("noholes", lambda x: x), Deviate(named("noholes", lambda x: x))), self.noholes)
            self.compare("CentrallyBinDeviate noholes const weights", CentrallyBin(centers, "noholes", Deviate("noholes")), CentrallyBin(centers, named("noholes", lambda x: x), Deviate(named("noholes", lambda x: x))), self.noholes)
            self.compare("CentrallyBinDeviate noholes positive weights", CentrallyBin(centers, "noholes", Deviate("noholes")), CentrallyBin(centers, named("noholes", lambda x: x), Deviate(named("noholes", lambda x: x))), self.noholes)
            self.compare("CentrallyBinDeviate noholes with weights", CentrallyBin(centers, "noholes", Deviate("noholes")), CentrallyBin(centers, named("noholes", lambda x: x), Deviate(named("noholes", lambda x: x))), self.noholes)
            self.compare("CentrallyBinDeviate noholes with holes", CentrallyBin(centers, "noholes", Deviate("noholes")), CentrallyBin(centers, named("noholes", lambda x: x), Deviate(named("noholes", lambda x: x))), self.noholes)
            self.compare("CentrallyBinDeviate holes w/o weights", CentrallyBin(centers, "withholes", Deviate("withholes")), CentrallyBin(centers, named("withholes", lambda x: x), Deviate(named("withholes", lambda x: x))), self.withholes)
            self.compare("CentrallyBinDeviate holes const weights", CentrallyBin(centers, "withholes", Deviate("withholes")), CentrallyBin(centers, named("withholes", lambda x: x), Deviate(named("withholes", lambda x: x))), self.withholes)
            self.compare("CentrallyBinDeviate holes positive weights", CentrallyBin(centers, "withholes", Deviate("withholes")), CentrallyBin(centers, named("withholes", lambda x: x), Deviate(named("withholes", lambda x: x))), self.withholes)
            self.compare("CentrallyBinDeviate holes with weights", CentrallyBin(centers, "withholes", Deviate("withholes")), CentrallyBin(centers, named("withholes", lambda x: x), Deviate(named("withholes", lambda x: x))), self.withholes)
            self.compare("CentrallyBinDeviate holes with holes", CentrallyBin(centers, "withholes", Deviate("withholes")), CentrallyBin(centers, named("withholes", lambda x: x), Deviate(named("withholes", lambda x: x))), self.withholes)
            self.compare("CentrallyBinDeviate holes with holes2", CentrallyBin(centers, "withholes", Deviate("withholes")), CentrallyBin(centers, named("withholes", lambda x: x), Deviate(named("withholes", lambda x: x))), self.withholes)

    # def testCategorize(self):
    #     if TestRootCling.ttreeFlat is not None:
    #         sys.stderr.write("\n")
    #         self.compare("Categorize noholes w/o weights", Categorize(lambda x: numpy.array(numpy.floor(x["noholes"]), dtype="<U5")), self.data, Categorize(lambda x: x), numpy.array(numpy.floor(self.noholes), dtype="<U5"), 1.0)
    #         self.compare("Categorize noholes const weights", Categorize(lambda x: numpy.array(numpy.floor(x["noholes"]), dtype="<U5")), self.data, Categorize(lambda x: x), numpy.array(numpy.floor(self.noholes), dtype="<U5"), 0.5)
    #         self.compare("Categorize noholes positive weights", Categorize(lambda x: numpy.array(numpy.floor(x["noholes"]), dtype="<U5")), self.data, Categorize(lambda x: x), numpy.array(numpy.floor(self.noholes), dtype="<U5"), self.positive)
    #         self.compare("Categorize noholes with weights", Categorize(lambda x: numpy.array(numpy.floor(x["noholes"]), dtype="<U5")), self.data, Categorize(lambda x: x), numpy.array(numpy.floor(self.noholes), dtype="<U5"), self.noholes)
    #         self.compare("Categorize noholes with holes", Categorize(lambda x: numpy.array(numpy.floor(x["noholes"]), dtype="<U5")), self.data, Categorize(lambda x: x), numpy.array(numpy.floor(self.noholes), dtype="<U5"), self.withholes)
    #         self.compare("Categorize holes w/o weights", Categorize(lambda x: numpy.array(numpy.floor(x["withholes"]), dtype="<U5")), self.data, Categorize(lambda x: x), numpy.array(numpy.floor(self.withholes), dtype="<U5"), 1.0)
    #         self.compare("Categorize holes const weights", Categorize(lambda x: numpy.array(numpy.floor(x["withholes"]), dtype="<U5")), self.data, Categorize(lambda x: x), numpy.array(numpy.floor(self.withholes), dtype="<U5"), 0.5)
    #         self.compare("Categorize holes positive weights", Categorize(lambda x: numpy.array(numpy.floor(x["withholes"]), dtype="<U5")), self.data, Categorize(lambda x: x), numpy.array(numpy.floor(self.withholes), dtype="<U5"), self.positive)
    #         self.compare("Categorize holes with weights", Categorize(lambda x: numpy.array(numpy.floor(x["withholes"]), dtype="<U5")), self.data, Categorize(lambda x: x), numpy.array(numpy.floor(self.withholes), dtype="<U5"), self.noholes)
    #         self.compare("Categorize holes with holes", Categorize(lambda x: numpy.array(numpy.floor(x["withholes"]), dtype="<U5")), self.data, Categorize(lambda x: x), numpy.array(numpy.floor(self.withholes), dtype="<U5"), self.withholes)
    #         self.compare("Categorize holes with holes2", Categorize(lambda x: numpy.array(numpy.floor(x["withholes"]), dtype="<U5")), self.data, Categorize(lambda x: x), numpy.array(numpy.floor(self.withholes), dtype="<U5"), self.withholes)

    # def testCategorizeTrans(self):
    #     if TestRootCling.ttreeFlat is not None:
    #         sys.stderr.write("\n")
    #         self.compare("CategorizeTrans noholes w/o weights", Categorize(lambda x: numpy.array(numpy.floor(x["noholes"]), dtype="<U5"), Count("0.5*weight")), self.data, Categorize(lambda x: x, Count("0.5*weight")), numpy.array(numpy.floor(self.noholes), dtype="<U5"), 1.0)
    #         self.compare("CategorizeTrans noholes const weights", Categorize(lambda x: numpy.array(numpy.floor(x["noholes"]), dtype="<U5"), Count("0.5*weight")), self.data, Categorize(lambda x: x, Count("0.5*weight")), numpy.array(numpy.floor(self.noholes), dtype="<U5"), 0.5)
    #         self.compare("CategorizeTrans noholes positive weights", Categorize(lambda x: numpy.array(numpy.floor(x["noholes"]), dtype="<U5"), Count("0.5*weight")), self.data, Categorize(lambda x: x, Count("0.5*weight")), numpy.array(numpy.floor(self.noholes), dtype="<U5"), self.positive)
    #         self.compare("CategorizeTrans noholes with weights", Categorize(lambda x: numpy.array(numpy.floor(x["noholes"]), dtype="<U5"), Count("0.5*weight")), self.data, Categorize(lambda x: x, Count("0.5*weight")), numpy.array(numpy.floor(self.noholes), dtype="<U5"), self.noholes)
    #         self.compare("CategorizeTrans noholes with holes", Categorize(lambda x: numpy.array(numpy.floor(x["noholes"]), dtype="<U5"), Count("0.5*weight")), self.data, Categorize(lambda x: x, Count("0.5*weight")), numpy.array(numpy.floor(self.noholes), dtype="<U5"), self.withholes)
    #         self.compare("CategorizeTrans holes w/o weights", Categorize(lambda x: numpy.array(numpy.floor(x["withholes"]), dtype="<U5"), Count("0.5*weight")), self.data, Categorize(lambda x: x, Count("0.5*weight")), numpy.array(numpy.floor(self.withholes), dtype="<U5"), 1.0)
    #         self.compare("CategorizeTrans holes const weights", Categorize(lambda x: numpy.array(numpy.floor(x["withholes"]), dtype="<U5"), Count("0.5*weight")), self.data, Categorize(lambda x: x, Count("0.5*weight")), numpy.array(numpy.floor(self.withholes), dtype="<U5"), 0.5)
    #         self.compare("CategorizeTrans holes positive weights", Categorize(lambda x: numpy.array(numpy.floor(x["withholes"]), dtype="<U5"), Count("0.5*weight")), self.data, Categorize(lambda x: x, Count("0.5*weight")), numpy.array(numpy.floor(self.withholes), dtype="<U5"), self.positive)
    #         self.compare("CategorizeTrans holes with weights", Categorize(lambda x: numpy.array(numpy.floor(x["withholes"]), dtype="<U5"), Count("0.5*weight")), self.data, Categorize(lambda x: x, Count("0.5*weight")), numpy.array(numpy.floor(self.withholes), dtype="<U5"), self.noholes)
    #         self.compare("CategorizeTrans holes with holes", Categorize(lambda x: numpy.array(numpy.floor(x["withholes"]), dtype="<U5"), Count("0.5*weight")), self.data, Categorize(lambda x: x, Count("0.5*weight")), numpy.array(numpy.floor(self.withholes), dtype="<U5"), self.withholes)
    #         self.compare("CategorizeTrans holes with holes2", Categorize(lambda x: numpy.array(numpy.floor(x["withholes"]), dtype="<U5"), Count("0.5*weight")), self.data, Categorize(lambda x: x, Count("0.5*weight")), numpy.array(numpy.floor(self.withholes), dtype="<U5"), self.withholes)

    # def testFractionBin(self):
    #     if TestRootCling.ttreeFlat is not None:
    #         sys.stderr.write("\n")
    #         self.compare("FractionBin noholes w/o weights", Fraction("noholes", Bin(100, -3.0, 3.0, "noholes")), Fraction(named("noholes", lambda x: x), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("FractionBin noholes const weights", Fraction("noholes", Bin(100, -3.0, 3.0, "noholes")), Fraction(named("noholes", lambda x: x), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("FractionBin noholes positive weights", Fraction("noholes", Bin(100, -3.0, 3.0, "noholes")), Fraction(named("noholes", lambda x: x), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("FractionBin noholes with weights", Fraction("noholes", Bin(100, -3.0, 3.0, "noholes")), Fraction(named("noholes", lambda x: x), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("FractionBin noholes with holes", Fraction("noholes", Bin(100, -3.0, 3.0, "noholes")), Fraction(named("noholes", lambda x: x), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("FractionBin holes w/o weights", Fraction("withholes", Bin(100, -3.0, 3.0, "withholes")), Fraction(named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("FractionBin holes const weights", Fraction("withholes", Bin(100, -3.0, 3.0, "withholes")), Fraction(named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("FractionBin holes positive weights", Fraction("withholes", Bin(100, -3.0, 3.0, "withholes")), Fraction(named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("FractionBin holes with weights", Fraction("withholes", Bin(100, -3.0, 3.0, "withholes")), Fraction(named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("FractionBin holes with holes", Fraction("withholes", Bin(100, -3.0, 3.0, "withholes")), Fraction(named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("FractionBin holes with holes2", Fraction("withholes", Bin(100, -3.0, 3.0, "withholes")), Fraction(named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)

    # def testStackBin(self):
    #     if TestRootCling.ttreeFlat is not None:
    #         sys.stderr.write("\n")
    #         cuts = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
    #         self.compare("StackBin noholes w/o weights", Stack(cuts, "noholes", Bin(100, -3.0, 3.0, "noholes")), Stack(cuts, named("noholes", lambda x: x), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("StackBin noholes const weights", Stack(cuts, "noholes", Bin(100, -3.0, 3.0, "noholes")), Stack(cuts, named("noholes", lambda x: x), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("StackBin noholes positive weights", Stack(cuts, "noholes", Bin(100, -3.0, 3.0, "noholes")), Stack(cuts, named("noholes", lambda x: x), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("StackBin noholes with weights", Stack(cuts, "noholes", Bin(100, -3.0, 3.0, "noholes")), Stack(cuts, named("noholes", lambda x: x), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("StackBin noholes with holes", Stack(cuts, "noholes", Bin(100, -3.0, 3.0, "noholes")), Stack(cuts, named("noholes", lambda x: x), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("StackBin holes w/o weights", Stack(cuts, "withholes", Bin(100, -3.0, 3.0, "withholes")), Stack(cuts, named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("StackBin holes const weights", Stack(cuts, "withholes", Bin(100, -3.0, 3.0, "withholes")), Stack(cuts, named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("StackBin holes positive weights", Stack(cuts, "withholes", Bin(100, -3.0, 3.0, "withholes")), Stack(cuts, named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("StackBin holes with weights", Stack(cuts, "withholes", Bin(100, -3.0, 3.0, "withholes")), Stack(cuts, named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("StackBin holes with holes", Stack(cuts, "withholes", Bin(100, -3.0, 3.0, "withholes")), Stack(cuts, named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("StackBin holes with holes2", Stack(cuts, "withholes", Bin(100, -3.0, 3.0, "withholes")), Stack(cuts, named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)

    # def testIrregularlyBinBin(self):
    #     if TestRootCling.ttreeFlat is not None:
    #         sys.stderr.write("\n")
    #         cuts = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
    #         self.compare("IrregularlyBinBin noholes w/o weights", IrregularlyBin(cuts, "noholes", Bin(100, -3.0, 3.0, "noholes")), IrregularlyBin(cuts, named("noholes", lambda x: x), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("IrregularlyBinBin noholes const weights", IrregularlyBin(cuts, "noholes", Bin(100, -3.0, 3.0, "noholes")), IrregularlyBin(cuts, named("noholes", lambda x: x), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("IrregularlyBinBin noholes positive weights", IrregularlyBin(cuts, "noholes", Bin(100, -3.0, 3.0, "noholes")), IrregularlyBin(cuts, named("noholes", lambda x: x), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("IrregularlyBinBin noholes with weights", IrregularlyBin(cuts, "noholes", Bin(100, -3.0, 3.0, "noholes")), IrregularlyBin(cuts, named("noholes", lambda x: x), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("IrregularlyBinBin noholes with holes", IrregularlyBin(cuts, "noholes", Bin(100, -3.0, 3.0, "noholes")), IrregularlyBin(cuts, named("noholes", lambda x: x), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("IrregularlyBinBin holes w/o weights", IrregularlyBin(cuts, "withholes", Bin(100, -3.0, 3.0, "withholes")), IrregularlyBin(cuts, named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("IrregularlyBinBin holes const weights", IrregularlyBin(cuts, "withholes", Bin(100, -3.0, 3.0, "withholes")), IrregularlyBin(cuts, named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("IrregularlyBinBin holes positive weights", IrregularlyBin(cuts, "withholes", Bin(100, -3.0, 3.0, "withholes")), IrregularlyBin(cuts, named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("IrregularlyBinBin holes with weights", IrregularlyBin(cuts, "withholes", Bin(100, -3.0, 3.0, "withholes")), IrregularlyBin(cuts, named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("IrregularlyBinBin holes with holes", IrregularlyBin(cuts, "withholes", Bin(100, -3.0, 3.0, "withholes")), IrregularlyBin(cuts, named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("IrregularlyBinBin holes with holes2", IrregularlyBin(cuts, "withholes", Bin(100, -3.0, 3.0, "withholes")), IrregularlyBin(cuts, named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)

    # def testSelectBin(self):
    #     if TestRootCling.ttreeFlat is not None:
    #         sys.stderr.write("\n")
    #         self.compare("SelectBin noholes w/o weights", Select("noholes", Bin(100, -3.0, 3.0, "noholes")), Select(named("noholes", lambda x: x), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("SelectBin noholes const weights", Select("noholes", Bin(100, -3.0, 3.0, "noholes")), Select(named("noholes", lambda x: x), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("SelectBin noholes positive weights", Select("noholes", Bin(100, -3.0, 3.0, "noholes")), Select(named("noholes", lambda x: x), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("SelectBin noholes with weights", Select("noholes", Bin(100, -3.0, 3.0, "noholes")), Select(named("noholes", lambda x: x), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("SelectBin noholes with holes", Select("noholes", Bin(100, -3.0, 3.0, "noholes")), Select(named("noholes", lambda x: x), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("SelectBin holes w/o weights", Select("withholes", Bin(100, -3.0, 3.0, "withholes")), Select(named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("SelectBin holes const weights", Select("withholes", Bin(100, -3.0, 3.0, "withholes")), Select(named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("SelectBin holes positive weights", Select("withholes", Bin(100, -3.0, 3.0, "withholes")), Select(named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("SelectBin holes with weights", Select("withholes", Bin(100, -3.0, 3.0, "withholes")), Select(named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("SelectBin holes with holes", Select("withholes", Bin(100, -3.0, 3.0, "withholes")), Select(named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("SelectBin holes with holes2", Select("withholes", Bin(100, -3.0, 3.0, "withholes")), Select(named("withholes", lambda x: x), Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)

    # def testLimitBin(self):
    #     if TestRootCling.ttreeFlat is not None:
    #         sys.stderr.write("\n")
    #         self.compare("LimitBin SIZE - 1 noholes w/o weights", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, "noholes"), Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("LimitBin SIZE - 1 noholes const weights", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, "noholes"), Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("LimitBin SIZE - 1 noholes positive weights", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, "noholes"), Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("LimitBin SIZE - 1 noholes with weights", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, "noholes"), Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("LimitBin SIZE - 1 noholes with holes", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, "noholes"), Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("LimitBin SIZE - 1 holes w/o weights", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, "withholes"), Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("LimitBin SIZE - 1 holes const weights", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, "withholes"), Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("LimitBin SIZE - 1 holes positive weights", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, "withholes"), Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("LimitBin SIZE - 1 holes with weights", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, "withholes"), Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("LimitBin SIZE - 1 holes with holes", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, "withholes"), Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("LimitBin SIZE - 1 holes with holes2", Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, "withholes"), Limit(self.SIZE - 1, Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("LimitBin SIZE noholes w/o weights", Limit(self.SIZE, Bin(100, -3.0, 3.0, "noholes"), Limit(self.SIZE, Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("LimitBin SIZE noholes const weights", Limit(self.SIZE, Bin(100, -3.0, 3.0, "noholes"), Limit(self.SIZE, Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("LimitBin SIZE noholes positive weights", Limit(self.SIZE, Bin(100, -3.0, 3.0, "noholes"), Limit(self.SIZE, Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("LimitBin SIZE noholes with weights", Limit(self.SIZE, Bin(100, -3.0, 3.0, "noholes"), Limit(self.SIZE, Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("LimitBin SIZE noholes with holes", Limit(self.SIZE, Bin(100, -3.0, 3.0, "noholes"), Limit(self.SIZE, Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("LimitBin SIZE holes w/o weights", Limit(self.SIZE, Bin(100, -3.0, 3.0, "withholes"), Limit(self.SIZE, Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("LimitBin SIZE holes const weights", Limit(self.SIZE, Bin(100, -3.0, 3.0, "withholes"), Limit(self.SIZE, Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("LimitBin SIZE holes positive weights", Limit(self.SIZE, Bin(100, -3.0, 3.0, "withholes"), Limit(self.SIZE, Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("LimitBin SIZE holes with weights", Limit(self.SIZE, Bin(100, -3.0, 3.0, "withholes"), Limit(self.SIZE, Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("LimitBin SIZE holes with holes", Limit(self.SIZE, Bin(100, -3.0, 3.0, "withholes"), Limit(self.SIZE, Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("LimitBin SIZE holes with holes2", Limit(self.SIZE, Bin(100, -3.0, 3.0, "withholes"), Limit(self.SIZE, Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)

    # def testLabelBin(self):
    #     if TestRootCling.ttreeFlat is not None:
    #         sys.stderr.write("\n")
    #         self.compare("LabelBin noholes w/o weights", Label(x=Bin(100, -3.0, 3.0, "noholes"), Label(x=Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("LabelBin noholes const weights", Label(x=Bin(100, -3.0, 3.0, "noholes"), Label(x=Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("LabelBin noholes positive weights", Label(x=Bin(100, -3.0, 3.0, "noholes"), Label(x=Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("LabelBin noholes with weights", Label(x=Bin(100, -3.0, 3.0, "noholes"), Label(x=Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("LabelBin noholes with holes", Label(x=Bin(100, -3.0, 3.0, "noholes"), Label(x=Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("LabelBin holes w/o weights", Label(x=Bin(100, -3.0, 3.0, "withholes"), Label(x=Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("LabelBin holes const weights", Label(x=Bin(100, -3.0, 3.0, "withholes"), Label(x=Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("LabelBin holes positive weights", Label(x=Bin(100, -3.0, 3.0, "withholes"), Label(x=Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("LabelBin holes with weights", Label(x=Bin(100, -3.0, 3.0, "withholes"), Label(x=Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("LabelBin holes with holes", Label(x=Bin(100, -3.0, 3.0, "withholes"), Label(x=Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("LabelBin holes with holes2", Label(x=Bin(100, -3.0, 3.0, "withholes"), Label(x=Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)

    # def testUntypedLabelBin(self):
    #     if TestRootCling.ttreeFlat is not None:
    #         sys.stderr.write("\n")
    #         self.compare("UntypedLabelBin noholes w/o weights", UntypedLabel(x=Bin(100, -3.0, 3.0, "noholes"), UntypedLabel(x=Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("UntypedLabelBin noholes const weights", UntypedLabel(x=Bin(100, -3.0, 3.0, "noholes"), UntypedLabel(x=Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("UntypedLabelBin noholes positive weights", UntypedLabel(x=Bin(100, -3.0, 3.0, "noholes"), UntypedLabel(x=Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("UntypedLabelBin noholes with weights", UntypedLabel(x=Bin(100, -3.0, 3.0, "noholes"), UntypedLabel(x=Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("UntypedLabelBin noholes with holes", UntypedLabel(x=Bin(100, -3.0, 3.0, "noholes"), UntypedLabel(x=Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("UntypedLabelBin holes w/o weights", UntypedLabel(x=Bin(100, -3.0, 3.0, "withholes"), UntypedLabel(x=Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("UntypedLabelBin holes const weights", UntypedLabel(x=Bin(100, -3.0, 3.0, "withholes"), UntypedLabel(x=Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("UntypedLabelBin holes positive weights", UntypedLabel(x=Bin(100, -3.0, 3.0, "withholes"), UntypedLabel(x=Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("UntypedLabelBin holes with weights", UntypedLabel(x=Bin(100, -3.0, 3.0, "withholes"), UntypedLabel(x=Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("UntypedLabelBin holes with holes", UntypedLabel(x=Bin(100, -3.0, 3.0, "withholes"), UntypedLabel(x=Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("UntypedLabelBin holes with holes2", UntypedLabel(x=Bin(100, -3.0, 3.0, "withholes"), UntypedLabel(x=Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)

    # def testIndexBin(self):
    #     if TestRootCling.ttreeFlat is not None:
    #         sys.stderr.write("\n")
    #         self.compare("IndexBin noholes w/o weights", Index(Bin(100, -3.0, 3.0, "noholes"), Index(Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("IndexBin noholes const weights", Index(Bin(100, -3.0, 3.0, "noholes"), Index(Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("IndexBin noholes positive weights", Index(Bin(100, -3.0, 3.0, "noholes"), Index(Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("IndexBin noholes with weights", Index(Bin(100, -3.0, 3.0, "noholes"), Index(Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("IndexBin noholes with holes", Index(Bin(100, -3.0, 3.0, "noholes"), Index(Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("IndexBin holes w/o weights", Index(Bin(100, -3.0, 3.0, "withholes"), Index(Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("IndexBin holes const weights", Index(Bin(100, -3.0, 3.0, "withholes"), Index(Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("IndexBin holes positive weights", Index(Bin(100, -3.0, 3.0, "withholes"), Index(Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("IndexBin holes with weights", Index(Bin(100, -3.0, 3.0, "withholes"), Index(Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("IndexBin holes with holes", Index(Bin(100, -3.0, 3.0, "withholes"), Index(Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("IndexBin holes with holes2", Index(Bin(100, -3.0, 3.0, "withholes"), Index(Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)

    # def testBranchBin(self):
    #     if TestRootCling.ttreeFlat is not None:
    #         sys.stderr.write("\n")
    #         self.compare("BranchBin noholes w/o weights", Branch(Bin(100, -3.0, 3.0, "noholes"), Branch(Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("BranchBin noholes const weights", Branch(Bin(100, -3.0, 3.0, "noholes"), Branch(Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("BranchBin noholes positive weights", Branch(Bin(100, -3.0, 3.0, "noholes"), Branch(Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("BranchBin noholes with weights", Branch(Bin(100, -3.0, 3.0, "noholes"), Branch(Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("BranchBin noholes with holes", Branch(Bin(100, -3.0, 3.0, "noholes"), Branch(Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
    #         self.compare("BranchBin holes w/o weights", Branch(Bin(100, -3.0, 3.0, "withholes"), Branch(Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("BranchBin holes const weights", Branch(Bin(100, -3.0, 3.0, "withholes"), Branch(Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("BranchBin holes positive weights", Branch(Bin(100, -3.0, 3.0, "withholes"), Branch(Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("BranchBin holes with weights", Branch(Bin(100, -3.0, 3.0, "withholes"), Branch(Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("BranchBin holes with holes", Branch(Bin(100, -3.0, 3.0, "withholes"), Branch(Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
    #         self.compare("BranchBin holes with holes2", Branch(Bin(100, -3.0, 3.0, "withholes"), Branch(Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)

    # def testBag(self):
    #     if TestRootCling.ttreeFlat is not None:
    #         sys.stderr.write("\n")
    #         self.compare("Bag noholes w/o weights", Bag("noholes"), Bag(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Bag noholes const weights", Bag("noholes"), Bag(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Bag noholes positive weights", Bag("noholes"), Bag(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Bag noholes with weights", Bag("noholes"), Bag(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Bag noholes with holes", Bag("noholes"), Bag(named("noholes", lambda x: x)), self.noholes)
    #         self.compare("Bag holes w/o weights", Bag("withholes"), Bag(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Bag holes const weights", Bag("withholes"), Bag(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Bag holes positive weights", Bag("withholes"), Bag(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Bag holes with weights", Bag("withholes"), Bag(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Bag holes with holes", Bag("withholes"), Bag(named("withholes", lambda x: x)), self.withholes)
    #         self.compare("Bag holes with holes2", Bag("withholes"), Bag(named("withholes", lambda x: x)), self.withholes)
