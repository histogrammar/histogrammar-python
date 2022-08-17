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
import sys
# import time
import unittest

from histogrammar.defs import Factory
from histogrammar.primitives.average import Average
from histogrammar.primitives.bag import Bag
from histogrammar.primitives.bin import Bin
from histogrammar.primitives.categorize import Categorize
from histogrammar.primitives.centrallybin import CentrallyBin
from histogrammar.primitives.collection import Branch, Index, Label, UntypedLabel
from histogrammar.primitives.count import Count
from histogrammar.primitives.deviate import Deviate
from histogrammar.primitives.fraction import Fraction
from histogrammar.primitives.irregularlybin import IrregularlyBin
from histogrammar.primitives.minmax import Minimize, Maximize
from histogrammar.primitives.select import Select
from histogrammar.primitives.sparselybin import SparselyBin
from histogrammar.primitives.stack import Stack
from histogrammar.primitives.sum import Sum

from histogrammar import util
from histogrammar.util import xrange, named
import histogrammar.version

tolerance = 1e-12
util.relativeTolerance = tolerance
util.absoluteTolerance = tolerance


class TestRootCling(unittest.TestCase):
    SIZE = 10000

    ttreeFlat = None
    ttreeEvent = None
    tchainFlat = None
    positive = []
    boolean = []
    noholes = []
    withholes = []
    withholes2 = []

    try:
        import ROOT
        ROOT.gInterpreter.AddIncludePath("tests/Event.h")
        ROOT.gInterpreter.ProcessLine(".L tests/Event.cxx")
        tfileFlat = ROOT.TFile("tests/flat.root")
        ttreeFlat = tfileFlat.Get("simple")
        tchainFlat = ROOT.TChain("simple")
        tchainFlat.Add("tests/flat.root")
        tfileBig = ROOT.TFile("tests/big.root")
        ttreeBig = tfileBig.Get("big")
        # tfileEvent = ROOT.TFile("tests/Event.root")
        # ttreeEvent = tfileEvent.Get("T")

        for row in ttreeFlat:
            positive.append(row.positive)
            boolean.append(row.boolean)
            noholes.append(row.noholes)
            withholes.append(row.withholes)
            withholes2.append(row.withholes2)

    except ImportError:
        pass

    def runTest(self):
        self.testTiming()
        self.testAAACount()
        self.testAAASum()
        self.testAAABin()
        self.testAAASelect()
        self.testSum()
        self.testAverage()
        self.testDeviate()
        self.testMinimize()
        self.testMaximize()
        self.testBin()
        self.testBinTrans()
        self.testBinAverage()
        self.testBinDeviate()
        self.testSparselyBin()
        self.testSparselyBinTrans()
        self.testSparselyBinAverage()
        self.testSparselyBinDeviate()
        self.testCentrallyBin()
        self.testCentrallyBinTrans()
        self.testCentrallyBinAverage()
        self.testCentrallyBinDeviate()
        self.testCategorize()
        self.testCategorizeTrans()
        self.testFractionBin()
        self.testStackBin()
        self.testIrregularlyBinBin()
        self.testSelectBin()
        self.testLabelBin()
        self.testUntypedLabelBin()
        self.testIndexBin()
        self.testBranchBin()
        self.testBag()

    # Timing

    def testTiming(self):
        which = TestRootCling.ttreeFlat

        if which is not None:
            which.AddBranchToCache("*", True)
            for row in which:
                pass
            # print which.GetCurrentFile().GetBytesRead(), which.GetCurrentFile().GetReadCalls()
            # print which.PrintCacheStats()
            for row in which:
                pass
            # print which.GetCurrentFile().GetBytesRead(), which.GetCurrentFile().GetReadCalls()
            # print which.PrintCacheStats()

            hg = Select("!boolean", Bin(100, -10, 10, "2 * noholes"))

            # print
            # print "Histogrammar JIT-compilation"

            hg.fill.root(which, 0, 1, debug=False)
            # print time.time() - startTime, which.GetCurrentFile().GetBytesRead(),
            # which.GetCurrentFile().GetReadCalls()

            # print
            # print "Histogrammar running"

            hg.fill.root(which)
            # print time.time() - startTime, which.GetCurrentFile().GetBytesRead(),
            # which.GetCurrentFile().GetReadCalls()

            hg.fill.root(which)
            # print time.time() - startTime, which.GetCurrentFile().GetBytesRead(),
            # which.GetCurrentFile().GetReadCalls()

            hg.fill.root(which)
            # print time.time() - startTime, which.GetCurrentFile().GetBytesRead(),
            # which.GetCurrentFile().GetReadCalls()

            controlTestCode = """class ControlTest {
public:
  TH1D histogram;

  ControlTest() : histogram("control1", "", 100, -10, 10) {}

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

            # print
            # print "ROOT C++ compilation"

            import ROOT
            ROOT.gInterpreter.Declare(controlTestCode)
            controlTest = ROOT.ControlTest()

            controlTest.fillall(which, 0, 1)

            controlTest.fillall(which, -1, -1)

            controlTest.fillall(which, -1, -1)

            controlTest.fillall(which, -1, -1)

            histogram = ROOT.TH1D("control2", "", 100, -10, 10)

            for row in which:
                if not row.boolean:
                    histogram.Fill(2 * row.noholes)
                break

            for row in which:
                if not row.boolean:
                    histogram.Fill(2 * row.noholes)

            for row in which:
                if not row.boolean:
                    histogram.Fill(2 * row.noholes)

            for row in which:
                if not row.boolean:
                    histogram.Fill(2 * row.noholes)

            which.Draw("2 * noholes >>+ control3", "!boolean", "goff")

            which.Draw("2 * noholes >>+ control3", "!boolean", "goff")

            which.Draw("2 * noholes >>+ control3", "!boolean", "goff")

            which.Draw("2 * noholes >>+ control3", "!boolean", "goff")

            import numpy
            table = {"boolean": numpy.empty(which.GetEntries(), dtype=numpy.int32),
                     "noholes": numpy.empty(which.GetEntries(), dtype=numpy.double)}

            for i, row in enumerate(which):
                table["boolean"][i] = row.boolean
                table["noholes"][i] = row.noholes

            hg = Select("np.logical_not(boolean)", Bin(100, -10, 10, "2 * noholes"))

            hg.fill.numpy(table)
            hg.fill.numpy(table)
            hg.fill.numpy(table)
            hg.fill.numpy(table)

            class Row(object):
                __slots__ = ["boolean", "noholes"]

                def __init__(self, boolean, noholes):
                    self.boolean = boolean
                    self.noholes = noholes

            table2 = []
            for i in xrange(len(table["boolean"])):
                table2.append(Row(table["boolean"][i], table["noholes"][i]))

            hg = Select(lambda t: not t.boolean, Bin(100, -10, 10, lambda t: 2 * t.noholes))

            for t in table2:
                hg.fill(t)

            for t in table2:
                hg.fill(t)

            for t in table2:
                hg.fill(t)

            for t in table2:
                hg.fill(t)

    # OriginalCount

    def testAAACount(self):
        if TestRootCling.tchainFlat is not None:
            hg = Count()
            hg.fill.root(TestRootCling.tchainFlat, debug=False)
            self.assertEqual(
                hg.toJson(), {
                    "version": histogrammar.version.specification, "type": "Count", "data": 10000})
            hg.fill.root(TestRootCling.tchainFlat, debug=False)
            self.assertEqual(
                hg.toJson(), {
                    "version": histogrammar.version.specification, "type": "Count", "data": 20000})

            hg = Count("0.5 * weight")
            hg.fill.root(TestRootCling.tchainFlat, debug=False)
            self.assertEqual(
                hg.toJson(), {
                    "version": histogrammar.version.specification, "type": "Count", "data": 5000})
            hg.fill.root(TestRootCling.tchainFlat, debug=False)
            self.assertEqual(
                hg.toJson(), {
                    "version": histogrammar.version.specification, "type": "Count", "data": 10000})

            hg = Count("double twice = weight * 2; twice")
            hg.fill.root(TestRootCling.tchainFlat, debug=False)
            self.assertEqual(
                hg.toJson(), {
                    "version": histogrammar.version.specification, "type": "Count", "data": 20000})
            hg.fill.root(TestRootCling.tchainFlat, debug=False)
            self.assertEqual(
                hg.toJson(), {
                    "version": histogrammar.version.specification, "type": "Count", "data": 40000})

        if TestRootCling.ttreeEvent is not None:
            hg = Count()
            hg.fill.root(TestRootCling.ttreeEvent, debug=False)
            self.assertEqual(
                hg.toJson(), {
                    "version": histogrammar.version.specification, "type": "Count", "data": 1000})
            hg.fill.root(TestRootCling.ttreeEvent, debug=False)
            self.assertEqual(
                hg.toJson(), {
                    "version": histogrammar.version.specification, "type": "Count", "data": 2000})

            hg = Count("0.5 * weight")
            hg.fill.root(TestRootCling.ttreeEvent, debug=False)
            self.assertEqual(hg.toJson(), {"version": histogrammar.version.specification, "type": "Count", "data": 500})
            hg.fill.root(TestRootCling.ttreeEvent, debug=False)
            self.assertEqual(
                hg.toJson(), {
                    "version": histogrammar.version.specification, "type": "Count", "data": 1000})

    # OriginalSum

    def testAAASum(self):
        if TestRootCling.tchainFlat is not None:
            hg = Sum("positive")
            hg.fill.root(TestRootCling.tchainFlat, debug=False)
            self.assertEqual(hg.toJson(), {"version": histogrammar.version.specification, "data": {
                             "sum": 7970.933535083706, "name": "positive", "entries": 10000}, "type": "Sum"})
            hg.fill.root(TestRootCling.tchainFlat, debug=False)
            self.assertEqual(hg.toJson(), {"version": histogrammar.version.specification, "data": {
                             "sum": 2*7970.933535083706, "name": "positive", "entries": 20000}, "type": "Sum"})

            hg = Sum("""2 * t("positive")""")
            hg.fill.root(TestRootCling.tchainFlat, debug=False)
            self.assertEqual(hg.toJson(), {"version": histogrammar.version.specification, "data": {
                             "sum": 2*7970.933535083706, "name": """2 * t("positive")""", "entries": 10000}, "type": "Sum"})
            hg.fill.root(TestRootCling.tchainFlat, debug=False)
            self.assertEqual(hg.toJson(), {"version": histogrammar.version.specification, "data": {
                             "sum": 4*7970.933535083706, "name": """2 * t("positive")""", "entries": 20000}, "type": "Sum"})

            hg = Sum("2 * noholes")
            hg.fill.root(TestRootCling.tchainFlat, debug=False)
            self.assertEqual(hg.toJson(), {"version": histogrammar.version.specification, "data": {
                             "sum": 137.62044119255137, "name": "2 * noholes", "entries": 10000}, "type": "Sum"})
            hg.fill.root(TestRootCling.tchainFlat, debug=False)
            self.assertEqual(hg.toJson(), {"version": histogrammar.version.specification, "data": {
                             "sum": 2*137.62044119255137, "name": "2 * noholes", "entries": 20000}, "type": "Sum"})

            hg = Sum("double twice = 2 * noholes; twice;")
            hg.fill.root(TestRootCling.tchainFlat, debug=False)
            self.assertEqual(hg.toJson(), {"version": histogrammar.version.specification, "data": {
                             "sum": 137.62044119255137, "name": "double twice = 2 * noholes; twice;", "entries": 10000}, "type": "Sum"})
            hg.fill.root(TestRootCling.tchainFlat, debug=False)
            self.assertEqual(hg.toJson(), {"version": histogrammar.version.specification, "data": {
                             "sum": 2*137.62044119255137, "name": "double twice = 2 * noholes; twice;", "entries": 20000}, "type": "Sum"})

            hg = Sum("twice")
            hg.fill.root(TestRootCling.tchainFlat, debug=False, twice="2 * noholes")
            self.assertEqual(hg.toJson(), {"version": histogrammar.version.specification, "data": {
                             "sum": 137.62044119255137, "name": "twice", "entries": 10000}, "type": "Sum"})
            hg.fill.root(TestRootCling.tchainFlat, debug=False, twice="2 * noholes")
            self.assertEqual(hg.toJson(), {"version": histogrammar.version.specification, "data": {
                             "sum": 2*137.62044119255137, "name": "twice", "entries": 20000}, "type": "Sum"})

            hg = Sum("quadruple")
            hg.fill.root(TestRootCling.tchainFlat, debug=False, quadruple="double x = 2 * noholes; x*2")
            self.assertEqual(hg.toJson(), {"version": histogrammar.version.specification, "data": {
                             "sum": 2*137.62044119255137, "name": "quadruple", "entries": 10000}, "type": "Sum"})
            hg.fill.root(TestRootCling.tchainFlat, debug=False, quadruple="double x = 2 * noholes; x*2")
            self.assertEqual(hg.toJson(), {"version": histogrammar.version.specification, "data": {
                             "sum": 4*137.62044119255137, "name": "quadruple", "entries": 20000}, "type": "Sum"})

        if TestRootCling.ttreeEvent is not None:
            hg = Sum("event.GetNtrack()")
            hg.fill.root(TestRootCling.ttreeEvent, debug=False)
            self.assertEqual(hg.toJson(), {"version": histogrammar.version.specification, "data": {
                             "sum": 599640, "name": "event.GetNtrack()", "entries": 1000}, "type": "Sum"})
            hg.fill.root(TestRootCling.ttreeEvent, debug=False)
            self.assertEqual(hg.toJson(), {"version": histogrammar.version.specification, "data": {
                             "sum": 2*599640, "name": "event.GetNtrack()", "entries": 2*1000}, "type": "Sum"})

    # OriginalBin

    def testAAABin(self):
        if TestRootCling.tchainFlat is not None:
            hg = Bin(20, -10, 10, "withholes")
            hg.fill.root(TestRootCling.tchainFlat, debug=False)

            self.assertEqual(hg.toJson(), {"version": histogrammar.version.specification, "data": {
                "nanflow:type": "Count",
                "name": "withholes",
                "nanflow": 96.0,
                "overflow:type": "Count",
                "values:type": "Count",
                "high": 10.0,
                "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 201.0, 1346.0, 3385.0, 3182.0, 1358.0, 211.0, 15.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "low": -10.0,
                "entries": 10000.0,
                "overflow": 99.0,
                "underflow": 96.0,
                "underflow:type": "Count"
            },
                "type": "Bin"})
            hg.fill.root(TestRootCling.tchainFlat, debug=False)
            self.assertEqual(hg.toJson(), {"version": histogrammar.version.specification, "data": {
                "nanflow:type": "Count",
                "name": "withholes",
                "nanflow": 2*96.0,
                "overflow:type": "Count",
                "values:type": "Count",
                "high": 10.0,
                "values": [2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*10.0, 2*201.0, 2*1346.0, 2*3385.0, 2*3182.0, 2*1358.0, 2*211.0, 2*15.0, 2*1.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0],
                "low": -10.0,
                "entries": 2*10000.0,
                "overflow": 2*99.0,
                "underflow": 2*96.0,
                "underflow:type": "Count"
            },
                "type": "Bin"})

            hg = Bin(20, -10, 10, "2 * withholes", Sum("positive"))
            hg.fill.root(TestRootCling.tchainFlat, debug=False)
            self.assertEqual(hg.toJson(), {"version": histogrammar.version.specification, "data": {
                "values:name": "positive",
                "nanflow:type": "Count",
                "name": "2 * withholes",
                "nanflow": 96.0,
                "overflow:type": "Count",
                "values:type": "Sum",
                "high": 10.0,
                "values": [
                    {"sum": 0.0, "entries": 0.0},
                    {"sum": 0.0, "entries": 0.0},
                    {"sum": 0.48081424832344055, "entries": 1.0},
                    {"sum": 10.879940822720528, "entries": 9.0},
                    {"sum": 43.35080977156758, "entries": 54.0},
                    {"sum": 113.69398449920118, "entries": 147.0},
                    {"sum": 349.6867558255326, "entries": 449.0},
                    {"sum": 729.5858678516815, "entries": 897.0},
                    {"sum": 1155.193773361767, "entries": 1451.0},
                    {"sum": 1520.5854493912775, "entries": 1934.0},
                    {"sum": 1436.6912576352042, "entries": 1796.0},
                    {"sum": 1116.2790022112895, "entries": 1386.0},
                    {"sum": 728.2537153647281, "entries": 922.0},
                    {"sum": 353.9190010114107, "entries": 436.0},
                    {"sum": 121.04832566762343, "entries": 158.0},
                    {"sum": 42.87702897598501, "entries": 53.0},
                    {"sum": 8.222344039008021, "entries": 13.0},
                    {"sum": 2.8457946181297302, "entries": 2.0},
                    {"sum": 0.36020421981811523, "entries": 1.0},
                    {"sum": 0.0, "entries": 0.0}
                ],
                "low": -10.0,
                "entries": 10000.0,
                "overflow": 99.0,
                "underflow": 96.0,
                "underflow:type": "Count"
            },
                "type": "Bin"})
            hg.fill.root(TestRootCling.tchainFlat, debug=False)
            self.assertEqual(hg.toJson(), {"version": histogrammar.version.specification, "data": {
                "values:name": "positive",
                "nanflow:type": "Count",
                "name": "2 * withholes",
                "nanflow": 2*96.0,
                "overflow:type": "Count",
                "values:type": "Sum",
                "high": 10.0,
                "values": [
                    {"sum": 2*0.0, "entries": 2*0.0},
                    {"sum": 2*0.0, "entries": 2*0.0},
                    {"sum": 2*0.48081424832344055, "entries": 2*1.0},
                    {"sum": 2*10.879940822720528, "entries": 2*9.0},
                    {"sum": 2*43.35080977156758, "entries": 2*54.0},
                    {"sum": 2*113.69398449920118, "entries": 2*147.0},
                    {"sum": 2*349.6867558255326, "entries": 2*449.0},
                    {"sum": 2*729.5858678516815, "entries": 2*897.0},
                    {"sum": 2*1155.193773361767, "entries": 2*1451.0},
                    {"sum": 2*1520.5854493912775, "entries": 2*1934.0},
                    {"sum": 2*1436.6912576352042, "entries": 2*1796.0},
                    {"sum": 2*1116.2790022112895, "entries": 2*1386.0},
                    {"sum": 2*728.2537153647281, "entries": 2*922.0},
                    {"sum": 2*353.9190010114107, "entries": 2*436.0},
                    {"sum": 2*121.04832566762343, "entries": 2*158.0},
                    {"sum": 2*42.87702897598501, "entries": 2*53.0},
                    {"sum": 2*8.222344039008021, "entries": 2*13.0},
                    {"sum": 2*2.8457946181297302, "entries": 2*2.0},
                    {"sum": 2*0.36020421981811523, "entries": 2*1.0},
                    {"sum": 2*0.0, "entries": 2*0.0}
                ],
                "low": -10.0,
                "entries": 2*10000.0,
                "overflow": 2*99.0,
                "underflow": 2*96.0,
                "underflow:type": "Count"
            },
                "type": "Bin"})

    def testAAASelect(self):
        if TestRootCling.tchainFlat is not None:
            hg = Select("boolean", Bin(20, -10, 10, "noholes"))
            hg.fill.root(TestRootCling.tchainFlat, debug=False)
            self.assertEqual(hg.toJson(), {"version": histogrammar.version.specification, "data": {
                "sub:type": "Bin",
                "data": {
                    "nanflow:type": "Count",
                    "name": "noholes",
                    "nanflow": 0.0,
                    "overflow:type": "Count",
                    "values:type": "Count",
                    "high": 10.0,
                    "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 22.0, 183.0, 425.0, 472.0, 181.0, 29.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "low": -10.0,
                    "entries": 1317.0,
                    "overflow": 0.0,
                    "underflow": 0.0,
                    "underflow:type": "Count"
                },
                "name": "boolean",
                "entries": 10000.0
            },
                "type": "Select"})
            hg.fill.root(TestRootCling.tchainFlat, debug=False)
            self.assertEqual(hg.toJson(), {"version": histogrammar.version.specification, "data": {
                "sub:type": "Bin",
                "data": {
                    "nanflow:type": "Count",
                    "name": "noholes",
                    "nanflow": 2*0.0,
                    "overflow:type": "Count",
                    "values:type": "Count",
                    "high": 10.0,
                    "values": [2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*3.0, 2*22.0, 2*183.0, 2*425.0, 2*472.0, 2*181.0, 2*29.0, 2*2.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0],
                    "low": -10.0,
                    "entries": 2*1317.0,
                    "overflow": 2*0.0,
                    "underflow": 2*0.0,
                    "underflow:type": "Count"
                },
                "name": "boolean",
                "entries": 2*10000.0
            },
                "type": "Select"})

            hg = Select("withholes / 2", Bin(20, -10, 10, "noholes"))
            hg.fill.root(TestRootCling.tchainFlat, debug=False)
            self.assertEqual(hg.toJson(), {"version": histogrammar.version.specification, "data": {
                "sub:type": "Bin",
                "data": {
                    "nanflow:type": "Count",
                    "name": "noholes",
                    "nanflow": 0.0,
                    "overflow:type": "Count",
                    "values:type": "Count",
                    "high": 10.0,
                    "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.7656523132161417, "inf", "inf", "inf", "inf", "inf", 40.84895628585768, 2.824571537630074, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "low": -10.0,
                    "entries": "inf",
                    "overflow": 0.0,
                    "underflow": 0.0,
                    "underflow:type": "Count"
                },
                "name": "withholes / 2",
                "entries": 10000.0
            },
                "type": "Select"})
            hg.fill.root(TestRootCling.tchainFlat, debug=False)
            self.assertEqual(hg.toJson(), {"version": histogrammar.version.specification, "data": {
                "sub:type": "Bin",
                "data": {
                    "nanflow:type": "Count",
                    "name": "noholes",
                    "nanflow": 2*0.0,
                    "overflow:type": "Count",
                    "values:type": "Count",
                    "high": 10.0,
                    "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2*3.7656523132161417, "inf", "inf", "inf", "inf", "inf", 2*40.84895628585768, 2*2.824571537630074, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0, 2*0.0],
                    "low": -10.0,
                    "entries": "inf",
                    "overflow": 2*0.0,
                    "underflow": 2*0.0,
                    "underflow:type": "Count"
                },
                "name": "withholes / 2",
                "entries": 2*10000.0
            },
                "type": "Select"})

    # Tests copied from Numpy

    def twosigfigs(self, number):
        return round(number, 1 - int(math.floor(math.log10(number))))

    def compare(self, name, hrc, hpy, pydata, debug=False):
        sys.stderr.write(name + "\n")

        hrc.fill.root(TestRootCling.tchainFlat, debug=debug)
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

        hrc.fill.root(TestRootCling.tchainFlat, debug=debug)
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

    def testSum(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            self.compare("Sum noholes", Sum("noholes"), Sum(named("noholes", lambda x: x)), self.noholes)
            self.compare("Sum holes", Sum("withholes"), Sum(named("withholes", lambda x: x)), self.withholes)

    def testAverage(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            self.compare("Average noholes", Average("noholes"), Average(named("noholes", lambda x: x)), self.noholes)
            self.compare(
                "Average holes",
                Average("withholes"),
                Average(
                    named(
                        "withholes",
                        lambda x: x)),
                self.withholes)

    def testDeviate(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            self.compare("Deviate noholes", Deviate("noholes"), Deviate(named("noholes", lambda x: x)), self.noholes)
            self.compare(
                "Deviate holes",
                Deviate("withholes"),
                Deviate(
                    named(
                        "withholes",
                        lambda x: x)),
                self.withholes)

    def testMinimize(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            self.compare("Minimize noholes", Minimize("noholes"), Minimize(named("noholes", lambda x: x)), self.noholes)
            self.compare(
                "Minimize holes",
                Minimize("withholes"),
                Minimize(
                    named(
                        "withholes",
                        lambda x: x)),
                self.withholes)

    def testMaximize(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            self.compare("Maximize noholes", Maximize("noholes"), Maximize(named("noholes", lambda x: x)), self.noholes)
            self.compare(
                "Maximize holes",
                Maximize("withholes"),
                Maximize(
                    named(
                        "withholes",
                        lambda x: x)),
                self.withholes)

    def testBin(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            for bins in [10, 100]:
                self.compare("Bin ({0} bins) noholes".format(bins), Bin(bins, -3.0, 3.0, "noholes"),
                             Bin(bins, -3.0, 3.0, named("noholes", lambda x: x)), self.noholes)
                self.compare("Bin ({0} bins) holes".format(bins), Bin(bins, -3.0, 3.0, "withholes"),
                             Bin(bins, -3.0, 3.0, named("withholes", lambda x: x)), self.withholes)

    def testBinTrans(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            for bins in [10, 100]:
                self.compare("BinTrans ({0} bins) noholes".format(bins), Bin(bins, -
                                                                             3.0, 3.0, "noholes", Count("0.5*weight")), Bin(bins, -
                                                                                                                            3.0, 3.0, named("noholes", lambda x: x), Count("0.5*weight")), self.noholes)
                self.compare("BinTrans ({0} bins) holes".format(bins), Bin(bins, -
                                                                           3.0, 3.0, "withholes", Count("0.5*weight")), Bin(bins, -
                                                                                                                            3.0, 3.0, named("withholes", lambda x: x), Count("0.5*weight")), self.withholes)

    def testBinAverage(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            for bins in [10, 100]:
                self.compare("BinAverage ({0} bins) noholes".format(bins), Bin(bins, -3.0, 3.0, "noholes", Average("noholes")), Bin(
                    bins, -3.0, 3.0, named("noholes", lambda x: x), Average(named("noholes", lambda x: x))), self.noholes)
                self.compare("BinAverage ({0} bins) holes".format(bins), Bin(bins, -3.0, 3.0, "withholes", Average("withholes")), Bin(
                    bins, -3.0, 3.0, named("withholes", lambda x: x), Average(named("withholes", lambda x: x))), self.withholes)

    def testBinDeviate(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            for bins in [10, 100]:
                self.compare("BinDeviate ({0} bins) noholes".format(bins), Bin(bins, -3.0, 3.0, "noholes", Deviate("noholes")), Bin(
                    bins, -3.0, 3.0, named("noholes", lambda x: x), Deviate(named("noholes", lambda x: x))), self.noholes)
                self.compare("BinDeviate ({0} bins) holes".format(bins), Bin(bins, -3.0, 3.0, "withholes", Deviate("withholes")), Bin(
                    bins, -3.0, 3.0, named("withholes", lambda x: x), Deviate(named("withholes", lambda x: x))), self.withholes)

    def testSparselyBin(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            self.compare(
                "SparselyBin noholes", SparselyBin(
                    0.1, "noholes"), SparselyBin(
                    0.1, named(
                        "noholes", lambda x: x)), self.noholes)
            self.compare(
                "SparselyBin holes", SparselyBin(
                    0.1, "withholes"), SparselyBin(
                    0.1, named(
                        "withholes", lambda x: x)), self.withholes)

    def testSparselyBinTrans(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            self.compare(
                "SparselyBinTrans noholes",
                SparselyBin(
                    0.1,
                    "noholes",
                    Count("0.5*weight")),
                SparselyBin(
                    0.1,
                    named(
                        "noholes",
                        lambda x: x),
                    Count("0.5*weight")),
                self.noholes)
            self.compare(
                "SparselyBinTrans holes",
                SparselyBin(
                    0.1,
                    "withholes",
                    Count("0.5*weight")),
                SparselyBin(
                    0.1,
                    named(
                        "withholes",
                        lambda x: x),
                    Count("0.5*weight")),
                self.withholes)

    def testSparselyBinAverage(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            self.compare(
                "SparselyBinAverage noholes", SparselyBin(
                    0.1, "noholes", Average("noholes")), SparselyBin(
                    0.1, named(
                        "noholes", lambda x: x), Average(
                        named(
                            "noholes", lambda x: x))), self.noholes)
            self.compare(
                "SparselyBinAverage holes", SparselyBin(
                    0.1, "withholes", Average("withholes")), SparselyBin(
                    0.1, named(
                        "withholes", lambda x: x), Average(
                        named(
                            "withholes", lambda x: x))), self.withholes)

    def testSparselyBinDeviate(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            self.compare(
                "SparselyBinDeviate noholes", SparselyBin(
                    0.1, "noholes", Deviate("noholes")), SparselyBin(
                    0.1, named(
                        "noholes", lambda x: x), Deviate(
                        named(
                            "noholes", lambda x: x))), self.noholes)
            self.compare(
                "SparselyBinDeviate holes", SparselyBin(
                    0.1, "withholes", Deviate("withholes")), SparselyBin(
                    0.1, named(
                        "withholes", lambda x: x), Deviate(
                        named(
                            "withholes", lambda x: x))), self.withholes)

    def testCentrallyBin(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            centers = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
            self.compare(
                "CentrallyBin noholes", CentrallyBin(
                    centers, "noholes"), CentrallyBin(
                    centers, named(
                        "noholes", lambda x: x)), self.noholes)
            self.compare(
                "CentrallyBin holes", CentrallyBin(
                    centers, "withholes"), CentrallyBin(
                    centers, named(
                        "withholes", lambda x: x)), self.withholes)

    def testCentrallyBinTrans(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            centers = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
            self.compare(
                "CentrallyBinTrans noholes",
                CentrallyBin(
                    centers,
                    "noholes",
                    Count("0.5*weight")),
                CentrallyBin(
                    centers,
                    named(
                        "noholes",
                        lambda x: x),
                    Count("0.5*weight")),
                self.noholes)
            self.compare(
                "CentrallyBinTrans holes",
                CentrallyBin(
                    centers,
                    "withholes",
                    Count("0.5*weight")),
                CentrallyBin(
                    centers,
                    named(
                        "withholes",
                        lambda x: x),
                    Count("0.5*weight")),
                self.withholes)

    def testCentrallyBinAverage(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            centers = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
            self.compare(
                "CentrallyBinAverage noholes", CentrallyBin(
                    centers, "noholes", Average("noholes")), CentrallyBin(
                    centers, named(
                        "noholes", lambda x: x), Average(
                        named(
                            "noholes", lambda x: x))), self.noholes)
            self.compare(
                "CentrallyBinAverage holes", CentrallyBin(
                    centers, "withholes", Average("withholes")), CentrallyBin(
                    centers, named(
                        "withholes", lambda x: x), Average(
                        named(
                            "withholes", lambda x: x))), self.withholes)

    def testCentrallyBinDeviate(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            centers = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
            self.compare(
                "CentrallyBinDeviate noholes", CentrallyBin(
                    centers, "noholes", Deviate("noholes")), CentrallyBin(
                    centers, named(
                        "noholes", lambda x: x), Deviate(
                        named(
                            "noholes", lambda x: x))), self.noholes)
            self.compare(
                "CentrallyBinDeviate holes", CentrallyBin(
                    centers, "withholes", Deviate("withholes")), CentrallyBin(
                    centers, named(
                        "withholes", lambda x: x), Deviate(
                        named(
                            "withholes", lambda x: x))), self.withholes)

    def testCategorize(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            self.compare("Categorize noholes", Categorize("to_string((int)floor(noholes))"), Categorize(
                named("to_string((int)floor(noholes))", lambda x: str(int(math.floor(x))))), self.noholes)

    def testCategorizeTrans(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            self.compare("CategorizeTrans noholes",
                         Categorize("to_string((int)floor(noholes))",
                                    Count("0.5*weight")),
                         Categorize(named("to_string((int)floor(noholes))",
                                          lambda x: str(int(math.floor(x)))),
                                    Count(lambda x: 0.5*x)),
                         self.noholes)

    def testFractionBin(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            self.compare("FractionBin noholes", Fraction("noholes", Bin(100, -
                                                                        3.0, 3.0, "noholes")), Fraction(named("noholes", lambda x: x), Bin(100, -
                                                                                                                                           3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("FractionBin holes", Fraction("withholes", Bin(100, -
                                                                        3.0, 3.0, "withholes")), Fraction(named("withholes", lambda x: x), Bin(100, -
                                                                                                                                               3.0, 3.0, named("withholes", lambda x: x))), self.withholes)

    def testStackBin(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            cuts = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
            self.compare("StackBin noholes", Stack(cuts, "noholes", Bin(100, -3.0, 3.0, "noholes")), Stack(cuts,
                                                                                                           named("noholes", lambda x: x), Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("StackBin holes", Stack(cuts, "withholes", Bin(100, -
                                                                        3.0, 3.0, "withholes")), Stack(cuts, named("withholes", lambda x: x), Bin(100, -
                                                                                                                                                  3.0, 3.0, named("withholes", lambda x: x))), self.withholes)

    def testIrregularlyBinBin(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            cuts = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
            self.compare("IrregularlyBinBin noholes", IrregularlyBin(cuts, "noholes", Bin(100, -
                                                                                          3.0, 3.0, "noholes")), IrregularlyBin(cuts, named("noholes", lambda x: x), Bin(100, -
                                                                                                                                                                         3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("IrregularlyBinBin holes", IrregularlyBin(cuts, "withholes", Bin(100, -
                                                                                          3.0, 3.0, "withholes")), IrregularlyBin(cuts, named("withholes", lambda x: x), Bin(100, -
                                                                                                                                                                             3.0, 3.0, named("withholes", lambda x: x))), self.withholes)

    def testSelectBin(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            self.compare("SelectBin noholes weights", Select("noholes", Bin(100, -
                                                                            3.0, 3.0, "noholes")), Select(named("noholes", lambda x: x), Bin(100, -
                                                                                                                                             3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("SelectBin holes", Select("withholes", Bin(100, -
                                                                    3.0, 3.0, "withholes")), Select(named("withholes", lambda x: x), Bin(100, -
                                                                                                                                         3.0, 3.0, named("withholes", lambda x: x))), self.withholes)

    def testLabelBin(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            self.compare("LabelBin single noholes", Label(x=Bin(100, -3.0, 3.0, "noholes")),
                         Label(x=Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("LabelBin identical noholes", Label(x=Bin(100, -
                                                                   3.0, 3.0, "noholes"), y=Bin(100, -
                                                                                               3.0, 3.0, "noholes")), Label(x=Bin(100, -
                                                                                                                                  3.0, 3.0, named("noholes", lambda x: x)), y=Bin(100, -
                                                                                                                                                                                  3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("LabelBin same struct noholes", Label(x=Bin(100, -
                                                                     3.0, 3.0, "noholes"), y=Bin(100, -
                                                                                                 2.0, 2.0, "noholes")), Label(x=Bin(100, -
                                                                                                                                    3.0, 3.0, named("noholes", lambda x: x)), y=Bin(100, -
                                                                                                                                                                                    2.0, 2.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("LabelBin trans noholes", Label(x=Bin(100, -
                                                               3.0, 3.0, "noholes"), y=Bin(100, -
                                                                                           3.0, 3.0, "noholes", Count("0.5*weight"))), Label(x=Bin(100, -
                                                                                                                                                   3.0, 3.0, named("noholes", lambda x: x)), y=Bin(100, -
                                                                                                                                                                                                   3.0, 3.0, named("noholes", lambda x: x), Count(lambda x: 0.5 *
                                                                                                                                                                                                                                                  x))), self.noholes)
            self.compare("LabelBin different structs noholes", Label(x=Bin(100, -
                                                                           3.0, 3.0, "noholes"), y=Bin(50, -
                                                                                                       3.0, 3.0, "noholes")), Label(x=Bin(100, -
                                                                                                                                          3.0, 3.0, named("noholes", lambda x: x)), y=Bin(50, -
                                                                                                                                                                                          3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("LabelBin single holes", Label(x=Bin(100, -3.0, 3.0, "withholes")),
                         Label(x=Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("LabelBin identical holes", Label(x=Bin(100, -
                                                                 3.0, 3.0, "withholes"), y=Bin(100, -
                                                                                               3.0, 3.0, "withholes")), Label(x=Bin(100, -
                                                                                                                                    3.0, 3.0, named("withholes", lambda x: x)), y=Bin(100, -
                                                                                                                                                                                      3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("LabelBin same struct holes", Label(x=Bin(100, -
                                                                   3.0, 3.0, "withholes"), y=Bin(100, -
                                                                                                 2.0, 2.0, "withholes")), Label(x=Bin(100, -
                                                                                                                                      3.0, 3.0, named("withholes", lambda x: x)), y=Bin(100, -
                                                                                                                                                                                        2.0, 2.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("LabelBin trans withholes", Label(x=Bin(100, -
                                                                 3.0, 3.0, "withholes"), y=Bin(100, -
                                                                                               3.0, 3.0, "withholes", Count("0.5*weight"))), Label(x=Bin(100, -
                                                                                                                                                         3.0, 3.0, named("withholes", lambda x: x)), y=Bin(100, -
                                                                                                                                                                                                           3.0, 3.0, named("withholes", lambda x: x), Count(lambda x: 0.5 *
                                                                                                                                                                                                                                                            x))), self.withholes)
            self.compare("LabelBin different structs holes", Label(x=Bin(100, -
                                                                         3.0, 3.0, "withholes"), y=Bin(50, -
                                                                                                       3.0, 3.0, "withholes")), Label(x=Bin(100, -
                                                                                                                                            3.0, 3.0, named("withholes", lambda x: x)), y=Bin(50, -
                                                                                                                                                                                              3.0, 3.0, named("withholes", lambda x: x))), self.withholes)

    def testUntypedLabelBin(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            self.compare("UntypedLabelBin single noholes", UntypedLabel(x=Bin(100, -3.0, 3.0, "noholes")),
                         UntypedLabel(x=Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("UntypedLabelBin identical noholes", UntypedLabel(x=Bin(100, -
                                                                                 3.0, 3.0, "noholes"), y=Bin(100, -
                                                                                                             3.0, 3.0, "noholes")), UntypedLabel(x=Bin(100, -
                                                                                                                                                       3.0, 3.0, named("noholes", lambda x: x)), y=Bin(100, -
                                                                                                                                                                                                       3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("UntypedLabelBin same struct noholes", UntypedLabel(x=Bin(100, -
                                                                                   3.0, 3.0, "noholes"), y=Bin(100, -
                                                                                                               2.0, 2.0, "noholes")), UntypedLabel(x=Bin(100, -
                                                                                                                                                         3.0, 3.0, named("noholes", lambda x: x)), y=Bin(100, -
                                                                                                                                                                                                         2.0, 2.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("UntypedLabelBin trans noholes", UntypedLabel(x=Bin(100, -
                                                                             3.0, 3.0, "noholes"), y=Bin(100, -
                                                                                                         3.0, 3.0, "noholes", Count("0.5*weight"))), UntypedLabel(x=Bin(100, -
                                                                                                                                                                        3.0, 3.0, named("noholes", lambda x: x)), y=Bin(100, -
                                                                                                                                                                                                                        3.0, 3.0, named("noholes", lambda x: x), Count(lambda x: 0.5 *
                                                                                                                                                                                                                                                                       x))), self.noholes)
            self.compare("UntypedLabelBin different structs noholes", UntypedLabel(x=Bin(100, -
                                                                                         3.0, 3.0, "noholes"), y=Bin(50, -
                                                                                                                     3.0, 3.0, "noholes")), UntypedLabel(x=Bin(100, -
                                                                                                                                                               3.0, 3.0, named("noholes", lambda x: x)), y=Bin(50, -
                                                                                                                                                                                                               3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("UntypedLabelBin deeply different structs noholes", UntypedLabel(x=Bin(100, -
                                                                                                3.0, 3.0, "noholes"), y=Bin(100, -
                                                                                                                            3.0, 3.0, "noholes", Sum("noholes"))), UntypedLabel(x=Bin(100, -
                                                                                                                                                                                      3.0, 3.0, named("noholes", lambda x: x)), y=Bin(100, -
                                                                                                                                                                                                                                      3.0, 3.0, named("noholes", lambda x: x), Sum(named("noholes", lambda x: x)))), self.noholes)
            self.compare("UntypedLabelBin single holes", UntypedLabel(x=Bin(100, -3.0, 3.0, "withholes")),
                         UntypedLabel(x=Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("UntypedLabelBin identical holes", UntypedLabel(x=Bin(100, -
                                                                               3.0, 3.0, "withholes"), y=Bin(100, -
                                                                                                             3.0, 3.0, "withholes")), UntypedLabel(x=Bin(100, -
                                                                                                                                                         3.0, 3.0, named("withholes", lambda x: x)), y=Bin(100, -
                                                                                                                                                                                                           3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("UntypedLabelBin same struct holes", UntypedLabel(x=Bin(100, -
                                                                                 3.0, 3.0, "withholes"), y=Bin(100, -
                                                                                                               2.0, 2.0, "withholes")), UntypedLabel(x=Bin(100, -
                                                                                                                                                           3.0, 3.0, named("withholes", lambda x: x)), y=Bin(100, -
                                                                                                                                                                                                             2.0, 2.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("UntypedLabelBin trans withholes", UntypedLabel(x=Bin(100, -
                                                                               3.0, 3.0, "withholes"), y=Bin(100, -
                                                                                                             3.0, 3.0, "withholes", Count("0.5*weight"))), UntypedLabel(x=Bin(100, -
                                                                                                                                                                              3.0, 3.0, named("withholes", lambda x: x)), y=Bin(100, -
                                                                                                                                                                                                                                3.0, 3.0, named("withholes", lambda x: x), Count(lambda x: 0.5 *
                                                                                                                                                                                                                                                                                 x))), self.withholes)
            self.compare("UntypedLabelBin different structs holes", UntypedLabel(x=Bin(100, -
                                                                                       3.0, 3.0, "withholes"), y=Bin(50, -
                                                                                                                     3.0, 3.0, "withholes")), UntypedLabel(x=Bin(100, -
                                                                                                                                                                 3.0, 3.0, named("withholes", lambda x: x)), y=Bin(50, -
                                                                                                                                                                                                                   3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("UntypedLabelBin deeply different structs holes", UntypedLabel(x=Bin(100, -
                                                                                              3.0, 3.0, "withholes"), y=Bin(100, -
                                                                                                                            3.0, 3.0, "withholes", Sum("withholes"))), UntypedLabel(x=Bin(100, -
                                                                                                                                                                                          3.0, 3.0, named("withholes", lambda x: x)), y=Bin(100, -
                                                                                                                                                                                                                                            3.0, 3.0, named("withholes", lambda x: x), Sum(named("withholes", lambda x: x)))), self.withholes)

    def testIndexBin(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            self.compare("IndexBin single noholes", Index(Bin(100, -3.0, 3.0, "noholes")),
                         Index(Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("IndexBin identical noholes", Index(Bin(100, -
                                                                 3.0, 3.0, "noholes"), Bin(100, -
                                                                                           3.0, 3.0, "noholes")), Index(Bin(100, -
                                                                                                                            3.0, 3.0, named("noholes", lambda x: x)), Bin(100, -
                                                                                                                                                                          3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("IndexBin same struct noholes", Index(Bin(100, -
                                                                   3.0, 3.0, "noholes"), Bin(100, -
                                                                                             2.0, 2.0, "noholes")), Index(Bin(100, -
                                                                                                                              3.0, 3.0, named("noholes", lambda x: x)), Bin(100, -
                                                                                                                                                                            2.0, 2.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("IndexBin trans noholes", Index(Bin(100, -
                                                             3.0, 3.0, "noholes"), Bin(100, -
                                                                                       3.0, 3.0, "noholes", Count("0.5*weight"))), Index(Bin(100, -
                                                                                                                                             3.0, 3.0, named("noholes", lambda x: x)), Bin(100, -
                                                                                                                                                                                           3.0, 3.0, named("noholes", lambda x: x), Count(lambda x: 0.5 *
                                                                                                                                                                                                                                          x))), self.noholes)
            self.compare("IndexBin different structs noholes", Index(Bin(100, -
                                                                         3.0, 3.0, "noholes"), Bin(50, -
                                                                                                   3.0, 3.0, "noholes")), Index(Bin(100, -
                                                                                                                                    3.0, 3.0, named("noholes", lambda x: x)), Bin(50, -
                                                                                                                                                                                  3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("IndexBin single holes", Index(Bin(100, -3.0, 3.0, "withholes")),
                         Index(Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("IndexBin identical holes", Index(Bin(100, -
                                                               3.0, 3.0, "withholes"), Bin(100, -
                                                                                           3.0, 3.0, "withholes")), Index(Bin(100, -
                                                                                                                              3.0, 3.0, named("withholes", lambda x: x)), Bin(100, -
                                                                                                                                                                              3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("IndexBin same struct holes", Index(Bin(100, -
                                                                 3.0, 3.0, "withholes"), Bin(100, -
                                                                                             2.0, 2.0, "withholes")), Index(Bin(100, -
                                                                                                                                3.0, 3.0, named("withholes", lambda x: x)), Bin(100, -
                                                                                                                                                                                2.0, 2.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("IndexBin trans withholes", Index(Bin(100, -
                                                               3.0, 3.0, "withholes"), Bin(100, -
                                                                                           3.0, 3.0, "withholes", Count("0.5*weight"))), Index(Bin(100, -
                                                                                                                                                   3.0, 3.0, named("withholes", lambda x: x)), Bin(100, -
                                                                                                                                                                                                   3.0, 3.0, named("withholes", lambda x: x), Count(lambda x: 0.5 *
                                                                                                                                                                                                                                                    x))), self.withholes)
            self.compare("IndexBin different structs holes", Index(Bin(100, -
                                                                       3.0, 3.0, "withholes"), Bin(50, -
                                                                                                   3.0, 3.0, "withholes")), Index(Bin(100, -
                                                                                                                                      3.0, 3.0, named("withholes", lambda x: x)), Bin(50, -
                                                                                                                                                                                      3.0, 3.0, named("withholes", lambda x: x))), self.withholes)

    def testBranchBin(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            self.compare("BranchBin single noholes", Branch(Bin(100, -3.0, 3.0, "noholes")),
                         Branch(Bin(100, -3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("BranchBin identical noholes", Branch(Bin(100, -
                                                                   3.0, 3.0, "noholes"), Bin(100, -
                                                                                             3.0, 3.0, "noholes")), Branch(Bin(100, -
                                                                                                                               3.0, 3.0, named("noholes", lambda x: x)), Bin(100, -
                                                                                                                                                                             3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("BranchBin same struct noholes", Branch(Bin(100, -
                                                                     3.0, 3.0, "noholes"), Bin(100, -
                                                                                               2.0, 2.0, "noholes")), Branch(Bin(100, -
                                                                                                                                 3.0, 3.0, named("noholes", lambda x: x)), Bin(100, -
                                                                                                                                                                               2.0, 2.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("BranchBin trans noholes", Branch(Bin(100, -
                                                               3.0, 3.0, "noholes"), Bin(100, -
                                                                                         3.0, 3.0, "noholes", Count("0.5*weight"))), Branch(Bin(100, -
                                                                                                                                                3.0, 3.0, named("noholes", lambda x: x)), Bin(100, -
                                                                                                                                                                                              3.0, 3.0, named("noholes", lambda x: x), Count(lambda x: 0.5 *
                                                                                                                                                                                                                                             x))), self.noholes)
            self.compare("BranchBin different structs noholes", Branch(Bin(100, -
                                                                           3.0, 3.0, "noholes"), Bin(50, -
                                                                                                     3.0, 3.0, "noholes")), Branch(Bin(100, -
                                                                                                                                       3.0, 3.0, named("noholes", lambda x: x)), Bin(50, -
                                                                                                                                                                                     3.0, 3.0, named("noholes", lambda x: x))), self.noholes)
            self.compare("BranchBin deeply different structs noholes", Branch(Bin(100, -
                                                                                  3.0, 3.0, "noholes"), Bin(100, -
                                                                                                            3.0, 3.0, "noholes", Sum("noholes"))), Branch(Bin(100, -
                                                                                                                                                              3.0, 3.0, named("noholes", lambda x: x)), Bin(100, -
                                                                                                                                                                                                            3.0, 3.0, named("noholes", lambda x: x), Sum(named("noholes", lambda x: x)))), self.noholes)
            self.compare("BranchBin single holes", Branch(Bin(100, -3.0, 3.0, "withholes")),
                         Branch(Bin(100, -3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("BranchBin identical holes", Branch(Bin(100, -
                                                                 3.0, 3.0, "withholes"), Bin(100, -
                                                                                             3.0, 3.0, "withholes")), Branch(Bin(100, -
                                                                                                                                 3.0, 3.0, named("withholes", lambda x: x)), Bin(100, -
                                                                                                                                                                                 3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("BranchBin same struct holes", Branch(Bin(100, -
                                                                   3.0, 3.0, "withholes"), Bin(100, -
                                                                                               2.0, 2.0, "withholes")), Branch(Bin(100, -
                                                                                                                                   3.0, 3.0, named("withholes", lambda x: x)), Bin(100, -
                                                                                                                                                                                   2.0, 2.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("BranchBin trans withholes", Branch(Bin(100, -
                                                                 3.0, 3.0, "withholes"), Bin(100, -
                                                                                             3.0, 3.0, "withholes", Count("0.5*weight"))), Branch(Bin(100, -
                                                                                                                                                      3.0, 3.0, named("withholes", lambda x: x)), Bin(100, -
                                                                                                                                                                                                      3.0, 3.0, named("withholes", lambda x: x), Count(lambda x: 0.5 *
                                                                                                                                                                                                                                                       x))), self.withholes)
            self.compare("BranchBin different structs holes", Branch(Bin(100, -
                                                                         3.0, 3.0, "withholes"), Bin(50, -
                                                                                                     3.0, 3.0, "withholes")), Branch(Bin(100, -
                                                                                                                                         3.0, 3.0, named("withholes", lambda x: x)), Bin(50, -
                                                                                                                                                                                         3.0, 3.0, named("withholes", lambda x: x))), self.withholes)
            self.compare("BranchBin deeply different structs holes", Branch(Bin(100, -
                                                                                3.0, 3.0, "withholes"), Bin(100, -
                                                                                                            3.0, 3.0, "withholes", Sum("withholes"))), Branch(Bin(100, -
                                                                                                                                                                  3.0, 3.0, named("withholes", lambda x: x)), Bin(100, -
                                                                                                                                                                                                                  3.0, 3.0, named("withholes", lambda x: x), Sum(named("withholes", lambda x: x)))), self.withholes)

    def testBag(self):
        if TestRootCling.tchainFlat is not None:
            sys.stderr.write("\n")
            self.compare(
                "Bag numbers noholes", Bag(
                    "noholes", "N"), Bag(
                    named(
                        "noholes", lambda x: x), "N"), self.noholes)
            self.compare(
                "Bag numeric vectors noholes", Bag(
                    "N2(noholes, noholes)", "N2"), Bag(
                    named(
                        "N2(noholes, noholes)", lambda x: [
                            x, x]), "N2"), self.noholes)
            self.compare("Bag strings noholes", Bag("to_string((int)floor(noholes))", "S"), Bag(
                named("to_string((int)floor(noholes))", lambda x: str(int(math.floor(x)))), "S"), self.noholes)
