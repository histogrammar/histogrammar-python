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
import random
import sys
import time
import unittest
from contextlib import suppress

import numpy as np

import histogrammar as hg
from histogrammar import util
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
from histogrammar.primitives.minmax import Maximize, Minimize
from histogrammar.primitives.select import Select
from histogrammar.primitives.sparselybin import SparselyBin
from histogrammar.primitives.stack import Stack
from histogrammar.primitives.sum import Sum
from histogrammar.util import xrange

tolerance = 1e-12
util.relativeTolerance = tolerance
util.absoluteTolerance = tolerance


class Pandas:
    def __enter__(self):
        try:
            import pandas  # noqa

            return pandas
        except ImportError:
            return None

    def __exit__(self, exc_type, exc_value, traceback):
        with suppress(ImportError):
            import pandas  # noqa


def makeSamples(SIZE, HOLES):
    empty = np.array([], dtype=float)

    rand = random.Random(12345)

    positive = np.array([abs(rand.gauss(0, 1)) + 1e-12 for i in xrange(SIZE)])
    assert all(x > 0.0 for x in positive)

    boolean = positive > 1.5

    noholes = np.array([rand.gauss(0, 1) for i in xrange(SIZE)])

    withholes = np.array([rand.gauss(0, 1) for i in xrange(SIZE)])
    for i in xrange(HOLES):
        withholes[rand.randint(0, SIZE)] = float("nan")
    for i in xrange(HOLES):
        withholes[rand.randint(0, SIZE)] = float("inf")
    for i in xrange(HOLES):
        withholes[rand.randint(0, SIZE)] = float("-inf")

    withholes2 = np.array([rand.gauss(0, 1) for i in xrange(SIZE)])
    for i in xrange(HOLES):
        withholes2[rand.randint(0, SIZE)] = float("nan")
    for i in xrange(HOLES):
        withholes2[rand.randint(0, SIZE)] = float("inf")
    for i in xrange(HOLES):
        withholes2[rand.randint(0, SIZE)] = float("-inf")

    return {
        "empty": empty,
        "positive": positive,
        "boolean": boolean,
        "noholes": noholes,
        "withholes": withholes,
        "withholes2": withholes2,
    }


def to_ns(x):
    """convert timestamp to nanosec since 1970-1-1"""
    import pandas as pd

    return pd.to_datetime(x).value


def unit(x):
    """unit return function"""
    return x


def get_test_histograms1():
    """Get set 1 of test histograms"""
    # dummy dataset with mixed types
    # convert timestamp (col D) to nanosec since 1970-1-1
    df = make_mixed_dataframe()
    df["date"] = df["D"].apply(to_ns)
    df["boolT"] = True
    df["boolF"] = False

    import pandas as pd

    import histogrammar as hg

    # building 1d-, 2d-, and 3d-histogram (iteratively)
    hist1 = hg.Categorize(unit("C"))
    hist2 = hg.Bin(5, 0, 5, unit("A"), value=hist1)
    hist3 = hg.SparselyBin(
        origin=pd.Timestamp("2009-01-01").value,
        binWidth=pd.Timedelta(days=1).value,
        quantity=unit("date"),
        value=hist2,
    )
    # fill them
    hist1.fill.numpy(df)
    hist2.fill.numpy(df)
    hist3.fill.numpy(df)

    return df, hist1, hist2, hist3


def get_test_histograms2():
    """Get set 2 of test histograms"""
    import histogrammar as hg

    # dummy dataset with mixed types
    df = make_mixed_dataframe()

    # building 1d-, 2d-histogram (iteratively)
    hist1 = hg.Categorize(unit("C"))
    hist2 = hg.Bin(5, 0, 5, unit("A"), value=hist1)
    hist3 = hg.Bin(5, 0, 5, unit("A"))
    hist4 = hg.Categorize(unit("C"), value=hist3)

    # fill them
    hist1.fill.numpy(df)
    hist2.fill.numpy(df)
    hist3.fill.numpy(df)
    hist4.fill.numpy(df)

    return df, hist1, hist2, hist3, hist4


def make_mixed_dataframe():
    import pandas as pd
    from pandas.core.indexes.datetimes import bdate_range

    return pd.DataFrame(
        {
            "A": [0.0, 1.0, 2.0, 3.0, 4.0],
            "B": [0.0, 1.0, 0.0, 1.0, 0.0],
            "C": ["foo1", "foo2", "foo3", "foo4", "foo5"],
            "D": bdate_range("1/1/2009", periods=5),
        }
    )


class TestNumpy(unittest.TestCase):
    def runTest(self):
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

    SIZE = 10000
    HOLES = 100
    data = makeSamples(SIZE, HOLES)
    empty = data["empty"]
    positive = data["positive"]
    boolean = data["boolean"]
    noholes = data["noholes"]
    withholes = data["withholes"]
    withholes2 = data["withholes2"]

    def twosigfigs(self, number):
        if number == 0:
            return 0
        return round(number, 1 - int(math.floor(math.log10(number))))

    def compare(self, name, hnp, npdata, hpy, pydata):
        npdata2 = npdata.copy()

        hnp2 = hnp.copy()
        hnp3 = hnp.copy()
        hpy2 = hpy.copy()
        hpy3 = hpy.copy()

        startTime = time.time()
        hnp.fill.numpy(npdata)
        numpyTime = time.time() - startTime
        # protect against zero time.
        numpyTime = max(numpyTime, 1e-10)

        if pydata.dtype != np.str_:
            for key in npdata:
                diff = (
                    (npdata[key] != npdata2[key])
                    & np.bitwise_not(np.isnan(npdata[key]))
                    & np.bitwise_not(np.isnan(npdata2[key]))
                )
                if np.any(diff):
                    msg = (
                        f"npdata has been modified:\n{npdata[key]}\n{npdata2[key]}\n{np.nonzero(diff)}\n"
                        f"{npdata[key][np.nonzero(diff)[0][0]]} vs {npdata2[key][np.nonzero(diff)[0][0]]}"
                    )
                    raise AssertionError(msg)

        hnp2.fill.numpy(npdata)
        hnp3.fill.numpy(npdata)
        hnp3.fill.numpy(npdata)
        assert (hnp + hnp2) == hnp3
        assert (hnp2 + hnp) == hnp3
        assert (hnp + hnp.zero()) == hnp2
        assert (hnp.zero() + hnp) == hnp2

        startTime = time.time()
        for d in pydata:
            dv = str(d) if isinstance(d, np.str_) else float(d)
            hpy.fill(dv)
        pyTime = time.time() - startTime
        # protect against zero time.
        pyTime = max(pyTime, 1e-10)

        for h in [hpy2, hpy3, hpy3]:
            for d in pydata:
                dv = str(d) if isinstance(d, np.str_) else float(d)
                h.fill(dv)

        assert (hpy + hpy) == hpy3
        assert (hpy + hpy2) == hpy3
        assert (hpy2 + hpy) == hpy3
        assert (hpy + hpy.zero()) == hpy2
        assert (hpy.zero() + hpy) == hpy2

        hnpj = json.dumps(hnp.toJson(), sort_keys=True)
        hpyj = json.dumps(hpy.toJson(), sort_keys=True)

        if Factory.fromJson(hnp.toJson()) != Factory.fromJson(hpy.toJson()):
            raise AssertionError(f"\n numpy: {hnpj}\npython: {hpyj}")
        msg = (
            f"{name:45s} | numpy: {numpyTime * 1000:.3f}ms python: {pyTime * 1000:.3f}"
            f"ms = {self.twosigfigs(pyTime / numpyTime):g}X speedup\n"
        )
        sys.stderr.write(msg)

        assert Factory.fromJson((hnp + hnp2).toJson()) == Factory.fromJson((hpy + hpy2).toJson())
        assert Factory.fromJson(hnp3.toJson()) == Factory.fromJson(hpy3.toJson())

    # Warmup: apparently, Numpy does some dynamic optimization that needs to warm up...
    if empty is not None:
        Sum(lambda x: x["empty"]).fill.numpy(data)
        Sum(lambda x: x["empty"]).fill.numpy(data)
        Sum(lambda x: x["empty"]).fill.numpy(data)
        Sum(lambda x: x["empty"]).fill.numpy(data)
        Sum(lambda x: x["empty"]).fill.numpy(data)

    def testSum(self):
        sys.stderr.write("\n")
        self.compare(
            "Sum no data",
            Sum(lambda x: x["empty"]),
            self.data,
            Sum(lambda x: x),
            self.empty,
        )
        self.compare(
            "Sum noholes",
            Sum(lambda x: x["noholes"]),
            self.data,
            Sum(lambda x: x),
            self.noholes,
        )
        self.compare(
            "Sum holes",
            Sum(lambda x: x["withholes"]),
            self.data,
            Sum(lambda x: x),
            self.withholes,
        )

    def testAverage(self):
        sys.stderr.write("\n")
        self.compare(
            "Average no data",
            Average(lambda x: x["empty"]),
            self.data,
            Average(lambda x: x),
            self.empty,
        )
        self.compare(
            "Average noholes",
            Average(lambda x: x["noholes"]),
            self.data,
            Average(lambda x: x),
            self.noholes,
        )
        self.compare(
            "Average holes",
            Average(lambda x: x["withholes"]),
            self.data,
            Average(lambda x: x),
            self.withholes,
        )

    def testDeviate(self):
        sys.stderr.write("\n")
        self.compare(
            "Deviate no data",
            Deviate(lambda x: x["empty"]),
            self.data,
            Deviate(lambda x: x),
            self.empty,
        )
        self.compare(
            "Deviate noholes",
            Deviate(lambda x: x["noholes"]),
            self.data,
            Deviate(lambda x: x),
            self.noholes,
        )
        self.compare(
            "Deviate holes",
            Deviate(lambda x: x["withholes"]),
            self.data,
            Deviate(lambda x: x),
            self.withholes,
        )

    def testMinimize(self):
        sys.stderr.write("\n")
        self.compare(
            "Minimize no data",
            Minimize(lambda x: x["empty"]),
            self.data,
            Minimize(lambda x: x),
            self.empty,
        )
        self.compare(
            "Minimize noholes",
            Minimize(lambda x: x["noholes"]),
            self.data,
            Minimize(lambda x: x),
            self.noholes,
        )
        self.compare(
            "Minimize holes",
            Minimize(lambda x: x["withholes"]),
            self.data,
            Minimize(lambda x: x),
            self.withholes,
        )

    def testMaximize(self):
        sys.stderr.write("\n")
        self.compare(
            "Maximize no data",
            Maximize(lambda x: x["empty"]),
            self.data,
            Maximize(lambda x: x),
            self.empty,
        )
        self.compare(
            "Maximize noholes",
            Maximize(lambda x: x["noholes"]),
            self.data,
            Maximize(lambda x: x),
            self.noholes,
        )
        self.compare(
            "Maximize holes",
            Maximize(lambda x: x["withholes"]),
            self.data,
            Maximize(lambda x: x),
            self.withholes,
        )

    def testBin(self):
        sys.stderr.write("\n")
        for bins in [10, 100]:
            self.compare(
                f"Bin ({bins} bins) no data",
                Bin(bins, -3.0, 3.0, lambda x: x["empty"]),
                self.data,
                Bin(bins, -3.0, 3.0, lambda x: x),
                self.empty,
            )
            self.compare(
                f"Bin ({bins} bins) noholes",
                Bin(bins, -3.0, 3.0, lambda x: x["noholes"]),
                self.data,
                Bin(bins, -3.0, 3.0, lambda x: x),
                self.noholes,
            )
            self.compare(
                f"Bin ({bins} bins) holes",
                Bin(bins, -3.0, 3.0, lambda x: x["withholes"]),
                self.data,
                Bin(bins, -3.0, 3.0, lambda x: x),
                self.withholes,
            )

    def testBinTrans(self):
        sys.stderr.write("\n")
        for bins in [10, 100]:
            self.compare(
                f"BinTrans ({bins} bins) no data",
                Bin(bins, -3.0, 3.0, lambda x: x["empty"], Count(lambda x: 0.5 * x)),
                self.data,
                Bin(bins, -3.0, 3.0, lambda x: x, Count(lambda x: 0.5 * x)),
                self.empty,
            )
            self.compare(
                f"BinTrans ({bins} bins) noholes",
                Bin(
                    bins,
                    -3.0,
                    3.0,
                    lambda x: x["noholes"],
                    Count(lambda x: 0.5 * x),
                ),
                self.data,
                Bin(bins, -3.0, 3.0, lambda x: x, Count(lambda x: 0.5 * x)),
                self.noholes,
            )
            self.compare(
                f"BinTrans ({bins} bins) holes",
                Bin(
                    bins,
                    -3.0,
                    3.0,
                    lambda x: x["withholes"],
                    Count(lambda x: 0.5 * x),
                ),
                self.data,
                Bin(bins, -3.0, 3.0, lambda x: x, Count(lambda x: 0.5 * x)),
                self.withholes,
            )

    def testBinAverage(self):
        sys.stderr.write("\n")
        for bins in [10, 100]:
            self.compare(
                f"BinAverage ({bins} bins) no data",
                Bin(
                    bins,
                    -3.0,
                    3.0,
                    lambda x: x["empty"],
                    Average(lambda x: x["empty"]),
                ),
                self.data,
                Bin(bins, -3.0, 3.0, lambda x: x, Average(lambda x: x)),
                self.empty,
            )
            self.compare(
                f"BinAverage ({bins} bins) noholes",
                Bin(
                    bins,
                    -3.0,
                    3.0,
                    lambda x: x["noholes"],
                    Average(lambda x: x["noholes"]),
                ),
                self.data,
                Bin(bins, -3.0, 3.0, lambda x: x, Average(lambda x: x)),
                self.noholes,
            )
            self.compare(
                f"BinAverage ({bins} bins) holes",
                Bin(
                    bins,
                    -3.0,
                    3.0,
                    lambda x: x["withholes"],
                    Average(lambda x: x["withholes"]),
                ),
                self.data,
                Bin(bins, -3.0, 3.0, lambda x: x, Average(lambda x: x)),
                self.withholes,
            )

    def testBinDeviate(self):
        sys.stderr.write("\n")
        for bins in [10, 100]:
            self.compare(
                f"BinDeviate ({bins} bins) no data",
                Bin(
                    bins,
                    -3.0,
                    3.0,
                    lambda x: x["empty"],
                    Deviate(lambda x: x["empty"]),
                ),
                self.data,
                Bin(bins, -3.0, 3.0, lambda x: x, Deviate(lambda x: x)),
                self.empty,
            )
            self.compare(
                f"BinDeviate ({bins} bins) noholes",
                Bin(
                    bins,
                    -3.0,
                    3.0,
                    lambda x: x["noholes"],
                    Deviate(lambda x: x["noholes"]),
                ),
                self.data,
                Bin(bins, -3.0, 3.0, lambda x: x, Deviate(lambda x: x)),
                self.noholes,
            )
            self.compare(
                f"BinDeviate ({bins} bins) holes",
                Bin(
                    bins,
                    -3.0,
                    3.0,
                    lambda x: x["withholes"],
                    Deviate(lambda x: x["withholes"]),
                ),
                self.data,
                Bin(bins, -3.0, 3.0, lambda x: x, Deviate(lambda x: x)),
                self.withholes,
            )

    def testSparselyBin(self):
        sys.stderr.write("\n")
        self.compare(
            "SparselyBin no data",
            SparselyBin(0.1, lambda x: x["empty"]),
            self.data,
            SparselyBin(0.1, lambda x: x),
            self.empty,
        )
        self.compare(
            "SparselyBin noholes",
            SparselyBin(0.1, lambda x: x["noholes"]),
            self.data,
            SparselyBin(0.1, lambda x: x),
            self.noholes,
        )
        self.compare(
            "SparselyBin holes",
            SparselyBin(0.1, lambda x: x["withholes"]),
            self.data,
            SparselyBin(0.1, lambda x: x),
            self.withholes,
        )

    def testSparselyBinTrans(self):
        sys.stderr.write("\n")
        self.compare(
            "SparselyBinTrans no data",
            SparselyBin(0.1, lambda x: x["empty"], Count(lambda x: 0.5 * x)),
            self.data,
            SparselyBin(0.1, lambda x: x, Count(lambda x: 0.5 * x)),
            self.empty,
        )
        self.compare(
            "SparselyBinTrans noholes",
            SparselyBin(0.1, lambda x: x["noholes"], Count(lambda x: 0.5 * x)),
            self.data,
            SparselyBin(0.1, lambda x: x, Count(lambda x: 0.5 * x)),
            self.noholes,
        )
        self.compare(
            "SparselyBinTrans holes",
            SparselyBin(0.1, lambda x: x["withholes"], Count(lambda x: 0.5 * x)),
            self.data,
            SparselyBin(0.1, lambda x: x, Count(lambda x: 0.5 * x)),
            self.withholes,
        )

    def testSparselyBinAverage(self):
        sys.stderr.write("\n")
        self.compare(
            "SparselyBinAverage no data",
            SparselyBin(0.1, lambda x: x["empty"], Average(lambda x: x["empty"])),
            self.data,
            SparselyBin(0.1, lambda x: x, Average(lambda x: x)),
            self.empty,
        )
        self.compare(
            "SparselyBinAverage noholes",
            SparselyBin(0.1, lambda x: x["noholes"], Average(lambda x: x["noholes"])),
            self.data,
            SparselyBin(0.1, lambda x: x, Average(lambda x: x)),
            self.noholes,
        )
        self.compare(
            "SparselyBinAverage holes",
            SparselyBin(0.1, lambda x: x["withholes"], Average(lambda x: x["withholes"])),
            self.data,
            SparselyBin(0.1, lambda x: x, Average(lambda x: x)),
            self.withholes,
        )

    def testSparselyBinDeviate(self):
        sys.stderr.write("\n")
        self.compare(
            "SparselyBinDeviate no data",
            SparselyBin(0.1, lambda x: x["empty"], Deviate(lambda x: x["empty"])),
            self.data,
            SparselyBin(0.1, lambda x: x, Deviate(lambda x: x)),
            self.empty,
        )
        self.compare(
            "SparselyBinDeviate noholes",
            SparselyBin(0.1, lambda x: x["noholes"], Deviate(lambda x: x["noholes"])),
            self.data,
            SparselyBin(0.1, lambda x: x, Deviate(lambda x: x)),
            self.noholes,
        )
        self.compare(
            "SparselyBinDeviate holes",
            SparselyBin(0.1, lambda x: x["withholes"], Deviate(lambda x: x["withholes"])),
            self.data,
            SparselyBin(0.1, lambda x: x, Deviate(lambda x: x)),
            self.withholes,
        )

    def testCentrallyBin(self):
        sys.stderr.write("\n")
        centers = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
        self.compare(
            "CentrallyBin no data",
            CentrallyBin(centers, lambda x: x["empty"]),
            self.data,
            CentrallyBin(centers, lambda x: x),
            self.empty,
        )
        self.compare(
            "CentrallyBin noholes",
            CentrallyBin(centers, lambda x: x["noholes"]),
            self.data,
            CentrallyBin(centers, lambda x: x),
            self.noholes,
        )
        self.compare(
            "CentrallyBin holes",
            CentrallyBin(centers, lambda x: x["withholes"]),
            self.data,
            CentrallyBin(centers, lambda x: x),
            self.withholes,
        )

    def testCentrallyBinTrans(self):
        sys.stderr.write("\n")
        centers = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
        self.compare(
            "CentrallyBinTrans no data",
            CentrallyBin(centers, lambda x: x["empty"], Count(lambda x: 0.5 * x)),
            self.data,
            CentrallyBin(centers, lambda x: x, Count(lambda x: 0.5 * x)),
            self.empty,
        )
        self.compare(
            "CentrallyBinTrans noholes",
            CentrallyBin(centers, lambda x: x["noholes"], Count(lambda x: 0.5 * x)),
            self.data,
            CentrallyBin(centers, lambda x: x, Count(lambda x: 0.5 * x)),
            self.noholes,
        )
        self.compare(
            "CentrallyBinTrans holes",
            CentrallyBin(centers, lambda x: x["withholes"], Count(lambda x: 0.5 * x)),
            self.data,
            CentrallyBin(centers, lambda x: x, Count(lambda x: 0.5 * x)),
            self.withholes,
        )

    def testCentrallyBinAverage(self):
        sys.stderr.write("\n")
        centers = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
        self.compare(
            "CentrallyBinAverage no data",
            CentrallyBin(centers, lambda x: x["empty"], Average(lambda x: x["empty"])),
            self.data,
            CentrallyBin(centers, lambda x: x, Average(lambda x: x)),
            self.empty,
        )
        self.compare(
            "CentrallyBinAverage noholes",
            CentrallyBin(centers, lambda x: x["noholes"], Average(lambda x: x["noholes"])),
            self.data,
            CentrallyBin(centers, lambda x: x, Average(lambda x: x)),
            self.noholes,
        )
        self.compare(
            "CentrallyBinAverage holes",
            CentrallyBin(centers, lambda x: x["withholes"], Average(lambda x: x["withholes"])),
            self.data,
            CentrallyBin(centers, lambda x: x, Average(lambda x: x)),
            self.withholes,
        )

    def testCentrallyBinDeviate(self):
        sys.stderr.write("\n")
        centers = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
        self.compare(
            "CentrallyBinDeviate no data",
            CentrallyBin(centers, lambda x: x["empty"], Deviate(lambda x: x["empty"])),
            self.data,
            CentrallyBin(centers, lambda x: x, Deviate(lambda x: x)),
            self.empty,
        )
        self.compare(
            "CentrallyBinDeviate noholes",
            CentrallyBin(centers, lambda x: x["noholes"], Deviate(lambda x: x["noholes"])),
            self.data,
            CentrallyBin(centers, lambda x: x, Deviate(lambda x: x)),
            self.noholes,
        )
        self.compare(
            "CentrallyBinDeviate holes",
            CentrallyBin(centers, lambda x: x["withholes"], Deviate(lambda x: x["withholes"])),
            self.data,
            CentrallyBin(centers, lambda x: x, Deviate(lambda x: x)),
            self.withholes,
        )

    def testCategorize(self):
        sys.stderr.write("\n")
        self.compare(
            "Categorize no data",
            Categorize(lambda x: np.array(np.floor(x["empty"]), dtype="<U5")),
            self.data,
            Categorize(lambda x: x),
            np.array(np.floor(self.empty), dtype="<U5"),
        )
        self.compare(
            "Categorize noholes",
            Categorize(lambda x: np.array(np.floor(x["noholes"]), dtype="<U5")),
            self.data,
            Categorize(lambda x: x),
            np.array(np.floor(self.noholes), dtype="<U5"),
        )
        self.compare(
            "Categorize holes",
            Categorize(lambda x: np.array(np.floor(x["withholes"]), dtype="<U5")),
            self.data,
            Categorize(lambda x: x),
            np.array(np.floor(self.withholes), dtype="<U5"),
        )

    def testCategorizeTrans(self):
        sys.stderr.write("\n")
        self.compare(
            "CategorizeTrans no data",
            Categorize(
                lambda x: np.array(np.floor(x["empty"]), dtype="<U5"),
                Count(lambda x: 0.5 * x),
            ),
            self.data,
            Categorize(lambda x: x, Count(lambda x: 0.5 * x)),
            np.array(np.floor(self.empty), dtype="<U5"),
        )
        self.compare(
            "CategorizeTrans noholes",
            Categorize(
                lambda x: np.array(np.floor(x["noholes"]), dtype="<U5"),
                Count(lambda x: 0.5 * x),
            ),
            self.data,
            Categorize(lambda x: x, Count(lambda x: 0.5 * x)),
            np.array(np.floor(self.noholes), dtype="<U5"),
        )
        self.compare(
            "CategorizeTrans holes",
            Categorize(
                lambda x: np.array(np.floor(x["withholes"]), dtype="<U5"),
                Count(lambda x: 0.5 * x),
            ),
            self.data,
            Categorize(lambda x: x, Count(lambda x: 0.5 * x)),
            np.array(np.floor(self.withholes), dtype="<U5"),
        )

    def testFractionBin(self):
        sys.stderr.write("\n")
        self.compare(
            "FractionBin no data",
            Fraction(lambda x: x["empty"], Bin(100, -3.0, 3.0, lambda x: x["empty"])),
            self.data,
            Fraction(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)),
            self.empty,
        )
        self.compare(
            "FractionBin noholes",
            Fraction(lambda x: x["noholes"], Bin(100, -3.0, 3.0, lambda x: x["noholes"])),
            self.data,
            Fraction(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)),
            self.noholes,
        )
        self.compare(
            "FractionBin holes",
            Fraction(
                lambda x: x["withholes"],
                Bin(100, -3.0, 3.0, lambda x: x["withholes"]),
            ),
            self.data,
            Fraction(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)),
            self.withholes,
        )

    def testStackBin(self):
        sys.stderr.write("\n")
        cuts = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
        self.compare(
            "StackBin no data",
            Stack(
                cuts,
                lambda x: x["empty"],
                Bin(100, -3.0, 3.0, lambda x: x["empty"]),
            ),
            self.data,
            Stack(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)),
            self.empty,
        )
        self.compare(
            "StackBin noholes",
            Stack(
                cuts,
                lambda x: x["noholes"],
                Bin(100, -3.0, 3.0, lambda x: x["noholes"]),
            ),
            self.data,
            Stack(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)),
            self.noholes,
        )
        self.compare(
            "StackBin holes",
            Stack(
                cuts,
                lambda x: x["withholes"],
                Bin(100, -3.0, 3.0, lambda x: x["withholes"]),
            ),
            self.data,
            Stack(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)),
            self.withholes,
        )

    def testIrregularlyBinBin(self):
        sys.stderr.write("\n")
        cuts = [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
        self.compare(
            "IrregularlyBinBin no data",
            IrregularlyBin(
                cuts,
                lambda x: x["empty"],
                Bin(100, -3.0, 3.0, lambda x: x["empty"]),
            ),
            self.data,
            IrregularlyBin(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)),
            self.empty,
        )
        self.compare(
            "IrregularlyBinBin noholes",
            IrregularlyBin(
                cuts,
                lambda x: x["noholes"],
                Bin(100, -3.0, 3.0, lambda x: x["noholes"]),
            ),
            self.data,
            IrregularlyBin(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)),
            self.noholes,
        )
        self.compare(
            "IrregularlyBinBin holes",
            IrregularlyBin(
                cuts,
                lambda x: x["withholes"],
                Bin(100, -3.0, 3.0, lambda x: x["withholes"]),
            ),
            self.data,
            IrregularlyBin(cuts, lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)),
            self.withholes,
        )

    def testSelectBin(self):
        sys.stderr.write("\n")
        self.compare(
            "SelectBin no data",
            Select(lambda x: x["empty"], Bin(100, -3.0, 3.0, lambda x: x["empty"])),
            self.data,
            Select(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)),
            self.empty,
        )
        self.compare(
            "SelectBin noholes",
            Select(lambda x: x["noholes"], Bin(100, -3.0, 3.0, lambda x: x["noholes"])),
            self.data,
            Select(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)),
            self.noholes,
        )
        self.compare(
            "SelectBin holes",
            Select(
                lambda x: x["withholes"],
                Bin(100, -3.0, 3.0, lambda x: x["withholes"]),
            ),
            self.data,
            Select(lambda x: x, Bin(100, -3.0, 3.0, lambda x: x)),
            self.withholes,
        )

    def testLabelBin(self):
        sys.stderr.write("\n")
        self.compare(
            "LabelBin no data",
            Label(x=Bin(100, -3.0, 3.0, lambda x: x["empty"])),
            self.data,
            Label(x=Bin(100, -3.0, 3.0, lambda x: x)),
            self.empty,
        )
        self.compare(
            "LabelBin noholes",
            Label(x=Bin(100, -3.0, 3.0, lambda x: x["noholes"])),
            self.data,
            Label(x=Bin(100, -3.0, 3.0, lambda x: x)),
            self.noholes,
        )
        self.compare(
            "LabelBin holes",
            Label(x=Bin(100, -3.0, 3.0, lambda x: x["withholes"])),
            self.data,
            Label(x=Bin(100, -3.0, 3.0, lambda x: x)),
            self.withholes,
        )

    def testUntypedLabelBin(self):
        sys.stderr.write("\n")
        self.compare(
            "UntypedLabelBin no data",
            UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x["empty"])),
            self.data,
            UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x)),
            self.empty,
        )
        self.compare(
            "UntypedLabelBin noholes",
            UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x["noholes"])),
            self.data,
            UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x)),
            self.noholes,
        )
        self.compare(
            "UntypedLabelBin holes",
            UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x["withholes"])),
            self.data,
            UntypedLabel(x=Bin(100, -3.0, 3.0, lambda x: x)),
            self.withholes,
        )

    def testIndexBin(self):
        sys.stderr.write("\n")
        self.compare(
            "IndexBin no data",
            Index(Bin(100, -3.0, 3.0, lambda x: x["empty"])),
            self.data,
            Index(Bin(100, -3.0, 3.0, lambda x: x)),
            self.empty,
        )
        self.compare(
            "IndexBin noholes",
            Index(Bin(100, -3.0, 3.0, lambda x: x["noholes"])),
            self.data,
            Index(Bin(100, -3.0, 3.0, lambda x: x)),
            self.noholes,
        )
        self.compare(
            "IndexBin holes",
            Index(Bin(100, -3.0, 3.0, lambda x: x["withholes"])),
            self.data,
            Index(Bin(100, -3.0, 3.0, lambda x: x)),
            self.withholes,
        )

    def testBranchBin(self):
        sys.stderr.write("\n")
        self.compare(
            "BranchBin no data",
            Branch(Bin(100, -3.0, 3.0, lambda x: x["empty"])),
            self.data,
            Branch(Bin(100, -3.0, 3.0, lambda x: x)),
            self.empty,
        )
        self.compare(
            "BranchBin noholes",
            Branch(Bin(100, -3.0, 3.0, lambda x: x["noholes"])),
            self.data,
            Branch(Bin(100, -3.0, 3.0, lambda x: x)),
            self.noholes,
        )
        self.compare(
            "BranchBin holes",
            Branch(Bin(100, -3.0, 3.0, lambda x: x["withholes"])),
            self.data,
            Branch(Bin(100, -3.0, 3.0, lambda x: x)),
            self.withholes,
        )

    def testBag(self):
        sys.stderr.write("\n")
        self.compare(
            "Bag no data",
            Bag(lambda x: x["empty"], "N"),
            self.data,
            Bag(lambda x: x, "N"),
            self.empty,
        )
        self.compare(
            "Bag noholes",
            Bag(lambda x: x["noholes"], "N"),
            self.data,
            Bag(lambda x: x, "N"),
            self.noholes,
        )
        self.compare(
            "Bag holes",
            Bag(lambda x: x["withholes"], "N"),
            self.data,
            Bag(lambda x: x, "N"),
            self.withholes,
        )


class TestPandas(unittest.TestCase):
    def runTest(self):
        self.test_n_dim()
        self.test_n_bins()
        self.test_num_bins()
        self.test_most_probable_value()
        self.test_bin_labels()
        self.test_bin_centers()
        self.test_bin_entries()
        self.test_bin_edges()
        self.test_bin_width()
        self.test_irregular()
        self.test_centrally()

    def test_n_dim(self):
        """Test dimension assigned to a histogram"""
        with Pandas() as pd:  # noqa
            if pd is None:
                return
            sys.stderr.write("\n")

            df, hist1, hist2, hist3 = get_test_histograms1()
            hist0 = hg.Count()

            assert hist0.n_dim == 0
            assert hist1.n_dim == 1
            assert hist2.n_dim == 2
            assert hist3.n_dim == 3

    def test_datatype(self):
        """Test dimension assigned to a histogram"""
        with Pandas() as pd:  # noqa
            if pd is None:
                return
            sys.stderr.write("\n")

            df, hist1, hist2, hist3 = get_test_histograms1()

            assert hist1.datatype == str
            np.testing.assert_array_equal(hist2.datatype, [np.number, str])
            np.testing.assert_array_equal(hist3.datatype, [np.datetime64, np.number, str])

    def test_n_bins(self):
        """Test getting the number of allocated bins"""
        with Pandas() as pd:  # noqa
            if pd is None:
                return
            sys.stderr.write("\n")

            df, hist1, hist2, hist3 = get_test_histograms1()

            assert hist1.n_bins == 5
            assert hist2.n_bins == 5
            assert hist3.n_bins == 5

    def test_num_bins(self):
        """Test getting the number of bins from lowest to highest bin"""
        with Pandas() as pd:  # noqa
            if pd is None:
                return
            sys.stderr.write("\n")

            df1 = pd.DataFrame({"A": [0, 2, 4, 5, 7, 9, 11, 13, 13, 15]})
            df2 = pd.DataFrame({"A": [2, 4, 4, 6, 8, 7, 10, 14, 17, 19]})

            # building 1d-, 2d-, and 3d-histogram (iteratively)
            hist2 = hg.SparselyBin(origin=0.0, binWidth=1.0, quantity=unit("A"))
            hist3 = hg.SparselyBin(origin=0.0, binWidth=1.0, quantity=unit("A"))
            hist4 = hg.Bin(num=20, low=0.0, high=20.0, quantity=unit("A"))
            hist5 = hg.Bin(num=20, low=0.0, high=20.0, quantity=unit("A"))
            hist6 = hg.Bin(num=201, low=0.0, high=1.005)

            # fill them
            hist2.fill.numpy(df1)
            hist3.fill.numpy(df2)
            hist4.fill.numpy(df1)
            hist5.fill.numpy(df2)

            assert hist2.num_bins() == 16
            assert hist3.num_bins() == 18
            assert hist4.num_bins() == 20
            assert hist5.num_bins() == 20
            assert hist6.num_bins() == 201

            assert hist2.num_bins(low=10, high=25) == 15
            assert hist3.num_bins(low=10, high=25) == 15
            assert hist4.num_bins(low=10, high=25) == 10
            assert hist5.num_bins(low=10, high=25) == 10
            assert hist6.num_bins(low=0.2089, high=0.9333) == 146

            assert hist2.num_bins(low=-10, high=28) == 38
            assert hist3.num_bins(low=-10, high=28) == 38
            assert hist4.num_bins(low=-10, high=28) == 20
            assert hist5.num_bins(low=-10, high=28) == 20
            assert hist6.num_bins(low=0.205, high=0.935) == 146

    def test_most_probable_value(self):
        """Test getting most probable value or label from histogram"""
        with Pandas() as pd:  # noqa
            if pd is None:
                return
            sys.stderr.write("\n")

            df1 = pd.DataFrame(
                {
                    "A": [0, 1, 2, 3, 4, 3, 2, 1, 1, 1],
                    "C": ["f1", "f3", "f4", "f3", "f4", "f2", "f2", "f1", "f3", "f4"],
                }
            )
            df2 = pd.DataFrame(
                {
                    "A": [2, 3, 4, 5, 7, 4, 6, 5, 7, 8],
                    "C": ["f7", "f3", "f5", "f8", "f9", "f2", "f3", "f6", "f7", "f7"],
                }
            )

            # building 1d-, 2d-, and 3d-histogram (iteratively)
            hist0 = hg.Categorize(unit("C"))
            hist1 = hg.Categorize(unit("C"))
            hist2 = hg.SparselyBin(origin=0.0, binWidth=1.0, quantity=unit("A"))
            hist3 = hg.SparselyBin(origin=0.0, binWidth=1.0, quantity=unit("A"))

            # fill them
            hist0.fill.numpy(df1)
            hist1.fill.numpy(df2)
            hist2.fill.numpy(df1)
            hist3.fill.numpy(df2)

            assert hist0.mpv == "f3"
            assert hist1.mpv == "f7"
            assert hist2.mpv == 1.5
            assert hist3.mpv == 4.5

    def test_bin_labels(self):
        """Test getting correct bin-labels from Categorize histograms"""
        with Pandas() as pd:  # noqa
            if pd is None:
                return
            sys.stderr.write("\n")

            df, hist1, hist2, hist3 = get_test_histograms1()

            np.testing.assert_array_equal(hist1.bin_labels(), ["foo1", "foo2", "foo3", "foo4", "foo5"])

    def test_bin_centers(self):
        """Test getting assigned bin-centers for Bin and SparselyBin histograms"""
        with Pandas() as pd:  # noqa
            if pd is None:
                return
            sys.stderr.write("\n")

            df1 = pd.DataFrame({"A": [0, 1, 2, 3, 4, 3, 2, 1, 1, 1]})
            df2 = pd.DataFrame({"A": [2, 3, 4, 5, 7, 4, 6, 5, 7, 8]})

            # histograms
            hist2 = hg.SparselyBin(origin=0.0, binWidth=1.0, quantity=unit("A"))
            hist3 = hg.SparselyBin(origin=0.0, binWidth=1.0, quantity=unit("A"))
            hist4 = hg.Bin(num=10, low=0.0, high=10.0, quantity=unit("A"))
            hist5 = hg.Bin(num=10, low=0.0, high=10.0, quantity=unit("A"))

            # fill them
            hist2.fill.numpy(df1)
            hist3.fill.numpy(df2)
            hist4.fill.numpy(df1)
            hist5.fill.numpy(df2)

            np.testing.assert_array_equal(hist2.bin_centers(), [0.5, 1.5, 2.5, 3.5, 4.5])
            np.testing.assert_array_equal(hist3.bin_centers(), [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
            np.testing.assert_array_equal(
                hist2.bin_centers(low=5, high=15),
                [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5],
            )
            np.testing.assert_array_equal(hist3.bin_centers(), [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
            np.testing.assert_array_equal(hist3.bin_centers(low=2.1, high=5.6), [2.5, 3.5, 4.5, 5.5])

            np.testing.assert_array_equal(hist4.bin_centers(), [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
            np.testing.assert_array_equal(hist5.bin_centers(), [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
            np.testing.assert_array_equal(hist4.bin_centers(low=5, high=15), [5.5, 6.5, 7.5, 8.5, 9.5])
            np.testing.assert_array_equal(
                hist5.bin_centers(low=2.1, high=9.1),
                [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
            )

    def test_bin_entries(self):
        """Test getting the number of bins for all assigned bins"""
        with Pandas() as pd:  # noqa
            if pd is None:
                return
            sys.stderr.write("\n")

            df1 = pd.DataFrame(
                {
                    "A": [0, 1, 2, 3, 4, 3, 2, 1, 1, 1],
                    "C": ["f1", "f3", "f4", "f3", "f4", "f2", "f2", "f1", "f3", "f4"],
                }
            )
            df2 = pd.DataFrame(
                {
                    "A": [2, 3, 4, 5, 7, 4, 6, 5, 7, 8],
                    "C": ["f7", "f3", "f5", "f8", "f9", "f2", "f3", "f6", "f7", "f7"],
                }
            )

            # building 1d-, 2d-, and 3d-histogram (iteratively)
            hist0 = hg.Categorize(unit("C"))
            hist1 = hg.Categorize(unit("C"))
            hist2 = hg.SparselyBin(origin=0.0, binWidth=1.0, quantity=unit("A"))
            hist3 = hg.SparselyBin(origin=0.0, binWidth=1.0, quantity=unit("A"))
            hist4 = hg.Bin(num=10, low=0.0, high=10.0, quantity=unit("A"))
            hist5 = hg.Bin(num=10, low=0.0, high=10.0, quantity=unit("A"))

            # fill them
            hist0.fill.numpy(df1)
            hist1.fill.numpy(df2)
            hist2.fill.numpy(df1)
            hist3.fill.numpy(df2)
            hist4.fill.numpy(df1)
            hist5.fill.numpy(df2)

            labels0 = hist0.bin_labels()
            labels1 = hist1.bin_labels()
            centers2 = hist2.bin_centers()
            centers3 = hist3.bin_centers()
            centers = hist4.bin_centers()

            np.testing.assert_array_equal(hist0.bin_entries(), [2.0, 2.0, 3.0, 3.0])
            np.testing.assert_array_equal(hist1.bin_entries(), [1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 1.0])
            np.testing.assert_array_equal(hist0.bin_entries(labels=labels1), [2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            np.testing.assert_array_equal(hist1.bin_entries(labels=labels0), [0.0, 1.0, 2.0, 0.0])

            np.testing.assert_array_equal(hist2.bin_entries(), [1.0, 4.0, 2.0, 2.0, 1.0])
            np.testing.assert_array_equal(hist3.bin_entries(), [1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0])
            np.testing.assert_array_equal(hist4.bin_entries(), [1.0, 4.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            np.testing.assert_array_equal(hist5.bin_entries(), [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 0.0])

            np.testing.assert_array_equal(hist2.bin_entries(xvalues=centers3), [2.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            np.testing.assert_array_equal(hist3.bin_entries(xvalues=centers2), [0.0, 0.0, 1.0, 1.0, 2.0])
            np.testing.assert_array_equal(
                hist2.bin_entries(xvalues=centers),
                [1.0, 4.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
            np.testing.assert_array_equal(
                hist3.bin_entries(xvalues=centers),
                [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 0.0],
            )

            np.testing.assert_array_equal(
                hist2.bin_entries(low=2.1, high=11.9),
                [2.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
            np.testing.assert_array_equal(hist3.bin_entries(low=1.1, high=5.4), [0.0, 1.0, 1.0, 2.0, 2.0])
            np.testing.assert_array_equal(
                hist4.bin_entries(low=2.1, high=11.9),
                [2.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
            np.testing.assert_array_equal(hist5.bin_entries(low=1.1, high=5.4), [0.0, 1.0, 1.0, 2.0, 2.0])

    def test_bin_edges(self):
        """Test getting the bin edges for requested ranges"""
        with Pandas() as pd:  # noqa
            if pd is None:
                return
            sys.stderr.write("\n")

            df1 = pd.DataFrame({"A": [0, 1, 2, 3, 4, 3, 2, 1, 1, 1]})
            df2 = pd.DataFrame({"A": [2, 3, 4, 5, 7, 4, 6, 5, 7, 8]})

            # building test histograms
            hist2 = hg.SparselyBin(origin=0.0, binWidth=1.0, quantity=unit("A"))
            hist3 = hg.SparselyBin(origin=0.0, binWidth=1.0, quantity=unit("A"))
            hist4 = hg.Bin(num=10, low=0.0, high=10.0, quantity=unit("A"))
            hist5 = hg.Bin(num=10, low=0.0, high=10.0, quantity=unit("A"))
            hist6 = hg.Bin(num=201, low=0.0, high=1.005)

            # fill them
            hist2.fill.numpy(df1)
            hist3.fill.numpy(df2)
            hist4.fill.numpy(df1)
            hist5.fill.numpy(df2)

            np.testing.assert_array_equal(hist2.bin_edges(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
            np.testing.assert_array_equal(hist3.bin_edges(), [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            np.testing.assert_array_equal(
                hist4.bin_edges(),
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            )
            np.testing.assert_array_equal(
                hist5.bin_edges(),
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            )

            np.testing.assert_array_equal(
                hist2.bin_edges(low=2.1, high=11.9),
                [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            )
            np.testing.assert_array_equal(hist3.bin_edges(low=1.1, high=6), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            np.testing.assert_array_equal(
                hist4.bin_edges(low=2.1, high=11.9),
                [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            )
            np.testing.assert_array_equal(hist5.bin_edges(low=1.1, high=5.4), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

            assert len(hist6.bin_edges()) == 202
            assert len(hist6.bin_edges(low=0.2089, high=0.9333)) == 147
            assert len(hist6.bin_edges(low=0.205, high=0.935)) == 147

    def test_bin_width(self):
        """Test getting the bin width of bin and sparselybin histograms"""
        with Pandas() as pd:  # noqa
            if pd is None:
                return
            sys.stderr.write("\n")

            df1 = pd.DataFrame({"A": [0, 1, 2, 3, 4, 3, 2, 1, 1, 1]})

            # building test histograms
            hist2 = hg.SparselyBin(origin=0.0, binWidth=1.0, quantity=unit("A"))
            hist3 = hg.SparselyBin(origin=0.0, binWidth=1.0, quantity=unit("A"))
            hist4 = hg.Bin(num=20, low=0.0, high=10.0, quantity=unit("A"))
            hist5 = hg.Bin(num=20, low=0.0, high=10.0, quantity=unit("A"))

            # fill them
            hist2.fill.numpy(df1)
            hist4.fill.numpy(df1)

            assert hist2.bin_width() == 1.0
            assert hist3.bin_width() == 1.0
            assert hist4.bin_width() == 0.5
            assert hist5.bin_width() == 0.5

    def test_irregular(self):
        """Test numpy functions of irregular histogram"""
        h = hg.IrregularlyBin([0, 10, 20, 40, 100])
        h.fillnumpy([-5, 5, 5, 50, 10, 100, 1000, 50, 50])

        np.testing.assert_array_equal(h.bin_entries(), [1.0, 2.0, 1.0, 0.0, 3.0, 2.0])
        np.testing.assert_array_equal(h.bin_edges(), [float("-inf"), 0.0, 10.0, 20.0, 40.0, 100.0, float("inf")])
        np.testing.assert_array_equal(h.bin_centers(), [float("-inf"), 5.0, 15.0, 30.0, 70.0, float("inf")])
        assert h.num_bins() == 6
        assert h.n_bins == 6
        np.testing.assert_almost_equal(h.mpv, 70.0)

        np.testing.assert_array_equal(h.bin_entries(10, 40), [1.0, 0.0])
        np.testing.assert_array_equal(h.bin_edges(10, 40), [10.0, 20.0, 40.0])
        np.testing.assert_array_equal(h.bin_centers(10, 40), [15.0, 30.0])
        assert h.num_bins(10, 40) == 2

        np.testing.assert_array_equal(h.bin_entries(5, 110), [2.0, 1.0, 0.0, 3.0, 2.0])
        np.testing.assert_array_equal(h.bin_edges(5, 110), [0.0, 10.0, 20.0, 40.0, 100.0, float("inf")])
        np.testing.assert_array_equal(h.bin_centers(5, 110), [5.0, 15.0, 30.0, 70.0, float("inf")])
        assert h.num_bins(5, 110) == 5

    def test_centrally(self):
        """Test numpy functions of centrally histogram"""
        h = hg.CentrallyBin([0, 10, 20, 40, 100])
        h.fillnumpy([-5, 5, 5, 50, 10, 100, 1000, 50, 50])

        np.testing.assert_array_equal(h.bin_entries(), [1.0, 3.0, 0.0, 3.0, 2.0])
        np.testing.assert_array_equal(h.bin_edges(), [float("-inf"), 5.0, 15.0, 30.0, 70.0, float("inf")])
        np.testing.assert_array_equal(h.bin_centers(), [0.0, 10.0, 20.0, 40.0, 100.0])
        assert h.num_bins() == 5
        assert h.n_bins == 5
        np.testing.assert_almost_equal(h.mpv, 10.0)

        np.testing.assert_array_equal(h.bin_entries(10, 40), [3.0, 0.0, 3.0])
        np.testing.assert_array_equal(h.bin_edges(10, 40), [5.0, 15.0, 30.0, 70.0])
        np.testing.assert_array_equal(h.bin_centers(10, 40), [10.0, 20.0, 40.0])
        assert h.num_bins(10, 40) == 3

        np.testing.assert_array_equal(h.bin_entries(5, 70), [3.0, 0.0, 3.0])
        np.testing.assert_array_equal(h.bin_edges(5, 70), [5.0, 15.0, 30.0, 70.0])
        np.testing.assert_array_equal(h.bin_centers(5, 70), [10.0, 20.0, 40.0])
        assert h.num_bins(5, 70) == 3

        np.testing.assert_array_equal(h.bin_entries(5, 110), [3.0, 0.0, 3.0, 2.0])
        np.testing.assert_array_equal(h.bin_edges(5, 110), [5.0, 15.0, 30.0, 70.0, float("inf")])
        np.testing.assert_array_equal(h.bin_centers(5, 110), [10.0, 20.0, 40.0, 100.0])
        assert h.num_bins(5, 110) == 4
