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

import math
import pickle
import sys
import unittest

from histogrammar import *

class TestOriginal(unittest.TestCase):
    simple = [3.4, 2.2, -1.8, 0.0, 7.3, -4.7, 1.6, 0.0, -3.0, -1.7]

    class Struct(object):
        def __init__(self, x, y, z, w):
            self.bool = x
            self.int = y
            self.double = z
            self.string = w
        def __repr__(self):
            return "Struct({}, {}, {}, {})".format(self.bool, self.int, self.double, self.string)

    struct = [
        Struct(True,  -2,  3.4, "one"),
        Struct(False, -1,  2.2, "two"),
        Struct(True,   0, -1.8, "three"),
        Struct(False,  1,  0.0, "four"),
        Struct(False,  2,  7.3, "five"),
        Struct(False,  3, -4.7, "six"),
        Struct(True,   4,  1.6, "seven"),
        Struct(True,   5,  0.0, "eight"),
        Struct(False,  6, -3.0, "nine"),
        Struct(True,   7, -1.7, "ten"),
        ]

    backward = list(reversed(struct))

    # straightforward mean and variance to complement the Tony Finch calculations used in the module

    @staticmethod
    def mean(x):
        if len(x) == 0:
            return 0.0
        else:
            return sum(x) / len(x)

    @staticmethod
    def meanWeighted(x, w):
        if not any(_ > 0.0 for _ in w):
            return 0.0
        else:
            w = list(w)
            return sum(xi * max(wi, 0.0) for xi, wi in zip(x, w)) / sum(_ for _ in w if _ > 0.0)

    @staticmethod
    def variance(x):
        if len(x) == 0:
            return 0.0
        else:
            return sum(math.pow(_, 2) for _ in x) / len(x) - math.pow(sum(x) / len(x), 2)

    @staticmethod
    def varianceWeighted(x, w):
        if not any(_ > 0.0 for _ in w):
            return 0.0
        else:
            w = list(w)
            return sum(xi**2 * max(wi, 0.0) for xi, wi in zip(x, w)) / sum(_ for _ in w if _ > 0.0) - math.pow(sum(xi * max(wi, 0.0) for xi, wi in zip(x, w)) / sum(_ for _ in w if _ > 0.0), 2)

    @staticmethod
    def mae(x):
        if len(x) == 0:
            return 0.0
        else:
            return sum(map(abs, x)) / len(x)

    @staticmethod
    def maeWeighted(x, w):
        if not any(_ > 0.0 for _ in w):
            return 0.0
        else:
            return sum(abs(xi) * max(wi, 0.0) for xi, wi in zip(x, w)) / sum(_ > 0.0 for _ in w)

    def checkJson(self, x):
        self.assertEqual(x.toJson(), Factory.fromJson(x.toJson()).toJson())

    def checkPickle(self, x):
        self.assertEqual(pickle.loads(pickle.dumps(x)), x)

    def checkName(self, x):
        repr(x)

    def runTest(self):
        pass

    ################################################################ Count

    def testCount(self):
        for i in xrange(11):
            left, right = self.simple[:i], self.simple[i:]

            leftCounting = Count()
            rightCounting = Count()

            for _ in left: leftCounting.fill(_)
            for _ in right: rightCounting.fill(_)

            self.assertEqual(leftCounting.entries, len(left))
            self.assertEqual(rightCounting.entries, len(right))

            finalResult = leftCounting + rightCounting

            self.assertEqual(finalResult.entries, len(self.simple))

            self.checkJson(leftCounting)
            self.checkPickle(leftCounting)
            self.checkName(leftCounting)

    def testCountWithFilter(self):
        for i in xrange(11):
            left, right = self.simple[:i], self.simple[i:]

            leftCounting = Select(named("something", lambda x: x > 0.0), Count())
            rightCounting = Select(named("something", lambda x: x > 0.0), Count())

            for _ in left: leftCounting.fill(_)
            for _ in right: rightCounting.fill(_)

            self.assertEqual(leftCounting.cut.entries, len(list(filter(lambda x: x > 0.0, left))))
            self.assertEqual(rightCounting.cut.entries, len(list(filter(lambda x: x > 0.0, right))))

            finalResult = leftCounting + rightCounting

            self.assertEqual(finalResult.cut.entries, len(list(filter(lambda x: x > 0.0, self.simple))))

            self.checkJson(leftCounting)
            self.checkJson(leftCounting)

    ################################################################ Sum

    def testSum(self):
        for i in xrange(11):
            left, right = self.simple[:i], self.simple[i:]

            leftSumming = Sum(named("something", lambda x: x))
            rightSumming = Sum(named("something", lambda x: x))

            for _ in left: leftSumming.fill(_)
            for _ in right: rightSumming.fill(_)

            self.assertAlmostEqual(leftSumming.sum, sum(left))
            self.assertAlmostEqual(rightSumming.sum, sum(right))

            finalResult = leftSumming + rightSumming

            self.assertAlmostEqual(finalResult.sum, sum(self.simple))

            self.checkJson(leftSumming)
            self.checkPickle(leftSumming)
            self.checkName(leftSumming)
       
    def testSumWithFilter(self):
        for i in xrange(11):
            left, right = self.struct[:i], self.struct[i:]

            leftSumming = Select(lambda x: x.bool, Sum(lambda x: x.double))
            rightSumming = Select(lambda x: x.bool, Sum(lambda x: x.double))

            for _ in left: leftSumming.fill(_)
            for _ in right: rightSumming.fill(_)

            self.assertAlmostEqual(leftSumming.cut.sum, sum(_.double for _ in left if _.bool))
            self.assertAlmostEqual(rightSumming.cut.sum, sum(_.double for _ in right if _.bool))

            finalResult = leftSumming + rightSumming

            self.assertAlmostEqual(finalResult.cut.sum, sum(_.double for _ in self.struct if _.bool))

            self.checkJson(leftSumming)
            self.checkJson(leftSumming)

    def testSumWithWeightingFactor(self):
        for i in xrange(11):
            left, right = self.struct[:i], self.struct[i:]

            leftSumming = Select(lambda x: x.int, Sum(lambda x: x.double))
            rightSumming = Select(lambda x: x.int, Sum(lambda x: x.double))

            for _ in left: leftSumming.fill(_)
            for _ in right: rightSumming.fill(_)

            self.assertAlmostEqual(leftSumming.cut.sum, sum(_.double * _.int for _ in left if _.int > 0))
            self.assertAlmostEqual(rightSumming.cut.sum, sum(_.double * _.int for _ in right if _.int > 0))

            finalResult = leftSumming + rightSumming

            self.assertAlmostEqual(finalResult.cut.sum, sum(_.double * _.int for _ in self.struct if _.int > 0))

            self.checkJson(leftSumming)
            self.checkPickle(leftSumming)
            self.checkName(leftSumming)

    def testSumStringFunctions(self):
        for i in xrange(11):
            left, right = self.simple[:i], self.simple[i:]

            leftSumming = Sum("_ + 1")
            rightSumming = Sum("datum + 1")

            for _ in left: leftSumming.fill(_)
            for _ in right: rightSumming.fill(_)

            self.assertAlmostEqual(leftSumming.sum, sum(left) + len(left))
            self.assertAlmostEqual(rightSumming.sum, sum(right) + len(right))

            finalResult = leftSumming + rightSumming

            self.assertAlmostEqual(finalResult.sum, sum(self.simple) + len(self.simple))

            self.checkJson(leftSumming)
            self.checkPickle(leftSumming)
            self.checkName(leftSumming)
       
    def testSumWithFilterStringFunctions(self):
        for i in xrange(11):
            left, right = self.struct[:i], self.struct[i:]

            leftSumming = Select("not bool", Sum("double + 1"))
            rightSumming = Select("not bool", Sum("double + 1"))

            for _ in left: leftSumming.fill(_)
            for _ in right: rightSumming.fill(_)

            self.assertAlmostEqual(leftSumming.cut.sum, sum(_.double + 1 for _ in left if not _.bool))
            self.assertAlmostEqual(rightSumming.cut.sum, sum(_.double + 1 for _ in right if not _.bool))

            finalResult = leftSumming + rightSumming

            self.assertAlmostEqual(finalResult.cut.sum, sum(_.double + 1 for _ in self.struct if not _.bool))

            self.checkJson(leftSumming)
            self.checkPickle(leftSumming)
            self.checkName(leftSumming)

    def testSumWithWeightingFactorStringFunctions(self):
        for i in xrange(11):
            left, right = self.struct[:i], self.struct[i:]

            leftSumming = Select("int", Sum("double * 2"))
            rightSumming = Select("int", Sum("double * 2"))

            for _ in left: leftSumming.fill(_)
            for _ in right: rightSumming.fill(_)

            self.assertAlmostEqual(leftSumming.cut.sum, sum(_.double * 2 * _.int for _ in left if _.int > 0))
            self.assertAlmostEqual(rightSumming.cut.sum, sum(_.double * 2 * _.int for _ in right if _.int > 0))

            finalResult = leftSumming + rightSumming

            self.assertAlmostEqual(finalResult.cut.sum, sum(_.double * 2 * _.int for _ in self.struct if _.int > 0))

            self.checkJson(leftSumming)
            self.checkPickle(leftSumming)
            self.checkName(leftSumming)

    ################################################################ Average

    def testAverage(self):
        for i in xrange(11):
            left, right = self.simple[:i], self.simple[i:]

            leftAveraging = Average(named("something", lambda x: x))
            rightAveraging = Average(named("something", lambda x: x))

            for _ in left: leftAveraging.fill(_)
            for _ in right: rightAveraging.fill(_)

            self.assertAlmostEqual(leftAveraging.mean, self.mean(left))
            self.assertAlmostEqual(rightAveraging.mean, self.mean(right))

            finalResult = leftAveraging + rightAveraging

            self.assertAlmostEqual(finalResult.mean, self.mean(self.simple))

            self.checkJson(leftAveraging)
            self.checkPickle(leftAveraging)
            self.checkName(leftAveraging)

    def testAverageWithFilter(self):
        for i in xrange(11):
            left, right = self.struct[:i], self.struct[i:]

            leftAveraging = Select(lambda x: x.bool, Average(lambda x: x.double))
            rightAveraging = Select(lambda x: x.bool, Average(lambda x: x.double))

            for _ in left: leftAveraging.fill(_)
            for _ in right: rightAveraging.fill(_)

            self.assertAlmostEqual(leftAveraging.cut.mean, self.mean([_.double for _ in left if _.bool]))
            self.assertAlmostEqual(rightAveraging.cut.mean, self.mean([_.double for _ in right if _.bool]))

            finalResult = leftAveraging + rightAveraging

            self.assertAlmostEqual(finalResult.cut.mean, self.mean([_.double for _ in self.struct if _.bool]))

            self.checkJson(leftAveraging)
            self.checkPickle(leftAveraging)
            self.checkName(leftAveraging)

    def testAverageWithWeightingFactor(self):
        for i in xrange(11):

            left, right = self.struct[:i], self.struct[i:]
            
            leftAveraging = Select(lambda x: x.int, Average(lambda x: x.double))
            rightAveraging = Select(lambda x: x.int, Average(lambda x: x.double))
            
            for _ in left: leftAveraging.fill(_)
            for _ in right: rightAveraging.fill(_)


            self.assertAlmostEqual(leftAveraging.cut.mean, self.meanWeighted(list(map(lambda _: _.double, left)), list(map(lambda _: _.int, left))))
            self.assertAlmostEqual(rightAveraging.cut.mean, self.meanWeighted(list(map(lambda _: _.double, right)), list(map(lambda _: _.int, right))))

            finalResult = leftAveraging + rightAveraging

            self.assertAlmostEqual(finalResult.cut.mean, self.meanWeighted(list(map(lambda _: _.double, self.struct)), list(map(lambda _: _.int, self.struct))))

            self.checkJson(leftAveraging)
            self.checkPickle(leftAveraging)
            self.checkName(leftAveraging)
        
    ################################################################ Deviate

    def testDeviate(self):
        for i in xrange(11):
            left, right = self.simple[:i], self.simple[i:]

            leftDeviating = Deviate(named("something", lambda x: x))
            rightDeviating = Deviate(named("something", lambda x: x))

            for _ in left: leftDeviating.fill(_)
            for _ in right: rightDeviating.fill(_)

            self.assertAlmostEqual(leftDeviating.variance, self.variance(left))
            self.assertAlmostEqual(rightDeviating.variance, self.variance(right))

            finalResult = leftDeviating + rightDeviating

            self.assertAlmostEqual(finalResult.variance, self.variance(self.simple))

            self.checkJson(leftDeviating)
            self.checkPickle(leftDeviating)
            self.checkName(leftDeviating)

    def testDeviateWithFilter(self):
        for i in xrange(11):
            left, right = self.struct[:i], self.struct[i:]

            leftDeviating = Select(lambda x: x.bool, Deviate(lambda x: x.double))
            rightDeviating = Select(lambda x: x.bool, Deviate(lambda x: x.double))

            for _ in left: leftDeviating.fill(_)
            for _ in right: rightDeviating.fill(_)
            

            self.assertAlmostEqual(leftDeviating.cut.variance, self.variance([_.double for _ in left if _.bool]))
            self.assertAlmostEqual(rightDeviating.cut.variance, self.variance([_.double for _ in right if _.bool]))

            finalResult = leftDeviating + rightDeviating

            self.assertAlmostEqual(finalResult.cut.variance, self.variance([_.double for _ in self.struct if _.bool]))

            self.checkJson(leftDeviating)
            self.checkPickle(leftDeviating)
            self.checkName(leftDeviating)
        
    def testDeviateWithWeightingFactor(self):
        for i in xrange(11):
            left, right = self.struct[:i], self.struct[i:]

            leftDeviating = Select(lambda x: x.int, Deviate(lambda x: x.double))
            rightDeviating = Select(lambda x: x.int, Deviate(lambda x: x.double))

            for _ in left: leftDeviating.fill(_)
            for _ in right: rightDeviating.fill(_)

            self.assertAlmostEqual(leftDeviating.cut.variance, self.varianceWeighted(list(map(lambda _: _.double, left)), list(map(lambda _: _.int, left))))
            self.assertAlmostEqual(rightDeviating.cut.variance, self.varianceWeighted(list(map(lambda _: _.double, right)), list(map(lambda _: _.int, right))))

            finalResult = leftDeviating + rightDeviating

            self.assertAlmostEqual(finalResult.cut.variance, self.varianceWeighted(list(map(lambda _: _.double, self.struct)), list(map(lambda _: _.int, self.struct))))

            self.checkJson(leftDeviating)
            self.checkPickle(leftDeviating)
            self.checkName(leftDeviating)

    ################################################################ Minimize

    def testMinimize(self):
        for i in xrange(11):
            left, right = self.simple[:i], self.simple[i:]

            leftMinimizing = Minimize(named("something", lambda x: x))
            rightMinimizing = Minimize(named("something", lambda x: x))

            for _ in left: leftMinimizing.fill(_)
            for _ in right: rightMinimizing.fill(_)

            if len(left) > 0:
                self.assertAlmostEqual(leftMinimizing.min, min(left))
            else:
                self.assertTrue(math.isnan(leftMinimizing.min))

            if len(right) > 0:
                self.assertAlmostEqual(rightMinimizing.min, min(right))
            else:
                self.assertTrue(math.isnan(rightMinimizing.min))

            finalResult = leftMinimizing + rightMinimizing

            self.assertAlmostEqual(finalResult.min, min(self.simple))

            self.checkJson(leftMinimizing)
            self.checkPickle(leftMinimizing)
            self.checkName(leftMinimizing)

    ################################################################ Maximize

    def testMaximize(self):
        for i in xrange(11):
            left, right = self.simple[:i], self.simple[i:]

            leftMaximizing = Maximize(named("something", lambda x: x))
            rightMaximizing = Maximize(named("something", lambda x: x))

            for _ in left: leftMaximizing.fill(_)
            for _ in right: rightMaximizing.fill(_)

            if len(left) > 0:
                self.assertAlmostEqual(leftMaximizing.max, max(left))
            else:
                self.assertTrue(math.isnan(leftMaximizing.max))

            if len(right) > 0:
                self.assertAlmostEqual(rightMaximizing.max, max(right))
            else:
                self.assertTrue(math.isnan(rightMaximizing.max))

            finalResult = leftMaximizing + rightMaximizing

            self.assertAlmostEqual(finalResult.max, max(self.simple))

            self.checkJson(leftMaximizing)
            self.checkPickle(leftMaximizing)
            self.checkName(leftMaximizing)

    ################################################################ Bag

    def testBag(self):
        one = Bag(named("something", lambda x: x), "N")
        for _ in self.simple: one.fill(_)
        self.assertEqual(one.values, {7.3: 1.0, 2.2: 1.0, -1.7: 1.0, -4.7: 1.0, 0.0: 2.0, -1.8: 1.0, -3.0: 1.0, 1.6: 1.0, 3.4: 1.0})

        two = Bag(lambda x: (x, x), "N2")
        for _ in self.simple: two.fill(_)
        self.assertEqual(two.values, {(7.3, 7.3): 1.0, (2.2, 2.2): 1.0, (-1.7, -1.7): 1.0, (-4.7, -4.7): 1.0, (0.0, 0.0): 2.0, (-1.8, -1.8): 1.0, (-3.0, -3.0): 1.0, (1.6, 1.6): 1.0, (3.4, 3.4): 1.0})

        three = Bag(lambda x: x.string[0], "S")
        for _ in self.struct: three.fill(_)
        self.assertEqual(three.values, {"n": 1.0, "e": 1.0, "t": 3.0, "s": 2.0, "f": 2.0, "o": 1.0})

        self.checkJson(one)
        self.checkJson(two)
        self.checkJson(three)
        self.checkPickle(one)
        self.checkPickle(two)
        self.checkPickle(three)
        self.checkName(one)
        self.checkName(two)
        self.checkName(three)

    def testBagWithLimit(self):
        one = Limit(20, Bag(lambda x: x.string, "S"))
        for _ in self.struct: one.fill(_)
        self.assertEqual(one.get.values, {"one": 1.0, "two": 1.0, "three": 1.0, "four": 1.0, "five": 1.0, "six": 1.0, "seven": 1.0, "eight": 1.0, "nine": 1.0, "ten": 1.0})

        two = Limit(9, Bag(lambda x: x.string, "S"))
        for _ in self.struct: two.fill(_)
        self.assertTrue(two.saturated)

        self.checkJson(one)
        self.checkJson(two)
        self.checkPickle(one)
        self.checkPickle(two)
        self.checkName(one)
        self.checkName(two)

    ################################################################ Bin

    def testBin(self):
        one = Bin(5, -3.0, 7.0, named("xaxis", lambda x: x))
        for _ in self.simple: one.fill(_)
        self.assertEqual(list(map(lambda _: _.entries, one.values)), [3.0, 2.0, 2.0, 1.0, 0.0])
        self.assertEqual(one.underflow.entries, 1.0)
        self.assertEqual(one.overflow.entries, 1.0)
        self.assertEqual(one.nanflow.entries, 0.0)

        two = Select(lambda x: x.bool, Bin(5, -3.0, 7.0, lambda x: x.double))
        for _ in self.struct: two.fill(_)

        self.assertEqual(list(map(lambda _: _.entries, two.cut.values)), [2.0, 1.0, 1.0, 1.0, 0.0])
        self.assertEqual(two.cut.underflow.entries, 0.0)
        self.assertEqual(two.cut.overflow.entries, 0.0)
        self.assertEqual(two.cut.nanflow.entries, 0.0)

        self.checkJson(one)
        self.checkJson(two)
        self.checkPickle(one)
        self.checkPickle(two)
        self.checkName(one)
        self.checkName(two)

    def testBinWithSum(self):
        one = Bin(5, -3.0, 7.0, named("xaxis", lambda x: x), Sum(named("yaxis", lambda x: 10.0)), Sum(lambda x: 10.0), Sum(lambda x: 10.0), Sum(lambda x: 10.0))
        for _ in self.simple: one.fill(_)
        self.assertEqual(list(map(lambda _: _.sum, one.values)), [30.0, 20.0, 20.0, 10.0, 0.0])
        self.assertEqual(one.underflow.sum, 10.0)
        self.assertEqual(one.overflow.sum, 10.0)
        self.assertEqual(one.nanflow.sum, 0.0)

        two = Select(lambda x: x.bool, Bin(5, -3.0, 7.0, lambda x: x.double, Sum(lambda x: 10.0), Sum(lambda x: 10.0), Sum(lambda x: 10.0), Sum(lambda x: 10.0)))
        for _ in self.struct: two.fill(_)

        self.assertEqual(list(map(lambda _: _.sum, two.cut.values)), [20.0, 10.0, 10.0, 10.0, 0.0])
        self.assertEqual(two.cut.underflow.sum, 0.0)
        self.assertEqual(two.cut.overflow.sum, 0.0)
        self.assertEqual(two.cut.nanflow.sum, 0.0)

        self.checkJson(one)
        self.checkJson(two)
        self.checkPickle(one)
        self.checkPickle(two)
        self.checkName(one)
        self.checkName(two)

    def testHistogram(self):
        one = Histogram(5, -3.0, 7.0, lambda x: x)

        for _ in self.simple: one.fill(_)
        self.assertEqual(one.numericalValues, [3.0, 2.0, 2.0, 1.0, 0.0])
        self.assertEqual(one.numericalUnderflow, 1.0)
        self.assertEqual(one.numericalOverflow, 1.0)
        self.assertEqual(one.numericalNanflow, 0.0)

        two = Histogram(5, -3.0, 7.0, lambda x: x.double, lambda x: x.bool)
        for _ in self.struct: two.fill(_)
        self.assertEqual(two.numericalValues, [2.0, 1.0, 1.0, 1.0, 0.0])
        self.assertEqual(two.numericalUnderflow, 0.0)
        self.assertEqual(two.numericalOverflow, 0.0)
        self.assertEqual(two.numericalNanflow, 0.0)

        self.checkJson(one)
        self.checkJson(two)
        self.checkPickle(one)
        self.checkPickle(two)
        self.checkName(one)
        self.checkName(two)

    def testPlotHistogram(self):
        one = Histogram(5, -3.0, 7.0, lambda x: x)
        map(lambda _: one.fill(_), self.simple)

        two = Histogram(5, -3.0, 7.0, lambda x: x.double, lambda x: x.bool)
        map(lambda _: two.fill(_), self.struct)

        try:
            if sys.version_info[0] == 2 and sys.version_info[1] == 6:
                raise ImportError   # Bokeh is not compatible with Python 2.6
            from histogrammar.plot.bokeh import plot,save,view
            glyph1 = one.bokeh("histogram")
            glyph2 = two.bokeh()
            c = plot(glyph1,glyph2)
            save(c,"plot_histogram.html")
            #self.checkHtml("example.html")
        except ImportError:
            pass

    def testPlotProfileErr(self):
        one = ProfileErr(5, -3.0, 7.0, lambda x: x, lambda x: x)
        map(lambda _: one.fill(_), self.simple)
    
        try:
            if sys.version_info[0] == 2 and sys.version_info[1] == 6:
                raise ImportError   # Bokeh is not compatible with Python 2.6
            from histogrammar.plot.bokeh import plot,save,view
            glyph = one.bokeh("errors")
            c = plot(glyph)
            save(c,"plot_errors.html")
            #self.checkHtml("example.html")
        except ImportError:
            pass

    def testPlotStack(self):
        one = Histogram(5, -3.0, 7.0, lambda x: x)
        two = Histogram(5, -3.0, 7.0, lambda x: x)

        labeling = Label(one=one, two=two)
        map(lambda _: labeling.fill(_), self.simple)

        try:
            if sys.version_info[0] == 2 and sys.version_info[1] == 6:
                raise ImportError   # Bokeh is not compatible with Python 2.6
            from histogrammar.plot.bokeh import plot,save,view
            s = Stack.build(one,two)
            glyph = s.bokeh()
            c = plot(glyph)
            save(c,"plot_stack.html")
            #self.checkHtml("example.html")
        except ImportError:
            pass

    ################################################################ SparselyBin

    def testSparselyBin(self):
        one = SparselyBin(1.0, named("something", lambda x: x))
        for _ in self.simple: one.fill(_)
        self.assertEqual([(i, v.entries) for i, v in sorted(one.bins.items())], [(-5, 1.0), (-3, 1.0), (-2, 2.0), (0, 2.0), (1, 1.0), (2, 1.0), (3, 1.0), (7, 1.0)])

        self.assertEqual(one.numFilled, 8)
        self.assertEqual(one.num, 13)
        self.assertEqual(one.low, -5.0)
        self.assertEqual(one.high, 8.0)

        self.checkJson(one)
        self.checkPickle(one)
        self.checkName(one)

        two = SparselyBin(1.0, named("something", lambda x: x), Sum(named("elsie", lambda x: x)))
        for _ in self.simple: two.fill(_)

        self.checkJson(two)
        self.checkPickle(two)
        self.checkName(two)

    ################################################################ CentrallyBin

    def testCentrallyBin(self):
        one = CentrallyBin([-3.0, -1.0, 0.0, 1.0, 3.0, 10.0], named("something", lambda x: x))
        self.assertEqual(one.center(1.5), 1.0)
        self.assertEqual(one.neighbors(1.0), (0.0, 3.0))
        self.assertEqual(one.neighbors(10.0), (3.0, None))
        self.assertEqual(one.range(-3.0), (float("-inf"), -2.0))
        self.assertEqual(one.range(-1.0), (-2.0, -0.5))
        self.assertEqual(one.range(0.0), (-0.5, 0.5))
        self.assertEqual(one.range(10.0), (6.5, float("inf")))

        for _ in self.simple: one.fill(_)

        self.assertEqual([(c, v.entries) for c, v in one.bins], [(-3.0,2.0), (-1.0,2.0), (0.0,2.0), (1.0,1.0), (3.0,2.0), (10.0,1.0)])

        self.checkJson(one)
        self.checkPickle(one)
        self.checkName(one)

        two = CentrallyBin([-3.0, -1.0, 0.0, 1.0, 3.0, 10.0], named("something", lambda x: x), Sum(named("elsie", lambda x: x)))

        self.checkJson(two)
        self.checkPickle(two)
        self.checkName(two)

    ################################################################ Fraction

    def testFraction(self):
        fracking = Fraction(named("something", lambda x: x > 0.0), Count())
        for _ in self.simple: fracking.fill(_)

        self.assertEqual(fracking.numerator.entries, 4.0)
        self.assertEqual(fracking.denominator.entries, 10.0)

        self.checkJson(fracking)
        self.checkPickle(fracking)
        self.checkName(fracking)

    def testFractionSum(self):
        fracking = Fraction(named("something", lambda x: x > 0.0), Sum(named("elsie", lambda x: x)))
        for _ in self.simple: fracking.fill(_)

        self.assertAlmostEqual(fracking.numerator.sum, 14.5)
        self.assertAlmostEqual(fracking.denominator.sum, 3.3)

        self.checkJson(fracking)
        self.checkPickle(fracking)
        self.checkName(fracking)

    def testFractionHistogram(self):
        fracking = Fraction(lambda x: x > 0.0, Histogram(5, -3.0, 7.0, lambda x: x))
        for _ in self.simple: fracking.fill(_)

        self.assertEqual(fracking.numerator.numericalValues, [0.0, 0.0, 2.0, 1.0, 0.0])
        self.assertEqual(fracking.denominator.numericalValues, [3.0, 2.0, 2.0, 1.0, 0.0])

        self.checkJson(fracking)
        self.checkPickle(fracking)
        self.checkName(fracking)

    ################################################################ Stack

    def testStack(self):
        stacking = Stack([0.0, 2.0, 4.0, 6.0, 8.0], named("something", lambda x: x), Count())
        for _ in self.simple: stacking.fill(_)        

        self.assertEqual([(k, v.entries) for k, v in stacking.bins], [(float("-inf"), 10.0), (0.0, 6.0), (2.0, 3.0), (4.0, 1.0), (6.0, 1.0), (8.0, 0.0)])

        self.checkJson(stacking)
        self.checkPickle(stacking)
        self.checkName(stacking)

    def testStackWithSum(self):
        stacking = Stack([0.0, 2.0, 4.0, 6.0, 8.0], named("something", lambda x: x), Sum(named("elsie", lambda x: x)))
        for _ in self.simple: stacking.fill(_)        

        self.assertEqual([(k, v.entries) for k, v in stacking.bins], [(float("-inf"), 10.0), (0.0, 6.0), (2.0, 3.0), (4.0, 1.0), (6.0, 1.0), (8.0, 0.0)])

        self.checkJson(stacking)
        self.checkPickle(stacking)
        self.checkName(stacking)

    ################################################################ IrregularlyBin

    def testIrregularlyBin(self):
        partitioning = IrregularlyBin([0.0, 2.0, 4.0, 6.0, 8.0], named("something", lambda x: x), Count())
        for _ in self.simple: partitioning.fill(_)

        self.assertEqual([(k, v.entries) for k, v in partitioning.bins], [(float("-inf"), 4.0), (0.0, 3.0), (2.0, 2.0), (4.0, 0.0), (6.0, 1.0), (8.0, 0.0)])

        self.checkJson(partitioning)
        self.checkPickle(partitioning)
        self.checkName(partitioning)

    def testIrregularlyBinSum(self):
        partitioning = IrregularlyBin([0.0, 2.0, 4.0, 6.0, 8.0], named("something", lambda x: x), Sum(named("elsie", lambda x: x)))
        for _ in self.simple: partitioning.fill(_)

        self.assertAlmostEqual(partitioning.bins[0][1].sum, -11.2)
        self.assertAlmostEqual(partitioning.bins[1][1].sum, 1.6)

        self.checkJson(partitioning)
        self.checkPickle(partitioning)
        self.checkName(partitioning)

    ################################################################ Categorize

    def testCategorize(self):
        categorizing = Categorize(named("something", lambda x: x.string[0]))
        for _ in self.struct: categorizing.fill(_)

        self.assertEqual(dict((k, v.entries) for k, v in categorizing.pairsMap.items()), {"n": 1.0, "e": 1.0, "t": 3.0, "s": 2.0, "f": 2.0, "o": 1.0})

        self.checkJson(categorizing)
        self.checkPickle(categorizing)
        self.checkName(categorizing)

        categorizing2 = Categorize(named("something", lambda x: x.string[0]), Sum(named("elsie", lambda x: x.double)))
        for _ in self.struct: categorizing2.fill(_)

        self.checkJson(categorizing2)
        self.checkPickle(categorizing2)
        self.checkName(categorizing2)

    ################################################################ Label

    def testLabel(self):
        one = Histogram(5, -3.0, 7.0, lambda x: x)
        two = Histogram(10, 0.0, 10.0, lambda x: x)
        three = Histogram(5, -3.0, 7.0, lambda x: 2*x)

        labeling = Label(one=one, two=two, three=three)

        for _ in self.simple: labeling.fill(_)

        self.assertEqual(labeling("one").numericalValues, [3.0, 2.0, 2.0, 1.0, 0.0])
        self.assertEqual(labeling("two").numericalValues, [2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        self.assertEqual(labeling("three").numericalValues, [0.0, 2.0, 0.0, 2.0, 1.0])

        self.checkJson(labeling)
        self.checkPickle(labeling)
        self.checkName(labeling)

    def testLabelDifferentCuts(self):
        one = Histogram(10, -10, 10, lambda x: x, lambda x: x > 0)
        two = Histogram(10, -10, 10, lambda x: x, lambda x: x > 5)
        three = Histogram(10, -10, 10, lambda x: x, lambda x: x < 5)

        labeling = Label(one=one, two=two, three=three)

        for _ in self.simple: labeling.fill(_)

        self.assertEqual(labeling("one").numericalValues, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0])
        self.assertEqual(labeling("two").numericalValues, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        self.assertEqual(labeling("three").numericalValues, [0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 2.0, 0.0, 0.0, 0.0])

        self.checkJson(labeling)
        self.checkPickle(labeling)
        self.checkName(labeling)

    ################################################################ UntypedLabel

    def testUntypedLabel(self):
        one = Histogram(5, -3.0, 7.0, lambda x: x)
        two = Histogram(10, 0.0, 10.0, lambda x: x)
        three = Histogram(5, -3.0, 7.0, lambda x: 2*x)

        labeling = UntypedLabel(one=one, two=two, three=three)

        for _ in self.simple: labeling.fill(_)

        self.assertEqual(labeling("one").numericalValues, [3.0, 2.0, 2.0, 1.0, 0.0])
        self.assertEqual(labeling("two").numericalValues, [2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        self.assertEqual(labeling("three").numericalValues, [0.0, 2.0, 0.0, 2.0, 1.0])

        self.checkJson(labeling)
        self.checkPickle(labeling)
        self.checkName(labeling)

    def testUntypedLabelDifferenCuts(self):
        one = Histogram(10, -10, 10, lambda x: x, lambda x: x > 0)
        two = Histogram(10, -10, 10, lambda x: x, lambda x: x > 5)
        three = Histogram(10, -10, 10, lambda x: x, lambda x: x < 5)

        labeling = UntypedLabel(one=one, two=two, three=three)

        for _ in self.simple: labeling.fill(_)

        self.assertEqual(labeling("one").numericalValues, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0])
        self.assertEqual(labeling("two").numericalValues, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        self.assertEqual(labeling("three").numericalValues, [0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 2.0, 0.0, 0.0, 0.0])

        self.checkJson(labeling)
        self.checkPickle(labeling)
        self.checkName(labeling)
        
    def testUntypedLabelMultipleTypes(self):
        one = Histogram(5, -3.0, 7.0, lambda x: x)
        two = Sum(lambda x: 1.0)
        three = Deviate(named("something", lambda x: x + 100.0))

        mapping = UntypedLabel(one=one, two=two, three=three)

        for _ in self.simple: mapping.fill(_)

        self.assertEqual(mapping("one").numericalValues, [3.0, 2.0, 2.0, 1.0, 0.0])
        self.assertEqual(mapping("two").sum, 10.0)
        self.assertAlmostEqual(mapping("three").entries, 10.0)
        self.assertAlmostEqual(mapping("three").mean, 100.33)
        self.assertAlmostEqual(mapping("three").variance, 10.8381)

        self.checkJson(mapping)
        self.checkPickle(mapping)
        self.checkName(mapping)

    ################################################################ Index

    def testIndex(self):
        one = Histogram(5, -3.0, 7.0, lambda x: x)
        two = Histogram(10, 0.0, 10.0, lambda x: x)
        three = Histogram(5, -3.0, 7.0, lambda x: 2*x)

        indexing = Index(one, two, three)

        for _ in self.simple: indexing.fill(_)

        self.assertEqual(indexing(0).numericalValues, [3.0, 2.0, 2.0, 1.0, 0.0])
        self.assertEqual(indexing(1).numericalValues, [2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        self.assertEqual(indexing(2).numericalValues, [0.0, 2.0, 0.0, 2.0, 1.0])

        self.checkJson(indexing)
        self.checkPickle(indexing)
        self.checkName(indexing)

    def testIndexDifferentCuts(self):
        one = Histogram(10, -10, 10, lambda x: x, lambda x: x > 0)
        two = Histogram(10, -10, 10, lambda x: x, lambda x: x > 5)
        three = Histogram(10, -10, 10, lambda x: x, lambda x: x < 5)

        indexing = Index(one, two, three)

        for _ in self.simple: indexing.fill(_)

        self.assertEqual(indexing(0).numericalValues, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0])
        self.assertEqual(indexing(1).numericalValues, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        self.assertEqual(indexing(2).numericalValues, [0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 2.0, 0.0, 0.0, 0.0])

        self.checkJson(indexing)
        self.checkPickle(indexing)
        self.checkName(indexing)

    ################################################################ Branch

    def testBranch(self):
        one = Histogram(5, -3.0, 7.0, lambda x: x)
        two = Count()
        three = Deviate(lambda x: x + 100.0)

        branching = Branch(one, two, three)

        for _ in self.simple: branching.fill(_)

        self.assertEqual(branching.i0.numericalValues, [3.0, 2.0, 2.0, 1.0, 0.0])
        self.assertEqual(branching.i0.numericalUnderflow, 1.0)
        self.assertEqual(branching.i0.numericalOverflow, 1.0)
        self.assertEqual(branching.i0.numericalNanflow, 0.0)

        self.assertEqual(branching.i1.entries, 10.0)

        self.assertAlmostEqual(branching.i2.entries, 10.0)
        self.assertAlmostEqual(branching.i2.mean, 100.33)
        self.assertAlmostEqual(branching.i2.variance, 10.8381)

        self.checkJson(branching)
        self.checkPickle(branching)
        self.checkName(branching)
        
    ################################################################ Usability in fold/aggregate

    # def testAggregate(self):
    #     pass
