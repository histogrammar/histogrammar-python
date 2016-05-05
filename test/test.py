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
import unittest

from histogrammar import *
from histogrammar.histogram import Histogram

class TestEverything(unittest.TestCase):
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

    ################################################################ Sum

    def testSum(self):
        for i in xrange(11):
            left, right = self.simple[:i], self.simple[i:]

            leftSumming = Sum(lambda x: x)
            rightSumming = Sum(lambda x: x)

            for _ in left: leftSumming.fill(_)
            for _ in right: rightSumming.fill(_)

            self.assertAlmostEqual(leftSumming.sum, sum(left))
            self.assertAlmostEqual(rightSumming.sum, sum(right))

            finalResult = leftSumming + rightSumming

            self.assertAlmostEqual(finalResult.sum, sum(self.simple))

            self.checkJson(leftSumming)
       
    def testSumWithFilter(self):
        for i in xrange(11):
            left, right = self.struct[:i], self.struct[i:]

            leftSumming = Sum(lambda x: x.double, lambda x: x.bool)
            rightSumming = Sum(lambda x: x.double, lambda x: x.bool)

            for _ in left: leftSumming.fill(_)
            for _ in right: rightSumming.fill(_)

            self.assertAlmostEqual(leftSumming.sum, sum(_.double for _ in left if _.bool))
            self.assertAlmostEqual(rightSumming.sum, sum(_.double for _ in right if _.bool))

            finalResult = leftSumming + rightSumming

            self.assertAlmostEqual(finalResult.sum, sum(_.double for _ in self.struct if _.bool))

            self.checkJson(leftSumming)

    def testSumWithWeightingFactor(self):
        for i in xrange(11):
            left, right = self.struct[:i], self.struct[i:]

            leftSumming = Sum(lambda x: x.double, lambda x: x.int)
            rightSumming = Sum(lambda x: x.double, lambda x: x.int)

            for _ in left: leftSumming.fill(_)
            for _ in right: rightSumming.fill(_)

            self.assertAlmostEqual(leftSumming.sum, sum(_.double * _.int for _ in left if _.int > 0))
            self.assertAlmostEqual(rightSumming.sum, sum(_.double * _.int for _ in right if _.int > 0))

            finalResult = leftSumming + rightSumming

            self.assertAlmostEqual(finalResult.sum, sum(_.double * _.int for _ in self.struct if _.int > 0))

            self.checkJson(leftSumming)

    def testSumStringFunctions(self):
        for i in xrange(11):
            left, right = self.simple[:i], self.simple[i:]

            leftSumming = Sum("datum + 1")
            rightSumming = Sum("datum + 1")

            for _ in left: leftSumming.fill(_)
            for _ in right: rightSumming.fill(_)

            self.assertAlmostEqual(leftSumming.sum, sum(left) + len(left))
            self.assertAlmostEqual(rightSumming.sum, sum(right) + len(right))

            finalResult = leftSumming + rightSumming

            self.assertAlmostEqual(finalResult.sum, sum(self.simple) + len(self.simple))

            self.checkJson(leftSumming)
       
    def testSumWithFilterStringFunctions(self):
        for i in xrange(11):
            left, right = self.struct[:i], self.struct[i:]

            leftSumming = Sum("double + 1", "not bool")
            rightSumming = Sum("double + 1", "not bool")

            for _ in left: leftSumming.fill(_)
            for _ in right: rightSumming.fill(_)

            self.assertAlmostEqual(leftSumming.sum, sum(_.double + 1 for _ in left if not _.bool))
            self.assertAlmostEqual(rightSumming.sum, sum(_.double + 1 for _ in right if not _.bool))

            finalResult = leftSumming + rightSumming

            self.assertAlmostEqual(finalResult.sum, sum(_.double + 1 for _ in self.struct if not _.bool))

            self.checkJson(leftSumming)

    def testSumWithWeightingFactorStringFunctions(self):
        for i in xrange(11):
            left, right = self.struct[:i], self.struct[i:]

            leftSumming = Sum("double * 2", "int")
            rightSumming = Sum("double * 2", "int")

            for _ in left: leftSumming.fill(_)
            for _ in right: rightSumming.fill(_)

            self.assertAlmostEqual(leftSumming.sum, sum(_.double * 2 * _.int for _ in left if _.int > 0))
            self.assertAlmostEqual(rightSumming.sum, sum(_.double * 2 * _.int for _ in right if _.int > 0))

            finalResult = leftSumming + rightSumming

            self.assertAlmostEqual(finalResult.sum, sum(_.double * 2 * _.int for _ in self.struct if _.int > 0))

            self.checkJson(leftSumming)

    ################################################################ Average

    def testAverage(self):
        for i in xrange(11):
            left, right = self.simple[:i], self.simple[i:]

            leftAveraging = Average(lambda x: x)
            rightAveraging = Average(lambda x: x)

            for _ in left: leftAveraging.fill(_)
            for _ in right: rightAveraging.fill(_)

            self.assertAlmostEqual(leftAveraging.mean, self.mean(left))
            self.assertAlmostEqual(rightAveraging.mean, self.mean(right))

            finalResult = leftAveraging + rightAveraging

            self.assertAlmostEqual(finalResult.mean, self.mean(self.simple))

            self.checkJson(leftAveraging)

    def testAverageWithFilter(self):
        for i in xrange(11):
            left, right = self.struct[:i], self.struct[i:]

            leftAveraging = Average(lambda x: x.double, lambda x: x.bool)
            rightAveraging = Average(lambda x: x.double, lambda x: x.bool)

            for _ in left: leftAveraging.fill(_)
            for _ in right: rightAveraging.fill(_)

            self.assertAlmostEqual(leftAveraging.mean, self.mean([_.double for _ in left if _.bool]))
            self.assertAlmostEqual(rightAveraging.mean, self.mean([_.double for _ in right if _.bool]))

            finalResult = leftAveraging + rightAveraging

            self.assertAlmostEqual(finalResult.mean, self.mean([_.double for _ in self.struct if _.bool]))

            self.checkJson(leftAveraging)

    def testAverageWithWeightingFactor(self):
        for i in xrange(11):
            left, right = self.struct[:i], self.struct[i:]

            leftAveraging = Average(lambda x: x.double, lambda x: x.int)
            rightAveraging = Average(lambda x: x.double, lambda x: x.int)

            for _ in left: leftAveraging.fill(_)
            for _ in right: rightAveraging.fill(_)

            self.assertAlmostEqual(leftAveraging.mean, self.meanWeighted(map(lambda _: _.double, left), map(lambda _: _.int, left)))
            self.assertAlmostEqual(rightAveraging.mean, self.meanWeighted(map(lambda _: _.double, right), map(lambda _: _.int, right)))

            finalResult = leftAveraging + rightAveraging

            self.assertAlmostEqual(finalResult.mean, self.meanWeighted(map(lambda _: _.double, self.struct), map(lambda _: _.int, self.struct)))

            self.checkJson(leftAveraging)

    ################################################################ Deviate

    def testDeviate(self):
        for i in xrange(11):
            left, right = self.simple[:i], self.simple[i:]

            leftDeviating = Deviate(lambda x: x)
            rightDeviating = Deviate(lambda x: x)

            for _ in left: leftDeviating.fill(_)
            for _ in right: rightDeviating.fill(_)

            self.assertAlmostEqual(leftDeviating.variance, self.variance(left))
            self.assertAlmostEqual(rightDeviating.variance, self.variance(right))

            finalResult = leftDeviating + rightDeviating

            self.assertAlmostEqual(finalResult.variance, self.variance(self.simple))

            self.checkJson(leftDeviating)

    def testDeviateWithFilter(self):
        for i in xrange(11):
            left, right = self.struct[:i], self.struct[i:]

            leftDeviating = Deviate(lambda x: x.double, lambda x: x.bool)
            rightDeviating = Deviate(lambda x: x.double, lambda x: x.bool)

            for _ in left: leftDeviating.fill(_)
            for _ in right: rightDeviating.fill(_)

            self.assertAlmostEqual(leftDeviating.variance, self.variance([_.double for _ in left if _.bool]))
            self.assertAlmostEqual(rightDeviating.variance, self.variance([_.double for _ in right if _.bool]))

            finalResult = leftDeviating + rightDeviating

            self.assertAlmostEqual(finalResult.variance, self.variance([_.double for _ in self.struct if _.bool]))

            self.checkJson(leftDeviating)

    def testDeviateWithWeightingFactor(self):
        for i in xrange(11):
            left, right = self.struct[:i], self.struct[i:]

            leftDeviating = Deviate(lambda x: x.double, lambda x: x.int)
            rightDeviating = Deviate(lambda x: x.double, lambda x: x.int)

            for _ in left: leftDeviating.fill(_)
            for _ in right: rightDeviating.fill(_)

            self.assertAlmostEqual(leftDeviating.variance, self.varianceWeighted(map(lambda _: _.double, left), map(lambda _: _.int, left)))
            self.assertAlmostEqual(rightDeviating.variance, self.varianceWeighted(map(lambda _: _.double, right), map(lambda _: _.int, right)))

            finalResult = leftDeviating + rightDeviating

            self.assertAlmostEqual(finalResult.variance, self.varianceWeighted(map(lambda _: _.double, self.struct), map(lambda _: _.int, self.struct)))

            self.checkJson(leftDeviating)

    ################################################################ AbsoluteErr

    def testAbsoluteErr(self):
        for i in xrange(11):
            left, right = self.simple[:i], self.simple[i:]

            leftAbsoluteErring = AbsoluteErr(lambda x: x)
            rightAbsoluteErring = AbsoluteErr(lambda x: x)

            for _ in left: leftAbsoluteErring.fill(_)
            for _ in right: rightAbsoluteErring.fill(_)

            self.assertAlmostEqual(leftAbsoluteErring.mae, self.mae(left))
            self.assertAlmostEqual(rightAbsoluteErring.mae, self.mae(right))

            finalResult = leftAbsoluteErring + rightAbsoluteErring

            self.assertAlmostEqual(finalResult.mae, self.mae(self.simple))

            self.checkJson(leftAbsoluteErring)
        
    ################################################################ Minimize

    def testMinimize(self):
        for i in xrange(11):
            left, right = self.simple[:i], self.simple[i:]

            leftMinimizing = Minimize(lambda x: x)
            rightMinimizing = Minimize(lambda x: x)

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

    ################################################################ Maximize

    # def testMaximize(self):
    #     pass

    ################################################################ Quantile

    # def testQuantile(self):
    #     pass

    ################################################################ Bag

    def testBag(self):
        one = Bag(lambda x: x)
        for _ in self.simple: one.fill(_)
        self.assertEqual(one.values, {(7.3,): 1.0, (2.2,): 1.0, (-1.7,): 1.0, (-4.7,): 1.0, (0.0,): 2.0, (-1.8,): 1.0, (-3.0,): 1.0, (1.6,): 1.0, (3.4,): 1.0})

        two = Bag(lambda x: x, limit = 5)
        for _ in self.simple: two.fill(_)
        self.assertEqual(two.values, None)

        self.checkJson(one)
        self.checkJson(two)

    ################################################################ Bin

    def testBin(self):
        one = Bin(5, -3.0, 7.0, lambda x: x)
        for _ in self.simple: one.fill(_)
        self.assertEqual(map(lambda _: _.entries, one.values), [3.0, 2.0, 2.0, 1.0, 0.0])
        self.assertEqual(one.underflow.entries, 1.0)
        self.assertEqual(one.overflow.entries, 1.0)
        self.assertEqual(one.nanflow.entries, 0.0)

        two = Bin(5, -3.0, 7.0, lambda x: x.double, lambda x: x.bool)
        for _ in self.struct: two.fill(_)

        self.assertEqual(map(lambda _: _.entries, two.values), [2.0, 1.0, 1.0, 1.0, 0.0])
        self.assertEqual(two.underflow.entries, 0.0)
        self.assertEqual(two.overflow.entries, 0.0)
        self.assertEqual(two.nanflow.entries, 0.0)

        self.checkJson(one)
        self.checkJson(two)

    def testHistogram(self):
        one = Histogram(5, -3.0, 7.0, lambda x: x)

        for _ in self.simple: one.fill(_)
        self.assertEqual(map(lambda _: _.entries, one.values), [3.0, 2.0, 2.0, 1.0, 0.0])
        self.assertEqual(one.underflow.entries, 1.0)
        self.assertEqual(one.overflow.entries, 1.0)
        self.assertEqual(one.nanflow.entries, 0.0)

        two = Histogram(5, -3.0, 7.0, lambda x: x.double, lambda x: x.bool)
        for _ in self.struct: two.fill(_)
        self.assertEqual(map(lambda _: _.entries, two.values), [2.0, 1.0, 1.0, 1.0, 0.0])
        self.assertEqual(two.underflow.entries, 0.0)
        self.assertEqual(two.overflow.entries, 0.0)
        self.assertEqual(two.nanflow.entries, 0.0)

        self.checkJson(one)
        self.checkJson(two)

    ################################################################ SparselyBin

    def testSparselyBin(self):
        one = SparselyBin(1.0, lambda x: x)
        for _ in self.simple: one.fill(_)
        self.assertEqual([(i, v.entries) for i, v in sorted(one.bins.items())], [(-5, 1.0), (-3, 1.0), (-2, 2.0), (0, 2.0), (1, 1.0), (2, 1.0), (3, 1.0), (7, 1.0)])

        self.assertEqual(one.numFilled, 8)
        self.assertEqual(one.num, 13)
        self.assertEqual(one.low, -5.0)
        self.assertEqual(one.high, 8.0)

        self.checkJson(one)

    ################################################################ CentrallyBin

    # def testCentrallyBin(self):
    #     pass

    ################################################################ AdaptivelyBin

    # def testAdaptivelyBin(self):
    #     pass

    ################################################################ Fraction

    def testFraction(self):
        fracking = Fraction(lambda x: x > 0.0, Count())
        for _ in self.simple: fracking.fill(_)

        self.assertEqual(fracking.numerator.entries, 4.0)
        self.assertEqual(fracking.denominator.entries, 10.0)

        self.checkJson(fracking)

    def testFractionSum(self):
        fracking = Fraction(lambda x: x > 0.0, Sum(lambda x: x))
        for _ in self.simple: fracking.fill(_)

        self.assertAlmostEqual(fracking.numerator.sum, 14.5)
        self.assertAlmostEqual(fracking.denominator.sum, 3.3)

        self.checkJson(fracking)

    def testFractionHistogram(self):
        fracking = Fraction(lambda x: x > 0.0, Histogram(5, -3.0, 7.0, lambda x: x))
        for _ in self.simple: fracking.fill(_)

        self.assertEqual(fracking.numerator.numericalValues, [0.0, 0.0, 2.0, 1.0, 0.0])
        self.assertEqual(fracking.denominator.numericalValues, [3.0, 2.0, 2.0, 1.0, 0.0])

        self.checkJson(fracking)

    ################################################################ Stack

    def testStack(self):
        stacking = Stack(Count(), lambda x: x, 0.0, 2.0, 4.0, 6.0, 8.0)
        for _ in self.simple: stacking.fill(_)        

        self.assertEqual([(k, v.entries) for k, v in stacking.cuts], [(float("-inf"), 10.0), (0.0, 6.0), (2.0, 3.0), (4.0, 1.0), (6.0, 1.0), (8.0, 0.0)])

        self.checkJson(stacking)

    ################################################################ Partition

    # def testPartition(self):
    #     pass

    ################################################################ Categorize

    def testCategorize(self):
        categorizing = Categorize(lambda x: x.string[0])
        for _ in self.struct: categorizing.fill(_)

        self.assertEqual({k: v.entries for k, v in categorizing.pairsMap.items()}, {"n": 1.0, "e": 1.0, "t": 3.0, "s": 2.0, "f": 2.0, "o": 1.0})

        self.checkJson(categorizing)

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
        
    def testUntypedLabelMultipleTypes(self):
        one = Histogram(5, -3.0, 7.0, lambda x: x)
        two = Sum(lambda x: 1.0)
        three = Deviate(lambda x: x + 100.0)

        mapping = UntypedLabel(one=one, two=two, three=three)

        for _ in self.simple: mapping.fill(_)

        self.assertEqual(mapping("one").numericalValues, [3.0, 2.0, 2.0, 1.0, 0.0])
        self.assertEqual(mapping("two").sum, 10.0)
        self.assertAlmostEqual(mapping("three").entries, 10.0)
        self.assertAlmostEqual(mapping("three").mean, 100.33)
        self.assertAlmostEqual(mapping("three").variance, 10.8381)

        self.checkJson(mapping)

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

    ################################################################ Branch

    def testBranch(self):
        one = Histogram(5, -3.0, 7.0, lambda x: x)
        two = Count()
        three = Deviate(lambda x: x + 100.0)

        branching = Branch(one, two, three)

        for _ in self.simple: branching.fill(_)

        self.assertEqual(branching.i0.numericalValues, [3.0, 2.0, 2.0, 1.0, 0.0])
        self.assertEqual(branching.i0.underflow.entries, 1.0)
        self.assertEqual(branching.i0.overflow.entries, 1.0)
        self.assertEqual(branching.i0.nanflow.entries, 0.0)

        self.assertEqual(branching.i1.entries, 10.0)

        self.assertAlmostEqual(branching.i2.entries, 10.0)
        self.assertAlmostEqual(branching.i2.mean, 100.33)
        self.assertAlmostEqual(branching.i2.variance, 10.8381)

        self.checkJson(branching)
        
    ################################################################ Usability in fold/aggregate

    # def testAggregate(self):
    #     pass
