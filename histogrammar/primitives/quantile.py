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

from histogrammar.defs import *
from histogrammar.util import *



## import random
## import numpy

## def attempt(p, stuff):
##     x, n = stuff[0]
##     estimate = x
##     entries = 0.0
##     cumulativeDeviation = 0.0
##     for x, n in stuff[1:]:
##         entries += n
##         if entries > 0.0:
##             cumulativeDeviation += abs(x - estimate)
##             learningRate = 1.5*cumulativeDeviation/entries**2
##             estimate += n * learningRate * (cmp(x, estimate) + 2.0 * p - 1.0)
##     return entries, cumulativeDeviation, estimate

## def flat(stuff):
##     for x, n in stuff:
##         if n == 1.0:
##             yield x
##         elif n == 2.0:
##             yield x
##             yield x
##         elif n == 3.0:
##             yield x
##             yield x

## stuff = [(random.choice([random.gauss(10, 1), random.gauss(6, 2), random.gauss(0, 2)]), float(random.randint(0, 3))) for i in xrange(1000)]

## plotme = open("/tmp/plotme.dat", "w")

## for i in xrange(100):
##     p = i / 100.0
##     entries1, cumulativeDeviation1, estimate1 = attempt(p, stuff[:10])
##     entries2, cumulativeDeviation2, estimate2 = attempt(p, stuff[10:20])
##     entries3, cumulativeDeviation3, estimate3 = attempt(p, stuff[20:100])
##     entries4, cumulativeDeviation4, estimate4 = attempt(p, stuff[100:800])
##     entries5, cumulativeDeviation5, estimate5 = attempt(p, stuff[800:])

##     entries, cumulativeDeviation, estimate = entries1, cumulativeDeviation1, estimate1

##     # entries += entries2
##     # cumulativeDeviation += abs(estimate2 - estimate)
##     # learningRate = 1.5*cumulativeDeviation/entries**2
##     # estimate += entries2 * learningRate * (cmp(estimate2, estimate) + 2.0 * p - 1.0)

##     # entries += entries3
##     # cumulativeDeviation += abs(estimate3 - estimate)
##     # learningRate = 1.5*cumulativeDeviation/entries**2
##     # estimate += entries3 * learningRate * (cmp(estimate3, estimate) + 2.0 * p - 1.0)

##     # entries += entries4
##     # cumulativeDeviation += abs(estimate4 - estimate)
##     # learningRate = 1.5*cumulativeDeviation/entries**2
##     # estimate += entries4 * learningRate * (cmp(estimate4, estimate) + 2.0 * p - 1.0)

##     # entries += entries5
##     # cumulativeDeviation += abs(estimate5 - estimate)
##     # learningRate = 1.5*cumulativeDeviation/entries**2
##     # estimate += entries5 * learningRate * (cmp(estimate5, estimate) + 2.0 * p - 1.0)

##     estimate = (entries1*estimate1 + entries2*estimate2 + entries3*estimate3 + entries4*estimate4 + entries5*estimate5) / (entries1 + entries2 + entries3 + entries4 + entries5)

##     print >> plotme, numpy.percentile(list(flat(stuff)), p*100.0), estimate

## plotme.close()



class Quantile(Factory, Container):
    @staticmethod
    def ed(entries, target, estimate):
        if entries < 0.0:
            raise ContainerException("entries ($entries) cannot be negative")
        out = Quantile(target, None, None)
        out.entries = float(entries)
        out.estimate = float(estimate)
        return out

    @staticmethod
    def ing(target, quantity, selection=unweighted):
        return Quantile(target, quantity, selection)

    def __init__(self, target, quantity, selection=unweighted):
        if target < 0.0 or target > 1.0:
            raise ContainerException("target ({}) must be between 0 and 1, inclusive".format(target))
        self.target = target
        self.quantity = quantity
        self.selection = selection
        self.entries = 0.0
        self.estimate = float("nan")
        self.cumulativeDeviation = 0.0
        super(Quantile, self).__init__()

    def zero(self): return Quantile(self.target, self.quantity, self.selection)

    def __add__(self, other):
        if isinstance(other, Quantile):
            if self.target == other.target:
                out = Quantile(self.target, self.quantity, self.selection)
                out.entries = self.entries + other.entries
                if math.isnan(self.estimate) and math.isnan(other.estimate):
                    out.estimate = float("nan")
                elif math.isnan(self.estimate):
                    out.estimate = other.estimate
                elif math.isnan(other.estimate):
                    out.estimate = self.estimate
                else:
                    out.estimate = (self.estimate*self.entries + other.estimate*other.entries) / (self.entries + other.entries)
                return out
            else:
                raise ContainerException("cannot add Quantiles because targets do not match ({} vs {})".format(self.target, other.target))
        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    def fill(self, datum, weight=1.0):
        if self.quantity is None or self.selection is None:
            raise RuntimeException("attempting to fill a container that has no fill rule")

        w = weight * self.selection(datum)
        if w > 0.0:
            q = self.quantity(datum)
            self.entries += w

            if math.isnan(self.estimate):
                self.estimate = q
            else:
                self.cumulativeDeviation += abs(q - self.estimate)
                learningRate = 1.5 * self.cumulativeDeviation / self.entries**2
                if q < self.estimate:
                    sgn = -1
                elif q > self.estimate:
                    sgn = 1
                else:
                    sgn = 0
                self.estimate = w * learningRate * (sgn + 2.0*self.target - 1.0)

    def toJsonFragment(self): return {
        "entries": self.entries,
        "target": self.target,
        "estimate": self.estimate,
        }

    @staticmethod
    def fromJsonFragment(json):
        if isinstance(json, dict) and set(json.keys()) == set(["entries", "target", "estimate"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = json["entries"]
            else:
                raise JsonFormatException(json["entries"], "Quantile.entries")

            if isinstance(json["target"], (int, long, float)):
                target = json["target"]
            else:
                raise JsonFormatException(json["target"], "Quantile.target")

            if isinstance(json["estimate"], (int, long, float)):
                estimate = json["estimate"]
            else:
                raise JsonFormatException(json["estimate"], "Quantile.estimate")

            return Quantile.ed(entries, target, estimate)

        else:
            raise JsonFormatException(json, "Quantile")

    def __repr__(self):
        return "Quantile[{}, {}]".format(self.target, self.estimate)

    def __eq__(self, other):
        return isinstance(other, Quantile) and self.quantity == other.quantity and self.selection == other.selection and exact(self.entries, other.entries) and exact(self.target, other.target) and exact(self.estimate, other.estimate)

    def __hash__(self):
        return hash((self.quantity, self.selection, self.entries, self.target, self.estimate))

Factory.register(Quantile)
