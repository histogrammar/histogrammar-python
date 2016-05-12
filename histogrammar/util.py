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

import bisect
import functools
import marshal
import math
import random
import types

@functools.total_ordering
class LessThanEverything(object):
    def __le__(self, other):
        return True
    def __eq__(self, other):
        return self is other

################################################################ handling key set comparisons with optional keys

def hasKeys(test, required, optional=set()):
    if not isinstance(test, set):
        test = set(test)
    if not isinstance(required, set):
        required = set(required)
    if not isinstance(optional, set):
        optional = set(optional)
    return required.issubset(test) and test.issubset(required.union(optional))

def maybeAdd(json, **pairs):
    if len(pairs) == 0:
        return json
    else:
        out = dict(json)
        for k, v in pairs.items():
            if v is not None:
                out[k] = v
        return out

################################################################ random sampling

class Reservoir(object):
    def __init__(self, limit, *initial):
        self.limit = limit
        self.numObserved = 0
        self.keyvalues = []
        for y, weight in initial:
            self.update(y, weight)

    def update(self, y, weight=1.0):
        self.numObserved += 1

        r = random.uniform(0.0, 1.0)**(1.0/weight)

        if self.numObserved <= self.limit:
            # insert this item in the list, keeping the list sorted, letting it grow
            index = bisect.bisect_left(self.keyvalues, (r, LessThanEverything()))
            self.keyvalues.insert(index, (r, (y, weight)))

        elif self.keyvalues[0][0] < r:
            # insert this item in the list, keeping the list sorted, keeping its size fixed by shifting the lowest values down
            index = max(bisect.bisect_left(self.keyvalues, (r, LessThanEverything())) - 1, 0)
            for i in xrange(index):
                self.keyvalues[i] = self.keyvalues[i + 1]
            self.keyvalues[index] = (r, (y, weight))

    @property
    def values(self): return [pair for r, pair in self.keyvalues]
    @property
    def size(self): return len(self.keyvalues)
    @property
    def isEmpty(self): return self.size == 0

################################################################ NaN handling

def exact(x, y):
    return (math.isnan(x) and math.isnan(y)) or x == y

def minplus(x, y):
    if math.isnan(x) and math.isnan(y):
        return float("nan")
    elif math.isnan(x):
        return y
    elif math.isnan(y):
        return x
    elif x < y:
        return x
    else:
        return y

def maxplus(x, y):
    if math.isnan(x) and math.isnan(y):
        return float("nan")
    elif math.isnan(x):
        return y
    elif math.isnan(y):
        return x
    elif x > y:
        return x
    else:
        return y

def floatToJson(x):
    if math.isnan(x):
        return "nan"
    elif math.isinf(x) and x > 0.0:
        return "inf"
    elif math.isinf(x):
        return "-inf"
    else:
        return x

################################################################ function tools

class Fcn(object):
    def __init__(self, fcn, varname="datum"):
        if isinstance(fcn, basestring):
            c = compile(fcn, "<string>", "eval")
            def function(datum):
                context = dict(globals(), **{varname: datum})
                try:
                    context.update(datum.__dict__)
                except AttributeError:
                    pass
                return eval(c, context)
            fcn = function

        if not isinstance(fcn, types.FunctionType):
            raise TypeError("quantity or selection function must be a function or string expression")
        self.fcn = fcn

    def __call__(self, *args, **kwds):
        return self.fcn(*args, **kwds)

    def __reduce__(self):
        refs = {n: self.fcn.func_globals[n] for n in self.fcn.func_code.co_names if n in self.fcn.func_globals}
        return (deserializeFcn, (self.__class__, marshal.dumps(self.fcn.func_code), self.fcn.func_name, self.fcn.func_defaults, self.fcn.func_closure, refs))

    def __repr__(self):
        return "Fcn({})".format(self.fcn)

class CachedFcn(Fcn):
    def __call__(self, *args, **kwds):
        if hasattr(self, "lastArgs") and hasattr(self, "lastKwds") and args == self.lastArgs and kwds == self.lastKwds:
            return self.lastReturn
        else:
            self.lastArgs = args
            self.lastKwds = kwds
            self.lastReturn = self.fcn(*args, **kwds)
            return self.lastReturn

    def __repr__(self):
        return "CachedFcn({})".format(self.fcn)

def deserializeFcn(cls, func_code, func_name, func_defaults, func_closure, refs):
    out = cls.__new__(cls)
    g = dict(globals(), **refs)
    out.fcn = types.FunctionType(marshal.loads(func_code), g, func_name, func_defaults, func_closure)
    return out

def serializable(fcn):
    if fcn is None:
        return None
    elif isinstance(fcn, Fcn):
        return fcn
    else:
        return Fcn(fcn)

def cache(fcn):
    if isinstance(fcn, CachedFcn):
        return fcn
    elif isinstance(fcn, Fcn):
        return CachedFcn(fcn.fcn)
    else:
        return CachedFcn(fcn)

################################################################ 1D clustering algorithm (used by AdaptivelyBin)

class Clustering1D(object):
    def __init__(self, num, tailDetail, value, values, min, max, entries):
        self.num = num
        self.tailDetail = tailDetail
        self.value = value
        self.values = values
        self.min = min
        self.max = max
        self.entries = entries

        self._mergeClusters()

    def _mergeClusters(self):
        while len(self.values) > self.num:
            smallestDistance = None
            nearestNeighbors = None
            lowIndex = None

            for index in xrange(len(self.values) - 1):
                x1, v1 = self.values[index]
                x2, v2 = self.values[index + 1]

                distanceMetric = (self.tailDetail  * (x2 - x1)/(self.max - self.min) +
                           (1.0 - self.tailDetail) * (v1.entries + v2.entries)/self.entries)

                if smallestDistance is None or distanceMetric < smallestDistance:
                    smallestDistance = distanceMetric
                    nearestNeighbors = (x1, v1), (x2, v2)
                    lowIndex = index

            (x1, v1), (x2, v2) = nearestNeighbors
            replacement = (x1 * v1.entries + x2 * v2.entries) / (v1.entries + v2.entries), v1 + v2

            del self.values[lowIndex]
            del self.values[lowIndex]
            self.values.insert(lowIndex, replacement)
            
    def update(self, x, datum, weight):
        if weight > 0.0:
            index = bisect.bisect_left(self.values, (x, LessThanEverything()))
            if len(self.values) > index and self.values[index][0] == x:
                self.values[index][1].fill(datum, weight)
            else:
                v = self.value.zero()
                v.fill(datum, weight)
                self.values.insert(index, (x, v))
                self._mergeClusters()

        if math.isnan(self.min) or x < self.min:
            self.min = x
        if math.isnan(self.max) or x > self.max:
            self.max = x

        self.entries += weight

    def merge(self, other):
        bins = {}

        for x, v in self.values:
            bins[x] = v.copy()          # replace them; don't update them in-place

        for x, v in other.values:
            if x in bins:
                bins[x] = bins[x] + v   # replace them; don't update them in-place
            else:
                bins[x] = v.copy()      # replace them; don't update them in-place

        return Clustering1D(self.num, self.tailDetail, self.value, sorted(bins.items()), minplus(self.min, other.min), maxplus(self.max, other.max), self.entries + other.entries)

    def __eq__(self, other):
        return self.num == other.num and exact(self.tailDetail, other.tailDetail) and self.values == other.values and exact(self.min, other.min) and exact(self.max, other.max) and exact(self.entries, other.entries)

    def __hash__(self):
        return hash((self.num, self.tailDetail, self.values, self.min, self.max, self.entries))

################################################################ interpretation of central bins as a distribution

class CentralBinsDistribution(object):
    def pdf(self, *xs):
        if len(xs) == 0:
            return self.pdfTimesEntries(xs[0]) / self.entries
        else:
            return [x / self.entries for x in self.pdfTimesEntries(*xs)]

    def cdf(self, *xs):
        if len(xs) == 0:
            return self.cdfTimesEntries(xs[0]) / self.entries
        else:
            return [x / self.entries for x in self.cdfTimesEntries(*xs)]

    def qf(self, *xs):
        if len(xs) == 0:
            return self.qfTimesEntries(xs[0]) * self.entries
        else:
            return [x * self.entries for x in self.qfTimesEntries(*xs)]

    def pdfTimesEntries(self, x, *xs):
        xs = [x] + list(xs)

        if len(self.bins) == 0 or math.isnan(self.min) or math.isnan(self.max):
            out = [0.0] * len(xs)

        elif len(self.bins) == 1:
            out = [float("inf") if x == bins[0][0] else 0.0 for x in xs]

        else:
            out = [0.0] * len(xs)

            left = self.min
            for i in xrange(len(self.bins)):
                if i < len(self.bins) - 1:
                    right = (self.bins[i][0] + self.bins[i + 1][0]) / 2.0
                else:
                    right = self.max

                entries = self.bins[i][1].entries

                for j, x in enumerate(xs):
                    if left <= x and x < right:
                        out[j] = entries / (right - left)

                left = right
            
        if len(xs) == 1:
            return out[0]
        else:
            return out

    def cdfTimesEntries(self, x, *xs):
        xs = [x] + list(xs)

        if len(self.bins) == 0 or math.isnan(self.min) or math.isnan(self.max):
            out = [0.0] * len(xs)

        elif len(self.bins) == 1:
            out = []
            for x in xs:
                if x < self.bins[0][0]:
                    out.append(0.0)
                elif x == self.bins[0][0]:
                    out.append(self.bins[0][1].entries / 2.0)
                else:
                    out.append(self.bins[0][1].entries)

        else:
            out = [0.0] * len(xs)

            left = self.min
            cumulative = 0.0
            for i in xrange(len(self.bins)):
                if i < len(self.bins) - 1:
                    right = (self.bins[i][0] + self.bins[i + 1][0]) / 2.0
                else:
                    right = self.max

                entries = self.bins[i][1].entries

                for j, x in enumerate(xs):
                    if left <= x and x < right:
                        out[j] = cumulative + entries * (x - left)/(right - left)

                left = right
                cumulative += entries

            for j, x in enumerate(xs):
                if x >= self.max:
                    out[j] = cumulative

        if len(xs) == 1:
            return out[0]
        else:
            return out

    def qfTimesEntries(self, y, *ys):
        ys = [y] + list(ys)

        if len(self.bins) == 0 or math.isnan(self.min) or math.isnan(self.max):
            out = [float("nan")] * len(ys)

        elif len(self.bins) == 1:
            out = [self.bins[0][0]] * len(ys)

        else:
            out = [self.min] * len(ys)

            left = self.min
            cumulative = 0.0
            for i in xrange(len(self.bins)):
                if i < len(self.bins) - 1:
                    right = (self.bins[i][0] + self.bins[i + 1][0]) / 2.0
                else:
                    right = self.max

                entries = self.bins[i][1].entries

                low = cumulative
                high = cumulative + entries

                for j, y in enumerate(ys):
                    if low <= y and y < high:
                        out[j] = left + (right - left)*(y - low)/(high - low)

                left = right
                cumulative += entries

            for j, y in enumerate(ys):
                if y >= cumulative:
                    out[j] = self.max

        if len(ys) == 1:
            return out[0]
        else:
            return out

class CentrallyBinMethods(object):
    @property
    def centersSet(self): return set(self.centers)
    @property
    def centers(self): return map(lambda (x, v): x, self.bins)
    @property
    def values(self): return map(lambda (x, v): v, self.bins)

    def index(self, x):
        closestIndex = bisect.bisect_left(self.bins, (x, LessThanEverything()))
        if closestIndex == len(self.bins):
            closestIndex = len(self.bins) - 1
        elif closestIndex > 0:
            x1 = self.bins[closestIndex - 1][0]
            x2 = self.bins[closestIndex][0]
            if abs(x - x1) < abs(x - x2):
                closestIndex = closestIndex - 1
        return closestIndex

    def center(self, x):
        return self.bins[self.index(x)][0]

    def value(self, x):
        return self.bins[self.index(x)][1]

    def nan(self, x):
        return math.isnan(x)

    def neighbors(self, center):
        closestIndex = self.index(center)
        if self.bins[closestIndex][0] != center:
            raise TypeError("position {} is not the exact center of a bin".format(center))
        elif closestIndex == 0:
            return None, self.bins[closestIndex + 1][0]
        elif closestIndex == len(self.bins) - 1:
            return self.bins[closestIndex - 1][0], None
        else:
            return self.bins[closestIndex - 1][0], self.bins[closestIndex + 1][0]

    def range(self, center):
        below, above = self.neighbors(center)    # is never None, None
        if below is None:
            return float("-inf"), (center + above)/2.0
        elif above is None:
            return (below + center)/2.0, float("inf")
        else:
            return (below + center)/2.0, (above + center)/2.0
