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

class UserFcn(object):
    def __init__(self, expr, name=None):
        self.expr = expr
        if isinstance(expr, basestring) and name is None:
            self.name = expr
        else:
            self.name = name

    def __call__(self, *args, **kwds):
        if not hasattr(self, "fcn"):
            if isinstance(self.expr, types.FunctionType):
                self.fcn = self.expr

            elif isinstance(self.expr, basestring):
                c = compile(self.expr, "<string>", "eval")

                # close over this state
                varname = [None]

                def function(datum):
                    context = dict(globals())
                    try:
                        context.update(datum.__dict__)
                    except AttributeError:
                        v, = varname
                        if v is None:
                            try:
                                v, = set(c.co_names) - set(context.keys())
                            except ValueError:
                                raise NameError("more than one unrecognized variable names in single-argument function: {}".format(set(c.co_names) - set(context.keys())))
                        context.update({v: datum})

                    return eval(c, context)

                self.fcn = function

            elif self.expr is None:
                raise TypeError("immutable container (created from JSON or .ed) cannot be filled")

            else:
                raise TypeError("unrecognized type for function: {}".format(type(self.expr)))

        return self.fcn(*args, **kwds)

    def __reduce__(self):
        if isinstance(self.expr, basestring) or self.expr is None:
            return (deserializeString, (self.__class__, self.expr, self.name))

        elif isinstance(self.expr, types.FunctionType):
            refs = {n: self.expr.func_globals[n] for n in self.expr.func_code.co_names if n in self.expr.func_globals}
            return (deserializeFunction, (self.__class__, marshal.dumps(self.expr.func_code), self.expr.func_name, self.expr.func_defaults, self.expr.func_closure, refs, self.name))

        else:
            raise TypeError("unrecognized type for function: {}".format(type(self.expr)))

    def __repr__(self):
        return "UserFcn({})".format(self.expr)

class CachedFcn(UserFcn):
    def __call__(self, *args, **kwds):
        if hasattr(self, "lastArgs") and hasattr(self, "lastKwds") and args == self.lastArgs and kwds == self.lastKwds:
            return self.lastReturn
        else:
            self.lastArgs = args
            self.lastKwds = kwds
            self.lastReturn = super(CachedFcn, self).__call__(*args, **kwds)
            return self.lastReturn

    def __repr__(self):
        return "CachedFcn({})".format(self.expr)

def deserializeString(cls, expr, name):
    out = cls.__new__(cls)
    out.expr = expr
    out.name = name
    return out

def deserializeFunction(cls, func_code, func_name, func_defaults, func_closure, refs, name):
    out = cls.__new__(cls)
    g = dict(globals(), **refs)
    out.expr = types.FunctionType(marshal.loads(func_code), g, func_name, func_defaults, func_closure)
    out.name = name
    return out

def serializable(fcn):
    if isinstance(fcn, UserFcn):
        return fcn
    else:
        return UserFcn(fcn)

def cached(fcn):
    if isinstance(fcn, CachedFcn):
        return fcn
    elif isinstance(fcn, UserFcn):
        return CachedFcn(fcn.expr, fcn.name)
    else:
        return CachedFcn(fcn)

def named(name, fcn):
    if isinstance(fcn, UserFcn) and fcn.name is not None:
        raise ValueError("two names applied to the same function: {} and {}".format(fcn.name, name))
    elif isinstance(fcn, CachedFcn):
        return CachedFcn(fcn.expr, name)
    elif isinstance(fcn, UserFcn):
        return UserFcn(fcn.expr, name)
    else:
        return UserFcn(fcn, name)

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
