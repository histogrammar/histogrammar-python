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

import bisect
import marshal
import math
import random
import types
import sys

# Definitions for python 2/3 compatability 
if sys.version_info[0] > 2:
    basestring = str
    xrange = range
    long = int
elif sys.version_info[0] == 2 and sys.version_info[1] > 6:
    from functools import total_ordering
else:
    # Copyright (c) 2012, Konsta Vesterinen (applies to Python 2.6 implementation of total_ordering only)
    #
    # All rights reserved.
    #
    # Redistribution and use in source and binary forms, with or without
    # modification, are permitted provided that the following conditions are met:
    #
    # * Redistributions of source code must retain the above copyright notice, this
    #   list of conditions and the following disclaimer.
    #
    # * Redistributions in binary form must reproduce the above copyright notice,
    #   this list of conditions and the following disclaimer in the documentation
    #   and/or other materials provided with the distribution.
    #
    # * The names of the contributors may not be used to endorse or promote products
    #   derived from this software without specific prior written permission.
    #
    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    # ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    # WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY DIRECT,
    # INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    # BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    # DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    # LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
    # OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
    # ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    def total_ordering(cls):
        """
        Backport to work with Python 2.6
        Class decorator that fills in missing ordering methods
        Code from: http://code.activestate.com/recipes/576685/
        Copied from: https://github.com/kvesteri/total-ordering
        """
        convert = {
            '__lt__': [
                ('__gt__', lambda self, other: not (self < other or self == other)),
                ('__le__', lambda self, other: self < other or self == other),
                ('__ge__', lambda self, other: not self < other)],
            '__le__': [
                ('__ge__', lambda self, other: not self <= other or self == other),
                ('__lt__', lambda self, other: self <= other and not self == other),
                ('__gt__', lambda self, other: not self <= other)],
            '__gt__': [
                ('__lt__', lambda self, other: not (self > other or self == other)),
                ('__ge__', lambda self, other: self > other or self == other),
                ('__le__', lambda self, other: not self > other)],
            '__ge__': [
                ('__le__', lambda self, other: (not self >= other) or self == other),
                ('__gt__', lambda self, other: self >= other and not self == other),
                ('__lt__', lambda self, other: not self >= other)]
        }
        roots = set(dir(cls)) & set(convert)
        if not roots:
            raise ValueError('must define at least one ordering operation: < > <= >=')
        root = max(roots)       # prefer __lt__ to __le__ to __gt__ to __ge__
        for opname, opfunc in convert[root]:
            if opname not in roots:
                opfunc.__name__ = opname
                opfunc.__doc__ = getattr(int, opname).__doc__
                setattr(cls, opname, opfunc)
        return cls
    
@total_ordering
class LessThanEverything(object):
    """An object that will always float the the beginning of a list in a sort."""
    def __le__(self, other):
        return True
    def __eq__(self, other):
        return self is other

def inheritdoc(cls):
    def _fn(fn):
        if fn.__name__ in cls.__dict__:
            fn.__doc__ = cls.__dict__[fn.__name__].__doc__
        return fn
    return _fn

################################################################ handling key set comparisons with optional keys

def hasKeys(test, required, optional=set()):
    """Checks to see if a dict from JSON has the right keys."""
    if not isinstance(test, set):
        test = set(test)
    if not isinstance(required, set):
        required = set(required)
    if not isinstance(optional, set):
        optional = set(optional)
    return required.issubset(test) and test.issubset(required.union(optional))

def maybeAdd(json, **pairs):
    """Adds key-value pairs to a dict for JSON if the value is not None."""
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
    """Utility for reservoir sampling on data of type RANGE.
      
    Collects up to ``limit`` data points (based on count, not sum of weights), including their weights. Once the ``limit`` is reached, points are deweighted such that the final sample is a statistically unbiased representative of the full sample.
      
    Merge step assumes that RANGE is immutable.
    """

    def __init__(self, limit, *initial):
        self.limit = limit
        self.numObserved = 0
        self.keyvalues = []
        for y, weight in initial:
            self.update(y, weight)

    def update(self, y, weight=1.0, randomGenerator=None):
        """Add a point to the reservoir.
        
        Args:
            y (RANGE): data point to insert or possibly insert in the sample
            weight (float): weight of the point; data in the final sample are represented in proportion to these weights (probability of a point with weight ``w`` to be in the final sample: ``w/W`` where ``W`` is the sum of all weights)
            randomGenerator (random.Random): optional random generator state, for deterministic tests with a given random seed
        """
        self.numObserved += 1

        if randomGenerator is None:
            r = random.uniform(0.0, 1.0)**(1.0/weight)
        else:
            r = randomGenerator.uniform(0.0, 1.0)**(1.0/weight)

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
    def values(self):
        """Get a snapshot of the sample."""
        return [pair for r, pair in self.keyvalues]
    @property
    def size(self):
        """Get the number of elements in the sample (saturates at ``limit``)."""
        return len(self.keyvalues)
    @property
    def isEmpty(self):
        """Determine if the sample is empty."""
        return self.size == 0

################################################################ NaN handling

relativeTolerance = 0.0
absoluteTolerance = 0.0

def numeq(x, y):
    """Introduces a ``===`` operator for all ``Double`` tolerance comparisons.
    
    Custom equality rules:
    
      - nan == nan (nans are used by some primitives to indicate missing data).
      - inf == inf and -inf == -inf (naturally, but has to be explicit given the following).
      - if ``histogrammar.util.relativeTolerance`` is greater than zero, numbers may differ by this small ratio.
      - if ``histogrammar.util.absoluteTolerance`` is greater than zero, numbers may differ by this small difference.
    
    Python's math.isclose algorithm is applied for non-NaNs:
    
        ``abs(x - y) <= max(relativeTolerance * max(abs(x), abs(y)), absoluteTolerance)``
   """
    if math.isnan(x) and math.isnan(y):
        return True
    elif math.isinf(x) and math.isinf(y):
        return (x > 0.0) == (y > 0.0)
    elif relativeTolerance > 0.0 and absoluteTolerance > 0.0:
        return abs(x - y) <= max(relativeTolerance * max(abs(x), abs(y)), absoluteTolerance)
    elif relativeTolerance > 0.0:
        return abs(x - y) <= relativeTolerance * max(abs(x), abs(y))
    elif absoluteTolerance > 0.0:
        return abs(x - y) <= absoluteTolerance
    else:
        return x == y

def minplus(x, y):
    """Rule for finding the minimum of two numbers, given the Histogrammar convention of representing the minimum of no data to be nan."""
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
    """Rule for finding the maximum of two numbers, given the Histogrammar convention of representing the maximum of no data to be nan."""
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
    """Custom rule for converting non-finite numbers to JSON as quoted strings: ``"inf"``, ``"-inf"``, and ``"nan"``. This avoids Python's bad habit of putting literal ``Infinity``, ``-Infinity``, and ``NaN`` in the JSON (without quotes)."""
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
    """Base trait for user functions.

    All functions passed to Histogrammar primitives get wrapped as UserFcn objects. Functions (instances of ``types.FunctionType``, not any callable) are used as-is and strings and deferred for later evaluation. If a string-based UserFcn is used in a normal ``fill`` operation, it gets compiled (once) as a Python function of the input structure's fields or a single-argument function for unstructured data.

    The string need not be interpreted this way: backends targeting JIT compilation can interpret the strings as C code; backends targeting GPUs and FPGAs can interpret them as CUDA/OpenCL or pin-out names. As usual with Histogrammar, the only platform-specific part is the user functions.

    UserFcns have a ``name`` parameter that may not be set. The user would ordinarily use the histogrammar.util.named function to give a function a name. Similarly, histogrammar.util.cached adds caching. (Naming and caching commute: they can be applied in either order.)

    UserFcns are also 100% serializable, so that Histogrammar trees can be pickled and they can be passed through PySpark.

    Note that the histogrammar.util.serializable function creates a UserFcn, avoids duplication, and commutes with histogrammar.util.named and histogrammar.util.cached (they can be applied in any order).
    """
    def __init__(self, expr, name=None):
        self.expr = expr
        if isinstance(expr, basestring) and name is None:
            self.name = expr
        elif isinstance(expr, types.FunctionType) and expr.__name__ != "<lambda>" and name is None:
            self.name = expr.__name__
        else:
            self.name = name
        if expr is not None and not isinstance(expr, (basestring, types.FunctionType)):
            raise TypeError("quantity ({0}) must be a string or function".format(expr))

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
                    context.update(math.__dict__)
                    try:
                        context.update(datum.__dict__)
                    except AttributeError:
                        v, = varname
                        if v is None:
                            try:
                                v, = set(c.co_names) - set(context.keys())
                            except ValueError:
                                raise NameError("more than one unrecognized variable names in single-argument function: {0}".format(set(c.co_names) - set(context.keys())))
                            varname[0] = v

                        context.update({v: datum})

                    return eval(c, context)

                self.fcn = function

            elif self.expr is None:
                raise TypeError("immutable container (created from JSON or .ed) cannot be filled")

            else:
                raise TypeError("unrecognized type for function: {0}".format(type(self.expr)))

        return self.fcn(*args, **kwds)

    def __reduce__(self):
        if isinstance(self.expr, basestring) or self.expr is None:
            return (deserializeString, (self.__class__, self.expr, self.name))

        elif isinstance(self.expr, types.FunctionType):
            refs = dict((n, self.expr.__globals__[n]) for n in self.expr.__code__.co_names if n in self.expr.__globals__)
            return (deserializeFunction, (self.__class__, marshal.dumps(self.expr.__code__), self.expr.__name__, self.expr.__defaults__, self.expr.__closure__, refs, self.name))

        else:
            raise TypeError("unrecognized type for function: {0}".format(type(self.expr)))

    def __repr__(self):
        return "UserFcn({0}, {1})".format(self.expr, self.name)

    def __eq__(self, other):
        out = isinstance(other, UserFcn) and self.name == other.name

        if isinstance(self.expr, types.FunctionType) and isinstance(other.expr, types.FunctionType):
            out = out and (self.expr.__code__.co_code == other.expr.__code__.co_code)
        else:
            out = out and (self.expr == other.expr)

        return out

    def __hash__(self):
        if isinstance(self.expr, types.FunctionType):
            return hash((None, self.expr.__code__.co_code, self.name))
        else:
            return hash((self.expr, self.name))

class CachedFcn(UserFcn):
    """Represents a cached UserFcn.
      
    Note that the histogrammar.util.cached function creates a CachedFcn, avoids duplication, and commutes with histogrammar.util.named and histogrammar.util.serializable (they can be applied in any order).
      
    **Example:**
      
    ::

        f = cached(lambda x: complexFunction(x))
        f(3.14)   # computes the function
        f(3.14)   # re-uses the old value
        f(4.56)   # computes the function again at a new point
    """

    try:
        import numpy
        np = numpy
    except ImportError:
        np = None

    def __call__(self, *args, **kwds):
        if hasattr(self, "lastArgs") and \
           len(args) == len(self.lastArgs) and \
           (all(x is y for x, y in zip(args, self.lastArgs)) or \
            (self.np is not None and all(self.np.array_equal(x, y) for x, y in zip(args, self.lastArgs))) or \
            (self.np is None and all(x == y for x, y in zip(args, self.lastArgs)))) and \
           set(kwds.keys()) == set(self.lastKwds.keys()) and \
           (all(kwds[k] is self.lastKwds[k] for k in kwds) or \
            (self.np is not None and all(self.np.array_equal(kwds[k], self.lastKwds[k]) for k in kwds)) or \
            (self.np is None and all(kwds[k] == self.lastKwds[k] for k in kwds))):
            return self.lastReturn
        else:
            self.lastArgs = args
            self.lastKwds = kwds
            self.lastReturn = super(CachedFcn, self).__call__(*args, **kwds)
            return self.lastReturn

    def __repr__(self):
        return "CachedFcn({0}, {1})".format(self.expr, self.name)

def deserializeString(cls, expr, name):
    """Used by Pickle to reconstruct a string-based histogrammar.util.UserFcn from Pickle data."""
    out = cls.__new__(cls)
    out.expr = expr
    out.name = name
    return out

def deserializeFunction(cls, __code__, __name__, __defaults__, __closure__, refs, name):
    """Used by Pickle to reconstruct a function-based histogrammar.util.UserFcn from Pickle data."""
    out = cls.__new__(cls)
    g = dict(globals(), **refs)
    out.expr = types.FunctionType(marshal.loads(__code__), g, __name__, __defaults__, __closure__)
    out.name = name
    return out

def serializable(fcn):
    """Create a serializable version of fcn (histogrammar.util.UserFcn), which can be a types.FunctionType or a string.

    Unlike the histogrammar.util.UserFcn constructor, this function avoids duplication (doubly wrapped objects) and commutes with histogrammar.util.cached and histogrammar.util.named (they can be applied in any order).
    """
    if isinstance(fcn, UserFcn):
        return fcn
    else:
        return UserFcn(fcn)

def cached(fcn):
    """Create a cached version of this function.

    Unlike the histogrammar.util.CachedFcn constructor, this function avoids duplication (doubly wrapped objects) and commutes with histogrammar.util.named and histogrammar.util.serializable (they can be applied in either order).
      
    **Example:**
      
    ::

        f = cached(lambda x: complexFunction(x))
        f(3.14)   # computes the function
        f(3.14)   # re-uses the old value
        f(4.56)   # computes the function again at a new point
    """

    if isinstance(fcn, CachedFcn):
        return fcn
    elif isinstance(fcn, UserFcn):
        return CachedFcn(fcn.expr, fcn.name)
    else:
        return CachedFcn(fcn)

def named(name, fcn):
    """Create a named, serializable version of fcn (histogrammar.util.UserFcn), which can be a types.FunctionType or a string.

    Unlike the histogrammar.util.UserFcn constructor, this function avoids duplication (doubly wrapped objects) and commutes with histogrammar.util.cached and histogrammar.util.serializable (they can be applied in any order).
    """
    if isinstance(fcn, UserFcn) and fcn.name is not None:
        raise ValueError("two names applied to the same function: {0} and {1}".format(fcn.name, name))
    elif isinstance(fcn, CachedFcn):
        return CachedFcn(fcn.expr, name)
    elif isinstance(fcn, UserFcn):
        return UserFcn(fcn.expr, name)
    else:
        return UserFcn(fcn, name)

################################################################ 1D clustering algorithm (used by AdaptivelyBin)

class Clustering1D(object):
    """Clusters data in one dimension for adaptive histogramming and approximating quantiles (such as the median) in one pass over the data.
      
    Adapted from `"A streaming parallel decision tree algorithm," <http://www.jmlr.org/papers/volume11/ben-haim10a/ben-haim10a.pdf>`_ Yael Ben-Haim and Elad Tom-Tov, *J. Machine Learning Research 11,* 2010.
    
    In the original paper, when the cluster-set needs to merge clusters (bins), it does so in increasing distance between neighboring bins. This algorithm also considers the content of the bins: the least-filled bins are merged first.
    
    The ``tailDetail`` parameter scales between extremes: ``tailDetail = 0`` *only* considers the content of the bins and ``tailDetail = 1`` *only* considers the distance between bins (pure Ben-Haim/Tom-Tov). Specifically, the first bins to be merged are the ones that minimize
    
    ::

        tailDetail*(x2 - x1)/(max - min) + (1.0 - tailDetail)*(v1 + v2)/entries
    
    where ``x1`` and ``x2`` are the positions of the neighboring bins, ``min`` and ``max`` are the most extreme data positions observed so far, ``v1`` and ``v2`` are the (weighted) number of entries in the neighboring bins, and ``entries`` is the total (weighted) number of entries. The corresponding objective function for pure Ben-Haim/Tom-Tov is just ``x2 - x1``.
    
    Args:
        num (int): Maximum number of bins (used as a constraint when merging).
        tailDetail (float): Between 0 and 1 inclusive: use 0 to focus on the bulk of the distribution and 1 to focus on the tails; see above for details.
        value (histogrammar.defs.Container): New value (note the ``=>``: expression is reevaluated every time a new value is needed).
        values (list of histogrammar.defs.Container): Containers for the surviving bins.
        min (float): Lowest observed value; used to interpret the first bin as a finite PDF (since the first bin technically extends to minus infinity).
        max (float): Highest observed value; used to interpret the last bin as a finite PDF (since the last bin technically extends to plus infinity).
        entries (float): Weighted number of entries (sum of all observed weights).
    """

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
        """Ben-Haim and Tom-Tov's "Algorithm 1" with min/max/entries tracking."""

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
        """Ben-Haim and Tom-Tov's "Algorithm 2" with min/max/entries tracking."""
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
        return self.num == other.num and numeq(self.tailDetail, other.tailDetail) and self.values == other.values and numeq(self.min, other.min) and numeq(self.max, other.max) and numeq(self.entries, other.entries)

    def __hash__(self):
        return hash((self.num, self.tailDetail, tuple(self.values), self.min, self.max, self.entries))

################################################################ interpretation of central bins as a distribution

class CentralBinsDistribution(object):
    """Mix-in for containers with non-uniform bins defined by centers (such as histogrammar.primitives.centralbin.CentrallyBin and histogrammar.primitives.adaptivebin.AdaptivelyBin)."""

    def pdf(self, *xs):
        """Probability distribution function (PDF) of one sample point.
      
        Computed as the ``entries`` of the corresponding bin divided by total number of entries divided by bin width.
        """
        if len(xs) == 0:
            return self.pdfTimesEntries(xs[0]) / self.entries
        else:
            return [x / self.entries for x in self.pdfTimesEntries(*xs)]

    def cdf(self, *xs):
        """Cumulative distribution function (CDF, or "accumulation function") of one sample point.
        
        Computed by adding bin contents from minus infinity to the point in question. This is a continuous, piecewise linear function.
        """
        if len(xs) == 0:
            return self.cdfTimesEntries(xs[0]) / self.entries
        else:
            return [x / self.entries for x in self.cdfTimesEntries(*xs)]

    def qf(self, *xs):
        """Quantile function (QF, or "inverse of the accumulation function") of one sample point.
       
        Computed like the CDF, but solving for the point in question, rather than integrating up to it. This is a continuous, piecewise linear function.
        """
        if len(xs) == 0:
            return self.qfTimesEntries(xs[0]) * self.entries
        else:
            return [x * self.entries for x in self.qfTimesEntries(*xs)]

    def pdfTimesEntries(self, x, *xs):
        """PDF without the non-unity number of entries removed (no division by zero when ``entries`` is zero)."""

        xs = [x] + list(xs)

        if len(self.bins) == 0 or math.isnan(self.min) or math.isnan(self.max):
            out = [0.0] * len(xs)

        elif len(self.bins) == 1:
            out = [float("inf") if x == self.bins[0][0] else 0.0 for x in xs]

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
        """CDF without the non-unity number of entries removed (no division by zero when ``entries`` is zero)."""

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
        """QF without the non-unity number of entries removed (no division by zero when ``entries`` is zero)."""

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
    def centersSet(self):
        """Set of centers of each bin."""
        return set(self.centers)
    @property
    def centers(self):
        """Iterable over the centers of each bin."""
        return map(lambda x: x[0], self.bins)
    @property
    def values(self):
        """Iterable over the containers associated with each bin."""
        return map(lambda x: x[0], self.bins)

    def index(self, x):
        """Find the closest index to ``x``."""
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
        """Return the exact center of the bin that ``x`` belongs to."""
        return self.bins[self.index(x)][0]

    def value(self, x):
        """Return the aggregator at the center of the bin that ``x`` belongs to."""
        return self.bins[self.index(x)][1]

    def nan(self, x):
        """Return ``true`` iff ``x`` is in the nanflow region (equal to ``NaN``)."""
        return math.isnan(x)

    def neighbors(self, center):
        """Find the lower and upper neighbors of a bin (given by exact bin center)."""
        closestIndex = self.index(center)
        if self.bins[closestIndex][0] != center:
            raise TypeError("position {0} is not the exact center of a bin".format(center))
        elif closestIndex == 0:
            return None, self.bins[closestIndex + 1][0]
        elif closestIndex == len(self.bins) - 1:
            return self.bins[closestIndex - 1][0], None
        else:
            return self.bins[closestIndex - 1][0], self.bins[closestIndex + 1][0]

    def range(self, center):
        """Get the low and high edge of a bin (given by exact bin center)."""
        below, above = self.neighbors(center)    # is never None, None
        if below is None:
            return float("-inf"), (center + above)/2.0
        elif above is None:
            return (below + center)/2.0, float("inf")
        else:
            return (below + center)/2.0, (above + center)/2.0
