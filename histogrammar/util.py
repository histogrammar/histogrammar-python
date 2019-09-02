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

import histogrammar.pycparser.c_ast

# Definitions for python 2/3 compatability 
if sys.version_info[0] > 2:
    basestring = str
    xrange = range
    long = int

def inheritdoc(cls):
    def _fn(fn):
        if fn.__name__ in cls.__dict__:
            fn.__doc__ = cls.__dict__[fn.__name__].__doc__
        return fn
    return _fn


################################################################ attach sub-methods to the fill and plot methods

class FillMethod(object):
    def __init__(self, container, fill):
        self.container = container
        self.fill = fill
        self.root = container.fillroot
        self.pycuda = container.fillpycuda
        self.numpy = container.fillnumpy
        self.sparksql = container.fillsparksql
    def __call__(self, *args, **kwds):
        return self.fill(*args, **kwds)

class PlotMethod(object):
    def __init__(self, container, plot):
        self.container = container
        self.plot = plot
        try:
            self.root = container.plotroot
        except (AttributeError, KeyError):
            pass
        try:
            self.bokeh = container.plotbokeh
        except (AttributeError, KeyError):
            pass
        try:
            self.matplotlib = container.plotmatplotlib
        except (AttributeError, KeyError):
            pass
    def __call__(self, *args, **kwds):
        return self.plot(*args, **kwds)

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

################################################################ inexact floating point and NaN handling

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

def floatOrNan(x):
    """NaN is not a good key in a hash map because it isn't equal to itself. histogrammar.primitives.Bag therefore uses the string ``"nan"`` as a substitute. This converts to the right JSON string representation."""
    x = float(x)
    if math.isnan(x):
        return "nan"
    else:
        return x

def floatToJson(x):
    """Custom rule for converting non-finite numbers to JSON as quoted strings: ``"inf"``, ``"-inf"``, and ``"nan"``. This avoids Python's bad habit of putting literal ``Infinity``, ``-Infinity``, and ``NaN`` in the JSON (without quotes)."""
    if x in ("nan", "inf", "-inf"):
        return x
    elif math.isnan(x):
        return "nan"
    elif math.isinf(x) and x > 0.0:
        return "inf"
    elif math.isinf(x):
        return "-inf"
    else:
        return x

def floatToC99(x):
    if math.isnan(x):
        return "NAN"
    elif math.isinf(x) and x > 0.0:
        return "INFINITY"
    elif math.isinf(x):
        return "-INFINITY"
    else:
        return str(x)

def rangeToJson(x):
    """Custom rule for converting numbers, one-dimensional vectors of numbers, and strings to JSON, converting non-finite nmbers to ``"inf"``, ``"-inf"``, and ``"nan"``. This avoids Python's bad habit of putting literal ``Infinity``, ``-Infinity``, and ``NaN`` in the JSON (without quotes)."""
    if isinstance(x, basestring):
        return x
    elif isinstance(x, (list, tuple)):
        return [floatToJson(xi) for xi in x]
    else:
        return floatToJson(x)

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
        
        if expr is None:
            ok = True
        elif isinstance(expr, basestring):
            ok = True
        elif isinstance(expr, types.FunctionType):
            ok = True
        else:
            try:
                from pyspark.sql.column import Column
            except ImportError:
                ok = False
            else:
                if isinstance(expr, Column):
                    if self.name is None:
                        self.name = str(expr)[7:-1]
                ok = True
        if not ok:
            raise TypeError("quantity ({0}) must be a string, function, or SparkSQL Column".format(expr))

        if name is not None and not isinstance(name, basestring):
            raise TypeError("function name must be a string, not {0} (perhaps your arguments are reversed)".format(name))

    def asSparkSQL(self):
        from pyspark.sql.column import Column
        if isinstance(self.expr, Column):
            return self.expr._jc
        else:
            raise TypeError("UserFcn is not a SparkSQL Column: " + repr(self))

    def __call__(self, *args, **kwds):
        if not hasattr(self, "fcn"):
            if isinstance(self.expr, types.FunctionType):
                self.fcn = self.expr

            elif isinstance(self.expr, basestring):
                c = compile(self.expr, "<string>", "eval")

                # close over this state
                varname = [None]

                try:
                    import numpy
                except ImportError:
                    numpy = None
                try:
                    import pandas
                except ImportError:
                    pandas = None

                def function(datum):
                    context = dict(globals())

                    # fill the namespace with math.* functions
                    context.update(math.__dict__)

                    # if you have Numpy, include numpy.* functions
                    if numpy is not None:
                        context["numpy"] = numpy
                        context["np"] = numpy

                    # if the datum is a dict, override the namespace with its dict keys
                    if isinstance(datum, dict):                # if it's a dict
                        context.update(datum)                  # use its items as variables

                    # if the datum is a Numpy record array, override the namespace with its field names
                    elif numpy is not None and isinstance(datum, numpy.core.records.recarray):
                        context.update(dict((n, datum[n]) for n in datum.dtype.names))

                    # if the datum is a Pandas DataFrame, override the namespace with its column names
                    elif pandas is not None and isinstance(datum, pandas.core.frame.DataFrame):
                        context.update(dict((n, datum[n].values) for n in datum.columns))

                    else:
                        try:
                            context.update(datum.__dict__)     # try to use its attributes as variables
                        except AttributeError:
                            v, = varname                       # otherwise, use the one and only variable
                            if v is None:                      # as the object (only discover it once)
                                v = set(c.co_names) - set(context.keys())
                                if len(v) > 1:
                                    raise NameError("more than one unrecognized variable names in single-argument function: {0}".format(set(c.co_names) - set(context.keys())))
                                elif len(v) == 0:
                                    v = None
                                else:
                                    v = list(v)[0]

                                varname[0] = v

                            if v is not None:
                                context.update({v: datum})

                    return eval(c, context)

                self.fcn = function

            elif self.expr is None:
                raise TypeError("immutable container (created from JSON or .ed) cannot be filled")

            else:
                try:
                    from pyspark.sql.column import Column
                except ImportError:
                    pass
                else:
                    if isinstance(self.expr, Column):
                        raise TypeError("cannot use SparkSQL Column with the normal fill method; use fill.sparksql")
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

def get_n_dim(cls):
    """Histogram dimension

    :returns: dimension of the histogram
    :rtype: int
    """
    if isinstance(cls, histogrammar.Count):
        return 0
    # histogram may have a subhistogram. Extract it and recurse
    if hasattr(cls, 'values'):
        hist = cls.values[0] if cls.values else histogrammar.Count()
    elif hasattr(cls, 'bins'):
        hist = list(cls.bins.values())[0] if cls.bins else histogrammar.Count()
    else:
        hist = histogrammar.Count()
    return 1 + get_n_dim(hist)
