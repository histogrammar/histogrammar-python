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

from histogrammar.defs import *
from histogrammar.util import *
from histogrammar.primitives.count import *

class Bin(Factory, Container):
    """Split a quantity into equally spaced bins between a low and high threshold and fill exactly one bin per datum.

    When composed with :doc:`Count <histogrammar.primitives.count.Count>`, this produces a standard histogram:

    ::

        Bin.ing(100, 0, 10, fill_x, Count.ing())

    and when nested, it produces a two-dimensional histogram:

    ::

        Bin.ing(100, 0, 10, fill_x,
          Bin.ing(100, 0, 10, fill_y, Count.ing()))

    Combining with [Deviate](#deviate-mean-and-variance) produces a physicist's "profile plot:"

    ::

        Bin.ing(100, 0, 10, fill_x, Deviate.ing(fill_y))

    and so on.
    """

    @staticmethod
    def ed(low, high, entries, values, underflow, overflow, nanflow):
        """Create a Bin that is only capable of being added.

        Parameters:
            low (float): the minimum-value edge of the first bin.
            high (float): the maximum-value edge of the last bin; must be strictly greater than `low`.
            entries (float): the number of entries.
            values (list of :doc:`Container <histogrammar.defs.Container>`): the filled sub-aggregators, one for each bin.
            underflow (:doc:`Container <histogrammar.defs.Container>`): the filled underflow bin.
            overflow (:doc:`Container <histogrammar.defs.Container>`): the filled overflow bin.
            nanflow (:doc:`Container <histogrammar.defs.Container>`): is the filled nanflow bin.
        """
        if not isinstance(low, (int, long, float)):
            raise TypeError("low ({0}) must be a number".format(low))
        if not isinstance(high, (int, long, float)):
            raise TypeError("high ({0}) must be a number".format(high))
        if not isinstance(entries, (int, long, float)):
            raise TypeError("entries ({0}) must be a number".format(entries))
        if not isinstance(values, (list, tuple)) and not all(isinstance(v, Container) for v in values):
            raise TypeError("values ({0}) must be a list of Containers".format(values))
        if not isinstance(underflow, Container):
            raise TypeError("underflow ({0}) must be a Container".format(underflow))
        if not isinstance(overflow, Container):
            raise TypeError("overflow ({0}) must be a Container".format(overflow))
        if not isinstance(nanflow, Container):
            raise TypeError("nanflow ({0}) must be a Container".format(nanflow))
        if low >= high:
            raise ValueError("low ({0}) must be less than high ({1})".format(low, high))
        if entries < 0.0:
            raise ValueError("entries ({0}) cannot be negative".format(entries))
        if len(values) < 1:
            raise ValueError("values ({0}) must have at least one element".format(values))

        out = Bin(len(values), low, high, None, None, underflow, overflow, nanflow)
        out.entries = float(entries)
        out.values = values
        return out.specialize()

    @staticmethod
    def ing(num, low, high, quantity, value=Count(), underflow=Count(), overflow=Count(), nanflow=Count()):
        """Synonym for ``__init__``."""
        return Bin(num, low, high, quantity, value, underflow, overflow, nanflow)

    def __init__(self, num, low, high, quantity, value=Count(), underflow=Count(), overflow=Count(), nanflow=Count()):
        """Create a Bin that is capable of being filled and added.

        Parameters:
            num (int): the number of bins; must be at least one.
            low (float): the minimum-value edge of the first bin.
            high (float): the maximum-value edge of the last bin; must be strictly greater than `low`.
            quantity (function returning float): computes the quantity of interest from the data.
            value (:doc:`Container <histogrammar.defs.Container>`): generates sub-aggregators to put in each bin.
            underflow (:doc:`Container <histogrammar.defs.Container>`): a sub-aggregator to use for data whose quantity is less than `low`.
            overflow (:doc:`Container <histogrammar.defs.Container>`): a sub-aggregator to use for data whose quantity is greater than or equal to `high`.
            nanflow (:doc:`Container <histogrammar.defs.Container>`): a sub-aggregator to use for data whose quantity is NaN.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
            values (list of :doc:`Container <histogrammar.defs.Container>`): the sub-aggregators in each bin.
        """

        if not isinstance(num, (int, long)):
            raise TypeError("num ({0}) must be an integer".format(num))
        if not isinstance(low, (int, long, float)):
            raise TypeError("low ({0}) must be a number".format(low))
        if not isinstance(high, (int, long, float)):
            raise TypeError("high ({0}) must be a number".format(high))
        if value is not None and not isinstance(value, Container):
            raise TypeError("value ({0}) must be a Container".format(value))
        if not isinstance(underflow, Container):
            raise TypeError("underflow ({0}) must be a Container".format(underflow))
        if not isinstance(overflow, Container):
            raise TypeError("overflow ({0}) must be a Container".format(overflow))
        if not isinstance(nanflow, Container):
            raise TypeError("nanflow ({0}) must be a Container".format(nanflow))
        if num < 1:
            raise ValueError("num ({0}) must be least one".format(num))
        if low >= high:
            raise ValueError("low ({0}) must be less than high ({1})".format(low, high))

        self.entries = 0.0
        self.low = float(low)
        self.high = float(high)
        self.quantity = serializable(quantity)
        if value is None:
            self.values = [None] * num
        else:
            self.values = [value.zero() for i in xrange(num)]
        self.underflow = underflow.copy()
        self.overflow = overflow.copy()
        self.nanflow = nanflow.copy()
        super(Bin, self).__init__()
        self.specialize()

    def histogram(self):
        """Return a plain histogram by converting all sub-aggregator values into :doc:`Counts <histogrammar.primitives.count.Count>`."""
        out = Bin(len(self.values), self.low, self.high, self.quantity, None, self.underflow.copy(), self.overflow.copy(), self.nanflow.copy())
        out.entries = float(self.entries)
        for i, v in enumerate(self.values):
            out.values[i] = Count.ed(v.entries)
        return out.specialize()

    @inheritdoc(Container)
    def zero(self): return Bin(len(self.values), self.low, self.high, self.quantity, self.values[0].zero(), self.underflow.zero(), self.overflow.zero(), self.nanflow.zero())

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Bin):
            if self.low != other.low:
                raise ContainerException("cannot add Bins because low differs ({0} vs {1})".format(self.low, other.low))
            if self.high != other.high:
                raise ContainerException("cannot add Bins because high differs ({0} vs {1})".format(self.high, other.high))
            if len(self.values) != len(other.values):
                raise ContainerException("cannot add Bins because nubmer of values differs ({0} vs {1})".format(len(self.values), len(other.values)))
            if len(self.values) == 0:
                raise ContainerException("cannot add Bins because number of values is zero")

            out = Bin(len(self.values), self.low, self.high, self.quantity, self.values[0], self.underflow + other.underflow, self.overflow + other.overflow, self.nanflow + other.nanflow)
            out.entries = self.entries + other.entries
            out.values = [x + y for x, y in zip(self.values, other.values)]
            return out.specialize()

        else:
            raise ContainerException("cannot add {0} and {1}".format(self.name, other.name))

    @property
    def num(self):
        """Number of bins."""
        return len(self.values)

    def bin(self, x):
        """Find the bin index associated with numerical value ``x``.
        
        @return -1 if ``x`` is out of range; the bin index otherwise.
        """
        if self.under(x) or self.over(x) or self.nan(x):
            return -1
        else:
            return int(math.floor(self.num * (x - self.low) / (self.high - self.low)))

    def under(self, x):
        """Return ``true`` iff ``x`` is in the underflow region (less than ``low``)."""
        return not math.isnan(x) and x < self.low

    def over(self, x):
        """Return ``true`` iff ``x`` is in the overflow region (greater than ``high``)."""
        return not math.isnan(x) and x >= self.high

    def nan(self, x):
        """Return ``true`` iff ``x`` is in the nanflow region (equal to ``NaN``)."""
        return math.isnan(x)

    @property
    def indexes(self):
        """Get a sequence of valid indexes."""
        return range(self.num)

    def range(self, index):
        """Get the low and high edge of a bin (given by index number)."""
        return ((self.high - self.low) * index / self.num + self.low, (self.high - self.low) * (index + 1) / self.num + self.low)

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()
        if weight > 0.0:
            q = self.quantity(datum)
            if not isinstance(q, (bool, int, long, float)):
                raise TypeError("function return value ({0}) must be boolean or number".format(q))

            if self.under(q):
                self.underflow.fill(datum, weight)
            elif self.over(q):
                self.overflow.fill(datum, weight)
            elif self.nan(q):
                self.nanflow.fill(datum, weight)
            else:
                self.values[self.bin(q)].fill(datum, weight)

            # no possibility of exception from here on out (for rollback)
            self.entries += weight

    def fillnp(self, data, weight=1.0):
        """Increment the aggregator by providing a one-dimensional Numpy array of ``data`` to the fill rule with given ``weight`` (number or array).

        This primitive is optimized with Numpy.

        The container is changed in-place.
        """
        self._checkForCrossReferences()

        import numpy
        if not isinstance(data, numpy.ndarray):
            data = numpy.array(data)
        assert len(data.shape) == 1
        length = data.shape[0]

        q = self.quantity(data)
        assert isinstance(q, numpy.ndarray)
        assert len(q.shape) == 1
        assert q.shape[0] == length

        if isinstance(weight, numpy.ndarray):
            assert len(weight.shape) == 1
            assert weight.shape[0] == length

        selection = numpy.isnan(q)
        self.nanflow.fillnp(data[selection], weight[selection] if isinstance(weight, numpy.ndarray) else weight)
        
        numpy.bitwise_not(selection, selection)
        data = data[selection]
        q = q[selection]
        if isinstance(weight, numpy.ndarray):
            weight = weight[selection]

        q = numpy.array(q, dtype=float)
        numpy.subtract(q, self.low, q)
        numpy.multiply(q, self.num, q)
        numpy.divide(q, self.high - self.low, q)

        selection = numpy.empty(q.shape, dtype=numpy.bool)        

        numpy.less(q, 0.0, selection)
        self.underflow.fillnp(data[selection], weight[selection] if isinstance(weight, numpy.ndarray) else weight)

        numpy.greater_equal(q, self.num, selection)
        self.overflow.fillnp(data[selection], weight[selection] if isinstance(weight, numpy.ndarray) else weight)

        numpy.floor(q, q)
        q = numpy.array(q, dtype=int)

        for index, value in enumerate(self.values):
            numpy.equal(q, index, selection)
            value.fillnp(data[selection], weight[selection] if isinstance(weight, numpy.ndarray) else weight)

        if isinstance(weight, numpy.ndarray):
            self.entries += float(weight[weight > 0.0].sum())
        elif weight > 0.0:
            self.entries += float(weight * length)

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return [self.underflow, self.overflow, self.nanflow] + self.values

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName):
        if getattr(self.values[0], "quantity", None) is not None:
            binsName = self.values[0].quantity.name
        elif getattr(self.values[0], "quantityName", None) is not None:
            binsName = self.values[0].quantityName
        else:
            binsName = None

        return maybeAdd({
            "low": floatToJson(self.low),
            "high": floatToJson(self.high),
            "entries": floatToJson(self.entries),
            "values:type": self.values[0].name,
            "values": [x.toJsonFragment(True) for x in self.values],
            "underflow:type": self.underflow.name,
            "underflow": self.underflow.toJsonFragment(False),
            "overflow:type": self.overflow.name,
            "overflow": self.overflow.toJsonFragment(False),
            "nanflow:type": self.nanflow.name,
            "nanflow": self.nanflow.toJsonFragment(False),
            }, **{"name": None if suppressName else self.quantity.name,
                  "values:name": binsName})

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["low", "high", "entries", "values:type", "values", "underflow:type", "underflow", "overflow:type", "overflow", "nanflow:type", "nanflow"], ["name", "values:name"]):
            if isinstance(json["low"], (int, long, float)):
                low = float(json["low"])
            else:
                raise JsonFormatException(json, "Bin.low")

            if isinstance(json["high"], (int, long, float)):
                high = float(json["high"])
            else:
                raise JsonFormatException(json, "Bin.high")

            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "Bin.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Bin.name")

            if isinstance(json["values:type"], basestring):
                valuesFactory = Factory.registered[json["values:type"]]
            else:
                raise JsonFormatException(json, "Bin.values:type")
            if isinstance(json.get("values:name", None), basestring):
                valuesName = json["values:name"]
            elif json.get("values:name", None) is None:
                valuesName = None
            else:
                raise JsonFormatException(json["values:name"], "Bin.values:name")
            if isinstance(json["values"], list):
                values = [valuesFactory.fromJsonFragment(x, valuesName) for x in json["values"]]
            else:
                raise JsonFormatException(json, "Bin.values")

            if isinstance(json["underflow:type"], basestring):
                underflowFactory = Factory.registered[json["underflow:type"]]
            else:
                raise JsonFormatException(json, "Bin.underflow:type")
            underflow = underflowFactory.fromJsonFragment(json["underflow"], None)

            if isinstance(json["overflow:type"], basestring):
                overflowFactory = Factory.registered[json["overflow:type"]]
            else:
                raise JsonFormatException(json, "Bin.overflow:type")
            overflow = overflowFactory.fromJsonFragment(json["overflow"], None)

            if isinstance(json["nanflow:type"], basestring):
                nanflowFactory = Factory.registered[json["nanflow:type"]]
            else:
                raise JsonFormatException(json, "Bin.nanflow:type")
            nanflow = nanflowFactory.fromJsonFragment(json["nanflow"], None)

            out = Bin.ed(low, high, entries, values, underflow, overflow, nanflow)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        else:
            raise JsonFormatException(json, "Bin")
        
    def __repr__(self):
        return "<Bin num={0} low={1} high={2} values={3} underflow={4} overflow={5} nanflow={6}>".format(len(self.values), self.low, self.high, self.values[0].name, self.underflow.name, self.overflow.name, self.nanflow.name)

    def __eq__(self, other):
        return isinstance(other, Bin) and numeq(self.low, other.low) and numeq(self.high, other.high) and self.quantity == other.quantity and numeq(self.entries, other.entries) and self.values == other.values and self.underflow == other.underflow and self.overflow == other.overflow and self.nanflow == other.nanflow

    def __hash__(self):
        return hash((self.low, self.high, self.quantity, self.entries, tuple(self.values), self.underflow, self.overflow, self.nanflow))

Factory.register(Bin)
