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
from histogrammar.primitives.count import *

class Bin(Factory, Container):
    @staticmethod
    def ed(low, high, entries, values, underflow, overflow, nanflow):
        if entries < 0.0:
            raise ContainerException("entries ($entries) cannot be negative")

        out = Bin(len(values), low, high, None, None, None, underflow, overflow, nanflow)
        out.entries = float(entries)
        out.values = values
        return out

    @staticmethod
    def ing(num, low, high, quantity, selection=unweighted, value=Count(), underflow=Count(), overflow=Count(), nanflow=Count()):
        return Bin(num, low, high, quantity, selection, value, underflow, overflow, nanflow)

    def __init__(self, num, low, high, quantity, selection=unweighted, value=Count(), underflow=Count(), overflow=Count(), nanflow=Count()):
        if low >= high:
            raise ContainerException("low ({}) must be less than high ({})".format(low, high))
        if num < 1:
            raise ContainerException("num ({}) must be least one".format(num))

        self.entries = 0.0
        self.low = float(low)
        self.high = float(high)
        self.quantity = quantity
        self.selection = selection
        if value is None:
            self.values = [None] * num
        else:
            self.values = [value.copy() for i in xrange(num)]
        self.underflow = underflow.copy()
        self.overflow = overflow.copy()
        self.nanflow = nanflow.copy()

    @property
    def zero(self): return Bin(len(self.values), self.low, self.high, self.quantity, self.selection, self.values[0].zero(), self.underflow.zero(), self.overflow.zero(), self.nanflow.zero())

    def __add__(self, other):
        if isinstance(other, Bin):
            if self.low != other.low:
                raise ContainerException("cannot add Bins because low differs ({} vs {})".format(self.low, other.low))
            if self.high != other.high:
                raise ContainerException("cannot add Bins because high differs ({} vs {})".format(self.high, other.high))
            if len(self.values) != len(other.values):
                raise ContainerException("cannot add Bins because nubmer of values differs ({} vs {})".format(len(self.values), len(other.values)))
            if len(self.values) == 0:
                raise ContainerException("cannot add Bins because number of values is zero")

            out = Bin(len(self.values), self.low, self.high, self.quantity, self.selection, self.values[0], self.underflow + other.underflow, self.overflow + other.overflow, self.nanflow + other.nanflow)
            out.values = [x + y for x, y in zip(self.values, other.values)]
            return out

        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    @property
    def num(self): return len(self.values)

    def bin(self, x):
        if self.under(x) or self.over(x) or self.nan(x):
            return -1
        else:
            return int(math.floor(self.num * (x - self.low) / (self.high - self.low)))

    def under(self, x): return not math.isnan(x) and x < self.low
    def over(self, x): return not math.isnan(x) and x >= self.high
    def nan(self, x): return math.isnan(x)

    @property
    def indexes(self): return range(self.num)
    def range(self, index): ((self.high - self.low) * index / self.num + self.low, (self.high - self.low) * (index + 1) / self.num + self.low)

    def fill(self, datum, weight=1.0):
        if self.quantity is None or self.selection is None:
            raise RuntimeException("attempting to fill a container that has no fill rule")

        w = weight * self.selection(datum)

        if w > 0.0:
            q = self.quantity(datum)

            self.entries += w
            if self.under(q):
                self.underflow.fill(datum, w)
            elif self.over(q):
                self.overflow.fill(datum, w)
            elif self.nan(q):
                self.nanflow.fill(datum, w)
            else:
                self.values[self.bin(q)].fill(datum, w)

    def toJsonFragment(self): return {
        "low": self.low,
        "high": self.high,
        "entries": self.entries,
        "values:type": self.values[0].name,
        "values": [x.toJsonFragment() for x in self.values],
        "underflow:type": self.underflow.name,
        "underflow": self.underflow.toJsonFragment(),
        "overflow:type": self.overflow.name,
        "overflow": self.overflow.toJsonFragment(),
        "nanflow:type": self.nanflow.name,
        "nanflow": self.nanflow.toJsonFragment(),
        }

    @staticmethod
    def fromJsonFragment(json):
        if isinstance(json, dict) and set(json.keys()) == set(["low", "high", "entries", "values:type", "values", "underflow:type", "underflow", "overflow:type", "overflow", "nanflow:type", "nanflow"]):
            if isinstance(json["low"], (int, long, float)):
                low = json["low"]
            else:
                raise JsonFormatException(json, "Bin.low")

            if isinstance(json["high"], (int, long, float)):
                high = json["high"]
            else:
                raise JsonFormatException(json, "Bin.high")

            if isinstance(json["entries"], (int, long, float)):
                entries = json["entries"]
            else:
                raise JsonFormatException(json, "Bin.entries")

            if isinstance(json["values:type"], basestring):
                valuesFactory = Factory.registered[json["values:type"]]
            else:
                raise JsonFormatException(json, "Bin.values:type")
            if isinstance(json["values"], list):
                values = [valuesFactory.fromJsonFragment(x) for x in json["values"]]
            else:
                raise JsonFormatException(json, "Bin.values")

            if isinstance(json["underflow:type"], basestring):
                underflowFactory = Factory.registered[json["underflow:type"]]
            else:
                raise JsonFormatException(json, "Bin.underflow:type")
            underflow = underflowFactory.fromJsonFragment(json["underflow"])

            if isinstance(json["overflow:type"], basestring):
                overflowFactory = Factory.registered[json["overflow:type"]]
            else:
                raise JsonFormatException(json, "Bin.overflow:type")
            overflow = overflowFactory.fromJsonFragment(json["overflow"])

            if isinstance(json["nanflow:type"], basestring):
                nanflowFactory = Factory.registered[json["nanflow:type"]]
            else:
                raise JsonFormatException(json, "Bin.nanflow:type")
            nanflow = nanflowFactory.fromJsonFragment(json["nanflow"])

            return Bin.ed(low, high, entries, values, underflow, overflow, nanflow)

        else:
            raise JsonFormatException(json, self.name)
        
    def __repr__(self):
        return "Bin[low={}, high={}, values=[{}..., size={}], underflow={}, overflow={}, nanflow={}]".format(self.low, self.high, repr(self.values[0]), len(self.values), repr(self.underflow), repr(self.overflow), repr(self.nanflow))

    def __eq__(self, other):
        return isinstance(other, Bin) and exact(self.low, other.low) and exact(self.high, other.high) and self.quantity == other.quantity and self.selection == other.selection and exact(self.entries, other.entries) and self.values == other.values and self.underflow == other.underflow and self.overflow == other.overflow and self.nanflow == other.nanflow

    def __hash__(self):
        return hash((self.low, self.high, self.quantity, self.selection, self.entries, self.values, self.underflow, self.overflow, self.nanflow))

Factory.register(Bin)
