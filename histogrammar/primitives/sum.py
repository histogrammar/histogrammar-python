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

from histogrammar.defs import *
from histogrammar.util import *

class Sum(Factory, Container):
    """Accumulate the (weighted) sum of a given quantity, calculated from the data.

    Sum differs from :doc:`Count <histogrammar.primitives.count.Count>` in that it computes a quantity on the spot, rather than percolating a product of weight metadata from nested primitives. Also unlike weights, the sum can add both positive and negative quantities (weights are always non-negative).
    """

    @staticmethod
    def ed(entries, sum):
        """Create a Sum that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            sum (float): the sum.
        """
        if not isinstance(entries, (int, long, float)):
            raise TypeError("entries ({}) must be a number".format(entries))
        if not isinstance(sum, (int, long, float)):
            raise TypeError("sum ({}) must be a number".format(sum))
        if entries < 0.0:
            raise ValueError("entries ({}) cannot be negative".format(entries))
        out = Sum(None)
        out.entries = float(entries)
        out.sum = float(sum)
        return out.specialize()

    @staticmethod
    def ing(quantity):
        """Synonym for ``__init__``."""
        return Sum(quantity)

    def __init__(self, quantity):
        """Create a Sum that is capable of being filled and added.

        Parameters:
            quantity (function returning float): computes the quantity of interest from the data.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
            sum (float): the running sum, initially 0.0.
        """
        self.quantity = serializable(quantity)
        self.entries = 0.0
        self.sum = 0.0
        super(Sum, self).__init__()
        self.specialize()

    @inheritdoc(Container)
    def zero(self): return Sum(self.quantity)

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Sum):
            out = Sum(self.quantity)
            out.entries = self.entries + other.entries
            out.sum = self.sum + other.sum
            return out.specialize()
        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()
        if weight > 0.0:
            q = self.quantity(datum)
            if not isinstance(q, (bool, int, long, float)):
                raise TypeError("function return value ({}) must be boolean or number".format(q))

            # no possibility of exception from here on out (for rollback)
            self.entries += weight
            self.sum += q * weight

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return []

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName): return maybeAdd({
        "entries": floatToJson(self.entries),
        "sum": floatToJson(self.sum),
        }, name=(None if suppressName else self.quantity.name))

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "sum"], ["name"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json["entries"], "Sum.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Sum.name")

            if isinstance(json["sum"], (int, long, float)):
                sum = float(json["sum"])
            else:
                raise JsonFormatException(json["sum"], "Sum.sum")

            out = Sum.ed(entries, sum)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        else:
            raise JsonFormatException(json, "Sum")
        
    def __repr__(self):
        return "<Sum sum={}>".format(self.sum)

    def __eq__(self, other):
        return isinstance(other, Sum) and self.quantity == other.quantity and numeq(self.entries, other.entries) and numeq(self.sum, other.sum)

    def __hash__(self):
        return hash((self.quantity, self.entries, self.sum))

Factory.register(Sum)
