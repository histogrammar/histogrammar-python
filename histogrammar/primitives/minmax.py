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

class Minimize(Factory, Container):
    """Find the minimum value of a given quantity. If no data are observed, the result is NaN.

    Unlike :doc:`Quantile <histogrammar.primitives.quantile.Quantile>` with a target of 0, Minimize is exact.
    """

    @staticmethod
    def ed(entries, min):
        """Create a Minimize that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            min (float): the lowest value of the quantity observed or NaN if no data were observed.
        """
        if not isinstance(entries, (int, long, float)):
            raise TypeError("entries ({}) must be a number".format(entries))
        if not isinstance(min, (int, long, float)):
            raise TypeError("min ({}) must be a number".format(min))
        if entries < 0.0:
            raise ValueError("entries ({}) cannot be negative".format(entries))
        out = Minimize(None)
        out.entries = float(entries)
        out.min = float(min)
        return out.specialize()

    @staticmethod
    def ing(quantity):
        """Synonym for ``__init__``."""
        return Minimize(quantity)

    def __init__(self, quantity):
        """Create a Minimize that is capable of being filled and added.

        Parameters:
            quantity (function returning float): computes the quantity of interest from the data.

        Other parameters:
            entries (float): the number of entries, initially 0.0. # 
            min (float): the lowest value of the quantity observed, initially NaN.
        """
        self.quantity = serializable(quantity)
        self.entries = 0.0
        self.min = float("nan")
        super(Minimize, self).__init__()
        self.specialize()

    @inheritdoc(Container)
    def zero(self): return Minimize(self.quantity)

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Minimize):
            out = Minimize(self.quantity)
            out.entries = self.entries + other.entries
            out.min = minplus(self.min, other.min)
            return out.specialize()
        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return []

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()
        if weight > 0.0:
            q = self.quantity(datum)
            if not isinstance(q, (bool, int, long, float)):
                raise TypeError("function return value ({}) must be boolean or number".format(q))

            # no possibility of exception from here on out (for rollback)
            self.entries += weight
            if math.isnan(self.min) or q < self.min:
                self.min = q

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName): return maybeAdd({
        "entries": floatToJson(self.entries),
        "min": floatToJson(self.min),
        }, name=(None if suppressName else self.quantity.name))

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "min"], ["name"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json["entries"], "Minimize.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Minimize.name")

            if json["min"] in ("nan", "inf", "-inf") or isinstance(json["min"], (int, long, float)):
                min = float(json["min"])
            else:
                raise JsonFormatException(json["min"], "Minimize.min")

            out = Minimize.ed(entries, min)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        else:
            raise JsonFormatException(json, "Minimize")
        
    def __repr__(self):
        return "<Minimize min={}>".format(self.min)

    def __eq__(self, other):
        return isinstance(other, Minimize) and self.quantity == other.quantity and numeq(self.entries, other.entries) and numeq(self.min, other.min)

    def __hash__(self):
        return hash((self.quantity, self.entries, self.min))

Factory.register(Minimize)

class Maximize(Factory, Container):
    """Find the maximum value of a given quantity. If no data are observed, the result is NaN.

    Unlike :doc:`Quantile <histogrammar.primitives.quantile.Quantile>` with a target of 1, Maximize is exact.
    """

    @staticmethod
    def ed(entries, max):
        """Create a Maximize that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            max (float): the highest value of the quantity observed or NaN if no data were observed.
        """
        if not isinstance(entries, (int, long, float)):
            raise TypeError("entries ({}) must be a number".format(entries))
        if not isinstance(max, (int, long, float)):
            raise TypeError("max ({}) must be a number".format(max))
        if entries < 0.0:
            raise ValueError("entries ({}) cannot be negative".format(entries))
        out = Maximize(None)
        out.entries = float(entries)
        out.max = float(max)
        return out.specialize()

    @staticmethod
    def ing(quantity):
        """Synonym for ``__init__``."""
        return Maximize(quantity)

    def __init__(self, quantity):
        """Create a Maximize that is capable of being filled and added.

        Parameters:
            quantity (function returning float): computes the quantity of interest from the data.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
            max (float): the highest value of the quantity observed, initially NaN.
        """
        self.quantity = serializable(quantity)
        self.entries = 0.0
        self.max = float("nan")
        super(Maximize, self).__init__()
        self.specialize()

    @inheritdoc(Container)
    def zero(self): return Maximize(self.quantity)

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Maximize):
            out = Maximize(self.quantity)
            out.entries = self.entries + other.entries
            out.max = maxplus(self.max, other.max)
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
            if math.isnan(self.max) or q > self.max:
                self.max = q

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return []

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName): return maybeAdd({
        "entries": floatToJson(self.entries),
        "max": floatToJson(self.max),
        }, name=(None if suppressName else self.quantity.name))

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "max"], ["name"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json["entries"], "Maximize.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Maximize.name")

            if json["max"] in ("nan", "inf", "-inf") or isinstance(json["max"], (int, long, float)):
                max = float(json["max"])
            else:
                raise JsonFormatException(json["max"], "Maximize.max")

            out = Maximize.ed(entries, max)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        else:
            raise JsonFormatException(json, "Maximize")
        
    def __repr__(self):
        return "<Maximize max={}>".format(self.max)

    def __eq__(self, other):
        return isinstance(other, Maximize) and self.quantity == other.quantity and numeq(self.entries, other.entries) and numeq(self.max, other.max)

    def __hash__(self):
        return hash((self.quantity, self.entries, self.max))

Factory.register(Maximize)
