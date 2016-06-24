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
from histogrammar.primitives.count import *

class SparselyBin(Factory, Container):
    """Split a quantity into equally spaced bins, creating them whenever their ``entries`` would be non-zero. Exactly one sub-aggregator is filled per datum.

    Use this when you have a distribution of known scale (bin width) but unknown domain (lowest and highest bin index).

    Unlike fixed-domain binning, this aggregator has the potential to use unlimited memory. A large number of *distinct* outliers can generate many unwanted bins.

    Like fixed-domain binning, the bins are indexed by integers, though they are 64-bit and may be negative.
    """

    @staticmethod
    def ed(binWidth, entries, contentType, bins, nanflow, origin):
        """
        * `binWidth` (double) is the width of a bin.
        * `entries` (double) is the number of entries.
        * `contentType` (string) is the value's sub-aggregator type (must be provided to determine type for the case when `bins` is empty).
        * `bins` (map from 64-bit integer to past-tense aggregator) is the non-empty bin indexes and their values.
        * `nanflow` (past-tense aggregator) is the filled nanflow bin.
        * `origin` (double) is the left edge of the bin whose index is zero.
        """
        if not isinstance(binWidth, (int, long, float)):
            raise TypeError("binWidth ({}) must be a number".format(binWidth))
        if not isinstance(entries, (int, long, float)):
            raise TypeError("entries ({}) must be a number".format(entries))
        if not isinstance(contentType, basestring):
            raise TypeError("contentType ({}) must be a string".format(contentType))
        if not isinstance(bins, dict) or not all(isinstance(k, (int, long)) and isinstance(v, Container) for k, v in bins.items()):
            raise TypeError("bins ({}) must be a map from 64-bit integers to Containers".format(bins))
        if not isinstance(nanflow, Container):
            raise TypeError("nanflow ({}) must be a Container".format(nanflow))
        if not isinstance(origin, (int, long, float)):
            raise TypeError("origin ({}) must be a number".format(origin))
        if entries < 0.0:
            raise ValueError("entries ({}) cannot be negative".format(entries))
        if binWidth <= 0.0:
            raise ValueError("binWidth ({}) must be greater than zero".format(binWidth))

        out = SparselyBin(binWidth, None, None, nanflow, origin)
        out.entries = entries
        out.contentType = contentType
        out.bins = bins
        return out.specialize()

    @staticmethod
    def ing(binWidth, quantity, value=Count(), nanflow=Count(), origin=0.0):
        """Synonym for ``__init__``."""
        return SparselyBin(binWidth, quantity, value, nanflow, origin)

    def __init__(self, binWidth, quantity, value=Count(), nanflow=Count(), origin=0.0):
        """
        * `binWidth` (double) is the width of a bin; must be strictly greater than zero.
        * `quantity` (function returning double) computes the quantity of interest from the data.
        * `value` (present-tense aggregator) generates sub-aggregators to put in each bin.
        * `nanflow` (present-tense aggregator) is a sub-aggregator to use for data whose quantity is NaN.
        * `origin` (double) is the left edge of the bin whose index is 0.
        * `entries` (mutable double) is the number of entries, initially 0.0.
        * `bins` (mutable map from 64-bit integer to present-tense aggregator) is the map, probably a hashmap, to fill with values when their `entries` become non-zero.
        """
        if not isinstance(binWidth, (int, long, float)):
            raise TypeError("binWidth ({}) must be a number".format(binWidth))
        if value is not None and not isinstance(value, Container):
            raise TypeError("value ({}) must be a Container".format(value))
        if not isinstance(nanflow, Container):
            raise TypeError("nanflow ({}) must be a Container".format(nanflow))
        if not isinstance(origin, (int, long, float)):
            raise TypeError("origin ({}) must be a number".format(origin))
        if binWidth <= 0.0:
            raise ValueError("binWidth ({}) must be greater than zero".format(binWidth))

        self.binWidth = binWidth
        self.entries = 0.0
        self.quantity = serializable(quantity)
        self.value = value
        if value is not None:
            self.contentType = self.value.name
        self.bins = {}
        self.nanflow = nanflow.copy()
        self.origin = origin
        super(SparselyBin, self).__init__()
        self.specialize()

    def histogram(self):
        out = SparselyBin(self.binWidth, self.quantity, Count(), self.nanflow.copy(), self.origin)
        out.entries = float(self.entries)
        out.contentType = "Count"
        for i, v in self.bins.items():
            out.bins[i] = Count.ed(v.entries)
        return out.specialize()

    @inheritdoc(Container)
    def zero(self): return SparselyBin(self.binWidth, self.quantity, self.value, self.nanflow.zero(), self.origin)

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, SparselyBin):
            if self.binWidth != other.binWidth:
                raise ContainerException("cannot add SparselyBins because binWidth differs ({} vs {})".format(self.binWidth, other.binWidth))
            if self.origin != other.origin:
                raise ContainerException("cannot add SparselyBins because origin differs ({} vs {})".format(self.origin, other.origin))

            out = SparselyBin(self.binWidth, self.quantity, self.value, self.nanflow + other.nanflow)
            out.entries = self.entries + other.entries
            out.bins = self.bins
            for i, v in other.bins.items():
                if i in out.bins:
                    out.bins[i] += v
                else:
                    out.bins[i] = v
            return out.specialize()

        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    @property
    def numFilled(self):
        return len(self.bins)

    @property
    def num(self):
        if len(self.bins) == 0:
            return 0
        else:
            return 1 + self.maxBin - self.minBin
    @property
    def minBin(self):
        if len(self.bins) == 0:
            return None
        else:
            return min(self.bins.keys())
    @property
    def maxBin(self):
        if len(self.bins) == 0:
            return None
        else:
            return max(self.bins.keys())
    @property
    def low(self):
        if len(self.bins) == 0:
            return None
        else:
            return self.minBin * self.binWidth + self.origin
    @property
    def high(self):
        if len(self.bins) == 0:
            return None
        else:
            return (self.maxBin + 1) * self.binWidth + self.origin
    def at(index):
        return self.bins.get(index, None)

    @property
    def indexes(self):
        return sorted(self.keys)

    def range(index):
        return (index * self.binWidth + self.origin, (index + 1) * self.binWidth + self.origin)
    
    def bin(self, x):
        if self.nan(x):
            return MIN_LONG
        else:
            return int(math.floor((x - self.origin) / self.binWidth))

    def nan(self, x): return math.isnan(x)

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()
        if weight > 0.0:
            q = self.quantity(datum)
            if not isinstance(q, (bool, int, long, float)):
                raise TypeError("function return value ({}) must be boolean or number".format(q))

            if self.nan(q):
                self.nanflow.fill(datum, weight)
            else:
                b = self.bin(q)
                if b not in self.bins:
                    self.bins[b] = self.value.zero()
                self.bins[b].fill(datum, weight)

            # no possibility of exception from here on out (for rollback)
            self.entries += weight

    @property
    def children(self):
        return [self.value, self.nanflow] + list(self.bins.values())

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName):
        if isinstance(self.value, Container):
            if getattr(self.value, "quantity", None) is not None:
                binsName = self.value.quantity.name
            elif getattr(self.value, "quantityName", None) is not None:
                binsName = self.value.quantityName
            else:
                binsName = None
        elif len(self.bins) > 0:
            if getattr(list(self.bins.values())[0], "quantity", None) is not None:
                binsName = list(self.bins.values())[0].quantity.name
            elif getattr(list(self.bins.values())[0], "quantityName", None) is not None:
                binsName = list(self.bins.values())[0].quantityName
            else:
                binsName = None
        else:
            binsName = None

        return maybeAdd({
            "binWidth": floatToJson(self.binWidth),
            "entries": floatToJson(self.entries),
            "bins:type": self.value.name if self.value is not None else self.contentType,
            "bins": {str(i): v.toJsonFragment(True) for i, v in self.bins.items()},
            "nanflow:type": self.nanflow.name,
            "nanflow": self.nanflow.toJsonFragment(False),
            "origin": self.origin,
            }, **{"name": None if suppressName else self.quantity.name,
                  "bins:name": binsName})

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["binWidth", "entries", "bins:type", "bins", "nanflow:type", "nanflow", "origin"], ["name", "bins:name"]):
            if isinstance(json["binWidth"], (int, long, float)):
                binWidth = float(json["binWidth"])
            else:
                raise JsonFormatException(json, "SparselyBin.binWidth")

            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "SparselyBin.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "SparselyBin.name")

            if isinstance(json["bins:type"], basestring):
                binsFactory = Factory.registered[json["bins:type"]]
            else:
                raise JsonFormatException(json, "SparselyBin.bins:type")
            if isinstance(json.get("bins:name", None), basestring):
                binsName = json["bins:name"]
            elif json.get("bins:name", None) is None:
                binsName = None
            else:
                raise JsonFormatException(json["bins:name"], "SparselyBin.bins:name")
            if isinstance(json["bins"], dict):
                for i in json["bins"]:
                    try:
                        int(i)
                    except ValueError:
                        raise JsonFormatException(i, "SparselyBin.bins key must be an integer")

                bins = {int(i): binsFactory.fromJsonFragment(v, binsName) for i, v in json["bins"].items()}

            else:
                raise JsonFormatException(json, "SparselyBin.bins")

            if isinstance(json["nanflow:type"], basestring):
                nanflowFactory = Factory.registered[json["nanflow:type"]]
            else:
                raise JsonFormatException(json, "Bin.nanflow:type")
            nanflow = nanflowFactory.fromJsonFragment(json["nanflow"], None)

            if isinstance(json["origin"], (int, long, float)):
                origin = json["origin"]
            else:
                raise JsonFormatException(json, "SparselyBin.origin")

            out = SparselyBin.ed(binWidth, entries, json["bins:type"], bins, nanflow, origin)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        else:
            raise JsonFormatException(json, "SparselyBin")
        
    def __repr__(self):
        if self.bins is None:
            contentType = self.contentType
        elif len(self.bins) == 0:
            contentType = self.value.name
        else:
            contentType = repr(min(self.bins.items())[1])
        return "<SparselyBin binWidth={} bins={} nanflow={}>".format(self.binWidth, self.value.name if self.value is not None else self.contentType, self.nanflow.name)

    def __eq__(self, other):
        return isinstance(other, SparselyBin) and numeq(self.binWidth, other.binWidth) and self.quantity == other.quantity and numeq(self.entries, other.entries) and self.bins == other.bins and self.nanflow == other.nanflow and numeq(self.origin, other.origin)

    def __hash__(self):
        return hash((self.binWidth, self.quantity, self.entries, tuple(sorted(self.bins.items())), self.nanflow, self.origin))

Factory.register(SparselyBin)
