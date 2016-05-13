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
    @staticmethod
    def ed(binWidth, entries, contentType, bins, nanflow, origin):
        if entries < 0.0:
            raise ContainerException("entries ({}) cannot be negative".format(entries))

        out = SparselyBin(binWidth, None, None, nanflow, origin)
        out.entries = entries
        out.contentType = contentType
        out.bins = bins
        return out

    @staticmethod
    def ing(binWidth, quantity, value=Count(), nanflow=Count(), origin=0.0):
        return SparselyBin(binWidth, quantity, value, nanflow, origin)

    def __init__(self, binWidth, quantity, value=Count(), nanflow=Count(), origin=0.0):
        if binWidth <= 0.0:
            raise ContainerException("binWidth ({}) must be greater than zero".format(binWidth))

        self.binWidth = binWidth
        self.entries = 0.0
        self.quantity = serializable(quantity)
        self.value = value
        self.bins = {}
        self.nanflow = nanflow.copy()
        self.origin = origin
        super(SparselyBin, self).__init__()

    def zero(self): return SparselyBin(self.binWidth, self.quantity, self.value, self.nanflow.zero(), self.origin)

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
            return out

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
            return min(*self.bins.keys())
    @property
    def maxBin(self):
        if len(self.bins) == 0:
            return None
        else:
            return max(*self.bins.keys())
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

    def fill(self, datum, weight=1.0):
        if weight > 0.0:
            q = self.quantity(datum)

            if self.nan(q):
                self.nanflow.fill(datum, weight)
            else:
                b = self.bin(q)
                if b not in self.bins:
                    self.bins[b] = self.value.copy()
                self.bins[b].fill(datum, weight)

            # no possibility of exception from here on out (for rollback)
            self.entries += weight

    def toJsonFragment(self): return maybeAdd({
        "binWidth": floatToJson(self.binWidth),
        "entries": floatToJson(self.entries),
        "bins:type": self.value.name if self.value is not None else self.contentType,
        "bins": {str(i): v.toJsonFragment() for i, v in self.bins.items()},
        "nanflow:type": self.nanflow.name,
        "nanflow": self.nanflow.toJsonFragment(),
        "origin": self.origin,
        }, name=self.quantity.name)

    @staticmethod
    def fromJsonFragment(json):
        if isinstance(json, dict) and hasKeys(json.keys(), ["binWidth", "entries", "bins:type", "bins", "nanflow:type", "nanflow", "origin"], ["name"]):
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
            if isinstance(json["bins"], dict):
                for i in json["bins"]:
                    try:
                        int(i)
                    except ValueError:
                        raise JsonFormatException(i, "SparselyBin.bins key must be an integer")

                bins = {int(i): binsFactory.fromJsonFragment(v) for i, v in json["bins"].items()}

            else:
                raise JsonFormatException(json, "SparselyBin.bins")

            if isinstance(json["nanflow:type"], basestring):
                nanflowFactory = Factory.registered[json["nanflow:type"]]
            else:
                raise JsonFormatException(json, "Bin.nanflow:type")
            nanflow = nanflowFactory.fromJsonFragment(json["nanflow"])

            if isinstance(json["origin"], (int, long, float)):
                origin = json["origin"]
            else:
                raise JsonFormatException(json, "SparselyBin.origin")

            out = SparselyBin.ed(binWidth, entries, json["bins:type"], bins, nanflow, origin)
            out.quantity.name = name
            return out

        else:
            raise JsonFormatException(json, "SparselyBin")
        
    def __repr__(self):
        if self.bins is None:
            contentType = self.contentType
        elif len(self.bins) == 0:
            contentType = self.value.name
        else:
            contentType = repr(min(self.bins.items())[1])
        return "SparselyBin[binWidth={}, bins=[{}, size={}], nanflow={}, origin={}]".format(self.binWidth, contentType, len(self.bins), self.nanflow, self.origin)

    def __eq__(self, other):
        return isinstance(other, SparselyBin) and exact(self.binWidth, other.binWidth) and self.quantity == other.quantity and exact(self.entries, other.entries) and self.bins == other.bins and self.nanflow == other.nanflow and self.origin == other.origin

    def __hash__(self):
        return hash((self.binWidth, self.quantity, self.entries, tuple(sorted(self.bins.items())), self.nanflow, self.origin))

Factory.register(SparselyBin)
