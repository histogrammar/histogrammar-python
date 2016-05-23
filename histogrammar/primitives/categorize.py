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
from histogrammar.primitives.count import *

class Categorize(Factory, Container):
    @staticmethod
    def ed(entries, contentType, **pairs):
        if entries < 0.0:
            raise ContainerException("entries ({}) cannot be negative".format(entries))

        out = Categorize(None, contentType)
        out.entries = float(entries)
        out.pairs = pairs
        return out

    @staticmethod
    def ing(quantity, value=Count()):
        return Categorize(quantity, value)

    def __init__(self, quantity, value=Count()):
        self.entries = 0.0
        self.quantity = serializable(quantity)
        self.value = value
        self.pairs = {}
        super(Categorize, self).__init__()

    @property
    def pairsMap(self): return self.pairs
    @property
    def size(self): return len(self.pairs)
    @property
    def keys(self): return self.pairs.keys()
    @property
    def values(self): return self.pairs.values()
    @property
    def keySet(self): return set(self.pairs.keys())

    def __call__(self, x): return self.pairs[x]
    def get(self, x): return self.pairs.get(x)
    def getOrElse(self, x, default): return self.pairs.get(x, default)

    def zero(self): return Categorize(self.quantity, self.value)

    def __add__(self, other):
        if isinstance(other, Categorize):
            out = Categorize(self.quantity, self.value)
            out.entries = self.entries + other.entries
            out.pairs = {}
            for k in self.keySet.union(other.keySet):
                if k in self.pairs and k in other.pairs:
                    out.pairs[k] = self.pairs[k] + other.pairs[k]
                elif k in self.pairs:
                    out.pairs[k] = self.pairs[k].copy()
                else:
                    out.pairs[k] = other.pairs[k].copy()
            return out

        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    def fill(self, datum, weight=1.0):
        if weight > 0.0:
            q = self.quantity(datum)

            if q not in self.pairs:
                self.pairs[q] = self.value.zero()
            self.pairs[q].fill(datum, weight)

            # no possibility of exception from here on out (for rollback)
            self.entries += weight

    @property
    def children(self):
        return [self.value] + self.pairs.values()

    def toJsonFragment(self, suppressName=False):
        if self.value is not None:
            if getattr(self.value, "quantity", None) is not None:
                binsName = self.value.quantity.name
            elif getattr(self.value, "quantityName", None) is not None:
                binsName = self.value.quantityName
            else:
                binsName = None
        elif len(self.bins) > 0:
            if getattr(self.bins[0][1], "quantity") is not None:
                binsName = self.bins[0][1].quantity.name
            elif getattr(self.bins[0][1], "quantityName") is not None:
                binsName = self.bins[0][1].quantityName
            else:
                binsName = None
        else:
            binsName = None

        return maybeAdd({
            "entries": floatToJson(self.entries),
            "type": self.value.name if isinstance(self.value, Container) else self.value,
            "data": {k: v.toJsonFragment(True) for k, v in self.pairs.items()},
            }, **{"name": None if suppressName else self.quantity.name,
                  "data:name": binsName})

    @staticmethod
    def fromJsonFragment(json, nameFromParent=None):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "type", "data"], ["name"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "Categorize.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Categorize.name")

            if isinstance(json["type"], basestring):
                contentType = json["type"]
                factory = Factory.registered[contentType]
            else:
                raise JsonFormatException(json, "Categorize.type")

            if isinstance(json["data"], dict):
                pairs = {k: factory.fromJsonFragment(v) for k, v in json["data"].items()}
            else:
                raise JsonFormatException(json, "Categorize.data")

            out = Categorize.ed(entries, contentType, **pairs)
            out.quantity.name = nameFromParent if name is None else name
            return out

        else:
            raise JsonFormatException(json, "Categorize")

    def __repr__(self):
        return "Categorize[{}..., size={}]".format(self.values[0] if self.size > 0 else self.value, self.size)

    def __eq__(self, other):
        return isinstance(other, Categorize) and exact(self.entries, other.entries) and self.quantity == other.quantity and self.pairs == other.pairs

    def __hash__(self):
        return hash((self.entries, self.quantity, tuple(sorted(self.pairs.items()))))

Factory.register(Categorize)
