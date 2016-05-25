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

class Stack(Factory, Container):
    @staticmethod
    def ed(entries, *cuts):
        if isinstance(entries, (int, long, float)):
            if entries < 0.0:
                raise ContainerException("entries ({}) cannot be negative".format(entries))
            out = Stack(None, None, *cuts)
            out.entries = float(entries)
            return out

        elif isinstance(entries, Container) and all(isinstance(x, Container) for x in cuts):
            ys = [entries] + list(cuts)
            entries = sum(y.entries for y in ys)
            cuts = []
            for i in xrange(len(ys)):
                cuts.append((float("nan"), reduce(lambda a, b: a + b, ys[i:])))
            return Stack.ed(entries, *cuts)

        else:
            raise TypeError("wrong arguments for Stack.ed")

    @staticmethod
    def ing(quantity, value, *cuts):
        return Stack(quantity, value, *cuts)

    def __init__(self, quantity, value, *cuts):
        self.entries = 0.0
        self.quantity = serializable(quantity)
        if value is None:
            self.cuts = cuts
        else:
            self.cuts = tuple((float(x), value.zero()) for x in (float("-inf"),) + cuts)
        super(Stack, self).__init__()

    @property
    def thresholds(self): return [k for k, v in self.cuts]
    @property
    def values(self): return [v for k, v in self.cuts]

    def zero(self):
        return Stack(self.quantity, None, *[(x, x.zero()) for x in cuts])

    def __add__(self, other):
        if isinstance(other, Stack):
            if self.thresholds != other.thresholds:
                raise ContainerException("cannot add Stack because cut thresholds differ")

            out = Stack(self.quantity, None, *[(k1, v1 + v2) for ((k1, v1), (k2, v2)) in zip(self.cuts, other.cuts)])
            out.entries = self.entries + other.entries
            return out

        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    def fill(self, datum, weight=1.0):
        if weight > 0.0:
            value = self.quantity(datum)
            for threshold, sub in self.cuts:
                if value >= threshold:
                    sub.fill(datum, weight)

            # no possibility of exception from here on out (for rollback)
            self.entries += weight

    @property
    def children(self):
        return self.values

    def toJsonFragment(self, suppressName):
        if getattr(self.cuts[0][1], "quantity", None) is not None:
            binsName = self.cuts[0][1].quantity.name
        elif getattr(self.cuts[0][1], "quantityName", None) is not None:
            binsName = self.cuts[0][1].quantityName
        else:
            binsName = None

        return maybeAdd({
            "entries": floatToJson(self.entries),
            "type": self.cuts[0][1].name,
            "data": [{"atleast": floatToJson(atleast), "data": sub.toJsonFragment(True)} for atleast, sub in self.cuts],
            }, **{"name": None if suppressName else self.quantity.name,
                  "data:name": binsName})

    @staticmethod
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "type", "data"], ["name", "data:name"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "Stack.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Stack.name")

            if isinstance(json["type"], basestring):
                factory = Factory.registered[json["type"]]
            else:
                raise JsonFormatException(json, "Stack.type")

            if isinstance(json.get("data:name", None), basestring):
                dataName = json["data:name"]
            elif json.get("data:name", None) is None:
                dataName = None
            else:
                raise JsonFormatException(json["data:name"], "Stack.data:name")

            if isinstance(json["data"], list):
                cuts = []
                for i, elementPair in enumerate(json["data"]):
                    if isinstance(elementPair, dict) and hasKeys(elementPair.keys(), ["atleast", "data"]):
                        if elementPair["atleast"] not in ("nan", "inf", "-inf") and not isinstance(elementPair["atleast"], (int, long, float)):
                            raise JsonFormatException(json, "Stack.data {} atleast".format(i))

                        cuts.append((float(elementPair["atleast"]), factory.fromJsonFragment(elementPair["data"], dataName)))

                    else:
                        raise JsonFormatException(json, "Stack.data {}".format(i))

                out = Stack.ed(entries, *cuts)
                out.quantity.name = nameFromParent if name is None else name
                return out

            else:
                raise JsonFormatException(json, "Stack.data")

        else:
            raise JsonFormatException(json, "Stack")

    def __repr__(self):
        return "Stack[{}, thresholds=[{}]]".format(self.cuts[0][1], ", ".join(map(str, self.thresholds)))

    def __eq__(self, other):
        return isinstance(other, Stack) and exact(self.entries, other.entries) and self.quantity == other.quantity and self.cuts == other.cuts

    def __hash__(self):
        return hash((self.entries, self.quantity, self.cuts))

Factory.register(Stack)
