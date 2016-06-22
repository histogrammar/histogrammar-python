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
import json

from histogrammar.defs import *
from histogrammar.util import *

class Bag(Factory, Container):
    @staticmethod
    def ed(entries, values):
        if not isinstance(entries, (int, long, float)):
            raise TypeError("entries ({}) must be a number".format(entries))
        if not isinstance(values, dict) and not all(isinstance(k, (int, long, float)) for k, v in values.items()):
            raise TypeError("values ({}) must be a dict from numbers to range type".format(values))
        if entries < 0.0:
            raise ValueError("entries ({}) cannot be negative".format(entries))
        out = Bag(None)
        out.entries = float(entries)
        out.values = values
        return out.specialize()

    @staticmethod
    def ing(quantity):
        return Bag(quantity)

    def __init__(self, quantity):
        self.quantity = serializable(quantity)
        self.entries = 0.0
        self.values = {}
        super(Bag, self).__init__()
        self.specialize()

    def zero(self): return Bag(self.quantity)

    def __add__(self, other):
        if isinstance(other, Bag):
            out = Bag(self.quantity)

            out.entries = self.entries + other.entries

            out.values = dict(self.values)
            for value, count in other.values.items():
                if value in out.values:
                    out.values[value] += count
                else:
                    out.values[value] = count

            return out.specialize()

        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()
        if weight > 0.0:
            q = self.quantity(datum)
            if not isinstance(q, (bool, int, long, float, basestring)) and not (isinstance(q, (list, tuple)) and all(isinstance(qi, (int, long, float)) for qi in q)):
                raise TypeError("function return value ({}) must be boolean, number, string, or list/tuple of numbers".format(q))
            if isinstance(q, list):
                q = tuple(q)

            # no possibility of exception from here on out (for rollback)
            self.entries += weight
            if q in self.values:
                self.values[q] += weight
            else:
                self.values[q] = weight

    @property
    def children(self):
        return []

    def toJsonFragment(self, suppressName): return maybeAdd({
        "entries": floatToJson(self.entries),
        "values": [{"w": n, "v": v} for v, n in sorted(self.values.items())],
        }, name=(None if suppressName else self.quantity.name))

    @staticmethod
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "values"], ["name"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = json["entries"]
            else:
                raise JsonFormatException(json["entries"], "Bag.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Bag.name")

            if json["values"] is None:
                values = None

            elif json["values"] is None or isinstance(json["values"], list):
                values = {}
                for i, nv in enumerate(json["values"]):
                    if isinstance(nv, dict) and hasKeys(nv.keys(), ["w", "v"]):
                        if isinstance(nv["w"], (int, long, float)):
                            n = float(nv["w"])
                        else:
                            raise JsonFormatException(nv["w"], "Bag.values {} n".format(i))

                        if isinstance(nv["v"], basestring):
                            v = nv["v"]
                        elif isinstance(nv["v"], (int, long, float)):
                            v = float(nv["v"])
                        elif isinstance(nv["v"], (list, tuple)):
                            for j, d in enumerate(nv["v"]):
                                if not isinstance(d, (int, long, float)):
                                    raise JsonFormatException(d, "Bag.values {} v {}".format(i, j))
                            v = tuple(map(float, nv["v"]))
                        else:
                            raise JsonFormatException(nv["v"], "Bag.values {} v".format(i))

                        values[v] = n

                    else:
                        raise JsonFormatException(nv, "Bag.values {}".format(i))

            elif json["values"] is None:
                values = None

            else:
                raise JsonFormatException(json["values"], "Bag.values")

            out = Bag.ed(entries, values)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        else:
            raise JsonFormatException(json, "Bag")
        
    def __repr__(self):
        return "<Bag size={}>".format(len(self.values))

    def __eq__(self, other):
        return isinstance(other, Bag) and self.quantity == other.quantity and numeq(self.entries, other.entries) and self.values == other.values

    def __hash__(self):
       return hash((self.quantity, self.entries, tuple(self.values.items())))

Factory.register(Bag)
