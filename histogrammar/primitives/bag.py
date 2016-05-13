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

class Bag(Factory, Container):
    @staticmethod
    def ed(entries, values):
        if entries < 0.0:
            raise ContainerException("entries ({}) cannot be negative".format(entries))
        out = Bag(None)
        out.entries = float(entries)
        out.values = values
        return out

    @staticmethod
    def ing(quantity):
        return Bag(quantity)

    def __init__(self, quantity):
        self.quantity = serializable(quantity)
        self.entries = 0.0
        self.values = {}
        super(Bag, self).__init__()

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

            return out

        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    def fill(self, datum, weight=1.0):
        if weight > 0.0:
            q = self.quantity(datum)

            if isinstance(q, list):
                q = tuple(map(float, q))
            elif not isinstance(q, (int, long, float, basestring, tuple)):
                raise ContainerException("fill rule for Bag must return a number, vector of numbers, or a string, not {}".format(q))

            # no possibility of exception from here on out (for rollback)
            self.entries += weight
            if q in self.values:
                self.values[q] += weight
            else:
                self.values[q] = weight

    def toJsonFragment(self): return maybeAdd({
        "entries": floatToJson(self.entries),
        "values": [{"n": n, "v": v} for v, n in sorted(self.values.items())],
        }, name=self.quantity.name)

    @staticmethod
    def fromJsonFragment(json):
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
                    if isinstance(nv, dict) and hasKeys(nv.keys(), ["n", "v"]):
                        if isinstance(nv["n"], (int, long, float)):
                            n = float(nv["n"])
                        else:
                            raise JsonFormatException(nv["n"], "Bag.values {} n".format(i))

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
            out.quantity.name = name
            return out

        else:
            raise JsonFormatException(json, "Bag")
        
    def __repr__(self):
        return "Bag[{}]".format("size=0" if len(self.values) == 0 else repr(self.values[0]) + "..., size=" + str(len(self.values)))

    def __eq__(self, other):
        return isinstance(other, Bag) and self.quantity == other.quantity and exact(self.entries, other.entries) and self.values == other.values

    def __hash__(self):
        return hash((self.quantity, self.entries, self.values))

Factory.register(Bag)
