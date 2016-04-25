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
    def ed(entries, limit, values):
        if entries < 0.0:
            raise ContainerException("entries ($entries) cannot be negative")
        out = Bag(None, None, limit)
        out.entries = float(entries)
        if out.limit is None:
            out.limit = None
        else:
            out.limit = float(limit)
        out.values = values
        return out

    @staticmethod
    def ing(quantity, selection=unweighted, limit=None):
        return Bag(quantity, selection, limit)

    def __init__(self, quantity, selection=unweighted, limit=None):
        self.quantity = serializable(quantity)
        self.selection = serializable(selection)
        self.entries = 0.0
        self.values = {}
        self.limit = limit
        super(Bag, self).__init__()

    @property
    def zero(self): return Bag(self.quantity, self.selection, self.limit)

    def __add__(self, other):
        if isinstance(other, Bag):
            out = Bag(self.quantity, self.selection, self.limit)

            out.entries = self.entries + other.entries

            if self.limit is not None and self.limit < out.entries:
                out.values = None
            else:
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
        if self.quantity is None or self.selection is None:
            raise RuntimeException("attempting to fill a container that has no fill rule")

        w = weight * self.selection(datum)
        if w > 0.0:
            q = self.quantity(datum)
            if isinstance(q, (int, long, float)):
                q = (q,)
            elif isinstance(q, list):
                q = tuple(q)

            self.entries += w

            if self.limit is not None and self.limit < self.entries:
                self.values = None
            else:
                if q in self.values:
                    self.values[q] += w
                else:
                    self.values[q] = w

    def toJsonFragment(self): return {
        "entries": self.entries,
        "limit": self.limit,
        "values": None if self.values is None else [{"n": n, "v": v} for v, n in sorted(self.values.items())],
        }

    @staticmethod
    def fromJsonFragment(json):
        if isinstance(json, dict) and set(json.keys()) == set(["entries", "values", "limit"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = json["entries"]
            else:
                raise JsonFormatException(json["entries"], "Bag.entries")

            if json["values"] is None:
                values = None

            elif json["values"] is None or isinstance(json["values"], list):
                values = {}
                for i, nv in enumerate(json["values"]):
                    if isinstance(nv, dict) and set(nv.keys()) == set(["n", "v"]):
                        if isinstance(nv["n"], (int, long, float)):
                            n = float(nv["n"])
                        else:
                            raise JsonFormatException(n, "Bag.values {} n".format(i))

                        if isinstance(nv["v"], (list, tuple)):
                            for j, d in enumerate(nv["v"]):
                                if not isinstance(d, (int, long, float)):
                                    raise JsonFormatException(d, "Bag.values {} v {}".format(i, j))
                            v = tuple(nv["v"])
                        else:
                            raise JsonFormatException(nv["v"], "Bag.values {} v".format(i))

                        values[v] = n

                    else:
                        raise JsonFormatException(nv, "Bag.values {}".format(i))

            elif json["values"] is None:
                values = None

            else:
                raise JsonFormatException(json["values"], "Bag.values")

            if json["limit"] is None or isinstance(json["limit"], (int, long, float)):
                limit = json["limit"]
            else:
                raise JsonFormatException(json["limit"], "Bag.limit")

            return Bag.ed(entries, limit, values)

        else:
            raise JsonFormatException(json, self.name)
        
    def __repr__(self):
        return "Bag[{}]".format("saturated" if self.limit is not None and self.limit < self.entries else "size=" + len(self.values))

    def __eq__(self, other):
        return isinstance(other, Bag) and self.quantity == other.quantity and self.selection == other.selection and exact(self.entries, other.entries) and self.limit == other.limit and self.values == other.values

    def __hash__(self):
        return hash((self.quantity, self.selection, self.entries, self.limit, self.values))

Factory.register(Bag)
