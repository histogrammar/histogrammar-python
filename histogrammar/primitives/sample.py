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

class Sample(Factory, Container):
    @staticmethod
    def ed(entries, limit, values):
        if entries < 0.0:
            raise ContainerException("entries ({}) cannot be negative".format(entries))
        out = Sample(limit, None)
        del out.reservoir
        out.entries = entries
        out._limit = limit
        out._values = values
        return out

    @staticmethod
    def ing(limit, quantity):
        return Sample(limit, quantity)

    def __init__(self, limit, quantity):
        if limit <= 0.0:
            raise ContainerException("limit ({}) cannot be negative".format(limit))
        self.entries = 0.0
        self.quantity = serializable(quantity)
        self.reservoir = Reservoir(limit)
        super(Sample, self).__init__()

    @property
    def limit(self):
        if hasattr(self, "reservoir"):
            return self.reservoir.limit
        else:
            return self._limit

    @property
    def values(self):
        if hasattr(self, "reservoir"):
            return self.reservoir.values
        else:
            return self._values

    @property
    def size(self):
        if hasattr(self, "reservoir"):
            return self.reservoir.size
        else:
            return len(self._values)

    @property
    def isEmpty(self):
        if hasattr(self, "reservoir"):
            return self.reservoir.isEmpty
        else:
            return len(self._values) == 0

    def zero(self):
        return Sample(self.limit, self.quantity)

    def __add__(self, other):
        if isinstance(other, Sample):
            if self.limit != other.limit:
                raise ContainerException("cannot add Ssample because limit differs ({} vs {})".format(self.limit, other.limit))

            newreservoir = Reservoir(self.limit, *self.values)
            for y, weight in other.values:
                newreservoir.update(y, weight)

            out = Sample(self.limit, self.quantity)
            out.entries = self.entries + other.entries
            if hasattr(self, "reservoir"):
                out.reservoir = newreservoir
            else:
                del out.reservoir
                out._values = newreservoir.values
            return out

        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    def fill(self, datum, weight=1.0):
        if weight > 0.0:
            q = self.quantity(datum)

            self.reservoir.update(q, weight)

            # no possibility of exception from here on out (for rollback)
            self.entries += weight

    def toJsonFragment(self): return {
        "entries": floatToJson(self.entries),
        "limit": floatToJson(self.limit),
        "values": [{"w": w, "v": y} for y, w in sorted(self.values, key=lambda (y, w): y)],
        }

    @staticmethod
    def fromJsonFragment(json):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "limit", "values"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = json["entries"]
            else:
                raise JsonFormatException(json["entries"], "Sample.entries")

            if isinstance(json["limit"], (int, long, float)):
                limit = json["limit"]
            else:
                raise JsonFormatException(json["limit"], "Sample.limit")

            if isinstance(json["values"], list):
                values = []
                for i, wv in enumerate(json["values"]):
                    if isinstance(wv, dict) and hasKeys(wv.keys(), ["w", "v"]):
                        if isinstance(wv["w"], (int, long, float)):
                            w = float(wv["w"])
                        else:
                            raise JsonFormatException(wv["w"], "Sample.values {} w".format(i))

                        if isinstance(wv["v"], basestring):
                            v = wv["v"]
                        elif isinstance(wv["v"], (int, long, float)):
                            v = float(wv["v"])
                        elif isinstance(wv["v"], (list, tuple)):
                            for j, d in enumerate(wv["v"]):
                                if not isinstance(d, (int, long, float)):
                                    raise JsonFormatException(d, "Sample.values {} v {}".format(i, j))
                            v = tuple(map(float, wv["v"]))
                        else:
                            raise JsonFormatException(wv["v"], "Sample.values {} v".format(i))

                        values.append((v, w))

                    else:
                        raise JsonFormatException(wv, "Sample.values {}".format(i))

            else:
                raise JsonFormatException(json["values"], "Sample.values")

            return Sample.ed(entries, limit, values)

        else:
            raise JsonFormatException(json, self.name)

    def __repr__(self):
        return "Sample[{}, size={}]".format("empty" if self.isEmpty else repr(self.values[0][0]) + "...", self.size)

    def __eq__(self, other):
        return isinstance(other, Sample) and self.entries == other.entries and self.quantity == other.quantity and self.limit == other.limit and self.values == other.values

    def __hash__(self):
        return hash((self.entries, self.quantity, self.limit, self.value))

Factory.register(Sample)
