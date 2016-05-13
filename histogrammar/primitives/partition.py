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

class Partition(Factory, Container):
    @staticmethod
    def ed(entries, *cuts):
        if entries < 0.0:
            raise ContainerException("entries ({}) cannot be negative".format(entries))

        out = Partition(None, None, *cuts)
        out.entries = float(entries)
        return out

    @staticmethod
    def ing(expression, value, *cuts):
        return Partition(expression, value, *cuts)

    def __init__(self, expression, value, *cuts):
        self.entries = 0.0
        self.expression = serializable(expression)
        if value is None:
            self.cuts = cuts
        else:
            self.cuts = tuple((float(x), value.zero()) for x in (float("-inf"),) + cuts)
        super(Partition, self).__init__()

    @property
    def thresholds(self): return [k for k, v in self.cuts]
    @property
    def values(self): return [v for k, v in self.cuts]

    def zero(self):
        return Partition(self.expression, None, *[(x, x.zero()) for x in cuts])

    def __add__(self, other):
        if isinstance(other, Partition):
            if self.thresholds != other.thresholds:
                raise ContainerException("cannot add Partition because cut thresholds differ")

            out = Partition(self.expression, None, *[(k1, v1 + v2) for ((k1, v1), (k2, v2)) in zip(self.cuts, other.cuts)])
            out.entries = self.entries + other.entries
            return out

        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    def fill(self, datum, weight=1.0):
        if weight > 0.0:
            value = self.expression(datum)
            for (low, sub), (high, _) in zip(self.cuts, self.cuts[1:] + (float("nan"), None)):
                if value >= low and not value >= high:
                    sub.fill(datum, weight)
                    break

            # no possibility of exception from here on out (for rollback)
            self.entries += weight

    def toJsonFragment(self): return maybeAdd({
        "entries": floatToJson(self.entries),
        "type": self.cuts[0][1].name,
        "data": [{"atleast": floatToJson(atleast), "data": sub.toJsonFragment()} for atleast, sub in self.cuts],
        }, name=self.expression.name)

    @staticmethod
    def fromJsonFragment(json):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "type", "data"], ["name"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "Partition.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Partition.name")

            if isinstance(json["type"], basestring):
                factory = Factory.registered[json["type"]]
            else:
                raise JsonFormatException(json, "Partition.type")

            if isinstance(json["data"], list):
                cuts = []
                for i, elementPair in enumerate(json["data"]):
                    if isinstance(elementPair, dict) and hasKeys(elementPair.keys(), ["atleast", "data"]):
                        if elementPair["atleast"] not in ("nan", "inf", "-inf") and not isinstance(elementPair["atleast"], (int, long, float)):
                            raise JsonFormatException(json, "Partition.data {} atleast".format(i))

                        cuts.append((float(elementPair["atleast"]), factory.fromJsonFragment(elementPair["data"])))

                    else:
                        raise JsonFormatException(json, "Partition.data {}".format(i))

                out = Partition.ed(entries, *cuts)
                out.expression.name = name
                return out

            else:
                raise JsonFormatException(json, "Partition.data")

        else:
            raise JsonFormatException(json, "Partition")

    def __repr__(self):
        return "Partition[{}, thresholds=[{}]]".format(self.cuts[0], ", ".join(map(str, self.thresholds)))

    def __eq__(self, other):
        return isinstance(other, Partition) and exact(self.entries, other.entries) and self.expression == other.expression and self.cuts == other.cuts

    def __hash__(self):
        return hash((self.entries, self.expression, self.cuts))

Factory.register(Partition)
