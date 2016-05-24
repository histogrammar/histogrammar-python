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

class Sum(Factory, Container):
    @staticmethod
    def ed(entries, sum):
        if entries < 0.0:
            raise ContainerException("entries ($entries) cannot be negative")
        out = Sum(None)
        out.entries = float(entries)
        out.sum = float(sum)
        return out

    @staticmethod
    def ing(quantity):
        return Sum(quantity)

    def __init__(self, quantity):
        self.quantity = serializable(quantity)
        self.entries = 0.0
        self.sum = 0.0
        super(Sum, self).__init__()

    def zero(self): return Sum(self.quantity)

    def __add__(self, other):
        if isinstance(other, Sum):
            out = Sum(self.quantity)
            out.entries = self.entries + other.entries
            out.sum = self.sum + other.sum
            return out
        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    def fill(self, datum, weight=1.0):
        if weight > 0.0:
            q = self.quantity(datum)

            # no possibility of exception from here on out (for rollback)
            self.entries += weight
            self.sum += q * weight

    @property
    def children(self):
        return []

    def toJsonFragment(self, suppressName): return maybeAdd({
        "entries": floatToJson(self.entries),
        "sum": floatToJson(self.sum),
        }, name=(None if suppressName else self.quantity.name))

    @staticmethod
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "sum"], ["name"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json["entries"], "Sum.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Sum.name")

            if isinstance(json["sum"], (int, long, float)):
                sum = float(json["sum"])
            else:
                raise JsonFormatException(json["sum"], "Sum.sum")

            out = Sum.ed(entries, sum)
            out.quantity.name = nameFromParent if name is None else name
            return out

        else:
            raise JsonFormatException(json, "Sum")
        
    def __repr__(self):
        return "Sum[{}]".format(self.sum)

    def __eq__(self, other):
        return isinstance(other, Sum) and self.quantity == other.quantity and exact(self.entries, other.entries) and exact(self.sum, other.sum)

    def __hash__(self):
        return hash((self.quantity, self.entries, self.sum))

Factory.register(Sum)
