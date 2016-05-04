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
        out = Sum(None, None)
        out.entries = float(entries)
        out.sum = float(sum)
        return out

    @staticmethod
    def ing(quantity, selection=unweighted):
        return Sum(quantity, selection)

    def __init__(self, quantity, selection=unweighted):
        self.quantity = serializable(quantity)
        self.selection = serializable(selection)
        self.entries = 0.0
        self.sum = 0.0
        super(Sum, self).__init__()

    def zero(self): return Sum(self.quantity, self.selection)

    def __add__(self, other):
        if isinstance(other, Sum):
            out = Sum(self.quantity, self.selection)
            out.entries = self.entries + other.entries
            out.sum = self.sum + other.sum
            return out
        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    def fill(self, datum, weight=1.0):
        if self.quantity is None or self.selection is None:
            raise RuntimeException("attempting to fill a container that has no fill rule")

        w = weight * self.selection(datum)
        if w > 0.0:
            q = self.quantity(datum)
            self.entries += w
            self.sum += q * w

    def toJsonFragment(self): return {
        "entries": self.entries,
        "sum": self.sum,
        }

    @staticmethod
    def fromJsonFragment(json):
        if isinstance(json, dict) and set(json.keys()) == set(["entries", "sum"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = json["entries"]
            else:
                raise JsonFormatException(json["entries"], "Sum.entries")

            if isinstance(json["sum"], (int, long, float)):
                sum = json["sum"]
            else:
                raise JsonFormatException(json["sum"], "Sum.sum")

            return Sum.ed(entries, sum)

        else:
            raise JsonFormatException(json, self.name)
        
    def __repr__(self):
        return "Sum[{}]".format(self.sum)

    def __eq__(self, other):
        return isinstance(other, Sum) and self.quantity == other.quantity and self.selection == other.selection and exact(self.entries, other.entries) and exact(self.sum, other.sum)

    def __hash__(self):
        return hash((self.quantity, self.selection, self.entries, self.sum))

Factory.register(Sum)
