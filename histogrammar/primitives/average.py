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

class Average(Factory, Container):
    @staticmethod
    def ed(entries, mean):
        if entries < 0.0:
            raise ContainerException("entries ($entries) cannot be negative")
        out = Average(None, None)
        out.entries = float(entries)
        out.mean = float(mean)
        return out

    @staticmethod
    def ing(quantity, selection=unweighted):
        return Average(quantity, selection)

    def __init__(self, quantity, selection=unweighted):
        self.quantity = serializable(quantity)
        self.selection = serializable(selection)
        self.entries = 0.0
        self.mean = 0.0
        super(Average, self).__init__()

    def zero(self): return Average(self.quantity, self.selection)

    def __add__(self, other):
        if isinstance(other, Average):
            out = Average(self.quantity, self.selection)
            out.entries = self.entries + other.entries
            out.mean = (self.entries*self.mean + other.entries*other.mean)/(self.entries + other.entries)
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
            delta = q - self.mean
            shift = delta * w / self.entries
            self.mean += shift

    def toJsonFragment(self): return {
        "entries": floatToJson(self.entries),
        "mean": floatToJson(self.mean),
        }

    @staticmethod
    def fromJsonFragment(json):
        if isinstance(json, dict) and set(json.keys()) == set(["entries", "mean"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json["entries"], "Average.entries")

            if isinstance(json["mean"], (int, long, float)):
                mean = float(json["mean"])
            else:
                raise JsonFormatException(json["mean"], "Average.mean")

            return Average.ed(entries, mean)

        else:
            raise JsonFormatException(json, self.name)
        
    def __repr__(self):
        return "Average[{}]".format(self.mean)

    def __eq__(self, other):
        return isinstance(other, Average) and self.quantity == other.quantity and self.selection == other.selection and exact(self.entries, other.entries) and exact(self.mean, other.mean)

    def __hash__(self):
        return hash((self.quantity, self.selection, self.entries, self.mean))

Factory.register(Average)
