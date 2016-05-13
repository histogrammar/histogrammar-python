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
        out = Average(None)
        out.entries = float(entries)
        out.mean = float(mean)
        return out

    @staticmethod
    def ing(quantity):
        return Average(quantity)

    def __init__(self, quantity):
        self.quantity = serializable(quantity)
        self.entries = 0.0
        self.mean = 0.0
        super(Average, self).__init__()

    def zero(self): return Average(self.quantity)

    def __add__(self, other):
        if isinstance(other, Average):
            out = Average(self.quantity)
            out.entries = self.entries + other.entries
            out.mean = (self.entries*self.mean + other.entries*other.mean)/(self.entries + other.entries)
            return out
        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    def fill(self, datum, weight=1.0):
        if weight > 0.0:
            q = self.quantity(datum)

            # no possibility of exception from here on out (for rollback)
            self.entries += weight
            delta = q - self.mean
            shift = delta * weight / self.entries
            self.mean += shift

    def toJsonFragment(self): return maybeAdd({
        "entries": floatToJson(self.entries),
        "mean": floatToJson(self.mean),
        }, name=self.quantity.name)

    @staticmethod
    def fromJsonFragment(json):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "mean"], ["name"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json["entries"], "Average.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Average.name")

            if isinstance(json["mean"], (int, long, float)):
                mean = float(json["mean"])
            else:
                raise JsonFormatException(json["mean"], "Average.mean")

            out = Average.ed(entries, mean)
            out.quantity.name = name
            return out

        else:
            raise JsonFormatException(json, "Average")
        
    def __repr__(self):
        return "Average[{}]".format(self.mean)

    def __eq__(self, other):
        return isinstance(other, Average) and self.quantity == other.quantity and exact(self.entries, other.entries) and exact(self.mean, other.mean)

    def __hash__(self):
        return hash((self.quantity, self.entries, self.mean))

Factory.register(Average)
