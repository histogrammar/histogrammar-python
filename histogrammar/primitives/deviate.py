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

class Deviate(Factory, Container):
    @staticmethod
    def ed(entries, mean, variance):
        if entries < 0.0:
            raise ContainerException("entries ($entries) cannot be negative")
        out = Deviate(None)
        out.entries = float(entries)
        out.mean = float(mean)
        out.varianceTimesEntries = float(variance)*float(entries)
        return out

    @staticmethod
    def ing(quantity):
        return Deviate(quantity)

    def __init__(self, quantity):
        self.quantity = serializable(quantity)
        self.entries = 0.0
        self.mean = 0.0
        self.varianceTimesEntries = 0.0
        super(Deviate, self).__init__()

    @property
    def variance(self):
        if self.entries == 0.0:
            return self.varianceTimesEntries
        else:
            return self.varianceTimesEntries/self.entries

    def zero(self): return Deviate(self.quantity)

    def __add__(self, other):
        if isinstance(other, Deviate):
            out = Deviate(self.quantity)
            out.entries = self.entries + other.entries
            out.mean = (self.entries*self.mean + other.entries*other.mean)/(self.entries + other.entries)
            out.varianceTimesEntries = self.varianceTimesEntries + other.varianceTimesEntries + self.entries*self.mean**2 + other.entries*other.mean**2 - 2.0*out.mean*(self.entries*self.mean + other.entries*other.mean) + out.mean*out.mean*out.entries
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
            self.varianceTimesEntries += weight * delta * (q - self.mean)

    def toJsonFragment(self): return {
        "entries": floatToJson(self.entries),
        "mean": floatToJson(self.mean),
        "variance": floatToJson(self.variance),
        }

    @staticmethod
    def fromJsonFragment(json):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "mean", "variance"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json["entries"], "Deviate.entries")

            if isinstance(json["mean"], (int, long, float)):
                mean = float(json["mean"])
            else:
                raise JsonFormatException(json["mean"], "Deviate.mean")

            if isinstance(json["variance"], (int, long, float)):
                variance = float(json["variance"])
            else:
                raise JsonFormatException(json["variance"], "Deviate.variance")

            return Deviate.ed(entries, mean, variance)

        else:
            raise JsonFormatException(json, self.name)
        
    def __repr__(self):
        return "Deviate[{}, {}]".format(self.mean, self.variance)

    def __eq__(self, other):
        return isinstance(other, Deviate) and self.quantity == other.quantity and exact(self.entries, other.entries) and exact(self.mean, other.mean) and exact(self.variance, other.variance)

    def __hash__(self):
        return hash((self.quantity, self.entries, self.mean, self.variance))

Factory.register(Deviate)
