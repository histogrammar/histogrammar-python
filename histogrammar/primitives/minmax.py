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

class Minimize(Factory, Container):
    @staticmethod
    def ed(entries, min):
        if entries < 0.0:
            raise ContainerException("entries ($entries) cannot be negative")
        out = Minimize(None, None)
        out.entries = float(entries)
        out.min = float(min)
        return out

    @staticmethod
    def ing(quantity, selection=unweighted):
        return Minimize(quantity, selection)

    def __init__(self, quantity, selection=unweighted):
        self.quantity = serializable(quantity)
        self.selection = serializable(selection)
        self.entries = 0.0
        self.min = float("nan")
        super(Minimize, self).__init__()

    def zero(self): return Minimize(self.quantity, self.selection)

    def __add__(self, other):
        if isinstance(other, Minimize):
            out = Minimize(self.quantity, self.selection)
            out.entries = self.entries + other.entries
            if math.isnan(self.min):
                out.min = other.min
            elif math.isnan(other.min):
                out.min = self.min
            elif self.min < other.min:
                out.min = self.min
            else:
                out.min = other.min
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
            if math.isnan(self.min) or q < self.min:
                self.min = q

    def toJsonFragment(self): return {
        "entries": self.entries,
        "min": self.min,
        }

    @staticmethod
    def fromJsonFragment(json):
        if isinstance(json, dict) and set(json.keys()) == set(["entries", "min"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = json["entries"]
            else:
                raise JsonFormatException(json["entries"], "Minimize.entries")

            if isinstance(json["min"], (int, long, float)):
                min = json["min"]
            else:
                raise JsonFormatException(json["min"], "Minimize.min")

            return Minimize.ed(entries, min)

        else:
            raise JsonFormatException(json, self.name)
        
    def __repr__(self):
        return "Minimize[{}]".format(self.min)

    def __eq__(self, other):
        return isinstance(other, Minimize) and self.quantity == other.quantity and self.selection == other.selection and exact(self.entries, other.entries) and exact(self.min, other.min)

    def __hash__(self):
        return hash((self.quantity, self.selection, self.entries, self.min))

Factory.register(Minimize)

class Maximize(Factory, Container):
    @staticmethod
    def ed(entries, max):
        if entries < 0.0:
            raise ContainerException("entries ($entries) cannot be negative")
        out = Maximize(None, None)
        out.entries = float(entries)
        out.max = float(max)
        return out

    @staticmethod
    def ing(quantity, selection=unweighted):
        return Maximize(quantity, selection)

    def __init__(self, quantity, selection=unweighted):
        self.quantity = serializable(quantity)
        self.selection = serializable(selection)
        self.entries = 0.0
        self.max = float("nan")
        super(Maximize, self).__init__()

    def zero(self): return Maximize(self.quantity, self.selection)

    def __add__(self, other):
        if isinstance(other, Maximize):
            out = Maximize(self.quantity, self.selection)
            out.entries = self.entries + other.entries
            if math.isnan(self.max):
                out.max = other.max
            elif math.isnan(other.max):
                out.max = self.max
            elif self.max > other.max:
                out.max = self.max
            else:
                out.max = other.max
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
            if math.isnan(self.max) or q > self.max:
                self.max = q

    def toJsonFragment(self): return {
        "entries": self.entries,
        "max": self.max,
        }

    @staticmethod
    def fromJsonFragment(json):
        if isinstance(json, dict) and set(json.keys()) == set(["entries", "max"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = json["entries"]
            else:
                raise JsonFormatException(json["entries"], "Maximize.entries")

            if isinstance(json["max"], (int, long, float)):
                max = json["max"]
            else:
                raise JsonFormatException(json["max"], "Maximize.max")

            return Maximize.ed(entries, max)

        else:
            raise JsonFormatException(json, self.name)
        
    def __repr__(self):
        return "Maximize[{}]".format(self.max)

    def __eq__(self, other):
        return isinstance(other, Maximize) and self.quantity == other.quantity and self.selection == other.selection and exact(self.entries, other.entries) and exact(self.max, other.max)

    def __hash__(self):
        return hash((self.quantity, self.selection, self.entries, self.max))

Factory.register(Maximize)
