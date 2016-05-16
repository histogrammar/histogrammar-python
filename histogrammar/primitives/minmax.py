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
        out = Minimize(None)
        out.entries = float(entries)
        out.min = float(min)
        return out

    @staticmethod
    def ing(quantity):
        return Minimize(quantity)

    def __init__(self, quantity):
        self.quantity = serializable(quantity)
        self.entries = 0.0
        self.min = float("nan")
        super(Minimize, self).__init__()

    def zero(self): return Minimize(self.quantity)

    def __add__(self, other):
        if isinstance(other, Minimize):
            out = Minimize(self.quantity)
            out.entries = self.entries + other.entries
            out.min = minplus(self.min, other.min)
            return out
        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    @property
    def children(self):
        return []

    def fill(self, datum, weight=1.0):
        if weight > 0.0:
            q = self.quantity(datum)

            # no possibility of exception from here on out (for rollback)
            self.entries += weight
            if math.isnan(self.min) or q < self.min:
                self.min = q

    def toJsonFragment(self): return maybeAdd({
        "entries": floatToJson(self.entries),
        "min": floatToJson(self.min),
        }, name=self.quantity.name)

    @staticmethod
    def fromJsonFragment(json):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "min"], ["name"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json["entries"], "Minimize.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Minimize.name")

            if json["min"] in ("nan", "inf", "-inf") or isinstance(json["min"], (int, long, float)):
                min = float(json["min"])
            else:
                raise JsonFormatException(json["min"], "Minimize.min")

            out = Minimize.ed(entries, min)
            out.quantity.name = name
            return out

        else:
            raise JsonFormatException(json, "Minimize")
        
    def __repr__(self):
        return "Minimize[{}]".format(self.min)

    def __eq__(self, other):
        return isinstance(other, Minimize) and self.quantity == other.quantity and exact(self.entries, other.entries) and exact(self.min, other.min)

    def __hash__(self):
        return hash((self.quantity, self.entries, self.min))

Factory.register(Minimize)

class Maximize(Factory, Container):
    @staticmethod
    def ed(entries, max):
        if entries < 0.0:
            raise ContainerException("entries ($entries) cannot be negative")
        out = Maximize(None)
        out.entries = float(entries)
        out.max = float(max)
        return out

    @staticmethod
    def ing(quantity):
        return Maximize(quantity)

    def __init__(self, quantity):
        self.quantity = serializable(quantity)
        self.entries = 0.0
        self.max = float("nan")
        super(Maximize, self).__init__()

    def zero(self): return Maximize(self.quantity)

    def __add__(self, other):
        if isinstance(other, Maximize):
            out = Maximize(self.quantity)
            out.entries = self.entries + other.entries
            out.max = maxplus(self.max, other.max)
            return out
        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    def fill(self, datum, weight=1.0):
        if weight > 0.0:
            q = self.quantity(datum)

            # no possibility of exception from here on out (for rollback)
            self.entries += weight
            if math.isnan(self.max) or q > self.max:
                self.max = q

    @property
    def children(self):
        return []

    def toJsonFragment(self): return maybeAdd({
        "entries": floatToJson(self.entries),
        "max": floatToJson(self.max),
        }, name=self.quantity.name)

    @staticmethod
    def fromJsonFragment(json):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "max"], ["name"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json["entries"], "Maximize.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Maximize.name")

            if json["max"] in ("nan", "inf", "-inf") or isinstance(json["max"], (int, long, float)):
                max = float(json["max"])
            else:
                raise JsonFormatException(json["max"], "Maximize.max")

            out = Maximize.ed(entries, max)
            out.quantity.name = name
            return out

        else:
            raise JsonFormatException(json, "Maximize")
        
    def __repr__(self):
        return "Maximize[{}]".format(self.max)

    def __eq__(self, other):
        return isinstance(other, Maximize) and self.quantity == other.quantity and exact(self.entries, other.entries) and exact(self.max, other.max)

    def __hash__(self):
        return hash((self.quantity, self.entries, self.max))

Factory.register(Maximize)
