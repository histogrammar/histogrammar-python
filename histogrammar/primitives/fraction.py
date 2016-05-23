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

class Fraction(Factory, Container):
    @staticmethod
    def ed(entries, numerator, denominator):
        if entries < 0.0:
            raise ContainerException("entries ({}) cannot be negative".format(entries))

        out = Fraction(None, None)
        out.entries = float(entries)
        out.numerator = numerator
        out.denominator = denominator
        return out

    @staticmethod
    def ing(quantity, value):
        return Fraction(quantity, value)

    def __init__(self, quantity, value):
        self.entries = 0.0
        self.quantity = serializable(quantity)
        if value is not None:
            self.numerator = value.zero()
            self.denominator = value.zero()
        super(Fraction, self).__init__()
        
    def zero(self):
        out = Fraction(self.quantity, None)
        out.numerator = self.numerator.zero()
        out.denominator = self.denominator.zero()
        return out

    def __add__(self, other):
        if isinstance(other, Fraction):
            out = Fraction(self.quantity, None)
            out.numerator = self.numerator + other.numerator
            out.denominator = self.denominator + other.denominator
            return out
        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    def fill(self, datum, weight=1.0):
        w = weight * self.quantity(datum)

        if weight > 0.0:
            self.denominator.fill(datum, weight)
        if w > 0.0:
            self.numerator.fill(datum, w)

        # no possibility of exception from here on out (for rollback)
        self.entries += weight

    @property
    def children(self):
        return [self.numerator, self.denominator]

    def toJsonFragment(self, suppressName=False):
        if getattr(self.numerator, "quantity", None) is not None:
            binsName = self.numerator.quantity.name
        elif getattr(self.numerator, "quantityName", None) is not None:
            binsName = self.numerator.quantityName
        else:
            binsName = None

        return maybeAdd({
            "entries": floatToJson(self.entries),
            "type": self.numerator.name,
            "numerator": self.numerator.toJsonFragment(True),
            "denominator": self.denominator.toJsonFragment(True),
            }, **{"name": None if suppressName else self.quantity.name,
                  "sub:name": binsName})

    @staticmethod
    def fromJsonFragment(json, nameFromParent=None):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "type", "numerator", "denominator"], ["name"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "Fraction.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Fraction.name")

            if isinstance(json["type"], basestring):
                factory = Factory.registered[json["type"]]
            else:
                raise JsonFormatException(json, "Fraction.type")

            numerator = factory.fromJsonFragment(json["numerator"])
            denominator = factory.fromJsonFragment(json["denominator"])

            out = Fraction.ed(entries, numerator, denominator)
            out.quantity.name = nameFromParent if name is None else name
            return out

        else:
            raise JsonFormatException(json, "Fraction")

    def __repr__(self):
        return "Fraction[{}, {}]".format(self.numerator, self.denominator)

    def __eq__(self, other):
        return isinstance(other, Fraction) and exact(self.entries, other.entries) and self.quantity == other.quantity and self.numerator == other.numerator and self.denominator == other.denominator

    def __hash__(self):
        return hash((self.entries, self.quantity, self.numerator, self.denominator))

Factory.register(Fraction)
