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
    def ing(numeratorSelection, value):
        return Fraction(numeratorSelection, value)

    def __init__(self, numeratorSelection, value):
        self.entries = 0.0
        self.numeratorSelection = numeratorSelection
        if value is not None:
            self.numerator = value.zero()
            self.denominator = value.zero()
        
    def zero(self):
        out = Fraction(self.numeratorSelection, None)
        out.numerator = self.numerator.zero()
        out.denominator = self.denominator.zero()
        return out

    def __add__(self, other):
        if isinstance(other, Fraction):
            out = Fraction(self.numeratorSelection, None)
            out.numerator = self.numerator + other.numerator
            out.denominator = self.denominator + other.denominator
            return out
        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    def fill(self, datum, weight=1.0):
        if self.numeratorSelection is None:
            raise RuntimeException("attempting to fill a container that has no fill rule")

        w = weight * self.numeratorSelection(datum)

        self.entries += weight
        if weight > 0.0:
            self.denominator.fill(datum, weight)
        if w > 0.0:
            self.numerator.fill(datum, w)

    def toJsonFragment(self): return {
        "entries": self.entries,
        "type": self.numerator.name,
        "numerator": self.numerator.toJsonFragment(),
        "denominator": self.denominator.toJsonFragment(),
        }

    @staticmethod
    def fromJsonFragment(json):
        if isinstance(json, dict) and set(json.keys()) == set(["entries", "type", "numerator", "denominator"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = json["entries"]
            else:
                raise JsonFormatException(json, "Fraction.entries")

            if isinstance(json["type"], basestring):
                factory = Factory.registered[json["type"]]
            else:
                raise JsonFormatException(json, "Fraction.type")

            numerator = factory.fromJsonFragment(json["numerator"])
            denominator = factory.fromJsonFragment(json["denominator"])

            return Fraction.ed(entries, numerator, denominator)

        else:
            raise JsonFormatException(json, "Fraction")

    def __repr__(self):
        return "Fraction[{}, {}]".format(self.numerator, self.denominator)

    def __eq__(self, other):
        return isinstance(other, Fraction) and exact(self.entries, other.entries) and self.numeratorSelection == other.numeratorSelection and self.numerator == other.numerator and self.denominator == other.denominator

    def __hash__(self):
        return hash((self.entries, self.numeratorSelection, self.numerator, self.denominator))

Factory.register(Fraction)
