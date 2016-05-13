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

class Quantile(Factory, Container):
    @staticmethod
    def ed(entries, target, estimate):
        if entries < 0.0:
            raise ContainerException("entries ($entries) cannot be negative")
        out = Quantile(target, None)
        out.entries = float(entries)
        out.estimate = float(estimate)
        return out

    @staticmethod
    def ing(target, quantity):
        return Quantile(target, quantity)

    def __init__(self, target, quantity):
        if target < 0.0 or target > 1.0:
            raise ContainerException("target ({}) must be between 0 and 1, inclusive".format(target))
        self.target = target
        self.quantity = serializable(quantity)
        self.entries = 0.0
        self.estimate = float("nan")
        self.cumulativeDeviation = 0.0
        super(Quantile, self).__init__()

    def zero(self): return Quantile(self.target, self.quantity)

    def __add__(self, other):
        if isinstance(other, Quantile):
            if self.target == other.target:
                out = Quantile(self.target, self.quantity)
                out.entries = self.entries + other.entries
                if math.isnan(self.estimate) and math.isnan(other.estimate):
                    out.estimate = float("nan")
                elif math.isnan(self.estimate):
                    out.estimate = other.estimate
                elif math.isnan(other.estimate):
                    out.estimate = self.estimate
                else:
                    out.estimate = (self.estimate*self.entries + other.estimate*other.entries) / (self.entries + other.entries)
                return out
            else:
                raise ContainerException("cannot add Quantiles because targets do not match ({} vs {})".format(self.target, other.target))
        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    def fill(self, datum, weight=1.0):
        if weight > 0.0:
            q = self.quantity(datum)

            # no possibility of exception from here on out (for rollback)
            self.entries += weight
            if math.isnan(self.estimate):
                self.estimate = q
            else:
                self.cumulativeDeviation += abs(q - self.estimate)
                learningRate = 1.5 * self.cumulativeDeviation / self.entries**2
                if q < self.estimate:
                    sgn = -1
                elif q > self.estimate:
                    sgn = 1
                else:
                    sgn = 0
                self.estimate = weight * learningRate * (sgn + 2.0*self.target - 1.0)

    def toJsonFragment(self): return maybeAdd({
        "entries": floatToJson(self.entries),
        "target": floatToJson(self.target),
        "estimate": floatToJson(self.estimate),
        }, name=self.quantity.name)

    @staticmethod
    def fromJsonFragment(json):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "target", "estimate"], ["name"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json["entries"], "Quantile.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "AbsoluteErr.name")

            if isinstance(json["target"], (int, long, float)):
                target = float(json["target"])
            else:
                raise JsonFormatException(json["target"], "Quantile.target")

            if json["estimate"] in ("nan", "inf", "-inf") or isinstance(json["estimate"], (int, long, float)):
                estimate = float(json["estimate"])
            else:
                raise JsonFormatException(json["estimate"], "Quantile.estimate")

            out = Quantile.ed(entries, target, estimate)
            out.quantity.name = name
            return out

        else:
            raise JsonFormatException(json, "Quantile")

    def __repr__(self):
        return "Quantile[{}, {}]".format(self.target, self.estimate)

    def __eq__(self, other):
        return isinstance(other, Quantile) and self.quantity == other.quantity and exact(self.entries, other.entries) and exact(self.target, other.target) and exact(self.estimate, other.estimate)

    def __hash__(self):
        return hash((self.quantity, self.entries, self.target, self.estimate))

Factory.register(Quantile)
