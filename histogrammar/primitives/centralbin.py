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

import bisect
import math

from histogrammar.defs import *
from histogrammar.util import *
from histogrammar.primitives.count import *

class CentrallyBin(Factory, Container, CentralBinsDistribution, CentrallyBinMethods):
    @staticmethod
    def ed(entries, bins, min, max, nanflow):
        if entries < 0.0:
            raise ContainerException("entries ({}) cannot be negative".format(entries))
        out = CentrallyBin(bins, None, None, None, nanflow)
        out.entries = entries
        out.bins = bins
        out.min = min
        out.max = max
        return out

    @staticmethod
    def ing(bins, quantity, selection=unweighted, value=Count(), nanflow=Count()):
        return CentrallyBin(bins, quantity, selection, value, nanflow)

    def __init__(self, bins, quantity, selection=unweighted, value=Count(), nanflow=Count()):
        if len(bins) < 2:
            raise ContainerException("number of bins ({}) must be at least two".format(len(bins)))

        self.entries = 0.0
        if value is None:
            self.bins = None
        else:
            self.bins = [(x, value.zero()) for x in sorted(bins)]
        self.min = float("nan")
        self.max = float("nan")

        self.quantity = quantity
        self.selection = selection
        self.value = value
        self.nanflow = nanflow

    def zero(self):
        return CentrallyBin(map(lambda (x, v): x, self.bins), self.quantity, self.selection, self.value, self.nanflow.zero())

    def __add__(self, other):
        if self.centers != other.centers:
            raise ContainerException("cannot add CentrallyBin because centers are different:\n    {}\nvs\n    {}".format(self.centers, other.centers))

        newbins = [(c1, v1 + v2) for (c1, v1), (_, v2) in zip(self.bins, other.bins)]

        out = CentrallyBin(map(lambda (x, v): x, self.bins), self.quantity, self.selection, self.value, self.nanflow + other.nanflow)
        out.entries = self.entries + other.entries
        out.bins = newbins
        out.min = minplus(self.min, other.min)
        out.max = maxplus(self.max, other.max)
        return out

    def fill(self, datum, weight=1.0):
        if self.quantity is None or self.selection is None:
            raise RuntimeException("attempting to fill a container that has no fill rule")

        w = weight * self.selection(datum)

        if w > 0.0:
            q = self.quantity(datum)

            self.entries += w
            if self.nan(q):
                self.nanflow.fill(datum, w)
            else:
                self.bins[self.index(q)][1].fill(datum, weight)

            if math.isnan(self.min) or q < self.min:
                self.min = q
            if math.isnan(self.max) or q > self.max:
                self.max = q

    def toJsonFragment(self): return {
        "entries": floatToJson(self.entries),
        "bins:type": self.bins[0][1].name,
        "bins": [{"center": floatToJson(c), "value": v.toJsonFragment()} for c, v in self.bins],
        "min": floatToJson(self.min),
        "max": floatToJson(self.max),
        "nanflow:type": self.nanflow.name,
        "nanflow": self.nanflow.toJsonFragment(),
        }

    @staticmethod
    def fromJsonFragment(json):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "bins:type", "bins", "min", "max", "nanflow:type", "nanflow"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "CentrallyBin.entries")

            if isinstance(json["bins:type"], basestring):
                factory = Factory.registered[json["bins:type"]]
            else:
                raise JsonFormatException(json, "CentrallyBin.bins:type")
            if isinstance(json["bins"], list):
                bins = []
                for i, binpair in enumerate(json["bins"]):
                    if isinstance(binpair, dict) and hasKeys(binpair.keys(), ["center", "value"]):
                        if isinstance(binpair["center"], (int, long, float)):
                            center = float(binpair["center"])
                        else:
                            JsonFormatException(binpair["center"], "CentrallyBin.bins {} center".format(i))
                        
                        bins.append((center, factory.fromJsonFragment(binpair["value"])))

                    else:
                        raise JsonFormatException(binpair, "CentrallyBin.bins {}".format(i))

            if json["min"] in ("nan", "inf", "-inf") or isinstance(json["min"], (int, long, float)):
                min = float(json["min"])
            else:
                raise JsonFormatException(json, "CentrallyBin.min")

            if json["max"] in ("nan", "inf", "-inf") or isinstance(json["max"], (int, long, float)):
                max = float(json["max"])
            else:
                raise JsonFormatException(json, "CentrallyBin.max")

            if isinstance(json["nanflow:type"], basestring):
                nanflowFactory = Factory.registered[json["nanflow:type"]]
            else:
                raise JsonFormatException(json, "CentrallyBin.nanflow:type")
            nanflow = nanflowFactory.fromJsonFragment(json["nanflow"])

            return CentrallyBin.ed(entries, bins, min, max, nanflow)

        else:
            raise JsonFormatException(json, "CentrallyBin")

    def __repr__(self):
        return "CentrallyBin[bins=[{}..., size={}], nanflow={}]".format(self.bins[0][1], len(self.bins), self.nanflow)

    def __eq__(self):
        return isinstance(other, CentrallyBin) and self.quantity == other.quantity and self.selection == other.selection and exact(self.entries, other.entries) and self.bins == other.bins and exact(self.min, other.min) and exact(self.max, other.max) and self.nanflow == other.nanflow

    def __hash__(self):
        return hash((self.quantity, self.selection, self.entries, self.bins, self.min, self.max, self.nanflow))

Factory.register(CentrallyBin)
