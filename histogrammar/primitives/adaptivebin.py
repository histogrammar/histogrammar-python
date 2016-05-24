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

class AdaptivelyBin(Factory, Container, CentralBinsDistribution, CentrallyBinMethods):
    @staticmethod
    def ed(entries, num, tailDetail, contentType, bins, min, max, nanflow):
        if entries < 0.0:
            raise ContainerException("entries ({}) cannot be negative".format(entries))

        out = AdaptivelyBin(None, num, tailDetail, None, nanflow)
        out.clustering.entries = entries
        out.clustering.values = bins
        out.clustering.min = min
        out.clustering.max = max
        out.contentType = contentType
        return out

    @staticmethod
    def ing(quantity, num=100, tailDetail=0.2, value=Count(), nanflow=Count()):
        return AdaptivelyBin(quantity, num, tailDetail, value, nanflow)

    def __init__(self, quantity, num=100, tailDetail=0.2, value=Count(), nanflow=Count()):
        if num < 2:
            raise ContainerException("number of bins ({}) must be at least two".format(num))
        if tailDetail < 0.0 or tailDetail > 1.0:
            raise ContainerException("tailDetail parameter ({}) must be between 0.0 and 1.0 inclusive".format(tailDetail))

        self.quantity = serializable(quantity)
        self.clustering = Clustering1D(num, tailDetail, value, [], float("nan"), float("nan"), 0.0)
        self.nanflow = nanflow.copy()
        super(AdaptivelyBin, self).__init__()

    @property
    def num(self): return self.clustering.num
    @property
    def tailDetail(self): return self.clustering.tailDetail
    @property
    def entries(self): return self.clustering.entries
    @entries.setter
    def entries(self, value):
        self.clustering.entries = value
    @property
    def bins(self): return self.clustering.values
    @property
    def min(self): return self.clustering.min
    @min.setter
    def min(self, value):
        self.clustering.min = min
    @property
    def max(self): return self.clustering.max
    @max.setter
    def max(self, value):
        self.clustering.max = max

    def zero(self):
        return AdaptivelyBin(self.quantity, self.num, self.tailDetail, self.clustering.value, self.nanflow.zero())

    def __add__(self, other):
        if self.num != other.num:
            raise ContainerException("cannot add AdaptivelyBin because number of bins is different ({} vs {})".format(self.num, other.num))
        if self.tailDetail != other.tailDetail:
            raise ContainerException("cannot add AdaptivelyBin because tailDetail parameter is different ({} vs {})".format(self.num, other.num))

        out = AdaptivelyBin(self.quantity, self.num, self.tailDetail, self.clustering.value, self.nanflow + other.nanflow)
        out.clustering = self.clustering.merge(other.clustering)
        return out
        
    def fill(self, datum, weight=1.0):
        if weight > 0.0:
            q = self.quantity(datum)
            self.clustering.update(q, datum, weight)

    @property
    def children(self):
        return [self.value, self.nanflow] + [v for c, v in self.bins]

    def toJsonFragment(self, suppressName):
        if isinstance(self.value, Container):
            if getattr(self.value, "quantity", None) is not None:
                binsName = self.value.quantity.name
            elif getattr(self.value, "quantityName", None) is not None:
                binsName = self.value.quantityName
            else:
                binsName = None
        elif len(self.bins) > 0:
            if getattr(self.bins[0][1], "quantity", None) is not None:
                binsName = self.bins[0][1].quantity.name
            elif getattr(self.bins[0][1], "quantityName", None) is not None:
                binsName = self.bins[0][1].quantityName
            else:
                binsName = None
        else:
            binsName = None

        return maybeAdd({
            "entries": floatToJson(self.entries),
            "num": self.num,
            "bins:type": self.clustering.value.name if self.clustering.value is not None else self.contentType,
            "bins": [{"center": c, "value": v.toJsonFragment(True)} for c, v in self.bins],
            "min": floatToJson(self.min),
            "max": floatToJson(self.max),
            "nanflow:type": self.nanflow.name,
            "nanflow": self.nanflow.toJsonFragment(False),
            "tailDetail": self.tailDetail,
            }, **{"name": None if suppressName else self.quantity.name,
                  "bins:name": binsName})

    @staticmethod
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "num", "bins:type", "bins", "min", "max", "nanflow:type", "nanflow", "tailDetail"], ["name", "bins:name"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "AdaptivelyBin.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "AdaptivelyBin.name")

            if isinstance(json["num"], (int, long)):
                num = int(json["num"])
            else:
                raise JsonFormatException(json, "AdaptivelyBin.num")

            if isinstance(json["bins:type"], basestring):
                contentType = json["bins:type"]
                factory = Factory.registered[contentType]
            else:
                raise JsonFormatException(json, "AdaptivelyBin.bins:type")
            if isinstance(json.get("bins:name", None), basestring):
                binsName = json["bins:name"]
            elif json.get("bins:name", None) is None:
                binsName = None
            else:
                raise JsonFormatException(json["bins:name"], "AdaptivelyBin.bins:name")
            if isinstance(json["bins"], list):
                bins = []
                for i, binpair in enumerate(json["bins"]):
                    if isinstance(binpair, dict) and hasKeys(binpair.keys(), ["center", "value"]):
                        if isinstance(binpair["center"], (int, long, float)):
                            center = float(binpair["center"])
                        else:
                            JsonFormatException(binpair["center"], "AdaptivelyBin.bins {} center".format(i))
                        
                        bins.append((center, factory.fromJsonFragment(binpair["value"], binsName)))

                    else:
                        raise JsonFormatException(binpair, "AdaptivelyBin.bins {}".format(i))

            if json["min"] in ("nan", "inf", "-inf") or isinstance(json["min"], (int, long, float)):
                min = float(json["min"])
            else:
                raise JsonFormatException(json, "AdaptivelyBin.min")

            if json["max"] in ("nan", "inf", "-inf") or isinstance(json["max"], (int, long, float)):
                max = float(json["max"])
            else:
                raise JsonFormatException(json, "AdaptivelyBin.max")

            if isinstance(json["nanflow:type"], basestring):
                nanflowFactory = Factory.registered[json["nanflow:type"]]
            else:
                raise JsonFormatException(json, "AdaptivelyBin.nanflow:type")
            nanflow = nanflowFactory.fromJsonFragment(json["nanflow"], None)

            if isinstance(json["tailDetail"], (int, long, float)):
                tailDetail = float(json["tailDetail"])
            else:
                raise JsonFormatException(json, "AdaptivelyBin.tailDetail")

            out = AdaptivelyBin.ed(entries, num, tailDetail, contentType, bins, min, max, nanflow)
            out.quantity.name = nameFromParent if name is None else name
            return out

        else:
            raise JsonFormatException(json, "AdaptivelyBin")

    def __repr__(self):
        if len(self.bins) > 0:
            v = self.bins[0][1]
        elif self.value is not None:
            v = self.value.name
        else:
            v = self.contentType
        return "AdaptivelyBin[bins=[{}..., size={}], nanflow={}]".format(v, len(self.bins), self.nanflow)

    def __eq__(self, other):
        return isinstance(other, AdaptivelyBin) and self.quantity == other.quantity and self.clustering == other.clustering

    def __hash__(self):
        return hash((self.quantity, self.clustering))

Factory.register(AdaptivelyBin)
