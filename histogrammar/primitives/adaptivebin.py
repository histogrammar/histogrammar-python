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
        if not isinstance(entries, (int, long, float)):
            raise TypeError("entries ({}) must be a number".format(entries))
        if not isinstance(num, (int, long)):
            raise TypeError("num ({}) must be an integer".format(num))
        if not isinstance(tailDetail, (int, long, float)):
            raise TypeError("tailDetail ({}) must be a number".format(tailDetail))
        if not isinstance(contentType, basestring):
            raise TypeError("contentType ({}) must be a string".format(contentType))
        if not isinstance(bins, (list, tuple)) and not all(isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], (int, long, float)) and isinstance(v[1], Container) for v in bins):
            raise TypeError("bins ({}) must be a list of number, Container pairs".format(bins))
        if not isinstance(min, (int, long, float)):
            raise TypeError("min ({}) must be a number".format(min))
        if not isinstance(max, (int, long, float)):
            raise TypeError("max ({}) must be a number".format(max))
        if not isinstance(nanflow, Container):
            raise TypeError("nanflow ({}) must be a Container".format(nanflow))
        if entries < 0.0:
            raise ValueError("entries ({}) cannot be negative".format(entries))
        if num < 2:
            raise ValueError("number of bins ({}) must be at least two".format(num))
        if tailDetail < 0.0 or tailDetail > 1.0:
            raise ValueError("tailDetail parameter ({}) must be between 0.0 and 1.0 inclusive".format(tailDetail))

        out = AdaptivelyBin(None, num, tailDetail, None, nanflow)
        out.clustering.entries = float(entries)
        out.clustering.values = bins
        out.clustering.min = min
        out.clustering.max = max
        out.contentType = contentType
        return out.specialize()

    @staticmethod
    def ing(quantity, num=100, tailDetail=0.2, value=Count(), nanflow=Count()):
        return AdaptivelyBin(quantity, num, tailDetail, value, nanflow)

    def __init__(self, quantity, num=100, tailDetail=0.2, value=Count(), nanflow=Count()):
        if not isinstance(num, (int, long)):
            raise TypeError("num ({}) must be an integer".format(num))
        if not isinstance(tailDetail, (int, long, float)):
            raise TypeError("tailDetail ({}) must be a number".format(tailDetail))
        if value is not None and not isinstance(value, Container):
            raise TypeError("value ({}) must be None or a Container".format(value))
        if not isinstance(nanflow, Container):
            raise TypeError("nanflow ({}) must be a Container".format(nanflow))
        if num < 2:
            raise ValueError("number of bins ({}) must be at least two".format(num))
        if tailDetail < 0.0 or tailDetail > 1.0:
            raise ValueError("tailDetail parameter ({}) must be between 0.0 and 1.0 inclusive".format(tailDetail))

        self.quantity = serializable(quantity)
        self.clustering = Clustering1D(num, tailDetail, value, [], float("nan"), float("nan"), 0.0)
        self.nanflow = nanflow.copy()
        super(AdaptivelyBin, self).__init__()
        self.specialize()

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

    def histogram(self):
        out = AdaptivelyBin(self.quantity, self.num, self.tailDetail, Count(), self.nanflow.copy())
        out.clustering.entries = float(self.entries)
        for i, v in self.clustering.values:
            out.clustering.values.append((i, Count.ed(v.entries)))
        out.clustering.min = self.min
        out.clustering.max = self.max
        out.clustering.contentType = "Count"
        return out.specialize()

    def zero(self):
        return AdaptivelyBin(self.quantity, self.num, self.tailDetail, self.clustering.value, self.nanflow.zero())

    def __add__(self, other):
        if self.num != other.num:
            raise ContainerException("cannot add AdaptivelyBin because number of bins is different ({} vs {})".format(self.num, other.num))
        if self.tailDetail != other.tailDetail:
            raise ContainerException("cannot add AdaptivelyBin because tailDetail parameter is different ({} vs {})".format(self.num, other.num))

        out = AdaptivelyBin(self.quantity, self.num, self.tailDetail, self.clustering.value, self.nanflow + other.nanflow)
        out.clustering = self.clustering.merge(other.clustering)
        return out.specialize()
        
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()
        if weight > 0.0:
            q = self.quantity(datum)
            if not isinstance(q, (bool, int, long, float)):
                raise TypeError("function return value ({}) must be boolean or number".format(q))

            self.clustering.update(q, datum, weight)

    @property
    def children(self):
        return [self.nanflow] + [v for c, v in self.bins]

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
            return out.specialize()

        else:
            raise JsonFormatException(json, "AdaptivelyBin")

    def __repr__(self):
        if len(self.bins) > 0:
            v = self.bins[0][1].name
        elif self.value is not None:
            v = self.value.name
        else:
            v = self.contentType
        return "<AdaptivelyBin bins={} size={} nanflow={}>".format(v, len(self.bins), self.nanflow.name)

    def __eq__(self, other):
        return isinstance(other, AdaptivelyBin) and self.quantity == other.quantity and self.clustering == other.clustering

    def __hash__(self):
        return hash((self.quantity, self.clustering))

Factory.register(AdaptivelyBin)
