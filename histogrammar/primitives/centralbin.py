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
    """Split a quantity into bins defined by irregularly spaced bin centers, with exactly one sub-aggregator filled per datum (the closest one).

    Unlike irregular bins defined by explicit ranges, irregular bins defined by bin centers are guaranteed to fully partition the space with no gaps and no overlaps. It could be viewed as cluster scoring in one dimension.

    The first and last bins cover semi-infinite domains, so it is unclear how to interpret them as part of the probability density function (PDF). Finite-width bins approximate the PDF in piecewise steps, but the first and last bins could be taken as zero (an underestimate) or as uniform from the most extreme point to the inner bin edge (an overestimate, but one that is compensated by underestimating the region just beyond the extreme point). For the sake of the latter interpretation, the minimum and maximum values are accumulated along with the bin values.
    """

    @staticmethod
    def ed(entries, bins, min, max, nanflow):
        """
        * `entries` (double) is the number of entries.
        * `bins` (list of double, past-tense aggregator pairs) is the list of bin centers and their accumulated data.
        * `min` (double) is the lowest value of the quantity observed or NaN if no data were observed.
        * `max` (double) is the highest value of the quantity observed or NaN if no data were observed.
        * `nanflow` (past-tense aggregator) is the filled nanflow bin.
        """
        if not isinstance(entries, (int, long, float)):
            raise TypeError("entries ({}) must be a number".format(entries))
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
        out = CentrallyBin(bins, None, None, nanflow)
        out.entries = entries
        out.bins = bins
        out.min = min
        out.max = max
        return out.specialize()

    @staticmethod
    def ing(bins, quantity, value=Count(), nanflow=Count()):
        """Synonym for ``__init__``."""
        return CentrallyBin(bins, quantity, value, nanflow)

    def __init__(self, bins, quantity, value=Count(), nanflow=Count()):
        """
        * `centers` (list of doubles) is the centers of all bins
        * `quantity` (function returning double) computes the quantity of interest from the data.
        * `value` (present-tense aggregator) generates sub-aggregators to put in each bin.
        * `nanflow` (present-tense aggregator) is a sub-aggregator to use for data whose quantity is NaN.
        * `entries` (mutable double) is the number of entries, initially 0.0.
        * `bins` (list of double, present-tense aggregator pairs) are the bin centers and sub-aggregators in each bin.
        * `min` (mutable double) is the lowest value of the quantity observed, initially NaN.
        * `max` (mutable double) is the highest value of the quantity observed, initially NaN.
        """

        if not isinstance(bins, (list, tuple)) and not all(isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], (int, long, float)) and isinstance(v[1], Container) for v in bins):
            raise TypeError("bins ({}) must be a list of number, Container pairs".format(bins))
        if value is not None and not isinstance(value, Container):
            raise TypeError("value ({}) must be None or a Container".format(value))
        if not isinstance(nanflow, Container):
            raise TypeError("nanflow ({}) must be a Container".format(nanflow))
        if len(bins) < 2:
            raise ValueError("number of bins ({}) must be at least two".format(len(bins)))

        self.entries = 0.0
        if value is None:
            self.bins = None
        else:
            self.bins = [(x, value.zero()) for x in sorted(bins)]
        self.min = float("nan")
        self.max = float("nan")

        self.quantity = serializable(quantity)
        self.value = value
        self.nanflow = nanflow.copy()

        super(CentrallyBin, self).__init__()
        self.specialize()

    def histogram(self):
        """Return a plain histogram by converting all sub-aggregator values into `Counts <histogrammar.primitives.count.Count>`_."""
        out = CentrallyBin(map(lambda x: x[0], self.bins), self.quantity, Count(), self.nanflow.copy())
        out.entries = self.entries
        for i, v in self.bins:
            out.bins[i] = Count.ed(v.entries)
        out.min = self.min
        out.max = self.max
        return out.specialize()

    @inheritdoc(Container)
    def zero(self):
        return CentrallyBin(map(lambda x: x[0], self.bins), self.quantity, self.value, self.nanflow.zero())

    @inheritdoc(Container)
    def __add__(self, other):
        if self.centers != other.centers:
            raise ContainerException("cannot add CentrallyBin because centers are different:\n    {}\nvs\n    {}".format(self.centers, other.centers))

        newbins = [(c1, v1 + v2) for (c1, v1), (_, v2) in zip(self.bins, other.bins)]

        out = CentrallyBin(map(lambda x: x[0], self.bins), self.quantity, self.value, self.nanflow + other.nanflow)
        out.entries = self.entries + other.entries
        out.bins = newbins
        out.min = minplus(self.min, other.min)
        out.max = maxplus(self.max, other.max)
        return out.specialize()

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()
        if weight > 0.0:
            q = self.quantity(datum)
            if not isinstance(q, (bool, int, long, float)):
                raise TypeError("function return value ({}) must be boolean or number".format(q))

            if self.nan(q):
                self.nanflow.fill(datum, weight)
            else:
                self.bins[self.index(q)][1].fill(datum, weight)

            # no possibility of exception from here on out (for rollback)
            self.entries += weight
            if math.isnan(self.min) or q < self.min:
                self.min = q
            if math.isnan(self.max) or q > self.max:
                self.max = q

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return [self.nanflow] + [v for c, v in self.bins]

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName):
        if getattr(self.bins[0][1], "quantity", None) is not None:
            binsName = self.bins[0][1].quantity.name
        elif getattr(self.bins[0][1], "quantityName", None) is not None:
            binsName = self.bins[0][1].quantityName
        else:
            binsName = None

        return maybeAdd({
            "entries": floatToJson(self.entries),
            "bins:type": self.bins[0][1].name,
            "bins": [{"center": floatToJson(c), "value": v.toJsonFragment(True)} for c, v in self.bins],
            "min": floatToJson(self.min),
            "max": floatToJson(self.max),
            "nanflow:type": self.nanflow.name,
            "nanflow": self.nanflow.toJsonFragment(False),
            }, **{"name": None if suppressName else self.quantity.name,
                  "bins:name": binsName})

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "bins:type", "bins", "min", "max", "nanflow:type", "nanflow"], ["name", "bins:name"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "CentrallyBin.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "CentrallyBin.name")

            if isinstance(json["bins:type"], basestring):
                factory = Factory.registered[json["bins:type"]]
            else:
                raise JsonFormatException(json, "CentrallyBin.bins:type")
            if isinstance(json.get("bins:name", None), basestring):
                binsName = json["bins:name"]
            elif json.get("bins:name", None) is None:
                binsName = None
            else:
                raise JsonFormatException(json["bins:name"], "CentrallyBin.bins:name")
            if isinstance(json["bins"], list):
                bins = []
                for i, binpair in enumerate(json["bins"]):
                    if isinstance(binpair, dict) and hasKeys(binpair.keys(), ["center", "value"]):
                        if isinstance(binpair["center"], (int, long, float)):
                            center = float(binpair["center"])
                        else:
                            JsonFormatException(binpair["center"], "CentrallyBin.bins {} center".format(i))
                        
                        bins.append((center, factory.fromJsonFragment(binpair["value"], binsName)))

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
            nanflow = nanflowFactory.fromJsonFragment(json["nanflow"], None)

            out = CentrallyBin.ed(entries, bins, min, max, nanflow)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        else:
            raise JsonFormatException(json, "CentrallyBin")

    def __repr__(self):
        return "<CentrallyBin bins={} size={} nanflow={}>".format(self.bins[0][1].name, len(self.bins), self.nanflow.name)

    def __eq__(self, other):
        return isinstance(other, CentrallyBin) and self.quantity == other.quantity and numeq(self.entries, other.entries) and self.bins == other.bins and numeq(self.min, other.min) and numeq(self.max, other.max) and self.nanflow == other.nanflow

    def __hash__(self):
        return hash((self.quantity, self.entries, tuple(self.bins), self.min, self.max, self.nanflow))

Factory.register(CentrallyBin)
