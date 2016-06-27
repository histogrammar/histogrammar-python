#!/usr/bin/env python

# Copyright 2016 DIANA-HEP
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
        """Create a CentrallyBin that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            bins (list of float, :doc:`Container <histogrammar.defs.Container>` pairs): the list of bin centers and their accumulated data.
            min (float): the lowest value of the quantity observed or NaN if no data were observed.
            max (float): the highest value of the quantity observed or NaN if no data were observed.
            nanflow (:doc:`Container <histogrammar.defs.Container>`): the filled nanflow bin.
        """
        if not isinstance(entries, (int, long, float)):
            raise TypeError("entries ({0}) must be a number".format(entries))
        if not isinstance(bins, (list, tuple)) and not all(isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], (int, long, float)) and isinstance(v[1], Container) for v in bins):
            raise TypeError("bins ({0}) must be a list of number, Container pairs".format(bins))
        if not isinstance(min, (int, long, float)):
            raise TypeError("min ({0}) must be a number".format(min))
        if not isinstance(max, (int, long, float)):
            raise TypeError("max ({0}) must be a number".format(max))
        if not isinstance(nanflow, Container):
            raise TypeError("nanflow ({0}) must be a Container".format(nanflow))
        if entries < 0.0:
            raise ValueError("entries ({0}) cannot be negative".format(entries))
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
        """Create a CentrallyBin that is capable of being filled and added.

        Parameters:
            centers (list of float): the centers of all bins
            quantity (function returning float): computes the quantity of interest from the data.
            value (:doc:`Container <histogrammar.defs.Container>`): generates sub-aggregators to put in each bin.
            nanflow (:doc:`Container <histogrammar.defs.Container>`): a sub-aggregator to use for data whose quantity is NaN.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
            bins (list of float, :doc:`Container <histogrammar.defs.Container>` pairs): the bin centers and sub-aggregators in each bin.
            min (float): the lowest value of the quantity observed, initially NaN.
            max (float): the highest value of the quantity observed, initially NaN.
        """

        if not isinstance(bins, (list, tuple)) and not all(isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], (int, long, float)) and isinstance(v[1], Container) for v in bins):
            raise TypeError("bins ({0}) must be a list of number, Container pairs".format(bins))
        if value is not None and not isinstance(value, Container):
            raise TypeError("value ({0}) must be None or a Container".format(value))
        if not isinstance(nanflow, Container):
            raise TypeError("nanflow ({0}) must be a Container".format(nanflow))
        if len(bins) < 2:
            raise ValueError("number of bins ({0}) must be at least two".format(len(bins)))

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
        """Return a plain histogram by converting all sub-aggregator values into :doc:`Counts <histogrammar.primitives.count.Count>`."""
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
            raise ContainerException("cannot add CentrallyBin because centers are different:\n    {0}\nvs\n    {1}".format(self.centers, other.centers))

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
                raise TypeError("function return value ({0}) must be boolean or number".format(q))

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

    def fillnp(self, data, weight=1.0):
        """Increment the aggregator by providing a one-dimensional Numpy array of ``data`` to the fill rule with given ``weight`` (number or array).

        This primitive is optimized with Numpy.

        The container is changed in-place.
        """
        self._checkForCrossReferences()

        import numpy
        if not isinstance(data, numpy.ndarray):
            data = numpy.array(data)
        assert len(data.shape) == 1
        length = data.shape[0]

        q = self.quantity(data)
        assert isinstance(q, numpy.ndarray)
        assert len(q.shape) == 1
        assert q.shape[0] == length

        if isinstance(weight, numpy.ndarray):
            assert len(weight.shape) == 1
            assert weight.shape[0] == length

        selection = numpy.isnan(q)
        self.nanflow.fillnp(data[selection], weight[selection] if isinstance(weight, numpy.ndarray) else weight)
        
        numpy.bitwise_not(selection, selection)
        data = data[selection]
        q = q[selection]
        if isinstance(weight, numpy.ndarray):
            weight = weight[selection]

        selection = numpy.empty(q.shape, dtype=numpy.bool)
        selection2 = numpy.empty(q.shape, dtype=numpy.bool)

        for index in xrange(len(self.bins)):
            if index == 0:
                high = (self.bins[index][0] + self.bins[index + 1][0])/2.0
                numpy.less(q, high, selection)

            elif index == len(self.bins) - 1:
                low = (self.bins[index - 1][0] + self.bins[index][0])/2.0
                numpy.greater_equal(q, low, selection)

            else:
                low = (self.bins[index - 1][0] + self.bins[index][0])/2.0
                high = (self.bins[index][0] + self.bins[index + 1][0])/2.0
                numpy.greater_equal(q, low, selection)
                numpy.less(q, high, selection2)
                numpy.bitwise_and(selection, selection2, selection)

            self.bins[index][1].fillnp(data[selection], weight[selection] if isinstance(weight, numpy.ndarray) else weight)

        if isinstance(weight, numpy.ndarray):
            self.entries += float(weight[weight > 0.0].sum())
        elif weight > 0.0:
            self.entries += float(weight * length)

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
                            JsonFormatException(binpair["center"], "CentrallyBin.bins {0} center".format(i))
                        
                        bins.append((center, factory.fromJsonFragment(binpair["value"], binsName)))

                    else:
                        raise JsonFormatException(binpair, "CentrallyBin.bins {0}".format(i))

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
        return "<CentrallyBin bins={0} size={1} nanflow={2}>".format(self.bins[0][1].name, len(self.bins), self.nanflow.name)

    def __eq__(self, other):
        return isinstance(other, CentrallyBin) and self.quantity == other.quantity and numeq(self.entries, other.entries) and self.bins == other.bins and numeq(self.min, other.min) and numeq(self.max, other.max) and self.nanflow == other.nanflow

    def __hash__(self):
        return hash((self.quantity, self.entries, tuple(self.bins), self.min, self.max, self.nanflow))

Factory.register(CentrallyBin)
