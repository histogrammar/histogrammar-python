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

class AdaptivelyBin(Factory, Container, CentralBinsDistribution, CentrallyBinMethods):
    """Adaptively partition a domain into bins and fill them at the same time using a clustering algorithm. Each input datum contributes to exactly one final bin.

    The algorithm is based on `"A streaming parallel decision tree algorithm," <http://www.jmlr.org/papers/volume11/ben-haim10a/ben-haim10a.pdf>`_ Yael Ben-Haim and Elad Tom-Tov, *J. Machine Learning Research 11,* 2010 with a small modification for display histograms.

    Yael Ben-Haim and Elad Tom-Tov's algorithm adds each new data point as a new bin containing a single value, then merges the closest bins if the total number of bins exceeds a maximum (like hierarchical clustering in one dimension).

    This tends to provide the most detail on the tails of a distribution (which have the most widely spaced bins), and is therefore a good alternative to [Quantile](#quantile-such-as-median-quartiles-quintiles-etc) for estimating extreme quantiles like 0.01 and 0.99.

    However, a histogram binned this way is less interesting for visualizing a distribution. Usually, the least interesting bins are the ones with the fewest entries, so one can consider merging the bins with the fewest entries, giving no detail on the tails.

    As a compromise, we introduce a "tail detail" hyperparameter that strikes a balance between the two extremes: the bins that are merged minimize

    ::

        tailDetail*(pos2 - pos1)/(max - min) + (1.0 - tailDetail)*(entries1 + entries2)/entries

    where ``pos1`` and ``pos2`` are the (ordered) positions of the two bins, ``min`` and ``max`` are the minimum and maximum positions of all entries, ``entries1`` and ``entries2`` are the number of entries in the two bins, and ``entries`` is the total number of entries in all bins. The denominators normalize the scales of domain position and number of entries so that ``tailDetail`` may be unitless and between 0.0 and 1.0 (inclusive).

    A value of ``tailDetail = 0.2`` is a good default.

    This algorithm is deterministic; the same input data yield the same histogram.
    """

    @staticmethod
    def ed(entries, num, tailDetail, contentType, bins, min, max, nanflow):
        """Create an AdaptivelyBin that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            num (int): specifies the maximum number of bins before merging.
            tailDetail (float): a value between 0.0 and 1.0 (inclusive) for choosing the pair of bins to merge (see above).
            contentType (str): the value's sub-aggregator type (must be provided to determine type for the case when `bins` is empty).
            bins (list of double, :doc:`Container <histogrammar.defs.Container>`): is the list of bin centers and bin contents. The domain of each bin is determined as in :doc:`CentrallyBin <histogrammar.primitives.centralbin.CentrallyBin>`.
            min (float): the lowest value of the quantity observed or NaN if no data were observed.
            max (float): the highest value of the quantity observed or NaN if no data were observed.
            nanflow (:doc:`Container <histogrammar.defs.Container>`): the filled nanflow bin.
        """

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
        """Synonym for ``__init__``."""
        return AdaptivelyBin(quantity, num, tailDetail, value, nanflow)

    def __init__(self, quantity, num=100, tailDetail=0.2, value=Count(), nanflow=Count()):
        """Create an AdaptivelyBin that is capable of being filled and added.

        Parameters:
            quantity (function returning float): computes the quantity of interest from the data.
            num (int): specifies the maximum number of bins before merging.
            tailDetail (float): a value between 0.0 and 1.0 (inclusive) for choosing the pair of bins to merge (see above).
            value (:doc:`Container <histogrammar.defs.Container>`): generates sub-aggregators to put in each bin.
            nanflow (:doc:`Container <histogrammar.defs.Container>`): is a sub-aggregator to use for data whose quantity is NaN.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
            bins (list of float, :doc:`Container <histogrammar.defs.Container>`): the list of bin centers and bin contents. The domain of each bin is determined as in :doc:`CentrallyBin <histogrammar.primitives.centralbin.CentrallyBin>`.
            min (float): the lowest value of the quantity observed, initially NaN.
            max (float): the highest value of the quantity observed, initially NaN.
        """

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
    def num(self):
        """Maximum number of bins (used as a constraint when merging)."""
        return self.clustering.num

    @property
    def tailDetail(self):
        """Clustering hyperparameter, between 0.0 and 1.0 inclusive: use 0.0 to focus on the bulk of the distribution and 1.0 to focus on the tails; see :doc:`Clustering1D <histogrammar.util.Clustering1D>` for details."""
        return self.clustering.tailDetail

    @property
    def entries(self):
        """Every :doc:`Container <histogrammar.defs.Container>` accumulates a sum of weights of observed data."""
        return self.clustering.entries

    @entries.setter
    def entries(self, value):
        self.clustering.entries = value

    @property
    def bins(self):
        """Center of bin, sub-aggregator pairs."""
        return self.clustering.values

    @property
    def min(self):
        """Minimum ``quantity`` observed so far (initially NaN)."""
        return self.clustering.min

    @min.setter
    def min(self, value):
        self.clustering.min = min

    @property
    def max(self):
        """Maximum ``quantity`` observed so far (initially NaN)."""
        return self.clustering.max

    @max.setter
    def max(self, value):
        self.clustering.max = max

    def histogram(self):
        """Return a plain histogram by converting all sub-aggregator values into :doc:`Counts <histogrammar.primitives.count.Count>`."""
        out = AdaptivelyBin(self.quantity, self.num, self.tailDetail, Count(), self.nanflow.copy())
        out.clustering.entries = float(self.entries)
        for i, v in self.clustering.values:
            out.clustering.values.append((i, Count.ed(v.entries)))
        out.clustering.min = self.min
        out.clustering.max = self.max
        out.clustering.contentType = "Count"
        return out.specialize()

    @inheritdoc(Container)
    def zero(self):
        return AdaptivelyBin(self.quantity, self.num, self.tailDetail, self.clustering.value, self.nanflow.zero())

    @inheritdoc(Container)
    def __add__(self, other):
        if self.num != other.num:
            raise ContainerException("cannot add AdaptivelyBin because number of bins is different ({} vs {})".format(self.num, other.num))
        if self.tailDetail != other.tailDetail:
            raise ContainerException("cannot add AdaptivelyBin because tailDetail parameter is different ({} vs {})".format(self.num, other.num))

        out = AdaptivelyBin(self.quantity, self.num, self.tailDetail, self.clustering.value, self.nanflow + other.nanflow)
        out.clustering = self.clustering.merge(other.clustering)
        return out.specialize()
        
    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()
        if weight > 0.0:
            q = self.quantity(datum)
            if not isinstance(q, (bool, int, long, float)):
                raise TypeError("function return value ({}) must be boolean or number".format(q))

            self.clustering.update(q, datum, weight)

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return [self.nanflow] + [v for c, v in self.bins]

    @inheritdoc(Container)
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
    @inheritdoc(Factory)
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
