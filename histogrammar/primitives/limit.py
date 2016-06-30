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

from histogrammar.defs import *
from histogrammar.util import *

################################################################ Limit

class Limit(Factory, Container):
    """Accumulate an aggregator until its number of entries reaches a predefined limit.

    Limit is intended to roll high-detail descriptions of small datasets over into low-detail descriptions of large datasets. For instance, a scatter plot is useful for small numbers of data points and heatmaps are useful for large ones. The following construction

    ::

        Bin(xbins, xlow, xhigh, lambda d: d.x,
          Bin(ybins, ylow, yhigh, lambda d: d.y,
            Limit(10.0, Bag(lambda d: [d.x, d.y]))))

    fills a scatter plot in all x-y bins that have fewer than 10 entries and only a number of entries above that. Postprocessing code would use the bin-by-bin numbers of entries to color a heatmap and the raw data points to show outliers in the nearly empty bins.

    Limit can effectively swap between two descriptions if it is embedded in a collection, such as :doc:`Branch <histogrammar.primitives.collection.Branch>`. All elements of the collection would be filled until the Limit saturates, leaving only the low-detail one. For instance, one could aggregate several :doc:`SparselyBin <histogrammar.primitives.sparsebin.SparselyBin>` histograms, each with a different ``binWidth``, and progressively eliminate them in order of increasing ``binWidth``.

    Note that Limit saturates when it reaches a specified *total weight,* not the number of data points in a :doc:`Bag <histogrammar.primitives.bag.Bag>`, so it is not certain to control memory use. However, the total weight is of more use to data analysis. (:doc:`Sample <histogrammar.primitives.sample.Sample>` puts a strict limit on memory use.)
    """

    @staticmethod
    def ed(entries, limit, contentType, value):
        """Create a Limit that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            limit (float): the maximum number of entries (inclusive).
            contentType (str): the value's sub-aggregator type (must be provided to determine type for the case when ``value`` has been deleted).
            value (:doc:`Container <histogrammar.defs.Container>` or ``None``) is the filled sub-aggregator if unsaturated, ``None`` if saturated.
        """
        if not isinstance(entries, (int, long, float)) and entries not in ("nan", "inf", "-inf"):
            raise TypeError("entries ({0}) must be a number".format(entries))
        if not isinstance(limit, (int, long, float)):
            raise TypeError("limit ({0}) must be a number".format(limit))
        if not isinstance(contentType, basestring):
            raise TypeError("contentType ({0}) must be a number".format(contentType))
        if value is not None and not isinstance(value, Container):
            raise TypeError("value ({0}) must be None or a Container".format(value))
        if entries < 0.0:
            raise ValueError("entries ({0}) cannot be negative".format(entries))

        out = Limit(limit, value)
        out.entries = float(entries)
        out.contentType = contentType
        return out.specialize()

    @staticmethod
    def ing(limit, value):
        """Synonym for ``__init__``."""
        return Limit(limit, value)

    def __init__(self, limit, value):
        """Create a Limit that is capable of being filled and added.

        Parameters:
            limit (float): the maximum number of entries (inclusive) before deleting the `value`.
            value (:doc:`Container <histogrammar.defs.Container>`): will only be filled until its number of entries exceeds the `limit`.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
            contentType (str): the value's sub-aggregator type (must be provided to determine type for the case when `value` has been deleted).
        """
        if not isinstance(limit, (int, long, float)):
            raise TypeError("limit ({0}) must be a number".format(limit))
        if value is not None and not isinstance(value, Container):
            raise TypeError("value ({0}) must be None or a Container".format(value))

        self.entries = 0.0
        self.limit = limit
        if value is None:
            self.contentType = None
        else:
            self.contentType = value.name
        self.value = value
        super(Limit, self).__init__()
        self.specialize()

    def __getattr__(self, attr):
        """Pass on searches for custom methods to the ``value``, so that Limit becomes effectively invisible."""
        if attr.startswith("__") and attr.endswith("__"):
            return getattr(Limit, attr)
        elif attr not in self.__dict__ and hasattr(self.__dict__["value"], attr):
            return getattr(self.__dict__["value"], attr)
        else:
            return self.__dict__[attr]

    @property
    def saturated(self):
        """True if ``entries`` exceeds ``limit`` and ``value`` is ``None``."""
        return self.value is None

    @property
    def get(self):
        """Get the value of ``value`` or raise an error if it is ``None``."""
        if self.value is None:
            raise TypeError("get called on Limit whose value is None")
        return self.value

    def getOrElse(self, default):
        """Get the value of ``value`` or return a default if it is ``None``."""
        if self.value is None:
            return default
        else:
            return self.value

    @inheritdoc(Container)
    def zero(self):
        return Limit.ed(0.0, self.limit, self.contentType, None if self.value is None else self.value.zero())

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Limit):
            if self.limit != other.limit:
                raise ContainerExeption("cannot add Limit because they have different limits ({0} vs {1})".format(self.limit, other.limit))
            else:
                newentries = self.entries + other.entries
                if newentries > self.limit:
                    newvalue = None
                else:
                    newvalue = self.value + other.value

                return Limit.ed(newentries, self.limit, self.contentType, newvalue)

        else:
            raise ContainerException("cannot add {0} and {1}".format(self.name, other.name))

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()
        if weight > 0.0:
            if self.entries + weight > self.limit:
                self.value = None
            elif self.value is not None:
                self.value.fill(datum, weight)

            # no possibility of exception from here on out (for rollback)
            self.entries += weight

    def _numpy(self, data, weights, shape):
        w = self.quantity(data)
        self._checkNPQuantity(w, shape)
        self._checkNPWeights(weights, shape)
        weights = self._makeNPWeights(weights, shape)

        newentries = weights.sum()

        if self.entries + newentries > self.limit:
            self.value = None
        elif self.value is not None:
            self.value._numpy(data, weights, shape)

        self.entries += float(newentries)

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return [] if self.value is None else [self.value]

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName): return {
        "entries": floatToJson(self.entries),
        "limit": floatToJson(self.limit),
        "type": self.contentType,
        "data": None if self.value is None else self.value.toJsonFragment(False),
        }

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "limit", "type", "data"]):
            if json["entries"] in ("nan", "inf", "-inf") or isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "Limit.entries")

            if json["limit"] in ("nan", "inf", "-inf") or isinstance(json["limit"], (int, long, float)):
                limit = float(json["limit"])
            else:
                raise JsonFormatException(json, "Limit.limit")

            if isinstance(json["type"], basestring):
                contentType = json["type"]
            else:
                raise JsonFormatException(json, "Limit.type")
            factory = Factory.registered[contentType]

            if json["data"] is None:
                value = None
            else:
                value = factory.fromJsonFragment(json["data"], None)

            return Limit.ed(entries, limit, contentType, value)

        else:
            raise JsonFormatException(json, "Limit")

    def __repr__(self):
        return "<Limit value={0}>".format("saturated" if self.saturated else self.value.name)

    def __eq__(self, other):
        return isinstance(other, Limit) and numeq(self.entries, other.entries) and numeq(self.limit, other.limit) and self.contentType == other.contentType and self.value == other.value

    def __ne__(self, other): return not self == other

    def __hash__(self):
        return hash((self.entries, self.limit, self.contentType, self.value))

Factory.register(Limit)

