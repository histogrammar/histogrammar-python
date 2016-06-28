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

class Average(Factory, Container):
    """Accumulate the weighted mean of a given quantity.

    Uses the numerically stable weighted mean algorithm described in `"Incremental calculation of weighted mean and variance," <http://www-uxsup.csx.cam.ac.uk/~fanf2/hermes/doc/antiforgery/stats.pdf>`_ Tony Finch, *Univeristy of Cambridge Computing Service,* 2009.
    """

    @staticmethod
    def ed(entries, mean):
        """Create an Average that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            mean (float): the mean.
        """

        if not isinstance(entries, (int, long, float)):
            raise TypeError("entries ({0}) must be a number".format(entries))
        if not isinstance(mean, (int, long, float)):
            raise TypeError("mean ({0}) must be a number".format(mean))
        if entries < 0.0:
            raise ValueError("entries ({0}) cannot be negative".format(entries))
        out = Average(None)
        out.entries = float(entries)
        out.mean = float(mean)
        return out.specialize()

    @staticmethod
    def ing(quantity):
        """Synonym for ``__init__``."""
        return Average(quantity)

    def __init__(self, quantity):
        """Create an Average that is capable of being filled and added.

        Parameters:
            quantity (function returning float): computes the quantity of interest from the data.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
            mean (float): the running mean, initially 0.0. Note that this value contributes to the total mean with weight zero (because `entries` is initially zero), so this arbitrary choice does not bias the final result.
        """
        self.quantity = serializable(quantity)
        self.entries = 0.0
        self.mean = 0.0
        super(Average, self).__init__()
        self.specialize()

    @inheritdoc(Container)
    def zero(self): return Average(self.quantity)

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Average):
            out = Average(self.quantity)
            out.entries = self.entries + other.entries
            if out.entries == 0.0:
                out.mean = (self.mean + other.mean)/2.0
            else:
                out.mean = (self.entries*self.mean + other.entries*other.mean)/(self.entries + other.entries)
            return out.specialize()
        else:
            raise ContainerException("cannot add {0} and {1}".format(self.name, other.name))

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()
        if weight > 0.0:
            q = self.quantity(datum)
            if not isinstance(q, (bool, int, long, float)):
                raise TypeError("function return value ({0}) must be boolean or number".format(q))

            # no possibility of exception from here on out (for rollback)
            self.entries += weight
            delta = q - self.mean
            shift = delta * weight / self.entries
            self.mean += shift

    def fillnp(self, data, weight=1.0):
        """Increment the aggregator by providing a one-dimensional Numpy array of ``data`` to the fill rule with given ``weight`` (number or array).

        This primitive is optimized with Numpy.

        The container is changed in-place.
        """
        self._checkForCrossReferences()

        import numpy
        data, weight = self._normalizenp(data, weight)
        if not isinstance(weight, numpy.ndarray) and weight <= 0.0: return
        q = self._computenp(data)

        ca, ma = self.entries, self.mean

        self._entriesnp(weight, data.shape[0])

        ca_plus_cb = self.entries
        if ca_plus_cb > 0.0:
            mb = numpy.average(q, weights=(weight if isinstance(weight, numpy.ndarray) else None))
            self.mean = float((ca*ma + (ca_plus_cb - ca)*mb) / ca_plus_cb)

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return []

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName): return maybeAdd({
        "entries": floatToJson(self.entries),
        "mean": floatToJson(self.mean),
        }, name=(None if suppressName else self.quantity.name))

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "mean"], ["name"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json["entries"], "Average.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Average.name")

            if isinstance(json["mean"], (int, long, float)):
                mean = float(json["mean"])
            else:
                raise JsonFormatException(json["mean"], "Average.mean")

            out = Average.ed(entries, mean)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        else:
            raise JsonFormatException(json, "Average")
        
    def __repr__(self):
        return "<Average mean={0}>".format(self.mean)

    def __eq__(self, other):
        return isinstance(other, Average) and self.quantity == other.quantity and numeq(self.entries, other.entries) and numeq(self.mean, other.mean)

    def __ne__(self, other): return not self == other

    def __hash__(self):
        return hash((self.quantity, self.entries, self.mean))

Factory.register(Average)
