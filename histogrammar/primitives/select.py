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

################################################################ Select

class Select(Factory, Container):
    """Filter or weight data according to a given selection.

    This primitive is a basic building block, intended to be used in conjunction with anything that needs a user-defined cut. In particular, a standard histogram often has a custom selection, and this can be built by nesting Select -> Bin -> Count.

    Select also resembles :doc:`Fraction <histogrammar.primitives.fraction.Fraction>`, but without the ``denominator``.

    The efficiency of a cut in a Select aggregator named ``x`` is simply ``x.cut.entries / x.entries`` (because all aggregators have an ``entries`` member).
    """

    @staticmethod
    def ed(entries, cut):
        """Create a Select that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            cut (:doc:`Container <histogrammar.defs.Container>`): the filled sub-aggregator.
        """
        if not isinstance(entries, (int, long, float)):
            raise TypeError("entries ({0}) must be a number".format(entries))
        if not isinstance(cut, Container):
            raise TypeError("cut ({0}) must be a Container".format(cut))
        if entries < 0.0:
            raise ValueError("entries ({0}) cannot be negative".format(entries))
        out = Select(None, cut)
        out.entries = entries
        return out.specialize()

    @staticmethod
    def ing(quantity, cut):
        """Synonym for ``__init__``."""
        return Select(quantity, cut)

    def __getattr__(self, attr):
        """Pass on searches for custom methods to the ``value``, so that Limit becomes effectively invisible."""
        if attr.startswith("__") and attr.endswith("__"):
            return getattr(Select, attr)
        elif attr not in self.__dict__ and hasattr(self.__dict__["cut"], attr):
            return getattr(self.__dict__["cut"], attr)
        else:
            return self.__dict__[attr]

    def __init__(self, quantity, cut):
        """Create a Select that is capable of being filled and added.

        Parameters:
            quantity (function returning bool or float): computes the quantity of interest from the data and interprets it as a selection (multiplicative factor on weight).
            cut (:doc:`Container <histogrammar.defs.Container>`): will only be filled with data that pass the cut, and which are weighted by the cut.

        Other Parameters:
            entries (float): the number of entries, initially 0.0.
        """
        if not isinstance(cut, Container):
            raise TypeError("cut ({0}) must be a Container".format(cut))
        self.entries = 0.0
        self.quantity = serializable(quantity)
        self.cut = cut
        super(Select, self).__init__()
        self.specialize()

    def fractionPassing(self):
        """Fraction of weights that pass the quantity."""
        return self.cut.entries / self.entries

    @inheritdoc(Container)
    def zero(self):
        return Select(self.quantity, self.cut.zero())

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Select):
            out = Select(self.quantity, self.cut + other.cut)
            out.entries = self.entries + other.entries
            return out.specialize()
        else:
            raise ContainerException("cannot add {0} and {1}".format(self.name, other.name))

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()
        if weight > 0.0:
            w = self.quantity(datum)
            if not isinstance(w, (bool, int, long, float)):
                raise TypeError("function return value ({0}) must be boolean or number".format(w))
            w *= weight

            if w > 0.0:
                self.cut.fill(datum, w)
            # no possibility of exception from here on out (for rollback)
            self.entries += weight

    def fillnp(self, data, weight=1.0):
        """Increment the aggregator by providing a one-dimensional Numpy array of ``data`` to the fill rule with given ``weight`` (number or array).

        This primitive is optimized with Numpy.

        The container is changed in-place.
        """
        self._checkForCrossReferences()

        import numpy
        data, weight = self._normalizenp(data, weight)
        if not isinstance(weight, numpy.ndarray) and weight <= 0.0: return
        w = self._computenp(data)

        numpy.multiply(w, weight, w)

        selection = (w > 0.0)
        self.cut.fillnp(data[selection], w[selection])

        self._entriesnp(weight, data.shape[0])

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return [self.cut]

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName): return maybeAdd({
        "entries": floatToJson(self.entries),
        "type": self.cut.name,
        "data": self.cut.toJsonFragment(False),
        }, name=(None if suppressName else self.quantity.name))

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "type", "data"], ["name"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "Select.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Select.name")

            if isinstance(json["type"], basestring):
                factory = Factory.registered[json["type"]]
            else:
                raise JsonFormatException(json, "Select.type")

            cut = factory.fromJsonFragment(json["data"], None)

            out = Select.ed(entries, cut)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        else:
            raise JsonFormatException(json, "Select")

    def __repr__(self):
        return "<Select cut={0}>".format(self.cut.name)

    def __eq__(self, other):
        return isinstance(other, Select) and numeq(self.entries, other.entries) and self.cut == other.cut

    def __ne__(self, other): return not self == other

    def __hash__(self):
        return hash((self.entries, self.cut))

Factory.register(Select)

