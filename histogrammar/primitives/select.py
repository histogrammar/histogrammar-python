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

import math
import numbers

from histogrammar.defs import (
    Container,
    ContainerException,
    Factory,
    JsonFormatException,
    identity,
)
from histogrammar.primitives.count import Count
from histogrammar.util import (
    basestring,
    datatype,
    floatToJson,
    hasKeys,
    inheritdoc,
    maybeAdd,
    n_dim,
    numeq,
    serializable,
)

# Select


class Select(Factory, Container):
    """Filter or weight data according to a given selection.

    This primitive is a basic building block, intended to be used in conjunction with anything that needs a
    user-defined cut. In particular, a standard histogram often has a custom selection, and this can be built by
    nesting Select -> Bin -> Count.

    Select also resembles :doc:`Fraction <histogrammar.primitives.fraction.Fraction>`, but without the ``denominator``.

    The efficiency of a cut in a Select aggregator named ``x`` is simply ``x.cut.entries / x.entries``
    (because all aggregators have an ``entries`` member).
    """

    @staticmethod
    def ed(entries, cut):
        """Create a Select that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            cut (:doc:`Container <histogrammar.defs.Container>`): the filled sub-aggregator.
        """
        if not isinstance(entries, numbers.Real) and entries not in (
            "nan",
            "inf",
            "-inf",
        ):
            raise TypeError(f"entries ({entries}) must be a number")
        if not isinstance(cut, Container):
            raise TypeError(f"cut ({cut}) must be a Container")
        if entries < 0.0:
            raise ValueError(f"entries ({entries}) cannot be negative")
        out = Select(None, cut)
        out.entries = float(entries)
        return out.specialize()

    @staticmethod
    def ing(quantity, cut=Count()):
        """Synonym for ``__init__``."""
        return Select(quantity, cut)

    def __getattr__(self, attr):
        """Pass on searches for custom methods to the ``value``, so that Limit becomes effectively invisible."""
        if attr.startswith("__") and attr.endswith("__"):
            return getattr(Select, attr)
        if attr not in self.__dict__ and hasattr(self.__dict__["cut"], attr):
            return getattr(self.__dict__["cut"], attr)
        return self.__dict__[attr]

    def __init__(self, quantity=identity, cut=Count()):
        """Create a Select that is capable of being filled and added.

        Parameters:
            quantity (function returning bool or float): computes the quantity of interest from the data and interprets
                it as a selection (multiplicative factor on weight).
            cut (:doc:`Container <histogrammar.defs.Container>`): will only be filled with data that pass the cut,
                and which are weighted by the cut.

        Other Parameters:
            entries (float): the number of entries, initially 0.0.
        """
        if not isinstance(cut, Container):
            raise TypeError(f"cut ({cut}) must be a Container")
        self.entries = 0.0
        self.quantity = serializable(identity(quantity) if isinstance(quantity, str) else quantity)
        self.cut = cut
        super().__init__()
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
        raise ContainerException(f"cannot add {self.name} and {other.name}")

    @inheritdoc(Container)
    def __iadd__(self, other):
        if isinstance(other, Select):
            self.entries += other.entries
            self.cut += other.cut
            return self
        raise ContainerException(f"cannot add {self.name} and {other.name}")

    @inheritdoc(Container)
    def __mul__(self, factor):
        if math.isnan(factor) or factor <= 0.0:
            return self.zero()
        out = self.zero()
        out.entries = factor * self.entries
        out.cut = self.cut * factor
        return out.specialize()

    @inheritdoc(Container)
    def __rmul__(self, factor):
        return self.__mul__(factor)

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()

        if weight > 0.0:
            w = self.quantity(datum)
            if not isinstance(w, numbers.Real):
                raise TypeError(f"function return value ({w}) must be boolean or number")
            w *= weight

            if w > 0.0:
                self.cut.fill(datum, w)
            # no possibility of exception from here on out (for rollback)
            self.entries += weight

    def _numpy(self, data, weights, shape):
        w = self.quantity(data)
        self._checkNPQuantity(w, shape)
        self._checkNPWeights(weights, shape)
        weights = self._makeNPWeights(weights, shape)

        import numpy

        w = w * weights
        w[numpy.isnan(w)] = 0.0
        w[w < 0.0] = 0.0

        self.cut._numpy(data, w, shape)

        # no possibility of exception from here on out (for rollback)
        self.entries += float(weights.sum())

    def _sparksql(self, jvm, converter):
        return converter.Select(self.quantity.asSparkSQL(), self.cut._sparksql(jvm, converter))

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return [self.cut]

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName):
        return maybeAdd(
            {
                "entries": floatToJson(self.entries),
                "sub:type": self.cut.name,
                "data": self.cut.toJsonFragment(False),
            },
            name=(None if suppressName else self.quantity.name),
        )

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "sub:type", "data"], ["name"]):
            if json["entries"] in ("nan", "inf", "-inf") or isinstance(json["entries"], numbers.Real):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "Select.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Select.name")

            if isinstance(json["sub:type"], basestring):
                factory = Factory.registered[json["sub:type"]]
            else:
                raise JsonFormatException(json, "Select.type")

            cut = factory.fromJsonFragment(json["data"], None)

            out = Select.ed(entries, cut)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        raise JsonFormatException(json, "Select")

    def __repr__(self):
        return f"<Select cut={self.cut.name}>"

    def __eq__(self, other):
        return isinstance(other, Select) and numeq(self.entries, other.entries) and self.cut == other.cut

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.entries, self.cut))


# extra properties: number of dimensions and datatypes of sub-hists
Select.n_dim = n_dim
Select.datatype = datatype

# register extra methods
Factory.register(Select)
