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

import numpy

from histogrammar.defs import (
    Container,
    ContainerException,
    Factory,
    JsonFormatException,
    identity,
)
from histogrammar.util import (
    basestring,
    datatype,
    floatToJson,
    hasKeys,
    inheritdoc,
    maxplus,
    maybeAdd,
    minplus,
    n_dim,
    numeq,
    serializable,
)


class Minimize(Factory, Container):
    """Find the minimum value of a given quantity. If no data are observed, the result is NaN."""

    @staticmethod
    def ed(entries, min):
        """Create a Minimize that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            min (float): the lowest value of the quantity observed or NaN if no data were observed.
        """
        if not isinstance(entries, numbers.Real) and entries not in (
            "nan",
            "inf",
            "-inf",
        ):
            raise TypeError(f"entries ({entries}) must be a number")
        if not isinstance(min, numbers.Real) and entries not in ("nan", "inf", "-inf"):
            raise TypeError(f"min ({min}) must be a number")
        if entries < 0.0:
            raise ValueError(f"entries ({entries}) cannot be negative")
        out = Minimize(None)
        out.entries = float(entries)
        out.min = float(min)
        return out.specialize()

    @staticmethod
    def ing(quantity):
        """Synonym for ``__init__``."""
        return Minimize(quantity)

    def __init__(self, quantity=identity):
        """Create a Minimize that is capable of being filled and added.

        Parameters:
            quantity (function returning float): computes the quantity of interest from the data.

        Other parameters:
            entries (float): the number of entries, initially 0.0. #
            min (float): the lowest value of the quantity observed, initially NaN.
        """
        self.quantity = serializable(identity(quantity) if isinstance(quantity, str) else quantity)
        self.entries = 0.0
        self.min = float("nan")
        super().__init__()
        self.specialize()

    @inheritdoc(Container)
    def zero(self):
        return Minimize(self.quantity)

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Minimize):
            out = Minimize(self.quantity)
            out.entries = self.entries + other.entries
            out.min = minplus(self.min, other.min)
            return out.specialize()
        raise ContainerException(f"cannot add {self.name} and {other.name}")

    @inheritdoc(Container)
    def __iadd__(self, other):
        both = self + other
        self.entries = both.entries
        self.min = both.min
        return self

    @inheritdoc(Container)
    def __mul__(self, factor):
        if math.isnan(factor) or factor <= 0.0:
            return self.zero()
        out = self.zero()
        out.entries = factor * self.entries
        out.min = self.min
        return out.specialize()

    @inheritdoc(Container)
    def __rmul__(self, factor):
        return self.__mul__(factor)

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return []

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()

        if weight > 0.0:
            q = self.quantity(datum)
            if not isinstance(q, numbers.Real):
                raise TypeError(f"function return value ({q}) must be boolean or number")

            # no possibility of exception from here on out (for rollback)
            self.entries += weight
            if math.isnan(self.min) or q < self.min:
                self.min = q

    def _numpy(self, data, weights, shape):
        q = self.quantity(data)
        self._checkNPQuantity(q, shape)
        self._checkNPWeights(weights, shape)
        weights = self._makeNPWeights(weights, shape)

        # no possibility of exception from here on out (for rollback)
        selection = numpy.isnan(q)
        numpy.bitwise_not(selection, selection)
        numpy.bitwise_and(selection, weights > 0.0, selection)
        q = q[selection]

        self.entries += float(weights.sum())

        if math.isnan(self.min):
            if q.shape[0] > 0:
                self.min = float(q.min())
        elif q.shape[0] > 0:
            self.min = min(self.min, float(q.min()))

    def _sparksql(self, jvm, converter):
        return converter.Minimize(self.quantity.asSparkSQL())

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName):
        return maybeAdd(
            {"entries": floatToJson(self.entries), "min": floatToJson(self.min)},
            name=(None if suppressName else self.quantity.name),
        )

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "min"], ["name"]):
            if json["entries"] in ("nan", "inf", "-inf") or isinstance(json["entries"], numbers.Real):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json["entries"], "Minimize.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Minimize.name")

            if json["min"] in ("nan", "inf", "-inf") or isinstance(json["min"], numbers.Real):
                min = float(json["min"])
            else:
                raise JsonFormatException(json["min"], "Minimize.min")

            out = Minimize.ed(entries, min)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        raise JsonFormatException(json, "Minimize")

    def __repr__(self):
        return f"<Minimize min={self.min}>"

    def __eq__(self, other):
        return (
            isinstance(other, Minimize)
            and self.quantity == other.quantity
            and numeq(self.entries, other.entries)
            and numeq(self.min, other.min)
        )

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.quantity, self.entries, self.min))


# extra properties: number of dimensions and datatypes of sub-hists
Minimize.n_dim = n_dim
Minimize.datatype = datatype

Factory.register(Minimize)


class Maximize(Factory, Container):
    """Find the maximum value of a given quantity. If no data are observed, the result is NaN."""

    @staticmethod
    def ed(entries, max):
        """Create a Maximize that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            max (float): the highest value of the quantity observed or NaN if no data were observed.
        """
        if not isinstance(entries, numbers.Real) and entries not in (
            "nan",
            "inf",
            "-inf",
        ):
            raise TypeError(f"entries ({entries}) must be a number")
        if not isinstance(max, numbers.Real) and entries not in ("nan", "inf", "-inf"):
            raise TypeError(f"max ({max}) must be a number")
        if entries < 0.0:
            raise ValueError(f"entries ({entries}) cannot be negative")
        out = Maximize(None)
        out.entries = float(entries)
        out.max = float(max)
        return out.specialize()

    @staticmethod
    def ing(quantity):
        """Synonym for ``__init__``."""
        return Maximize(quantity)

    def __init__(self, quantity=identity):
        """Create a Maximize that is capable of being filled and added.

        Parameters:
            quantity (function returning float): computes the quantity of interest from the data.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
            max (float): the highest value of the quantity observed, initially NaN.
        """
        self.quantity = serializable(identity(quantity) if isinstance(quantity, str) else quantity)
        self.entries = 0.0
        self.max = float("nan")
        super().__init__()
        self.specialize()

    @inheritdoc(Container)
    def zero(self):
        return Maximize(self.quantity)

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Maximize):
            out = Maximize(self.quantity)
            out.entries = self.entries + other.entries
            out.max = maxplus(self.max, other.max)
            return out.specialize()
        raise ContainerException(f"cannot add {self.name} and {other.name}")

    @inheritdoc(Container)
    def __iadd__(self, other):
        both = self + other
        self.entries = both.entries
        self.max = both.max
        return self

    @inheritdoc(Container)
    def __mul__(self, factor):
        if math.isnan(factor) or factor <= 0.0:
            return self.zero()
        out = self.zero()
        out.entries = factor * self.entries
        out.max = self.max
        return out.specialize()

    @inheritdoc(Container)
    def __rmul__(self, factor):
        return self.__mul__(factor)

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()

        if weight > 0.0:
            q = self.quantity(datum)
            if not isinstance(q, numbers.Real):
                raise TypeError(f"function return value ({q}) must be boolean or number")

            # no possibility of exception from here on out (for rollback)
            self.entries += weight
            if math.isnan(self.max) or q > self.max:
                self.max = q

    def _numpy(self, data, weights, shape):
        q = self.quantity(data)
        self._checkNPQuantity(q, shape)
        self._checkNPWeights(weights, shape)
        weights = self._makeNPWeights(weights, shape)

        # no possibility of exception from here on out (for rollback)
        selection = numpy.isnan(q)
        numpy.bitwise_not(selection, selection)
        numpy.bitwise_and(selection, weights > 0.0, selection)
        q = q[selection]

        self.entries += float(weights.sum())

        if math.isnan(self.max):
            if q.shape[0] > 0:
                self.max = float(q.max())
        elif q.shape[0] > 0:
            self.max = max(self.max, float(q.max()))

    def _sparksql(self, jvm, converter):
        return converter.Maximize(self.quantity.asSparkSQL())

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return []

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName):
        return maybeAdd(
            {"entries": floatToJson(self.entries), "max": floatToJson(self.max)},
            name=(None if suppressName else self.quantity.name),
        )

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "max"], ["name"]):
            if json["entries"] in ("nan", "inf", "-inf") or isinstance(json["entries"], numbers.Real):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json["entries"], "Maximize.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Maximize.name")

            if json["max"] in ("nan", "inf", "-inf") or isinstance(json["max"], numbers.Real):
                max = float(json["max"])
            else:
                raise JsonFormatException(json["max"], "Maximize.max")

            out = Maximize.ed(entries, max)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        raise JsonFormatException(json, "Maximize")

    def __repr__(self):
        return f"<Maximize max={self.max}>"

    def __eq__(self, other):
        return (
            isinstance(other, Maximize)
            and self.quantity == other.quantity
            and numeq(self.entries, other.entries)
            and numeq(self.max, other.max)
        )

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.quantity, self.entries, self.max))


# extra properties: number of dimensions and datatypes of sub-hists
Maximize.n_dim = n_dim
Maximize.datatype = datatype

# register extra methods
Factory.register(Maximize)
