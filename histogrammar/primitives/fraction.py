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


class Fraction(Factory, Container):
    """Accumulate two aggregators, one numerator and one denominator

    Accumulate two aggregators, one containing only entries that pass a given selection (numerator) and another
    that contains all entries (denominator).

    The aggregator may be a simple :doc:`Count <histogrammar.primitives.count.Count>` to measure the efficiency of a
    cut, a :doc:`Bin <histogrammar.primitives.bin.Bin>` to plot a turn-on curve, or anything else to be tested with
    and without a cut.

    As a side effect of NaN values returning false for any comparison, a NaN return value from the selection is
    treated as a failed cut (the denominator is filled but the numerator is not).
    """

    @staticmethod
    def ed(entries, numerator, denominator):
        """Create a Fraction that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            numerator: (:doc:`Container <histogrammar.defs.Container>`): the filled numerator.
            denominator (:doc:`Container <histogrammar.defs.Container>`): the filled denominator.
        """
        if not isinstance(entries, numbers.Real) and entries not in (
            "nan",
            "inf",
            "-inf",
        ):
            raise TypeError(f"entries ({entries}) must be a number")
        if not isinstance(numerator, Container):
            raise TypeError(f"numerator ({numerator}) must be a Container")
        if not isinstance(denominator, Container):
            raise TypeError(f"denominator ({denominator}) must be a Container")
        if entries < 0.0:
            raise ValueError(f"entries ({entries}) cannot be negative")

        out = Fraction(None, None)
        out.entries = float(entries)
        out.numerator = numerator
        out.denominator = denominator
        return out.specialize()

    @staticmethod
    def ing(quantity, value=Count()):
        """Synonym for ``__init__``."""
        return Fraction(quantity, value)

    def __init__(self, quantity=identity, value=Count()):
        """Create a Fraction that is capable of being filled and added.

        Parameters:
            quantity (function returning bool or float): computes the quantity of interest from the data and interprets
                it as a selection (multiplicative factor on weight).
            value (:doc:`Container <histogrammar.defs.Container>`): generates sub-aggregators for the numerator and
                denominator.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
            numerator (:doc:`Container <histogrammar.defs.Container>`): the sub-aggregator of entries that pass
                the selection.
            denominator (:doc:`Container <histogrammar.defs.Container>`): the sub-aggregator of all entries.
        """
        if value is not None and not isinstance(value, Container):
            raise TypeError(f"value ({value}) must be None or a Container")
        self.entries = 0.0
        self.quantity = serializable(identity(quantity) if isinstance(quantity, str) else quantity)
        if value is not None:
            self.numerator = value.zero()
            self.denominator = value.zero()
        super().__init__()
        self.specialize()

    @staticmethod
    def build(numerator, denominator):
        """Create a Fraction out of pre-existing containers, which might have been aggregated on different streams.

        Parameters:
            numerator (:doc:`Container <histogrammar.defs.Container>`): the filled numerator.
            denominator (:doc:`Container <histogrammar.defs.Container>`): the filled denominator.

        This function will attempt to combine the ``numerator`` and ``denominator``, so they must have the same
        binning/bounds/etc.
        """
        if not isinstance(numerator, Container):
            raise TypeError(f"numerator ({numerator}) must be a Container")
        if not isinstance(denominator, Container):
            raise TypeError(f"denominator ({denominator}) must be a Container")
        # check for compatibility
        numerator + denominator
        # return object
        return Fraction.ed(denominator.entries, numerator, denominator)

    @inheritdoc(Container)
    def zero(self):
        out = Fraction(self.quantity, None)
        out.numerator = self.numerator.zero()
        out.denominator = self.denominator.zero()
        return out.specialize()

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Fraction):
            out = Fraction(self.quantity, None)
            out.entries = self.entries + other.entries
            out.numerator = self.numerator + other.numerator
            out.denominator = self.denominator + other.denominator
            return out.specialize()
        raise ContainerException(f"cannot add {self.name} and {other.name}")

    @inheritdoc(Container)
    def __iadd__(self, other):
        if isinstance(other, Fraction):
            self.entries += other.entries
            self.numerator += other.numerator
            self.denominator += other.denominator
            return self
        raise ContainerException(f"cannot add {self.name} and {other.name}")

    @inheritdoc(Container)
    def __mul__(self, factor):
        if math.isnan(factor) or factor <= 0.0:
            return self.zero()
        out = self.zero()
        out.entries = factor * self.entries
        out.numerator = self.numerator * factor
        out.denominator = self.denominator * factor
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

            self.denominator.fill(datum, weight)
            if w > 0.0:
                self.numerator.fill(datum, w)

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

        self.numerator._numpy(data, w, shape)
        self.denominator._numpy(data, weights, shape)

        # no possibility of exception from here on out (for rollback)
        self.entries += float(weights.sum())

    def _sparksql(self, jvm, converter):
        return converter.Fraction(self.quantity.asSparkSQL(), self.numerator._sparksql(jvm, converter))

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return [self.numerator, self.denominator]

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName):
        if getattr(self.numerator, "quantity", None) is not None:
            binsName = self.numerator.quantity.name
        elif getattr(self.numerator, "quantityName", None) is not None:
            binsName = self.numerator.quantityName
        else:
            binsName = None

        return maybeAdd(
            {
                "entries": floatToJson(self.entries),
                "sub:type": self.numerator.name,
                "numerator": self.numerator.toJsonFragment(True),
                "denominator": self.denominator.toJsonFragment(True),
            },
            **{
                "name": None if suppressName else self.quantity.name,
                "sub:name": binsName,
            },
        )

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(
            json.keys(),
            ["entries", "sub:type", "numerator", "denominator"],
            ["name", "sub:name"],
        ):
            if json["entries"] in ("nan", "inf", "-inf") or isinstance(json["entries"], numbers.Real):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "Fraction.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Fraction.name")

            if isinstance(json["sub:type"], basestring):
                factory = Factory.registered[json["sub:type"]]
            else:
                raise JsonFormatException(json, "Fraction.type")

            if isinstance(json.get("sub:name", None), basestring):
                subName = json["sub:name"]
            elif json.get("sub:name", None) is None:
                subName = None
            else:
                raise JsonFormatException(json["sub:name"], "Fraction.sub:name")

            numerator = factory.fromJsonFragment(json["numerator"], subName)
            denominator = factory.fromJsonFragment(json["denominator"], subName)

            out = Fraction.ed(entries, numerator, denominator)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        raise JsonFormatException(json, "Fraction")

    def __repr__(self):
        return f"<Fraction values={self.numerator.name}>"

    def __eq__(self, other):
        return (
            isinstance(other, Fraction)
            and numeq(self.entries, other.entries)
            and self.quantity == other.quantity
            and self.numerator == other.numerator
            and self.denominator == other.denominator
        )

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.entries, self.quantity, self.numerator, self.denominator))


# extra properties: number of dimensions and datatypes of sub-hists
Fraction.n_dim = n_dim
Fraction.datatype = datatype

# register extra methods
Factory.register(Fraction)
