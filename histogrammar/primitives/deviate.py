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


class Deviate(Factory, Container):
    """Accumulate the weighted mean and weighted variance of a given quantity.

    The variance is computed around the mean, not zero.

    Uses the numerically stable weighted mean and weighted variance algorithms described
    in `"Incremental calculation of weighted mean and variance,"
    <http://www-uxsup.csx.cam.ac.uk/~fanf2/hermes/doc/antiforgery/stats.pdf>`_ Tony Finch,
    *Univeristy of Cambridge Computing Service,* 2009.
    """

    @staticmethod
    def ed(entries, mean, variance):
        """Create a Deviate that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            mean (float): the mean.
            variance (float): the variance.
        """
        if not isinstance(entries, numbers.Real) and entries not in (
            "nan",
            "inf",
            "-inf",
        ):
            raise TypeError(f"entries ({entries}) must be a number")
        if not isinstance(mean, numbers.Real) and entries not in ("nan", "inf", "-inf"):
            raise TypeError(f"mean ({mean}) must be a number")
        if not isinstance(variance, numbers.Real) and entries not in (
            "nan",
            "inf",
            "-inf",
        ):
            raise TypeError(f"variance ({variance}) must be a number")
        if entries < 0.0:
            raise ValueError(f"entries ({entries}) cannot be negative")
        out = Deviate(None)
        out.entries = float(entries)
        out.mean = float(mean)
        out.varianceTimesEntries = float(variance) * float(entries)
        return out.specialize()

    @staticmethod
    def ing(quantity):
        """Synonym for ``__init__``."""
        return Deviate(quantity)

    def __init__(self, quantity=identity):
        """Create a Deviate that is capable of being filled and added.

        Parameters:
            quantity (function returning float): computes the quantity of interest from the data.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
            mean (float): the running mean, initially NaN.
            variance (float): the running variance, initially NaN.
        """
        self.quantity = serializable(identity(quantity) if isinstance(quantity, str) else quantity)
        self.entries = 0.0
        self.mean = float("nan")
        self.varianceTimesEntries = float("nan")
        super().__init__()
        self.specialize()

    @property
    def variance(self):
        """Weighted variance of the quantity."""
        if self.entries == 0.0:
            return self.varianceTimesEntries
        return self.varianceTimesEntries / self.entries

    @variance.setter
    def variance(self, value):
        self.varianceTimesEntries = value * self.entries

    @inheritdoc(Container)
    def zero(self):
        return Deviate(self.quantity)

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Deviate):
            out = Deviate(self.quantity)
            out.entries = self.entries + other.entries
            if self.entries == 0.0:
                out.mean = other.mean
                out.varianceTimesEntries = other.varianceTimesEntries
            elif other.entries == 0.0:
                out.mean = self.mean
                out.varianceTimesEntries = self.varianceTimesEntries
            else:
                out.mean = (self.entries * self.mean + other.entries * other.mean) / (self.entries + other.entries)
                out.varianceTimesEntries = (
                    self.varianceTimesEntries
                    + other.varianceTimesEntries
                    + self.entries * self.mean * self.mean
                    + other.entries * other.mean * other.mean
                    - 2.0 * out.mean * (self.entries * self.mean + other.entries * other.mean)
                    + out.mean * out.mean * out.entries
                )
            return out.specialize()
        raise ContainerException(f"cannot add {self.name} and {other.name}")

    @inheritdoc(Container)
    def __iadd__(self, other):
        both = self + other
        self.entries = both.entries
        self.mean = both.mean
        self.varianceTimesEntries = both.varianceTimesEntries
        return self

    @inheritdoc(Container)
    def __mul__(self, factor):
        if math.isnan(factor) or factor <= 0.0:
            return self.zero()
        out = self.zero()
        out.entries = factor * self.entries
        out.mean = self.mean
        out.varianceTimesEntries = factor * self.varianceTimesEntries
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
            if self.entries == 0.0:
                self.mean = q
                self.varianceTimesEntries = 0.0
            self.entries += weight

            if math.isnan(self.mean) or math.isnan(q):
                self.mean = float("nan")
                self.varianceTimesEntries = float("nan")

            elif math.isinf(self.mean) or math.isinf(q):
                if math.isinf(self.mean) and math.isinf(q) and self.mean * q < 0.0:
                    self.mean = float("nan")  # opposite-sign infinities is bad
                elif math.isinf(q):
                    self.mean = q  # mean becomes infinite with sign of q
                else:
                    pass  # mean and variance are already infinite
                if math.isinf(self.entries) or math.isnan(self.entries):
                    self.mean = float("nan")  # non-finite denominator is bad

                # any infinite value makes the variance NaN
                self.varianceTimesEntries = float("nan")

            else:  # handle finite case
                delta = q - self.mean
                shift = delta * weight / self.entries
                self.mean += shift
                self.varianceTimesEntries += weight * delta * (q - self.mean)

    def _numpy(self, data, weights, shape):
        q = self.quantity(data)
        self._checkNPQuantity(q, shape)
        self._checkNPWeights(weights, shape)
        weights = self._makeNPWeights(weights, shape)

        # no possibility of exception from here on out (for rollback)
        ca, ma, sa = self.entries, self.mean, self.varianceTimesEntries
        if ca == 0.0:
            ma = 0.0
            sa = 0.0

        import numpy

        selection = weights > 0.0
        q = q[selection]
        weights = weights[selection]

        self.entries += float(weights.sum())
        ca_plus_cb = self.entries

        if math.isinf(ca_plus_cb):
            self.mean = float("nan")
            self.varianceTimesEntries = float("nan")

        elif ca_plus_cb > 0.0:
            cb = ca_plus_cb - ca
            mb = numpy.average(q, weights=weights)
            sb = cb * numpy.average((q - mb) * (q - mb), weights=weights)
            self.mean = float((ca * ma + (ca_plus_cb - ca) * mb) / ca_plus_cb)
            self.varianceTimesEntries = float(
                sa
                + sb
                + ca * ma * ma
                + cb * mb * mb
                - 2.0 * self.mean * (ca * ma + cb * mb)
                + self.mean * self.mean * ca_plus_cb
            )

    def _sparksql(self, jvm, converter):
        return converter.Deviate(self.quantity.asSparkSQL())

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return []

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName):
        return maybeAdd(
            {
                "entries": floatToJson(self.entries),
                "mean": floatToJson(self.mean),
                "variance": floatToJson(self.variance),
            },
            name=(None if suppressName else self.quantity.name),
        )

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "mean", "variance"], ["name"]):
            if json["entries"] in ("nan", "inf", "-inf") or isinstance(json["entries"], numbers.Real):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json["entries"], "Deviate.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Deviate.name")

            if json["mean"] in ("nan", "inf", "-inf") or isinstance(json["mean"], numbers.Real):
                mean = float(json["mean"])
            else:
                raise JsonFormatException(json["mean"], "Deviate.mean")

            if json["variance"] in ("nan", "inf", "-inf") or isinstance(json["variance"], numbers.Real):
                variance = float(json["variance"])
            else:
                raise JsonFormatException(json["variance"], "Deviate.variance")

            out = Deviate.ed(entries, mean, variance)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        raise JsonFormatException(json, "Deviate")

    def __repr__(self):
        return f"<Deviate mean={self.mean} variance={self.variance}>"

    def __eq__(self, other):
        return (
            isinstance(other, Deviate)
            and self.quantity == other.quantity
            and numeq(self.entries, other.entries)
            and numeq(self.mean, other.mean)
            and numeq(self.variance, other.variance)
        )

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.quantity, self.entries, self.mean, self.variance))


# extra properties: number of dimensions and datatypes of sub-hists
Deviate.n_dim = n_dim
Deviate.datatype = datatype

# register extra methods
Factory.register(Deviate)
