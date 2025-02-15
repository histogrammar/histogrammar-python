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
    xrange,
)


class Stack(Factory, Container):
    """Accumulates a suite of aggregators, each filtered with a tighter selection on the same quantity.

    This is a generalization of :doc:`Fraction <histogrammar.primitives.fraction.Fraction>`, which fills two
    aggregators, one with a cut, the other without. Stack fills ``N + 1`` aggregators with ``N`` successively tighter
    cut thresholds. The first is always filled (like the denominator of Fraction), the second is filled if the
    computed quantity exceeds its threshold, the next is filled if the computed quantity exceeds a higher threshold,
    and so on.

    The thresholds are presented in increasing order and the computed value must be greater than or equal to a
    threshold to fill the corresponding bin, and therefore the number of entries in each filled bin is greatest in the
    first and least in the last.

    Although this aggregation could be visualized as a stack of histograms, stacked histograms usually represent a
    different thing: data from different sources, rather than different cuts on the same source. For example, it is
    common to stack Monte Carlo samples from different backgrounds to show that they add up to the observed data.
    The Stack aggregator does not make plots of this type because aggregation trees in Histogrammar draw data from
    exactly one source.

    To make plots from different sources in Histogrammar, one must perform separate aggregation runs. It may then be
    convenient to stack the results of those runs as though they were created with a Stack aggregation, so that
    plotting code can treat both cases uniformly. For this reason, Stack has an alternate constructor to build a
    Stack manually from distinct aggregators, even if those aggregators came from different aggregation runs.
    """

    @staticmethod
    def ed(entries, bins, nanflow):
        """Create a Stack that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            bins (list of float, :doc:`Container <histogrammar.defs.Container>` pairs): the ``N + 1`` thresholds and
                sub-aggregator pairs.
            nanflow (:doc:`Container <histogrammar.defs.Container>`): the filled nanflow bin.
        """
        if not isinstance(entries, numbers.Real) and entries not in (
            "nan",
            "inf",
            "-inf",
        ):
            raise TypeError(f"entries ({entries}) must be a number")
        if not isinstance(bins, (list, tuple)) and not all(
            isinstance(v, (list, tuple))
            and len(v) == 2
            and isinstance(v[0], numbers.Real)
            and isinstance(v[1], Container)
            for v in bins
        ):
            raise TypeError(f"bins ({bins}) must be a list of number, Container pairs")
        if not isinstance(nanflow, Container):
            raise TypeError(f"nanflow ({nanflow}) must be a Container")
        if entries < 0.0:
            raise ValueError(f"entries ({entries}) cannot be negative")
        out = Stack(bins, None, None, nanflow)
        out.entries = float(entries)
        return out.specialize()

    @staticmethod
    def ing(thresholds, quantity, value=Count(), nanflow=Count()):
        """Synonym for ``__init__``."""
        return Stack(thresholds, quantity, value, nanflow)

    def __init__(self, thresholds, quantity=identity, value=Count(), nanflow=Count()):
        """Create a Stack that is capable of being filled and added.

        Parameters:
            thresholds (list of floats): specifies ``N`` cut thresholds, so the Stack will fill ``N + 1`` aggregators,
                each overlapping the last.
            quantity (function returning float): computes the quantity of interest from the data.
            value (:doc:`Container <histogrammar.defs.Container>`): generates sub-aggregators for each bin.
            nanflow (:doc:`Container <histogrammar.defs.Container>`): a sub-aggregator to use for data whose quantity
                is NaN.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
            bins (list of float, :doc:`Container <histogrammar.defs.Container>` pairs): the ``N + 1`` thresholds and
                sub-aggregators. (The first threshold is minus infinity; the rest are the ones specified
                by ``thresholds``).
        """
        if not isinstance(thresholds, (list, tuple)) and not all(
            isinstance(v, (list, tuple))
            and len(v) == 2
            and isinstance(v[0], numbers.Real)
            and isinstance(v[1], Container)
            for v in thresholds
        ):
            raise TypeError(f"thresholds ({thresholds}) must be a list of number, Container pairs")
        if value is not None and not isinstance(value, Container):
            raise TypeError(f"value ({value}) must be None or a Container")
        if not isinstance(nanflow, Container):
            raise TypeError(f"nanflow ({nanflow}) must be a Container")
        self.entries = 0.0
        self.quantity = serializable(identity(quantity) if isinstance(quantity, str) else quantity)
        if value is None:
            self.bins = tuple(thresholds)
        else:
            self.bins = tuple((float(x), value.zero()) for x in (float("-inf"),) + tuple(thresholds))
        self.nanflow = nanflow.copy()
        super().__init__()
        self.specialize()

    @staticmethod
    def build(*ys):
        """Create a Stack out of pre-existing containers, which might have been aggregated on different streams.

        Parameters:
            aggregators (list of :doc:`Container <histogrammar.defs.Container>`): this function will attempt to add
                them, so they must also have the same binning/bounds/etc.
        """
        from functools import reduce

        if not all(isinstance(y, Container) for y in ys):
            raise TypeError("ys must all be Containers")
        entries = sum(y.entries for y in ys)
        bins = []
        for i in xrange(len(ys)):
            bins.append((float("nan"), reduce(lambda a, b: a + b, ys[i:])))
        return Stack.ed(entries, bins, Count.ed(0.0))

    @property
    def thresholds(self):
        """Cut thresholds (first items of ``bins``)."""
        return [k for k, v in self.bins]

    @property
    def values(self):
        """Sub-aggregators (second items of ``bins``)."""
        return [v for k, v in self.bins]

    @inheritdoc(Container)
    def zero(self):
        return Stack(
            [(c, v.zero()) for c, v in self.bins],
            self.quantity,
            None,
            self.nanflow.zero(),
        )

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Stack):
            if self.thresholds != other.thresholds:
                raise ContainerException("cannot add Stack because cut thresholds differ")

            out = Stack(
                [(k1, v1 + v2) for ((k1, v1), (k2, v2)) in zip(self.bins, other.bins)],
                self.quantity,
                None,
                self.nanflow + other.nanflow,
            )
            out.entries = self.entries + other.entries
            return out.specialize()

        raise ContainerException(f"cannot add {self.name} and {other.name}")

    @inheritdoc(Container)
    def __iadd__(self, other):
        if isinstance(other, Stack):
            if self.thresholds != other.thresholds:
                raise ContainerException("cannot add Stack because cut thresholds differ")
            self.entries += other.entries
            for (k1, v1), (k2, v2) in zip(self.bins, other.bins):
                v1 += v2  # noqa: PLW2901
            self.nanflow += other.nanflow
            return self
        raise ContainerException(f"cannot add {self.name} and {other.name}")

    @inheritdoc(Container)
    def __mul__(self, factor):
        if math.isnan(factor) or factor <= 0.0:
            return self.zero()
        out = self.zero()
        out.entries = factor * self.entries
        out.bins = [(c, v * factor) for (c, v) in self.bins]
        out.nanflow = self.nanflow * factor
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

            if math.isnan(q):
                self.nanflow.fill(datum, weight)
            else:
                for threshold, sub in self.bins:
                    if q >= threshold:
                        sub.fill(datum, weight)

            # no possibility of exception from here on out (for rollback)
            self.entries += weight

    def _numpy(self, data, weights, shape):
        q = self.quantity(data)
        self._checkNPQuantity(q, shape)
        self._checkNPWeights(weights, shape)
        weights = self._makeNPWeights(weights, shape)
        newentries = weights.sum()

        selection = numpy.isnan(q)
        numpy.bitwise_not(selection, selection)
        subweights = weights.copy()
        subweights[selection] = 0.0
        self.nanflow._numpy(data, subweights, shape)

        # avoid nan warning in calculations by flinging the nans elsewhere
        numpy.bitwise_not(selection, selection)
        q = numpy.array(q, dtype=numpy.float64)
        q[selection] = float("-inf")
        weights = weights.copy()
        weights[selection] = 0.0

        selection = numpy.empty(q.shape, dtype=bool)
        for threshold, sub in self.bins:
            numpy.less(q, threshold, selection)
            subweights[:] = weights
            subweights[selection] = 0.0

            sub._numpy(data, subweights, shape)

        # no possibility of exception from here on out (for rollback)
        self.entries += float(newentries)

    def _sparksql(self, jvm, converter):
        return converter.Stack(
            [e for e, v in self.bins[1:]],
            self.quantity.asSparkSQL(),
            self.bins[0][1]._sparksql(jvm, converter),
            self.nanflow._sparksql(jvm, converter),
        )

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return [self.nanflow] + self.values

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName):
        if getattr(self.bins[0][1], "quantity", None) is not None:
            binsName = self.bins[0][1].quantity.name
        elif getattr(self.bins[0][1], "quantityName", None) is not None:
            binsName = self.bins[0][1].quantityName
        else:
            binsName = None

        return maybeAdd(
            {
                "entries": floatToJson(self.entries),
                "bins:type": self.bins[0][1].name,
                "bins": [
                    {"atleast": floatToJson(atleast), "data": sub.toJsonFragment(True)} for atleast, sub in self.bins
                ],
                "nanflow:type": self.nanflow.name,
                "nanflow": self.nanflow.toJsonFragment(False),
            },
            **{
                "name": None if suppressName else self.quantity.name,
                "bins:name": binsName,
            },
        )

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(
            json.keys(),
            ["entries", "bins:type", "bins", "nanflow:type", "nanflow"],
            ["name", "bins:name"],
        ):
            if json["entries"] in ("nan", "inf", "-inf") or isinstance(json["entries"], numbers.Real):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "Stack.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Stack.name")

            if isinstance(json["bins:type"], basestring):
                factory = Factory.registered[json["bins:type"]]
            else:
                raise JsonFormatException(json, "Stack.bins:type")

            if isinstance(json.get("bins:name", None), basestring):
                dataName = json["bins:name"]
            elif json.get("bins:name", None) is None:
                dataName = None
            else:
                raise JsonFormatException(json["bins:name"], "Stack.bins:name")

            if isinstance(json["nanflow:type"], basestring):
                nanflowFactory = Factory.registered[json["nanflow:type"]]
            else:
                raise JsonFormatException(json, "Stack.nanflow:type")
            nanflow = nanflowFactory.fromJsonFragment(json["nanflow"], None)

            if isinstance(json["bins"], list):
                bins = []
                for i, elementPair in enumerate(json["bins"]):
                    if isinstance(elementPair, dict) and hasKeys(elementPair.keys(), ["atleast", "data"]):
                        if elementPair["atleast"] not in (
                            "nan",
                            "inf",
                            "-inf",
                        ) and not isinstance(elementPair["atleast"], numbers.Real):
                            raise JsonFormatException(json, f"Stack.bins {i} atleast")

                        bins.append(
                            (
                                float(elementPair["atleast"]),
                                factory.fromJsonFragment(elementPair["data"], dataName),
                            )
                        )

                    else:
                        raise JsonFormatException(json, f"Stack.bins {i}")

                out = Stack.ed(entries, bins, nanflow)
                out.quantity.name = nameFromParent if name is None else name
                return out.specialize()

            raise JsonFormatException(json, "Stack.bins")

        raise JsonFormatException(json, "Stack")

    def __repr__(self):
        return "<Stack values={} thresholds=({}) nanflow={}>".format(
            self.bins[0][1].name,
            ", ".join([str(x) for x in self.thresholds]),
            self.nanflow.name,
        )

    def __eq__(self, other):
        return (
            isinstance(other, Stack)
            and numeq(self.entries, other.entries)
            and self.quantity == other.quantity
            and all(numeq(c1, c2) and v1 == v2 for (c1, v1), (c2, v2) in zip(self.bins, other.bins))
            and self.nanflow == other.nanflow
        )

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.entries, self.quantity, self.bins, self.nanflow))

    def bin_entries(self):
        """Returns bin values

        :returns: numpy array with numbers of entries for all threshold bins
        :rtype: numpy.array
        """
        return numpy.array([v.entries for v in self.values])


# extra properties: number of dimensions and datatypes of sub-hists
Stack.n_dim = n_dim
Stack.datatype = datatype

# register extra methods
Factory.register(Stack)
