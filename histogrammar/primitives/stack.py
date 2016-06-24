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

import types

from histogrammar.defs import *
from histogrammar.util import *
from histogrammar.primitives.count import *

class Stack(Factory, Container):
    """Accumulates a suite of aggregators, each filtered with a tighter selection on the same quantity.

    This is a generalization of :doc:`Fraction <histogrammar.primitives.fraction.Fraction>`, which fills two aggregators, one with a cut, the other without. Stack fills ``N + 1`` aggregators with ``N`` successively tighter cut thresholds. The first is always filled (like the denominator of Fraction), the second is filled if the computed quantity exceeds its threshold, the next is filled if the computed quantity exceeds a higher threshold, and so on.

    The thresholds are presented in increasing order and the computed value must be greater than or equal to a threshold to fill the corresponding bin, and therefore the number of entries in each filled bin is greatest in the first and least in the last.

    Although this aggregation could be visualized as a stack of histograms, stacked histograms usually represent a different thing: data from different sources, rather than different cuts on the same source. For example, it is common to stack Monte Carlo samples from different backgrounds to show that they add up to the observed data. The Stack aggregator does not make plots of this type because aggregation trees in Histogrammar draw data from exactly one source.

    To make plots from different sources in Histogrammar, one must perform separate aggregation runs. It may then be convenient to stack the results of those runs as though they were created with a Stack aggregation, so that plotting code can treat both cases uniformly. For this reason, Stack has an alternate constructor to build a Stack manually from distinct aggregators, even if those aggregators came from different aggregation runs.
    """

    @staticmethod
    def ed(entries, cuts, nanflow):
        """Create a Stack that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            cuts (list of float, :doc:`Container <histogrammar.defs.Container>` pairs): the ``N + 1`` thresholds and sub-aggregator pairs.
            nanflow (:doc:`Container <histogrammar.defs.Container>`): the filled nanflow bin.
        """
        if not isinstance(entries, (int, long, float)):
            raise TypeError("entries ({0}) must be a number".format(entries))
        if not isinstance(cuts, (list, tuple)) and not all(isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], (int, long, float)) and isinstance(v[1], Container) for v in cuts):
            raise TypeError("cuts ({0}) must be a list of number, Container pairs".format(cuts))
        if not isinstance(nanflow, Container):
            raise TypeError("nanflow ({0}) must be a Container".format(nanflow))
        if entries < 0.0:
            raise ValueError("entries ({0}) cannot be negative".format(entries))
        out = Stack(cuts, None, None, nanflow)
        out.entries = float(entries)
        return out.specialize()

    @staticmethod
    def ing(cuts, quantity, value, nanflow=Count()):
        """Synonym for ``__init__``."""
        return Stack(cuts, quantity, value, nanflow)

    def __init__(self, cuts, quantity, value, nanflow=Count()):
        """Create a Stack that is capable of being filled and added.

        Parameters:
            thresholds (list of floats): specifies ``N`` cut thresholds, so the Stack will fill ``N + 1`` aggregators, each overlapping the last.
            quantity (function returning float): computes the quantity of interest from the data.
            value (:doc:`Container <histogrammar.defs.Container>`): generates sub-aggregators for each bin.
            nanflow (:doc:`Container <histogrammar.defs.Container>`): a sub-aggregator to use for data whose quantity is NaN.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
            cuts (list of float, :doc:`Container <histogrammar.defs.Container>` pairs): the ``N + 1`` thresholds and sub-aggregators. (The first threshold is minus infinity; the rest are the ones specified by ``thresholds``).
        """
        if not isinstance(cuts, (list, tuple)) and not all(isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], (int, long, float)) and isinstance(v[1], Container) for v in cuts):
            raise TypeError("cuts ({0}) must be a list of number, Container pairs".format(cuts))
        if value is not None and not isinstance(value, Container):
            raise TypeError("value ({0}) must be None or a Container".format(value))
        if not isinstance(nanflow, Container):
            raise TypeError("nanflow ({0}) must be a Container".format(nanflow))
        self.entries = 0.0
        self.quantity = serializable(quantity)
        if value is None:
            self.cuts = tuple(cuts)
        else:
            self.cuts = tuple((float(x), value.zero()) for x in (float("-inf"),) + tuple(cuts))
        self.nanflow = nanflow
        super(Stack, self).__init__()
        self.specialize()

    @staticmethod
    def build(*ys):
        """Create a Stack out of pre-existing containers, which might have been aggregated on different streams.

        Parameters:
            aggregators (list of :doc:`Container <histogrammar.defs.Container>`): this function will attempt to add them, so they must also have the same binning/bounds/etc.
        """
        from functools import reduce
        if not all(isinstance(y, Container) for y in ys):
            raise TypeError("ys must all be Containers")
        entries = sum(y.entries for y in ys)
        cuts = []
        for i in xrange(len(ys)):
            cuts.append((float("nan"), reduce(lambda a, b: a + b, ys[i:])))
        return Stack.ed(entries, cuts, Count.ed(0.0))

    @property
    def thresholds(self):
        """Cut thresholds (first items of ``cuts``)."""
        return [k for k, v in self.cuts]

    @property
    def values(self):
        """Sub-aggregators (second items of ``cuts``)."""
        return [v for k, v in self.cuts]

    @inheritdoc(Container)
    def zero(self):
        return Stack([(x, x.zero()) for x in cuts], self.quantity, None, self.nanflow.zero())

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Stack):
            if self.thresholds != other.thresholds:
                raise ContainerException("cannot add Stack because cut thresholds differ")

            out = Stack([(k1, v1 + v2) for ((k1, v1), (k2, v2)) in zip(self.cuts, other.cuts)], self.quantity, None, self.nanflow + other.nanflow)
            out.entries = self.entries + other.entries
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

            if math.isnan(q):
                self.nanflow.fill(datum, weight)
            else:
                for threshold, sub in self.cuts:
                    if q >= threshold:
                        sub.fill(datum, weight)

            # no possibility of exception from here on out (for rollback)
            self.entries += weight

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return [self.nanflow] + self.values

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName):
        if getattr(self.cuts[0][1], "quantity", None) is not None:
            binsName = self.cuts[0][1].quantity.name
        elif getattr(self.cuts[0][1], "quantityName", None) is not None:
            binsName = self.cuts[0][1].quantityName
        else:
            binsName = None

        return maybeAdd({
            "entries": floatToJson(self.entries),
            "type": self.cuts[0][1].name,
            "data": [{"atleast": floatToJson(atleast), "data": sub.toJsonFragment(True)} for atleast, sub in self.cuts],
            "nanflow:type": self.nanflow.name,
            "nanflow": self.nanflow.toJsonFragment(False),
            }, **{"name": None if suppressName else self.quantity.name,
                  "data:name": binsName})

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "type", "data", "nanflow:type", "nanflow"], ["name", "data:name"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "Stack.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Stack.name")

            if isinstance(json["type"], basestring):
                factory = Factory.registered[json["type"]]
            else:
                raise JsonFormatException(json, "Stack.type")

            if isinstance(json.get("data:name", None), basestring):
                dataName = json["data:name"]
            elif json.get("data:name", None) is None:
                dataName = None
            else:
                raise JsonFormatException(json["data:name"], "Stack.data:name")

            if isinstance(json["nanflow:type"], basestring):
                nanflowFactory = Factory.registered[json["nanflow:type"]]
            else:
                raise JsonFormatException(json, "Stack.nanflow:type")
            nanflow = nanflowFactory.fromJsonFragment(json["nanflow"], None)

            if isinstance(json["data"], list):
                cuts = []
                for i, elementPair in enumerate(json["data"]):
                    if isinstance(elementPair, dict) and hasKeys(elementPair.keys(), ["atleast", "data"]):
                        if elementPair["atleast"] not in ("nan", "inf", "-inf") and not isinstance(elementPair["atleast"], (int, long, float)):
                            raise JsonFormatException(json, "Stack.data {0} atleast".format(i))

                        cuts.append((float(elementPair["atleast"]), factory.fromJsonFragment(elementPair["data"], dataName)))

                    else:
                        raise JsonFormatException(json, "Stack.data {0}".format(i))

                out = Stack.ed(entries, cuts, nanflow)
                out.quantity.name = nameFromParent if name is None else name
                return out.specialize()

            else:
                raise JsonFormatException(json, "Stack.data")

        else:
            raise JsonFormatException(json, "Stack")

    def __repr__(self):
        return "<Stack values={0} thresholds=({1}) nanflow={2}>".format(self.cuts[0][1].name, ", ".join(map(str, self.thresholds)), self.nanflow.name)

    def __eq__(self, other):
        return isinstance(other, Stack) and numeq(self.entries, other.entries) and self.quantity == other.quantity and all(numeq(c1, c2) and v1 == v2 for (c1, v1), (c2, v2) in zip(self.cuts, other.cuts)) and self.nanflow == other.nanflow

    def __hash__(self):
        return hash((self.entries, self.quantity, self.cuts, self.nanflow))

Factory.register(Stack)
