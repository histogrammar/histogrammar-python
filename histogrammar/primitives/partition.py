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

import numbers

from histogrammar.defs import *
from histogrammar.util import *
from histogrammar.primitives.count import *

class Partition(Factory, Container):
    """Accumulate a suite of aggregators, each between two thresholds, filling exactly one per datum.

    This is a variation on :doc:`Stack <histogrammar.primitives.stack.Stack>`, which fills ``N + 1`` aggregators with ``N`` successively tighter cut thresholds. Partition fills ``N + 1`` aggregators in the non-overlapping intervals between ``N`` thresholds.

    Partition is also similar to :doc:`CentrallyBin <histogrammar.primitives.centralbin.CentrallyBin>`, in that they both partition a space into irregular subdomains with no gaps and no overlaps. However, CentrallyBin is defined by bin centers and Partition is defined by bin edges, the first and last of which are at negative and positive infinity.
    """

    @staticmethod
    def ed(entries, cuts, nanflow):
        """Create a Partition that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            cuts (list of float, :doc:`Container <histogrammar.defs.Container>` pairs): the ``N + 1`` thresholds and sub-aggregator pairs.
            nanflow (:doc:`Container <histogrammar.defs.Container>`): the filled nanflow bin.
        """
        if not isinstance(entries, numbers.Real) and entries not in ("nan", "inf", "-inf"):
            raise TypeError("entries ({0}) must be a number".format(entries))
        if not isinstance(cuts, (list, tuple)) and not all(isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], numbers.Real) and isinstance(v[1], Container) for v in cuts):
            raise TypeError("cuts ({0}) must be a list of number, Container pairs".format(cuts))
        if not isinstance(nanflow, Container):
            raise TypeError("nanflow ({0}) must be a Container".format(nanflow))
        if entries < 0.0:
            raise ValueError("entries ({0}) cannot be negative".format(entries))

        out = Partition(cuts, None, None, nanflow)
        out.entries = float(entries)
        return out.specialize()

    @staticmethod
    def ing(cuts, quantity, value, nanflow=Count()):
        """Synonym for ``__init__``."""
        return Partition(cuts, quantity, value, nanflow)

    def __init__(self, thresholds, quantity, value, nanflow=Count()):
        """Create a Partition that is capable of being filled and added.

        Parameters:
            thresholds (list of float) specifies ``N`` cut thresholds, so the Partition will fill ``N + 1`` aggregators in distinct intervals.
            quantity (function returning float): computes the quantity of interest from the data.
            value (:doc:`Container <histogrammar.defs.Container>`): generates sub-aggregators for each bin.
            nanflow (:doc:`Container <histogrammar.defs.Container>`): a sub-aggregator to use for data whose quantity is NaN.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
            cuts (list of float, :doc:`Container <histogrammar.defs.Container>` pairs): the ``N + 1`` thresholds and sub-aggregators. (The first threshold is minus infinity; the rest are the ones specified by ``thresholds``).
        """
        if not isinstance(thresholds, (list, tuple)) and not all(isinstance(v, numbers.Real) for v in thresholds):
            raise TypeError("thresholds ({0}) must be a list of numbers".format(thresholds))
        if value is not None and not isinstance(value, Container):
            raise TypeError("value ({0}) must be None or a Container".format(value))
        if not isinstance(nanflow, Container):
            raise TypeError("nanflow ({0}) must be a Container".format(nanflow))

        self.entries = 0.0
        self.quantity = serializable(quantity)
        if value is None:
            self.cuts = tuple(thresholds)
        else:
            self.cuts = tuple((float(x), value.zero()) for x in (float("-inf"),) + tuple(thresholds))
        self.nanflow = nanflow.copy()
        super(Partition, self).__init__()
        self.specialize()

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
        return Partition([(c, v.zero()) for c, v in self.cuts], self.quantity, None, self.nanflow.zero())

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Partition):
            if self.thresholds != other.thresholds:
                raise ContainerException("cannot add Partition because cut thresholds differ")

            out = Partition([(k1, v1 + v2) for ((k1, v1), (k2, v2)) in zip(self.cuts, other.cuts)], self.quantity, None, self.nanflow + other.nanflow)
            out.entries = self.entries + other.entries
            return out.specialize()

        else:
            raise ContainerException("cannot add {0} and {1}".format(self.name, other.name))

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()

        if weight > 0.0:
            q = self.quantity(datum)
            if not isinstance(q, numbers.Real):
                raise TypeError("function return value ({0}) must be boolean or number".format(q))

            if math.isnan(q):
                self.nanflow.fill(datum, weight)
            else:
                for (low, sub), (high, _) in zip(self.cuts, self.cuts[1:] + ((float("nan"), None),)):
                    if q >= low and not q >= high:
                        sub.fill(datum, weight)
                        break

            # no possibility of exception from here on out (for rollback)
            self.entries += weight

    def _numpy(self, data, weights, shape):
        q = self.quantity(data)
        self._checkNPQuantity(q, shape)
        self._checkNPWeights(weights, shape)
        weights = self._makeNPWeights(weights, shape)
        newentries = weights.sum()

        import numpy

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

        # FIXME: the case of all Counts could be optimized with numpy.histogram (see CentrallyBin for an example)

        selection = numpy.empty(q.shape, dtype=numpy.bool)
        selection2 = numpy.empty(q.shape, dtype=numpy.bool)
        subweights = weights.copy()
        for (low, sub), (high, _) in zip(self.cuts, self.cuts[1:] + ((float("nan"), None),)):
            numpy.greater_equal(q, low, selection)
            numpy.greater_equal(q, high, selection2)
            numpy.bitwise_not(selection2, selection2)
            numpy.bitwise_and(selection, selection2, selection)
            numpy.bitwise_not(selection, selection)

            subweights[:] = weights
            subweights[selection] = 0.0

            sub._numpy(data, subweights, shape)

        # no possibility of exception from here on out (for rollback)
        self.entries += float(newentries)

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
            if json["entries"] in ("nan", "inf", "-inf") or isinstance(json["entries"], numbers.Real):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "Partition.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Partition.name")

            if isinstance(json["type"], basestring):
                factory = Factory.registered[json["type"]]
            else:
                raise JsonFormatException(json, "Partition.type")

            if isinstance(json.get("data:name", None), basestring):
                dataName = json["data:name"]
            elif json.get("data:name", None) is None:
                dataName = None
            else:
                raise JsonFormatException(json["data:name"], "Partition.data:name")

            if isinstance(json["nanflow:type"], basestring):
                nanflowFactory = Factory.registered[json["nanflow:type"]]
            else:
                raise JsonFormatException(json, "Partition.nanflow:type")
            nanflow = nanflowFactory.fromJsonFragment(json["nanflow"], None)

            if isinstance(json["data"], list):
                cuts = []
                for i, elementPair in enumerate(json["data"]):
                    if isinstance(elementPair, dict) and hasKeys(elementPair.keys(), ["atleast", "data"]):
                        if elementPair["atleast"] not in ("nan", "inf", "-inf") and not isinstance(elementPair["atleast"], numbers.Real):
                            raise JsonFormatException(json, "Partition.data {0} atleast".format(i))

                        cuts.append((float(elementPair["atleast"]), factory.fromJsonFragment(elementPair["data"], dataName)))

                    else:
                        raise JsonFormatException(json, "Partition.data {0}".format(i))

                out = Partition.ed(entries, cuts, nanflow)
                out.quantity.name = nameFromParent if name is None else name
                return out.specialize()

            else:
                raise JsonFormatException(json, "Partition.data")

        else:
            raise JsonFormatException(json, "Partition")

    def __repr__(self):
        return "<Partition values={0} thresholds=({1}) nanflow={2}>".format(self.cuts[0][1].name, ", ".join([str(x) for x in self.thresholds]), self.nanflow.name)

    def __eq__(self, other):
        return isinstance(other, Partition) and numeq(self.entries, other.entries) and self.quantity == other.quantity and all(numeq(c1, c2) and v1 == v2 for (c1, v1), (c2, v2) in zip(self.cuts, other.cuts)) and self.nanflow == other.nanflow

    def __ne__(self, other): return not self == other

    def __hash__(self):
        return hash((self.entries, self.quantity, self.cuts, self.nanflow))

Factory.register(Partition)
