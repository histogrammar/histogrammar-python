#!/usr/bin/env python

# Copyright 2016 Jim Pivarski
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
        if not isinstance(entries, (int, long, float)):
            raise TypeError("entries ({}) must be a number".format(entries))
        if not isinstance(cuts, (list, tuple)) and not all(isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], (int, long, float)) and isinstance(v[1], Container) for v in cuts):
            raise TypeError("cuts ({}) must be a list of number, Container pairs".format(cuts))
        if not isinstance(nanflow, Container):
            raise TypeError("nanflow ({}) must be a Container".format(nanflow))
        if entries < 0.0:
            raise ValueError("entries ({}) cannot be negative".format(entries))

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
        if not isinstance(thresholds, (list, tuple)) and not all(isinstance(v, (int, long, float)) for v in thresholds):
            raise TypeError("thresholds ({}) must be a list of numbers".format(thresholds))
        if value is not None and not isinstance(value, Container):
            raise TypeError("value ({}) must be None or a Container".format(value))
        if not isinstance(nanflow, Container):
            raise TypeError("nanflow ({}) must be a Container".format(nanflow))

        self.entries = 0.0
        self.quantity = serializable(quantity)
        if value is None:
            self.cuts = tuple(thresholds)
        else:
            self.cuts = tuple((float(x), value.zero()) for x in (float("-inf"),) + tuple(thresholds))
        self.nanflow = nanflow
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
        return Partition([(x, x.zero()) for x in cuts], self.quantity, None, self.nanflow.zero())

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Partition):
            if self.thresholds != other.thresholds:
                raise ContainerException("cannot add Partition because cut thresholds differ")

            out = Partition([(k1, v1 + v2) for ((k1, v1), (k2, v2)) in zip(self.cuts, other.cuts)], self.quantity, None, self.nanflow + other.nanflow)
            out.entries = self.entries + other.entries
            return out.specialize()

        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()
        if weight > 0.0:
            q = self.quantity(datum)
            if not isinstance(q, (bool, int, long, float)):
                raise TypeError("function return value ({}) must be boolean or number".format(q))

            if math.isnan(q):
                self.nanflow.fill(datum, weight)
            else:
                for (low, sub), (high, _) in zip(self.cuts, self.cuts[1:] + ((float("nan"), None),)):
                    if q >= low and not q >= high:
                        sub.fill(datum, weight)
                        break

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
                        if elementPair["atleast"] not in ("nan", "inf", "-inf") and not isinstance(elementPair["atleast"], (int, long, float)):
                            raise JsonFormatException(json, "Partition.data {} atleast".format(i))

                        cuts.append((float(elementPair["atleast"]), factory.fromJsonFragment(elementPair["data"], dataName)))

                    else:
                        raise JsonFormatException(json, "Partition.data {}".format(i))

                out = Partition.ed(entries, cuts, nanflow)
                out.quantity.name = nameFromParent if name is None else name
                return out.specialize()

            else:
                raise JsonFormatException(json, "Partition.data")

        else:
            raise JsonFormatException(json, "Partition")

    def __repr__(self):
        return "<Partition values={} thresholds=({}) nanflow={}>".format(self.cuts[0][1].name, ", ".join(map(str, self.thresholds)), self.nanflow.name)

    def __eq__(self, other):
        return isinstance(other, Partition) and numeq(self.entries, other.entries) and self.quantity == other.quantity and all(numeq(c1, c2) and v1 == v2 for (c1, v1), (c2, v2) in zip(self.cuts, other.cuts)) and self.nanflow == other.nanflow

    def __hash__(self):
        return hash((self.entries, self.quantity, self.cuts, self.nanflow))

Factory.register(Partition)
