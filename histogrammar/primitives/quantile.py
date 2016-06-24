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

import math

from histogrammar.defs import *
from histogrammar.util import *

class Quantile(Factory, Container):
    """Estimate a quantile, such as 0.5 for median, (0.25, 0.75) for quartiles, or (0.2, 0.4, 0.6, 0.8) for quintiles, etc.

    **Note:** this is an inexact heuristic! In general, it is not possible to derive an exact quantile in a single pass over a dataset (without accumulating a large fraction of the dataset in memory). To interpret this statistic, refer to the fill and merge algorithms below.

    The quantile aggregator dynamically minimizes the mean absolute error between the current estimate and the target quantile, with a learning rate that depends on the cumulative deviations. The algorithm is deterministic: the same data always yields the same final estimate.

    This statistic has the best accuracy for quantiles near the middle of the distribution, such as the median (0.5), and the worst accuracy for quantiles near the edges, such as the first or last percentile (0.01 or 0.99). Use the specialized aggregators for the :doc:`Minimize <histogrammar.primitives.minmax.Minimize>` (0.0) or :doc:`Maximize <histogrammar.primitives.minmax.Maximize>` (1.0) of a distribution, since those aggregators are exact.

    Another alternative is to use :doc:`AdaptivelyBin <histogrammar.primitives.adaptivebin.AdaptivelyBin>` to histogram the distribution and then estimate quantiles from the histogram bins. AdaptivelyBin with ``tailDetail == 1.0`` maximizes detail on the tails of the distribution (Yael Ben-Haim and Elad Tom-Tov's original algorithm), providing the best estimates of extreme quantiles like 0.01 and 0.99.
    """

    @staticmethod
    def ed(entries, target, estimate):
        """Create a Quantile that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            target (float): the value between 0.0 and 1.0 (inclusive), indicating the quantile approximated.
            estimate (float): the best estimate of where `target` of the distribution is below this value and `1.0 - target` of the distribution is above.
        """
        if not isinstance(entries, (int, long, float)):
            raise TypeError("entries ({}) must be a number".format(entries))
        if not isinstance(target, (int, long, float)):
            raise TypeError("target ({}) must be a number".format(target))
        if not isinstance(estimate, (int, long, float)):
            raise TypeError("estimate ({}) must be a number".format(estimate))
        if entries < 0.0:
            raise ValueError("entries ({}) cannot be negative".format(entries))
        out = Quantile(target, None)
        out.entries = float(entries)
        out.estimate = float(estimate)
        return out.specialize()

    @staticmethod
    def ing(target, quantity):
        """Synonym for ``__init__``."""
        return Quantile(target, quantity)

    def __init__(self, target, quantity):
        """Create a Quantile that is capable of being filled and added.

        Parameters:
            target (float): a value between 0.0 and 1.0 (inclusive), indicating the quantile to approximate.
            quantity (function returning float): computes the quantity of interest from the data.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
            estimate (float): the best estimate of where `target` of the distribution is below this value and `1.0 - target` of the distribution is above. Initially, this value is NaN.
            cumulativeDeviation (float): the sum of absolute error between observed values and the current `estimate` (which moves). Initially, this value is 0.0.
        """
        if not isinstance(target, (int, long, float)):
            raise TypeError("target ({}) must be a number".format(target))
        if target < 0.0 or target > 1.0:
            raise ValueError("target ({}) must be between 0 and 1, inclusive".format(target))
        self.target = target
        self.quantity = serializable(quantity)
        self.entries = 0.0
        self.estimate = float("nan")
        self.cumulativeDeviation = 0.0
        super(Quantile, self).__init__()
        self.specialize()

    @inheritdoc(Container)
    def zero(self): return Quantile(self.target, self.quantity)

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Quantile):
            if self.target == other.target:
                out = Quantile(self.target, self.quantity)
                out.entries = self.entries + other.entries
                if math.isnan(self.estimate) and math.isnan(other.estimate):
                    out.estimate = float("nan")
                elif math.isnan(self.estimate):
                    out.estimate            = other.estimate
                    out.cumulativeDeviation = other.cumulativeDeviation
                elif math.isnan(other.estimate):
                    out.estimate            = self.estimate
                    out.cumulativeDeviation = self.cumulativeDeviation
                elif out.entries == 0.0:
                    out.estimate            = (self.estimate + other.estimate) / 2.0
                    out.cumulativeDeviation = (self.cumulativeDeviation + other.cumulativeDeviation)/2.0
                else:
                    out.estimate = (self.estimate*self.entries + other.estimate*other.entries) / (self.entries + other.entries)
                return out.specialize()
            else:
                raise ContainerException("cannot add Quantiles because targets do not match ({} vs {})".format(self.target, other.target))
        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()
        if weight > 0.0:
            q = self.quantity(datum)
            if not isinstance(q, (bool, int, long, float)):
                raise TypeError("function return value ({}) must be boolean or number".format(q))

            # no possibility of exception from here on out (for rollback)
            self.entries += weight
            if math.isnan(self.estimate):
                self.estimate = q
            else:
                self.cumulativeDeviation += abs(q - self.estimate)
                learningRate = 1.5 * self.cumulativeDeviation / self.entries**2
                if q < self.estimate:
                    sgn = -1
                elif q > self.estimate:
                    sgn = 1
                else:
                    sgn = 0
                self.estimate = weight * learningRate * (sgn + 2.0*self.target - 1.0)

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return []

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName): return maybeAdd({
        "entries": floatToJson(self.entries),
        "target": floatToJson(self.target),
        "estimate": floatToJson(self.estimate),
        }, name=(None if suppressName else self.quantity.name))

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "target", "estimate"], ["name"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json["entries"], "Quantile.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "AbsoluteErr.name")

            if isinstance(json["target"], (int, long, float)):
                target = float(json["target"])
            else:
                raise JsonFormatException(json["target"], "Quantile.target")

            if json["estimate"] in ("nan", "inf", "-inf") or isinstance(json["estimate"], (int, long, float)):
                estimate = float(json["estimate"])
            else:
                raise JsonFormatException(json["estimate"], "Quantile.estimate")

            out = Quantile.ed(entries, target, estimate)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        else:
            raise JsonFormatException(json, "Quantile")

    def __repr__(self):
        return "<Quantile target={} estimate={}>".format(self.target, self.estimate)

    def __eq__(self, other):
        return isinstance(other, Quantile) and self.quantity == other.quantity and numeq(self.entries, other.entries) and numeq(self.target, other.target) and numeq(self.estimate, other.estimate) and numeq(self.cumulativeDeviation, other.cumulativeDeviation)

    def __hash__(self):
        return hash((self.quantity, self.entries, self.target, self.estimate))

Factory.register(Quantile)
