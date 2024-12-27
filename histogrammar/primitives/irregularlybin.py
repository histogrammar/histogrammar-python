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

import bisect
import math
import numbers

import numpy as np

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


class IrregularlyBin(Factory, Container):
    """Accumulate a suite of aggregators, each between two thresholds, filling exactly one per datum.

    This is a variation on :doc:`Stack <histogrammar.primitives.stack.Stack>`, which fills ``N + 1`` aggregators
    with ``N`` successively tighter cut thresholds. IrregularlyBin fills ``N + 1`` aggregators in the non-overlapping
    intervals between ``N`` thresholds.

    IrregularlyBin is also similar to :doc:`CentrallyBin <histogrammar.primitives.centrallybin.CentrallyBin>`, in that
    they both partition a space into irregular subdomains with no gaps and no overlaps. However, CentrallyBin is
    defined by bin centers and IrregularlyBin is defined by bin edges, the first and last of which are at negative
    and positive infinity.
    """

    @staticmethod
    def ed(entries, bins, nanflow):
        """Create a IrregularlyBin that is only capable of being added.

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

        out = IrregularlyBin(bins, None, None, nanflow)
        out.entries = float(entries)
        return out.specialize()

    @staticmethod
    def ing(edges, quantity, value=Count(), nanflow=Count()):
        """Synonym for ``__init__``."""
        return IrregularlyBin(edges, quantity, value, nanflow)

    def __init__(self, edges, quantity=identity, value=Count(), nanflow=Count()):
        """Create a IrregularlyBin that is capable of being filled and added.

        Parameters:
            edges (list of float) specifies ``N`` cut thresholds, so the IrregularlyBin will fill ``N + 1`` aggregators
                in distinct intervals.
            quantity (function returning float): computes the quantity of interest from the data.
            value (:doc:`Container <histogrammar.defs.Container>`): generates sub-aggregators for each bin.
            nanflow (:doc:`Container <histogrammar.defs.Container>`): a sub-aggregator to use for data whose quantity
                is NaN.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
            bins (list of float, :doc:`Container <histogrammar.defs.Container>` pairs): the ``N + 1`` thresholds and
                sub-aggregators. (The first threshold is minus infinity; the rest are the ones specified by ``edges``).
        """
        if not isinstance(edges, (list, tuple)) and not all(isinstance(v, numbers.Real) for v in edges):
            raise TypeError(f"edges ({edges}) must be a list of numbers")
        if value is not None and not isinstance(value, Container):
            raise TypeError(f"value ({value}) must be None or a Container")
        if not isinstance(nanflow, Container):
            raise TypeError(f"nanflow ({nanflow}) must be a Container")

        self.entries = 0.0
        self.quantity = serializable(identity(quantity) if isinstance(quantity, str) else quantity)
        if value is None:
            self.bins = tuple(edges)
        else:
            self.bins = tuple((float(x), value.zero()) for x in (float("-inf"),) + tuple(edges))
        self.nanflow = nanflow.copy()
        super().__init__()
        self.specialize()

    @property
    def thresholds(self):
        """Cut thresholds (first items of ``bins``)."""
        return [k for k, v in self.bins]

    @property
    def edges(self):
        """Cut thresholds (first items of ``bins``)."""
        return [k for k, v in self.bins]

    @property
    def values(self):
        """Sub-aggregators (second items of ``bins``)."""
        return [v for k, v in self.bins]

    @property
    def n_bins(self):
        """Get number of bins, consistent with SparselyBin and Categorize"""
        return len(self.bins)

    def _lower_index(self, x):
        """Find lower index of bin corresponding to ``x``."""
        edges = self.edges
        return max(0, bisect.bisect(edges, x) - 1)

    def _upper_index(self, x):
        """Find upper index of bin corresponding to ``x``."""
        edges = self.edges
        if x in edges:
            return max(0, edges.index(x) - 1)
        return max(0, bisect.bisect(edges, x) - 1)

    def num_bins(self, low=None, high=None):
        """Returns number of bins of a given (sub-)range

        Possible to set range with low and high params

        :param low: lower edge of range, default is None
        :param high: higher edge of range, default is None
        :returns: number of bins in range
        :rtype: int
        """
        # trivial cases first
        if low is None and high is None:
            return len(self.bins)
        # catch weird cases
        if low is not None and high is not None and low > high:
            raise RuntimeError(f"low {low} greater than high {high}")
        # lowest, highest edge reset
        if low is None:
            low = -np.inf
        if high is None:
            high = np.inf
        # bin indices
        lidx = self._lower_index(low)
        hidx = self._upper_index(high)
        return hidx - lidx + 1

    def bin_entries(self, low=None, high=None, xvalues=[]):
        """Returns bin values

        Possible to set range with low and high params, and list of selected x-values

        :param low: lower edge of range, default is None
        :param high: higher edge of range, default is None
        :param xvalues: list of x-values to get entries of, alternative to low and high
        :returns: numpy array with numbers of entries for selected bins
        :rtype: numpy.array
        """
        # trivial case
        if low is None and high is None and len(xvalues) == 0:
            return np.array([b[1].entries for b in self.bins])
        # catch weird cases
        if low is not None and high is not None and len(xvalues) == 0:
            if low > high:
                raise RuntimeError(f"low {low} greater than high {high}")
        # entries at request list of x-values
        elif len(xvalues) > 0:
            return np.array([(self.bins[self.index(x)])[1].entries for x in xvalues])
        # lowest, highest edge reset
        if low is None:
            low = -np.inf
        if high is None:
            high = np.inf
        # return bin entries
        lidx = self._lower_index(low)
        hidx = self._upper_index(high)
        return np.array([(self.bins[i])[1].entries for i in range(lidx, hidx + 1)])

    def bin_edges(self, low=None, high=None):
        """Returns bin edges

        :param low: lower edge of range, default is None
        :param high: higher edge of range, default is None
        :returns: numpy array with bin edges for selected range
        :rtype: numpy.array
        """
        # catch weird cases
        if low is not None and high is not None and low > high:
            raise RuntimeError(f"low {low} greater than high {high}")
        # lowest, highest edge reset
        if low is None:
            low = -np.inf
        if high is None:
            high = np.inf
        # return bin edges
        all_edges = np.concatenate([self.edges, [np.inf]])
        lidx = self._lower_index(low)
        hidx = self._upper_index(high)
        return all_edges[lidx : hidx + 2]

    def bin_centers(self, low=None, high=None):
        """Returns bin centers

        :param low: lower edge of range, default is None
        :param high: higher edge of range, default is None
        :returns: numpy array with bin centers for selected range
        :rtype: numpy.array
        """
        bin_edges = self.bin_edges(low, high)
        return (bin_edges[:-1] + bin_edges[1:]) / 2

    def bin_width(self):
        """Returns bin widths"""
        edges = self.edges[1:]  # cut out -inf
        return np.diff(edges)

    def _center_from_key(self, edge):
        idx = self.edges.index(edge)
        return self.bin_centers()[idx]

    @property
    def mpv(self):
        """Return bin-center of most probable value"""
        bin_entries = self.bin_entries()
        bin_centers = self.bin_centers()
        # if two max elements are equal, this will return the element with the lowest index.
        max_idx = max(enumerate(bin_entries), key=lambda x: x[1])[0]
        return bin_centers[max_idx]

    @inheritdoc(Container)
    def zero(self):
        return IrregularlyBin(
            [(c, v.zero()) for c, v in self.bins],
            self.quantity,
            None,
            self.nanflow.zero(),
        )

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, IrregularlyBin):
            if self.thresholds != other.thresholds:
                raise ContainerException("cannot add IrregularlyBin because cut thresholds differ")

            out = IrregularlyBin(
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
        if isinstance(other, IrregularlyBin):
            if self.thresholds != other.thresholds:
                raise ContainerException("cannot add IrregularlyBin because cut thresholds differ")
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
                for (low, sub), (high, _) in zip(self.bins, self.bins[1:] + ((float("nan"), None),)):
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

        selection = np.isnan(q)
        np.bitwise_not(selection, selection)
        subweights = weights.copy()
        subweights[selection] = 0.0
        self.nanflow._numpy(data, subweights, shape)

        # avoid nan warning in calculations by flinging the nans elsewhere
        np.bitwise_not(selection, selection)
        q = np.array(q, dtype=np.float64)
        q[selection] = float("-inf")
        weights = weights.copy()
        weights[selection] = 0.0

        # FIXME: the case of all Counts could be optimized with np.histogram (see CentrallyBin for an example)

        selection = np.empty(q.shape, dtype=bool)
        selection2 = np.empty(q.shape, dtype=bool)
        subweights = weights.copy()
        for (low, sub), (high, _) in zip(self.bins, self.bins[1:] + ((float("nan"), None),)):
            np.greater_equal(q, low, selection)
            np.greater_equal(q, high, selection2)
            np.bitwise_not(selection2, selection2)
            np.bitwise_and(selection, selection2, selection)
            np.bitwise_not(selection, selection)

            subweights[:] = weights
            subweights[selection] = 0.0

            sub._numpy(data, subweights, shape)

        # no possibility of exception from here on out (for rollback)
        self.entries += float(newentries)

    def _sparksql(self, jvm, converter):
        return converter.IrregularlyBin(
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
                raise JsonFormatException(json, "IrregularlyBin.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "IrregularlyBin.name")

            if isinstance(json["bins:type"], basestring):
                factory = Factory.registered[json["bins:type"]]
            else:
                raise JsonFormatException(json, "IrregularlyBin.bins:type")

            if isinstance(json.get("bins:name", None), basestring):
                dataName = json["bins:name"]
            elif json.get("bins:name", None) is None:
                dataName = None
            else:
                raise JsonFormatException(json["bins:name"], "IrregularlyBin.bins:name")

            if isinstance(json["nanflow:type"], basestring):
                nanflowFactory = Factory.registered[json["nanflow:type"]]
            else:
                raise JsonFormatException(json, "IrregularlyBin.nanflow:type")
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
                            raise JsonFormatException(json, f"IrregularlyBin.bins {i} atleast")

                        bins.append(
                            (
                                float(elementPair["atleast"]),
                                factory.fromJsonFragment(elementPair["data"], dataName),
                            )
                        )

                    else:
                        raise JsonFormatException(json, f"IrregularlyBin.bins {i}")

                out = IrregularlyBin.ed(entries, bins, nanflow)
                out.quantity.name = nameFromParent if name is None else name
                return out.specialize()

            raise JsonFormatException(json, "IrregularlyBin.bins")

        raise JsonFormatException(json, "IrregularlyBin")

    def __repr__(self):
        return "<IrregularlyBin values={} thresholds=({}) nanflow={}>".format(
            self.bins[0][1].name,
            ", ".join([str(x) for x in self.thresholds]),
            self.nanflow.name,
        )

    def __eq__(self, other):
        return (
            isinstance(other, IrregularlyBin)
            and numeq(self.entries, other.entries)
            and self.quantity == other.quantity
            and all(numeq(c1, c2) and v1 == v2 for (c1, v1), (c2, v2) in zip(self.bins, other.bins))
            and self.nanflow == other.nanflow
        )

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.entries, self.quantity, self.bins, self.nanflow))


# extra properties: number of dimensions and datatypes of sub-hists
IrregularlyBin.n_dim = n_dim
IrregularlyBin.datatype = datatype

# register extra methods
Factory.register(IrregularlyBin)
