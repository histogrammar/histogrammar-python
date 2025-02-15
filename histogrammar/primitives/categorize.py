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


class Categorize(Factory, Container):
    """Split a given quantity by its categorical value and fill only one category per datum.

    A bar chart may be thought of as a histogram with string-valued (categorical) bins, so this is the equivalent
    of :doc:`Bin <histogrammar.primitives.bin.Bin>` for bar charts. The order of the strings is deferred to the
    visualization stage.

    Unlike :doc:`SparselyBin <histogrammar.primitives.sparselybin.SparselyBin>`, this aggregator has the potential to
    use unlimited memory. A large number of *distinct* categories can generate many unwanted bins.
    """

    @staticmethod
    def ed(entries, contentType, binsAsDict=None, **bins):
        """Create a Categorize that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            contentType (str): the value's sub-aggregator type (must be provided to determine type for the case when
            `bins` is empty). bins (dict from str to :doc:`Container <histogrammar.defs.Container>`): the non-empty
            bin categories and their values.
        """
        if not isinstance(entries, numbers.Real) and entries not in (
            "nan",
            "inf",
            "-inf",
        ):
            raise TypeError(f"entries ({entries}) must be a number")
        if not isinstance(contentType, basestring):
            raise TypeError(f"contentType ({contentType}) must be a string")
        if not all(isinstance(k, basestring) and isinstance(v, Container) for k, v in bins.items()):
            raise TypeError(f"bins ({bins}) must be a dict from strings to Containers")
        if entries < 0.0:
            raise ValueError(f"entries ({entries}) cannot be negative")

        out = Categorize(None, None)
        out.entries = float(entries)
        if binsAsDict is None:
            out.bins = {}
        else:
            out.bins = binsAsDict
        out.bins.update(bins)
        out.contentType = contentType
        return out.specialize()

    @staticmethod
    def ing(quantity, value=Count()):
        """Synonym for ``__init__``."""
        return Categorize(quantity, value)

    def __init__(self, quantity=identity, value=Count()):
        """Create a Categorize that is capable of being filled and added.

        Parameters:
            quantity (function returning float): computes the quantity of interest from the data.
            value (:doc:`Container <histogrammar.defs.Container>`): generates sub-aggregators to put in each bin.

        Other Parameters:
            entries (float): the number of entries, initially 0.0.
            bins (dict from str to :doc:`Container <histogrammar.defs.Container>`): the map, probably a hashmap, to
            fill with values when their `entries` become non-zero.
        """
        if value is not None and not isinstance(value, Container):
            raise TypeError(f"value ({value}) must be None or a Container")
        self.entries = 0.0
        self.quantity = serializable(identity(quantity) if isinstance(quantity, str) else quantity)
        self.value = value
        self.bins = {}
        if value is not None:
            self.contentType = value.name
        else:
            self.contentType = "Count"
        super().__init__()
        self.specialize()

    @property
    def binsMap(self):
        """Input ``bins`` as a key-value map."""
        return self.bins

    @property
    def size(self):
        """Number of ``bins``."""
        return len(self.bins)

    @property
    def keys(self):
        """Iterable over the keys of the ``bins``."""
        return self.bins.keys()

    @property
    def values(self):
        """Iterable over the values of the ``bins``."""
        return list(self.bins.values())

    @property
    def keySet(self):
        """Set of keys among the ``bins``."""
        return set(self.bins.keys())

    def __call__(self, x):
        """Attempt to get key ``x``, throwing an exception if it does not exist."""
        return self.bins[x]

    def get(self, x):
        """Attempt to get key ``x``, returning ``None`` if it does not exist."""
        return self.bins.get(x)

    def getOrElse(self, x, default):
        """Attempt to get key ``x``, returning an alternative if it does not exist."""
        return self.bins.get(x, default)

    @inheritdoc(Container)
    def zero(self):
        return Categorize(self.quantity, self.value)

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Categorize):
            out = Categorize(self.quantity, self.value)
            out.entries = self.entries + other.entries
            out.bins = {}
            for k in self.keySet.union(other.keySet):
                if k in self.bins and k in other.bins:
                    out.bins[k] = self.bins[k] + other.bins[k]
                elif k in self.bins:
                    out.bins[k] = self.bins[k].copy()
                else:
                    out.bins[k] = other.bins[k].copy()
            return out.specialize()

        raise ContainerException(f"cannot add {self.name} and {other.name}")

    @inheritdoc(Container)
    def __iadd__(self, other):
        if isinstance(other, Categorize):
            self.entries += other.entries
            for k in self.keySet.union(other.keySet):
                if k in self.bins and k in other.bins:
                    self.bins[k] += other.bins[k]
                elif k not in self.bins and k in other.bins:
                    self.bins[k] = other.bins[k].copy()
            return self
        raise ContainerException(f"cannot add {self.name} and {other.name}")

    @inheritdoc(Container)
    def __mul__(self, factor):
        if math.isnan(factor) or factor <= 0.0:
            return self.zero()
        out = self.zero()
        out.entries = factor * self.entries
        for k, v in self.bins.items():
            out.bins[k] = v * factor
        return out.specialize()

    @inheritdoc(Container)
    def __rmul__(self, factor):
        return self.__mul__(factor)

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()

        if weight > 0.0:
            q = self.quantity(datum)
            if isinstance(q, (basestring, bool)):
                pass
            elif q is None or np.isnan(q):
                q = "NaN"
            if not isinstance(q, (basestring, bool)):
                raise TypeError(f"function return value ({q}) must be a string or bool")

            if q not in self.bins:
                self.bins[q] = self.value.zero()
            self.bins[q].fill(datum, weight)

            # no possibility of exception from here on out (for rollback)
            self.entries += weight

    def _numpy(self, data, weights, shape):
        q = self.quantity(data)
        if isinstance(q, (list, tuple)):
            q = np.array(q)
        self._checkNPQuantity(q, shape)

        if (
            isinstance(weights, (float, int))
            and weights == 1
            or isinstance(weights, np.ndarray)
            and np.all(weights == 1)
        ):
            all_weights_one = True
        else:
            all_weights_one = False
        self._checkNPWeights(weights, shape)
        weights = self._makeNPWeights(weights, shape)
        newentries = weights.sum()

        subweights = weights.copy()
        subweights[weights < 0.0] = 0.0

        if self.n_dim == 1 and all_weights_one and isinstance(self.value, Count):
            # special case of filling single array where all weights are 1
            uniques, counts = np.unique(q, return_counts=True)

            for c, x in zip(counts, uniques):
                xval = x
                if isinstance(xval, (basestring, bool)):
                    pass
                elif xval is None or np.isnan(xval):
                    xval = "NaN"
                if xval not in self.bins:
                    self.bins[xval] = self.value.zero()
                self.bins[xval]._numpy(None, c, [None])
        else:
            # all other cases ...
            selection = np.empty(q.shape, dtype=bool)
            uniques, inverse = np.unique(q, return_inverse=True)

            # no possibility of exception from here on out (for rollback)
            for i, x in enumerate(uniques):
                xval = x
                if isinstance(xval, (basestring, bool)):
                    pass
                elif xval is None or np.isnan(xval):
                    xval = "NaN"
                if xval not in self.bins:
                    self.bins[xval] = self.value.zero()

                # passing on the full array seems faster for one- AND multi-dim histograms
                np.not_equal(inverse, i, selection)
                subweights[:] = weights
                subweights[selection] = 0.0
                self.bins[xval]._numpy(data, subweights, shape)

        self.entries += float(newentries)

    def _sparksql(self, jvm, converter):
        return converter.Categorize(self.quantity.asSparkSQL(), self.value._sparksql(jvm, converter))

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return [self.value] + list(self.bins.values())

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName):
        if isinstance(self.value, Container):
            if getattr(self.value, "quantity", None) is not None:
                binsName = self.value.quantity.name
            elif getattr(self.value, "quantityName", None) is not None:
                binsName = self.value.quantityName
            else:
                binsName = None
        elif len(self.bins) > 0:
            if getattr(list(self.bins.values())[0], "quantity", None) is not None:
                binsName = list(self.bins.values())[0].quantity.name
            elif getattr(list(self.bins.values())[0], "quantityName", None) is not None:
                binsName = list(self.bins.values())[0].quantityName
            else:
                binsName = None
        else:
            binsName = None

        if len(self.bins) > 0:
            bins_type = list(self.bins.values())[0].name
        elif self.value is not None:
            bins_type = self.value.name
        else:
            bins_type = self.contentType

        return maybeAdd(
            {
                # for json serialization all keys need to be strings, else json libs throws TypeError
                # e.g. boolean keys get converted to strings here
                "entries": floatToJson(self.entries),
                "bins:type": bins_type,
                "bins": {str(k): v.toJsonFragment(True) for k, v in self.bins.items()},
            },
            **{
                "name": None if suppressName else self.quantity.name,
                "bins:name": binsName,
            },
        )

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "bins:type", "bins"], ["name", "bins:name"]):
            if json["entries"] in ("nan", "inf", "-inf") or isinstance(json["entries"], numbers.Real):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "Categorize.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Categorize.name")

            if isinstance(json["bins:type"], basestring):
                contentType = json["bins:type"]
                factory = Factory.registered[contentType]
            else:
                raise JsonFormatException(json, "Categorize.bins:type")

            if isinstance(json.get("bins:name", None), basestring):
                dataName = json["bins:name"]
            elif json.get("bins:name", None) is None:
                dataName = None
            else:
                raise JsonFormatException(json["bins:name"], "Categorize.bins:name")

            if isinstance(json["bins"], dict):
                bins = {k: factory.fromJsonFragment(v, dataName) for k, v in json["bins"].items()}
            else:
                raise JsonFormatException(json, "Categorize.bins")

            out = Categorize.ed(entries, contentType, **bins)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        raise JsonFormatException(json, "Categorize")

    def __repr__(self):
        vals = self.values[0].name if self.size > 0 else self.value.name if self.value is not None else self.contentType
        return f"<Categorize values={vals} size={self.size}"

    def __eq__(self, other):
        return (
            isinstance(other, Categorize)
            and numeq(self.entries, other.entries)
            and self.quantity == other.quantity
            and self.bins == other.bins
        )

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.entries, self.quantity, tuple(sorted(self.bins.items()))))

    @property
    def n_bins(self):
        """Get number of bins, consistent with SparselyBin and Categorize"""
        return self.size

    def bin_entries(self, labels=[]):
        """Returns bin values

        :param list labels: get entries for list of selected labels. When empty return for all labels found.
        :returns: array of bin-entries
        :rtype: numpy.array
        """
        if len(labels) == 0:
            return np.array([self.bins[i].entries for i in self.bins])
        entries = [self.bins[lab].entries if lab in self.bins else 0.0 for lab in labels]
        return np.array(entries)

    def bin_labels(self, max_length=-1):
        """Returns bin labels

        :param int max_length: maximum length of a label. Default is full length.
        :returns: array of labels
        :rtype: numpy.array
        """
        if max_length == -1:
            return np.array(list(self.bins.keys()))

        labels = []
        for i, key in enumerate(self.bins.keys()):
            try:
                label = str(key)
                if max_length > 0:
                    label = label[:max_length]
            except BaseException:
                label = "bin_%d" % i
            labels.append(label)
        return np.array(labels)

    def bin_centers(self, max_length=-1):
        """Returns bin labels

        Compatible function call with Bin and SparselyBin

        :param int max_length: maximum length of a label. Default if full length.
        :returns: array of labels
        :rtype: numpy.array
        """
        return self.bin_labels(max_length)

    def _center_from_key(self, bin_key):
        return bin_key

    @property
    def mpv(self):
        """Return bin-label of most probable value"""
        bin_entries = self.bin_entries()
        bin_labels = self.bin_labels()

        # if two max elements are equal, this will return the element with the lowest index.
        max_idx = max(enumerate(bin_entries), key=lambda x: x[1])[0]
        return bin_labels[max_idx]


# extra properties: number of dimensions and datatypes of sub-hists
Categorize.n_dim = n_dim
Categorize.datatype = datatype

# register extra methods such as plotting
Factory.register(Categorize)
