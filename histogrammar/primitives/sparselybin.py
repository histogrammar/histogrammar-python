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

from histogrammar.defs import Container, Factory, identity, JsonFormatException, ContainerException
from histogrammar.util import n_dim, datatype, serializable, inheritdoc, maybeAdd, floatToJson, hasKeys, numeq, \
    basestring, long
from histogrammar.primitives.count import Count

LONG_NAN = -9223372036854775808
LONG_MINUSINF = -9223372036854775807
LONG_PLUSINF = 9223372036854775807


class SparselyBin(Factory, Container):
    """Split a quantity into equally spaced bins, creating them whenever their ``entries`` would be non-zero.

    Exactly one sub-aggregator is filled per datum.

    Use this when you have a distribution of known scale (bin width) but unknown domain (lowest and highest bin index).

    Unlike fixed-domain binning, this aggregator has the potential to use unlimited memory. A large number
    of *distinct* outliers can generate many unwanted bins.

    Like fixed-domain binning, the bins are indexed by integers, though they are 64-bit and may be negative.
    Bin indexes below ``-(2**63 - 1)`` are put in the ``-(2**63 - 1)`` are bin and indexes above ``(2**63 - 1)``
    are put in the ``(2**63 - 1)`` bin.
    """

    @staticmethod
    def ed(binWidth, entries, contentType, bins, nanflow, origin):
        """Create a SparselyBin that is only capable of being added.

        Parameters:
            binWidth (float): the width of a bin.
            entries (float): the number of entries.
            contentType (str): the value's sub-aggregator type (must be provided to determine type for the case
                when `bins` is empty).
            bins (dict from int to :doc:`Container <histogrammar.defs.Container>`): the non-empty bin indexes and
                their values.
            nanflow (:doc:`Container <histogrammar.defs.Container>`): the filled nanflow bin.
            origin (float): the left edge of the bin whose index is zero.
        """
        if not isinstance(binWidth, numbers.Real):
            raise TypeError("binWidth ({0}) must be a number".format(binWidth))
        if not isinstance(entries, numbers.Real) and entries not in ("nan", "inf", "-inf"):
            raise TypeError("entries ({0}) must be a number".format(entries))
        if not isinstance(contentType, basestring):
            raise TypeError("contentType ({0}) must be a string".format(contentType))
        if not isinstance(bins, dict) or \
                not all(isinstance(k, (int, long)) and isinstance(v, Container) for k, v in bins.items()):
            raise TypeError("bins ({0}) must be a map from 64-bit integers to Containers".format(bins))
        if not isinstance(nanflow, Container):
            raise TypeError("nanflow ({0}) must be a Container".format(nanflow))
        if not isinstance(origin, numbers.Real):
            raise TypeError("origin ({0}) must be a number".format(origin))
        if entries < 0.0:
            raise ValueError("entries ({0}) cannot be negative".format(entries))
        if binWidth <= 0.0:
            raise ValueError("binWidth ({0}) must be greater than zero".format(binWidth))

        out = SparselyBin(binWidth, None, None, nanflow, origin)
        out.entries = float(entries)
        out.contentType = contentType
        out.bins = bins
        return out.specialize()

    @staticmethod
    def ing(binWidth, quantity, value=Count(), nanflow=Count(), origin=0.0):
        """Synonym for ``__init__``."""
        return SparselyBin(binWidth, quantity, value, nanflow, origin)

    def __init__(self, binWidth, quantity=identity, value=Count(), nanflow=Count(), origin=0.0):
        """Create a SparselyBin that is capable of being filled and added.

        Parameters:
            binWidth (float): the width of a bin; must be strictly greater than zero.
            quantity (function returning float): computes the quantity of interest from the data.
            value (:doc:`Container <histogrammar.defs.Container>`): generates sub-aggregators to put in each bin.
            nanflow (:doc:`Container <histogrammar.defs.Container>`): a sub-aggregator to use for data whose quantity
                is NaN.
            origin (float): the left edge of the bin whose index is 0.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
            bins (dict from int to :doc:`Container <histogrammar.defs.Container>`): the map, probably a hashmap, to
                fill with values when their `entries` become non-zero.
        """
        if not isinstance(binWidth, numbers.Real):
            raise TypeError("binWidth ({0}) must be a number".format(binWidth))
        if value is not None and not isinstance(value, Container):
            raise TypeError("value ({0}) must be a Container".format(value))
        if not isinstance(nanflow, Container):
            raise TypeError("nanflow ({0}) must be a Container".format(nanflow))
        if not isinstance(origin, numbers.Real):
            raise TypeError("origin ({0}) must be a number".format(origin))
        if binWidth <= 0.0:
            raise ValueError("binWidth ({0}) must be greater than zero".format(binWidth))

        self.binWidth = float(binWidth)
        self.entries = 0.0
        self.quantity = serializable(identity(quantity) if isinstance(quantity, str) else quantity)
        self.value = value
        if value is not None:
            self.contentType = value.name
        else:
            self.contentType = "Count"
        self.bins = {}
        self.nanflow = nanflow.copy()
        self.origin = float(origin)
        super(SparselyBin, self).__init__()
        self.specialize()

    def histogram(self):
        """Return a plain histogram by converting all sub-aggregator values into Counts"""
        out = SparselyBin(self.binWidth, self.quantity, Count(), self.nanflow.copy(), self.origin)
        out.entries = float(self.entries)
        out.contentType = "Count"
        for i, v in self.bins.items():
            out.bins[i] = Count.ed(v.entries)
        return out.specialize()

    @inheritdoc(Container)
    def zero(self):
        return SparselyBin(self.binWidth, self.quantity, self.value, self.nanflow.zero(), self.origin)

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, SparselyBin):
            if self.binWidth != other.binWidth:
                raise ContainerException(
                    "cannot add SparselyBins because binWidth differs ({0} vs {1})".format(
                        self.binWidth, other.binWidth))
            if self.origin != other.origin:
                raise ContainerException(
                    "cannot add SparselyBins because origin differs ({0} vs {1})".format(
                        self.origin, other.origin))

            out = SparselyBin(
                self.binWidth,
                self.quantity,
                self.value.copy() if self.value is not None else None,
                self.nanflow + other.nanflow,
                self.origin)
            out.entries = self.entries + other.entries
            out.bins = self.bins.copy()
            for i, v in other.bins.items():
                if i in out.bins:
                    out.bins[i] = out.bins[i] + v
                else:
                    out.bins[i] = v
            return out.specialize()

        else:
            raise ContainerException("cannot add {0} and {1}".format(self.name, other.name))

    @inheritdoc(Container)
    def __iadd__(self, other):
        if isinstance(other, SparselyBin):
            if self.binWidth != other.binWidth:
                raise ContainerException(
                    "cannot add SparselyBins because binWidth differs ({0} vs {1})".format(
                        self.binWidth, other.binWidth))
            if self.origin != other.origin:
                raise ContainerException(
                    "cannot add SparselyBins because origin differs ({0} vs {1})".format(
                        self.origin, other.origin))
            self.entries += other.entries
            for i, v in other.bins.items():
                if i in self.bins:
                    self.bins[i] += v
                else:
                    self.bins[i] = v.copy()
            self.nanflow += other.nanflow
            return self
        else:
            raise ContainerException("cannot add {0} and {1}".format(self.name, other.name))

    @inheritdoc(Container)
    def __mul__(self, factor):
        if math.isnan(factor) or factor <= 0.0:
            return self.zero()
        else:
            out = self.zero()
            out.entries = factor * self.entries
            out.bins = dict((c, v * factor) for (c, v) in self.bins.items())
            out.value = self.value.copy() if self.value is not None else None
            out.nanflow = self.nanflow * factor
            return out.specialize()

    @inheritdoc(Container)
    def __rmul__(self, factor):
        return self.__mul__(factor)

    @property
    def numFilled(self):
        """The number of non-empty bins."""
        return len(self.bins)

    @property
    def num(self):
        """The number of bins between the first non-empty one (inclusive) and the last non-empty one (exclusive)."""
        if len(self.bins) == 0:
            return 0
        else:
            return 1 + self.maxBin - self.minBin

    @property
    def minBin(self):
        """The first non-empty bin or None if no values have been accumulated."""
        if len(self.bins) == 0:
            return None
        else:
            return min(self.bins.keys())

    @property
    def maxBin(self):
        """The last non-empty bin or None if no values have been accumulated."""
        if len(self.bins) == 0:
            return None
        else:
            return max(self.bins.keys())

    @property
    def low(self):
        """The low edge of the first non-empty bin or None if no values have been accumulated."""
        if len(self.bins) == 0:
            return None
        else:
            return self.minBin * self.binWidth + self.origin

    @property
    def high(self):
        """The high edge of the last non-empty bin or None if no values have been accumulated."""
        if len(self.bins) == 0:
            return None
        else:
            return (self.maxBin + 1) * self.binWidth + self.origin

    def at(self, index):
        """Extract the container at a given index, if it exists."""
        return self.bins.get(index, None)

    @property
    def indexes(self):
        """Get a sequence of filled indexes."""
        return sorted(self.keys)

    def range(self, index):
        """Get the low and high edge of a bin (given by index number)."""
        return (index * self.binWidth + self.origin, (index + 1) * self.binWidth + self.origin)

    def bin(self, x):
        """Find the bin index associated with numerical value ``x``.

        Returns `-2**63` if `x` is `NaN`, the bin index if it is between `-(2**63 - 1)` and `(2**63 - 1)`,
        otherwise saturate at the endpoints.
        """
        if self.nan(x):
            return LONG_NAN
        else:
            softbin = (x - self.origin) / self.binWidth
            if softbin <= LONG_MINUSINF:
                return LONG_MINUSINF
            elif softbin >= LONG_PLUSINF:
                return LONG_PLUSINF
            else:
                return int(math.floor(softbin))

    def nan(self, x):
        """Return ``true`` iff ``x`` is in the nanflow region (equal to ``NaN``)."""
        return math.isnan(x)

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()

        if weight > 0.0:
            q = self.quantity(datum)
            if not isinstance(q, numbers.Real):
                raise TypeError("function return value ({0}) must be boolean or number".format(q))

            if self.nan(q):
                self.nanflow.fill(datum, weight)
            else:
                b = self.bin(q)
                if b not in self.bins:
                    self.bins[b] = self.value.copy()
                self.bins[b].fill(datum, weight)
            # no possibility of exception from here on out (for rollback)
            self.entries += weight

    def _cppGenerateCode(self, parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes,
                         derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode, fillPrefix,
                         fillIndent, weightVars, weightVarStack, tmpVarTypes):
        normexpr = self._c99QuantityExpr(
            parser,
            generator,
            inputFieldNames,
            inputFieldTypes,
            derivedFieldTypes,
            derivedFieldExprs,
            None)

        initCode.append(" " * initIndent + self._c99ExpandPrefix(*initPrefix) + ".entries = 0.0;")
        initCode.append(" " * initIndent + self._c99ExpandPrefix(*initPrefix) + ".bins.clear();")
        fillCode.append(" " * fillIndent + self._c99ExpandPrefix(*fillPrefix) +
                        ".entries += " + weightVarStack[-1] + ";")

        fillCode.append(" " * fillIndent + "if (std::isnan({0})) {{".format(normexpr))
        self.nanflow._c99GenerateCode(parser,
                                      generator,
                                      inputFieldNames,
                                      inputFieldTypes,
                                      derivedFieldTypes,
                                      derivedFieldExprs,
                                      storageStructs,
                                      initCode,
                                      initPrefix + (("var",
                                                     "nanflow"),
                                                    ),
                                      initIndent,
                                      fillCode,
                                      fillPrefix + (("var",
                                                     "nanflow"),
                                                    ),
                                      fillIndent + 2,
                                      weightVars,
                                      weightVarStack,
                                      tmpVarTypes)
        fillCode.append(" " * fillIndent + "}")
        fillCode.append(" " * fillIndent + "else {")

        softbin = "softbin_" + str(len(tmpVarTypes))
        tmpVarTypes[softbin] = "double"
        key = "key_" + str(len(tmpVarTypes))
        tmpVarTypes[key] = "Long64_t"
        value = "value_" + str(len(tmpVarTypes))
        tmpVarTypes[value] = self.value._c99StorageType() + "*"

        fillCode.append("""{indent}{softbin} = ({q} - {origin}) * {scale};
{indent}if ({softbin} <= {LONG_MINUSINF})
{indent}  {key} = {LONG_MINUSINF};
{indent}else if ({softbin} >= {LONG_PLUSINF})
{indent}  {key} = {LONG_PLUSINF};
{indent}else
{indent}  {key} = floor({softbin});
{indent}if ({bins}.find({key}) == {bins}.end())
{indent}  {bins}[{key}] = {prototype};    // copy
{indent}{value} = &({bins}[{key}]);    // reference""".format(
            indent=" " * (fillIndent + 2),
            softbin=softbin,
            q=normexpr,
            origin=self.origin,
            scale=1.0/self.binWidth,
            key=key,
            value=value,
            prototype=self._c99ExpandPrefix(*fillPrefix) + ".value",
            bins=self._c99ExpandPrefix(*fillPrefix) + ".bins",
            LONG_MINUSINF=LONG_MINUSINF,
            LONG_PLUSINF=LONG_PLUSINF
        ))

        self.value._c99GenerateCode(parser,
                                    generator,
                                    inputFieldNames,
                                    inputFieldTypes,
                                    derivedFieldTypes,
                                    derivedFieldExprs,
                                    storageStructs,
                                    initCode,
                                    initPrefix + (("var",
                                                   "value"),
                                                  ),
                                    initIndent,
                                    fillCode,
                                    (("var",
                                      "(*" + value + ")"),
                                     ),
                                    fillIndent + 2,
                                    weightVars,
                                    weightVarStack,
                                    tmpVarTypes)

        fillCode.append(" " * fillIndent + "}")

        storageStructs[self._c99StructName()] = """
  typedef struct {{
    double entries;
    {2} nanflow;
    {1} value;
    std::unordered_map<Long64_t, {1}> bins;
    {1}& getValues(Long64_t i) {{ return bins[i]; }}
  }} {0};
""".format(self._c99StructName(), self.value._c99StorageType(), self.nanflow._c99StorageType())

    def _c99GenerateCode(self, parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes,
                         derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode, fillPrefix,
                         fillIndent, weightVars, weightVarStack, tmpVarTypes):
        raise NotImplementedError("no C99-compliant implementation of SparselyBin (only C++)")

    def _clingUpdate(self, filler, *extractorPrefix):
        obj = self._clingExpandPrefix(filler, *extractorPrefix)
        self.entries += obj.entries

        for i in obj.bins:
            key = i.first
            if key not in self.bins:
                self.bins[key] = self.value.copy()
            self.bins[key]._clingUpdate(obj, ("func", ["getValues", key]))

        self.nanflow._clingUpdate(obj, ("var", "nanflow"))

    def _c99StructName(self):
        return "Sb" + self.value._c99StructName() + self.nanflow._c99StructName()

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

        # switch to float here like in bin.py else numpy throws
        # TypeError on trivial integer cases such as:
        # >>> q = numpy.array([1,2,3,4])
        # >>> np.divide(q,1,q)
        # >>> np.floor(q,q)
        q = numpy.array(q, dtype=numpy.float64)
        neginfs = numpy.isneginf(q)
        posinfs = numpy.isposinf(q)

        numpy.subtract(q, self.origin, q)
        numpy.divide(q, self.binWidth, q)
        numpy.floor(q, q)
        q = numpy.array(q, dtype=numpy.int64)
        q[neginfs] = LONG_MINUSINF
        q[posinfs] = LONG_PLUSINF

        selected = q[weights > 0.0]

        selection = numpy.empty(q.shape, dtype=numpy.bool)
        for index in numpy.unique(selected):
            if index != LONG_NAN:
                bin = self.bins.get(index)
                if bin is None:
                    bin = self.value.zero()
                    self.bins[index] = bin

                numpy.not_equal(q, index, selection)
                subweights[:] = weights
                subweights[selection] = 0.0
                bin._numpy(data, subweights, shape)

        # no possibility of exception from here on out (for rollback)
        self.entries += float(newentries)

    def _sparksql(self, jvm, converter):
        return converter.SparselyBin(self.binWidth, self.quantity.asSparkSQL(), self.value._sparksql(
            jvm, converter), self.nanflow._sparksql(jvm, converter), self.origin)

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return [self.value, self.nanflow] + list(self.bins.values())

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

        return maybeAdd({
            "binWidth": floatToJson(self.binWidth),
            "entries": floatToJson(self.entries),
            "bins:type": bins_type,
            "bins": dict((str(i), v.toJsonFragment(True)) for i, v in self.bins.items()),
            "nanflow:type": self.nanflow.name,
            "nanflow": self.nanflow.toJsonFragment(False),
            "origin": self.origin,
        }, **{"name": None if suppressName else self.quantity.name,
              "bins:name": binsName})

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(
                json.keys(), ["binWidth", "entries", "bins:type", "bins", "nanflow:type", "nanflow", "origin"],
                ["name", "bins:name"]):
            if json["binWidth"] in ("nan", "inf", "-inf") or isinstance(json["binWidth"], numbers.Real):
                binWidth = float(json["binWidth"])
            else:
                raise JsonFormatException(json, "SparselyBin.binWidth")

            if json["entries"] in ("nan", "inf", "-inf") or isinstance(json["entries"], numbers.Real):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "SparselyBin.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "SparselyBin.name")

            if isinstance(json["bins:type"], basestring):
                binsFactory = Factory.registered[json["bins:type"]]
            else:
                raise JsonFormatException(json, "SparselyBin.bins:type")
            if isinstance(json.get("bins:name", None), basestring):
                binsName = json["bins:name"]
            elif json.get("bins:name", None) is None:
                binsName = None
            else:
                raise JsonFormatException(json["bins:name"], "SparselyBin.bins:name")
            if isinstance(json["bins"], dict):
                for i in json["bins"]:
                    try:
                        int(i)
                    except ValueError:
                        raise JsonFormatException(i, "SparselyBin.bins key must be an integer")

                bins = dict((int(i), binsFactory.fromJsonFragment(v, binsName)) for i, v in json["bins"].items())

            else:
                raise JsonFormatException(json, "SparselyBin.bins")

            if isinstance(json["nanflow:type"], basestring):
                nanflowFactory = Factory.registered[json["nanflow:type"]]
            else:
                raise JsonFormatException(json, "Bin.nanflow:type")
            nanflow = nanflowFactory.fromJsonFragment(json["nanflow"], None)

            if json["origin"] in ("nan", "inf", "-inf") or isinstance(json["origin"], numbers.Real):
                origin = json["origin"]
            else:
                raise JsonFormatException(json, "SparselyBin.origin")

            out = SparselyBin.ed(binWidth, entries, json["bins:type"], bins, nanflow, origin)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        else:
            raise JsonFormatException(json, "SparselyBin")

    def __repr__(self):
        if self.bins is None:
            contentType = self.contentType
        elif len(self.bins) == 0 and self.value is not None:
            contentType = self.value.name
        elif len(self.bins) > 0:
            contentType = (min(self.bins.items())[1]).name
        elif hasattr(self, 'contentType'):
            contentType = self.contentType
        else:  # revert to default
            contentType = 'Count'
        return "<SparselyBin binWidth={0} bins={1} nanflow={2}>".format(
            self.binWidth, self.value.name if self.value is not None else contentType, self.nanflow.name)

    def __eq__(self, other):
        return isinstance(other, SparselyBin) and numeq(self.binWidth, other.binWidth) and \
               self.quantity == other.quantity and numeq(self.entries, other.entries) and self.bins == other.bins and \
               self.nanflow == other.nanflow and numeq(self.origin, other.origin)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.binWidth, self.quantity, self.entries, tuple(
            sorted(self.bins.items())), self.nanflow, self.origin))

    @property
    def n_bins(self):
        """Get number of bins, consistent with SparselyBin and Categorize """
        return self.size

    def num_bins(self, low=None, high=None):
        """
        Returns number of bins

        Possible to set range with low and high params

        :param low: lower edge of range, default is None
        :param high: higher edge of range, default is None
        :returns: number of bins in range
        :rtype: int
        """
        import numpy as np
        # sparse hist not filled
        if self.minBin is None or self.maxBin is None:
            return 0
        # trivial case
        if low is None and high is None:
            return (self.maxBin - self.minBin + 1)
        # catch weird cases
        elif low is not None and high is not None:
            if low > high:
                raise RuntimeError('low {low} greater than high {high}'.format(low=low, high=high))
        # lowest edge
        if low is None:
            minBin = self.minBin
        else:
            minBin = self.bin(low)
        # highest case
        if high is None:
            maxBin = self.maxBin
        else:
            maxBin = self.bin(high)
            if np.isclose(high, self.origin + self.bin_width() * maxBin):
                maxBin -= 1
        nbins = maxBin - minBin + 1
        return nbins

    def bin_width(self):
        """
        Returns bin width
        """
        return self.binWidth

    def bin_edges(self, low=None, high=None):
        """
        Returns bin_edges

        Possible to set range with low and high params

        :param low: lower edge of range, default is None
        :param high: higher edge of range, default is None
        :returns: numpy array with bin edges for selected range
        :rtype: numpy.array
        """
        import numpy as np
        # sparse hist not filled
        if self.minBin is None or self.maxBin is None:
            return np.array([self.origin, self.origin + 1])
        # trivial cases first
        if low is None and high is None:
            num_bins = self.maxBin - self.minBin + 1
            return np.linspace(self.low, self.high, num_bins + 1)
        # catch weird cases
        elif low is not None and high is not None:
            if low > high:
                raise RuntimeError('low {low} greater than high {high}'.format(low=low, high=high))
        # number of bins, before low/high adjustments
        num_bins = self.num_bins(low, high)
        # lowest edge
        if low is None:
            low = self.low
        else:
            minBin = self.bin(low)
            low = self.origin + self.bin_width() * minBin
        # highest edge
        if high is None:
            high = self.high
        else:
            maxBin = self.bin(high)
            if np.isclose(high, self.origin + self.bin_width() * maxBin):
                maxBin -= 1
            high = self.origin + self.bin_width() * (maxBin + 1)
        edges = np.linspace(low, high, num_bins + 1)
        return edges

    def bin_entries(self, low=None, high=None, xvalues=[]):
        """
        Returns bin values

        Possible to set range with low and high params or dedicated x-values

        :param low: lower edge of range, default is None
        :param high: higher edge of range, default is None
        :param xvalues: list of x-values to get entries of, alternative to low and high
        :returns: numpy array with numbers of entries for selected bins
        :rtype: numpy.array
        """
        import numpy as np
        # sparse hist not filled
        if self.minBin is None or self.maxBin is None:
            return np.array([])
        # trivial cases first
        if low is None and high is None and len(xvalues) == 0:
            entries = [self.bins[i].entries if i in self.bins else 0.0 for i in range(self.minBin, self.maxBin + 1)]
            return np.array(entries)
        # catch weird cases
        elif low is not None and high is not None and len(xvalues) == 0:
            if low > high:
                raise RuntimeError('low {low} greater than high {high}'.format(low=low, high=high))
        # entries at request list of x-values
        elif len(xvalues) > 0:
            entries = [self.bins[self.bin(x)].entries if self.bin(x) in self.bins else 0.0 for x in xvalues]
            return np.array(entries)
        # lowest edge
        if low is None:
            minBin = self.minBin
        else:
            minBin = self.bin(low)
        # highest case
        if high is None:
            maxBin = self.maxBin
        else:
            maxBin = self.bin(high)
            if np.isclose(high, self.origin + self.bin_width() * maxBin):
                maxBin -= 1
        entries = [self.bins[i].entries if i in self.bins else 0.0 for i in range(minBin, maxBin + 1)]
        return np.array(entries)

    def bin_centers(self, low=None, high=None):
        """
        Returns bin centers

        Possible to set range with low and high params

        :param low: lower edge of range, default is None
        :param high: higher edge of range, default is None
        :returns: numpy array with bin centers for selected range
        :rtype: numpy.array
        """
        import numpy as np
        bin_edges = self.bin_edges(low, high)
        centers = [(bin_edges[i] + bin_edges[i + 1]) / 2. for i in range(len(bin_edges) - 1)]
        return np.array(centers)

    @property
    def mpv(self):
        """Return bin-center of most probable value
        """
        bin_entries = self.bin_entries()
        bin_centers = self.bin_centers()

        # if two max elements are equal, this will return the element with the lowest index.
        max_idx = max(enumerate(bin_entries), key=lambda x: x[1])[0]
        bc = bin_centers[max_idx]
        return bc

    def _center_from_key(self, bin_key):
        xc = (bin_key + 0.5) * self.binWidth + self.origin
        return xc


# extra properties: number of dimensions and datatypes of sub-hists
SparselyBin.n_dim = n_dim
SparselyBin.datatype = datatype

# register extra methods such as plotting
Factory.register(SparselyBin)
