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

import json
import math
import numbers
import struct

from histogrammar.defs import Container, Factory, identity, JsonFormatException, ContainerException
from histogrammar.util import n_dim, datatype, serializable, inheritdoc, maybeAdd, floatToJson, hasKeys, numeq, \
    floatToC99, basestring, xrange
from histogrammar.primitives.count import Count


class CentrallyBin(Factory, Container):
    """Split a quantity into bins defined by irregularly spaced bin centers.

    Unlike irregular bins defined by explicit ranges, irregular bins defined by bin centers are guaranteed to
    fully partition the space with no gaps and no overlaps. It could be viewed as cluster scoring in one dimension.
    """

    @staticmethod
    def ed(entries, bins, nanflow):
        """Create a CentrallyBin that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            bins (list of float, :doc:`Container <histogrammar.defs.Container>` pairs): the list of bin centers and
                their accumulated data.
            nanflow (:doc:`Container <histogrammar.defs.Container>`): the filled nanflow bin.
        """
        if not isinstance(entries, numbers.Real) and entries not in ("nan", "inf", "-inf"):
            raise TypeError("entries ({0}) must be a number".format(entries))
        if not isinstance(bins, (list, tuple)) and not all(isinstance(v, (list, tuple)) and len(
                v) == 2 and isinstance(v[0], numbers.Real) and isinstance(v[1], Container) for v in bins):
            raise TypeError("bins ({0}) must be a list of number, Container pairs".format(bins))
        if not isinstance(nanflow, Container):
            raise TypeError("nanflow ({0}) must be a Container".format(nanflow))
        if entries < 0.0:
            raise ValueError("entries ({0}) cannot be negative".format(entries))
        out = CentrallyBin(bins, None, None, nanflow)
        out.entries = float(entries)
        out.bins = bins
        return out.specialize()

    @staticmethod
    def ing(centers, quantity, value=Count(), nanflow=Count()):
        """Synonym for ``__init__``."""
        return CentrallyBin(centers, quantity, value, nanflow)

    def __init__(self, centers, quantity=identity, value=Count(), nanflow=Count()):
        """Create a CentrallyBin that is capable of being filled and added.

        Parameters:
            centers (list of float): the centers of all bins
            quantity (function returning float): computes the quantity of interest from the data.
            value (:doc:`Container <histogrammar.defs.Container>`): generates sub-aggregators to put in each bin.
            nanflow (:doc:`Container <histogrammar.defs.Container>`): a sub-aggregator to use for data whose quantity
                is NaN.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
            bins (list of float, :doc:`Container <histogrammar.defs.Container>` pairs): the bin centers and
                sub-aggregators in each bin.
        """

        if not isinstance(centers, (list, tuple)) and not all(isinstance(v, (list, tuple)) and len(
                v) == 2 and isinstance(v[0], numbers.Real) and isinstance(v[1], Container) for v in centers):
            raise TypeError("centers ({0}) must be a list of number, Container pairs".format(centers))
        if value is not None and not isinstance(value, Container):
            raise TypeError("value ({0}) must be None or a Container".format(value))
        if not isinstance(nanflow, Container):
            raise TypeError("nanflow ({0}) must be a Container".format(nanflow))
        if len(centers) < 2:
            raise ValueError("number of centers ({0}) must be at least two".format(len(centers)))

        self.entries = 0.0
        if value is None:
            self.bins = None
        else:
            self.bins = [(float(x), value.zero()) for x in sorted(centers)]

        self.quantity = serializable(identity(quantity) if isinstance(quantity, str) else quantity)
        self.value = value
        self.nanflow = nanflow.copy()

        super(CentrallyBin, self).__init__()
        self.specialize()

    def histogram(self):
        """Return a plain histogram by converting all sub-aggregator values into Counts"""
        out = CentrallyBin([c for c, v in self.bins], self.quantity, Count(), self.nanflow.copy())
        out.entries = self.entries
        for i, v in self.bins:
            out.bins[i] = Count.ed(v.entries)
        return out.specialize()

    @property
    def centersSet(self):
        """Set of centers of each bin."""
        return set(self.centers)

    @property
    def centers(self):
        """Iterable over the centers of each bin."""
        return [c for c, v in self.bins]

    @property
    def values(self):
        """Iterable over the containers associated with each bin."""
        return [v for c, v in self.bins]

    def index(self, x, greater=True):
        """Find the closest index to ``x``."""
        for index in xrange(len(self.bins)):
            if index == len(self.bins) - 1:
                return index
            thisCenter = self.bins[index][0]
            nextCenter = self.bins[index + 1][0]
            if greater:
                if x < (thisCenter + nextCenter)/2.0:
                    return index
            else:
                if x <= (thisCenter + nextCenter)/2.0:
                    return index

    def _lower_index(self, x):
        return self.index(x)

    def _upper_index(self, x):
        return self.index(x, greater=False)

    def center(self, x):
        """Return the exact center of the bin that ``x`` belongs to."""
        return self.bins[self.index(x)][0]

    def value(self, x):
        """Return the aggregator at the center of the bin that ``x`` belongs to."""
        return self.bins[self.index(x)][1]

    def nan(self, x):
        """Return ``true`` iff ``x`` is in the nanflow region (equal to ``NaN``)."""
        return math.isnan(x)

    def neighbors(self, center):
        """Find the lower and upper neighbors of a bin (given by exact bin center)."""
        closestIndex = self.index(center)
        if self.bins[closestIndex][0] != center:
            raise TypeError("position {0} is not the exact center of a bin".format(center))
        elif closestIndex == 0:
            return None, self.bins[closestIndex + 1][0]
        elif closestIndex == len(self.bins) - 1:
            return self.bins[closestIndex - 1][0], None
        else:
            return self.bins[closestIndex - 1][0], self.bins[closestIndex + 1][0]

    def range(self, center):
        """Get the low and high edge of a bin (given by exact bin center)."""
        below, above = self.neighbors(center)    # is never None, None
        if below is None:
            return float("-inf"), (center + above)/2.0
        elif above is None:
            return (below + center)/2.0, float("inf")
        else:
            return (below + center)/2.0, (above + center)/2.0

    @property
    def n_bins(self):
        """Get number of bins, consistent with SparselyBin and Categorize """
        return len(self.bins)

    def num_bins(self, low=None, high=None):
        """
        Returns number of bins of a given (sub-)range

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
        elif low is not None and high is not None:
            if low > high:
                raise RuntimeError('low {low} greater than high {high}'.format(low=low, high=high))
        # lowest, highest edge reset
        if low is None:
            low = float("-inf")
        if high is None:
            high = float("inf")
        # return number of bins
        lidx = self._lower_index(low)
        hidx = self._upper_index(high)
        return hidx - lidx + 1

    def bin_entries(self, low=None, high=None, xvalues=[]):
        """
        Returns bin values

        Possible to set range with low and high params, and list of selected x-values

        :param low: lower edge of range, default is None
        :param high: higher edge of range, default is None
        :param xvalues: list of x-values to get entries of, alternative to low and high
        :returns: numpy array with numbers of entries for selected bins
        :rtype: numpy.array
        """
        import numpy as np
        # trivial case
        if low is None and high is None and len(xvalues) == 0:
            return np.array([b[1].entries for b in self.bins])
        # catch weird cases
        elif low is not None and high is not None and len(xvalues) == 0:
            if low > high:
                raise RuntimeError('low {low} greater than high {high}'.format(low=low, high=high))
        # entries at request list of x-values
        elif len(xvalues) > 0:
            return np.array([(self.bins[self.index(x)])[1].entries for x in xvalues])
        # lowest, highest edge reset
        if low is None:
            low = float("-inf")
        if high is None:
            high = float("inf")
        # return bin entries
        lidx = self._lower_index(low)
        hidx = self._upper_index(high)
        return np.array([(self.bins[i])[1].entries for i in xrange(lidx, hidx + 1)])

    def bin_edges(self, low=None, high=None):
        """
        Returns bin edges

        :param low: lower edge of range, default is None
        :param high: higher edge of range, default is None
        :returns: numpy array with bin edges for selected range
        :rtype: numpy.array
        """
        import numpy as np
        # catch weird cases
        if low is not None and high is not None and low > high:
            raise RuntimeError('low {low} greater than high {high}'.format(low=low, high=high))
        # lowest, highest edge reset
        if low is None:
            low = float('-inf')
        if high is None:
            high = float('inf')
        # return bin edges
        lidx = self._lower_index(low)
        hidx = self._upper_index(high)
        lowest_edge = self.range(self.center(low))[0]
        upper_edges = np.array([self.range(c)[1] for c in self.centers[lidx: hidx + 1]])
        bin_edges = np.concatenate([[lowest_edge], upper_edges])
        return bin_edges

    def bin_centers(self, low=None, high=None):
        """
        Returns bin centers

        :param low: lower edge of range, default is None
        :param high: higher edge of range, default is None
        :returns: numpy array with bin centers for selected range
        :rtype: numpy.array
        """
        import numpy as np
        # catch weird cases
        if low is not None and high is not None and low > high:
            raise RuntimeError('low {low} greater than high {high}'.format(low=low, high=high))
        # lowest, highest edge reset
        if low is None:
            low = -np.inf
        if high is None:
            high = np.inf
        # return bin entries
        lidx = self._lower_index(low)
        hidx = self._upper_index(high)
        return np.array(self.centers[lidx: hidx + 1])

    def _center_from_key(self, center):
        return center

    @property
    def mpv(self):
        """Return bin-center of most probable value
        """
        bin_entries = self.bin_entries()
        # if two max elements are equal, this will return the element with the lowest index.
        max_idx = max(enumerate(bin_entries), key=lambda x: x[1])[0]
        bc = self.centers[max_idx]
        return bc

    @inheritdoc(Container)
    def zero(self):
        return CentrallyBin([c for c, v in self.bins], self.quantity, self.value, self.nanflow.zero())

    @inheritdoc(Container)
    def __add__(self, other):
        if self.centers != other.centers:
            raise ContainerException(
                "cannot add CentrallyBin because centers are different:\n    {0}\nvs\n    {1}".format(
                    self.centers, other.centers))

        newbins = [(c1, v1 + v2) for (c1, v1), (_, v2) in zip(self.bins, other.bins)]

        out = CentrallyBin([c for c, v in self.bins], self.quantity, self.value, self.nanflow + other.nanflow)
        out.entries = self.entries + other.entries
        out.bins = newbins
        return out.specialize()

    @inheritdoc(Container)
    def __iadd__(self, other):
        if self.centers != other.centers:
            raise ContainerException(
                "cannot add CentrallyBin because centers are different:\n    {0}\nvs\n    {1}".format(
                    self.centers, other.centers))
        self.entries += other.entries
        for (c1, v1), (_, v2) in zip(self.bins, other.bins):
            v1 += v2
        self.nanflow += other.nanflow
        return self

    @inheritdoc(Container)
    def __mul__(self, factor):
        if math.isnan(factor) or factor <= 0.0:
            return self.zero()
        else:
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
                raise TypeError("function return value ({0}) must be boolean or number".format(q))

            if self.nan(q):
                self.nanflow.fill(datum, weight)
            else:
                self.bins[self.index(q)][1].fill(datum, weight)

            # no possibility of exception from here on out (for rollback)
            self.entries += weight

    def _cppGenerateCode(self, parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes,
                         derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode, fillPrefix,
                         fillIndent, weightVars, weightVarStack, tmpVarTypes):
        return self._c99GenerateCode(parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes,
                                     derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode,
                                     fillPrefix, fillIndent, weightVars, weightVarStack, tmpVarTypes)

    def _c99GenerateCode(self, parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes,
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

        bin = "bin_" + str(len(tmpVarTypes))
        tmpVarTypes[bin] = "int"

        initCode.append(" " * initIndent + "for ({0} = 0;  {0} < {1};  ++{0}) {{".format(bin, len(self.bins)))

        fillCode.append(" " * fillIndent +
                        "  const double edges[{0}] = {{{1}}};".format(len(self.values) - 1,
                                                                      ", ".join(floatToC99((self.bins[idx - 1][0] +
                                                                                            self.bins[idx][0])/2.0)
                                                                                for idx in xrange(1, len(self.bins)))))

        fillCode.append(" " * fillIndent + "  for ({0} = 0;  {0} < {1};  ++{0}) {{".format(bin, len(self.bins) - 1))
        fillCode.append(" " * fillIndent + "    if ({0} < edges[{1}])".format(normexpr, bin))
        fillCode.append(" " * fillIndent + "      break;")
        fillCode.append(" " * fillIndent + "  }")

        self.bins[0][1]._c99GenerateCode(parser,
                                         generator,
                                         inputFieldNames,
                                         inputFieldTypes,
                                         derivedFieldTypes,
                                         derivedFieldExprs,
                                         storageStructs,
                                         initCode,
                                         initPrefix + (("var",
                                                        "values"),
                                                       ("index",
                                                        bin)),
                                         initIndent + 2,
                                         fillCode,
                                         fillPrefix + (("var",
                                                        "values"),
                                                       ("index",
                                                        bin)),
                                         fillIndent + 2,
                                         weightVars,
                                         weightVarStack,
                                         tmpVarTypes)

        initCode.append(" " * initIndent + "}")
        fillCode.append(" " * fillIndent + "}")

        storageStructs[self._c99StructName()] = """
  typedef struct {{
    double entries;
    {3} nanflow;
    {1} values[{2}];
    {1}& getValues(int i) {{ return values[i]; }}
  }} {0};
""".format(self._c99StructName(),
           self.bins[0][1]._c99StorageType(),
           len(self.values),
           self.nanflow._c99StorageType())

    def _clingUpdate(self, filler, *extractorPrefix):
        obj = self._clingExpandPrefix(filler, *extractorPrefix)
        self.entries += obj.entries
        for i in xrange(len(self.values)):
            self.bins[i][1]._clingUpdate(obj, ("func", ["getValues", i]))
        self.nanflow._clingUpdate(obj, ("var", "nanflow"))

    def _c99StructName(self):
        return "Cb" + str(len(self.bins)) + self.bins[0][1]._c99StructName() + self.nanflow._c99StructName()

    def _cudaGenerateCode(self, parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes,
                          derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode, fillPrefix,
                          fillIndent, combineCode, totalPrefix, itemPrefix, combineIndent, jsonCode, jsonPrefix,
                          jsonIndent, weightVars, weightVarStack, tmpVarTypes, suppressName):
        normexpr = self._cudaQuantityExpr(
            parser,
            generator,
            inputFieldNames,
            inputFieldTypes,
            derivedFieldTypes,
            derivedFieldExprs,
            None)

        initCode.append(" " * initIndent + self._c99ExpandPrefix(*initPrefix) + ".entries = 0.0f;")
        fillCode.append(" " * fillIndent + "atomicAdd(&" + self._c99ExpandPrefix(*fillPrefix) + ".entries, " +
                        weightVarStack[-1] + ");")
        combineCode.append(
            " " *
            combineIndent +
            "atomicAdd(&" +
            self._c99ExpandPrefix(
                *
                totalPrefix) +
            ".entries, " +
            self._c99ExpandPrefix(
                *
                itemPrefix) +
            ".entries);")
        jsonCode.append(" " * jsonIndent + "fprintf(out, \"{\\\"entries\\\": \");")
        jsonCode.append(" " * jsonIndent + "floatToJson(out, " + self._c99ExpandPrefix(*jsonPrefix) + ".entries);")

        fillCode.append(" " * fillIndent + "if (isnan({0})) {{".format(normexpr))
        jsonCode.append(
            " " *
            jsonIndent +
            "fprintf(out, \", \\\"nanflow:type\\\": \\\"" +
            self.nanflow.name +
            "\\\"\");")
        jsonCode.append(" " * jsonIndent + "fprintf(out, \", \\\"nanflow\\\": \");")
        self.nanflow._cudaGenerateCode(parser,
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
                                       initIndent + 2,
                                       fillCode,
                                       fillPrefix + (("var",
                                                      "nanflow"),
                                                     ),
                                       fillIndent + 2,
                                       combineCode,
                                       totalPrefix + (("var",
                                                       "nanflow"),
                                                      ),
                                       itemPrefix + (("var",
                                                      "nanflow"),
                                                     ),
                                       combineIndent,
                                       jsonCode,
                                       jsonPrefix + (("var",
                                                      "nanflow"),
                                                     ),
                                       jsonIndent,
                                       weightVars,
                                       weightVarStack,
                                       tmpVarTypes,
                                       False)
        fillCode.append(" " * fillIndent + "}")
        fillCode.append(" " * fillIndent + "else {")

        bin = "bin_" + str(len(tmpVarTypes))
        tmpVarTypes[bin] = "int"

        initCode.append(" " * initIndent + "for ({0} = 0;  {0} < {1};  ++{0}) {{".format(bin, len(self.bins)))

        fillCode.append(" " * fillIndent + "  const float edges[{0}] = {{{1}}};".format(
            len(self.values) - 1, ", ".join(floatToC99((self.bins[index - 1][0] + self.bins[index][0])/2.0)
                                            for index in xrange(1, len(self.bins)))))
        fillCode.append(" " * fillIndent + "  for ({0} = 0;  {0} < {1};  ++{0}) {{".format(bin, len(self.bins) - 1))
        fillCode.append(" " * fillIndent + "    if ({0} < edges[{1}])".format(normexpr, bin))
        fillCode.append(" " * fillIndent + "      break;")
        fillCode.append(" " * fillIndent + "  }")

        combineCode.append(" " * combineIndent + "for ({0} = 0;  {0} < {1}; ++{0}) {{".format(bin, len(self.bins)))

        jsonCode.append(
            " " *
            jsonIndent +
            "fprintf(out, \", \\\"bins:type\\\": \\\"" +
            self.bins[0][1].name +
            "\\\"\");")
        if hasattr(self.bins[0][1], "quantity") and self.bins[0][1].quantity.name is not None:
            jsonCode.append(
                " " *
                jsonIndent +
                "fprintf(out, \", \\\"bins:name\\\": \\\"" +
                self.bins[0][1].quantity.name +
                "\\\"\");")
        jsonCode.append(" " * jsonIndent + "{")
        jsonCode.append(" " * jsonIndent + "  const float centers[{0}] = {{{1}}};".format(
            len(self.values), ", ".join(floatToC99(center) for center, value in self.bins)))
        jsonCode.append(" " * jsonIndent + "  fprintf(out, \", \\\"bins\\\": [\");")
        jsonCode.append(" " * jsonIndent + "  for ({0} = 0;  {0} < {1};  ++{0}) {{".format(bin, len(self.values)))
        jsonCode.append(" " * jsonIndent + "    fprintf(out, \"{\\\"center\\\": \");")
        jsonCode.append(" " * jsonIndent + "    floatToJson(out, centers[" + bin + "]);")
        jsonCode.append(" " * jsonIndent + "    fprintf(out, \", \\\"data\\\": \");")

        self.bins[0][1]._cudaGenerateCode(parser,
                                          generator,
                                          inputFieldNames,
                                          inputFieldTypes,
                                          derivedFieldTypes,
                                          derivedFieldExprs,
                                          storageStructs,
                                          initCode,
                                          initPrefix + (("var",
                                                         "values"),
                                                        ("index",
                                                         bin)),
                                          initIndent + 2,
                                          fillCode,
                                          fillPrefix + (("var",
                                                         "values"),
                                                        ("index",
                                                         bin)),
                                          fillIndent + 2,
                                          combineCode,
                                          totalPrefix + (("var",
                                                          "values"),
                                                         ("index",
                                                          bin)),
                                          itemPrefix + (("var",
                                                         "values"),
                                                        ("index",
                                                         bin)),
                                          combineIndent + 2,
                                          jsonCode,
                                          jsonPrefix + (("var",
                                                         "values"),
                                                        ("index",
                                                         bin)),
                                          jsonIndent + 4,
                                          weightVars,
                                          weightVarStack,
                                          tmpVarTypes,
                                          True)

        initCode.append(" " * initIndent + "}")
        fillCode.append(" " * fillIndent + "}")
        combineCode.append(" " * combineIndent + "}")
        jsonCode.append(" " * jsonIndent + "    fprintf(out, \"}\");")
        jsonCode.append(" " * jsonIndent + "    if ({0} != {1})".format(bin, len(self.values) - 1))
        jsonCode.append(" " * jsonIndent + "      fprintf(out, \", \");")
        jsonCode.append(" " * jsonIndent + "  }")
        jsonCode.append(" " * jsonIndent + "}")

        if suppressName or self.quantity.name is None:
            jsonCode.append(" " * jsonIndent + "fprintf(out, \"]}\");")
        else:
            jsonCode.append(" " * jsonIndent + "fprintf(out, \"], \\\"name\\\": " +
                            json.dumps(json.dumps(self.quantity.name))[1:-1] + "}\");")

        storageStructs[self._c99StructName()] = """
  typedef struct {{
    float entries;
    {3} nanflow;
    {1} values[{2}];
  }} {0};
""".format(self._c99StructName(),
           self.bins[0][1]._cudaStorageType(),
           len(self.values),
           self.nanflow._cudaStorageType())

    def _cudaUnpackAndFill(self, data, bigendian, alignment):
        format = "<f"
        entries, = struct.unpack(format, data[:struct.calcsize(format)])
        self.entries += entries
        data = data[struct.calcsize(format):]

        data = self.nanflow._cudaUnpackAndFill(data, bigendian, alignment)

        for center, value in self.bins:
            data = value._cudaUnpackAndFill(data, bigendian, alignment)

        return data

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
        q[selection] = 0.0
        weights = weights.copy()
        weights[selection] = 0.0

        if all(isinstance(v, Count) and v.transform is identity for c, v in self.bins) and numpy.all(
                numpy.isfinite(q)) and numpy.all(numpy.isfinite(weights)):

            h, _ = numpy.histogram(q, [float("-inf")] + [(c1 + c2)/2.0 for (c1, v1), (c2, v2)
                                                         in zip(self.bins[:-1], self.bins[1:])] + [float("inf")],
                                   weights=weights)

            for hi, (c, v) in zip(h, self.bins):
                v.fill(None, float(hi))

        else:
            selection = numpy.empty(q.shape, dtype=numpy.bool)
            selection2 = numpy.empty(q.shape, dtype=numpy.bool)

            for index in xrange(len(self.bins)):
                if index == 0:
                    high = (self.bins[index][0] + self.bins[index + 1][0])/2.0
                    numpy.greater_equal(q, high, selection)

                elif index == len(self.bins) - 1:
                    low = (self.bins[index - 1][0] + self.bins[index][0])/2.0
                    numpy.less(q, low, selection)

                else:
                    low = (self.bins[index - 1][0] + self.bins[index][0])/2.0
                    high = (self.bins[index][0] + self.bins[index + 1][0])/2.0
                    numpy.less(q, low, selection)
                    numpy.greater_equal(q, high, selection2)
                    numpy.bitwise_or(selection, selection2, selection)

                subweights[:] = weights
                subweights[selection] = 0.0
                self.bins[index][1]._numpy(data, subweights, shape)

        # no possibility of exception from here on out (for rollback)
        self.entries += float(newentries)

    def _sparksql(self, jvm, converter):
        return converter.CentrallyBin([c for c, v in self.bins], self.quantity.asSparkSQL(
        ), self.bins[0][1]._sparksql(jvm, converter), self.nanflow._sparksql(jvm, converter))

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return [self.nanflow] + [v for c, v in self.bins]

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName):
        if getattr(self.bins[0][1], "quantity", None) is not None:
            binsName = self.bins[0][1].quantity.name
        elif getattr(self.bins[0][1], "quantityName", None) is not None:
            binsName = self.bins[0][1].quantityName
        else:
            binsName = None

        return maybeAdd({
            "entries": floatToJson(self.entries),
            "bins:type": self.bins[0][1].name,
            "bins": [{"center": floatToJson(c), "data": v.toJsonFragment(True)} for c, v in self.bins],
            "nanflow:type": self.nanflow.name,
            "nanflow": self.nanflow.toJsonFragment(False),
        }, **{"name": None if suppressName else self.quantity.name,
              "bins:name": binsName})

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(
                json.keys(), ["entries", "bins:type", "bins", "nanflow:type", "nanflow"], ["name", "bins:name"]):
            if json["entries"] in ("nan", "inf", "-inf") or isinstance(json["entries"], numbers.Real):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "CentrallyBin.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "CentrallyBin.name")

            if isinstance(json["bins:type"], basestring):
                factory = Factory.registered[json["bins:type"]]
            else:
                raise JsonFormatException(json, "CentrallyBin.bins:type")
            if isinstance(json.get("bins:name", None), basestring):
                binsName = json["bins:name"]
            elif json.get("bins:name", None) is None:
                binsName = None
            else:
                raise JsonFormatException(json["bins:name"], "CentrallyBin.bins:name")
            if isinstance(json["bins"], list):
                bins = []
                for i, binpair in enumerate(json["bins"]):
                    if isinstance(binpair, dict) and hasKeys(binpair.keys(), ["center", "data"]):
                        if binpair["center"] in ("nan", "inf", "-inf") or isinstance(binpair["center"], numbers.Real):
                            center = float(binpair["center"])
                        else:
                            JsonFormatException(binpair["center"], "CentrallyBin.bins {0} center".format(i))

                        bins.append((center, factory.fromJsonFragment(binpair["data"], binsName)))

                    else:
                        raise JsonFormatException(binpair, "CentrallyBin.bins {0}".format(i))

            if isinstance(json["nanflow:type"], basestring):
                nanflowFactory = Factory.registered[json["nanflow:type"]]
            else:
                raise JsonFormatException(json, "CentrallyBin.nanflow:type")
            nanflow = nanflowFactory.fromJsonFragment(json["nanflow"], None)

            out = CentrallyBin.ed(entries, bins, nanflow)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        else:
            raise JsonFormatException(json, "CentrallyBin")

    def __repr__(self):
        return "<CentrallyBin bins={0} size={1} nanflow={2}>".format(
            self.bins[0][1].name, len(self.bins), self.nanflow.name)

    def __eq__(self, other):
        return isinstance(other, CentrallyBin) and self.quantity == other.quantity and numeq(
            self.entries, other.entries) and self.bins == other.bins and self.nanflow == other.nanflow

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.quantity, self.entries, tuple(self.bins), self.nanflow))


# extra properties: number of dimensions and datatypes of sub-hists
CentrallyBin.n_dim = n_dim
CentrallyBin.datatype = datatype

# register extra methods
Factory.register(CentrallyBin)
