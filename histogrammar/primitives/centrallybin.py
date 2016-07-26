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

from histogrammar.defs import *
from histogrammar.primitives.count import *

class CentrallyBin(Factory, Container):
    """Split a quantity into bins defined by irregularly spaced bin centers, with exactly one sub-aggregator filled per datum (the closest one).

    Unlike irregular bins defined by explicit ranges, irregular bins defined by bin centers are guaranteed to fully partition the space with no gaps and no overlaps. It could be viewed as cluster scoring in one dimension.
    """

    @staticmethod
    def ed(entries, bins, nanflow):
        """Create a CentrallyBin that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            bins (list of float, :doc:`Container <histogrammar.defs.Container>` pairs): the list of bin centers and their accumulated data.
            nanflow (:doc:`Container <histogrammar.defs.Container>`): the filled nanflow bin.
        """
        if not isinstance(entries, numbers.Real) and entries not in ("nan", "inf", "-inf"):
            raise TypeError("entries ({0}) must be a number".format(entries))
        if not isinstance(bins, (list, tuple)) and not all(isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], numbers.Real) and isinstance(v[1], Container) for v in bins):
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
    def ing(bins, quantity, value=Count(), nanflow=Count()):
        """Synonym for ``__init__``."""
        return CentrallyBin(bins, quantity, value, nanflow)

    def __init__(self, bins, quantity, value=Count(), nanflow=Count()):
        """Create a CentrallyBin that is capable of being filled and added.

        Parameters:
            centers (list of float): the centers of all bins
            quantity (function returning float): computes the quantity of interest from the data.
            value (:doc:`Container <histogrammar.defs.Container>`): generates sub-aggregators to put in each bin.
            nanflow (:doc:`Container <histogrammar.defs.Container>`): a sub-aggregator to use for data whose quantity is NaN.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
            bins (list of float, :doc:`Container <histogrammar.defs.Container>` pairs): the bin centers and sub-aggregators in each bin.
        """

        if not isinstance(bins, (list, tuple)) and not all(isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], numbers.Real) and isinstance(v[1], Container) for v in bins):
            raise TypeError("bins ({0}) must be a list of number, Container pairs".format(bins))
        if value is not None and not isinstance(value, Container):
            raise TypeError("value ({0}) must be None or a Container".format(value))
        if not isinstance(nanflow, Container):
            raise TypeError("nanflow ({0}) must be a Container".format(nanflow))
        if len(bins) < 2:
            raise ValueError("number of bins ({0}) must be at least two".format(len(bins)))

        self.entries = 0.0
        if value is None:
            self.bins = None
        else:
            self.bins = [(x, value.zero()) for x in sorted(bins)]

        self.quantity = serializable(quantity)
        self.value = value
        self.nanflow = nanflow.copy()

        super(CentrallyBin, self).__init__()
        self.specialize()

    def histogram(self):
        """Return a plain histogram by converting all sub-aggregator values into :doc:`Counts <histogrammar.primitives.count.Count>`."""
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

    def index(self, x):
        """Find the closest index to ``x``."""
        for index in xrange(len(self.bins)):
            if index == len(self.bins) - 1:
                return index
            thisCenter = self.bins[index][0]
            nextCenter = self.bins[index + 1][0]
            if x < (thisCenter + nextCenter)/2.0:
                return index

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

    @inheritdoc(Container)
    def zero(self):
        return CentrallyBin([c for c, v in self.bins], self.quantity, self.value, self.nanflow.zero())

    @inheritdoc(Container)
    def __add__(self, other):
        if self.centers != other.centers:
            raise ContainerException("cannot add CentrallyBin because centers are different:\n    {0}\nvs\n    {1}".format(self.centers, other.centers))

        newbins = [(c1, v1 + v2) for (c1, v1), (_, v2) in zip(self.bins, other.bins)]

        out = CentrallyBin([c for c, v in self.bins], self.quantity, self.value, self.nanflow + other.nanflow)
        out.entries = self.entries + other.entries
        out.bins = newbins
        return out.specialize()

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

    def _clingGenerateCode(self, parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode, fillPrefix, fillIndent, weightVars, weightVarStack, tmpVarTypes):
        normexpr = self._clingQuantityExpr(parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, None)

        initCode.append(" " * initIndent + self._clingExpandPrefixCpp(*initPrefix) + ".entries = 0.0;")
        fillCode.append(" " * fillIndent + self._clingExpandPrefixCpp(*fillPrefix) + ".entries += " + weightVarStack[-1] + ";")

        fillCode.append(" " * fillIndent + "if (std::isnan({0})) {{".format(normexpr))
        self.nanflow._clingGenerateCode(parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, storageStructs, initCode, initPrefix + (("var", "nanflow"),), initIndent, fillCode, fillPrefix + (("var", "nanflow"),), fillIndent + 2, weightVars, weightVarStack, tmpVarTypes)
        fillCode.append(" " * fillIndent + "}")
        fillCode.append(" " * fillIndent + "else {")

        bin = "bin_" + str(len(tmpVarTypes))
        tmpVarTypes[bin] = "int"

        initCode.append(" " * initIndent + "for ({0} = 0;  {0} < {1};  ++{0}) {{".format(bin, len(self.bins)))
    
        fillCode.append(" " * fillIndent + "  const double edges[{0}] = {{{1}}};".format(
            len(self.values) - 1,
            ", ".join(str((self.bins[index - 1][0] + self.bins[index][0])/2.0) for index in xrange(1, len(self.bins)))))

        fillCode.append(" " * fillIndent + "  for ({0} = 0;  {0} < {1};  ++{0}) {{".format(bin, len(self.bins) - 1))
        fillCode.append(" " * fillIndent + "    if ({0} < edges[{1}])".format(normexpr, bin))
        fillCode.append(" " * fillIndent + "      break;")
        fillCode.append(" " * fillIndent + "  }")

        self.bins[0][1]._clingGenerateCode(parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, storageStructs, initCode, initPrefix + (("var", "values"), ("index", bin)), initIndent + 2, fillCode, fillPrefix + (("var", "values"), ("index", bin)), fillIndent + 2, weightVars, weightVarStack, tmpVarTypes)

        initCode.append(" " * initIndent + "}")
        fillCode.append(" " * fillIndent + "}")

        storageStructs[self._clingStructName()] = """
  typedef struct {{
    double entries;
    {3} nanflow;
    {1} values[{2}];
    {1}& getValues(int i) {{ return values[i]; }}
  }} {0};
""".format(self._clingStructName(),
           self.bins[0][1]._clingStorageType(),
           len(self.values),
           self.nanflow._clingStorageType())

    def _clingUpdate(self, filler, *extractorPrefix):
        obj = self._clingExpandPrefixPython(filler, *extractorPrefix)
        self.entries += obj.entries
        for i in xrange(len(self.values)):
            self.bins[i][1]._clingUpdate(obj, ("func", ["getValues", i]))
        self.nanflow._clingUpdate(obj, ("var", "nanflow"))

    def _clingStructName(self):
        return "Cb" + str(len(self.bins)) + self.bins[0][1]._clingStructName() + self.nanflow._clingStructName()

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

        if all(isinstance(v, Count) and v.transform is identity for c, v in self.bins) and numpy.all(numpy.isfinite(q)) and numpy.all(numpy.isfinite(weights)):

            h, _ = numpy.histogram(q, [float("-inf")] + [(c1 + c2)/2.0 for (c1, v1), (c2, v2) in zip(self.bins[:-1], self.bins[1:])] + [float("inf")], weights=weights)

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
            "bins": [{"center": floatToJson(c), "value": v.toJsonFragment(True)} for c, v in self.bins],
            "nanflow:type": self.nanflow.name,
            "nanflow": self.nanflow.toJsonFragment(False),
            }, **{"name": None if suppressName else self.quantity.name,
                  "bins:name": binsName})

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "bins:type", "bins", "nanflow:type", "nanflow"], ["name", "bins:name"]):
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
                    if isinstance(binpair, dict) and hasKeys(binpair.keys(), ["center", "value"]):
                        if binpair["center"] in ("nan", "inf", "-inf") or isinstance(binpair["center"], numbers.Real):
                            center = float(binpair["center"])
                        else:
                            JsonFormatException(binpair["center"], "CentrallyBin.bins {0} center".format(i))
                        
                        bins.append((center, factory.fromJsonFragment(binpair["value"], binsName)))

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
        return "<CentrallyBin bins={0} size={1} nanflow={2}>".format(self.bins[0][1].name, len(self.bins), self.nanflow.name)

    def __eq__(self, other):
        return isinstance(other, CentrallyBin) and self.quantity == other.quantity and numeq(self.entries, other.entries) and self.bins == other.bins and self.nanflow == other.nanflow

    def __ne__(self, other): return not self == other

    def __hash__(self):
        return hash((self.quantity, self.entries, tuple(self.bins), self.nanflow))

Factory.register(CentrallyBin)
