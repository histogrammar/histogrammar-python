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

from histogrammar.defs import *
from histogrammar.util import *
from histogrammar.primitives.count import *

class Bin(Factory, Container):
    """Split a quantity into equally spaced bins between a low and high threshold and fill exactly one bin per datum.

    When composed with :doc:`Count <histogrammar.primitives.count.Count>`, this produces a standard histogram:

    ::

        Bin.ing(100, 0, 10, fill_x, Count.ing())

    and when nested, it produces a two-dimensional histogram:

    ::

        Bin.ing(100, 0, 10, fill_x,
          Bin.ing(100, 0, 10, fill_y, Count.ing()))

    Combining with [Deviate](#deviate-mean-and-variance) produces a physicist's "profile plot:"

    ::

        Bin.ing(100, 0, 10, fill_x, Deviate.ing(fill_y))

    and so on.
    """

    @staticmethod
    def ed(low, high, entries, values, underflow, overflow, nanflow):
        """Create a Bin that is only capable of being added.

        Parameters:
            low (float): the minimum-value edge of the first bin.
            high (float): the maximum-value edge of the last bin; must be strictly greater than `low`.
            entries (float): the number of entries.
            values (list of :doc:`Container <histogrammar.defs.Container>`): the filled sub-aggregators, one for each bin.
            underflow (:doc:`Container <histogrammar.defs.Container>`): the filled underflow bin.
            overflow (:doc:`Container <histogrammar.defs.Container>`): the filled overflow bin.
            nanflow (:doc:`Container <histogrammar.defs.Container>`): is the filled nanflow bin.
        """
        if not isinstance(low, numbers.Real) and entries not in ("nan", "inf", "-inf"):
            raise TypeError("low ({0}) must be a number".format(low))
        if not isinstance(high, numbers.Real) and entries not in ("nan", "inf", "-inf"):
            raise TypeError("high ({0}) must be a number".format(high))
        if not isinstance(entries, numbers.Real) and entries not in ("nan", "inf", "-inf"):
            raise TypeError("entries ({0}) must be a number".format(entries))
        if not isinstance(values, (list, tuple)) and not all(isinstance(v, Container) for v in values):
            raise TypeError("values ({0}) must be a list of Containers".format(values))
        if not isinstance(underflow, Container):
            raise TypeError("underflow ({0}) must be a Container".format(underflow))
        if not isinstance(overflow, Container):
            raise TypeError("overflow ({0}) must be a Container".format(overflow))
        if not isinstance(nanflow, Container):
            raise TypeError("nanflow ({0}) must be a Container".format(nanflow))
        if low >= high:
            raise ValueError("low ({0}) must be less than high ({1})".format(low, high))
        if entries < 0.0:
            raise ValueError("entries ({0}) cannot be negative".format(entries))
        if len(values) < 1:
            raise ValueError("values ({0}) must have at least one element".format(values))

        out = Bin(len(values), float(low), float(high), None, None, underflow, overflow, nanflow)
        out.entries = float(entries)
        out.values = values
        return out.specialize()

    @staticmethod
    def ing(num, low, high, quantity, value=Count(), underflow=Count(), overflow=Count(), nanflow=Count()):
        """Synonym for ``__init__``."""
        return Bin(num, low, high, quantity, value, underflow, overflow, nanflow)

    def __init__(self, num, low, high, quantity, value=Count(), underflow=Count(), overflow=Count(), nanflow=Count()):
        """Create a Bin that is capable of being filled and added.

        Parameters:
            num (int): the number of bins; must be at least one.
            low (float): the minimum-value edge of the first bin.
            high (float): the maximum-value edge of the last bin; must be strictly greater than `low`.
            quantity (function returning float): computes the quantity of interest from the data.
            value (:doc:`Container <histogrammar.defs.Container>`): generates sub-aggregators to put in each bin.
            underflow (:doc:`Container <histogrammar.defs.Container>`): a sub-aggregator to use for data whose quantity is less than `low`.
            overflow (:doc:`Container <histogrammar.defs.Container>`): a sub-aggregator to use for data whose quantity is greater than or equal to `high`.
            nanflow (:doc:`Container <histogrammar.defs.Container>`): a sub-aggregator to use for data whose quantity is NaN.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
            values (list of :doc:`Container <histogrammar.defs.Container>`): the sub-aggregators in each bin.
        """

        if not isinstance(num, (int, long)):
            raise TypeError("num ({0}) must be an integer".format(num))
        if not isinstance(low, numbers.Real):
            raise TypeError("low ({0}) must be a number".format(low))
        if not isinstance(high, numbers.Real):
            raise TypeError("high ({0}) must be a number".format(high))
        if value is not None and not isinstance(value, Container):
            raise TypeError("value ({0}) must be a Container".format(value))
        if not isinstance(underflow, Container):
            raise TypeError("underflow ({0}) must be a Container".format(underflow))
        if not isinstance(overflow, Container):
            raise TypeError("overflow ({0}) must be a Container".format(overflow))
        if not isinstance(nanflow, Container):
            raise TypeError("nanflow ({0}) must be a Container".format(nanflow))
        if num < 1:
            raise ValueError("num ({0}) must be least one".format(num))
        if low >= high:
            raise ValueError("low ({0}) must be less than high ({1})".format(low, high))

        self.entries = 0.0
        self.low = float(low)
        self.high = float(high)
        self.quantity = serializable(quantity)
        if value is None:
            self.values = [None] * num
        else:
            self.values = [value.zero() for i in xrange(num)]
        self.underflow = underflow.copy()
        self.overflow = overflow.copy()
        self.nanflow = nanflow.copy()
        super(Bin, self).__init__()
        self.specialize()

    def histogram(self):
        """Return a plain histogram by converting all sub-aggregator values into :doc:`Counts <histogrammar.primitives.count.Count>`."""
        out = Bin(len(self.values), self.low, self.high, self.quantity, None, self.underflow.copy(), self.overflow.copy(), self.nanflow.copy())
        out.entries = float(self.entries)
        for i, v in enumerate(self.values):
            out.values[i] = Count.ed(v.entries)
        return out.specialize()

    @inheritdoc(Container)
    def zero(self): return Bin(len(self.values), self.low, self.high, self.quantity, self.values[0].zero(), self.underflow.zero(), self.overflow.zero(), self.nanflow.zero())

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Bin):
            if self.low != other.low:
                raise ContainerException("cannot add Bins because low differs ({0} vs {1})".format(self.low, other.low))
            if self.high != other.high:
                raise ContainerException("cannot add Bins because high differs ({0} vs {1})".format(self.high, other.high))
            if len(self.values) != len(other.values):
                raise ContainerException("cannot add Bins because nubmer of values differs ({0} vs {1})".format(len(self.values), len(other.values)))
            if len(self.values) == 0:
                raise ContainerException("cannot add Bins because number of values is zero")

            out = Bin(len(self.values), self.low, self.high, self.quantity, self.values[0], self.underflow + other.underflow, self.overflow + other.overflow, self.nanflow + other.nanflow)
            out.entries = self.entries + other.entries
            out.values = [x + y for x, y in zip(self.values, other.values)]
            return out.specialize()

        else:
            raise ContainerException("cannot add {0} and {1}".format(self.name, other.name))

    @inheritdoc(Container)
    def __iadd__(self, other):
        if isinstance(other, Bin):
            if self.low != other.low:
                raise ContainerException("cannot add Bins because low differs ({0} vs {1})".format(self.low, other.low))
            if self.high != other.high:
                raise ContainerException("cannot add Bins because high differs ({0} vs {1})".format(self.high, other.high))
            if len(self.values) != len(other.values):
                raise ContainerException("cannot add Bins because nubmer of values differs ({0} vs {1})".format(len(self.values), len(other.values)))
            if len(self.values) == 0:
                raise ContainerException("cannot add Bins because number of values is zero")
            self.entries += other.entries
            for x, y in zip(self.values, other.values):
                x += y
            self.underflow += other.underflow
            self.overflow += other.overflow
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
            for i, v in enumerate(self.values):
                out.values[i] = v * factor
            out.overflow = self.overflow * factor
            out.underflow = self.underflow * factor
            out.nanflow = self.nanflow * factor
            return out.specialize()

    @inheritdoc(Container)
    def __rmul__(self, factor):
        return self.__mul__(factor)

    @property
    def num(self):
        """Number of bins."""
        return len(self.values)

    def bin(self, x):
        """Find the bin index associated with numerical value ``x``.
        
        @return -1 if ``x`` is out of range; the bin index otherwise.
        """
        if self.under(x) or self.over(x) or self.nan(x):
            return -1
        else:
            return int(math.floor(self.num * (x - self.low) / (self.high - self.low)))

    def under(self, x):
        """Return ``true`` iff ``x`` is in the underflow region (less than ``low``)."""
        return not math.isnan(x) and x < self.low

    def over(self, x):
        """Return ``true`` iff ``x`` is in the overflow region (greater than ``high``)."""
        return not math.isnan(x) and x >= self.high

    def nan(self, x):
        """Return ``true`` iff ``x`` is in the nanflow region (equal to ``NaN``)."""
        return math.isnan(x)

    @property
    def indexes(self):
        """Get a sequence of valid indexes."""
        return range(self.num)

    def range(self, index):
        """Get the low and high edge of a bin (given by index number)."""
        return ((self.high - self.low) * index / self.num + self.low, (self.high - self.low) * (index + 1) / self.num + self.low)

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()

        if weight > 0.0:
            q = self.quantity(datum)
            if not isinstance(q, numbers.Real):
                raise TypeError("function return value ({0}) must be boolean or number".format(q))

            if self.under(q):
                self.underflow.fill(datum, weight)
            elif self.over(q):
                self.overflow.fill(datum, weight)
            elif self.nan(q):
                self.nanflow.fill(datum, weight)
            else:
                self.values[self.bin(q)].fill(datum, weight)

            # no possibility of exception from here on out (for rollback)
            self.entries += weight

    def _cppGenerateCode(self, parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode, fillPrefix, fillIndent, weightVars, weightVarStack, tmpVarTypes):
        return self._c99GenerateCode(parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode, fillPrefix, fillIndent, weightVars, weightVarStack, tmpVarTypes)

    def _c99GenerateCode(self, parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode, fillPrefix, fillIndent, weightVars, weightVarStack, tmpVarTypes):
        normexpr = self._c99QuantityExpr(parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, None)

        initCode.append(" " * initIndent + self._c99ExpandPrefix(*initPrefix) + ".entries = 0.0;")
        fillCode.append(" " * fillIndent + self._c99ExpandPrefix(*fillPrefix) + ".entries += " + weightVarStack[-1] + ";")

        fillCode.append(" " * fillIndent + "if (std::isnan({0})) {{".format(normexpr))
        self.nanflow._c99GenerateCode(parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, storageStructs, initCode, initPrefix + (("var", "nanflow"),), initIndent, fillCode, fillPrefix + (("var", "nanflow"),), fillIndent + 2, weightVars, weightVarStack, tmpVarTypes)
        fillCode.append(" " * fillIndent + "}")

        fillCode.append(" " * fillIndent + "else if ({0} < {1}) {{".format(normexpr, self.low))
        self.underflow._c99GenerateCode(parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, storageStructs, initCode, initPrefix + (("var", "underflow"),), initIndent, fillCode, fillPrefix + (("var", "underflow"),), fillIndent + 2, weightVars, weightVarStack, tmpVarTypes)
        fillCode.append(" " * fillIndent + "}")

        fillCode.append(" " * fillIndent + "else if ({0} >= {1}) {{".format(normexpr, self.high))
        self.overflow._c99GenerateCode(parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, storageStructs, initCode, initPrefix + (("var", "overflow"),), initIndent, fillCode, fillPrefix + (("var", "overflow"),), fillIndent + 2, weightVars, weightVarStack, tmpVarTypes)
        fillCode.append(" " * fillIndent + "}")

        fillCode.append(" " * fillIndent + "else {")

        bin = "bin_" + str(len(tmpVarTypes))
        tmpVarTypes[bin] = "int"
        initCode.append(" " * initIndent + "for ({0} = 0;  {0} < {1};  ++{0}) {{".format(bin, len(self.values)))

        fillCode.append(" " * (fillIndent + 2) + "{0} = floor(({1} - {2}) * {3});".format(bin, normexpr, self.low, len(self.values)/(self.high - self.low)))

        self.values[0]._c99GenerateCode(parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, storageStructs, initCode, initPrefix + (("var", "values"), ("index", bin)), initIndent + 2, fillCode, fillPrefix + (("var", "values"), ("index", bin)), fillIndent + 2, weightVars, weightVarStack, tmpVarTypes)

        initCode.append(" " * initIndent + "}")
        fillCode.append(" " * fillIndent + "}")

        storageStructs[self._c99StructName()] = """
  typedef struct {{
    double entries;
    {3} underflow;
    {4} overflow;
    {5} nanflow;
    {1} values[{2}];
    {1}& getValues(int i) {{ return values[i]; }}
  }} {0};
""".format(self._c99StructName(), self.values[0]._c99StorageType(), len(self.values), self.underflow._c99StorageType(), self.overflow._c99StorageType(), self.nanflow._c99StorageType())

    def _clingUpdate(self, filler, *extractorPrefix):
        obj = self._clingExpandPrefix(filler, *extractorPrefix)
        self.entries += obj.entries
        for i in xrange(len(self.values)):
            self.values[i]._clingUpdate(obj, ("func", ["getValues", i]))
        self.underflow._clingUpdate(obj, ("var", "underflow"))
        self.overflow._clingUpdate(obj, ("var", "overflow"))
        self.nanflow._clingUpdate(obj, ("var", "nanflow"))

    def _c99StructName(self):
        return "Bn" + str(len(self.values)) + self.values[0]._c99StructName() + self.underflow._c99StructName() + self.overflow._c99StructName() + self.nanflow._c99StructName()

    def _cudaGenerateCode(self, parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode, fillPrefix, fillIndent, combineCode, totalPrefix, itemPrefix, combineIndent, jsonCode, jsonPrefix, jsonIndent, weightVars, weightVarStack, tmpVarTypes, suppressName):
        normexpr = self._cudaQuantityExpr(parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, None)

        initCode.append(" " * initIndent + self._c99ExpandPrefix(*initPrefix) + ".entries = 0.0f;")
        fillCode.append(" " * fillIndent + "atomicAdd(&" + self._c99ExpandPrefix(*fillPrefix) + ".entries, " + weightVarStack[-1] + ");")
        combineCode.append(" " * combineIndent + "atomicAdd(&" + self._c99ExpandPrefix(*totalPrefix) + ".entries, " + self._c99ExpandPrefix(*itemPrefix) + ".entries);")
        jsonCode.append(" " * jsonIndent + "fprintf(out, \"{\\\"low\\\": " + str(self.low) + ", \\\"high\\\": " + str(self.high) + ", \\\"entries\\\": \");")
        jsonCode.append(" " * jsonIndent + "floatToJson(out, " + self._c99ExpandPrefix(*jsonPrefix) + ".entries);")

        fillCode.append(" " * fillIndent + "if (isnan({0})) {{".format(normexpr))
        jsonCode.append(" " * jsonIndent + "fprintf(out, \", \\\"nanflow:type\\\": \\\"" + self.nanflow.name + "\\\"\");")
        jsonCode.append(" " * jsonIndent + "fprintf(out, \", \\\"nanflow\\\": \");")
        self.nanflow._cudaGenerateCode(parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, storageStructs, initCode, initPrefix + (("var", "nanflow"),), initIndent + 2, fillCode, fillPrefix + (("var", "nanflow"),), fillIndent + 2, combineCode, totalPrefix + (("var", "nanflow"),), itemPrefix + (("var", "nanflow"),), combineIndent, jsonCode, jsonPrefix + (("var", "nanflow"),), jsonIndent, weightVars, weightVarStack, tmpVarTypes, False)
        fillCode.append(" " * fillIndent + "}")

        fillCode.append(" " * fillIndent + "else if ({0} < {1}) {{".format(normexpr, self.low))
        jsonCode.append(" " * jsonIndent + "fprintf(out, \", \\\"underflow:type\\\": \\\"" + self.underflow.name + "\\\"\");")
        jsonCode.append(" " * jsonIndent + "fprintf(out, \", \\\"underflow\\\": \");")
        self.underflow._cudaGenerateCode(parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, storageStructs, initCode, initPrefix + (("var", "underflow"),), initIndent + 2, fillCode, fillPrefix + (("var", "underflow"),), fillIndent + 2, combineCode, totalPrefix + (("var", "underflow"),), itemPrefix + (("var", "underflow"),), combineIndent, jsonCode, jsonPrefix + (("var", "underflow"),), jsonIndent, weightVars, weightVarStack, tmpVarTypes, False)
        fillCode.append(" " * fillIndent + "}")

        fillCode.append(" " * fillIndent + "else if ({0} >= {1}) {{".format(normexpr, self.high))
        jsonCode.append(" " * jsonIndent + "fprintf(out, \", \\\"overflow:type\\\": \\\"" + self.overflow.name + "\\\"\");")
        jsonCode.append(" " * jsonIndent + "fprintf(out, \", \\\"overflow\\\": \");")
        self.overflow._cudaGenerateCode(parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, storageStructs, initCode, initPrefix + (("var", "overflow"),), initIndent + 2, fillCode, fillPrefix + (("var", "overflow"),), fillIndent + 2, combineCode, totalPrefix + (("var", "overflow"),), itemPrefix + (("var", "overflow"),), combineIndent, jsonCode, jsonPrefix + (("var", "overflow"),), jsonIndent, weightVars, weightVarStack, tmpVarTypes, False)
        fillCode.append(" " * fillIndent + "}")

        fillCode.append(" " * fillIndent + "else {")

        bin = "bin_" + str(len(tmpVarTypes))
        tmpVarTypes[bin] = "int"

        initCode.append(" " * initIndent + "for ({0} = 0;  {0} < {1};  ++{0}) {{".format(bin, len(self.values)))

        fillCode.append(" " * (fillIndent + 2) + "{0} = floor(({1} - {2}) * {3});".format(bin, normexpr, self.low, len(self.values)/(self.high - self.low)))

        combineCode.append(" " * combineIndent + "for ({0} = 0;  {0} < {1}; ++{0}) {{".format(bin, len(self.values)))

        jsonCode.append(" " * jsonIndent + "fprintf(out, \", \\\"values:type\\\": \\\"" + self.values[0].name + "\\\"\");")
        if hasattr(self.values[0], "quantity") and self.values[0].quantity.name is not None:
            jsonCode.append(" " * jsonIndent + "fprintf(out, \", \\\"values:name\\\": \\\"" + self.values[0].quantity.name + "\\\"\");")
        jsonCode.append(" " * jsonIndent + "fprintf(out, \", \\\"values\\\": [\");")
        jsonCode.append(" " * jsonIndent + "for ({0} = 0;  {0} < {1};  ++{0}) {{".format(bin, len(self.values)))
        self.values[0]._cudaGenerateCode(parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, storageStructs, initCode, initPrefix + (("var", "values"), ("index", bin)), initIndent + 2, fillCode, fillPrefix + (("var", "values"), ("index", bin)), fillIndent + 2, combineCode, totalPrefix + (("var", "values"), ("index", bin)), itemPrefix + (("var", "values"), ("index", bin)), combineIndent + 2, jsonCode, jsonPrefix + (("var", "values"), ("index", bin)), jsonIndent + 2, weightVars, weightVarStack, tmpVarTypes, True)

        initCode.append(" " * initIndent + "}")

        fillCode.append(" " * fillIndent + "}")

        combineCode.append(" " * combineIndent + "}")

        jsonCode.append(" " * jsonIndent + "  if ({0} != {1})".format(bin, len(self.values) - 1))
        jsonCode.append(" " * jsonIndent + "    fprintf(out, \", \");")
        jsonCode.append(" " * jsonIndent + "}")

        if suppressName or self.quantity.name is None:
            jsonCode.append(" " * jsonIndent + "fprintf(out, \"]}\");")
        else:
            jsonCode.append(" " * jsonIndent + "fprintf(out, \"], \\\"name\\\": " + json.dumps(json.dumps(self.quantity.name))[1:-1] + "}\");")

        storageStructs[self._c99StructName()] = """
  typedef struct {{
    float entries;
    {3} underflow;
    {4} overflow;
    {5} nanflow;
    {1} values[{2}];
  }} {0};
""".format(self._c99StructName(), self.values[0]._cudaStorageType(), len(self.values), self.underflow._cudaStorageType(), self.overflow._cudaStorageType(), self.nanflow._cudaStorageType())

    def _cudaUnpackAndFill(self, data, bigendian, alignment):
        format = "<f"
        entries, = struct.unpack(format, data[:struct.calcsize(format)])
        self.entries += entries
        data = data[struct.calcsize(format):]

        data = self.underflow._cudaUnpackAndFill(data, bigendian, alignment)
        data = self.overflow._cudaUnpackAndFill(data, bigendian, alignment)
        data = self.nanflow._cudaUnpackAndFill(data, bigendian, alignment)

        for value in self.values:
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
        q[selection] = self.high
        weights = weights.copy()
        weights[selection] = 0.0

        numpy.greater_equal(q, self.low, selection)
        subweights[:] = weights
        subweights[selection] = 0.0
        self.underflow._numpy(data, subweights, shape)

        numpy.less(q, self.high, selection)
        subweights[:] = weights
        subweights[selection] = 0.0
        self.overflow._numpy(data, subweights, shape)

        if all(isinstance(value, Count) and value.transform is identity for value in self.values) and numpy.all(numpy.isfinite(q)) and numpy.all(numpy.isfinite(weights)):
            # Numpy defines histograms as including the upper edge of the last bin only, so drop that
            weights[q == self.high] == 0.0

            h, _ = numpy.histogram(q, self.num, (self.low, self.high), weights=weights)

            for hi, value in zip(h, self.values):
                value.fill(None, float(hi))

        else:
            q = numpy.array(q, dtype=numpy.float64)
            numpy.subtract(q, self.low, q)
            numpy.multiply(q, self.num, q)
            numpy.divide(q, self.high - self.low, q)
            numpy.floor(q, q)
            q = numpy.array(q, dtype=int)

            for index, value in enumerate(self.values):
                numpy.not_equal(q, index, selection)
                subweights[:] = weights
                subweights[selection] = 0.0
                value._numpy(data, subweights, shape)

        # no possibility of exception from here on out (for rollback)
        self.entries += float(newentries)

    def _sparksql(self, jvm, converter):
        return converter.Bin(len(self.values), self.low, self.high, self.quantity.asSparkSQL(), self.values[0]._sparksql(jvm, converter), self.underflow._sparksql(jvm, converter), self.overflow._sparksql(jvm, converter), self.nanflow._sparksql(jvm, converter))

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return [self.underflow, self.overflow, self.nanflow] + self.values

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName):
        if getattr(self.values[0], "quantity", None) is not None:
            binsName = self.values[0].quantity.name
        elif getattr(self.values[0], "quantityName", None) is not None:
            binsName = self.values[0].quantityName
        else:
            binsName = None

        return maybeAdd({
            "low": floatToJson(self.low),
            "high": floatToJson(self.high),
            "entries": floatToJson(self.entries),
            "values:type": self.values[0].name,
            "values": [x.toJsonFragment(True) for x in self.values],
            "underflow:type": self.underflow.name,
            "underflow": self.underflow.toJsonFragment(False),
            "overflow:type": self.overflow.name,
            "overflow": self.overflow.toJsonFragment(False),
            "nanflow:type": self.nanflow.name,
            "nanflow": self.nanflow.toJsonFragment(False),
            }, **{"name": None if suppressName else self.quantity.name,
                  "values:name": binsName})

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["low", "high", "entries", "values:type", "values", "underflow:type", "underflow", "overflow:type", "overflow", "nanflow:type", "nanflow"], ["name", "values:name"]):
            if json["low"] in ("nan", "inf", "-inf") or isinstance(json["low"], numbers.Real):
                low = float(json["low"])
            else:
                raise JsonFormatException(json, "Bin.low")

            if json["high"] in ("nan", "inf", "-inf") or isinstance(json["high"], numbers.Real):
                high = float(json["high"])
            else:
                raise JsonFormatException(json, "Bin.high")

            if json["entries"] in ("nan", "inf", "-inf") or isinstance(json["entries"], numbers.Real):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "Bin.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Bin.name")

            if isinstance(json["values:type"], basestring):
                valuesFactory = Factory.registered[json["values:type"]]
            else:
                raise JsonFormatException(json, "Bin.values:type")
            if isinstance(json.get("values:name", None), basestring):
                valuesName = json["values:name"]
            elif json.get("values:name", None) is None:
                valuesName = None
            else:
                raise JsonFormatException(json["values:name"], "Bin.values:name")
            if isinstance(json["values"], list):
                values = [valuesFactory.fromJsonFragment(x, valuesName) for x in json["values"]]
            else:
                raise JsonFormatException(json, "Bin.values")

            if isinstance(json["underflow:type"], basestring):
                underflowFactory = Factory.registered[json["underflow:type"]]
            else:
                raise JsonFormatException(json, "Bin.underflow:type")
            underflow = underflowFactory.fromJsonFragment(json["underflow"], None)

            if isinstance(json["overflow:type"], basestring):
                overflowFactory = Factory.registered[json["overflow:type"]]
            else:
                raise JsonFormatException(json, "Bin.overflow:type")
            overflow = overflowFactory.fromJsonFragment(json["overflow"], None)

            if isinstance(json["nanflow:type"], basestring):
                nanflowFactory = Factory.registered[json["nanflow:type"]]
            else:
                raise JsonFormatException(json, "Bin.nanflow:type")
            nanflow = nanflowFactory.fromJsonFragment(json["nanflow"], None)

            out = Bin.ed(low, high, entries, values, underflow, overflow, nanflow)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        else:
            raise JsonFormatException(json, "Bin")
        
    def __repr__(self):
        return "<Bin num={0} low={1} high={2} values={3} underflow={4} overflow={5} nanflow={6}>".format(len(self.values), self.low, self.high, self.values[0].name, self.underflow.name, self.overflow.name, self.nanflow.name)

    def __eq__(self, other):
        return isinstance(other, Bin) and numeq(self.low, other.low) and numeq(self.high, other.high) and self.quantity == other.quantity and numeq(self.entries, other.entries) and self.values == other.values and self.underflow == other.underflow and self.overflow == other.overflow and self.nanflow == other.nanflow

    def __ne__(self, other): return not self == other

    def __hash__(self):
        return hash((self.low, self.high, self.quantity, self.entries, tuple(self.values), self.underflow, self.overflow, self.nanflow))

Factory.register(Bin)
