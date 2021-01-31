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
    basestring


class Average(Factory, Container):
    """Accumulate the weighted mean of a given quantity.

    Uses the numerically stable weighted mean algorithm described in `"Incremental calculation of weighted mean
    and variance," <http://www-uxsup.csx.cam.ac.uk/~fanf2/hermes/doc/antiforgery/stats.pdf>`_ Tony Finch,
    *Univeristy of Cambridge Computing Service,* 2009.
    """

    @staticmethod
    def ed(entries, mean):
        """Create an Average that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            mean (float): the mean.
        """

        if not isinstance(entries, numbers.Real) and entries not in ("nan", "inf", "-inf"):
            raise TypeError("entries ({0}) must be a number".format(entries))
        if not isinstance(mean, numbers.Real) and entries not in ("nan", "inf", "-inf"):
            raise TypeError("mean ({0}) must be a number".format(mean))
        if entries < 0.0:
            raise ValueError("entries ({0}) cannot be negative".format(entries))
        out = Average(None)
        out.entries = float(entries)
        out.mean = float(mean)
        return out.specialize()

    @staticmethod
    def ing(quantity):
        """Synonym for ``__init__``."""
        return Average(quantity)

    def __init__(self, quantity=identity):
        """Create an Average that is capable of being filled and added.

        Parameters:
            quantity (function returning float): computes the quantity of interest from the data.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
            mean (float): the running mean, initially NaN.
        """
        self.quantity = serializable(identity(quantity) if isinstance(quantity, str) else quantity)
        self.entries = 0.0
        self.mean = float("nan")
        super(Average, self).__init__()
        self.specialize()

    @inheritdoc(Container)
    def zero(self):
        return Average(self.quantity)

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Average):
            out = Average(self.quantity)
            out.entries = self.entries + other.entries
            if self.entries == 0.0:
                out.mean = other.mean
            elif other.entries == 0.0:
                out.mean = self.mean
            else:
                out.mean = (self.entries*self.mean + other.entries*other.mean)/(self.entries + other.entries)
            return out.specialize()
        else:
            raise ContainerException("cannot add {0} and {1}".format(self.name, other.name))

    @inheritdoc(Container)
    def __iadd__(self, other):
        both = self + other
        self.entries = both.entries
        self.mean = both.mean
        return self

    @inheritdoc(Container)
    def __mul__(self, factor):
        if math.isnan(factor) or factor <= 0.0:
            return self.zero()
        else:
            out = self.zero()
            out.entries = factor * self.entries
            out.mean = self.mean
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

            # no possibility of exception from here on out (for rollback)
            if self.entries == 0.0:
                self.mean = q
            self.entries += weight

            if math.isnan(self.mean) or math.isnan(q):
                self.mean = float("nan")

            elif math.isinf(self.mean) or math.isinf(q):
                if math.isinf(self.mean) and math.isinf(q) and self.mean * q < 0.0:
                    self.mean = float("nan")       # opposite-sign infinities is bad
                elif math.isinf(q):
                    self.mean = q                  # mean becomes infinite with sign of q
                else:
                    pass                           # mean is already infinite
                if math.isinf(self.entries) or math.isnan(self.entries):
                    self.mean = float("nan")       # non-finite denominator is bad

            else:                                  # handle finite case
                delta = q - self.mean
                shift = delta * weight / self.entries
                self.mean += shift

    def _cppGenerateCode(self, parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes,
                         derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode, fillPrefix,
                         fillIndent, weightVars, weightVarStack, tmpVarTypes):
        return self._c99GenerateCode(parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes,
                                     derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode,
                                     fillPrefix, fillIndent, weightVars, weightVarStack, tmpVarTypes)

    def _c99GenerateCode(self, parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes,
                         derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode, fillPrefix,
                         fillIndent, weightVars, weightVarStack, tmpVarTypes):
        initCode.append(" " * initIndent + self._c99ExpandPrefix(*initPrefix) + ".entries = 0.0;")
        initCode.append(" " * initIndent + self._c99ExpandPrefix(*initPrefix) + ".mean = 0.0;")

        normexpr = self._c99QuantityExpr(
            parser,
            generator,
            inputFieldNames,
            inputFieldTypes,
            derivedFieldTypes,
            derivedFieldExprs,
            None)
        fillCode.append(" " * fillIndent + self._c99ExpandPrefix(*fillPrefix) + ".entries += " +
                        weightVarStack[-1] + ";")

        delta = "delta_" + str(len(tmpVarTypes))
        tmpVarTypes[delta] = "double"
        shift = "shift_" + str(len(tmpVarTypes))
        tmpVarTypes[shift] = "double"

        fillCode.append("""{indent}if (std::isnan({mean})  ||  std::isnan({q})) {{
{indent}  {mean} = NAN;
{indent}}}
{indent}else if (std::isinf({mean})  ||  std::isinf({q})) {{
{indent}  if (std::isinf({mean})  &&  std::isinf({q})  &&  {mean} * {q} < 0.0)
{indent}    {mean} = NAN;
{indent}  else if (std::isinf({q}))
{indent}    {mean} = {q};
{indent}  else
{indent}    {{ }}
{indent}  if (std::isinf({entries})  ||  std::isnan({entries}))
{indent}    {mean} = NAN;
{indent}}}
{indent}else {{
{indent}  {delta} = {q} - {mean};
{indent}  {shift} = {delta} * {weight} / {entries};
{indent}  {mean} += {shift};
{indent}}}""".format(indent=" " * fillIndent,
                     entries=self._c99ExpandPrefix(*fillPrefix) + ".entries",
                     mean=self._c99ExpandPrefix(*fillPrefix) + ".mean",
                     q=normexpr,
                     delta=delta,
                     shift=shift,
                     weight=weightVarStack[-1]))

        storageStructs[self._c99StructName()] = """
  typedef struct {{
    double entries;
    double mean;
  }} {0};
""".format(self._c99StructName())

    def _clingUpdate(self, filler, *extractorPrefix):
        obj = self._clingExpandPrefix(filler, *extractorPrefix)

        entries = self.entries + obj.entries
        if self.entries == 0.0:
            mean = obj.mean
        elif obj.entries == 0.0:
            mean = self.mean
        else:
            mean = (self.entries*self.mean + obj.entries*obj.mean)/(self.entries + obj.entries)

        self.entries = entries
        self.mean = mean

    def _c99StructName(self):
        return "Av"

    def _cudaGenerateCode(self, parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes,
                          derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode,
                          fillPrefix, fillIndent, combineCode, totalPrefix, itemPrefix, combineIndent, jsonCode,
                          jsonPrefix, jsonIndent, weightVars, weightVarStack, tmpVarTypes, suppressName):
        initCode.append(" " * initIndent + self._c99ExpandPrefix(*initPrefix) + ".entries = 0.0f;")
        initCode.append(" " * initIndent + self._c99ExpandPrefix(*initPrefix) + ".sum = 0.0f;")

        normexpr = self._cudaQuantityExpr(
            parser,
            generator,
            inputFieldNames,
            inputFieldTypes,
            derivedFieldTypes,
            derivedFieldExprs,
            None)
        fillCode.append(" " * fillIndent + "atomicAdd(&" +
                        self._c99ExpandPrefix(*fillPrefix) + ".entries, " + weightVarStack[-1] + ");")
        fillCode.append(" " * fillIndent + "atomicAdd(&" + self._c99ExpandPrefix(*fillPrefix) +
                        ".sum, " + weightVarStack[-1] + " * " + normexpr + ");")

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
        combineCode.append(
            " " *
            combineIndent +
            "atomicAdd(&" +
            self._c99ExpandPrefix(
                *
                totalPrefix) +
            ".sum, " +
            self._c99ExpandPrefix(
                *
                itemPrefix) +
            ".sum);")

        jsonCode.append(" " * jsonIndent + "fprintf(out, \"{\\\"entries\\\": \");")
        jsonCode.append(" " * jsonIndent + "floatToJson(out, " + self._c99ExpandPrefix(*jsonPrefix) + ".entries);")
        jsonCode.append(" " * jsonIndent + "fprintf(out, \", \\\"mean\\\": \");")
        jsonCode.append(" " * jsonIndent + "if (" + self._c99ExpandPrefix(*jsonPrefix) + ".entries == 0.0f)")
        jsonCode.append(" " * jsonIndent + "  fprintf(out, \"\\\"nan\\\"\");")
        jsonCode.append(" " * jsonIndent + "else")
        jsonCode.append(
            " " *
            jsonIndent +
            "  floatToJson(out, " +
            self._c99ExpandPrefix(
                *
                jsonPrefix) +
            ".sum / " +
            self._c99ExpandPrefix(
                *
                jsonPrefix) +
            ".entries);")
        if suppressName or self.quantity.name is None:
            jsonCode.append(" " * jsonIndent + "fprintf(out, \"}\");")
        else:
            jsonCode.append(" " * jsonIndent + "fprintf(out, \", \\\"name\\\": " +
                            json.dumps(json.dumps(self.quantity.name))[1:-1] + "}\");")

        storageStructs[self._c99StructName()] = """
  typedef struct {{
    float entries;
    float sum;
  }} {0};
""".format(self._c99StructName())

    def _cudaUnpackAndFill(self, data, bigendian, alignment):
        format = "<ff"
        objentries, objsum = struct.unpack(format, data[:struct.calcsize(format)])

        entries = self.entries + objentries
        if self.entries == 0.0:
            if objentries == 0.0:
                mean = float("nan")
            else:
                mean = objsum / objentries
        elif objentries == 0.0:
            mean = self.mean
        else:
            mean = (self.entries*self.mean + objsum)/(self.entries + objentries)

        self.entries = entries
        self.mean = mean
        return data[struct.calcsize(format):]

    def _numpy(self, data, weights, shape):
        q = self.quantity(data)
        self._checkNPQuantity(q, shape)
        self._checkNPWeights(weights, shape)
        weights = self._makeNPWeights(weights, shape)

        # no possibility of exception from here on out (for rollback)
        ca, ma = self.entries, self.mean
        if ca == 0.0:
            ma = 0.0

        import numpy
        selection = weights > 0.0
        q = q[selection]
        weights = weights[selection]

        self.entries += float(weights.sum())
        ca_plus_cb = self.entries

        if math.isinf(ca_plus_cb):
            self.mean = float("nan")
        elif ca_plus_cb > 0.0:
            mb = numpy.average(q, weights=weights)
            self.mean = float((ca*ma + (ca_plus_cb - ca)*mb) / ca_plus_cb)

    def _sparksql(self, jvm, converter):
        return converter.Average(self.quantity.asSparkSQL())

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return []

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName):
        return maybeAdd({"entries": floatToJson(self.entries), "mean": floatToJson(self.mean)},
                        name=(None if suppressName else self.quantity.name))

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "mean"], ["name"]):
            if json["entries"] in ("nan", "inf", "-inf") or isinstance(json["entries"], numbers.Real):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json["entries"], "Average.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Average.name")

            if json["mean"] in ("nan", "inf", "-inf") or isinstance(json["mean"], numbers.Real):
                mean = float(json["mean"])
            else:
                raise JsonFormatException(json["mean"], "Average.mean")

            out = Average.ed(entries, mean)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        else:
            raise JsonFormatException(json, "Average")

    def __repr__(self):
        return "<Average mean={0}>".format(self.mean)

    def __eq__(self, other):
        return isinstance(other, Average) and self.quantity == other.quantity and numeq(
            self.entries, other.entries) and numeq(self.mean, other.mean)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.quantity, self.entries, self.mean))


# extra properties: number of dimensions and datatypes of sub-hists
Average.n_dim = n_dim
Average.datatype = datatype

# register extra methods such as plotting
Factory.register(Average)
