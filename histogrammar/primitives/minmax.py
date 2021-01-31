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
    basestring, minplus, maxplus


class Minimize(Factory, Container):
    """Find the minimum value of a given quantity. If no data are observed, the result is NaN."""

    @staticmethod
    def ed(entries, min):
        """Create a Minimize that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            min (float): the lowest value of the quantity observed or NaN if no data were observed.
        """
        if not isinstance(entries, numbers.Real) and entries not in ("nan", "inf", "-inf"):
            raise TypeError("entries ({0}) must be a number".format(entries))
        if not isinstance(min, numbers.Real) and entries not in ("nan", "inf", "-inf"):
            raise TypeError("min ({0}) must be a number".format(min))
        if entries < 0.0:
            raise ValueError("entries ({0}) cannot be negative".format(entries))
        out = Minimize(None)
        out.entries = float(entries)
        out.min = float(min)
        return out.specialize()

    @staticmethod
    def ing(quantity):
        """Synonym for ``__init__``."""
        return Minimize(quantity)

    def __init__(self, quantity=identity):
        """Create a Minimize that is capable of being filled and added.

        Parameters:
            quantity (function returning float): computes the quantity of interest from the data.

        Other parameters:
            entries (float): the number of entries, initially 0.0. #
            min (float): the lowest value of the quantity observed, initially NaN.
        """
        self.quantity = serializable(identity(quantity) if isinstance(quantity, str) else quantity)
        self.entries = 0.0
        self.min = float("nan")
        super(Minimize, self).__init__()
        self.specialize()

    @inheritdoc(Container)
    def zero(self):
        return Minimize(self.quantity)

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Minimize):
            out = Minimize(self.quantity)
            out.entries = self.entries + other.entries
            out.min = minplus(self.min, other.min)
            return out.specialize()
        else:
            raise ContainerException("cannot add {0} and {1}".format(self.name, other.name))

    @inheritdoc(Container)
    def __iadd__(self, other):
        both = self + other
        self.entries = both.entries
        self.min = both.min
        return self

    @inheritdoc(Container)
    def __mul__(self, factor):
        if math.isnan(factor) or factor <= 0.0:
            return self.zero()
        else:
            out = self.zero()
            out.entries = factor * self.entries
            out.min = self.min
            return out.specialize()

    @inheritdoc(Container)
    def __rmul__(self, factor):
        return self.__mul__(factor)

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return []

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()

        if weight > 0.0:
            q = self.quantity(datum)
            if not isinstance(q, numbers.Real):
                raise TypeError("function return value ({0}) must be boolean or number".format(q))

            # no possibility of exception from here on out (for rollback)
            self.entries += weight
            if math.isnan(self.min) or q < self.min:
                self.min = q

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
        initCode.append(" " * initIndent + self._c99ExpandPrefix(*initPrefix) + ".min = NAN;")

        normexpr = self._c99QuantityExpr(
            parser,
            generator,
            inputFieldNames,
            inputFieldTypes,
            derivedFieldTypes,
            derivedFieldExprs,
            None)
        fillCode.append(" " * fillIndent + self._c99ExpandPrefix(*fillPrefix) +
                        ".entries += " + weightVarStack[-1] + ";")

        fillCode.append(" " * fillIndent + "if (std::isnan({min})  ||  {q} < {min}) {min} = {q};".format(
            min=self._c99ExpandPrefix(*fillPrefix) + ".min",
            q=normexpr))

        storageStructs[self._c99StructName()] = """
  typedef struct {{
    double entries;
    double min;
  }} {0};
""".format(self._c99StructName())

    def _clingUpdate(self, filler, *extractorPrefix):
        obj = self._clingExpandPrefix(filler, *extractorPrefix)
        self.entries = self.entries + obj.entries
        self.min = minplus(self.min, obj.min)

    def _c99StructName(self):
        return "Mn"

    def _cudaGenerateCode(self, parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes,
                          derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode, fillPrefix,
                          fillIndent, combineCode, totalPrefix, itemPrefix, combineIndent, jsonCode, jsonPrefix,
                          jsonIndent, weightVars, weightVarStack, tmpVarTypes, suppressName):
        old = "old_" + str(len(tmpVarTypes))
        tmpVarTypes[old] = "int"
        assumed = "assumed_" + str(len(tmpVarTypes))
        tmpVarTypes[assumed] = "float"
        trial = "trial_" + str(len(tmpVarTypes))
        tmpVarTypes[trial] = "float"

        initCode.append(
            " " *
            initIndent +
            "(void){old}; (void){assumed}; (void){trial};  // not used; ignore warnings".format(
                old=old,
                assumed=assumed,
                trial=trial))
        jsonCode.append(
            " " *
            jsonIndent +
            "(void){old}; (void){assumed}; (void){trial};  // not used; ignore warnings".format(
                old=old,
                assumed=assumed,
                trial=trial))

        initCode.append(" " * initIndent + self._c99ExpandPrefix(*initPrefix) + ".entries = 0.0f;")
        initCode.append(" " * initIndent + self._c99ExpandPrefix(*initPrefix) + ".min = CUDART_NAN_F;")

        normexpr = self._cudaQuantityExpr(
            parser,
            generator,
            inputFieldNames,
            inputFieldTypes,
            derivedFieldTypes,
            derivedFieldExprs,
            None)
        fillCode.append("""{indent}atomicAdd(&{prefix}.entries, {weight});
{indent}{old} = *(int*)(&{prefix}.min);
{indent}do {{
{indent}  {assumed} = *(float*)(&{old});
{indent}  if (isnan({assumed})  ||  {q} < {assumed})
{indent}    {trial} = {q};
{indent}  else
{indent}    {trial} = {assumed};
{indent}  {old} = atomicCAS((int*)(&{prefix}.min), *(int*)(&{assumed}), *(int*)(&{trial}));
{indent}}} while (*(int*)(&{assumed}) != {old});
""".format(indent=" " * fillIndent,
           prefix=self._c99ExpandPrefix(*fillPrefix),
           weight=weightVarStack[-1],
           old=old,
           assumed=assumed,
           trial=trial,
           q=normexpr))

        combineCode.append("""{indent}atomicAdd(&{total}.entries, {item}.entries);
{indent}{old} = *(int*)(&{total}.min);
{indent}do {{
{indent}  {assumed} = *(float*)(&{old});
{indent}  if (isnan({assumed}))
{indent}    {trial} = {item}.min;
{indent}  else if (isnan({item}.min))
{indent}    {trial} = {assumed};
{indent}  else if ({assumed} < {item}.min)
{indent}    {trial} = {assumed};
{indent}  else
{indent}    {trial} = {item}.min;
{indent}  {old} = atomicCAS((int*)(&{total}.min), *(int*)(&{assumed}), *(int*)(&{trial}));
{indent}}} while (*(int*)(&{assumed}) != {old});
""".format(indent=" " * combineIndent,
           total=self._c99ExpandPrefix(*totalPrefix),
           item=self._c99ExpandPrefix(*itemPrefix),
           old=old,
           assumed=assumed,
           trial=trial))

        jsonCode.append(" " * jsonIndent + "fprintf(out, \"{\\\"entries\\\": \");")
        jsonCode.append(" " * jsonIndent + "floatToJson(out, " + self._c99ExpandPrefix(*jsonPrefix) + ".entries);")
        jsonCode.append(" " * jsonIndent + "fprintf(out, \", \\\"min\\\": \");")
        jsonCode.append(" " * jsonIndent + "floatToJson(out, " + self._c99ExpandPrefix(*jsonPrefix) + ".min);")
        if suppressName or self.quantity.name is None:
            jsonCode.append(" " * jsonIndent + "fprintf(out, \"}\");")
        else:
            jsonCode.append(" " * jsonIndent + "fprintf(out, \", \\\"name\\\": " +
                            json.dumps(json.dumps(self.quantity.name))[1:-1] + "}\");")

        storageStructs[self._c99StructName()] = """
  typedef struct {{
    float entries;
    float min;
  }} {0};
""".format(self._c99StructName())

    def _cudaUnpackAndFill(self, data, bigendian, alignment):
        format = "<ff"
        objentries, objmin = struct.unpack(format, data[:struct.calcsize(format)])
        self.entries = self.entries + objentries
        self.min = minplus(self.min, objmin)
        return data[struct.calcsize(format):]

    def _numpy(self, data, weights, shape):
        q = self.quantity(data)
        self._checkNPQuantity(q, shape)
        self._checkNPWeights(weights, shape)
        weights = self._makeNPWeights(weights, shape)

        # no possibility of exception from here on out (for rollback)
        import numpy
        selection = numpy.isnan(q)
        numpy.bitwise_not(selection, selection)
        numpy.bitwise_and(selection, weights > 0.0, selection)
        q = q[selection]

        self.entries += float(weights.sum())

        if math.isnan(self.min):
            if q.shape[0] > 0:
                self.min = float(q.min())
        else:
            if q.shape[0] > 0:
                self.min = min(self.min, float(q.min()))

    def _sparksql(self, jvm, converter):
        return converter.Minimize(self.quantity.asSparkSQL())

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName):
        return maybeAdd({"entries": floatToJson(self.entries),
                         "min": floatToJson(self.min)},
                        name=(None if suppressName else self.quantity.name))

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "min"], ["name"]):
            if json["entries"] in ("nan", "inf", "-inf") or isinstance(json["entries"], numbers.Real):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json["entries"], "Minimize.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Minimize.name")

            if json["min"] in ("nan", "inf", "-inf") or isinstance(json["min"], numbers.Real):
                min = float(json["min"])
            else:
                raise JsonFormatException(json["min"], "Minimize.min")

            out = Minimize.ed(entries, min)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        else:
            raise JsonFormatException(json, "Minimize")

    def __repr__(self):
        return "<Minimize min={0}>".format(self.min)

    def __eq__(self, other):
        return isinstance(other, Minimize) and self.quantity == other.quantity and numeq(
            self.entries, other.entries) and numeq(self.min, other.min)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.quantity, self.entries, self.min))


# extra properties: number of dimensions and datatypes of sub-hists
Minimize.n_dim = n_dim
Minimize.datatype = datatype

Factory.register(Minimize)


class Maximize(Factory, Container):
    """Find the maximum value of a given quantity. If no data are observed, the result is NaN."""

    @staticmethod
    def ed(entries, max):
        """Create a Maximize that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            max (float): the highest value of the quantity observed or NaN if no data were observed.
        """
        if not isinstance(entries, numbers.Real) and entries not in ("nan", "inf", "-inf"):
            raise TypeError("entries ({0}) must be a number".format(entries))
        if not isinstance(max, numbers.Real) and entries not in ("nan", "inf", "-inf"):
            raise TypeError("max ({0}) must be a number".format(max))
        if entries < 0.0:
            raise ValueError("entries ({0}) cannot be negative".format(entries))
        out = Maximize(None)
        out.entries = float(entries)
        out.max = float(max)
        return out.specialize()

    @staticmethod
    def ing(quantity):
        """Synonym for ``__init__``."""
        return Maximize(quantity)

    def __init__(self, quantity=identity):
        """Create a Maximize that is capable of being filled and added.

        Parameters:
            quantity (function returning float): computes the quantity of interest from the data.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
            max (float): the highest value of the quantity observed, initially NaN.
        """
        self.quantity = serializable(identity(quantity) if isinstance(quantity, str) else quantity)
        self.entries = 0.0
        self.max = float("nan")
        super(Maximize, self).__init__()
        self.specialize()

    @inheritdoc(Container)
    def zero(self):
        return Maximize(self.quantity)

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Maximize):
            out = Maximize(self.quantity)
            out.entries = self.entries + other.entries
            out.max = maxplus(self.max, other.max)
            return out.specialize()
        else:
            raise ContainerException("cannot add {0} and {1}".format(self.name, other.name))

    @inheritdoc(Container)
    def __iadd__(self, other):
        both = self + other
        self.entries = both.entries
        self.max = both.max
        return self

    @inheritdoc(Container)
    def __mul__(self, factor):
        if math.isnan(factor) or factor <= 0.0:
            return self.zero()
        else:
            out = self.zero()
            out.entries = factor * self.entries
            out.max = self.max
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
            self.entries += weight
            if math.isnan(self.max) or q > self.max:
                self.max = q

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
        initCode.append(" " * initIndent + self._c99ExpandPrefix(*initPrefix) + ".max = NAN;")

        normexpr = self._c99QuantityExpr(
            parser,
            generator,
            inputFieldNames,
            inputFieldTypes,
            derivedFieldTypes,
            derivedFieldExprs,
            None)
        fillCode.append(" " * fillIndent + self._c99ExpandPrefix(*fillPrefix) +
                        ".entries += " + weightVarStack[-1] + ";")

        fillCode.append(" " * fillIndent + "if (std::isnan({max})  ||  {q} > {max}) {max} = {q};".format(
            max=self._c99ExpandPrefix(*fillPrefix) + ".max",
            q=normexpr))

        storageStructs[self._c99StructName()] = """
  typedef struct {{
    double entries;
    double max;
  }} {0};
""".format(self._c99StructName())

    def _clingUpdate(self, filler, *extractorPrefix):
        obj = self._clingExpandPrefix(filler, *extractorPrefix)
        self.entries = self.entries + obj.entries
        self.max = maxplus(self.max, obj.max)

    def _c99StructName(self):
        return "Mx"

    def _cudaGenerateCode(self, parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes,
                          derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode, fillPrefix,
                          fillIndent, combineCode, totalPrefix, itemPrefix, combineIndent, jsonCode, jsonPrefix,
                          jsonIndent, weightVars, weightVarStack, tmpVarTypes, suppressName):
        old = "old_" + str(len(tmpVarTypes))
        tmpVarTypes[old] = "int"
        assumed = "assumed_" + str(len(tmpVarTypes))
        tmpVarTypes[assumed] = "float"
        trial = "trial_" + str(len(tmpVarTypes))
        tmpVarTypes[trial] = "float"

        initCode.append(
            " " *
            initIndent +
            "(void){old}; (void){assumed}; (void){trial};  // not used; ignore warnings".format(
                old=old,
                assumed=assumed,
                trial=trial))
        jsonCode.append(
            " " *
            jsonIndent +
            "(void){old}; (void){assumed}; (void){trial};  // not used; ignore warnings".format(
                old=old,
                assumed=assumed,
                trial=trial))

        initCode.append(" " * initIndent + self._c99ExpandPrefix(*initPrefix) + ".entries = 0.0f;")
        initCode.append(" " * initIndent + self._c99ExpandPrefix(*initPrefix) + ".max = CUDART_NAN_F;")

        normexpr = self._cudaQuantityExpr(
            parser,
            generator,
            inputFieldNames,
            inputFieldTypes,
            derivedFieldTypes,
            derivedFieldExprs,
            None)
        fillCode.append("""{indent}atomicAdd(&{prefix}.entries, {weight});
{indent}{old} = *(int*)(&{prefix}.max);
{indent}do {{
{indent}  {assumed} = *(float*)(&{old});
{indent}  if (isnan({assumed})  ||  {q} > {assumed})
{indent}    {trial} = {q};
{indent}  else
{indent}    {trial} = {assumed};
{indent}  {old} = atomicCAS((int*)(&{prefix}.max), *(int*)(&{assumed}), *(int*)(&{trial}));
{indent}}} while (*(int*)(&{assumed}) != {old});
""".format(indent=" " * fillIndent,
           prefix=self._c99ExpandPrefix(*fillPrefix),
           weight=weightVarStack[-1],
           old=old,
           assumed=assumed,
           trial=trial,
           q=normexpr))

        combineCode.append("""{indent}atomicAdd(&{total}.entries, {item}.entries);
{indent}{old} = *(int*)(&{total}.max);
{indent}do {{
{indent}  {assumed} = *(float*)(&{old});
{indent}  if (isnan({assumed}))
{indent}    {trial} = {item}.max;
{indent}  else if (isnan({item}.max))
{indent}    {trial} = {assumed};
{indent}  else if ({assumed} > {item}.max)
{indent}    {trial} = {assumed};
{indent}  else
{indent}    {trial} = {item}.max;
{indent}  {old} = atomicCAS((int*)(&{total}.max), *(int*)(&{assumed}), *(int*)(&{trial}));
{indent}}} while (*(int*)(&{assumed}) != {old});
""".format(indent=" " * combineIndent,
           total=self._c99ExpandPrefix(*totalPrefix),
           item=self._c99ExpandPrefix(*itemPrefix),
           old=old,
           assumed=assumed,
           trial=trial))

        jsonCode.append(" " * jsonIndent + "fprintf(out, \"{\\\"entries\\\": \");")
        jsonCode.append(" " * jsonIndent + "floatToJson(out, " + self._c99ExpandPrefix(*jsonPrefix) + ".entries);")
        jsonCode.append(" " * jsonIndent + "fprintf(out, \", \\\"max\\\": \");")
        jsonCode.append(" " * jsonIndent + "floatToJson(out, " + self._c99ExpandPrefix(*jsonPrefix) + ".max);")
        if suppressName or self.quantity.name is None:
            jsonCode.append(" " * jsonIndent + "fprintf(out, \"}\");")
        else:
            jsonCode.append(" " * jsonIndent + "fprintf(out, \", \\\"name\\\": " +
                            json.dumps(json.dumps(self.quantity.name))[1:-1] + "}\");")

        storageStructs[self._c99StructName()] = """
  typedef struct {{
    float entries;
    float max;
  }} {0};
""".format(self._c99StructName())

    def _cudaUnpackAndFill(self, data, bigendian, alignment):
        format = "<ff"
        objentries, objmax = struct.unpack(format, data[:struct.calcsize(format)])
        self.entries = self.entries + objentries
        self.max = maxplus(self.max, objmax)
        return data[struct.calcsize(format):]

    def _numpy(self, data, weights, shape):
        q = self.quantity(data)
        self._checkNPQuantity(q, shape)
        self._checkNPWeights(weights, shape)
        weights = self._makeNPWeights(weights, shape)

        # no possibility of exception from here on out (for rollback)
        import numpy
        selection = numpy.isnan(q)
        numpy.bitwise_not(selection, selection)
        numpy.bitwise_and(selection, weights > 0.0, selection)
        q = q[selection]

        self.entries += float(weights.sum())

        if math.isnan(self.max):
            if q.shape[0] > 0:
                self.max = float(q.max())
        else:
            if q.shape[0] > 0:
                self.max = max(self.max, float(q.max()))

    def _sparksql(self, jvm, converter):
        return converter.Maximize(self.quantity.asSparkSQL())

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return []

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName):
        return maybeAdd({"entries": floatToJson(self.entries),
                         "max": floatToJson(self.max)},
                        name=(None if suppressName else self.quantity.name))

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "max"], ["name"]):
            if json["entries"] in ("nan", "inf", "-inf") or isinstance(json["entries"], numbers.Real):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json["entries"], "Maximize.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Maximize.name")

            if json["max"] in ("nan", "inf", "-inf") or isinstance(json["max"], numbers.Real):
                max = float(json["max"])
            else:
                raise JsonFormatException(json["max"], "Maximize.max")

            out = Maximize.ed(entries, max)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        else:
            raise JsonFormatException(json, "Maximize")

    def __repr__(self):
        return "<Maximize max={0}>".format(self.max)

    def __eq__(self, other):
        return isinstance(other, Maximize) and self.quantity == other.quantity and numeq(
            self.entries, other.entries) and numeq(self.max, other.max)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.quantity, self.entries, self.max))


# extra properties: number of dimensions and datatypes of sub-hists
Maximize.n_dim = n_dim
Maximize.datatype = datatype

# register extra methods
Factory.register(Maximize)
