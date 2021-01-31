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
import struct

from histogrammar.defs import Container, Factory, identity, JsonFormatException, ContainerException
from histogrammar.util import n_dim, datatype, serializable, inheritdoc, floatToJson, numeq


class Count(Factory, Container):
    """Count entries by accumulating the sum of all observed weights or a sum of transformed weights

    (e.g. collect the sum of squares of weights).

    An optional ``transform`` function can be applied to the weights before summing.
    To accumulate the sum of squares of weights, use:

    ::
        lambda x: x**2

    for instance. This is unlike any other primitive's ``quantity`` function in that its domain is
    the *weights* (always double), not *data* (any type).
    """

    @staticmethod
    def ed(entries):
        """Create a Count that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
        """
        if not isinstance(entries, numbers.Real) and entries not in ("nan", "inf", "-inf"):
            raise TypeError("entries ({0}) must be a number".format(entries))
        if entries < 0.0:
            raise ValueError("entries ({0}) cannot be negative".format(entries))
        out = Count()
        out.entries = float(entries)
        return out.specialize()

    @staticmethod
    def ing(transform=identity):
        """Synonym for ``__init__``."""
        return Count(transform)

    def __init__(self, transform=identity):
        """Create a Count that is capable of being filled and added.

        Parameters:
            transform (function from float to float): transforms each weight.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
        """
        self.entries = 0.0
        self.transform = serializable(transform)
        super(Count, self).__init__()
        self.specialize()

    @inheritdoc(Container)
    def zero(self):
        return Count(self.transform)

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Count):
            out = Count(self.transform)
            out.entries = self.entries + other.entries
            return out.specialize()
        else:
            raise ContainerException("cannot add {0} and {1}".format(self.name, other.name))

    @inheritdoc(Container)
    def __iadd__(self, other):
        if isinstance(other, Count):
            self.entries += other.entries
            return self
        else:
            raise ContainerException("cannot add {0} and {1}".format(self.name, other.name))

    @inheritdoc(Container)
    def __mul__(self, factor):
        if self.transform != identity or \
           not callable(self.transform.expr) or \
           (hasattr(self.transform.expr, "func_code") and
            self.transform.expr.func_code.co_code != identity.expr.func_code.co_code) or \
            (hasattr(self.transform.expr, "__code__") and
             self.transform.expr.__code__.co_code != identity.expr.__code__.co_code):
            raise ContainerException("Cannot scalar-multiply Count with a non-identity transform.")
        elif math.isnan(factor) or factor <= 0.0:
            return self.zero()
        else:
            out = self.zero()
            out.entries = factor * self.entries
            return out.specialize()

    @inheritdoc(Container)
    def __rmul__(self, factor):
        return self.__mul__(factor)

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()

        if weight > 0.0:
            t = self.transform(weight)
            if not isinstance(t, numbers.Real):
                raise TypeError("function return value ({0}) must be boolean or number".format(t))

            # no possibility of exception from here on out (for rollback)
            self.entries += t

    def _cppGenerateCode(self, parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes,
                         derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode, fillPrefix,
                         fillIndent, weightVars, weightVarStack, tmpVarTypes):
        return self._c99GenerateCode(parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes,
                                     derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode,
                                     fillPrefix, fillIndent, weightVars, weightVarStack, tmpVarTypes)

    def _c99GenerateCode(self, parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes,
                         derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode, fillPrefix,
                         fillIndent, weightVars, weightVarStack, tmpVarTypes):
        initCode.append(" " * initIndent + self._c99ExpandPrefix(*initPrefix) + " = 0.0;")
        if self.transform is not identity:
            normexpr = self._c99QuantityExpr(parser,
                                             generator,
                                             inputFieldNames,
                                             inputFieldTypes,
                                             derivedFieldTypes,
                                             derivedFieldExprs,
                                             weightVarStack[-1])
            fillCode.append(" " * fillIndent + self._c99ExpandPrefix(*fillPrefix) + " += " + normexpr + ";")
        else:
            fillCode.append(" " * fillIndent + self._c99ExpandPrefix(*fillPrefix) + " += " + weightVarStack[-1] + ";")

    def _clingUpdate(self, filler, *extractorPrefix):
        self.entries += self._clingExpandPrefix(filler, *extractorPrefix)

    def _c99StorageType(self):
        return "double"

    def _c99StructName(self):
        return "Ct"

    def _cudaGenerateCode(self, parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes,
                          derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode, fillPrefix,
                          fillIndent, combineCode, totalPrefix, itemPrefix, combineIndent, jsonCode, jsonPrefix,
                          jsonIndent, weightVars, weightVarStack, tmpVarTypes, suppressName):
        initCode.append(" " * initIndent + self._c99ExpandPrefix(*initPrefix) + " = 0.0f;")

        if self.transform is not identity:
            normexpr = self._cudaQuantityExpr(parser,
                                              generator,
                                              inputFieldNames,
                                              inputFieldTypes,
                                              derivedFieldTypes,
                                              derivedFieldExprs,
                                              weightVarStack[-1])
            fillCode.append(
                " " *
                fillIndent +
                "atomicAdd(&" +
                self._c99ExpandPrefix(
                    *
                    fillPrefix) +
                ", " +
                normexpr +
                ");")
        else:
            fillCode.append(" " * fillIndent + "atomicAdd(&" + self._c99ExpandPrefix(*fillPrefix) +
                            ", " + weightVarStack[-1] + ");")

        combineCode.append(
            " " *
            combineIndent +
            "atomicAdd(&" +
            self._c99ExpandPrefix(
                *
                totalPrefix) +
            ", " +
            self._c99ExpandPrefix(
                *
                itemPrefix) +
            ");")

        jsonCode.append(" " * jsonIndent + "floatToJson(out, " + self._c99ExpandPrefix(*jsonPrefix) + ");")

    def _cudaUnpackAndFill(self, data, bigendian, alignment):
        format = "<f"
        entries, = struct.unpack(format, data[:struct.calcsize(format)])
        self.entries += entries
        return data[struct.calcsize(format):]

    def _cudaStorageType(self):
        return "float"

    def _numpy(self, data, weights, shape):
        import numpy
        if isinstance(weights, numpy.ndarray):
            assert len(weights.shape) == 1
            if shape[0] is not None:
                assert weights.shape[0] == shape[0]

            if self.transform is identity:
                self.entries += float(weights.sum())
            else:
                t = self.transform(weights)
                assert len(t.shape) == 1
                if shape[0] is not None:
                    assert t.shape[0] == shape[0]
                self.entries += float(t.sum())

        elif shape[0] is not None:
            if self.transform is identity:
                self.entries += weights * shape[0]
            else:
                t = self.transform(numpy.array([weights]))
                assert len(t.shape) == 1
                assert t.shape[0] == 1
                self.entries += float(t[0])

        else:
            raise ValueError("cannot use Numpy to fill an isolated Count (unless the weights are given as an array)")

    def _sparksql(self, jvm, converter):
        return converter.Count()   # TODO: handle transform

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return []

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName):
        return floatToJson(self.entries)

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if json in ("nan", "inf", "-inf") or isinstance(json, numbers.Real):
            return Count.ed(float(json))
        else:
            raise JsonFormatException(json, "Count")

    def __repr__(self):
        return "<Count {0}>".format(self.entries)

    def __eq__(self, other):
        return isinstance(other, Count) and numeq(self.entries, other.entries) and self.transform == other.transform

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.entries, self.transform))


# extra properties: number of dimensions and datatypes of sub-hists
Count.n_dim = n_dim
Count.datatype = datatype

# register extra methods
Factory.register(Count)
