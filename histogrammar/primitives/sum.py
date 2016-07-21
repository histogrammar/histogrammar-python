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

import numbers

from histogrammar.defs import *
from histogrammar.util import *

class Sum(Factory, Container):
    """Accumulate the (weighted) sum of a given quantity, calculated from the data.

    Sum differs from :doc:`Count <histogrammar.primitives.count.Count>` in that it computes a quantity on the spot, rather than percolating a product of weight metadata from nested primitives. Also unlike weights, the sum can add both positive and negative quantities (weights are always non-negative).
    """

    @staticmethod
    def ed(entries, sum):
        """Create a Sum that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            sum (float): the sum.
        """
        if not isinstance(entries, numbers.Real) and entries not in ("nan", "inf", "-inf"):
            raise TypeError("entries ({0}) must be a number".format(entries))
        if not isinstance(sum, numbers.Real) and entries not in ("nan", "inf", "-inf"):
            raise TypeError("sum ({0}) must be a number".format(sum))
        if entries < 0.0:
            raise ValueError("entries ({0}) cannot be negative".format(entries))
        out = Sum(None)
        out.entries = float(entries)
        out.sum = float(sum)
        return out.specialize()

    @staticmethod
    def ing(quantity):
        """Synonym for ``__init__``."""
        return Sum(quantity)

    def __init__(self, quantity):
        """Create a Sum that is capable of being filled and added.

        Parameters:
            quantity (function returning float): computes the quantity of interest from the data.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
            sum (float): the running sum, initially 0.0.
        """
        self.quantity = serializable(quantity)
        self.entries = 0.0
        self.sum = 0.0
        super(Sum, self).__init__()
        self.specialize()

    @inheritdoc(Container)
    def zero(self): return Sum(self.quantity)

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Sum):
            out = Sum(self.quantity)
            out.entries = self.entries + other.entries
            out.sum = self.sum + other.sum
            return out.specialize()
        else:
            raise ContainerException("cannot add {0} and {1}".format(self.name, other.name))

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0, method=None):
        self._checkForCrossReferences()

        if weight > 0.0:
            q = self.quantity(datum)
            if not isinstance(q, numbers.Real):
                raise TypeError("function return value ({0}) must be boolean or number".format(q))

            # no possibility of exception from here on out (for rollback)
            self.entries += weight
            self.sum += q * weight

    def _clingGenerateCode(self, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode, fillIndent):
        if not isinstance(self.quantity.expr, basestring):
            raise ContainerException("Sum.quantity must be provided as a C++ string to use with Cling")

        storageStructs[self._clingStructName()] = """
  typedef struct {0} {{
    double entries;
    double sum;
  }} {0};
""".format(self._clingStructName())

        initCode.append(" " * initIndent + self._clingExpandPrefixCpp(initPrefix) + ".entries = 0.0;")
        initCode.append(" " * initIndent + self._clingExpandPrefixCpp(initPrefix) + ".sum = 0.0;")

        normexpr = self._clingNormalizeTTreeExpr(inputFieldNames, inputFieldTypes, self.quantity.expr)
        if self._clingInputFieldRef(self.quantity.expr):
            fillCode.append(" " * fillIndent + self._clingExpandPrefixCpp(initPrefix) + ".entries += weight;")
            fillCode.append(" " * fillIndent + self._clingExpandPrefixCpp(initPrefix) + ".sum += " + normexpr + ";")

        else:
            derivedFieldName = None
            for name, expr in derivedFieldExprs.items():
                if expr == normexpr:
                    derivedFieldName = name
                    break
            if derivedFieldName is None:
                derivedFieldName = "quantity_" + str(len(derivedFieldExprs))
                derivedFieldExprs[derivedFieldName] = normexpr
                derivedFieldTypes[derivedFieldName] = "double"
            fillCode.append(" " * fillIndent + self._clingExpandPrefixCpp(initPrefix) + ".entries += weight;")
            fillCode.append(" " * fillIndent + self._clingExpandPrefixCpp(initPrefix) + ".sum += " + derivedFieldName + ";")

    def _clingUpdate(self, filler, extractorPrefix):
        obj = self._clingExpandPrefixPython(filler, extractorPrefix)
        self.entries += obj.entries
        self.sum += obj.sum

    def _clingStorageType(self):
        return self._clingStructName()

    def _clingStructName(self):
        return "Sm"

    def _numpy(self, data, weights, shape):
        q = self.quantity(data)
        self._checkNPQuantity(q, shape)
        self._checkNPWeights(weights, shape)
        weights = self._makeNPWeights(weights, shape)

        # no possibility of exception from here on out (for rollback)
        self.entries += float(weights.sum())

        import numpy
        selection = numpy.isnan(q)
        numpy.bitwise_not(selection, selection)
        numpy.bitwise_and(selection, weights > 0.0, selection)
        q = q[selection]
        weights = weights[selection]
        q *= weights

        self.sum += float(q.sum())

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return []

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName): return maybeAdd({
        "entries": floatToJson(self.entries),
        "sum": floatToJson(self.sum),
        }, name=(None if suppressName else self.quantity.name))

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "sum"], ["name"]):
            if json["entries"] in ("nan", "inf", "-inf") or isinstance(json["entries"], numbers.Real):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json["entries"], "Sum.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Sum.name")

            if json["sum"] in ("nan", "inf", "-inf") or isinstance(json["sum"], numbers.Real):
                sum = float(json["sum"])
            else:
                raise JsonFormatException(json["sum"], "Sum.sum")

            out = Sum.ed(entries, sum)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        else:
            raise JsonFormatException(json, "Sum")
        
    def __repr__(self):
        return "<Sum sum={0}>".format(self.sum)

    def __eq__(self, other):
        return isinstance(other, Sum) and self.quantity == other.quantity and numeq(self.entries, other.entries) and numeq(self.sum, other.sum)

    def __ne__(self, other): return not self == other

    def __hash__(self):
        return hash((self.quantity, self.entries, self.sum))

Factory.register(Sum)
