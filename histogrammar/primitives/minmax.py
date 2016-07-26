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
from histogrammar.util import *

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

    def __init__(self, quantity):
        """Create a Minimize that is capable of being filled and added.

        Parameters:
            quantity (function returning float): computes the quantity of interest from the data.

        Other parameters:
            entries (float): the number of entries, initially 0.0. # 
            min (float): the lowest value of the quantity observed, initially NaN.
        """
        self.quantity = serializable(quantity)
        self.entries = 0.0
        self.min = float("nan")
        super(Minimize, self).__init__()
        self.specialize()

    @inheritdoc(Container)
    def zero(self): return Minimize(self.quantity)

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Minimize):
            out = Minimize(self.quantity)
            out.entries = self.entries + other.entries
            out.min = minplus(self.min, other.min)
            return out.specialize()
        else:
            raise ContainerException("cannot add {0} and {1}".format(self.name, other.name))

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

    def _clingGenerateCode(self, parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode, fillPrefix, fillIndent, weightVars, weightVarStack, tmpVarTypes):
        initCode.append(" " * initIndent + self._clingExpandPrefixCpp(*initPrefix) + ".entries = 0.0;")
        initCode.append(" " * initIndent + self._clingExpandPrefixCpp(*initPrefix) + ".min = NAN;")

        normexpr = self._clingQuantityExpr(parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, None)
        fillCode.append(" " * fillIndent + self._clingExpandPrefixCpp(*fillPrefix) + ".entries += " + weightVarStack[-1] + ";")

        fillCode.append(" " * fillIndent + "if (std::isnan({min})  ||  {q} < {min}) {min} = {q};".format(
            min = self._clingExpandPrefixCpp(*fillPrefix) + ".min",
            q = normexpr))

        storageStructs[self._clingStructName()] = """
  typedef struct {{
    double entries;
    double min;
  }} {0};
""".format(self._clingStructName())

    def _clingUpdate(self, filler, *extractorPrefix):
        obj = self._clingExpandPrefixPython(filler, *extractorPrefix)
        self.entries = self.entries + obj.entries
        self.min = minplus(self.min, obj.min)

    def _clingStructName(self):
        return "Mn"

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

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName): return maybeAdd({
        "entries": floatToJson(self.entries),
        "min": floatToJson(self.min),
        }, name=(None if suppressName else self.quantity.name))

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
        return isinstance(other, Minimize) and self.quantity == other.quantity and numeq(self.entries, other.entries) and numeq(self.min, other.min)

    def __ne__(self, other): return not self == other

    def __hash__(self):
        return hash((self.quantity, self.entries, self.min))

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

    def __init__(self, quantity):
        """Create a Maximize that is capable of being filled and added.

        Parameters:
            quantity (function returning float): computes the quantity of interest from the data.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
            max (float): the highest value of the quantity observed, initially NaN.
        """
        self.quantity = serializable(quantity)
        self.entries = 0.0
        self.max = float("nan")
        super(Maximize, self).__init__()
        self.specialize()

    @inheritdoc(Container)
    def zero(self): return Maximize(self.quantity)

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

    def _clingGenerateCode(self, parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode, fillPrefix, fillIndent, weightVars, weightVarStack, tmpVarTypes):
        initCode.append(" " * initIndent + self._clingExpandPrefixCpp(*initPrefix) + ".entries = 0.0;")
        initCode.append(" " * initIndent + self._clingExpandPrefixCpp(*initPrefix) + ".max = NAN;")

        normexpr = self._clingQuantityExpr(parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, None)
        fillCode.append(" " * fillIndent + self._clingExpandPrefixCpp(*fillPrefix) + ".entries += " + weightVarStack[-1] + ";")

        fillCode.append(" " * fillIndent + "if (std::isnan({max})  ||  {q} > {max}) {max} = {q};".format(
            max = self._clingExpandPrefixCpp(*fillPrefix) + ".max",
            q = normexpr))

        storageStructs[self._clingStructName()] = """
  typedef struct {{
    double entries;
    double max;
  }} {0};
""".format(self._clingStructName())

    def _clingUpdate(self, filler, *extractorPrefix):
        obj = self._clingExpandPrefixPython(filler, *extractorPrefix)
        self.entries = self.entries + obj.entries
        self.max = maxplus(self.max, obj.max)

    def _clingStructName(self):
        return "Mx"

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

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return []

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName): return maybeAdd({
        "entries": floatToJson(self.entries),
        "max": floatToJson(self.max),
        }, name=(None if suppressName else self.quantity.name))

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
        return isinstance(other, Maximize) and self.quantity == other.quantity and numeq(self.entries, other.entries) and numeq(self.max, other.max)

    def __ne__(self, other): return not self == other

    def __hash__(self):
        return hash((self.quantity, self.entries, self.max))

Factory.register(Maximize)
