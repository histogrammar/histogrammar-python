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

################################################################ Select

class Select(Factory, Container):
    """Filter or weight data according to a given selection.

    This primitive is a basic building block, intended to be used in conjunction with anything that needs a user-defined cut. In particular, a standard histogram often has a custom selection, and this can be built by nesting Select -> Bin -> Count.

    Select also resembles :doc:`Fraction <histogrammar.primitives.fraction.Fraction>`, but without the ``denominator``.

    The efficiency of a cut in a Select aggregator named ``x`` is simply ``x.cut.entries / x.entries`` (because all aggregators have an ``entries`` member).
    """

    @staticmethod
    def ed(entries, cut):
        """Create a Select that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            cut (:doc:`Container <histogrammar.defs.Container>`): the filled sub-aggregator.
        """
        if not isinstance(entries, numbers.Real) and entries not in ("nan", "inf", "-inf"):
            raise TypeError("entries ({0}) must be a number".format(entries))
        if not isinstance(cut, Container):
            raise TypeError("cut ({0}) must be a Container".format(cut))
        if entries < 0.0:
            raise ValueError("entries ({0}) cannot be negative".format(entries))
        out = Select(None, cut)
        out.entries = float(entries)
        return out.specialize()

    @staticmethod
    def ing(quantity, cut):
        """Synonym for ``__init__``."""
        return Select(quantity, cut)

    def __getattr__(self, attr):
        """Pass on searches for custom methods to the ``value``, so that Limit becomes effectively invisible."""
        if attr.startswith("__") and attr.endswith("__"):
            return getattr(Select, attr)
        elif attr not in self.__dict__ and hasattr(self.__dict__["cut"], attr):
            return getattr(self.__dict__["cut"], attr)
        else:
            return self.__dict__[attr]

    def __init__(self, quantity, cut):
        """Create a Select that is capable of being filled and added.

        Parameters:
            quantity (function returning bool or float): computes the quantity of interest from the data and interprets it as a selection (multiplicative factor on weight).
            cut (:doc:`Container <histogrammar.defs.Container>`): will only be filled with data that pass the cut, and which are weighted by the cut.

        Other Parameters:
            entries (float): the number of entries, initially 0.0.
        """
        if not isinstance(cut, Container):
            raise TypeError("cut ({0}) must be a Container".format(cut))
        self.entries = 0.0
        self.quantity = serializable(quantity)
        self.cut = cut
        super(Select, self).__init__()
        self.specialize()

    def fractionPassing(self):
        """Fraction of weights that pass the quantity."""
        return self.cut.entries / self.entries

    @inheritdoc(Container)
    def zero(self):
        return Select(self.quantity, self.cut.zero())

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Select):
            out = Select(self.quantity, self.cut + other.cut)
            out.entries = self.entries + other.entries
            return out.specialize()
        else:
            raise ContainerException("cannot add {0} and {1}".format(self.name, other.name))

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()

        if weight > 0.0:
            w = self.quantity(datum)
            if not isinstance(w, numbers.Real):
                raise TypeError("function return value ({0}) must be boolean or number".format(w))
            w *= weight

            if w > 0.0:
                self.cut.fill(datum, w)
            # no possibility of exception from here on out (for rollback)
            self.entries += weight

    def _clingGenerateCode(self, parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, storageStructs, initCode, prefix, initIndent, fillCode, fillIndent, weightVars, weightVarStack, tmpVarTypes):
        initCode.append(" " * initIndent + self._clingExpandPrefixCpp(*prefix) + ".entries = 0.0;")

        normexpr = self._clingQuantityExpr(parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, None)

        fillCode.append(" " * fillIndent + self._clingExpandPrefixCpp(*prefix) + ".entries += " + weightVars[-1] + ";")
        fillCode.append(" " * fillIndent + """if (!isnan({0})  &&  {0} > 0.0) {{""".format(normexpr))

        weightVars.append("weight_" + str(len(weightVars)))
        weightVarStack = weightVarStack + (weightVars[-1],)
        fillCode.append(" " * (fillIndent + 2) + """{0} = {1} * {2};""".format(weightVarStack[-1], weightVarStack[-2], normexpr))
        self.cut._clingGenerateCode(parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, storageStructs, initCode, prefix + (("var", "cut"),), initIndent, fillCode, fillIndent + 2, weightVars, weightVarStack, tmpVarTypes)

        fillCode.append(" " * fillIndent + "}")

        storageStructs[self._clingStructName()] = """
  typedef struct {{
    double entries;
    {1} cut;
  }} {0};
""".format(self._clingStructName(), self.cut._clingStorageType())

    def _clingUpdate(self, filler, *extractorPrefix):
        obj = self._clingExpandPrefixPython(filler, *extractorPrefix)
        self.entries += obj.entries
        self.cut._clingUpdate(obj, ("var", "cut"))

    def _clingStructName(self):
        return "Se" + self.cut._clingStructName()

    def _numpy(self, data, weights, shape):
        w = self.quantity(data)
        self._checkNPQuantity(w, shape)
        self._checkNPWeights(weights, shape)
        weights = self._makeNPWeights(weights, shape)

        import numpy
        w = w * weights
        w[numpy.isnan(w)] = 0.0
        w[w < 0.0] = 0.0

        self.cut._numpy(data, w, shape)

        # no possibility of exception from here on out (for rollback)
        self.entries += float(weights.sum())

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return [self.cut]

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName): return maybeAdd({
        "entries": floatToJson(self.entries),
        "type": self.cut.name,
        "data": self.cut.toJsonFragment(False),
        }, name=(None if suppressName else self.quantity.name))

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "type", "data"], ["name"]):
            if json["entries"] in ("nan", "inf", "-inf") or isinstance(json["entries"], numbers.Real):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "Select.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Select.name")

            if isinstance(json["type"], basestring):
                factory = Factory.registered[json["type"]]
            else:
                raise JsonFormatException(json, "Select.type")

            cut = factory.fromJsonFragment(json["data"], None)

            out = Select.ed(entries, cut)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        else:
            raise JsonFormatException(json, "Select")

    def __repr__(self):
        return "<Select cut={0}>".format(self.cut.name)

    def __eq__(self, other):
        return isinstance(other, Select) and numeq(self.entries, other.entries) and self.cut == other.cut

    def __ne__(self, other): return not self == other

    def __hash__(self):
        return hash((self.entries, self.cut))

Factory.register(Select)

