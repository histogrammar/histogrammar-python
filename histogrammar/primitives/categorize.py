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
from histogrammar.primitives.count import *

class Categorize(Factory, Container):
    """Split a given quantity by its categorical value and fill only one category per datum.

    A bar chart may be thought of as a histogram with string-valued (categorical) bins, so this is the equivalent of :doc:`Bin <histogrammar.primitives.bin.Bin>` for bar charts. The order of the strings is deferred to the visualization stage.

    Unlike :doc:`SparselyBin <histogrammar.primitives.sparselybin.SparselyBin>`, this aggregator has the potential to use unlimited memory. A large number of *distinct* categories can generate many unwanted bins.
    """

    @staticmethod
    def ed(entries, contentType, pairsAsDict=None, **pairs):
        """Create a Categorize that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            contentType (str): the value's sub-aggregator type (must be provided to determine type for the case when `bins` is empty).
            pairs (dict from str to :doc:`Container <histogrammar.defs.Container>`): the non-empty bin categories and their values.
        """
        if not isinstance(entries, numbers.Real) and entries not in ("nan", "inf", "-inf"):
            raise TypeError("entries ({0}) must be a number".format(entries))
        if not isinstance(contentType, basestring):
            raise TypeError("contentType ({0}) must be a string".format(contentType))
        if not all(isinstance(k, basestring) and isinstance(v, Container) for k, v in pairs.items()):
            raise TypeError("pairs ({0}) must be a dict from strings to Containers".format(pairs))
        if entries < 0.0:
            raise ValueError("entries ({0}) cannot be negative".format(entries))

        out = Categorize(None, None)
        out.entries = float(entries)
        if pairsAsDict is None:
            out.pairs = {}
        else:
            out.pairs = pairsAsDict
        out.pairs.update(pairs)
        out.contentType = contentType
        return out.specialize()

    @staticmethod
    def ing(quantity, value=Count()):
        """Synonym for ``__init__``."""
        return Categorize(quantity, value)

    def __init__(self, quantity, value=Count()):
        """Create a Categorize that is capable of being filled and added.

        Parameters:
            quantity (function returning float): computes the quantity of interest from the data.
            value (:doc:`Container <histogrammar.defs.Container>`): generates sub-aggregators to put in each bin.

        Other Parameters:
            entries (float): the number of entries, initially 0.0.
            pairs (dict from str to :doc:`Container <histogrammar.defs.Container>`): the map, probably a hashmap, to fill with values when their `entries` become non-zero.
        """
        if value is not None and not isinstance(value, Container):
            raise TypeError("value ({0}) must be None or a Container".format(value))
        self.entries = 0.0
        self.quantity = serializable(quantity)
        self.value = value
        self.pairs = {}
        if value is not None:
            self.contentType = str(value.factory.name)
        super(Categorize, self).__init__()
        self.specialize()

    @property
    def pairsMap(self):
        """Input ``pairs`` as a key-value map."""
        return self.pairs

    @property
    def size(self):
        """Number of ``pairs``."""
        return len(self.pairs)

    @property
    def keys(self):
        """Iterable over the keys of the ``pairs``."""
        return self.pairs.keys()

    @property
    def values(self):
        """Iterable over the values of the ``pairs``."""
        return list(self.pairs.values())

    @property
    def keySet(self):
        """Set of keys among the ``pairs``."""
        return set(self.pairs.keys())

    def __call__(self, x):
        """Attempt to get key ``x``, throwing an exception if it does not exist."""
        return self.pairs[x]

    def get(self, x):
        """Attempt to get key ``x``, returning ``None`` if it does not exist."""
        return self.pairs.get(x)

    def getOrElse(self, x, default):
        """Attempt to get key ``x``, returning an alternative if it does not exist."""
        return self.pairs.get(x, default)

    @inheritdoc(Container)
    def zero(self): return Categorize(self.quantity, self.value)

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Categorize):
            out = Categorize(self.quantity, self.value)
            out.entries = self.entries + other.entries
            out.pairs = {}
            for k in self.keySet.union(other.keySet):
                if k in self.pairs and k in other.pairs:
                    out.pairs[k] = self.pairs[k] + other.pairs[k]
                elif k in self.pairs:
                    out.pairs[k] = self.pairs[k].copy()
                else:
                    out.pairs[k] = other.pairs[k].copy()
            return out.specialize()

        else:
            raise ContainerException("cannot add {0} and {1}".format(self.name, other.name))

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()

        if weight > 0.0:
            q = self.quantity(datum)
            if not isinstance(q, basestring):
                raise TypeError("function return value ({0}) must be a string".format(q))

            if q not in self.pairs:
                self.pairs[q] = self.value.zero()
            self.pairs[q].fill(datum, weight)

            # no possibility of exception from here on out (for rollback)
            self.entries += weight

    def _clingGenerateCode(self, parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode, fillPrefix, fillIndent, weightVars, weightVarStack, tmpVarTypes):
        normexpr = self._clingQuantityExpr(parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, None)

        initCode.append(" " * initIndent + self._clingExpandPrefixCpp(*initPrefix) + ".entries = 0.0;")
        initCode.append(" " * initIndent + self._clingExpandPrefixCpp(*initPrefix) + ".pairs.clear();")
        fillCode.append(" " * fillIndent + self._clingExpandPrefixCpp(*fillPrefix) + ".entries += " + weightVarStack[-1] + ";")

        value = "value_" + str(len(tmpVarTypes))
        tmpVarTypes[value] = self.value._clingStorageType() + "*"

        fillCode.append("""{indent}if ({pairs}.find({q}) == {pairs}.end())
{indent}  {pairs}[{q}] = {prototype};    // copy
{indent}{value} = &({pairs}[{q}]);    // reference""".format(
            indent = " " * fillIndent,
            q = normexpr,
            value = value,
            prototype = self._clingExpandPrefixCpp(*fillPrefix) + ".value",
            pairs = self._clingExpandPrefixCpp(*fillPrefix) + ".pairs"))

        self.value._clingGenerateCode(parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, storageStructs, initCode, initPrefix + (("var", "value"),), initIndent, fillCode, (("var", "(*" + value + ")"),), fillIndent, weightVars, weightVarStack, tmpVarTypes)

        storageStructs[self._clingStructName()] = """
  typedef struct {{
    double entries;
    {1} value;
    std::unordered_map<std::string, {1}> pairs;
    {1}& getValues(std::string i) {{ return pairs[i]; }}
  }} {0};
""".format(self._clingStructName(), self.value._clingStorageType())

    def _clingUpdate(self, filler, *extractorPrefix):
        obj = self._clingExpandPrefixPython(filler, *extractorPrefix)
        self.entries += obj.entries

        for i in obj.pairs:
            key = i.first
            if key not in self.pairs:
                self.pairs[key] = self.value.copy()
            self.pairs[key]._clingUpdate(obj, ("func", ["getValues", key]))

    def _clingStructName(self):
        return "Cz" + self.value._clingStructName()

    def _numpy(self, data, weights, shape):
        q = self.quantity(data)
        self._checkNPQuantity(q, shape)
        self._checkNPWeights(weights, shape)
        weights = self._makeNPWeights(weights, shape)

        # no possibility of exception from here on out (for rollback)
        for x, w in zip(q, weights):
            if w > 0.0:
                if x not in self.pairs:
                    self.pairs[x] = self.value.zero()
                self.pairs[x].fill(x, w)

        # no possibility of exception from here on out (for rollback)
        self.entries += float(weights.sum())

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return [self.value] + list(self.pairs.values())

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName):
        if isinstance(self.value, Container):
            if getattr(self.value, "quantity", None) is not None:
                binsName = self.value.quantity.name
            elif getattr(self.value, "quantityName", None) is not None:
                binsName = self.value.quantityName
            else:
                binsName = None
        elif len(self.pairs) > 0:
            if getattr(list(self.pairs.values())[0], "quantity", None) is not None:
                binsName = list(self.pairs.values())[0].quantity.name
            elif getattr(list(self.pairs.values())[0], "quantityName", None) is not None:
                binsName = list(self.pairs.values())[0].quantityName
            else:
                binsName = None
        else:
            binsName = None

        return maybeAdd({
            "entries": floatToJson(self.entries),
            "type": self.value.name if self.value is not None else self.contentType,
            "data": dict((k, v.toJsonFragment(True)) for k, v in self.pairs.items()),
            }, **{"name": None if suppressName else self.quantity.name,
                  "data:name": binsName})

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "type", "data"], ["name", "data:name"]):
            if json["entries"] in ("nan", "inf", "-inf") or isinstance(json["entries"], numbers.Real):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "Categorize.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Categorize.name")

            if isinstance(json["type"], basestring):
                contentType = json["type"]
                factory = Factory.registered[contentType]
            else:
                raise JsonFormatException(json, "Categorize.type")

            if isinstance(json.get("data:name", None), basestring):
                dataName = json["data:name"]
            elif json.get("data:name", None) is None:
                dataName = None
            else:
                raise JsonFormatException(json["data:name"], "Categorize.data:name")

            if isinstance(json["data"], dict):
                pairs = dict((k, factory.fromJsonFragment(v, dataName)) for k, v in json["data"].items())
            else:
                raise JsonFormatException(json, "Categorize.data")

            out = Categorize.ed(entries, contentType, **pairs)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        else:
            raise JsonFormatException(json, "Categorize")

    def __repr__(self):
        return "<Categorize values={0} size={1}".format(self.values[0].name if self.size > 0 else self.value.name if self.value is not None else self.contentType, self.size)

    def __eq__(self, other):
        return isinstance(other, Categorize) and numeq(self.entries, other.entries) and self.quantity == other.quantity and self.pairs == other.pairs

    def __ne__(self, other): return not self == other

    def __hash__(self):
        return hash((self.entries, self.quantity, tuple(sorted(self.pairs.items()))))

Factory.register(Categorize)
