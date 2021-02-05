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

from histogrammar.defs import Container, Factory, identity, JsonFormatException, ContainerException
from histogrammar.util import n_dim, datatype, serializable, inheritdoc, maybeAdd, floatToJson, hasKeys, numeq, \
    floatOrNan, rangeToJson, basestring, xrange


class Bag(Factory, Container):
    """Accumulate raw numbers, vectors of numbers, or strings, with identical values merged.

    A bag is the appropriate data type for scatter plots: a container that collects raw values, maintaining
    multiplicity but not order. (A "bag" is also known as a "multiset.") Conceptually, it is a mapping from distinct
    raw values to the number of observations: when two instances of the same raw value are observed, one key is stored
    and their weights add.

    Although the user-defined function may return scalar numbers, fixed-dimension vectors of numbers, or categorical
    strings, it may not mix range types. For the purposes of Label and Index (which can only collect aggregators of
    a single type), bags with different ranges are different types.
    """

    @staticmethod
    def ed(entries, values, range):
        """Create a Bag that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            values (dict from float, tuple of floats, or str to float): the number of entries for each unique item.
            range ("N", "N#" where "#" is a positive integer, or "S"): the data type: number, vector of numbers,
                or string.
        """

        if not isinstance(entries, numbers.Real) and entries not in ("nan", "inf", "-inf"):
            raise TypeError("entries ({0}) must be a number".format(entries))
        if not isinstance(values, dict) and not all(isinstance(k, numbers.Real) for k, v in values.items()):
            raise TypeError("values ({0}) must be a dict from numbers to range type".format(values))
        if float(entries) < 0.0:
            raise ValueError("entries ({0}) cannot be negative".format(entries))
        out = Bag(None, range)
        out.entries = float(entries)
        out.values = values
        return out.specialize()

    @staticmethod
    def ing(quantity):
        """Synonym for ``__init__``."""
        return Bag(quantity)

    def __init__(self, quantity=identity, range="S"):
        """Create a Bag that is capable of being filled and added.

        Parameters:
            quantity (function returning a float, a tuple of floats, or a str): computes the quantity of interest from
            the data.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
            values (dict from quantity return type to float): the number of entries for each unique item.
            range ("N", "N#" where "#" is a positive integer, or "S"): the data type: number, vector of numbers,
            or string. default is "S".
        """
        self.quantity = serializable(identity(quantity) if isinstance(quantity, str) else quantity)
        self.entries = 0.0
        self.values = {}
        self.range = range
        try:
            self.dimension = int(range[1:])
        except BaseException:
            self.dimension = 0
        super(Bag, self).__init__()
        self.specialize()

    @inheritdoc(Container)
    def zero(self):
        return Bag(self.quantity, self.range)

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Bag):
            if self.range != other.range:
                raise ContainerException(
                    "cannot add Bag because range differs ({0} vs {1})".format(
                        self.range, other.range))

            out = Bag(self.quantity, self.range)

            out.entries = self.entries + other.entries

            out.values = dict(self.values)
            for value, count in other.values.items():
                if value in out.values:
                    out.values[value] += count
                else:
                    out.values[value] = count

            return out.specialize()

        else:
            raise ContainerException("cannot add {0} and {1}".format(self.name, other.name))

    @inheritdoc(Container)
    def __iadd__(self, other):
        self.entries = other.entries
        self.values = other.values
        return self

    @inheritdoc(Container)
    def __mul__(self, factor):
        if math.isnan(factor) or factor <= 0.0:
            return self.zero()
        else:
            out = self.zero()
            out.entries = factor * self.entries
            for value, count in self.values.items():
                out.values[value] = factor * count
            return out.specialize()

    @inheritdoc(Container)
    def __rmul__(self, factor):
        return self.__mul__(factor)

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()

        if weight > 0.0:
            q = self.quantity(datum)
            self._update(q, weight)

    def _update(self, q, weight):
        if self.range == "S":
            if not isinstance(q, basestring):
                raise TypeError("function return value ({0}) must be a string for range {1}".format(q, self.range))

        elif self.range == "N":
            try:
                q = floatOrNan(q)
            except BaseException:
                raise TypeError("function return value ({0}) must be a number for range {1}".format(q, self.range))

        else:
            try:
                q = tuple(floatOrNan(qi) for qi in q)
                assert len(q) == self.dimension
            except BaseException:
                raise TypeError(
                    "function return value ({0}) must be a list/tuple of numbers with length {1} for range {2}".format(
                        q, self.dimension, self.range))

        # no possibility of exception from here on out (for rollback)
        self.entries += weight
        if q in self.values:
            self.values[q] += weight
        else:
            self.values[q] = weight

    def _cppGenerateCode(self, parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes,
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
        initCode.append(" " * initIndent + self._c99ExpandPrefix(*initPrefix) + ".values.clear();")
        fillCode.append(" " * fillIndent + self._c99ExpandPrefix(*fillPrefix) +
                        ".entries += " + weightVarStack[-1] + ";")

        fillCode.append("""{indent}if ({values}.find({q}) == {values}.end())
{indent}  {values}[{q}] = 0.0;
{indent}{values}[{q}] += {weight};""".format(
            indent=" " * fillIndent,
            values=self._c99ExpandPrefix(*fillPrefix) + ".values",
            q=normexpr,
            weight=weightVarStack[-1]
        ))

        if self.range[0] == "N" and len(self.range) > 1:
            storageStructs[self.range] = """
  class {0} {{
    public:
      double {1};
      {0}({2}): {3} {{ }}
      {0}(const {0}& other): {4} {{ }}
      {0}(): {5} {{ }}
      Bool_t operator<(const {0}& other) const {{
        {6}
        else return false;
      }}
  }};
""".format(self.range,
                ", ".join("v" + str(i) for i in xrange(self.dimension)),
                ", ".join("double v" + str(i) for i in xrange(self.dimension)),
                ", ".join("v" + str(i) + "(v" + str(i) + ")" for i in xrange(self.dimension)),
                ", ".join("v" + str(i) + "(other.v" + str(i) + ")" for i in xrange(self.dimension)),
                ", ".join("v" + str(i) + "(0.0)" for i in xrange(self.dimension)),
                "\n        ".join(
                    ("else " if i != 0 else "") +
                    "if (v" +
                    str(i) +
                    " < other.v" +
                    str(i) +
                    ") return true;" for i in xrange(
                        self.dimension))
           )

        storageStructs[self._c99StructName()] = """
  typedef struct {{
    double entries;
    std::map<{1}, double> values;
    double getValues({1} i) {{ return values[i]; }}
  }} {0};
""".format(self._c99StructName(), "double" if self.range == "N" else "std::string" if self.range == "S" else self.range)

    def _c99GenerateCode(self, parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes,
                         derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode, fillPrefix,
                         fillIndent, weightVars, weightVarStack, tmpVarTypes):
        raise NotImplementedError("no C99-compliant implementation of Bag (only C++)")

    def _clingUpdate(self, filler, *extractorPrefix):
        obj = self._clingExpandPrefix(filler, *extractorPrefix)
        self.entries += obj.entries

        for i in obj.values:
            key = i.first
            if self.range[0] == "N" and len(self.range) > 1:
                key = tuple(getattr(key, "v" + str(x)) for x in xrange(self.dimension))
            if key not in self.values:
                self.values[key] = 0.0
            self.values[key] += i.second

    def _c99StructName(self):
        return "Bg" + self.range + "_"

    def _numpy(self, data, weights, shape):
        import numpy
        q = self.quantity(data)
        assert isinstance(q, numpy.ndarray)
        if shape[0] is None:
            shape[0] = q.shape[0]
        else:
            assert q.shape[0] == shape[0]

        self._checkNPWeights(weights, shape)
        weights = self._makeNPWeights(weights, shape)

        for x, w in zip(q, weights):
            if w > 0.0:
                if isinstance(x, numpy.ndarray):
                    x = x.tolist()
                self._update(x, float(w))

    def _sparksql(self, jvm, converter):
        return converter.Bag(self.quantity.asSparkSQL(), self.range)

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return []

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName):
        if self.range == "N":
            aslist = sorted(x for x in self.values.items() if x[0] != "nan")
            if "nan" in self.values:
                aslist.append(("nan", self.values["nan"]))

        elif self.range[0] == "N":
            class Sorter(object):
                def __init__(self, x):
                    self.x = x

                def __lt__(self, other):
                    for xi, yi in zip(self.x, other.x):
                        if isinstance(xi, str) and isinstance(yi, float):
                            return False
                        elif isinstance(xi, float) and isinstance(yi, str):
                            return True
                        elif xi < yi:
                            return True
                        elif xi > yi:
                            return False
                    return False
            aslist = sorted((x for x in self.values.items()), key=lambda y: tuple(Sorter(z) for z in y))

        else:
            aslist = sorted(x for x in self.values.items())

        return maybeAdd({
            "entries": floatToJson(self.entries),
            "values": [{"w": floatToJson(n), "v": rangeToJson(v)} for v, n in aslist],
            "range": self.range,
        }, name=(None if suppressName else self.quantity.name))

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "values", "range"], ["name"]):
            if json["entries"] in ("nan", "inf", "-inf") or isinstance(json["entries"], numbers.Real):
                entries = json["entries"]
            else:
                raise JsonFormatException(json["entries"], "Bag.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Bag.name")

            if json["values"] is None:
                values = None

            elif json["values"] is None or isinstance(json["values"], list):
                values = {}
                for i, nv in enumerate(json["values"]):
                    if isinstance(nv, dict) and hasKeys(nv.keys(), ["w", "v"]):
                        if nv["w"] in ("nan", "inf", "-inf") or isinstance(nv["w"], numbers.Real):
                            n = float(nv["w"])
                        else:
                            raise JsonFormatException(nv["w"], "Bag.values {0} n".format(i))

                        if nv["v"] in ("nan", "inf", "-inf") or isinstance(nv["v"], numbers.Real):
                            v = floatOrNan(nv["v"])
                        elif isinstance(nv["v"], basestring):
                            v = nv["v"]
                        elif isinstance(nv["v"], (list, tuple)):
                            for j, d in enumerate(nv["v"]):
                                if d not in ("nan", "inf", "-inf") and not isinstance(d, numbers.Real):
                                    raise JsonFormatException(d, "Bag.values {0} v {1}".format(i, j))
                            v = tuple(map(floatOrNan, nv["v"]))
                        else:
                            raise JsonFormatException(nv["v"], "Bag.values {0} v".format(i))

                        values[v] = n

                    else:
                        raise JsonFormatException(nv, "Bag.values {0}".format(i))

            elif json["values"] is None:
                values = None

            else:
                raise JsonFormatException(json["values"], "Bag.values")

            if isinstance(json["range"], basestring):
                range = json["range"]
            else:
                raise JsonFormatException(json["range"], "Bag.range")

            out = Bag.ed(entries, values, range)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        else:
            raise JsonFormatException(json, "Bag")

    def __repr__(self):
        return "<Bag size={0} range={1}>".format(len(self.values), self.range)

    def __eq__(self, other):
        if len(self.values) != len(other.values):
            return False

        if self.range != other.range:
            return False

        if self.range == "N":
            one = sorted(x for x in self.values.items() if x[0] != "nan") + [("nan", self.values.get("nan"))]
            two = sorted(x for x in other.values.items() if x[0] != "nan") + [("nan", other.values.get("nan"))]

        elif self.range[0] == "N":
            class Sorter(object):
                def __init__(self, x):
                    self.x = x

                def __lt__(self, other):
                    for xi, yi in zip(self.x, other.x):
                        if isinstance(xi, str) and isinstance(yi, float):
                            return False
                        elif isinstance(xi, float) and isinstance(yi, str):
                            return True
                        elif xi < yi:
                            return True
                        elif xi > yi:
                            return False
                    return False
            one = sorted((x for x in self.values.items()), key=lambda y: tuple(Sorter(z) for z in y))
            two = sorted((x for x in other.values.items()), key=lambda y: tuple(Sorter(z) for z in y))

        else:
            one = sorted(x for x in self.values.items())
            two = sorted(x for x in other.values.items())

        for (v1, w1), (v2, w2) in zip(one, two):
            if isinstance(v1, basestring) and isinstance(v2, basestring):
                if v1 != v2:
                    return False
            elif isinstance(v1, numbers.Real) and isinstance(v2, numbers.Real):
                if not numeq(v1, v2):
                    return False
            elif isinstance(v1, tuple) and isinstance(v2, tuple) and len(v1) == len(v2):
                for v1i, v2i in zip(v1, v2):
                    if isinstance(v1i, numbers.Real) and isinstance(v2i, numbers.Real):
                        if not numeq(v1i, v2i):
                            return False
                    elif isinstance(v1i, basestring) and isinstance(v2i, basestring):
                        if v1i != v2i:
                            return False
                    else:
                        return False
            else:
                return False

            if v1 == "nan" and v2 == "nan" and w1 is None and w2 is None:
                pass
            elif isinstance(w1, numbers.Real) and isinstance(w2, numbers.Real):
                if not numeq(w1, w2):
                    return False
            else:
                return False

        return isinstance(other, Bag) and self.quantity == other.quantity and numeq(self.entries, other.entries)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.quantity, self.entries, tuple(self.values.items()), self.range))


# extra properties: number of dimensions and datatypes of sub-hists
Bag.n_dim = n_dim
Bag.datatype = datatype

# register extra methods such as plotting
Factory.register(Bag)
