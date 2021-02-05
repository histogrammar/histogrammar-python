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

from histogrammar.defs import Container, Factory, JsonFormatException, ContainerException
from histogrammar.util import inheritdoc, floatToJson, hasKeys, numeq, basestring, xrange


class Collection(object):
    def _c99CanonicalOrder(self, items):
        return sorted((v._c99StructName(), k, v) for k, v in items)

    def _c99StructName(self):
        letter = self.name[0]
        out = [letter, "_"]
        last = None
        count = 0
        for s, k, v in self._c99CanonicalOrder(self.pairs.items()):
            if s != last:
                if last is not None:
                    out.append(str(count))
                    out.append("_")
                out.append(s)
                count = 0
            count += 1
            last = s
        out.extend([str(count), "_", letter.lower()])
        return "".join(out)

    def _c99Struct(self):
        out = ["""
  typedef struct {
    """]
        last = None
        count = 0
        n = 0
        lastN = 0
        lastType = 0
        for s, k, v in self._c99CanonicalOrder(self.pairs.items()):
            if s != last:
                if last is not None:
                    out.append("[{0}];\n    ".format(count))
                    out.append("{0}& getSub{1}(int i) {{ return sub{1}[i]; }}\n    ".format(lastType, lastN))
                out.append("{0} sub{1}".format(s, n))
                lastType = s
                lastN = n
                n += 1
                count = 0
            count += 1
            last = s
        out.append(
            """[{0}];\n    {1}& getSub{2}(int i) {{ return sub{2}[i]; }}\n    double entries;\n  }} {3};""".format(
                count,
                lastType,
                lastN,
                self._c99StructName()))
        return "".join(out)

    def _cppGenerateCode(self, parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes,
                         derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode, fillPrefix,
                         fillIndent, weightVars, weightVarStack, tmpVarTypes):
        return self._c99GenerateCode(parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes,
                                     derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode,
                                     fillPrefix, fillIndent, weightVars, weightVarStack, tmpVarTypes)

    def _c99GenerateCode(self, parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes,
                         derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode, fillPrefix,
                         fillIndent, weightVars, weightVarStack, tmpVarTypes):
        last = None
        n = 0
        i = 0
        for s, k, v in self._c99CanonicalOrder(self.pairs.items()):
            if last is not None and s != last:
                n += 1
                i = 0
            v._c99GenerateCode(parser,
                               generator,
                               inputFieldNames,
                               inputFieldTypes,
                               derivedFieldTypes,
                               derivedFieldExprs,
                               storageStructs,
                               initCode,
                               initPrefix + (("var",
                                              "sub" + str(n)),
                                             ("index",
                                              i)),
                               initIndent,
                               fillCode,
                               fillPrefix + (("var",
                                              "sub" + str(n)),
                                             ("index",
                                              i)),
                               fillIndent,
                               weightVars,
                               weightVarStack,
                               tmpVarTypes)
            i += 1
            last = s

        initCode.append(" " * initIndent + self._c99ExpandPrefix(*initPrefix) + ".entries = 0.0;")
        fillCode.append(" " * fillIndent + self._c99ExpandPrefix(*fillPrefix) +
                        ".entries += " + weightVarStack[-1] + ";")

        storageStructs[self._c99StructName()] = self._c99Struct()

    def _clingUpdate(self, filler, *extractorPrefix):
        obj = self._clingExpandPrefix(filler, *extractorPrefix)
        self.entries += obj.entries
        last = None
        n = 0
        i = 0
        for s, k, v in self._c99CanonicalOrder(self.pairs.items()):
            if last is not None and s != last:
                n += 1
                i = 0
            v._clingUpdate(obj, ("func", ["getSub" + str(n), i]))
            i += 1
            last = s

    def _cudaStruct(self):
        out = ["""
  typedef struct {
    """]
        last = None
        n = 0
        count = 0
        for s, k, v in self._c99CanonicalOrder(self.pairs.items()):
            if s != last:
                if last is not None:
                    out.append("[{0}];\n    ".format(count))
                out.append("{0} sub{1}".format(s, n))
                n += 1
                count = 0
            count += 1
            last = s
        out.append("""[{0}];\n    float entries;\n  }} {1};""".format(count, self._c99StructName()))
        return "".join(out)

    def _cudaGenerateCode(self, parser, generator, inputFieldNames, inputFieldTypes, derivedFieldTypes,
                          derivedFieldExprs, storageStructs, initCode, initPrefix, initIndent, fillCode, fillPrefix,
                          fillIndent, combineCode, totalPrefix, itemPrefix, combineIndent, jsonCode, jsonPrefix,
                          jsonIndent, weightVars, weightVarStack, tmpVarTypes, suppressName):
        tmpJsonCode = []

        if isinstance(self, (Label, Index)):
            tmpJsonCode.append(
                " " *
                jsonIndent +
                "fprintf(out, \"{\\\"sub:type\\\": \\\"" +
                self.values[0].name +
                "\\\", \\\"data\\\": \");")
        else:
            tmpJsonCode.append(" " * jsonIndent + "fprintf(out, \"{\\\"data\\\": \");")

        if isinstance(self, (Label, UntypedLabel)):
            tmpJsonCode.append(" " * jsonIndent + "fprintf(out, \"{\");")
        else:
            tmpJsonCode.append(" " * jsonIndent + "fprintf(out, \"[\");")

        if isinstance(self, Branch):
            rightOrder = {}
            lastJsonCode = len(tmpJsonCode)

        last = None
        n = 0
        i = 0
        for s, k, v in self._c99CanonicalOrder(self.pairs.items()):
            if not isinstance(self, Branch) and last is not None:
                tmpJsonCode.append(" " * jsonIndent + "fprintf(out, \", \");")
            if isinstance(self, (Label, UntypedLabel)):
                tmpJsonCode.append(" " * jsonIndent + "fprintf(out, \"" + json.dumps(json.dumps(k))[1:-1] + ": \");")
            if isinstance(self, (UntypedLabel, Branch)):
                tmpJsonCode.append(
                    " " *
                    jsonIndent +
                    "fprintf(out, \"{\\\"type\\\": \\\"" +
                    v.name +
                    "\\\", \\\"data\\\": \");")

            if last is not None and s != last:
                n += 1
                i = 0
            v._cudaGenerateCode(parser,
                                generator,
                                inputFieldNames,
                                inputFieldTypes,
                                derivedFieldTypes,
                                derivedFieldExprs,
                                storageStructs,
                                initCode,
                                initPrefix + (("var",
                                               "sub" + str(n)),
                                              ("index",
                                               i)),
                                initIndent,
                                fillCode,
                                fillPrefix + (("var",
                                               "sub" + str(n)),
                                              ("index",
                                               i)),
                                fillIndent,
                                combineCode,
                                totalPrefix + (("var",
                                                "sub" + str(n)),
                                               ("index",
                                                i)),
                                itemPrefix + (("var",
                                               "sub" + str(n)),
                                              ("index",
                                               i)),
                                combineIndent,
                                tmpJsonCode,
                                jsonPrefix + (("var",
                                               "sub" + str(n)),
                                              ("index",
                                               i)),
                                jsonIndent,
                                weightVars,
                                weightVarStack,
                                tmpVarTypes,
                                suppressName)

            i += 1
            last = s

            if isinstance(self, (UntypedLabel, Branch)):
                tmpJsonCode.append(" " * jsonIndent + "fprintf(out, \"}\");")

            if isinstance(self, Branch):
                rightOrder[k] = tmpJsonCode[lastJsonCode:]
                tmpJsonCode = tmpJsonCode[:lastJsonCode]

        if isinstance(self, Branch):
            for k in xrange(len(rightOrder)):
                if k != 0:
                    tmpJsonCode.append(" " * jsonIndent + "fprintf(out, \", \");")
                tmpJsonCode.extend(rightOrder[k])

        if isinstance(self, (Label, UntypedLabel)):
            tmpJsonCode.append(" " * jsonIndent + "fprintf(out, \"}\");")
        else:
            tmpJsonCode.append(" " * jsonIndent + "fprintf(out, \"]\");")

        initCode.append(" " * initIndent + self._c99ExpandPrefix(*initPrefix) + ".entries = 0.0f;")
        fillCode.append(" " * fillIndent + "atomicAdd(&" + self._c99ExpandPrefix(*fillPrefix) + ".entries, " +
                        weightVarStack[-1] + ");")
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
        tmpJsonCode.append(" " * jsonIndent + "fprintf(out, \", \\\"entries\\\": \");")
        tmpJsonCode.append(" " * jsonIndent + "floatToJson(out, " + self._c99ExpandPrefix(*jsonPrefix) + ".entries);")
        tmpJsonCode.append(" " * jsonIndent + "fprintf(out, \"}\");")

        jsonCode.extend(tmpJsonCode)

        storageStructs[self._c99StructName()] = self._cudaStruct()

    def _cudaUnpackAndFill(self, data, bigendian, alignment):
        last = None
        n = 0
        i = 0
        for s, k, v in self._c99CanonicalOrder(self.pairs.items()):
            if last is not None and s != last:
                n += 1
                i = 0
            data = v._cudaUnpackAndFill(data, bigendian, alignment)
            i += 1
            last = s

        format = "<f"
        entries, = struct.unpack(format, data[:struct.calcsize(format)])
        self.entries += entries
        data = data[struct.calcsize(format):]
        return data

# Label


class Label(Factory, Container, Collection):
    """Accumulate any number of aggregators of the same type and label them with strings.

    Every sub-aggregator is filled with every input datum. This primitive simulates a directory of aggregators.
    For sub-directories, nest collections within the Label collection.

    Note that all sub-aggregators within a Label must have the *same type* (e.g. histograms of different binnings,
    but all histograms). To collect objects of *different types* with string-based look-up keys,
    use :doc:`UntypedLabel <histogrammar.primitives.collection.UntypedLabel>`.

    To collect aggregators of the *same type* without naming them,
    use :doc:`Index <histogrammar.primitives.collection.Index>`.
    To collect aggregators of *different types* without naming them,
    use :doc:`Branch <histogrammar.primitives.collection.Branch>`.

    In strongly typed languages, the restriction to a single type allows nested objects to be extracted without casting.
    """
    @staticmethod
    def ed(entries, pairsAsDict=None, **pairs):
        """Create a Label that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            pairs (list of str, :doc:`Container <histogrammar.defs.Container>` pairs): the collection of
                filled aggregators.
        """
        if not isinstance(entries, numbers.Real) and entries not in ("nan", "inf", "-inf"):
            raise TypeError("entries ({0}) must be a number".format(entries))
        if not all(isinstance(k, basestring) and isinstance(v, Container) for k, v in pairs.items()):
            raise TypeError("pairs ({0}) must be a dict from strings to Containers".format(pairs))
        if entries < 0.0:
            raise ValueError("entries ({0}) cannot be negative".format(entries))

        if pairsAsDict is None:
            pairsAsDict = {}
        pairsAsDict.update(pairs)
        out = Label(**pairsAsDict)
        out.entries = float(entries)
        return out.specialize()

    @staticmethod
    def ing(**pairs):
        """Synonym for ``__init__``."""
        return Label(**pairs)

    def __init__(self, **pairs):
        """Create a Label that is capable of being filled and added.

        Parameters:
            pairs (list of str, :doc:`Container <histogrammar.defs.Container>` pairs): the collection of aggregators
                to fill.

        Other Parameters:
            entries (float): the number of entries, initially 0.0.
        """
        if not all(isinstance(k, basestring) and isinstance(v, Container) for k, v in pairs.items()):
            raise TypeError("pairs ({0}) must be a dict from strings to Containers".format(pairs))
        if any(not isinstance(x, basestring) for x in pairs.keys()):
            raise ValueError("all Label keys must be strings")
        if len(pairs) < 1:
            raise ValueError("at least one pair required")

        contentType = list(pairs.values())[0].name
        if any(x.name != contentType for x in pairs.values()):
            raise ContainerException("all Label values must have the same type")
        if contentType == "Bag":
            rangeType = list(pairs.values())[0].range
            if any(x.range != rangeType for x in pairs.values()):
                raise ContainerException("all Label values must have the same type")

        self.entries = 0.0
        self.pairs = pairs

        super(Label, self).__init__()
        self.specialize()

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

    def __call__(self, x, *rest):
        """Attempt to get key ``index``, throwing an exception if it does not exist."""
        if len(rest) == 0:
            return self.pairs[x]
        else:
            return self.pairs[x](*rest)

    def get(self, x):
        """Attempt to get key ``x``, returning ``None`` if it does not exist."""
        return self.pairs.get(x, None)

    def getOrElse(self, x, default):
        """Attempt to get key ``x``, returning an alternative if it does not exist."""
        return self.pairs.get(x, default)

    @inheritdoc(Container)
    def zero(self):
        return Label(**dict((k, v.zero()) for k, v in self.pairs.items()))

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Label):
            if self.keySet != other.keySet:
                raise ContainerException(
                    "cannot add Labels because keys differ:\n    {0}\n    {1}".format(
                        ", ".join(
                            sorted(
                                self.keys)), ", ".join(
                            sorted(
                                other.keys))))

            out = Label(**dict((k, self(k) + other(k)) for k in self.keys))
            out.entries = self.entries + other.entries
            return out.specialize()

        else:
            raise ContainerException("cannot add {0} and {1}".format(self.name, other.name))

    @inheritdoc(Container)
    def __iadd__(self, other):
        if isinstance(other, Label):
            if self.keySet != other.keySet:
                raise ContainerException(
                    "cannot add Labels because keys differ:\n    {0}\n    {1}".format(
                        ", ".join(
                            sorted(
                                self.keys)), ", ".join(
                            sorted(
                                other.keys))))
            self.entries += other.entries
            for k in self.keys:
                v = self(k)
                v += other(k)
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
            for k, v in self.pairs.items():
                out.pairs[k] = v * factor
            return out.specialize()

    @inheritdoc(Container)
    def __rmul__(self, factor):
        return self.__mul__(factor)

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()

        if weight > 0.0:
            for x in self.values:
                x.fill(datum, weight)

            # no possibility of exception from here on out (for rollback)
            self.entries += weight

    def _numpy(self, data, weights, shape):
        if shape[0] is not None:
            self._checkNPWeights(weights, shape)
            weights = self._makeNPWeights(weights, shape)

        for x in self.values:
            x._numpy(data, weights, shape)

        # no possibility of exception from here on out (for rollback)
        import numpy
        if isinstance(weights, numpy.ndarray):
            self.entries += float(weights.sum())
        else:
            self.entries += float(weights * shape[0])

    def _sparksql(self, jvm, converter):
        return converter.Label([jvm.scala.Tuple2(k, v._sparksql(jvm, converter)) for k, v in self.pairs.items()])

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return self.values

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName):
        return {"entries": floatToJson(self.entries),
                "sub:type": self.values[0].name,
                "data": dict((k, v.toJsonFragment(False)) for k, v in self.pairs.items())
                }

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "sub:type", "data"]):
            if json["entries"] in ("nan", "inf", "-inf") or isinstance(json["entries"], numbers.Real):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "Label.entries")

            if isinstance(json["sub:type"], basestring):
                factory = Factory.registered[json["sub:type"]]
            else:
                raise JsonFormatException(json, "Label.sub:type")

            if isinstance(json["data"], dict):
                pairs = dict((k, factory.fromJsonFragment(v, None)) for k, v in json["data"].items())
            else:
                raise JsonFormatException(json, "Label.data")

            return Label.ed(entries, **pairs)

        else:
            raise JsonFormatException(json, "Label")

    def __repr__(self):
        return "<Label values={0} size={1}>".format(self.values[0].name, self.size)

    def __eq__(self, other):
        return isinstance(other, Label) and numeq(self.entries, other.entries) and self.pairs == other.pairs

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.entries, tuple(sorted(self.pairs.items()))))


Factory.register(Label)

# UntypedLabel


class UntypedLabel(Factory, Container, Collection):
    """Accumulate any number of aggregators of any type and label them with strings.

    Every sub-aggregator is filled with every input datum.

    This primitive simulates a directory of aggregators. For sub-directories, nest collections within the UntypedLabel.

    Note that sub-aggregators within an UntypedLabel may have *different types*. In strongly typed languages, this
    flexibility poses a problem: nested objects must be type-cast before they can be used. To collect objects of
    the *same type* with string-based look-up keys, use :doc:`Label <histogrammar.primitives.collection.Label>`.

    To collect aggregators of the *same type* without naming them,
    use :doc:`Index <histogrammar.primitives.collection.Index>`.
    To collect aggregators of *different types* without naming them,
    use :doc:`Branch <histogrammar.primitives.collection.Branch>`.
    """

    @staticmethod
    def ed(entries, pairsAsDict=None, **pairs):
        """Create an UntypedLabel that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            pairs (list of str, :doc:`Container <histogrammar.defs.Container>` pairs): the collection of filled
                aggregators.
        """
        if not isinstance(entries, numbers.Real) and entries not in ("nan", "inf", "-inf"):
            raise TypeError("entries ({0}) must be a number".format(entries))
        if not all(isinstance(k, basestring) and isinstance(v, Container) for k, v in pairs.items()):
            raise TypeError("pairs ({0}) must be a dict from strings to Containers".format(pairs))
        if entries < 0.0:
            raise ValueError("entries ({0}) cannot be negative".format(entries))

        if pairsAsDict is None:
            pairsAsDict = {}
        pairsAsDict.update(pairs)
        out = UntypedLabel(**pairsAsDict)
        out.entries = float(entries)
        return out.specialize()

    @staticmethod
    def ing(**pairs):
        """Synonym for ``__init__``."""
        return UntypedLabel(**pairs)

    def __init__(self, **pairs):
        """Create an UntypedLabel that is capable of being filled and added.

        Parameters:
            pairs (list of str, :doc:`Container <histogrammar.defs.Container>` pairs): the collection of aggregators
                to fill.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
        """
        if not all(isinstance(k, basestring) and isinstance(v, Container) for k, v in pairs.items()):
            raise TypeError("pairs ({0}) must be a dict from strings to Containers".format(pairs))

        self.entries = 0.0
        self.pairs = pairs

        super(UntypedLabel, self).__init__()
        self.specialize()

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

    def __call__(self, x, *rest):
        """Attempt to get key ``index``, throwing an exception if it does not exist."""
        if len(rest) == 0:
            return self.pairs[x]
        else:
            return self.pairs[x](*rest)

    def get(self, x):
        """Attempt to get key ``x``, returning ``None`` if it does not exist."""
        return self.pairs.get(x, None)

    def getOrElse(self, x, default):
        """Attempt to get key ``x``, returning an alternative if it does not exist."""
        return self.pairs.get(x, default)

    @inheritdoc(Container)
    def zero(self):
        return UntypedLabel(**dict((k, v.zero()) for k, v in self.pairs.items()))

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, UntypedLabel):
            if self.keySet != other.keySet:
                raise ContainerException(
                    "cannot add UntypedLabels because keys differ:\n    {0}\n    {1}".format(
                        ", ".join(
                            sorted(
                                self.keys)), ", ".join(
                            sorted(
                                other.keys))))

            out = UntypedLabel(**dict((k, self(k) + other(k)) for k in self.keys))
            out.entries = self.entries + other.entries
            return out.specialize()

        else:
            raise ContainerException("cannot add {0} and {1}".format(self.name, other.name))

    @inheritdoc(Container)
    def __iadd__(self, other):
        if isinstance(other, UntypedLabel):
            if self.keySet != other.keySet:
                raise ContainerException(
                    "cannot add UntypedLabels because keys differ:\n    {0}\n    {1}".format(
                        ", ".join(
                            sorted(
                                self.keys)), ", ".join(
                            sorted(
                                other.keys))))
            self.entries += other.entries
            for k in self.keys:
                v = self(k)
                v += other(k)
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
            for k, v in self.pairs.items():
                out.pairs[k] = v * factor
            return out.specialize()

    @inheritdoc(Container)
    def __rmul__(self, factor):
        return self.__mul__(factor)

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()

        if weight > 0.0:
            for x in self.values:
                x.fill(datum, weight)

            # no possibility of exception from here on out (for rollback)
            self.entries += weight

    def _numpy(self, data, weights, shape):
        if shape[0] is not None:
            self._checkNPWeights(weights, shape)
            weights = self._makeNPWeights(weights, shape)

        for x in self.values:
            x._numpy(data, weights, shape)

        # no possibility of exception from here on out (for rollback)
        import numpy
        if isinstance(weights, numpy.ndarray):
            self.entries += float(weights.sum())
        else:
            self.entries += float(weights * shape[0])

    def _sparksql(self, jvm, converter):
        return converter.UntypedLabel([jvm.scala.Tuple2(k, v._sparksql(jvm, converter)) for k, v in self.pairs.items()])

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return self.values

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName):
        return {"entries": floatToJson(self.entries),
                "data": dict((k, {"type": v.name, "data": v.toJsonFragment(False)}) for k, v in self.pairs.items())}

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "data"]):
            if json["entries"] in ("nan", "inf", "-inf") or isinstance(json["entries"], numbers.Real):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "UntypedLabel.entries")

            if isinstance(json["data"], dict):
                pairs = {}
                for k, v in json["data"].items():
                    if isinstance(v, dict) and hasKeys(v.keys(), ["type", "data"]):
                        factory = Factory.registered[v["type"]]
                        pairs[k] = factory.fromJsonFragment(v["data"], None)

                    else:
                        raise JsonFormatException(k, "UntypedLabel.data {0}".format(v))

            else:
                raise JsonFormatException(json, "UntypedLabel.data")

            return UntypedLabel.ed(entries, **pairs).specialize()

        else:
            raise JsonFormatException(json, "UntypedLabel")

    def __repr__(self):
        return "<UntypedLabel size={0}>".format(self.size)

    def __eq__(self, other):
        return isinstance(other, UntypedLabel) and numeq(self.entries, other.entries) and self.pairs == other.pairs

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.entries, tuple(sorted(self.pairs.items()))))


Factory.register(UntypedLabel)

# Index


class Index(Factory, Container, Collection):
    """Accumulate any number of aggregators of the same type in a list.

    Every sub-aggregator is filled with every input datum.

    This primitive provides an anonymous collection of aggregators (unless the integer index is taken to have special
    meaning, but generally such bookkeeping should be encoded in strings). Indexes can be nested to create
    two-dimensional ordinal grids of aggregators. (Use :doc:`Bin <histogrammar.primitives.bin.Bin>` if the space is
    to have a metric interpretation.)

    Note that all sub-aggregators within an Index must have the *same type* (e.g. histograms of different binnings,
    but all histograms). To collect objects of *different types,* still indexed by integer,
    use :doc:`Branch <histogrammar.primitives.collection.Branch>`.

    To collect aggregators of the *same type* with string-based labels,
    use :doc:`Label <histogrammar.primitives.collection.Label>`.
    To collect aggregators of *different types* with string-based labels,
    use :doc:`UntypedLabel <histogrammar.primitives.collection.UntypedLabel>`.

    In strongly typed languages, the restriction to a single type allows nested objects to be extracted without casting.
    """

    @staticmethod
    def ed(entries, *values):
        """Create an Index that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            values (list of :doc:`Container <histogrammar.defs.Container>`): the collection of filled aggregators.
        """
        if len(values) == 1 and isinstance(values[0], (list, tuple)):
            values = values[0]

        if not isinstance(entries, numbers.Real) and entries not in ("nan", "inf", "-inf"):
            raise TypeError("entries ({0}) must be a number".format(entries))
        if not all(isinstance(v, Container) for v in values):
            raise TypeError("values ({0}) must be a list of Containers".format(values))
        if entries < 0.0:
            raise ValueError("entries ({0}) cannot be negative".format(entries))

        out = Index(*values)
        out.entries = float(entries)
        return out.specialize()

    @staticmethod
    def ing(*values):
        """Synonym for ``__init__``."""
        return Index(*values)

    def __init__(self, *values):
        """Create an Index that is capable of being filled and added.

        Parameters:
            values (list of :doc:`Container <histogrammar.defs.Container>`): the collection of aggregators to fill.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
        """
        if not all(isinstance(v, Container) for v in values):
            raise TypeError("values ({0}) must be a list of Containers".format(values))
        if len(values) < 1:
            raise ContainerException("at least one value required")
        contentType = values[0].name
        if any(x.name != contentType for x in values):
            raise ValueError("all Index values must have the same type")
        if contentType == "Bag":
            rangeType = values[0].range
            if any(x.range != rangeType for x in values):
                raise ValueError("all Index values must have the same type")

        self.entries = 0.0
        self.values = values

        super(Index, self).__init__()
        self.specialize()

    @property
    def size(self):
        """Number of ``values``."""
        return len(self.values)

    @property
    def pairs(self):
        return dict(enumerate(self.values))

    def __call__(self, i, *rest):
        """Attempt to get key ``index``, throwing an exception if it does not exist."""
        if len(rest) == 0:
            return self.values[i]
        else:
            return self.values[i](*rest)

    def get(self, i):
        """Attempt to get index ``i``, returning ``None`` if it does not exist."""
        if i < 0 or i >= len(self.values):
            return None
        else:
            return self.values[i]

    def getOrElse(self, i, default):
        """Attempt to get index ``i``, returning an alternative if it does not exist."""
        if i < 0 or i >= len(self.values):
            return default
        else:
            return self.values[i]

    @inheritdoc(Container)
    def zero(self):
        return Index(*[x.zero() for x in self.values])

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Index):
            if self.size != other.size:
                raise ContainerException(
                    "cannot add Indexes because they have different sizes: ({0} vs {1})".format(
                        self.size, other.size))

            out = Index(*[x + y for x, y in zip(self.values, other.values)])
            out.entries = self.entries + other.entries
            return out.specialize()

        else:
            raise ContainerException("cannot add {0} and {1}".format(self.name, other.name))

    @inheritdoc(Container)
    def __iadd__(self, other):
        if isinstance(other, Index):
            if self.size != other.size:
                raise ContainerException(
                    "cannot add Indexes because they have different sizes: ({0} vs {1})".format(
                        self.size, other.size))
            self.entries += other.entries
            for x, y in zip(self.values, other.values):
                x += y
            return self
        else:
            raise ContainerException("cannot add {0} and {1}".format(self.name, other.name))

    @inheritdoc(Container)
    def __mul__(self, factor):
        if math.isnan(factor) or factor <= 0.0:
            return self.zero()
        else:
            out = Index(*[x * factor for x in self.values])
            out.entries = factor * self.entries
            return out.specialize()

    @inheritdoc(Container)
    def __rmul__(self, factor):
        return self.__mul__(factor)

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()

        if weight > 0.0:
            for x in self.values:
                x.fill(datum, weight)

            # no possibility of exception from here on out (for rollback)
            self.entries += weight

    def _numpy(self, data, weights, shape):
        if shape[0] is not None:
            self._checkNPWeights(weights, shape)
            weights = self._makeNPWeights(weights, shape)

        for x in self.values:
            x._numpy(data, weights, shape)

        # no possibility of exception from here on out (for rollback)
        import numpy
        if isinstance(weights, numpy.ndarray):
            self.entries += float(weights.sum())
        else:
            self.entries += float(weights * shape[0])

    def _sparksql(self, jvm, converter):
        return converter.Index([v._sparksql(jvm, converter) for v in self.values])

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return self.values

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName):
        return {"entries": floatToJson(self.entries),
                "sub:type": self.values[0].name,
                "data": [x.toJsonFragment(False) for x in self.values]}

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "sub:type", "data"]):
            if json["entries"] in ("nan", "inf", "-inf") or isinstance(json["entries"], numbers.Real):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "Index.entries")

            if isinstance(json["sub:type"], basestring):
                factory = Factory.registered[json["sub:type"]]
            else:
                raise JsonFormatException(json, "Index.sub:type")

            if isinstance(json["data"], list):
                values = [factory.fromJsonFragment(x, None) for x in json["data"]]
            else:
                raise JsonFormatException(json, "Index.data")

            return Index.ed(entries, *values).specialize()

        else:
            raise JsonFormatException(json, "Index")

    def __repr__(self):
        return "<Index values={0} size={1}>".format(self.values[0].name, self.size)

    def __eq__(self, other):
        return isinstance(other, Index) and numeq(self.entries, other.entries) and self.values == other.values

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.entries, tuple(self.values)))


Factory.register(Index)

# Branch


class Branch(Factory, Container, Collection):
    """Accumulate aggregators of different types, indexed by i0 through i9.

    Every sub-aggregator is filled with every input datum.

    This primitive provides an anonymous collection of aggregators of *different types,* usually for gluing together
    various statistics. For instance, if the following associates a sum of weights to every bin in a histogram,

    ::

        Bin.ing(100, 0, 1, lambda d: d.x,
        Sum.ing(lambda d: d.weight))

    the following would associate the sum of weights and the sum of squared weights to every bin:

    ::

        Bin.ing(100, 0, 1, lambda d: d.x,
        Branch.ing(Sum.ing(lambda d: d.weight), Sum.ing(lambda d: d.weight**2)))

    Branch is a basic building block for complex aggregators. The limitation to ten branches, indexed from i0 to i9,
    is a concession to type inference in statically typed languages. It is not a fundamental limit, but the
    type-metaprogramming becomes increasingly complex as branches are added. Error messages may be convoluted as
    the compiler presents internals of the type-metaprogramming in response to a user's simple mistake.

    Therefore, individual implementations may allow more than ten branches, but the Histogrammar standard only r
    equires ten.

    To collect an unlimited number of aggregators of the *same type* without naming them,
    use :doc:`Index <histogrammar.primitives.collection.Index>`.
    To collect aggregators of the *same type* with string-based labels,
    use :doc:`Label <histogrammar.primitives.collection.Label>`.
    To collect aggregators of *different types* with string-based labels,
    use :doc:`UntypedLabel <histogrammar.primitives.collection.UntypedLabel>`.
    """

    @staticmethod
    def ed(entries, *values):
        """Create a Branch that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            values (list of :doc:`Container <histogrammar.defs.Container>`): the collection of filled aggregators.
        """
        if len(values) == 1 and isinstance(values[0], (list, tuple)):
            values = values[0]

        if not isinstance(entries, numbers.Real) and entries not in ("nan", "inf", "-inf"):
            raise TypeError("entries ({0}) must be a number".format(entries))
        if not all(isinstance(v, Container) for v in values):
            raise TypeError("values ({0}) must be a list of Containers".format(values))
        if entries < 0.0:
            raise ValueError("entries ({0}) cannot be negative".format(entries))

        out = Branch(*values)
        out.entries = float(entries)
        return out.specialize()

    @staticmethod
    def ing(*values):
        """Synonym for ``__init__``."""
        return Branch(*values)

    def __init__(self, *values):
        """Create a Branch that is capable of being filled and added.

        Parameters:
            values (list of :doc:`Container <histogrammar.defs.Container>`): the collection of aggregators to fill.

        Other Parameters:
            entries (float): the number of entries, initially 0.0.
        """
        if not all(isinstance(v, Container) for v in values):
            raise TypeError("values ({0}) must be a list of Containers".format(values))
        if len(values) < 1:
            raise ValueError("at least one value required")

        self.entries = 0.0
        self.values = values

        for i, x in enumerate(values):
            setattr(self, "i" + str(i), x)

        super(Branch, self).__init__()
        self.specialize()

    @property
    def size(self):
        """Return the number of containers."""
        return len(self.values)

    @property
    def pairs(self):
        return dict(enumerate(self.values))

    def __call__(self, i, *rest):
        """Attempt to get key ``index``, throwing an exception if it does not exist."""
        if len(rest) == 0:
            return self.values[i]
        else:
            return self.values[i](*rest)

    def get(self, i):
        """Attempt to get index ``i``, returning ``None`` if it does not exist."""
        if i < 0 or i >= len(self.values):
            return None
        else:
            return self.values[i]

    def getOrElse(self, i, default):
        """Attempt to get index ``i``, returning an alternative if it does not exist."""
        if i < 0 or i >= len(self.values):
            return default
        else:
            return self.values[i]

    @inheritdoc(Container)
    def zero(self):
        return Branch(*[x.zero() for x in self.values])

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Branch):
            if self.size != other.size:
                raise ContainerException(
                    "cannot add Branches because they have different sizes: ({0} vs {1})".format(
                        self.size, other.size))

            out = Branch(*[x + y for x, y in zip(self.values, other.values)])
            out.entries = self.entries + other.entries
            return out.specialize()

        else:
            raise ContainerException("cannot add {0} and {1}".format(self.name, other.name))

    @inheritdoc(Container)
    def __iadd__(self, other):
        if isinstance(other, Branch):
            if self.size != other.size:
                raise ContainerException(
                    "cannot add Branches because they have different sizes: ({0} vs {1})".format(
                        self.size, other.size))
            self.entries += other.entries
            for x, y in zip(self.values, other.values):
                x += y
            return self
        else:
            raise ContainerException("cannot add {0} and {1}".format(self.name, other.name))

    @inheritdoc(Container)
    def __mul__(self, factor):
        if math.isnan(factor) or factor <= 0.0:
            return self.zero()
        else:
            out = Branch(*[x * factor for x in self.values])
            out.entries = factor * self.entries
            return out.specialize()

    @inheritdoc(Container)
    def __rmul__(self, factor):
        return self.__mul__(factor)

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()

        if weight > 0.0:
            for x in self.values:
                x.fill(datum, weight)

            # no possibility of exception from here on out (for rollback)
            self.entries += weight

    def _numpy(self, data, weights, shape):
        if shape[0] is not None:
            self._checkNPWeights(weights, shape)
            weights = self._makeNPWeights(weights, shape)

        for x in self.values:
            x._numpy(data, weights, shape)

        # no possibility of exception from here on out (for rollback)
        import numpy
        if isinstance(weights, numpy.ndarray):
            self.entries += float(weights.sum())
        else:
            self.entries += float(weights * shape[0])

    def _sparksql(self, jvm, converter):
        return converter.Branch(*[v._sparksql(jvm, converter) for v in self.values])

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return self.values

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName):
        return {"entries": floatToJson(self.entries),
                "data": [{"type": x.name, "data": x.toJsonFragment(False)} for x in self.values]}

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "data"]):
            if json["entries"] in ("nan", "inf", "-inf") or isinstance(json["entries"], numbers.Real):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "Branch.entries")

            if isinstance(json["data"], list):
                values = []
                for i, x in enumerate(json["data"]):
                    if isinstance(x, dict) and hasKeys(x.keys(), ["type", "data"]):
                        if isinstance(x["type"], basestring):
                            factory = Factory.registered[x["type"]]
                        else:
                            raise JsonFormatException(x, "Branch.data {0} type".format(i))
                        values.append(factory.fromJsonFragment(x["data"], None))

            else:
                raise JsonFormatException(json, "Branch.data")

            return Branch.ed(entries, *values)

        else:
            raise JsonFormatException(json, "Branch")

    def __repr__(self):
        return "<Branch {0}>".format(" ".join("i" + str(i) + "=" + v.name for i, v in enumerate(self.values)))

    def __eq__(self, other):
        return isinstance(other, Branch) and numeq(self.entries, other.entries) and self.values == other.values

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.entries, tuple(self.values)))


Factory.register(Branch)
