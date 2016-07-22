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

import collections
import json as jsonlib
import math
import re

from histogrammar.util import *

class ContainerException(Exception):
    """Exception type for improperly configured containers."""
    pass

class InvalidJsonException(Exception):
    """Exception type for strings that cannot be parsed because they are not proper JSON."""
    def __init__(self, message):
        super(InvalidJsonException, self).__init__("invalid JSON: {0}".format(message))

class JsonFormatException(Exception):
    """Exception type for unexpected JSON structure, thrown by ``fromJson`` methods."""
    def __init__(self, x, context):
        super(JsonFormatException, self).__init__("wrong JSON format for {0}: {1}".format(context, jsonlib.dumps(x)))

class Factory(object):
    """Interface for a container factory, always named as imperative verbs, such as "Count" and "Bin".
    
    Each factory has:
    
       - a custom ``__call__`` method to create an active container than can aggregate data.
       - a custom ``ed`` method to create a fixed container that cannot aggregate data, only merge with the ``+`` operator.
       - a uniform ``fromJsonFragment`` method that can reconstruct a fixed container from its JSON representation. This is used by the ``Factory`` object's ``fromJson`` entry point. (Click on the "t" in a circle in the upper-left to see the ``Factory`` object's documentation, rather than the ``Factory`` trait.

    In Python, no class distinction is made between active and fixed containers (e.g. "Counting" and "Counted" are both just "Count"). The distinction is maintained at runtime by which methods are available.

    Also particular to Python, the Container classes are their own Factories. Thus, ``Count.ing()`` makes a ``Count``.
   """

    registered = {}
    
    @staticmethod
    def register(factory):
        """Add a new ``Factory`` to the registry, introducing a new container type on the fly. General users usually wouldn't do this, but they could. This method is used internally to define the standard container types."""
        Factory.registered[factory.__name__] = factory

    def __init__(self):
        self._checkedForCrossReferences = False

    def specialize(self):
        """Explicitly invoke histogrammar.specialized.addImplicitMethods on this object, usually right after construction (in each of the methods that construct: ``__init__``, ``ed``, ``ing``, ``fromJsonFragment``, etc).

        Objects used as default parameter arguments are created too early for this to be possible, since they are created before the histogrammar.specialized module can be defined. These objects wouldn't satisfy any of ``addImplicitMethod``'s checks anyway.
        """
        try:
            import histogrammar.specialized
            histogrammar.specialized.addImplicitMethods(self)
        except (ImportError, AttributeError):
            pass
        return self

    @staticmethod
    def fromJsonFragment(json, nameFromParent):
        """Reconstructs a container of known type from JSON. General users should call the ``Factory`` object's ``fromJson``, which uses header data to identify the container type. (This is called by ``fromJson``.)"""
        raise NotImplementedError

    @staticmethod
    def fromJson(json):
        """User's entry point for reconstructing a container from JSON text."""

        if isinstance(json, basestring):
            json = jsonlib.loads(json)

        if isinstance(json, dict) and set(json.keys()) == set(["type", "data"]):
            if isinstance(json["type"], basestring):
                name = json["type"]
            else:
                raise JsonFormatException(json["type"], "Factory.type")

            if name not in Factory.registered:
                raise JsonFormatException(json, "unrecognized container (is it a custom container that hasn't been registered?): {0}".format(name))

            return Factory.registered[name].fromJsonFragment(json["data"], None)

        else:
            raise JsonFormatException(json, "Factory")
        
class Container(object):
    """Interface for classes that contain aggregated data, such as "Count" or "Bin".
    
    Containers are monoids: they have a neutral element (``zero``) and an associative operator (``+``). Thus, partial sums aggregated in parallel can be combined arbitrarily.
    """

    @property
    def name(self):
        """Name of the concrete ``Factory`` as a string; used to label the container type in JSON."""
        return self.__class__.__name__
    @property
    def factory(self):
        """Reference to the container's factory for runtime reflection."""
        return self.__class__

    def zero(self):
        """Create an empty container with the same parameters as this one. The original is unaffected. """
        raise NotImplementedError

    def __add__(self, other):
        """Add two containers of the same type. The originals are unaffected. """
        raise NotImplementedError

    def fill(self, datum, weight=1.0):
        """Increment the aggregator by providing one ``datum`` to the fill rule with a given ``weight``.
      
        Usually all containers in a collection of histograms take the same input data by passing it recursively through the tree. Quantities to plot are specified by the individual container's lambda functions.
      
        The container is changed in-place.
        """
        raise NotImplementedError

    def copy(self):
        """Copy this container, making a clone with no reference to the original. """
        return self + self.zero()

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        raise NotImplementedError

    def _checkForCrossReferences(self, memo=None):
        if not self._checkedForCrossReferences:
            if memo is None:
                memo = set()
            if any(x is self for x in memo):
                raise ContainerException("cannot fill a tree that contains the same aggregator twice: {0}".format(self))
            memo.add(self)
            for child in self.children:
                child._checkForCrossReferences(memo)
            self._checkedForCrossReferences = True

    def toJson(self):
        """Convert this container to dicts and lists representing JSON (dropping its ``fill`` method).
       
        Note that the dicts and lists can be turned into a string with ``json.dumps``.
        """
        return {"type": self.name, "data": self.toJsonFragment(False)}

    def toJsonFragment(self, suppressName):
        """Used internally to convert the container to JSON without its ``"type"`` header."""
        raise NotImplementedError

    _clingClassNameNumber = 0
    def cling(self, ttree, start=-1, end=-1, debug=False):
        self._checkForCrossReferences()

        if not hasattr(self, "_clingFiller"):
            import ROOT

            inputFieldNames = {}
            inputFieldTypes = {}
            for branch in ttree.GetListOfBranches():
                if branch.GetClassName() == "":
                    # FIXME: what about split vs unsplit?
                    # FIXME: what about leaves that are arrays?
                    for leaf in branch.GetListOfLeaves():
                        if leaf.IsA() == ROOT.TLeafO.Class():
                            inputFieldTypes[leaf.GetName()] = "bool"
                        elif leaf.IsA() == ROOT.TLeafB.Class() and leaf.IsUnsigned():
                            inputFieldTypes[leaf.GetName()] = "unsigned char"
                        elif leaf.IsA() == ROOT.TLeafB.Class():
                            inputFieldTypes[leaf.GetName()] = "char"
                        elif leaf.IsA() == ROOT.TLeafS.Class() and leaf.IsUnsigned():
                            inputFieldTypes[leaf.GetName()] = "unsigned short"
                        elif leaf.IsA() == ROOT.TLeafS.Class():
                            inputFieldTypes[leaf.GetName()] = "short"
                        elif leaf.IsA() == ROOT.TLeafI.Class() and leaf.IsUnsigned():
                            inputFieldTypes[leaf.GetName()] = "UInt_t"
                        elif leaf.IsA() == ROOT.TLeafI.Class():
                            inputFieldTypes[leaf.GetName()] = "Int_t"
                        elif leaf.IsA() == ROOT.TLeafL.Class() and leaf.IsUnsigned():
                            inputFieldTypes[leaf.GetName()] = "ULong64_t"
                        elif leaf.IsA() == ROOT.TLeafL.Class():
                            inputFieldTypes[leaf.GetName()] = "Long64_t"
                        elif leaf.IsA() == ROOT.TLeafF.Class():
                            inputFieldTypes[leaf.GetName()] = "float"
                        elif leaf.IsA() == ROOT.TLeafD.Class():
                            inputFieldTypes[leaf.GetName()] = "double"
                        elif leaf.IsA() == ROOT.TLeafC.Class():
                            raise NotImplementedError("TODO: TLeafC (string)")
                        elif leaf.IsA() == ROOT.TLeafElement.Class():
                            raise NotImplementedError("TODO: TLeafElement")
                        elif leaf.IsA() == ROOT.TLeafObject.Class():
                            raise NotImplementedError("TODO: TLeafObject")
                        else:
                            raise NotImplementedError("unknown leaf type: " + repr(leaf))

                else:
                    inputFieldTypes[branch.GetName()] = branch.GetClassName() + "*"
                    
            derivedFieldTypes = {}
            derivedFieldExprs = {}

            storageStructs = collections.OrderedDict()
            initCode = []
            fillCode = []
            weightVars = ["weight_0"]
            weightVarStack = ("weight_0",)
            tmpVarTypes = {}
            self._clingGenerateCode(inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs, storageStructs, initCode, (("var", "storage"),), 4, fillCode, 6, weightVars, weightVarStack, tmpVarTypes)

            className = "HistogrammarClingFiller_" + str(Container._clingClassNameNumber)
            Container._clingClassNameNumber += 1
            classCode = """class {0} {{
public:
{1}
{2}{3}
{4}{5}  {6} storage;

  bool isnan(double x) {{ return x != x; }}
  bool isnan(float x) {{ return x != x; }}
  bool isnan(int x) {{ return false; }}

  void fillall(TTree* ttree, Long64_t start, Long64_t end) {{
{7}
    weight_0 = 1.0;
{8}
    if (start < 0) start = 0;
    if (end < 0) end = ttree->GetEntries();
    for (;  start < end;  ++start) {{
      // ttree->GetEntry(start);
      continue;
{9}{10}
    }}

    ttree->ResetBranchAddresses();
  }}
}};
""".format(className,
           "".join(storageStructs.values()),
           "".join("  double " + n + ";\n" for n in weightVars),
           "".join("  " + t + " " + self._clingNormalizeTTreeName(n) + ";\n" for n, t in inputFieldTypes.items() if self._clingNormalizeTTreeName(n) in inputFieldNames),
           "".join("  " + t + " " + n + ";\n" for n, t in derivedFieldTypes.items()),
           "".join("  " + t + " " + n + ";\n" for n, t in tmpVarTypes.items()),
           self._clingStorageType(),
           "\n".join(initCode),
           "".join("    ttree->SetBranchAddress(" + jsonlib.dumps(key) + ", &" + n + ");\n" for n, key in inputFieldNames.items()),
           "".join("      " + n + " = " + e + ";\n" for n, e in derivedFieldExprs.items()),
           "\n".join(fillCode))

            if debug:
                print("line |")
                print("\n".join("{0:4d} | {1}".format(i + 1, line) for i, line in enumerate(classCode.split("\n"))))
            if not ROOT.gInterpreter.Declare(classCode):
                if debug:
                    raise SyntaxError("Could not compile the above")
                else:
                    raise SyntaxError("Could not compile (rerun with debug=True to see the generated C++ code)")

            self._clingFiller = getattr(ROOT, className)()

        # we already have a _clingFiller; just fill
        self._clingFiller.fillall(ttree, start, end)
        self._clingUpdate(self._clingFiller, ("var", "storage"))
                
    def _clingExpandPrefixCpp(self, *prefix):
        out = ""
        for t, x in prefix:
            if t == "var":
                if len(out) == 0:
                    out += x
                else:
                    out += "." + x
            elif t == "index":
                out += "[" + str(x) + "]"
            else:
                raise Exception((t, x))
        return out

    def _clingExpandPrefixPython(self, obj, *prefix):
        for t, x in prefix:
            if t == "var":
                obj = getattr(obj, x)
            elif t == "index":
                obj = obj.__getitem__(x)
            elif t == "func":
                name = x[0]
                args = x[1:]
                obj = getattr(obj, name)(*args)
            else:
                raise NotImplementedError((t, x))
        return obj

    def _clingQuantityExpr(self, inputFieldNames, inputFieldTypes, derivedFieldTypes, derivedFieldExprs):
        if not isinstance(self.quantity.expr, basestring):
            raise ContainerException(self.factory.name + ".quantity must be provided as a C++ string to use with Cling")

        normexpr = self._clingNormalizeTTreeExpr(inputFieldNames, inputFieldTypes, self.quantity.expr)
        if self._clingInputFieldRef(self.quantity.expr):
            return normexpr
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
            return derivedFieldName

    def _clingInputFieldRef(self, code):
        return re.match("^\s*[a-zA-Z_][a-zA-Z0-9]*\s*$", code) is not None

    def _clingNormalizeTTreeName(self, key):
        return "input_" + key    # FIXME: add rules to turn arbitrary strings into valid C++ variable names

    def _clingNormalizeTTreeExpr(self, inputFieldNames, inputFieldTypes, expr):
        expr = " " + expr + " "
        for name, t in inputFieldTypes.items():
            freeword = r"(\W)" + name + r"(\W)"
            if re.search(freeword, expr) is not None:
                norm = self._clingNormalizeTTreeName(name)
                inputFieldNames[norm] = name
                if t.endswith("*"):
                    norm = "(*" + norm + ")"
                return re.sub(freeword, r"\1" + norm + r"\2", expr).strip()
        return expr.strip()

    def _clingStorageType(self):
        return self._clingStructName()

    def _clingStructName(self):
        raise NotImplementedError

    def numpy(self, data, weights=1.0):
        import numpy
        self._checkForCrossReferences()

        if isinstance(weights, numpy.ndarray):
            assert len(weights.shape) == 1

            original = weights
            weights = numpy.array(weights, dtype=numpy.float64)
            weights[numpy.isnan(weights)] = 0.0
            weights[weights < 0.0] = 0.0

            shape = [weights.shape[0]]

        elif math.isnan(weights) or weights <= 0.0:
            return

        else:
            shape = [None]

        self._numpy(data, weights, shape)

    def _checkNPQuantity(self, q, shape):
        import numpy
        assert isinstance(q, numpy.ndarray)
        assert len(q.shape) == 1
        if shape[0] is None:
            shape[0] = q.shape[0]
        else:
            assert q.shape[0] == shape[0]

    def _checkNPWeights(self, weights, shape):
        import numpy
        if isinstance(weights, numpy.ndarray):
            assert len(weights.shape) == 1
            assert weights.shape[0] == shape[0]

    def _makeNPWeights(self, weights, shape):
        import numpy
        if isinstance(weights, numpy.ndarray):
            return weights
        else:
            return weights * numpy.ones(shape, dtype=numpy.float64)

# useful functions

unweighted = named("unweighted", lambda datum: 1.0)
identity = named("identity", lambda x: x)
square = named("square", lambda x: x)

# functions for Spark's RDD.aggregate

def increment(container, datum):
    """Increment function for Apache Spark's ``aggregate`` method.
    * 
    * Typical use: ``filledHistogram = datasetRDD.aggregate(initialHistogram, increment, combine)`` where ``datasetRDD`` is a collection of ``initialHistogram``'s input type.
   """
    container.fill(datum)
    return container

def combine(container1, container2):
    """Combine function for Apache Spark's ``aggregate`` method.
     
    Typical use: ``filledHistogram = datasetRDD.aggregate(initialHistogram)(increment, combine)`` where ``datasetRDD`` is a collection of ``initialHistogram``'s input type.
   """
    return container1 + container2

# symbols for Branch paths (Index paths use integers, Label/UntypedLabel paths use strings)

i0 = 0
i1 = 1
i2 = 2
i3 = 3
i4 = 4
i5 = 5
i6 = 6
i7 = 7
i8 = 8
i9 = 9
