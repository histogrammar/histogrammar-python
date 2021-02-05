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

import base64
import datetime
import json as jsonlib
import math
import random
import re
try:
    from collections import OrderedDict
except ImportError:
    class OrderedDict(object):
        def __init__(self):
            self.pairs = {}
            self.keys = []

        def __setitem__(self, key, value):
            self.pairs[key] = value
            if key not in self.keys:
                self.keys.append(key)

        def __getitem__(self, key):
            return self.pairs[key]

        def values(self):
            return [self.pairs[k] for k in self.keys]

        def items(self):
            return [(k, self.pairs[k]) for k in self.keys]

        def __iter__(self):
            return iter(self.keys)

        def __len__(self):
            return len(self.keys)

from histogrammar.util import FillMethod, PlotMethod, basestring, xrange, named
from histogrammar.parsing import C99SourceToAst
from histogrammar.parsing import C99AstToSource
from histogrammar.pycparser import c_ast
import histogrammar.version


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
       - a custom ``ed`` method to create a fixed container that cannot aggregate data, only merge with
         the ``+`` operator.
       - a uniform ``fromJsonFragment`` method that can reconstruct a fixed container from its JSON representation.
         This is used by the ``Factory`` object's ``fromJson`` entry point. (Click on the "t" in a circle in the
        upper-left to see the ``Factory`` object's documentation, rather than the ``Factory`` trait.

    In Python, no class distinction is made between active and fixed containers (e.g. "Counting" and "Counted" are
    both just "Count"). The distinction is maintained at runtime by which methods are available.

    Also particular to Python, the Container classes are their own Factories. Thus, ``Count.ing()`` makes a ``Count``.
    """

    registered = {}

    @staticmethod
    def register(factory):
        """Add a new ``Factory`` to the registry, introducing a new container type on the fly.

        General users usually wouldn't do this, but they could. This method is used internally to define the
        standard container types.
        """
        Factory.registered[factory.__name__] = factory

    def __init__(self):
        self._checkedForCrossReferences = False

    def specialize(self):
        """Explicitly invoke histogrammar.specialized.addImplicitMethods on this object.

        Usually right after construction (in each of the methods that construct: ``__init__``, ``ed``, ``ing``,
        ``fromJsonFragment``, etc).

        Objects used as default parameter arguments are created too early for this to be possible,
        since they are created before the histogrammar.specialized module can be defined.
        These objects wouldn't satisfy any of ``addImplicitMethod``'s checks anyway.
        """
        try:
            import histogrammar.specialized
            histogrammar.specialized.addImplicitMethods(self)
        except (ImportError, AttributeError):
            pass
        self.fill = FillMethod(self, self.fill)
        self.plot = PlotMethod(self, self.plot)
        return self

    @staticmethod
    def fromJsonFragment(json, nameFromParent):
        """Reconstructs a container of known type from JSON.

        General users should call the ``Factory`` object's ``fromJson``, which uses header data to identify the
        container type. (This is called by ``fromJson``.)
        """
        raise NotImplementedError

    @staticmethod
    def fromJsonFile(fileName):
        return Factory.fromJson(jsonlib.load(open(fileName)))

    @staticmethod
    def fromJsonString(json):
        return Factory.fromJson(jsonlib.loads(json))

    @staticmethod
    def fromJson(json):
        """User's entry point for reconstructing a container from JSON text."""

        if isinstance(json, basestring):
            json = jsonlib.loads(json)

        if isinstance(json, dict) and "type" in json and "data" in json and "version" in json:
            if isinstance(json["version"], basestring):
                if not histogrammar.version.compatible(json["version"]):
                    raise ContainerException(
                        "cannot read a Histogrammar {0} document with histogrammar-python version {1}".format(
                            json["version"], histogrammar.version.version))
            else:
                raise JsonFormatException(json["version"], "Factory.version")

            if isinstance(json["type"], basestring):
                name = json["type"]
            else:
                raise JsonFormatException(json["type"], "Factory.type")

            if name not in Factory.registered:
                raise JsonFormatException(json, "unrecognized container (is it a custom container "
                                                "that hasn't been registered?): {0}".format(name))

            return Factory.registered[name].fromJsonFragment(json["data"], None)

        else:
            raise JsonFormatException(json, "Factory")


class Container(object):
    """Interface for classes that contain aggregated data, such as "Count" or "Bin".

    Containers are monoids: they have a neutral element (``zero``) and an associative operator (``+``).
    Thus, partial sums aggregated in parallel can be combined arbitrarily.
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
        """Add two containers of the same type. The originals are unaffected."""
        raise NotImplementedError

    def __iadd__(self, other):
        """Add other to self; other is unaffected, but self is changed in place."""
        raise NotImplementedError

    def __mul__(self, factor):
        """Reweight the contents in all nested aggregators by a scalar factor

        As though they had been filled with a different weight. The original is unaffected.
        """
        raise NotImplementedError

    def __rmul__(self, factor):
        """Reweight the contents in all nested aggregators by a scalar factor

        As though they had been filled with a different weight. The original is unaffected.
        """
        raise NotImplementedError

    def fill(self, datum, weight=1.0):
        """Increment the aggregator by providing one ``datum`` to the fill rule with a given ``weight``.

        Usually all containers in a collection of histograms take the same input data by passing it recursively
        through the tree. Quantities to plot are specified by the individual container's lambda functions.

        The container is changed in-place.
        """
        raise NotImplementedError

    def plot(self, httpServer=None, **parameters):
        """Generate a VEGA visualization and serve it via HTTP."""
        raise NotImplementedError

    def __getstate__(self):
        state = dict(self.__dict__)
        del state["fill"]
        del state["plot"]
        return state

    def __setstate__(self, dict):
        self.__dict__ = dict
        self.fill = FillMethod(self, self.fill)
        self.plot = PlotMethod(self, self.plot)

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

    def toJsonFile(self, fileName):
        return jsonlib.dump(self.toJson(), open(fileName, "w"))

    def toJsonString(self):
        return jsonlib.dumps(self.toJson())

    def toJson(self):
        """Convert this container to dicts and lists representing JSON (dropping its ``fill`` method).

        Note that the dicts and lists can be turned into a string with ``json.dumps``.
        """
        return {"type": self.name, "data": self.toJsonFragment(False), "version": histogrammar.version.specification}

    def toJsonFragment(self, suppressName):
        """Used internally to convert the container to JSON without its ``"type"`` header."""
        raise NotImplementedError

    def toImmutable(self):
        """Return a copy of this container

        As though it was created by the ``ed`` function or from JSON (the \"immutable form\" in languages that
        support it, not Python).
        """
        return Factory.fromJson(self.toJson())

    _clingClassNameNumber = 0

    def fillroot(self, ttree, start=-1, end=-1, debug=False, debugOnError=True, **exprs):
        self._checkForCrossReferences()

        if not hasattr(self, "_clingFiller"):
            import ROOT

            parser = C99SourceToAst()
            generator = C99AstToSource()

            inputFieldNames = {}
            inputFieldTypes = {}
            for branch in ttree.GetListOfBranches():
                if branch.GetClassName() == "":
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

                        inputFieldTypes[leaf.GetName()] += "*" * leaf.GetTitle().count("[")

                else:
                    inputFieldTypes[branch.GetName()] = branch.GetClassName() + "*"

            derivedFieldTypes = {}
            derivedFieldExprs = {}

            storageStructs = OrderedDict()
            initCode = []
            fillCode = []
            weightVars = ["weight_0"]
            weightVarStack = ("weight_0",)
            tmpVarTypes = {}

            for name, expr in exprs.items():
                self._clingAddExpr(
                    parser,
                    generator,
                    name,
                    expr,
                    inputFieldNames,
                    inputFieldTypes,
                    derivedFieldTypes,
                    derivedFieldExprs)

            self._cppGenerateCode(
                parser,
                generator,
                inputFieldNames,
                inputFieldTypes,
                derivedFieldTypes,
                derivedFieldExprs,
                storageStructs,
                initCode,
                (("var",
                  "storage"),
                 ),
                4,
                fillCode,
                (("var",
                  "storage"),
                 ),
                6,
                weightVars,
                weightVarStack,
                tmpVarTypes)

            className = "HistogrammarClingFiller_" + str(Container._clingClassNameNumber)
            Container._clingClassNameNumber += 1
            classCode = """class {0} {{
public:
{1}
{2}{3}
{4}{5}  {6} storage;

  void init() {{
{7}
    weight_0 = 1.0;
  }}

  void fillall(TTree* ttree, Long64_t start, Long64_t end) {{
    ttree->SetBranchStatus("*", 0);
{11}
    init();
{8}
    if (start < 0) start = 0;
    if (end < 0) end = ttree->GetEntries();
    for (;  start < end;  ++start) {{
      ttree->GetEntry(start);
{9}{10}
    }}

    ttree->ResetBranchAddresses();
  }}
}};
""".format(className,
                "".join(storageStructs.values()),
                "".join("  double " + n + ";\n" for n in weightVars),
                "".join("  " + t + " " + self._cppNormalizeInputName(n) + ";\n" for n,
                        t in inputFieldTypes.items() if self._cppNormalizeInputName(n) in inputFieldNames),
                "".join("  " + t + " " + n + ";\n" for n, t in derivedFieldTypes.items() if t != "auto"),
                "".join("  " + t + " " + n + ";\n" for n, t in tmpVarTypes.items()),
                self._cppStorageType(),
                "\n".join(initCode),
                "".join("    ttree->SetBranchAddress(" + jsonlib.dumps(key) +
                        ", &" + n + ");\n" for n, key in inputFieldNames.items()),
                "".join(x for x in derivedFieldExprs.values()),
                "\n".join(fillCode),
                "".join("    ttree->SetBranchStatus(\"" + key + "\", 1);\n" for key in inputFieldNames.values()))

            if debug:
                print("line |")
                print("\n".join("{0:4d} | {1}".format(i + 1, line) for i, line in enumerate(classCode.split("\n"))))
            if not ROOT.gInterpreter.Declare(classCode):
                if debug:
                    raise SyntaxError("Could not compile the above")
                elif debugOnError:
                    raise SyntaxError(
                        "Could not compile the following:\n\n" +
                        "\n".join(
                            "{0:4d} | {1}".format(
                                i +
                                1,
                                line) for i,
                            line in enumerate(
                                classCode.split("\n"))))
                else:
                    raise SyntaxError("Could not compile (rerun with debug=True to see the generated C++ code)")

            self._clingFiller = getattr(ROOT, className)()

        # we already have a _clingFiller; just fill
        self._clingFiller.fillall(ttree, start, end)
        self._clingUpdate(self._clingFiller, ("var", "storage"))

    _cudaNamespaceNumber = 0

    def cuda(self, namespace=True, namespaceName=None, writeSize=False, commentMain=True,
             split=False, testData=[round(random.gauss(0, 1), 2) for x in xrange(10)], **exprs):
        parser = C99SourceToAst()
        generator = C99AstToSource()

        inputFieldNames = OrderedDict()
        inputFieldTypes = {}
        derivedFieldTypes = {}
        derivedFieldExprs = OrderedDict()
        storageStructs = OrderedDict()
        initCode = []
        fillCode = []
        combineCode = []
        jsonCode = []
        weightVars = ["weight_0"]
        weightVarStack = ("weight_0",)
        tmpVarTypes = {}

        for name, expr in exprs.items():
            self._cudaAddExpr(
                parser,
                generator,
                name,
                expr,
                inputFieldNames,
                inputFieldTypes,
                derivedFieldTypes,
                derivedFieldExprs)

        self._cudaGenerateCode(
            parser,
            generator,
            inputFieldNames,
            inputFieldTypes,
            derivedFieldTypes,
            derivedFieldExprs,
            storageStructs,
            initCode,
            (("var",
              "(*aggregator)"),
             ),
            4,
            fillCode,
            (("var",
              "(*aggregator)"),
             ),
            4,
            combineCode,
            (("var",
              "(*total)"),
             ),
            (("var",
              "(*item)"),
             ),
            4,
            jsonCode,
            (("var",
              "(*aggregator)"),
             ),
            4,
            weightVars,
            weightVarStack,
            tmpVarTypes,
            False)

        if namespaceName is None:
            namespaceName = "HistogrammarCUDA_" + str(Container._cudaNamespaceNumber)
            Container._cudaNamespaceNumber += 1

        out = '''// Auto-generated on {timestamp:%Y-%m-%d %H:%M:%S}
// If you edit this file, it will be hard to swap it out for another auto-generated copy.

#ifndef {NS}
#define {NS}

#include <stdio.h>
#include <math_constants.h>

/////////////////////////////////////////////////////////////////// declarations

namespace {ns} {{
  // How the aggregator is laid out in memory (CPU main memory and GPU shared memory).{typedefs}
  // Convenient name for the whole aggregator.
  typedef {lastStructName} Aggregator;

  // Specific logic of how to zero out the aggregator.
  __device__ void zero(Aggregator* aggregator);

  // Specific logic of how to increment the aggregator with input values.
  __device__ void increment(Aggregator* aggregator{comma}{inputArgList});

  // Specific logic of how to combine two aggregators.
  __device__ void combine(Aggregator* total, Aggregator* item);

  // Used by toJson.
  __host__ void floatToJson(FILE* out, float x);

  // Specific logic of how to print out the aggregator.
  __host__ void toJson(Aggregator* aggregator, FILE* out);

  // Generic blockId calculation (3D is the most general).
  __device__ int blockId();

  // Generic blockSize calculation (3D is the most general).
  __device__ int blockSize();

  // Generic threadId calculation (3D is the most general).
  __device__ int threadId();

  // Wrapper for CUDA calls to report errors.
  void errorCheck(cudaError_t code);

  // User-level API for initializing the aggregator (from CPU or GPU).
  //
  //   aggregators: array of aggregators to fill in parallel.
  //   numAggregators: number of aggregators to fill.
  //
  __global__ void initialize(Aggregator* aggregators, int numAggregators);

  // User-level API for filling the aggregator with a single value (from GPU).
  //
  //   aggregators: array of aggregators to fill in parallel.
  //   numAggregators: number of aggregators to fill.
  //   input_*: the input variables you used in the aggregator\'s fill rule.
  //
  __device__ void fill(Aggregator* aggregators, int numAggregators{comma}{inputArgList});

  // User-level API for filling the aggregator with arrays of values (from CPU or GPU).
  //
  //   aggregators: array of aggregators to fill in parallel.
  //   numAggregators: number of aggregators to fill.
  //   input_*: the input variables you used in the aggregator\'s fill rule.
  //   numDataPoints: the number of values in each input_* array.
  //
  __global__ void fillAll(Aggregator* aggregators, int numAggregators{comma}{inputArgStarList}, int numDataPoints);

  // User-level API for combining all aggregators in a block (from CPU or GPU).
  //
  // Assumes that at least `numAggregators` threads are all running at the same time (can be mutually
  // synchronized with `__syncthreads()`. Generally, this means that `extract` should be called on
  // no more than one block, which suggests that `numAggregators` ought to be equal to the maximum
  // number of threads per block.
  //
  //   aggregators: array of aggregators to fill in parallel.
  //   numAggregators: number of aggregators to fill.
  //   result: single output
  //
  __global__ void extract(Aggregator* aggregators, int numAggregators, Aggregator* result);

  // Test function provides an example of how to use the API.
  //
  //   numAggregators: number of aggregators to fill.
  //   numBlocks: number of independent blocks to run.
  //   numThreadsPerBlock: number of threads to run in each block.
  //   input_*: the input variables you used in the aggregator\'s fill rule.
  //   numDataPoints: the number of values in each input_* array.
  //
  void test(int numAggregators, int numBlocks, int numThreadsPerBlock, {inputArgStarList}{comma}int numDataPoints);
}}

/////////////////////////////////////////////////////////////////// implementations

namespace {ns} {{{writeSize}
  // Specific logic of how to zero out the aggregator.
  __device__ void zero(Aggregator* aggregator) {{
{tmpVarDeclarations}{initCode}
  }}

  // Specific logic of how to increment the aggregator with input values.
  __device__ void increment(Aggregator* aggregator{comma}{inputArgList}) {{
    const float weight_0 = 1.0f;
{weightVarDeclarations}{tmpVarDeclarations}{quantities}
{fillCode}
  }}

  // Specific logic of how to combine two aggregators.
  __device__ void combine(Aggregator* total, Aggregator* item) {{
{tmpVarDeclarations}{combineCode}
  }}

  // Used by toJson.
  __host__ void floatToJson(FILE* out, float x) {{
    if (isnan(x))
      fprintf(out, "\\"nan\\"");
    else if (isinf(x)  &&  x > 0.0f)
      fprintf(out, "\\"inf\\"");
    else if (isinf(x))
      fprintf(out, "\\"-inf\\"");
    else
      fprintf(out, "%g", x);
  }}

  // Specific logic of how to print out the aggregator.
  __host__ void toJson(Aggregator* aggregator, FILE* out) {{
{tmpVarDeclarations}    fprintf(out, "{{\\"version\\": \\"{specVersion}\\", \\"type\\": \\"{factoryName}\\", \\"data\\": ");
{jsonCode}
    fprintf(out, "}}\\n");
  }}

  // Generic blockId calculation (3D is the most general).
  __device__ int blockId() {{
    return blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  }}

  // Generic blockSize calculation (3D is the most general).
  __device__ int blockSize() {{
    return blockDim.x * blockDim.y * blockDim.z;
  }}

  // Generic threadId calculation (3D is the most general).
  __device__ int threadId() {{
    return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
  }}

  // Wrapper for CUDA calls to report errors.
  void errorCheck(cudaError_t code) {{
    if (code != cudaSuccess) {{
      fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(code));
      exit(code);
    }}
  }}

  // User-level API for initializing the aggregator (from CPU or GPU).
  //
  //   aggregators: array of aggregators to fill in parallel.
  //   numAggregators: number of aggregators to fill.
  //
  __global__ void initialize(Aggregator* aggregators, int numAggregators) {{
    zero(&aggregators[(threadId() + blockId() * blockSize()) % numAggregators]);
  }}

  // User-level API for filling the aggregator with a single value (from GPU).
  //
  //   aggregators: array of aggregators to fill in parallel.
  //   numAggregators: number of aggregators to fill.
  //   input_*: the input variables you used in the aggregator\'s fill rule.
  //
  __device__ void fill(Aggregator* aggregators, int numAggregators{comma}{inputArgList}) {{
    increment(&aggregators[(threadId() + blockId() * blockSize()) % numAggregators]{comma}{inputList});
  }}

  // User-level API for filling the aggregator with arrays of values (from CPU or GPU).
  //
  //   aggregators: array of aggregators to fill in parallel.
  //   numAggregators: number of aggregators to fill.
  //   input_*: the input variables you used in the aggregator\'s fill rule.
  //   numDataPoints: the number of values in each input_* array.
  //
  __global__ void fillAll(Aggregator* aggregators, int numAggregators{comma}{inputArgStarList}, int numDataPoints) {{
    int id = threadId() + blockId() * blockSize();
    if (id < numDataPoints)
      fill(aggregators, numAggregators{comma}{inputListId});
  }}

  // User-level API for combining all aggregators in a block (from CPU or GPU).
  //
  // Assumes that at least `numAggregators` threads are all running at the same time (can be mutually
  // synchronized with `__syncthreads()`. Generally, this means that `extract` should be called on
  // no more than one block, which suggests that `numAggregators` ought to be equal to the maximum
  // number of threads per block.
  //
  //   aggregators: array of aggregators to fill in parallel.
  //   numAggregators: number of aggregators to fill.
  //   result: single output
  //
  __global__ void extract(Aggregator* aggregators, int numAggregators, Aggregator* result) {{
    // merge down in log(N) steps until the thread with id == 0 has the total for this block
    int id = threadId();

    // i should be the first power of 2 larger than numAggregators/2
    int i = 1;
    while (2*i < numAggregators) i <<= 1;

    // iteratively split the sample and combine the upper half into the lower half
    while (i != 0) {{
      if (id < i  &&  id + i < numAggregators) {{
        Aggregator* ours = &aggregators[id % numAggregators];
        Aggregator* theirs = &aggregators[(id + i) % numAggregators];
        combine(ours, theirs);
      }}

      // every iteration should be in lock-step across threads in this block
      __syncthreads();
      i >>= 1;
    }}

    // return the result, which is in thread 0\'s copy (aggregators[0])
    if (id == 0) {{
      Aggregator* blockLocal = &result[blockId()];
      memcpy(blockLocal, aggregators, sizeof(Aggregator));
    }}
  }}

  // Test function provides an example of how to use the API.
  //
  //   numAggregators: number of aggregators to fill.
  //   numBlocks: number of independent blocks to run.
  //   numThreadsPerBlock: number of threads to run in each block.
  //   input_*: the input variables you used in the aggregator\'s fill rule.
  //   numDataPoints: the number of values in each input_* array.
  //
  void test(int numAggregators, int numBlocks, int numThreadsPerBlock, {inputArgStarList}{comma}int numDataPoints) {{
    // Create the aggregators and call initialize on them.
    Aggregator* aggregators;
    cudaMalloc((void**)&aggregators, numAggregators * sizeof(Aggregator));
    initialize<<<1, numThreadsPerBlock>>>(aggregators, numAggregators);
    errorCheck(cudaPeekAtLastError());
    errorCheck(cudaDeviceSynchronize());

{copyTestData}
    // Call fill next, using the same number of blocks, threads per block, and memory allocation.
    // fillAll is a __global__ function that takes arrays; fill is a __device__ function that
    // takes single entries. Use the latter if filling from your GPU application.
    fillAll<<<numBlocks, numThreadsPerBlock>>>(aggregators, numAggregators{comma}{gpuList}, numDataPoints);
    errorCheck(cudaPeekAtLastError());
    errorCheck(cudaDeviceSynchronize());

{freeTestData}
    // Call extract and give it an Aggregator to overwrite.
    Aggregator* resultGPU;
    cudaMalloc((void**)&resultGPU, sizeof(Aggregator));
    extract<<<1, numThreadsPerBlock>>>(aggregators, numAggregators, resultGPU);

    // Now you can free the collection of subaggregators from the GPU.
    cudaFree(aggregators);

    Aggregator resultCPU;
    cudaMemcpy(&resultCPU, resultGPU, sizeof(Aggregator), cudaMemcpyDeviceToHost);

    // Now you can free the total aggregator from the GPU.
    cudaFree(resultGPU);

    // This Aggregator can be written to stdout as JSON for other Histogrammar programs to interpret
    // (and plot).
    toJson(&resultCPU, stdout);
  }}
}}

// Optional main runs a tiny test.
{startComment}
int main(int argc, char** argv) {{
  int numAggregators = 5;
  int numBlocks = 2;
  int numThreadsPerBlock = 5;

  int numDataPoints = 10;
{initTestData}
  {ns}::test(numAggregators, numBlocks, numThreadsPerBlock, {inputList}{comma}numDataPoints);
}}
{endComment}

#endif  // {NS}
'''.format(timestamp=datetime.datetime.now(),
           ns=namespaceName,
           NS=namespaceName.upper(),
           writeSize="""  __global__ void write_size(size_t *output) {{
    *output = sizeof(Aggregator);
  }}
""" if writeSize else "",
           specVersion=histogrammar.version.specification,
           factoryName=self.name,
           typedefs="".join(storageStructs.values()),
           lastStructName="float" if self._c99StructName() == "Ct" else self._c99StructName(),
           initCode="\n".join(initCode),
           fillCode="\n".join(fillCode),
           combineCode="\n".join(combineCode),
           quantities="".join(derivedFieldExprs.values()),
           jsonCode="\n".join(jsonCode),
           comma=", " if len(inputFieldNames) > 0 else "",
           inputList=", ".join(norm for norm, name in inputFieldNames.items()),
           gpuList=", ".join("gpu_" + name for norm, name in inputFieldNames.items()),
           inputListId=", ".join(norm + "[id]" for norm, name in inputFieldNames.items()),
           inputArgList=", ".join(inputFieldTypes[name] + " " + norm for norm, name in inputFieldNames.items()),
           inputArgStarList=", ".join(inputFieldTypes[name] + "* " + norm for norm, name in inputFieldNames.items()),
           weightVarDeclarations="".join("  float " + n + ";\n" for n in weightVars if n != "weight_0"),
           tmpVarDeclarations="".join("    " + t + " " + n + ";\n" for n, t in tmpVarTypes.items()),
           copyTestData="".join('''    float* gpu_{1};
    errorCheck(cudaMalloc((void**)&gpu_{1}, numDataPoints * sizeof(float)));
    errorCheck(cudaMemcpy(gpu_{1}, {0}, numDataPoints * sizeof(float), cudaMemcpyHostToDevice));
'''.format(norm, name) for norm, name in inputFieldNames.items()),
           freeTestData="".join('''    errorCheck(cudaFree(gpu_{0}));
'''.format(name) for name in inputFieldNames.values()),
           initTestData="".join('''  float {0}[10] = {{{1}}};
'''.format(norm, ", ".join(str(float(x)) + "f"
                           for x in (testData[name] if isinstance(testData, dict) else testData)))
                                for norm, name in inputFieldNames.items()),
           startComment="/*" if commentMain else "",
           endComment="*/" if commentMain else ""
           )

        if split:
            ifndefIndex = out.find("#ifndef")
            splitIndex = out.find("/////////////////////////////////////////////////////////////////// implementations")
            endifIndex = out.find("#endif")
            doth = out[:splitIndex] + out[endifIndex:]
            dotcu = out[:ifndefIndex] + "#include \"" + namespaceName + ".h\"\n\n" + out[splitIndex:endifIndex]
            return doth, dotcu

        else:
            return out

    def fillpycuda(self, length=None, **exprs):
        import numpy
        import pycuda.autoinit
        import pycuda.driver
        import pycuda.compiler
        import pycuda.gpuarray

        parser = C99SourceToAst()
        generator = C99AstToSource()

        inputFieldNames = OrderedDict()
        inputFieldTypes = {}
        derivedFieldTypes = {}
        derivedFieldExprs = OrderedDict()
        storageStructs = OrderedDict()
        initCode = []
        fillCode = []
        combineCode = []
        jsonCode = []
        weightVars = ["weight_0"]
        weightVarStack = ("weight_0",)
        tmpVarTypes = {}

        inputArrays = {}
        for name, expr in exprs.items():
            if isinstance(expr, pycuda.driver.DeviceAllocation):
                inputArrays[name] = expr
            elif isinstance(expr, numpy.ndarray):
                inputArrays[name] = expr.astype(numpy.float32)
                if len(inputArrays[name].shape) != 1:
                    raise ValueError("Numpy arrays must be one-dimensional")
            elif not isinstance(expr, basestring) and hasattr(expr, "__iter__"):
                inputArrays[name] = numpy.array(expr).astype(numpy.float32)
            else:
                self._cudaAddExpr(
                    parser,
                    generator,
                    name,
                    expr,
                    inputFieldNames,
                    inputFieldTypes,
                    derivedFieldTypes,
                    derivedFieldExprs)

        self._cudaGenerateCode(
            parser,
            generator,
            inputFieldNames,
            inputFieldTypes,
            derivedFieldTypes,
            derivedFieldExprs,
            storageStructs,
            initCode,
            (("var",
              "(*aggregator)"),
             ),
            4,
            fillCode,
            (("var",
              "(*aggregator)"),
             ),
            4,
            combineCode,
            (("var",
              "(*total)"),
             ),
            (("var",
              "(*item)"),
             ),
            4,
            jsonCode,
            (("var",
              "(*aggregator)"),
             ),
            6,
            weightVars,
            weightVarStack,
            tmpVarTypes,
            False)

        arguments = []
        for name in inputFieldNames.values():
            if name not in inputArrays:
                raise ValueError("no input supplied for \"" + name + "\"")
            if isinstance(inputArrays[name], numpy.ndarray):
                if length is None or inputArrays[name].shape[0] < length:
                    length = inputArrays[name].shape[0]
                arguments.append(pycuda.driver.In(inputArrays[name]))
            else:
                arguments.append(inputArrays[name])

        if length is None:
            raise ValueError(
                "no arrays specified as input fields in the aggregator to get length from "
                "(and length not specified explicitly)")

        module = pycuda.compiler.SourceModule(self.cuda(namespace=False, writeSize=True))

        numThreadsPerBlock = min(pycuda.driver.Context.get_device().get_attribute(
            pycuda.driver.device_attribute.MAX_THREADS_PER_BLOCK), length)
        numBlocks = int(math.ceil(float(length) / float(numThreadsPerBlock)))
        numAggregators = numThreadsPerBlock

        write_size = module.get_function("write_size")
        initialize = module.get_function("initialize")
        fillAll = module.get_function("fillAll")
        extract = module.get_function("extract")

        aggregatorSize = pycuda.gpuarray.empty((), dtype=numpy.uintp)
        write_size(aggregatorSize, block=(1, 1, 1), grid=(1, 1))
        pycuda.driver.Context.synchronize()
        aggregatorSize = int(aggregatorSize.get())

        aggregators = pycuda.driver.InOut(numpy.zeros(numAggregators * aggregatorSize, dtype=numpy.uint8))
        result = numpy.zeros(aggregatorSize, dtype=numpy.uint8)

        initialize(aggregators, numpy.intc(numAggregators), block=(numThreadsPerBlock, 1, 1), grid=(1, 1))

        fillAll(aggregators, numpy.intc(numAggregators), *(arguments + [numpy.intc(length)]),
                block=(numThreadsPerBlock, 1, 1), grid=(numBlocks, 1))

        extract(
            aggregators,
            numpy.intc(numAggregators),
            pycuda.driver.Out(result),
            block=(
                numThreadsPerBlock,
                1,
                1),
            grid=(
                1,
                1))

        pycuda.driver.Context.synchronize()
        self._cudaUnpackAndFill(result.tostring(), False, 4)    # TODO: determine bigendian, alignment and use them!

    def _cppExpandPrefix(self, *prefix):
        return self._c99ExpandPrefix(*prefix)

    def _c99ExpandPrefix(self, *prefix):
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

    def _clingExpandPrefix(self, obj, *prefix):
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

    def _cppNormalizeInputName(self, key):
        return self._c99NormalizeInputName(key)

    def _c99NormalizeInputName(self, key):
        if re.match("^[a-zA-Z0-9]*$", key) is not None:
            return "input_" + key
        else:
            return "input_" + base64.b64encode(key).replace("+", "_1").replace("/", "_2").replace("=", "")

    def _cppNormalizeExpr(self, ast, inputFieldNames, inputFieldTypes, weightVar):
        return self._c99NormalizeExpr(ast, inputFieldNames, inputFieldTypes, weightVar)

    def _c99NormalizeExpr(self, ast, inputFieldNames, inputFieldTypes, weightVar):
        # interpret raw identifiers as tree field names IF they're in the tree (otherwise, leave them alone)
        if isinstance(ast, c_ast.ID):
            if weightVar is not None and ast.name == "weight":
                ast.name = weightVar
            elif ast.name in inputFieldTypes:
                norm = self._cppNormalizeInputName(ast.name)
                inputFieldNames[norm] = ast.name
                if inputFieldTypes[ast.name].endswith("*"):
                    norm = "(" + ("*" * inputFieldTypes[ast.name].count("*")) + norm + ")"
                ast.name = norm

        elif isinstance(ast, c_ast.FuncCall):
            # t("field name") for field names that aren't valid C identifiers
            if isinstance(ast.name, c_ast.ID) and ast.name.name == "t" and isinstance(ast.args, c_ast.ExprList) and \
                    len(ast.args.exprs) == 1 and isinstance(ast.args.exprs[0], c_ast.Constant) and \
                    ast.args.exprs[0].type == "string":
                ast = self._cppNormalizeExpr(
                    c_ast.ID(
                        jsonlib.loads(
                            ast.args.exprs[0].value)),
                    inputFieldNames,
                    inputFieldTypes,
                    weightVar)
            # ordinary function: don't translate the name (let function names live in
            # a different namespace from variables)
            elif isinstance(ast.name, c_ast.ID):
                if ast.args is not None:
                    ast.args = self._cppNormalizeExpr(ast.args, inputFieldNames, inputFieldTypes, weightVar)
            # weird function: calling the result of an evaluation, probably an overloaded operator() in C++
            else:
                ast.name = self._cppNormalizeExpr(ast.name, inputFieldNames, inputFieldTypes, weightVar)
                if ast.args is not None:
                    ast.args = self._cppNormalizeExpr(ast.args, inputFieldNames, inputFieldTypes, weightVar)

        # only the top (x) of a dotted expression (x.y.z) should be interpreted as a field name
        elif isinstance(ast, c_ast.StructRef):
            self._cppNormalizeExpr(ast.name, inputFieldNames, inputFieldTypes, weightVar)

        # anything else
        else:
            for fieldName, fieldValue in ast.children():
                m = re.match(r"([^[]+)\[([0-9]+)\]", fieldName)
                if m is not None:
                    tmp = getattr(ast, m.group(1))
                    tmp.__setitem__(int(m.group(2)), self._cppNormalizeExpr(
                        fieldValue, inputFieldNames, inputFieldTypes, weightVar))
                else:
                    setattr(
                        ast,
                        fieldName,
                        self._cppNormalizeExpr(
                            fieldValue,
                            inputFieldNames,
                            inputFieldTypes,
                            weightVar))

        return ast

    def _cudaNormalizeExpr(self, ast, inputFieldNames, inputFieldTypes, weightVar, derivedFieldExprs, intermediates):
        if isinstance(ast, c_ast.ID):
            if weightVar is not None and ast.name == "weight":
                ast.name = weightVar
            elif ast.name in derivedFieldExprs:
                norm = "quantity_" + ast.name
                ast.name = norm
            elif ast.name not in intermediates:
                norm = "input_" + ast.name
                inputFieldNames[norm] = ast.name
                inputFieldTypes[ast.name] = "float"
                ast.name = norm

        elif isinstance(ast, c_ast.Decl):
            intermediates.add(ast.name)
            self._cudaNormalizeExpr(
                ast.init,
                inputFieldNames,
                inputFieldTypes,
                weightVar,
                derivedFieldExprs,
                intermediates)

        elif isinstance(ast, c_ast.FuncCall):
            # ordinary function: don't translate the name (let function names live in
            # a different namespace from variables)
            if isinstance(ast.name, c_ast.ID):
                if ast.args is not None:
                    ast.args = self._cudaNormalizeExpr(
                        ast.args,
                        inputFieldNames,
                        inputFieldTypes,
                        weightVar,
                        derivedFieldExprs,
                        intermediates)
            # weird function: calling the result of an evaluation, probably an overloaded operator() in C++
            else:
                ast.name = self._cudaNormalizeExpr(
                    ast.name,
                    inputFieldNames,
                    inputFieldTypes,
                    weightVar,
                    derivedFieldExprs,
                    intermediates)
                if ast.args is not None:
                    ast.args = self._cudaNormalizeExpr(
                        ast.args,
                        inputFieldNames,
                        inputFieldTypes,
                        weightVar,
                        derivedFieldExprs,
                        intermediates)

        # only the top (x) of a dotted expression (x.y.z) should be interpreted as a field name
        elif isinstance(ast, c_ast.StructRef):
            self._cudaNormalizeExpr(
                ast.name,
                inputFieldNames,
                inputFieldTypes,
                weightVar,
                derivedFieldExprs,
                intermediates)

        # anything else
        else:
            for fieldName, fieldValue in ast.children():
                m = re.match(r"([^[]+)\[([0-9]+)\]", fieldName)
                if m is not None:
                    tmp = getattr(ast, m.group(1))
                    tmp.__setitem__(int(m.group(2)),
                                    self._cudaNormalizeExpr(fieldValue, inputFieldNames, inputFieldTypes, weightVar,
                                                            derivedFieldExprs, intermediates))
                else:
                    setattr(
                        ast,
                        fieldName,
                        self._cudaNormalizeExpr(
                            fieldValue,
                            inputFieldNames,
                            inputFieldTypes,
                            weightVar,
                            derivedFieldExprs,
                            intermediates))

        return ast

    def _cppQuantityExpr(self, parser, generator, inputFieldNames, inputFieldTypes,
                         derivedFieldTypes, derivedFieldExprs, weightVar):
        return self._c99QuantityExpr(parser, generator, inputFieldNames, inputFieldTypes,
                                     derivedFieldTypes, derivedFieldExprs, weightVar)

    def _c99QuantityExpr(self, parser, generator, inputFieldNames, inputFieldTypes,
                         derivedFieldTypes, derivedFieldExprs, weightVar):
        if weightVar is not None:
            if not isinstance(self.transform.expr, basestring):
                raise ContainerException("Count.transform must be provided as a C99 string when used with Cling")
            try:
                ast = parser(self.transform.expr)
            except Exception as err:
                raise SyntaxError("""Couldn't parse C99 expression "{0}": {1}""".format(self.transform.expr, str(err)))
        else:
            if not isinstance(self.quantity.expr, basestring):
                raise ContainerException(self.name + ".quantity must be provided as a C99 string when used with Cling")
            try:
                ast = parser(self.quantity.expr)
            except Exception as err:
                raise SyntaxError("""Couldn't parse C99 expression "{0}": {1}""".format(self.quantity.expr, str(err)))

        ast = [self._cppNormalizeExpr(x, inputFieldNames, inputFieldTypes, weightVar) for x in ast]

        if len(ast) == 1 and isinstance(ast[0], c_ast.ID):
            return generator(ast)

        else:
            normexpr = generator(ast)
            derivedFieldName = None
            for name, expr in derivedFieldExprs.items():
                if expr == normexpr:
                    derivedFieldName = name
                    break

            if derivedFieldName is None:
                derivedFieldName = "quantity_" + str(len(derivedFieldExprs))
                if len(ast) > 1:
                    derivedFieldExprs[derivedFieldName] = "      {\n        " + ";\n        ".join(
                        generator(x) for x in ast[:-1]) + ";\n        " + \
                                                          derivedFieldName + " = " + \
                                                          generator(ast[-1]) + ";\n      }\n"
                else:
                    derivedFieldExprs[derivedFieldName] = "      " + derivedFieldName + " = " + normexpr + ";\n"

                if self.name == "Categorize":
                    derivedFieldTypes[derivedFieldName] = "std::string"
                elif self.name == "Bag":
                    if self.range == "S":
                        derivedFieldTypes[derivedFieldName] = "std::string"
                    elif self.range == "N":
                        derivedFieldTypes[derivedFieldName] = "double"
                    else:
                        derivedFieldTypes[derivedFieldName] = self.range
                else:
                    derivedFieldTypes[derivedFieldName] = "double"

            return derivedFieldName

    def _cudaQuantityExpr(self, parser, generator, inputFieldNames, inputFieldTypes,
                          derivedFieldTypes, derivedFieldExprs, weightVar):
        if weightVar is not None:
            if not isinstance(self.transform.expr, basestring):
                raise ContainerException("Count.transform must be provided as a C99 string when used with CUDA")
            try:
                ast = parser(self.transform.expr)
            except Exception as err:
                raise SyntaxError("""Couldn't parse C99 expression "{0}": {1}""".format(self.transform.expr, str(err)))
        else:
            if not isinstance(self.quantity.expr, basestring):
                raise ContainerException(self.name + ".quantity must be provided as a C99 string when used with CUDA")
            try:
                ast = parser(self.quantity.expr)
            except Exception as err:
                raise SyntaxError("""Couldn't parse C99 expression "{0}": {1}""".format(self.quantity.expr, str(err)))

        intermediates = set()
        ast = [
            self._cudaNormalizeExpr(
                x,
                inputFieldNames,
                inputFieldTypes,
                weightVar,
                derivedFieldExprs,
                intermediates) for x in ast]

        if len(ast) == 1 and isinstance(ast[0], c_ast.ID):
            return generator(ast)

        else:
            normexpr = generator(ast)
            derivedFieldName = None
            for name, expr in derivedFieldExprs.items():
                if expr == normexpr:
                    derivedFieldName = name
                    break

            if derivedFieldName is None:
                derivedFieldName = "quantity_" + str(len(derivedFieldExprs))
                if len(ast) > 1:
                    derivedFieldExprs[derivedFieldName] = "    float " + derivedFieldName + \
                                                          ";\n    {\n      " + ";\n      ".join(
                        generator(x) for x in ast[:-1]) + ";\n      " + derivedFieldName + " = " + \
                                                          generator(ast[-1]) + ";\n    }\n"
                else:
                    derivedFieldExprs[derivedFieldName] = "    float " + derivedFieldName + " = " + normexpr + ";\n"

                derivedFieldTypes[derivedFieldName] = "float"

            return derivedFieldName

    def _clingAddExpr(self, parser, generator, name, expr, inputFieldNames,
                      inputFieldTypes, derivedFieldTypes, derivedFieldExprs):
        if not isinstance(expr, basestring):
            raise ContainerException("expressions like {0} must be provided as a C99 string".format(name))
        try:
            ast = parser(expr)
        except Exception as err:
            raise SyntaxError("""Couldn't parse C99 expression "{0}": {1}""".format(expr, str(err)))

        ast = [self._cppNormalizeExpr(x, inputFieldNames, inputFieldTypes, None) for x in ast]

        if len(ast) > 1:
            derivedFieldExprs[name] = "      auto " + name + " = [this]{\n        " + ";\n        ".join(
                generator(x) for x in ast[:-1]) + ";\n        return " + generator(ast[-1]) + ";\n      }();\n"
        else:
            derivedFieldExprs[name] = "      auto " + name + " = " + generator(ast[0]) + ";\n"
        derivedFieldTypes[name] = "auto"

    def _cudaAddExpr(self, parser, generator, name, expr, inputFieldNames,
                     inputFieldTypes, derivedFieldTypes, derivedFieldExprs):
        if not isinstance(expr, basestring):
            raise ContainerException("expressions like {0} must be provided as a C99 string".format(name))
        try:
            ast = parser(expr)
        except Exception as err:
            raise SyntaxError("""Couldn't parse C99 expression "{0}": {1}""".format(expr, str(err)))

        intermediates = set()
        ast = [
            self._cudaNormalizeExpr(
                x,
                inputFieldNames,
                inputFieldTypes,
                None,
                derivedFieldExprs,
                intermediates) for x in ast]

        if len(ast) > 1:
            derivedFieldExprs[name] = "    float quantity_" + name + ";\n    {\n      " + ";\n      ".join(
                generator(x) for x in ast[:-1]) + ";\n      quantity_" + name + " = " + \
                                      generator(ast[-1]) + ";\n    }\n"
        else:
            derivedFieldExprs[name] = "    float quantity_" + name + " = " + generator(ast[0]) + ";\n"
        derivedFieldTypes[name] = "float"

    def _cppStorageType(self):
        return self._c99StorageType()

    def _c99StorageType(self):
        return self._c99StructName()

    def _cudaStorageType(self):
        return self._c99StructName()

    def fillnumpy(self, data):
        self._checkForCrossReferences()
        self._numpy(data, 1.0, [None])

    def _checkNPQuantity(self, q, shape):
        import numpy
        if isinstance(q, (list, tuple)):
            q = numpy.array(q)
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

    def fillsparksql(self, df):
        converter = df._sc._jvm.org.dianahep.histogrammar.sparksql.pyspark.AggregatorConverter()
        agg = self._sparksql(df._sc._jvm, converter)
        result = converter.histogrammar(df._jdf, agg)
        delta = Factory.fromJson(jsonlib.loads(result.toJsonString()))
        self += delta

# useful functions


unweighted = named("unweighted", lambda datum: 1.0)
identity = named("identity", lambda x: x)
square = named("square", lambda x: x)

# functions for Spark's RDD.aggregate


def increment(container, datum):
    """Increment function for Apache Spark's ``aggregate`` method.

    Typical use: ``filledHistogram = datasetRDD.aggregate(initialHistogram, increment, combine)``
    where ``datasetRDD`` is a collection of ``initialHistogram``'s input type.
    """
    container.fill(datum)
    return container


def combine(container1, container2):
    """Combine function for Apache Spark's ``aggregate`` method.

    Typical use: ``filledHistogram = datasetRDD.aggregate(initialHistogram)(increment, combine)``
    where ``datasetRDD`` is a collection of ``initialHistogram``'s input type.
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
