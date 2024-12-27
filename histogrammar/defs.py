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

import json as jsonlib
from pathlib import Path

import numpy

import histogrammar.version
from histogrammar.util import FillMethod, PlotMethod, basestring, named


class ContainerException(Exception):
    """Exception type for improperly configured containers."""


class InvalidJsonException(Exception):
    """Exception type for strings that cannot be parsed because they are not proper JSON."""

    def __init__(self, message):
        super().__init__(f"invalid JSON: {message}")


class JsonFormatException(Exception):
    """Exception type for unexpected JSON structure, thrown by ``fromJson`` methods."""

    def __init__(self, x, context):
        super().__init__(f"wrong JSON format for {context}: {jsonlib.dumps(x)}")


class Factory:
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
            # MB 20220517: warning, this is a slow function.
            # Adding functions to each object, ideally avoid this.
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
        path = Path(fileName)
        return Factory.fromJson(jsonlib.load(path.open()))

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
                        "cannot read a Histogrammar {} document with histogrammar-python version {}".format(
                            json["version"], histogrammar.version.version
                        )
                    )
            else:
                raise JsonFormatException(json["version"], "Factory.version")

            if isinstance(json["type"], basestring):
                name = json["type"]
            else:
                raise JsonFormatException(json["type"], "Factory.type")

            if name not in Factory.registered:
                raise JsonFormatException(
                    json,
                    "unrecognized container (is it a custom container " f"that hasn't been registered?): {name}",
                )

            return Factory.registered[name].fromJsonFragment(json["data"], None)

        raise JsonFormatException(json, "Factory")


class Container:
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
        """Create an empty container with the same parameters as this one. The original is unaffected."""
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
        # used by pickling
        state = dict(self.__dict__)
        for s in ["fill", "plot"]:
            # these states are set dynamically by FillMethod and PlotMethod, in factory.specialize().
            # MB 20220517: turned off specialize() for Count objects,
            # for which specialized fill and plot methods are not needed.
            if s in state:
                del state[s]
        return state

    def __setstate__(self, dict):
        # used by unpickling
        self.__dict__ = dict
        self.fill = FillMethod(self, self.fill)
        self.plot = PlotMethod(self, self.plot)

    def copy(self):
        """Copy this container, making a clone with no reference to the original."""
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
                raise ContainerException(f"cannot fill a tree that contains the same aggregator twice: {self}")
            memo.add(self)
            for child in self.children:
                child._checkForCrossReferences(memo)
            self._checkedForCrossReferences = True

    def toJsonFile(self, fileName):
        path = Path(fileName)
        return jsonlib.dump(self.toJson(), path.open(mode="w"))

    def toJsonString(self):
        return jsonlib.dumps(self.toJson())

    def toJson(self):
        """Convert this container to dicts and lists representing JSON (dropping its ``fill`` method).

        Note that the dicts and lists can be turned into a string with ``json.dumps``.
        """
        return {
            "type": self.name,
            "data": self.toJsonFragment(False),
            "version": histogrammar.version.specification,
        }

    def toJsonFragment(self, suppressName):
        """Used internally to convert the container to JSON without its ``"type"`` header."""
        raise NotImplementedError

    def toImmutable(self):
        """Return a copy of this container

        As though it was created by the ``ed`` function or from JSON (the \"immutable form\" in languages that
        support it, not Python).
        """
        return Factory.fromJson(self.toJson())

    def fillnumpy(self, data, weights=1.0):
        self._checkForCrossReferences()
        self._numpy(data, weights, shape=[None])

    def _checkNPQuantity(self, q, shape):
        if isinstance(q, (list, tuple)):
            q = numpy.array(q)
        assert isinstance(q, numpy.ndarray)
        assert len(q.shape) == 1
        if shape[0] is None:
            shape[0] = q.shape[0]
        else:
            assert q.shape[0] == shape[0]

    def _checkNPWeights(self, weights, shape):
        if isinstance(weights, numpy.ndarray):
            assert len(weights.shape) == 1
            assert weights.shape[0] == shape[0]

    def _makeNPWeights(self, weights, shape):
        if isinstance(weights, numpy.ndarray):
            return weights
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
