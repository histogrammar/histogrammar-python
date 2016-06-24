#!/usr/bin/env python

# Copyright 2016 Jim Pivarski
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
import random

from histogrammar.defs import *
from histogrammar.util import *

MIN_LONG = -2**63
MAX_LONG = 2**63 - 1

class Sample(Factory, Container):
    """Accumulate raw numbers, vectors of numbers, or strings, randomly replacing them with Reservoir Sampling when the number of values exceeds a limit.

    Sample collects raw values without attempting to group them by distinct value (as [Bag](#bag-accumulate-values-for-scatter-plots) does), up to a given maximum _number_ of entries (unlike [Limit](#limit-keep-detail-until-entries-is-large), which rolls over at a given total weight). The reason for the limit on Sample is purely to conserve memory.

    The maximum number of entries and the data type together determine the size of the working set. If new values are added after this set is full, individual values will be randomly chosen for replacement. The probability of replacement is proportional to an entry's weight and it decreases with time, such that the final sample is a representative subset of all observed values, without preference for early values or late values.

    This algorithm is known as weighted Reservoir Sampling, and it is non-deterministic. Each evaluation will likely result in a different final set.

    Specifically, the algorithm implemented here was described in ["Weighted random sampling with a reservoir," Pavlos S. Efraimidis and Paul G. Spirakis, _Information Processing Letters 97 (5): 181-185,_ 2005 (doi:10.1016/j.ipl.2005.11.003)](http://www.sciencedirect.com/science/article/pii/S002001900500298X).

    Although the user-defined function may return scalar numbers, fixed-dimension vectors of numbers, or categorical strings, it may not mix types. Different Sample primitives in an analysis tree may collect different types.
    """

    @staticmethod
    def ed(entries, limit, values, randomSeed=None):
        """
        * `entries` (double) is the number of entries.
        * `limit` (32-bit integer) is the maximum number of entries to store before replacement. This is a strict _number_ of entries, unaffected by weights.
        * `values` (list of quantity return type, double, double triples) is the set of collected values with their weights. Its size is at most `limit` and it may contain duplicates.
        * `randomSeed` (long integer or None) an optional random seed to make the sampling deterministic.
        * `randomGenerator` (random generator state or None) platform-dependent representation of the random generator's state if a `randomSeed` was provided. The random generator's sequence of values must be unaffected by any other random sampling elsewhere in the environment, including other Sampled instances.
        """
        if not isinstance(entries, (int, long, float)):
            raise TypeError("entries ({}) must be a number".format(entries))
        if not isinstance(limit, (int, long, float)):
            raise TypeError("limit ({}) must be a number".format(limit))
        if not isinstance(values, (list, tuple)) and not all(isinstance(v, (list, tuple)) and len(v) == 3 and isinstance(v[1], (int, long, float)) and isinstance(v[2], (int, long, float)) for v in values):
            raise TypeError("values ({}) must be a list of quantity return type, number, number triples".format(values))
        if randomSeed is not None and not isinstance(randomSeed, (int, long)):
            raise TypeError("randomSeed ({}) must be None or a number".format(randomSeed))
        if entries < 0.0:
            raise ValueError("entries ({}) cannot be negative".format(entries))
        out = Sample(limit, None, randomSeed)
        del out.reservoir
        out.entries = entries
        out._limit = limit
        out._values = values
        return out.specialize()

    @staticmethod
    def ing(limit, quantity, randomSeed=None):
        """Synonym for ``__init__``."""
        return Sample(limit, quantity, randomSeed)

    def __init__(self, limit, quantity, randomSeed=None):
        """
        * `limit` (32-bit integer) is the maximum number of entries to store before replacement. This is a strict _number_ of entries, unaffected by weights.
        * `quantity` (function returning a double, a vector of doubles, or a string) computes the quantity of interest from the data.
        * `randomSeed` (long integer or None) an optional random seed to make the sampling deterministic.
        * `entries` (mutable double) is the number of entries, initially 0.0.
        * `values` (mutable, list of quantity return type, double, double triplets) is the set of collected values with their weights and a random number (see algorithm below), sorted by the random number. Its size is at most `limit` and it may contain duplicates.
        * `randomGenerator` (random generator state or None) platform-dependent representation of the random generator's state if a `randomSeed` was provided. The random generator's sequence of values must be unaffected by any other random sampling elsewhere in the environment, including other Sampling instances.
        """
        if not isinstance(limit, (int, long, float)):
            raise TypeError("limit ({}) must be a number".format(limit))
        if randomSeed is not None and not isinstance(randomSeed, (int, long)):
            raise TypeError("randomSeed ({}) must be None or a number".format(randomSeed))
        if limit <= 0.0:
            raise ValueError("limit ({}) cannot be negative".format(limit))
        self.entries = 0.0
        self.quantity = serializable(quantity)
        self.reservoir = Reservoir(limit)
        if randomSeed is None:
            self.randomGenerator = None
        else:
            self.randomGenerator = random.Random(randomSeed)
        super(Sample, self).__init__()
        self._limit = limit
        self.specialize()
        

    @property
    def limit(self):
        if hasattr(self, "reservoir"):
            return self.reservoir.limit
        else:
            return self._limit

    @property
    def values(self):
        if hasattr(self, "reservoir"):
            return self.reservoir.values
        else:
            return self._values

    @property
    def size(self):
        if hasattr(self, "reservoir"):
            return self.reservoir.size
        else:
            return len(self._values)

    @property
    def isEmpty(self):
        if hasattr(self, "reservoir"):
            return self.reservoir.isEmpty
        else:
            return len(self._values) == 0

    def zero(self):
        if self.randomGenerator is None:
            newseed = None
        else:
            newseed = self.randomGenerator.randint(-2**63, 2**63 - 1)
        return Sample(self.limit, self.quantity, newseed)

    def __add__(self, other):
        if isinstance(other, Sample):
            if self.limit != other.limit:
                raise ContainerException("cannot add Ssample because limit differs ({} vs {})".format(self.limit, other.limit))

            if self.randomGenerator is not None and other.randomGenerator is not None:
                newSeed = self.randomGenerator.randint(MIN_LONG, MAX_LONG) + other.randomGenerator.randint(MIN_LONG, MAX_LONG)
                if newSeed > MAX_LONG:
                    newSeed -= MAX_LONG - MIN_LONG
                if newSeed < MIN_LONG:
                    newSeed += MAX_LONG - MIN_LONG
                newGenerator = random.Random(newSeed)
            elif self.randomGenerator is not None:
                newGenerator = random.Random(self.randomGenerator.randint(MIN_LONG, MAX_LONG))
            elif other.randomGenerator is not None:
                newGenerator = random.Random(other.randomGenerator.randint(MIN_LONG, MAX_LONG))
            else:
                newGenerator = None

            newreservoir = Reservoir(self.limit, *self.values)
            for y, weight in other.values:
                newreservoir.update(y, weight, newGenerator)
                
            out = Sample(self.limit, self.quantity, None)
            out.entries = self.entries + other.entries
            if hasattr(self, "reservoir"):
                out.reservoir = newreservoir
            else:
                del out.reservoir
                out._values = newreservoir.values
            out.randomGenerator = newGenerator
            return out.specialize()

        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()
        if weight > 0.0:
            q = self.quantity(datum)
            if not isinstance(q, (bool, int, long, float, basestring)) and not (isinstance(q, (list, tuple)) and all(isinstance(qi, (int, long, float)) for qi in q)):
                raise TypeError("function return value ({}) must be boolean, number, string, or list/tuple of numbers".format(q))

            self.reservoir.update(q, weight, self.randomGenerator)

            # no possibility of exception from here on out (for rollback)
            self.entries += weight

    @property
    def children(self):
        return []

    def toJsonFragment(self, suppressName): return maybeAdd({
        "entries": floatToJson(self.entries),
        "limit": floatToJson(self.limit),
        "values": [{"w": w, "v": y} for y, w in sorted(self.values, key=lambda y_w: y_w[0])],
        }, name=self.quantity.name, seed=self.randomGenerator.randint(MIN_LONG, MAX_LONG) if self.randomGenerator is not None else None)

    @staticmethod
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "limit", "values"], ["name", "seed"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = json["entries"]
            else:
                raise JsonFormatException(json["entries"], "Sample.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "Sample.name")

            if isinstance(json["limit"], (int, long, float)):
                limit = json["limit"]
            else:
                raise JsonFormatException(json["limit"], "Sample.limit")

            if isinstance(json["values"], list):
                values = []
                for i, wv in enumerate(json["values"]):
                    if isinstance(wv, dict) and hasKeys(wv.keys(), ["w", "v"]):
                        if isinstance(wv["w"], (int, long, float)):
                            w = float(wv["w"])
                        else:
                            raise JsonFormatException(wv["w"], "Sample.values {} w".format(i))

                        if isinstance(wv["v"], basestring):
                            v = wv["v"]
                        elif isinstance(wv["v"], (int, long, float)):
                            v = float(wv["v"])
                        elif isinstance(wv["v"], (list, tuple)):
                            for j, d in enumerate(wv["v"]):
                                if not isinstance(d, (int, long, float)):
                                    raise JsonFormatException(d, "Sample.values {} v {}".format(i, j))
                            v = tuple(map(float, wv["v"]))
                        else:
                            raise JsonFormatException(wv["v"], "Sample.values {} v".format(i))

                        values.append((v, w))

                    else:
                        raise JsonFormatException(wv, "Sample.values {}".format(i))

            else:
                raise JsonFormatException(json["values"], "Sample.values")

            if isinstance(json.get("seed", None), (int, long)):
                seed = json["seed"]
            elif json.get("seed", None) is None:
                seed = None
            else:
                raise JsonFormatException(json["seed"], "Sample.seed")

            out = Sample.ed(entries, limit, values, seed)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        else:
            raise JsonFormatException(json, "Sample")

    def __repr__(self):
        return "<Sample size={}>".format(self.size)

    def __eq__(self, other):
        return isinstance(other, Sample) and self.entries == other.entries and self.quantity == other.quantity and self.limit == other.limit and self.values == other.values and (self.randomGenerator is None) == (other.randomGenerator is None)

    def __hash__(self):
        return hash((self.entries, self.quantity, self.limit, tuple(self.values), (self.randomGenerator is None)))

Factory.register(Sample)
