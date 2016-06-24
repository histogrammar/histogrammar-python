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

from histogrammar.defs import *
from histogrammar.util import *

class Collection(object): pass

################################################################ Label

class Label(Factory, Container, Collection):
    """Accumulate any number of aggregators of the same type and label them with strings. Every sub-aggregator is filled with every input datum.

    This primitive simulates a directory of aggregators. For sub-directories, nest collections within the Label collection.

    Note that all sub-aggregators within a Label must have the _same type_ (e.g. histograms of different binnings, but all histograms). To collect objects of _different types_ with string-based look-up keys, use [UntypedLabel](#untypedlabel-directory-of-different-types).

    To collect aggregators of the _same type_ without naming them, use [Index](#index-list-with-integer-keys). To collect aggregators of _different types_ without naming them, use [Branch](#branch-tuple-of-different-types).

    In strongly typed languages, the restriction to a single type allows nested objects to be extracted without casting.
    """
    @staticmethod
    def ed(entries, **pairs):
        """
        * `entries` (double) is the number of entries.
        * `pairs` (list of string, past-tense aggregator pairs) is the collection of filled aggregators.
        * `pairsMap` (map of the above, probably a hashmap) is intended for fast look-ups.
        """
        if not isinstance(entries, (int, long, float)):
            raise TypeError("entries ({}) must be a number".format(entries))
        if not all(isinstance(k, basestring) and isinstance(v, Container) for k, v in pairs.items()):
            raise TypeError("pairs ({}) must be a dict from strings to Containers".format(pairs))
        if entries < 0.0:
            raise ValueError("entries ({}) cannot be negative".format(entries))

        out = Label(**pairs)
        out.entries = float(entries)
        return out.specialize()

    @staticmethod
    def ing(**pairs):
        """Synonym for ``__init__``."""
        return Label(**pairs)

    def __init__(self, **pairs):
        """
        * `pairs` (list of string, present-tense aggregator pairs) is the collection of aggregators to fill.
        * `pairsMap` (map of the above, probably a hashmap) is intended for fast look-ups.
        * `entries` (mutable double) is the number of entries, initially 0.0.
        """
        if not all(isinstance(k, basestring) and isinstance(v, Container) for k, v in pairs.items()):
            raise TypeError("pairs ({}) must be a dict from strings to Containers".format(pairs))
        if any(not isinstance(x, basestring) for x in pairs.keys()):
            raise ValueError("all Label keys must be strings")
        if len(pairs) < 1:
            raise ValueError("at least one pair required")

        contentType = list(pairs.values())[0].name
        if any(x.name != contentType for x in pairs.values()):
            raise ContainerException("all Label values must have the same type")

        self.entries = 0.0
        self.pairs = pairs

        super(Label, self).__init__()
        self.specialize()

    @property
    def pairsMap(self): return self.pairs
    @property
    def size(self): return len(self.pairs)
    @property
    def keys(self): return self.pairs.keys()
    @property
    def values(self): return list(self.pairs.values())
    @property
    def keySet(self): return set(self.pairs.keys())

    def __call__(self, x, *rest):
        if len(rest) == 0:
            return self.pairs[x]
        else:
            return self.pairs[x](*rest)
    def get(self, x): return self.pairs.get(x, None)
    def getOrElse(self, x, default): return self.pairs.get(x, default)

    def zero(self): return Label(**{k: v.zero() for k, v in self.pairs.items()})

    def __add__(self, other):
        if isinstance(other, Label):
            if self.keySet != other.keySet:
                raise ContainerException("cannot add Labels because keys differ:\n    {}\n    {}".format(", ".join(sorted(self.keys)), ", ".join(sorted(other.keys))))

            out = Label(**{k: self(k) + other(k) for k in self.keys})
            out.entries = self.entries + other.entries
            return out.specialize()

        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()
        for x in self.values:
            x.fill(datum, weight)
        # no possibility of exception from here on out (for rollback)
        self.entries += weight

    @property
    def children(self):
        return self.values

    def toJsonFragment(self, suppressName): return {
        "entries": floatToJson(self.entries),
        "type": self.values[0].name,
        "data": {k: v.toJsonFragment(False) for k, v in self.pairs.items()},
        }

    @staticmethod
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "type", "data"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "Label.entries")

            if isinstance(json["type"], basestring):
                factory = Factory.registered[json["type"]]
            else:
                raise JsonFormatException(json, "Label.type")

            if isinstance(json["data"], dict):
                pairs = {k: factory.fromJsonFragment(v, None) for k, v in json["data"].items()}
            else:
                raise JsonFormatException(json, "Label.data")

            return Label.ed(entries, **pairs)

        else:
            raise JsonFormatException(json, "Label")

    def __repr__(self):
        return "<Label values={} size={}>".format(self.values[0].name, self.size)

    def __eq__(self, other):
        return isinstance(other, Label) and numeq(self.entries, other.entries) and self.pairs == other.pairs

    def __hash__(self):
        return hash((self.entries, tuple(sorted(self.pairs.items()))))

Factory.register(Label)

################################################################ UntypedLabel

class UntypedLabel(Factory, Container, Collection):
    """Accumulate any number of aggregators of any type and label them with strings. Every sub-aggregator is filled with every input datum.

    This primitive simulates a directory of aggregators. For sub-directories, nest collections within the UntypedLabel.

    Note that sub-aggregators within an UntypedLabel may have _different types_. In strongly typed languages, this flexibility poses a problem: nested objects must be type-cast before they can be used. To collect objects of the _same type_ with string-based look-up keys, use [Label](#label-directory-with-string-based-keys).

    To collect aggregators of the _same type_ without naming them, use [Index](#index-list-with-integer-keys). To collect aggregators of _different types_ without naming them, use [Branch](#branch-tuple-of-different-types).
    """

    @staticmethod
    def ed(entries, **pairs):
        """
        * `entries` (double) is the number of entries.
        * `pairs` (list of string, past-tense aggregator pairs) is the collection of filled aggregators.
        * `pairsMap` (map of the above, probably a hashmap) is intended for fast look-ups.
        """
        if not isinstance(entries, (int, long, float)):
            raise TypeError("entries ({}) must be a number".format(entries))
        if not all(isinstance(k, basestring) and isinstance(v, Container) for k, v in pairs.items()):
            raise TypeError("pairs ({}) must be a dict from strings to Containers".format(pairs))
        if entries < 0.0:
            raise ValueError("entries ({}) cannot be negative".format(entries))

        out = UntypedLabel(**pairs)
        out.entries = float(entries)
        return out.specialize()

    @staticmethod
    def ing(**pairs):
        """Synonym for ``__init__``."""
        return UntypedLabel(**pairs)

    def __init__(self, **pairs):
        """
        * `pairs` (list of string, present-tense aggregator pairs) is the collection of aggregators to fill.
        * `pairsMap` (map of the above, probably a hashmap) is intended for fast look-ups.
        * `entries` (mutable double) is the number of entries, initially 0.0.
        """
        if not all(isinstance(k, basestring) and isinstance(v, Container) for k, v in pairs.items()):
            raise TypeError("pairs ({}) must be a dict from strings to Containers".format(pairs))

        self.entries = 0.0
        self.pairs = pairs

        super(UntypedLabel, self).__init__()
        self.specialize()

    @property
    def pairsMap(self): return self.pairs
    @property
    def size(self): return len(self.pairs)
    @property
    def keys(self): return self.pairs.keys()
    @property
    def values(self): return list(self.pairs.values())
    @property
    def keySet(self): return set(self.pairs.keys())

    def __call__(self, x, *rest):
        if len(rest) == 0:
            return self.pairs[x]
        else:
            return self.pairs[x](*rest)
    def get(self, x): return self.pairs.get(x, None)
    def getOrElse(self, x, default): return self.pairs.get(x, default)

    def zero(self): return UntypedLabel(**{k: v.zero() for k, v in self.pairs.items()})

    def __add__(self, other):
        if isinstance(other, UntypedLabel):
            if self.keySet != other.keySet:
                raise ContainerException("cannot add UntypedLabels because keys differ:\n    {}\n    {}".format(", ".join(sorted(self.keys)), ", ".join(sorted(other.keys))))

            out = UntypedLabel(**{k: self(k) + other(k) for k in self.keys})
            out.entries = self.entries + other.entries
            return out.specialize()

        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()
        for x in self.values:
            x.fill(datum, weight)
        # no possibility of exception from here on out (for rollback)
        self.entries += weight

    @property
    def children(self):
        return self.values

    def toJsonFragment(self, suppressName): return {
        "entries": floatToJson(self.entries),
        "data": {k: {"type": v.name, "data": v.toJsonFragment(False)} for k, v in self.pairs.items()},
        }

    @staticmethod
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "data"]):
            if isinstance(json["entries"], (int, long, float)):
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
                        raise JsonFormatException(k, "UntypedLabel.data {}".format(v))

            else:
                raise JsonFormatException(json, "UntypedLabel.data")

            return UntypedLabel.ed(entries, **pairs).specialize()

        else:
            raise JsonFormatException(json, "UntypedLabel")

    def __repr__(self):
        return "<UntypedLabel size={}>".format(self.size)

    def __eq__(self, other):
        return isinstance(other, UntypedLabel) and numeq(self.entries, other.entries) and self.pairs == other.pairs

    def __hash__(self):
        return hash((self.entries, tuple(sorted(self.pairs.items()))))

Factory.register(UntypedLabel)

################################################################ Index

class Index(Factory, Container, Collection):
    """Accumulate any number of aggregators of the same type in a list. Every sub-aggregator is filled with every input datum.

    This primitive provides an anonymous collection of aggregators (unless the integer index is taken to have special meaning, but generally such bookkeeping should be encoded in strings). Indexes can be nested to create two-dimensional ordinal grids of aggregators. (Use [Bin](#bin-regular-binning-for-histograms) if the space is to have a metric interpretation.)

    Note that all sub-aggregators within an Index must have the _same type_ (e.g. histograms of different binnings, but all histograms). To collect objects of _different types,_ still indexed by integer, use [Branch](#branch-tuple-of-different-types).

    To collect aggregators of the _same type_ with string-based labels, use [Label](#label-directory-with-string-based-keys). To collect aggregators of _different types_ with string-based labels, use [UntypedLabel](#untypedlabel-directory-of-different-types).

    In strongly typed languages, the restriction to a single type allows nested objects to be extracted without casting.
    """

    @staticmethod
    def ed(entries, *values):
        """
        * `entries` (double) is the number of entries.
        * `values` (list of past-tense aggregators) is the collection of filled aggregators.
        """
        if not isinstance(entries, (int, long, float)):
            raise TypeError("entries ({}) must be a number".format(entries))
        if not all(isinstance(v, Container) for v in values):
            raise TypeError("values ({}) must be a list of Containers".format(values))
        if entries < 0.0:
            raise ValueError("entries ({}) cannot be negative".format(entries))

        out = Index(*values)
        out.entries = float(entries)
        return out.specialize()

    @staticmethod
    def ing(*values):
        """Synonym for ``__init__``."""
        return Index(*values)

    def __init__(self, *values):
        """
        * `values` (list of present-tense aggregators) is the collection of aggregators to fill.
        * `entries` (mutable double) is the number of entries, initially 0.0.
        """
        if not all(isinstance(v, Container) for v in values):
            raise TypeError("values ({}) must be a list of Containers".format(values))
        if len(values) < 1:
            raise ContainerException("at least one value required")
        contentType = values[0].name
        if any(x.name != contentType for x in values):
            raise ValueError("all Index values must have the same type")

        self.entries = 0.0
        self.values = values

        super(Index, self).__init__()
        self.specialize()

    @property
    def size(self): return len(self.values)

    def __call__(self, i, *rest):
        if len(rest) == 0:
            return self.values[i]
        else:
            return self.values[i](*rest)
    def get(self, i):
        if i < 0 or i >= len(self.values):
            return None
        else:
            return self.values[i]

    def getOrElse(self, x, default):
        if i < 0 or i >= len(self.values):
            return default
        else:
            return self.values[i]

    def zero(self): return Index(*[x.zero() for x in self.values])

    def __add__(self, other):
        if isinstance(other, Index):
            if self.size != other.size:
                raise ContainerException("cannot add Indexes because they have different sizes: ({} vs {})".format(self.size, other.size))

            out = Index(*[x + y for x, y in zip(self.values, other.values)])
            out.entries = self.entries + other.entries
            return out.specialize()

        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()
        for x in self.values:
            x.fill(datum, weight)
        # no possibility of exception from here on out (for rollback)
        self.entries += weight

    @property
    def children(self):
        return self.values

    def toJsonFragment(self, suppressName): return {
        "entries": floatToJson(self.entries),
        "type": self.values[0].name,
        "data": [x.toJsonFragment(False) for x in self.values],
        }

    @staticmethod
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "type", "data"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "Index.entries")

            if isinstance(json["type"], basestring):
                factory = Factory.registered[json["type"]]
            else:
                raise JsonFormatException(json, "Index.type")

            if isinstance(json["data"], list):
                values = [factory.fromJsonFragment(x, None) for x in json["data"]]
            else:
                raise JsonFormatException(json, "Index.data")

            return Index.ed(entries, *values).specialize()

        else:
            raise JsonFormatException(json, "Index")

    def __repr__(self):
        return "<Index values={} size={}>".format(self.values[0].name, self.size)

    def __eq__(self, other):
        return isinstance(other, Index) and numeq(self.entries, other.entries) and self.values == other.values

    def __hash__(self):
        return hash((self.entries, tuple(self.values)))

Factory.register(Index)

################################################################ Branch

class Branch(Factory, Container, Collection):
    """Accumulate aggregators of different types, indexed by i0 through i9. Every sub-aggregator is filled with every input datum.

       This primitive provides an anonymous collection of aggregators of _different types,_ usually for gluing together various statistics. For instance, if the following associates a sum of weights to every bin in a histogram,

       ```python
       Bin.ing(100, 0, 1, lambda d: d.x,
         Sum.ing(lambda d: d.weight))
       ```

       the following would associate the sum of weights and the sum of squared weights to every bin:

       ```python
       Bin.ing(100, 0, 1, lambda d: d.x,
         Branch.ing(Sum.ing(lambda d: d.weight),
                    Sum.ing(lambda d: d.weight**2)))
       ```

       Branch is a basic building block for complex aggregators. The limitation to ten branches, indexed from i0 to i9, is a concession to type inference in statically typed languages. It is not a fundamental limit, but the type-metaprogramming becomes increasingly complex as branches are added. Error messages may be convoluted as the compiler presents internals of the type-metaprogramming in response to a user's simple mistake.

       Therefore, individual implementations may allow more than ten branches, but the Histogrammar standard only requires ten.

       To collect an unlimited number of aggregators of the _same type_ without naming them, use [Index](#index-list-with-integer-keys). To collect aggregators of the _same type_ with string-based labels, use [Label](#label-directory-with-string-based-keys). To collect aggregators of _different types_ with string-based labels, use [UntypedLabel](#untypedlabel-directory-of-different-types).
       """

    @staticmethod
    def ed(entries, *values):
        """
        * `entries` (double) is the number of entries.
        * `values` (list of past-tense aggregators) is the collection of filled aggregators.
        """
        if not isinstance(entries, (int, long, float)):
            raise TypeError("entries ({}) must be a number".format(entries))
        if not all(isinstance(v, Container) for v in values):
            raise TypeError("values ({}) must be a list of Containers".format(values))
        if entries < 0.0:
            raise ValueError("entries ({}) cannot be negative".format(entries))

        out = Branch(*values)
        out.entries = float(entries)
        return out.specialize()

    @staticmethod
    def ing(*values):
        """Synonym for ``__init__``."""
        return Branch(*values)

    def __init__(self, *values):
        """
        * `values` (list of present-tense aggregators) is the collection of aggregators to fill.
        * `entries` (mutable double) is the number of entries, initially 0.0.
        """
        if not all(isinstance(v, Container) for v in values):
            raise TypeError("values ({}) must be a list of Containers".format(values))
        if len(values) < 1:
            raise ValueError("at least one value required")

        self.entries = 0.0
        self.values = values

        for i, x in enumerate(values):
            setattr(self, "i" + str(i), x)

        super(Branch, self).__init__()
        self.specialize()

    @property
    def size(self): return len(self.values)

    def __call__(self, i, *rest):
        if len(rest) == 0:
            return self.values[i]
        else:
            return self.values[i](*rest)
    def get(self, i):
        if i < 0 or i >= len(self.values):
            return None
        else:
            return self.values[i]

    def getOrElse(self, x, default):
        if i < 0 or i >= len(self.values):
            return default
        else:
            return self.values[i]

    def zero(self): return Branch(*[x.zero() for x in self.values])

    def __add__(self, other):
        if isinstance(other, Branch):
            if self.size != other.size:
                raise ContainerException("cannot add Branches because they have different sizes: ({} vs {})".format(self.size, other.size))

            out = Branch(*[x + y for x, y in zip(self.values, other.values)])
            out.entries = self.entries + other.entries
            return out.specialize()

        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()
        for x in self.values:
            x.fill(datum, weight)
        # no possibility of exception from here on out (for rollback)
        self.entries += weight

    @property
    def children(self):
        return self.values

    def toJsonFragment(self, suppressName): return {
        "entries": floatToJson(self.entries),
        "data": [{"type": x.name, "data": x.toJsonFragment(False)} for x in self.values],
        }

    @staticmethod
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "data"]):
            if isinstance(json["entries"], (int, long, float)):
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
                            raise JsonFormatException(x, "Branch.data {} type".format(i))
                        values.append(factory.fromJsonFragment(x["data"], None))

            else:
                raise JsonFormatException(json, "Branch.data")

            return Branch.ed(entries, *values)

        else:
            raise JsonFormatException(json, "Branch")
        
    def __repr__(self):
        return "<Branch {}>".format(" ".join("i" + str(i) + "=" + v.name for i, v in enumerate(self.values)))

    def __eq__(self, other):
        return isinstance(other, Branch) and numeq(self.entries, other.entries) and self.values == other.values

    def __hash__(self):
        return hash((self.entries, tuple(self.values)))

Factory.register(Branch)
