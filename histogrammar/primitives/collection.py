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
    @staticmethod
    def ed(entries, **pairs):
        if entries < 0.0:
            raise ContainerException("entries ({}) cannot be negative".format(entries))

        out = Label(**pairs)
        out.entries = float(entries)
        return out

    @staticmethod
    def ing(**pairs):
        return Label(**pairs)

    def __init__(self, **pairs):
        if any(not isinstance(x, basestring) for x in pairs.keys()):
            raise ContainerException("all Label keys must be strings")
        if len(pairs) < 1:
            raise ContainerException("at least one pair required")
        contentType = list(pairs.values())[0].name
        if any(x.name != contentType for x in pairs.values()):
            raise ContainerException("all Label values must have the same type")

        self.entries = 0.0
        self.pairs = pairs

        super(Label, self).__init__()

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
            return out

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
    @staticmethod
    def ed(entries, **pairs):
        if entries < 0.0:
            raise ContainerException("entries ({}) cannot be negative".format(entries))

        out = UntypedLabel(**pairs)
        out.entries = float(entries)
        return out

    @staticmethod
    def ing(**pairs):
        return UntypedLabel(**pairs)

    def __init__(self, **pairs):
        if any(not isinstance(x, basestring) for x in pairs.keys()):
            raise ContainerException("all UntypedLabel keys must be strings")

        self.entries = 0.0
        self.pairs = pairs

        super(UntypedLabel, self).__init__()

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
            return out

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

            return UntypedLabel.ed(entries, **pairs)

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
    @staticmethod
    def ed(entries, *values):
        if entries < 0.0:
            raise ContainerException("entries ({}) cannot be negative".format(entries))

        out = Index(*values)
        out.entries = float(entries)
        return out

    @staticmethod
    def ing(*values):
        return Index(*values)

    def __init__(self, *values):
        if len(values) < 1:
            raise ContainerException("at least one value required")
        contentType = values[0].name
        if any(x.name != contentType for x in values):
            raise ContainerException("all Index values must have the same type")

        self.entries = 0.0
        self.values = values

        super(Index, self).__init__()

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
            return out

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

            return Index.ed(entries, *values)

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
    @staticmethod
    def ed(entries, *values):
        if entries < 0.0:
            raise ContainerException("entries ({}) cannot be negative".format(entries))

        out = Branch(*values)
        out.entries = float(entries)
        return out

    @staticmethod
    def ing(*values):
        return Branch(*values)

    def __init__(self, *values):
        if len(values) < 1:
            raise ContainerException("at least one value required")

        self.entries = 0.0
        self.values = values

        for i, x in enumerate(values):
            setattr(self, "i" + str(i), x)

        super(Branch, self).__init__()

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
            return out

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
