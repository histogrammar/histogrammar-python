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

################################################################ Limit

class Limit(Factory, Container):
    @staticmethod
    def ed(entries, limit, contentType, value):
        if entries < 0.0:
            raise ContainerException("entries ({}) cannot be negative".format(entries))

        out = Limit(value, limit)
        out.entries = entries
        out.contentType = contentType
        return out.specialize()

    @staticmethod
    def ing(value, limit): return Limit(value, limit)

    def __init__(self, value, limit):
        self.entries = 0.0
        self.limit = limit
        if value is None:
            self.contentType = None
        else:
            self.contentType = value.name
        self.value = value
        super(Limit, self).__init__()
        self.specialize()

    @property
    def saturated(self): return self.value is None
    @property
    def get(self):
        if self.value is None:
            raise TypeError("get called on Limit whose value is None")
        return self.value
    def getOrElse(self, default):
        if self.value is None:
            return default
        else:
            return self.value

    def zero(self):
        return Limit.ed(0.0, self.limit, self.contentType, None if self.value is None else self.value.zero())

    def __add__(self, other):
        if isinstance(other, Limit):
            if self.limit != other.limit:
                raise ContainerExeption("cannot add Limit because they have different limits ({} vs {})".format(self.limit, other.limit))
            else:
                newentries = self.entries + other.entries
                if newentries > self.limit:
                    newvalue = None
                else:
                    newvalue = self.value + other.value

                return Limit.ed(newentries, self.limit, self.contentType, newvalue)

        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()
        if self.entries + weight > self.limit:
            self.value = None
        elif self.value is not None:
            self.value.fill(datum, weight)

        # no possibility of exception from here on out (for rollback)
        self.entries += weight

    @property
    def children(self):
        return [] if self.value is None else [self.value]

    def toJsonFragment(self, suppressName): return {
        "entries": floatToJson(self.entries),
        "limit": floatToJson(self.limit),
        "type": self.contentType,
        "data": None if self.value is None else self.value.toJsonFragment(False),
        }

    @staticmethod
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "limit", "type", "data"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json, "Limit.entries")

            if isinstance(json["limit"], (int, long, float)):
                limit = float(json["limit"])
            else:
                raise JsonFormatException(json, "Limit.limit")

            if isinstance(json["type"], basestring):
                contentType = json["type"]
            else:
                raise JsonFormatException(json, "Limit.type")
            factory = Factory.registered[contentType]

            if json["data"] is None:
                value = None
            else:
                value = factory.fromJsonFragment(json["data"], None)

            return Limit.ed(entries, limit, contentType, value)

        else:
            raise JsonFormatException(json, "Limit")

    def __repr__(self):
        return "<Limit value={}>".format("saturated" if self.saturated else self.value.name)

    def __eq__(self, other):
        return isinstance(other, Limit) and numeq(self.entries, other.entries) and numeq(self.limit, other.limit) and self.contentType == other.contentType and self.value == other.value

    def __hash__(self):
        return hash((self.entries, self.limit, self.contentType, self.value))

Factory.register(Limit)

