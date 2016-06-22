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

untransformed = serializable(lambda x: x)

class Count(Factory, Container):
    @staticmethod
    def ed(entries):
        if entries < 0.0:
            raise ContainerException("entries ($entries) cannot be negative")
        out = Count()
        out.entries = float(entries)
        return out.specialize()

    @staticmethod
    def ing(transform=untransformed):
        return Count(transform)

    def __init__(self, transform=untransformed):
        self.entries = 0.0
        self.transform = serializable(transform)
        super(Count, self).__init__()
        self.specialize()
    
    def zero(self): return Count(self.transform)

    def __add__(self, other):
        if isinstance(other, Count):
            out = Count(self.transform)
            out.entries = self.entries + other.entries
            return out.specialize()
        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()
        # no possibility of exception from here on out (for rollback)
        if weight > 0.0:
            self.entries += self.transform(weight)

    @property
    def children(self):
        return []

    def toJsonFragment(self, suppressName): return floatToJson(self.entries)

    @staticmethod
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, (int, long, float)):
            return Count.ed(float(json))
        else:
            raise JsonFormatException(json, "Count")
        
    def __repr__(self):
        return "<Count {}>".format(self.entries)

    def __eq__(self, other):
        return isinstance(other, Count) and numeq(self.entries, other.entries) and self.transform == other.transform

    def __hash__(self):
        return hash((self.entries, self.transform))

Factory.register(Count)
