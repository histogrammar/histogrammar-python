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

class Count(Factory, Container):
    @staticmethod
    def ed(entries):
        if entries < 0.0:
            raise ContainerException("entries ($entries) cannot be negative")
        out = Count()
        out.entries = float(entries)
        return out

    @staticmethod
    def ing():
        return Count()

    def __init__(self):
        self.entries = 0.0
        super(Count, self).__init__()

    def zero(self): return Count()

    def __add__(self, other):
        if isinstance(other, Count):
            out = Count()
            out.entries = self.entries + other.entries
            return out
        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    def fill(self, datum, weight=1.0):
        # no possibility of exception from here on out (for rollback)
        if weight > 0.0:
            self.entries += weight

    def toJsonFragment(self): return floatToJson(self.entries)

    @staticmethod
    def fromJsonFragment(json):
        if isinstance(json, (int, long, float)):
            return Count.ed(float(json))
        else:
            raise JsonFormatException(json, "Count")
        
    def __repr__(self):
        return "Count[{}]".format(self.entries)

    def __eq__(self, other):
        return isinstance(other, Count) and exact(self.entries, other.entries)

    def __hash__(self):
        return hash(self.entries)

Factory.register(Count)
