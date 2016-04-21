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

class Count(Factory, Container):
    def __init__(self, entries=0.0):
        self.entries = float(entries)

    @property
    def zero(self): return Count(0.0)

    def __add__(self, other):
        if isinstance(other, Count):
            return Count(self.entries + other.entries)
        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    def fill(datum, weight=1.0):
        if weight > 0.0:
            self.entries += weight

    def toJsonFragment(self): return self.entries

    @staticmethod
    def fromJsonFragment(json):
        if isinstance(json, (int, long, float)):
            return Count(json)
        else:
            raise JsonFormatException(json, self.name)
        
    def __repr__(self):
        return "Count({})".format(self.entries)

    def __eq__(self, other):
        return isinstance(other, Count) and self.entries == other.entries

    def __hash__(self):
        return hash(self.entries)

Factory.register(Count)
