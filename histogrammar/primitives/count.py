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

identity = serializable(lambda x: x)

class Count(Factory, Container):
    """Count entries by accumulating the sum of all observed weights or a sum of transformed weights (e.g. sum of squares of weights).

    An optional ``transform`` function can be applied to the weights before summing. To accumulate the sum of squares of weights, use

    ::

        lambda x: x**2

    for instance. This is unlike any other primitive's ``quantity`` function in that its domain is the *weights* (always double), not *data* (any type).
    """

    @staticmethod
    def ed(entries):
        """
        * `entries` (double) is the number of entries.
        """
        if not isinstance(entries, (int, long, float)):
            raise TypeError("entries ({}) must be a number".format(entries))
        if entries < 0.0:
            raise ValueError("entries ({}) cannot be negative".format(entries))
        out = Count()
        out.entries = float(entries)
        return out.specialize()

    @staticmethod
    def ing(transform=identity):
        """Synonym for ``__init__``."""
        return Count(transform)

    def __init__(self, transform=identity):
        """
        * `entries` (mutable double) is the number of entries, initially 0.0.
        * `transform` (function from double to double) transforms each weight.
        """
        self.entries = 0.0
        self.transform = serializable(transform)
        super(Count, self).__init__()
        self.specialize()
    
    @inheritdoc(Container)
    def zero(self): return Count(self.transform)

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, Count):
            out = Count(self.transform)
            out.entries = self.entries + other.entries
            return out.specialize()
        else:
            raise ContainerException("cannot add {} and {}".format(self.name, other.name))

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()
        if weight > 0.0:
            t = self.transform(weight)
            if not isinstance(t, (bool, int, long, float)):
                raise TypeError("function return value ({}) must be boolean or number".format(t))

            # no possibility of exception from here on out (for rollback)
            self.entries += t

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return []

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName): return floatToJson(self.entries)

    @staticmethod
    @inheritdoc(Factory)
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
