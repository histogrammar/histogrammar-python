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

from histogrammar.defs import *
from histogrammar.util import *

class AbsoluteErr(Factory, Container):
    """Accumulate the weighted mean absolute error (MAE) of a quantity around zero.

    The MAE is sometimes used as a replacement for the standard deviation, associated with medians, rather than means. However, this aggregator makes no attempt to estimate a median. If used as an "error," it should be used on a quantity whose nominal value is zero, such as a residual.
    """

    @staticmethod
    def ed(entries, mae):
        """Create an AbsoluteErr that is only capable of being added.

        Parameters:
            entries (float): the number of entries.
            mae (float): the mean absolute error.
        """

        if not isinstance(entries, (int, long, float)):
            raise TypeError("entries ({0}) must be a number".format(entries))
        if not isinstance(mae, (int, long, float)):
            raise TypeError("mae ({0}) must be a number".format(mae))
        if entries < 0.0:
            raise ValueError("entries ({0}) cannot be negative".format(entries))
        out = AbsoluteErr(None)
        out.entries = float(entries)
        out.absoluteSum = float(mae)*float(entries)
        return out.specialize()

    @staticmethod
    def ing(quantity):
        """Synonym for ``__init__``."""
        return AbsoluteErr(quantity)

    def __init__(self, quantity):
        """Create an AbsoluteErr that is capable of being filled and added.

        Parameters:
            quantity (function returning float): computes the quantity of interest from the data.

        Other parameters:
            entries (float): the number of entries, initially 0.0.
            mae (float): the mean absolute error.
        """

        self.quantity = serializable(quantity)
        self.entries = 0.0
        self.absoluteSum = 0.0
        super(AbsoluteErr, self).__init__()
        self.specialize()

    @property
    def mae(self):
        """The mean absolute error."""
        if self.entries == 0.0:
            return self.absoluteSum
        else:
            return self.absoluteSum/self.entries

    @inheritdoc(Container)
    def zero(self): return AbsoluteErr(self.quantity)

    @inheritdoc(Container)
    def __add__(self, other):
        if isinstance(other, AbsoluteErr):
            out = AbsoluteErr(self.quantity)
            out.entries = self.entries + other.entries
            out.absoluteSum = self.entries*self.mae + other.entries*other.mae
            return out.specialize()
        else:
            raise ContainerException("cannot add {0} and {1}".format(self.name, other.name))

    @inheritdoc(Container)
    def fill(self, datum, weight=1.0):
        self._checkForCrossReferences()
        if weight > 0.0:
            q = self.quantity(datum)
            try:
                q = float(q)
            except:
                raise TypeError("function return value ({0}) must be boolean or number".format(q))

            # no possibility of exception from here on out (for rollback)
            self.entries += weight
            self.absoluteSum += abs(q) * weight

    def fillnp(self, data, weight=1.0):
        """Increment the aggregator by providing a one-dimensional Numpy array of ``data`` to the fill rule with given ``weight`` (number or array).

        This primitive is optimized with Numpy.

        The container is changed in-place.
        """
        self._checkForCrossReferences()

        import numpy
        data, weight = self._normalizenp(data, weight)
        if not isinstance(weight, numpy.ndarray) and weight <= 0.0: return
        q = self._computenp(data)

        self._entriesnp(weight, data.shape[0])
        numpy.absolute(q, q)
        numpy.multiply(q, weight, q)
        self.absoluteSum += float(q.sum())

    @property
    def children(self):
        """List of sub-aggregators, to make it possible to walk the tree."""
        return []

    @inheritdoc(Container)
    def toJsonFragment(self, suppressName): return maybeAdd({
        "entries": floatToJson(self.entries),
        "mae": floatToJson(self.mae),
        }, name=(None if suppressName else self.quantity.name))

    @staticmethod
    @inheritdoc(Factory)
    def fromJsonFragment(json, nameFromParent):
        if isinstance(json, dict) and hasKeys(json.keys(), ["entries", "mae"], ["name"]):
            if isinstance(json["entries"], (int, long, float)):
                entries = float(json["entries"])
            else:
                raise JsonFormatException(json["entries"], "AbsoluteErr.entries")

            if isinstance(json.get("name", None), basestring):
                name = json["name"]
            elif json.get("name", None) is None:
                name = None
            else:
                raise JsonFormatException(json["name"], "AbsoluteErr.name")

            if isinstance(json["mae"], (int, long, float)):
                mae = float(json["mae"])
            else:
                raise JsonFormatException(json["mae"], "AbsoluteErr.mae")

            out = AbsoluteErr.ed(entries, mae)
            out.quantity.name = nameFromParent if name is None else name
            return out.specialize()

        else:
            raise JsonFormatException(json, "AbsoluteErr")
        
    def __repr__(self):
        return "<AbsoluteErr mae={0}>".format(self.mae)

    def __eq__(self, other):
        return isinstance(other, AbsoluteErr) and self.quantity == other.quantity and numeq(self.entries, other.entries) and numeq(self.mae, other.mae)

    def __ne__(self, other): return not self == other

    def __hash__(self):
        return hash((self.quantity, self.entries, self.mae))

Factory.register(AbsoluteErr)
