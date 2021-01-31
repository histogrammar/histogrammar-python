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

import json
import math
import sys

# Definitions for python 2/3 compatability
if sys.version_info[0] > 2:
    basestring = str
    long = int

MAX_REPR = 50


class JsonObject(dict):
    def __init__(self, *pairs, **kwarg):
        if isinstance(pairs, dict):
            self._pairs = tuple(pairs.items())
        else:
            self._pairs = pairs
        if len(kwarg) > 0:
            self._pairs = self._pairs + tuple(kwarg.items())

        if any(not isinstance(kv, tuple) or len(kv) != 2 for kv in self._pairs):
            raise TypeError("JsonObject pairs must all be two-element tuples")

        if any(not isinstance(k, basestring) or not (v is None or isinstance(
                v, (basestring, bool, int, long, float, JsonObject, JsonArray))) for k, v in self._pairs):
            raise TypeError(
                "JsonObject keys must be strings and values must be (string, bool, int, float, JsonObject, JsonArray)")

    def toJsonString(self, prefix="", indent=2):
        out = ["{\n", prefix, " "]
        first = True
        for k, v in self._pairs:
            if first:
                first = False
            else:
                out.append(",\n")
                out.append(prefix)
                out.append(" ")
            out.append(json.dumps(k))
            out.append(": ")
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                raise ValueError("cannot JSON-serialize NaN or Infinity")
            elif isinstance(v, (basestring, bool, int, long, float)):
                v = json.dumps(v)
            else:
                v = v.toJsonString(prefix + (" " * indent), indent)
            out.append(v)
        out.append("\n")
        out.append(prefix)
        out.append("}")
        return "".join(out)

    def _index(self, key):
        for i, (k, v) in enumerate(self._pairs):
            if k == key:
                return i
        return -1

    def set(self, *path, **kwds):
        if "to" not in kwds:
            raise TypeError("missing keyword argument 'to' in set(path, to=value)")
        elif len(kwds) != 1:
            raise TypeError("unrecognized keyword arguments in set(path, to=value)")
        value = kwds["to"]

        if len(path) < 1:
            raise TypeError("missing path in set(path, to=value)")
        key = path[0]
        index = self._index(key)
        if len(path) == 1:
            if index == -1:
                return JsonObject(*(self._pairs + ((key, value),)))
            else:
                return JsonObject(*[(key, value) if k == key else (k, v) for k, v in self._pairs])
        else:
            if index == -1:
                raise ValueError("JsonObject field {0} does not contain path ({1})".format(
                    repr(key), ", ".join(map(repr, path[1:]))))
            elif not isinstance(self._pairs[index][1], (JsonObject, JsonArray)):
                raise ValueError("JsonObject field {0} does not contain path ({1})".format(
                    repr(key), ", ".join(map(repr, path[1:]))))
            else:
                return JsonObject(*[(k, v.set(*path[1:], **kwds)) if k == key else (k, v) for k, v in self._pairs])

    def without(self, *path):
        if len(path) < 1:
            raise TypeError("missing path in without(path)")
        key = path[0]
        index = self._index(key)
        if len(path) == 1:
            if index == -1:
                return self
            else:
                return JsonObject(*[(k, v) for k, v in self._pairs if k != key])
        else:
            if index == -1:
                raise ValueError("JsonObject field {0} does not contain path ({1})".format(
                    repr(key), ", ".join(map(repr, path[1:]))))
            elif not isinstance(self._pairs[index][1], (JsonObject, JsonArray)):
                raise ValueError("JsonObject field {0} does not contain path ({1})".format(
                    repr(key), ", ".join(map(repr, path[1:]))))
            else:
                return JsonObject(*[(k, v.without(*path[1:])) if k == key else (k, v) for k, v in self._pairs])

    def overlay(self, other):
        out = self
        for k, v in other.items():
            out = out.set(k, to=v)
        return out

    # override built-in dict methods

    def __cmp__(self, other):
        def cmp(a, b):
            return (a > b) - (a < b)
        return cmp(dict(self._pairs), dict(other._pairs))

    def __contains__(self, key):
        return any(k == key for k, v in self._pairs)

    def __delattr_(self, key):
        raise TypeError("JsonObject cannot be changed in-place; no immutable equivalent")

    def __delitem__(self, key):
        raise TypeError("JsonObject cannot be changed in-place; use .without(key)")

    def __eq__(self, other):
        return isinstance(other, JsonObject) and self._pairs == other._pairs

    def __format__(self, format_spec):
        return str(self)

    def __getitem__(self, key):
        index = self._index(key)
        if index == -1:
            raise KeyError(key)
        else:
            return self._pairs[index][1]

    def __hash__(self):
        return hash(("JsonObject", self._pairs))

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return len(self._pairs)

    def __reduce__(self):
        return self.__reduce_ex__(0)

    def __reduce_ex__(self, protocol):
        return (self.__class__, self._pairs)

    def __repr__(self):
        out = "{"
        first = True
        for k, v in self._pairs:
            if first:
                first = False
            else:
                out += ", "
            if len(out) > MAX_REPR - 1:
                break
            if isinstance(v, (basestring, bool, int, long, float)):
                v = json.dumps(v)
            else:
                v = repr(v)
            out += json.dumps(k) + ": " + v
        if len(out) > MAX_REPR - 1:
            out = out[:(MAX_REPR - 4)] + "..."
        return out + "}"

    def __setitem__(self, key, value):
        raise TypeError("JsonObject cannot be changed in-place; use .set(path, to=value)")

    def __sizeof__(self):
        return super(dict, self).__sizeof__()

    def __str__(self):
        out = ["{"]
        first = True
        for k, v in self._pairs:
            if first:
                first = False
            else:
                out.append(",")
            out.append(json.dumps(k))
            out.append(":")
            if isinstance(v, (basestring, bool, int, long, float)):
                v = json.dumps(v)
            else:
                v = str(v)
            out.append(v)
        out.append("}")
        return "".join(out)

    def clear(self):
        raise TypeError("JsonObject cannot be changed in-place; use JsonObject() constructor to make a new one")

    def copy(self):
        return self   # because we're immutable

    def __copy__(self):
        return self   # because we're immutable

    def __deepcopy__(self, memo):
        return self   # because we're immutable

    def get(self, key, default=None):
        index = self._index(key)
        if index == -1:
            return default
        else:
            return self._pairs[index][1]

    def has_key(self, key):
        return key in self

    def items(self):
        for k, v in self._pairs:
            yield k, v

    def iteritems(self):
        return self.items()

    def iterkeys(self):
        return self.keys()

    def itervalues(self):
        return self.values()

    def keys(self):
        for k, v in self._pairs:
            yield k

    def pop(self, key, default=None):
        raise TypeError("JsonObject cannot be changed in-place; no immutable equivalent")

    def popitem(self, key, default=None):
        raise TypeError("JsonObject cannot be changed in-place; no immutable equivalent")

    def setdefault(self, key, default=None):
        raise TypeError("JsonObject cannot be changed in-place; no immutable equivalent")

    def update(self, other):
        raise TypeError("JsonObject cannot be changed in-place; use .overlay(other)")

    def values(self):
        for k, v in self._pairs:
            yield v

    def viewitems(self):
        return self.items()

    def viewkeys(self):
        return self.keys()

    def viewvalues(self):
        return self.values()


class JsonArray(tuple):
    def __init__(self, *values):
        self._values = values
        if any(not (v is None or isinstance(v, (basestring, bool, int, long, float, JsonObject, JsonArray)))
               for v in self._values):
            raise TypeError("JsonArray values must be (string, bool, int, float, JsonObject, JsonArray)")

    def toJsonString(self, prefix="", indent=2):
        out = [prefix, "[\n", prefix, " "]
        first = True
        for v in self._values:
            if first:
                first = False
            else:
                out.append(",\n")
                out.append(prefix)
                out.append(" ")
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                raise ValueError("cannot JSON-serialize NaN or Infinity")
            elif isinstance(v, (basestring, bool, int, long, float)):
                v = json.dumps(v)
            else:
                v = v.toJsonString(prefix + (" " * indent), indent)
            out.append(v)
        out.append("\n")
        out.append(prefix)
        out.append("]")
        return "".join(out)

    # override built-in tuple methods

# __add__
# __cmp__
# __contains__
# __delattr__
# __delitem__
# __eq__
# __format__
# __getitem__
# __getnewargs__
# __getslice__
# __hash__
# __iter__
# __len__
# __mul__
# __reduce__
# __reduce_ex__

    def __repr__(self):
        out = "["
        first = True
        for v in self._values:
            if first:
                first = False
            else:
                out += ", "
            if len(out) > MAX_REPR - 1:
                break
            out += repr(v)
        if len(out) > MAX_REPR - 1:
            out = out[:(MAX_REPR - 4)] + "..."
        return out + "]"

# __rmul__
# __sizeof__

    def __str__(self):
        out = ["["]
        first = False
        for v in self._values:
            if first:
                first = False
            else:
                out.append(",")
            if isinstance(v, (basestring, bool, int, long, float)):
                v = json.dumps(v)
            else:
                v = str(v)
            out.append(v)
        out.append("]")
        return "".join(out)

# count
# index
