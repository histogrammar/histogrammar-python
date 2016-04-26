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

import marshal
import types

class Fcn(object):
    def __init__(self, fcn, varname="datum"):
        if isinstance(fcn, basestring):
            c = compile(fcn, "<string>", "eval")
            def function(datum):
                context = dict(globals(), **{varname: datum})
                try:
                    context.update(datum.__dict__)
                except AttributeError:
                    pass
                return eval(c, context)
            fcn = function

        if not isinstance(fcn, types.FunctionType):
            raise TypeError("quantity or selection function must be a function or string expression")
        self.fcn = fcn

    def __call__(self, *args, **kwds):
        return self.fcn(*args, **kwds)

    def __reduce__(self):
        refs = {n: self.fcn.func_globals[n] for n in self.fcn.func_code.co_names if n in self.fcn.func_globals}
        return (deserializeFcn, (self.__class__, marshal.dumps(self.fcn.func_code), self.fcn.func_name, self.fcn.func_defaults, self.fcn.func_closure, refs))

    def __repr__(self):
        return "Fcn({})".format(self.fcn)

class CachedFcn(Fcn):
    def __call__(self, *args, **kwds):
        if hasattr(self, "lastArgs") and hasattr(self, "lastKwds") and args == self.lastArgs and kwds == self.lastKwds:
            return self.lastReturn
        else:
            self.lastArgs = args
            self.lastKwds = kwds
            self.lastReturn = self.fcn(*args, **kwds)
            return self.lastReturn

    def __repr__(self):
        return "CachedFcn({})".format(self.fcn)

def deserializeFcn(cls, func_code, func_name, func_defaults, func_closure, refs):
    out = cls.__new__(cls)
    g = dict(globals(), **refs)
    out.fcn = types.FunctionType(marshal.loads(func_code), g, func_name, func_defaults, func_closure)
    return out

def serializable(fcn):
    if fcn is None:
        return None
    elif isinstance(fcn, Fcn):
        return fcn
    else:
        return Fcn(fcn)

def cache(fcn):
    if isinstance(fcn, CachedFcn):
        return fcn
    elif isinstance(fcn, Fcn):
        return CachedFcn(fcn.fcn)
    else:
        return CachedFcn(fcn)
