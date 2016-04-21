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

import json

class ContainerException(Exception):
    pass

class InvalidJsonException(Exception):
    def __init__(self, message):
        super(InvalidJsonException, self).__init__("invalid JSON: {}".format(message))

class JsonFormatException(Exception):
    def __init__(self, x, context):
        super(JsonFormatException, self).__init__("wrong JSON format for {}: {}".format(context, json.dumps(x)))

class Factory(object):
    @property
    def name(self): raise NotImplementedError
    @property
    def help(self): raise NotImplementedError
    @property
    def detailedHelp(self): raise NotImplementedError

    def fromJsonFragment(x): raise NotImplementedError

    registered = {}

    @classmethod
    def register(cls, factory):
        cls.registered[factory.name] = factory

    @classmethod
    def get(cls, name):
        try:
            return cls.registered[name]
        except KeyError:
            raise InvalidJsonException("unrecognized container (is it a custom container that hasn't been registered?): " + name)

    @classmethod
    def fromJson(x):
        if isinstance(x, basestring):
            x = json.loads(x)

        if isinstance(x, dict) and set(x.keys()) == set(["type", "data"]):
            if isinstance(x["type"], basestring):
                name = x["type"]
            else:
                raise JsonFormatException(x["type"], "Factory.type")

            return Factory.get(name).fromJsonFormat(x["data"])

        else:
            raise JsonFormatException(x, "Factory")
        
class Container(object):
    @property
    def factory(self): raise NotImplementedError
    @property
    def entries(self): return self._entries
    @property
    def zero(self): raise NotImplementedError
    @property
    def __add__(self, other): raise NotImplementedError

    def copy(self): return self + self.zero
    def toJson(self): return {"type": self.factory.name, "data": self.toJsonFragment()}
    def toJsonFragment(self): raise NotImplementedError

class Aggregator(Container):
    @property
    def entries(self): return self._entries
    @entries.setter
    def entries(self, value):
        self._entries = value

def help():
    
