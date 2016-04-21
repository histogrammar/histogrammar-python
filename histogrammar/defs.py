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

import json as jsonlib

class ContainerException(Exception):
    pass

class InvalidJsonException(Exception):
    def __init__(self, message):
        super(InvalidJsonException, self).__init__("invalid JSON: {}".format(message))

class JsonFormatException(Exception):
    def __init__(self, x, context):
        super(JsonFormatException, self).__init__("wrong JSON format for {}: {}".format(context, json.dumps(x)))

class Factory(object):
    registered = {}
    
    @staticmethod
    def register(factory):
        Factory.registered[factory.__name__] = factory

    @staticmethod
    def fromJson(json):
        if isinstance(json, basestring):
            json = jsonlib.loads(json)

        if isinstance(json, dict) and set(json.keys()) == set(["type", "data"]):
            if isinstance(json["type"], basestring):
                name = json["type"]
            else:
                raise JsonFormatException(json["type"], "Factory.type")

            if name not in Factory.registered:
                raise JsonFormatException(json, "unrecognized container (is it a custom container that hasn't been registered?): {}".format(name))

            return Factory.registered[name].fromJsonFragment(json["data"])

        else:
            raise JsonFormatException(json, "Factory")
        
class Container(object):
    @property
    def name(self): return self.__class__.__name__
    @property
    def factory(self): return self.__class__

    @property
    def zero(self): raise NotImplementedError
    def __add__(self, other): raise NotImplementedError
    def fill(datum, weight=1.0): raise NotImplementedError

    def copy(self): return self + self.zero

    def toJson(self): return {"type": self.name, "data": self.toJsonFragment()}
    def toJsonFragment(self): raise NotImplementedError
    def __repr__(self): raise NotImplementedError

def unweighted(datum): return 1.0
