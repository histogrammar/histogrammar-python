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

import histogrammar.pycparser.c_parser
import histogrammar.pycparser.c_generator
import histogrammar.pycparser.c_ast


class C99SourceToAst(object):
    def __init__(self, wholeFile=False):
        self.wholeFile = wholeFile
        self.parser = histogrammar.pycparser.c_parser.CParser(
            lextab="histogrammar.pycparser.lextab", yacctab="histogrammar.pycparser.yacctab")

    def __call__(self, src):
        if self.wholeFile:
            return self.parser.parse(src)
        else:
            src = "void wrappedAsFcn() {" + src + ";}"
            ast = self.parser.parse(src).ext[0].body.block_items
            if len(ast) < 1:
                raise SyntaxError("empty expression")
            else:
                return [x for x in ast if not isinstance(x, histogrammar.pycparser.c_ast.EmptyStatement)]


class C99AstToSource(object):
    def __init__(self):
        self.generator = histogrammar.pycparser.c_generator.CGenerator()

    def __call__(self, ast):
        if isinstance(ast, (list, tuple)):
            return "; ".join(self.generator.visit(x).strip() for x in ast)
        return self.generator.visit(ast).strip()
