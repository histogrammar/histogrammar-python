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

import importlib
import inspect
from pathlib import Path

modules = [
    "histogrammar.defs",
    "histogrammar.specialized",
    "histogrammar.util",
    "histogrammar.version",
    "histogrammar.primitives.average",
    "histogrammar.primitives.bag",
    "histogrammar.primitives.bin",
    "histogrammar.primitives.categorize",
    "histogrammar.primitives.centrallybin",
    "histogrammar.primitives.collection",
    "histogrammar.primitives.count",
    "histogrammar.primitives.deviate",
    "histogrammar.primitives.fraction",
    "histogrammar.primitives.irregularlybin",
    "histogrammar.primitives.minmax",
    "histogrammar.primitives.select",
    "histogrammar.primitives.sparselybin",
    "histogrammar.primitives.stack",
    "histogrammar.primitives.sum",
    "histogrammar.plot.bokeh",
    "histogrammar.plot.root",
]

modules = {name: importlib.import_module(name) for name in modules}

documented = []
for moduleName, module in modules.items():
    for objName in dir(module):
        obj = getattr(module, objName)
        if not objName.startswith("_") and callable(obj) and obj.__module__ == moduleName:
            print(objName, obj)
            documented.append(moduleName + "." + objName)
            path = Path("docs/" + moduleName + "." + objName + ".rst")
            if inspect.isclass(obj):
                path.write_text(
                    """:orphan:

{0}
{1}

.. autoclass:: {0}
    :members:
    :special-members: __init__, __add__
    :inherited-members:
    :show-inheritance:
""".format(
                        moduleName + "." + objName,
                        "=" * (len(moduleName) + len(objName) + 1),
                    )
                )
            else:
                path.write_text(
                    """:orphan:

{0}
{1}

.. autofunction:: {0}
""".format(
                        moduleName + "." + objName,
                        "=" * (len(moduleName) + len(objName) + 1),
                    )
                )
