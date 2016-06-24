#!/usr/bin/env python

import importlib
import inspect

modules = [
    "histogrammar.defs",
    "histogrammar.specialized",
    "histogrammar.util",
    "histogrammar.version",
    "histogrammar.primitives.absoluteerr",
    "histogrammar.primitives.adaptivebin",
    "histogrammar.primitives.average",
    "histogrammar.primitives.bag",
    "histogrammar.primitives.bin",
    "histogrammar.primitives.categorize",
    "histogrammar.primitives.centralbin",
    "histogrammar.primitives.collection",
    "histogrammar.primitives.count",
    "histogrammar.primitives.deviate",
    "histogrammar.primitives.fraction",
    "histogrammar.primitives.limit",
    "histogrammar.primitives.minmax",
    "histogrammar.primitives.partition",
    "histogrammar.primitives.quantile",
    "histogrammar.primitives.sample",
    "histogrammar.primitives.select",
    "histogrammar.primitives.sparsebin",
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
            print objName, obj
            documented.append(moduleName + "." + objName)
            if inspect.isclass(obj):
                open("docs/" + moduleName + "." + objName + ".rst", "w").write(''':orphan:

{0}
{1}

.. autoclass:: {0}
    :members:
    :special-members: __init__, __add__
    :inherited-members:
    :show-inheritance:
'''.format(moduleName + "." + objName, "=" * (len(moduleName) + len(objName) + 1)))
            else:
                open("docs/" + moduleName + "." + objName + ".rst", "w").write(''':orphan:

{0}
{1}

.. autofunction:: {0}
'''.format(moduleName + "." + objName, "=" * (len(moduleName) + len(objName) + 1)))
