# Copyright (c) 2020 ING Wholesale Banking Advanced Analytics
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


# Resources lookup file for histogrammar
from importlib import resources

from histogrammar import notebooks, test_data

# data files that are shipped with histogrammar.
_DATA = {_.name: _ for _ in resources.files(test_data).iterdir()}

# Tutorial notebooks
_NOTEBOOK = {p.name: p for p in resources.files(notebooks).iterdir() if p.suffix == ".ipynb"}

# Resource types
_RESOURCES = {"data": _DATA, "notebook": _NOTEBOOK}


def _resource(resource_type, name: str) -> str:
    """Return the full path filename of a resource.

    :param str resource_type: The type of the resource.
    :param str  name: The name of the resource.
    :returns: The full path filename of the fixture data set.
    :rtype: str
    :raises FileNotFoundError: If the resource cannot be found.
    """
    full_path = _RESOURCES[resource_type].get(name, None)

    if full_path and full_path.exists():
        return str(full_path)

    raise FileNotFoundError(f'Could not find {resource_type} "{name!s}"! Does it exist?')


def data(name: str) -> str:
    """Return the full path filename of a shipped data file.

    :param str name: The name of the data.
    :returns: The full path filename of the data.
    :rtype: str
    :raises FileNotFoundError: If the data cannot be found.
    """
    return _resource("data", name)


def notebook(name: str) -> str:
    """Return the full path filename of a tutorial notebook.

    :param str name: The name of the notebook.
    :returns: The full path filename of the notebook.
    :rtype: str
    :raises FileNotFoundError: If the notebook cannot be found.
    """
    return _resource("notebook", name)
