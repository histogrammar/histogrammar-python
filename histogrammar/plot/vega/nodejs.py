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
import subprocess
import tempfile
import os


def write(vegaSpec, outputFile, format=None):
    """Use the 'vega' package in Nodejs to write to SVG or PNG files.

    Unlike interactive plotting, this does not require a round trip through a web browser, but it does require a
    Nodejs installation on your computer (to evaluate the Javascript).

    To install the prerequisites on an Ubuntu system, do

        # Cairo dependencies for generating PNG:
        sudo apt-get install install libcairo2-dev libjpeg-dev libgif-dev libpango1.0-dev build-essential g++
        # Nodejs and its package manager, npm:
        sudo apt-get install npm

        # Get the 'vega' package with npm; user-install, not global (no sudo)!
        npm install vega

    Parameters:
        vegaSpec (string or dict): JSON string or its dict-of-dicts equivalent
        outputFile (string or None): output file name or None to return output as a string
        format ('svg', 'png', or None): None (default) guesses format from outputFile extension
    """

    if format is None and outputFile is None:
        format = "svg"
    elif format is None and outputFile.endswith(".svg"):
        format = "svg"
    elif format is None and outputFile.endswith(".png"):
        format = "png"
    else:
        raise IOError("Could not infer format from outputFile")

    if format == "png":
        cmd = "vg2png"
    elif format == "svg":
        cmd = "vg2svg"
    else:
        raise IOError("Only 'png' and 'svg' output is supported.")

    npmbin = subprocess.Popen(["npm", "bin"], stdout=subprocess.PIPE)
    if npmbin.wait() == 0:
        npmbin = npmbin.stdout.read().strip()
    else:
        raise IOError("Nodejs Package Manager 'npm' must be installed to use nodejs.write function.")

    tmp = tempfile.NamedTemporaryFile(delete=False)

    if isinstance(vegaSpec, dict):
        vegaSpec = json.dump(tmp, vegaSpec)
    else:
        tmp.write(vegaSpec)

    tmp.close()

    if outputFile is None:
        vg2x = subprocess.Popen([cmd, tmp.name], stdout=subprocess.PIPE, env=dict(
            os.environ, PATH=npmbin + ":" + os.environ.get("PATH", "")))
        if vg2x.wait() == 0:
            return vg2x.stdout.read()
        else:
            os.unlink(tmp.name)
            raise IOError("Command '{0}' failed; if it's not installed, install it with 'npm install vega'".format(cmd))

    else:
        vg2x = subprocess.Popen([cmd, tmp.name, outputFile], stdout=subprocess.PIPE,
                                env=dict(os.environ, PATH=npmbin + ":" + os.environ.get("PATH", "")))
        if vg2x.wait() != 0:
            os.unlink(tmp.name)
            raise IOError("Command '{0}' failed; if it's not installed, install it with 'npm install vega'".format(cmd))
