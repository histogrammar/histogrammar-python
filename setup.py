# flake8: noqa

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

from setuptools import setup, find_packages

import histogrammar.version

with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

# read the contents of abstract file
with open("README.rst", encoding="utf-8") as f:
    long_description = f.read()

setup(name="histogrammar",
      version=histogrammar.version.__version__,
      packages=find_packages(),
      scripts=["scripts/hgwatch"],
      description="Composable histogram primitives for distributed data reduction.",
      long_description=long_description,
      long_description_content_type="text/x-rst",
      python_requires=">=3.6",
      author="Jim Pivarski (DIANA-HEP)",
      author_email="pivarski@fnal.gov",
      maintainer="Max Baak",
      maintainer_email="maxbaak@gmail.com",
      url="https://histogrammar.github.io/histogrammar-docs",
      download_url="https://github.com/histogrammar/histogrammar-python",
      license="Apache Software License v2",
      test_suite="tests",
      install_requires=REQUIREMENTS,
      classifiers=["Development Status :: 5 - Production/Stable",
                     "Environment :: Console",
                     "Intended Audience :: Science/Research",
                     "License :: OSI Approved :: Apache Software License",
                     "Topic :: Scientific/Engineering :: Information Analysis",
                     "Topic :: Scientific/Engineering :: Mathematics",
                     "Topic :: Scientific/Engineering :: Physics",
                   ],
      # files to be shipped with the installation, under: popmon/popmon/
      # after installation, these can be found with the functions in resources.py
      package_data=dict(
          histogrammar=[
              "test_data/*.csv.gz",
              "test_data/*.json*",
              "notebooks/*tutorial*.ipynb",
          ]
      ),
      platforms="Any",
      )
