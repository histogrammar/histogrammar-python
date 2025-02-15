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

# handy monkey patch functions for pandas and spark dataframes
import histogrammar.dfinterface
from histogrammar.convenience import (
    Histogram,
    Profile,
    ProfileErr,
    SparselyHistogram,
    SparselyProfile,
    SparselyProfileErr,
    TwoDimensionallyHistogram,
    TwoDimensionallySparselyHistogram,
)
from histogrammar.defs import Container, Factory
from histogrammar.primitives.average import Average
from histogrammar.primitives.bag import Bag
from histogrammar.primitives.bin import Bin
from histogrammar.primitives.categorize import Categorize
from histogrammar.primitives.centrallybin import CentrallyBin
from histogrammar.primitives.collection import (
    Branch,
    Collection,
    Index,
    Label,
    UntypedLabel,
)
from histogrammar.primitives.count import Count
from histogrammar.primitives.deviate import Deviate
from histogrammar.primitives.fraction import Fraction
from histogrammar.primitives.irregularlybin import IrregularlyBin
from histogrammar.primitives.minmax import Maximize, Minimize
from histogrammar.primitives.select import Select
from histogrammar.primitives.sparselybin import SparselyBin
from histogrammar.primitives.stack import Stack
from histogrammar.primitives.sum import Sum
