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

# "Public" methods; what we want to attach to the Histogram as a mix-in.

from bokeh.charts.builder import Builder, create_and_build
from bokeh.models import FactorRange, Range1d
from bokeh.charts.glyphs import BarGlyph
from bokeh.charts.properties import Dimension
from bokeh.charts.attributes import ColorAttr, CatAttr
from bokeh.models.sources import ColumnDataSource
from bokeh.charts.builders.bar_builder import BarBuilder

class HistogramMethods(object):
    def bokeh(self, label=None, values=None, color=None, stack=None, group=None, agg="sum",
        xscale="categorical", yscale="linear", xgrid=False, ygrid=True,
        continuous_range=None, **kw):

        if continuous_range and not isinstance(continuous_range, Range1d):
            raise ValueError(
                    "continuous_range must be an instance of bokeh.models.ranges.Range1d"
            )

        if label is not None and values is None:
            kw['label_only'] = True
            if (agg == 'sum') or (agg == 'mean'):
                agg = 'count'
                values = label

        # The continuous_range is the y_range
        y_range = continuous_range
        kw['label'] = label
        kw['values'] = values
        kw['color'] = color
        kw['stack'] = stack
        kw['group'] = group
        kw['agg'] = agg
        kw['xscale'] = xscale
        kw['yscale'] = yscale
        kw['xgrid'] = xgrid
        kw['ygrid'] = ygrid
        kw['y_range'] = y_range

        return create_and_build(BarBuilder, self.numericalValues, **kw)


    def plot(self):
        pass
    def save(self):
        pass
    def view(self): 
        pass

class ProfileMethods(object):
    pass

