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
from __future__ import absolute_import

class HistogramMethods(object):
    def bokeh(self, glyphType="bar", label=None, color=None, stack=None, group=None,
        xscale="categorical", yscale="linear", xgrid=False, ygrid=True,
        continuous_range=None, **kw):

        from bokeh.charts.builder import create_and_build
        from bokeh.charts.builders.bar_builder import BarBuilder
        from bokeh.charts.builders.histogram_builder import HistogramBuilder
        from bokeh.charts.builders.boxplot_builder import BoxPlotBuilder
        from bokeh.models import Range1d

        if continuous_range and not isinstance(continuous_range, Range1d):
            raise ValueError(
                    "continuous_range must be an instance of bokeh.models.ranges.Range1d"
            )

        if label is not None and values is None: kw['label_only'] = True

        # The continuous_range is the y_range
        y_range = continuous_range
        kw['label'] = label
        kw['color'] = color
        kw['stack'] = stack
        kw['group'] = group
        kw['xscale'] = xscale
        kw['yscale'] = yscale
        kw['xgrid'] = xgrid
        kw['ygrid'] = ygrid
        kw['y_range'] = y_range

        if glyphType == "box": return create_and_build(BoxPlotBuilder, self.numericalValues, **kw)
        if glyphType == "histogram": return create_and_build(HistogramBuilder, self.numericalValues, **kw)
        else: return create_and_build(BarBuilder, self.numericalValues, **kw) 

    def plot(self,chart,fname="default.html"):
        from bokeh.charts import output_file,show

        output_file(fname)
        #show(chart) 
        
    def save(self):
        pass

    def view(self): 
        pass

class ProfileMethods(object):
    pass

