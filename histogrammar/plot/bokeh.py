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
    def bokeh(self,glyphType="line",glyphSize=1,fill_color="red",line_color="black",line_alpha=1,fill_alpha=0.1,line_dash='solid'):

        #glyphs
        from bokeh.models.glyphs import Rect, Segment, Line, Patches, Arc
        from bokeh.models.renderers import GlyphRenderer
        from bokeh.models.markers import (Marker, Asterisk, Circle, CircleCross, CircleX, Cross,
                      Diamond, DiamondCross, InvertedTriangle, Square,
                      SquareCross, SquareX, Triangle, X)

        #data 
        from bokeh.models import ColumnDataSource

        #Parameters of the histogram
        l = self.low 
        h = self.high
        num = self.num
        bin_width = (h-l)/num
        x = list()
        center = l
        for _ in range(num):
            x.append(center+bin_width/2)
            center += bin_width
        y = self.numericalValues

        source = ColumnDataSource(data=dict(x=x, y=y))

        glyph = None
        if glyphType == "square": glyph = Square(x='x', y='y',line_color=line_color,fill_color=fill_color,line_alpha=line_alpha,size=glyphSize,line_dash=line_dash)   
        elif glyphType == "diamond": glyph = Diamond(x='x', y='y',line_color=line_color,fill_color=fill_color,line_alpha=line_alpha,size=glyphSize,line_dash=line_dash)
        elif glyphType == "cross": glyph = Cross(x='x', y='y',line_color=line_color,fill_color=fill_color,line_alpha=line_alpha,size=glyphSize,line_dash=line_dash)
        elif glyphType == "triangle": glyph = Triangle(x='x', y='y',line_color=line_color,fill_color=fill_color,line_alpha=line_alpha,size=glyphSize,line_dash=line_dash)
        elif glyphType == "circle": glyph = Circle(x='x', y='y',line_color=line_color,fill_color=fill_color,line_alpha=line_alpha,size=glyphSize,line_dash=line_dash)
        elif glyphType == "rect": glyph = Rect(x='x', y='y', width=bin_width, height=0.1, fill_alpha=fill_alpha, line_color=line_color, fill_color=fill_color)
        elif glyphType == "histogram": 
            h = y
            y = [yy/2 for yy in y]
            source = ColumnDataSource(dict(x=x, y=y, h=h))
            glyph = Rect(x='x', y='y', width=bin_width, height='h', fill_alpha=fill_alpha, line_color=line_color, fill_color=fill_color)
        else: glyph = Line(x='x', y='y',line_color=line_color,line_alpha=line_alpha,line_width=glyphSize,line_dash=line_dash)

        return GlyphRenderer(glyph=glyph,data_source=source)

class SparselyHistogramMethods(object):
    pass

class ProfileMethods(object):
    pass

class SparselyProfileMethods(object):
    pass

class ProfileErrMethods(object):
    def bokeh(self,glyphType="line",glyphSize=1,fill_color="red",line_color="black",line_alpha=1,fill_alpha=0.1,line_dash='solid'):

        #glyphs
        from bokeh.models.glyphs import Rect, Segment, Line, Patches, Arc
        from bokeh.models.renderers import GlyphRenderer
        from bokeh.models.markers import (Marker, Asterisk, Circle, CircleCross, CircleX, Cross,
                      Diamond, DiamondCross, InvertedTriangle, Square,
                      SquareCross, SquareX, Triangle, X)

        #data 
        from bokeh.models import ColumnDataSource

        from math import sqrt

        #Parameters of the histogram
        l = self.low
        h = self.high
        num = self.num
        bin_width = (h-l)/num
        x = list()
        y = list()
        center = l
        for v in self.values:
            y.append(v.mean)
            x.append(center+bin_width/2)
            center += bin_width

        source = ColumnDataSource(data=dict(x=x, y=y))

        glyph = None
        if glyphType == "square": glyph = Square(x='x', y='y',line_color=line_color,fill_color=fill_color,line_alpha=line_alpha,size=glyphSize,line_dash=line_dash)
        elif glyphType == "diamond": glyph = Diamond(x='x', y='y',line_color=line_color,fill_color=fill_color,line_alpha=line_alpha,size=glyphSize,line_dash=line_dash)
        elif glyphType == "cross": glyph = Cross(x='x', y='y',line_color=line_color,fill_color=fill_color,line_alpha=line_alpha,size=glyphSize,line_dash=line_dash)
        elif glyphType == "triangle": glyph = Triangle(x='x', y='y',line_color=line_color,fill_color=fill_color,line_alpha=line_alpha,size=glyphSize,line_dash=line_dash)
        elif glyphType == "circle": glyph = Circle(x='x', y='y',line_color=line_color,fill_color=fill_color,line_alpha=line_alpha,size=glyphSize,line_dash=line_dash)
        elif glyphType == "errors":
            w = [bin_width for _ in x]
            h = [sqrt(v.variance/v.entries) if v.entries > 0 else 0.0 for v in self.values]
            source = ColumnDataSource(dict(x=x, y=y, w=w, h=h))
            glyph = Rect(x='x', y='y', width='w', height='h', fill_alpha=fill_alpha, line_color=line_color, fill_color=fill_color)
        elif glyphType == "histogram":
            w = [bin_width for _ in x]
            h = y
            y = [yy/2 for yy in y]
            source = ColumnDataSource(dict(x=x, y=y, w=w, h=h))
            glyph = Rect(x='x', y='y', width='w', height='h', fill_alpha=fill_alpha, line_color=line_color, fill_color=fill_color)
        else: glyph = Line(x='x', y='y',line_color=line_color,line_alpha=line_alpha,line_width=glyphSize,line_dash=line_dash)

        return GlyphRenderer(glyph=glyph,data_source=source)


class SparselyProfileErrMethods(object):
    pass

class StackedHistogramMethods(object):
    nMaxStacked = 7
    glyphTypeDefaults = ["circle"]*nMaxStacked
    glyphSizeDefaults = [1]*nMaxStacked
    fillColorDefaults = ["red"]*nMaxStacked
    lineColorDefaults = ["black"]*nMaxStacked
    lineAlphaDefaults = [1]*nMaxStacked
    fillAlphaDefaults = [0.1]*nMaxStacked
    lineDashDefaults = ["solid"]*nMaxStacked

    def bokeh(self,glyphTypes=glyphTypeDefaults,glyphSizes=glyphSizeDefaults,fillColors=fillColorDefaults,lineColors=lineColorDefaults,lineAlphas=lineAlphaDefaults,fillAlphas=fillAlphaDefaults,lineDashes = lineDashDefaults):
        nTypes = len(glyphTypes)
        assert nTypes == len(glyphSizes)
        assert nTypes == len(fillColors)
        assert nTypes == len(lineColors)
        assert nTypes == len(lineAlphas)
        assert nTypes == len(fillAlphas)
        assert nTypes == len(lineDashes)

        stackedGlyphs = list()
        #for ichild, p in enumerate(self.children,start=1):
        for ichild in range(1,len(self.children)):
            print(ichild)
            stackedGlyphs.append(self.children[ichild].bokeh(glyphTypes[ichild],glyphSizes[ichild],fillColors[ichild],lineColors[ichild],lineAlphas[ichild],fillAlphas[ichild],lineDashes[ichild]))

        return stackedGlyphs

class PartitionedHistogramMethods(object):
    pass

class FractionedHistogramMethods(object):
    pass

class TwoDimensionallyHistogramMethods(object):
    pass

class SparselyTwoDimensionallyHistogramMethods(object):
    pass


def plot(xLabel='x',yLabel='y',*args):

    from bokeh.models import DataRange1d, Plot, LinearAxis, Grid
    from bokeh.models import PanTool, WheelZoomTool

    xdr = DataRange1d()
    ydr = DataRange1d()

    plot = Plot(x_range=xdr, y_range=ydr, min_border=80)

    extra = list()
    if type(xLabel) is not str: 
        extra.append(xLabel)
        xLabel = 'x'
    elif type(yLabel) is not str:
        extra.append(yLabel)
        yLabel = 'y'
   
    args = extra+list(args) 
    for renderer in args:
         if type(renderer) is not list: 
             plot.renderers.append(renderer)
         else: 
             plot.renderers.extend(renderer)

    #axes
    xaxis = LinearAxis(axis_label=xLabel)
    plot.add_layout(xaxis, 'below')
    yaxis = LinearAxis(axis_label=yLabel)
    plot.add_layout(yaxis, 'left')
    #add grid to the plot 
    #plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
    #plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))

    #interactive tools
    plot.add_tools(PanTool(), WheelZoomTool()) #, SaveTool())

    return plot




def save(plot,fname):    
    #SaveTool https://github.com/bokeh/bokeh/blob/118b6a765ee79232b1fef0e82ed968a9dbb0e17f/examples/models/line.py
    from bokeh.io import save, output_file
    output_file(fname)
    save(plot)

def view(plot):
    from bokeh.plotting import curdoc
    curdoc().add_root(plot)
