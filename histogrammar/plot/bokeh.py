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
            w = [bin_width for _ in x]
            h = y
            y = [yy/2 for yy in y]
            source = ColumnDataSource(dict(x=x, y=y, w=w, h=h))
            glyph = Rect(x='x', y='y', width='w', height='h', fill_alpha=fill_alpha, line_color=line_color, fill_color=fill_color)
        else: glyph = Line(x='x', y='y',line_color=line_color,line_alpha=line_alpha,line_width=glyphSize,line_dash=line_dash)

        return GlyphRenderer(glyph=glyph,data_source=source)


def plot(xLabel='x',yLabel='y',**kwargs):
    from bokeh.models import DataRange1d, Plot, LinearAxis, Grid
    from bokeh.models import PanTool, WheelZoomTool

    xdr = DataRange1d()
    ydr = DataRange1d()

    plot = Plot(x_range=xdr, y_range=ydr, min_border=80)

    for _,renderer in kwargs.items():
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
    from bokeh.io import save
    save(plot,fname)

def view(plot):
    #FIXME tests with the bokeh serve pending
    from bokeh.plotting import show
    show(plot)
    #document = Document()
    #session = push_session(document)
    #document.add_root(plot)
    #session.show(plot)

class ProfileMethods(object):
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
        #elif glyphType == "errors": glyph = Rect(x='x', y='y', width=bin_width, height=0.1, fill_alpha=fill_alpha, line_color=line_color, fill_color=fill_color)
        elif glyphType == "histogram":
            w = [bin_width for _ in x]
            h = y
            y = [yy/2 for yy in y]
            source = ColumnDataSource(dict(x=x, y=y, w=w, h=h))
            glyph = Rect(x='x', y='y', width='w', height='h', fill_alpha=fill_alpha, line_color=line_color, fill_color=fill_color)
        else: glyph = Line(x='x', y='y',line_color=line_color,line_alpha=line_alpha,line_width=glyphSize,line_dash=line_dash)

        return GlyphRenderer(glyph=glyph,data_source=source)
