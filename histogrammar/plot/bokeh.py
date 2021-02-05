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

# "Public" methods; what we want to attach to the Histogram as a mix-in.
from __future__ import absolute_import

import math

# python 2/3 compatibility fixes
# from histogrammar.util import *


class HistogramMethods(object):
    def plotbokeh(self, glyphType="line", glyphSize=1, fillColor="red",
                  lineColor="black", lineAlpha=1, fillAlpha=0.1, lineDash='solid'):

        # glyphs
        from bokeh.models.glyphs import Rect, Line
        from bokeh.models.renderers import GlyphRenderer
        from bokeh.models.markers import (Circle, Cross,
                                          Diamond, Square,
                                          Triangle)

        # data
        from bokeh.models import ColumnDataSource

        # Parameters of the histogram
        lo = self.low
        hi = self.high
        num = self.num
        bin_width = (hi-lo)/num
        x = list()
        center = lo
        for _ in range(num):
            x.append(center+bin_width/2)
            center += bin_width
        y = self.numericalValues
        ci = [2.*v for v in self.confidenceIntervalValues()]

        source = ColumnDataSource(data=dict(x=x, y=y, ci=ci))

        glyph = None
        if glyphType == "square":
            glyph = Square(
                x='x',
                y='y',
                line_color=lineColor,
                fill_color=fillColor,
                line_alpha=lineAlpha,
                size=glyphSize,
                line_dash=lineDash)
        elif glyphType == "diamond":
            glyph = Diamond(
                x='x',
                y='y',
                line_color=lineColor,
                fill_color=fillColor,
                line_alpha=lineAlpha,
                size=glyphSize,
                line_dash=lineDash)
        elif glyphType == "cross":
            glyph = Cross(
                x='x',
                y='y',
                line_color=lineColor,
                fill_color=fillColor,
                line_alpha=lineAlpha,
                size=glyphSize,
                line_dash=lineDash)
        elif glyphType == "triangle":
            glyph = Triangle(
                x='x',
                y='y',
                line_color=lineColor,
                fill_color=fillColor,
                line_alpha=lineAlpha,
                size=glyphSize,
                line_dash=lineDash)
        elif glyphType == "circle":
            glyph = Circle(
                x='x',
                y='y',
                line_color=lineColor,
                fill_color=fillColor,
                line_alpha=lineAlpha,
                size=glyphSize,
                line_dash=lineDash)
        elif glyphType == "rect":
            glyph = Rect(
                x='x',
                y='y',
                width=bin_width,
                height=0.1,
                fill_alpha=fillAlpha,
                line_color=lineColor,
                fill_color=fillColor)
        elif glyphType == "errors":
            glyph = Rect(
                x='x',
                y='y',
                width=bin_width,
                height='ci',
                fill_alpha=fillAlpha,
                line_color=lineColor,
                fill_color=fillColor)
        elif glyphType == "histogram":
            h = y
            y = [yy/2 for yy in y]
            source = ColumnDataSource(dict(x=x, y=y, h=h))
            glyph = Rect(
                x='x',
                y='y',
                width=bin_width,
                height='h',
                fill_alpha=fillAlpha,
                line_color=lineColor,
                fill_color=fillColor)
        else:
            glyph = Line(
                x='x',
                y='y',
                line_color=lineColor,
                line_alpha=lineAlpha,
                line_width=glyphSize,
                line_dash=lineDash)

        return GlyphRenderer(glyph=glyph, data_source=source)


class SparselyHistogramMethods(object):
    def plotbokeh(self, glyphType="line", glyphSize=1, fillColor="red",
                  lineColor="black", lineAlpha=1, fillAlpha=0.1, lineDash='solid'):

        # glyphs
        from bokeh.models.glyphs import Rect, Line
        from bokeh.models.renderers import GlyphRenderer
        from bokeh.models.markers import (Circle, Cross,
                                          Diamond, Square,
                                          Triangle)

        # data
        from bokeh.models import ColumnDataSource

        # Parameters of the histogram
        lo = self.low
        hi = self.high
        num = self.numFilled
        bin_width = (hi-lo)/num
        x = list()
        center = lo
        for _ in range(num):
            x.append(center+bin_width/2)
            center += bin_width
        y = [v.entries for _, v in sorted(self.bins.items())]

        source = ColumnDataSource(data=dict(x=x, y=y))

        glyph = None
        if glyphType == "square":
            glyph = Square(
                x='x',
                y='y',
                line_color=lineColor,
                fill_color=fillColor,
                line_alpha=lineAlpha,
                size=glyphSize,
                line_dash=lineDash)
        elif glyphType == "diamond":
            glyph = Diamond(
                x='x',
                y='y',
                line_color=lineColor,
                fill_color=fillColor,
                line_alpha=lineAlpha,
                size=glyphSize,
                line_dash=lineDash)
        elif glyphType == "cross":
            glyph = Cross(
                x='x',
                y='y',
                line_color=lineColor,
                fill_color=fillColor,
                line_alpha=lineAlpha,
                size=glyphSize,
                line_dash=lineDash)
        elif glyphType == "triangle":
            glyph = Triangle(
                x='x',
                y='y',
                line_color=lineColor,
                fill_color=fillColor,
                line_alpha=lineAlpha,
                size=glyphSize,
                line_dash=lineDash)
        elif glyphType == "circle":
            glyph = Circle(
                x='x',
                y='y',
                line_color=lineColor,
                fill_color=fillColor,
                line_alpha=lineAlpha,
                size=glyphSize,
                line_dash=lineDash)
        elif glyphType == "rect":
            glyph = Rect(
                x='x',
                y='y',
                width=bin_width,
                height=0.1,
                fill_alpha=fillAlpha,
                line_color=lineColor,
                fill_color=fillColor)
        elif glyphType == "errors":
            ci = [2.*v for v in self.confidenceIntervalValues()]
            source = ColumnDataSource(data=dict(x=x, y=y, ci=ci))
            glyph = Rect(
                x='x',
                y='y',
                width=bin_width,
                height='ci',
                fill_alpha=fillAlpha,
                line_color=lineColor,
                fill_color=fillColor)
        elif glyphType == "histogram":
            h = y
            y = [yy/2 for yy in y]
            source = ColumnDataSource(dict(x=x, y=y, h=h))
            glyph = Rect(
                x='x',
                y='y',
                width=bin_width,
                height='h',
                fill_alpha=fillAlpha,
                line_color=lineColor,
                fill_color=fillColor)
        else:
            glyph = Line(
                x='x',
                y='y',
                line_color=lineColor,
                line_alpha=lineAlpha,
                line_width=glyphSize,
                line_dash=lineDash)

        return GlyphRenderer(glyph=glyph, data_source=source)


class CategorizeHistogramMethods(object):
    pass


class IrregularlyHistogramMethods(object):
    pass


class CentrallyHistogramMethods(object):
    pass


class ProfileMethods(object):
    def plotbokeh(self, glyphType="line", glyphSize=1, fillColor="red",
                  lineColor="black", lineAlpha=1, fillAlpha=0.1, lineDash='solid'):

        # glyphs
        from bokeh.models.glyphs import Rect, Line
        from bokeh.models.renderers import GlyphRenderer
        from bokeh.models.markers import (Circle, Cross,
                                          Diamond, Square,
                                          Triangle)

        # data
        from bokeh.models import ColumnDataSource

        # Parameters of the histogram
        lo = self.low
        hi = self.high
        num = self.num
        bin_width = (hi-lo)/num
        x = list()
        y = list()
        center = lo
        for v in self.values:
            if not math.isnan(v.mean):
                y.append(v.mean)
                x.append(center+bin_width/2)
                center += bin_width

        source = ColumnDataSource(data=dict(x=x, y=y))

        glyph = None
        if glyphType == "square":
            glyph = Square(
                x='x',
                y='y',
                line_color=lineColor,
                fill_color=fillColor,
                line_alpha=lineAlpha,
                size=glyphSize,
                line_dash=lineDash)
        elif glyphType == "diamond":
            glyph = Diamond(
                x='x',
                y='y',
                line_color=lineColor,
                fill_color=fillColor,
                line_alpha=lineAlpha,
                size=glyphSize,
                line_dash=lineDash)
        elif glyphType == "cross":
            glyph = Cross(
                x='x',
                y='y',
                line_color=lineColor,
                fill_color=fillColor,
                line_alpha=lineAlpha,
                size=glyphSize,
                line_dash=lineDash)
        elif glyphType == "triangle":
            glyph = Triangle(
                x='x',
                y='y',
                line_color=lineColor,
                fill_color=fillColor,
                line_alpha=lineAlpha,
                size=glyphSize,
                line_dash=lineDash)
        elif glyphType == "circle":
            glyph = Circle(
                x='x',
                y='y',
                line_color=lineColor,
                fill_color=fillColor,
                line_alpha=lineAlpha,
                size=glyphSize,
                line_dash=lineDash)
        elif glyphType == "histogram":
            w = [bin_width for _ in x]
            h = y
            y = [yy/2 for yy in y]
            source = ColumnDataSource(dict(x=x, y=y, w=w, h=h))
            glyph = Rect(
                x='x',
                y='y',
                width='w',
                height='h',
                fill_alpha=fillAlpha,
                line_color=lineColor,
                fill_color=fillColor)
        else:
            glyph = Line(
                x='x',
                y='y',
                line_color=lineColor,
                line_alpha=lineAlpha,
                line_width=glyphSize,
                line_dash=lineDash)

        return GlyphRenderer(glyph=glyph, data_source=source)


class SparselyProfileMethods(object):
    pass


class ProfileErrMethods(object):
    def plotbokeh(self, glyphType="line", glyphSize=1, fillColor="red",
                  lineColor="black", lineAlpha=1, fillAlpha=0.1, lineDash='solid'):

        # glyphs
        from bokeh.models.glyphs import Rect, Line
        from bokeh.models.renderers import GlyphRenderer
        from bokeh.models.markers import (Circle, Cross,
                                          Diamond, Square,
                                          Triangle)

        # data
        from bokeh.models import ColumnDataSource

        from math import sqrt

        # Parameters of the histogram
        lo = self.low
        hi = self.high
        num = self.num
        bin_width = (hi-lo)/num
        x = list()
        y = list()
        center = lo
        for v in self.values:
            if not math.isnan(v.mean):
                y.append(v.mean)
                x.append(center+bin_width/2)
                center += bin_width

        source = ColumnDataSource(data=dict(x=x, y=y))

        glyph = None
        if glyphType == "square":
            glyph = Square(
                x='x',
                y='y',
                line_color=lineColor,
                fill_color=fillColor,
                line_alpha=lineAlpha,
                size=glyphSize,
                line_dash=lineDash)
        elif glyphType == "diamond":
            glyph = Diamond(
                x='x',
                y='y',
                line_color=lineColor,
                fill_color=fillColor,
                line_alpha=lineAlpha,
                size=glyphSize,
                line_dash=lineDash)
        elif glyphType == "cross":
            glyph = Cross(
                x='x',
                y='y',
                line_color=lineColor,
                fill_color=fillColor,
                line_alpha=lineAlpha,
                size=glyphSize,
                line_dash=lineDash)
        elif glyphType == "triangle":
            glyph = Triangle(
                x='x',
                y='y',
                line_color=lineColor,
                fill_color=fillColor,
                line_alpha=lineAlpha,
                size=glyphSize,
                line_dash=lineDash)
        elif glyphType == "circle":
            glyph = Circle(
                x='x',
                y='y',
                line_color=lineColor,
                fill_color=fillColor,
                line_alpha=lineAlpha,
                size=glyphSize,
                line_dash=lineDash)
        elif glyphType == "errors":
            w = [bin_width for _ in x]
            h = [sqrt(v.variance/v.entries) if v.entries > 0 else 0.0 for v in self.values]
            source = ColumnDataSource(dict(x=x, y=y, w=w, h=h))
            glyph = Rect(
                x='x',
                y='y',
                width='w',
                height='h',
                fill_alpha=fillAlpha,
                line_color=lineColor,
                fill_color=fillColor)
        elif glyphType == "histogram":
            w = [bin_width for _ in x]
            h = y
            y = [yy/2 for yy in y]
            source = ColumnDataSource(dict(x=x, y=y, w=w, h=h))
            glyph = Rect(
                x='x',
                y='y',
                width='w',
                height='h',
                fill_alpha=fillAlpha,
                line_color=lineColor,
                fill_color=fillColor)
        else:
            glyph = Line(
                x='x',
                y='y',
                line_color=lineColor,
                line_alpha=lineAlpha,
                line_width=glyphSize,
                line_dash=lineDash)

        return GlyphRenderer(glyph=glyph, data_source=source)


class SparselyProfileErrMethods(object):
    pass


class StackedHistogramMethods(object):
    nMaxStacked = 10
    glyphTypeDefaults = ["circle"]*nMaxStacked
    glyphSizeDefaults = [1]*nMaxStacked
    fillColorDefaults = ["red"]*nMaxStacked
    lineColorDefaults = ["red"]*nMaxStacked
    lineAlphaDefaults = [1]*nMaxStacked
    fillAlphaDefaults = [0.1]*nMaxStacked
    lineDashDefaults = ["solid"]*nMaxStacked

    def plotbokeh(self, glyphTypes=glyphTypeDefaults, glyphSizes=glyphSizeDefaults, fillColors=fillColorDefaults,
                  lineColors=lineColorDefaults, lineAlphas=lineAlphaDefaults, fillAlphas=fillAlphaDefaults,
                  lineDashes=lineDashDefaults):
        nChildren = len(self.children)-1

        assert len(glyphSizes) >= nChildren
        assert len(glyphTypes) >= nChildren
        assert len(fillColors) >= nChildren
        assert len(lineColors) >= nChildren
        assert len(lineAlphas) >= nChildren
        assert len(fillAlphas) >= nChildren
        assert len(lineDashes) >= nChildren

        stackedGlyphs = list()
        # for ichild, p in enumerate(self.children,start=1):
        for ichild in range(nChildren):
            stackedGlyphs.append(self.children[ichild+1].plotbokeh(glyphTypes[ichild],
                                                                   glyphSizes[ichild],
                                                                   fillColors[ichild],
                                                                   lineColors[ichild],
                                                                   lineAlphas[ichild],
                                                                   fillAlphas[ichild],
                                                                   lineDashes[ichild]))

        return stackedGlyphs


class PartitionedHistogramMethods(object):
    pass


class FractionedHistogramMethods(object):
    pass


class TwoDimensionallyHistogramMethods(object):
    pass


class SparselyTwoDimensionallyHistogramMethods(object):
    pass


class IrregularlyTwoDimensionallyHistogramMethods(object):
    pass


class CentrallyTwoDimensionallyHistogramMethods(object):
    pass


def plot(xLabel='x', yLabel='y', *args):

    from bokeh.models import DataRange1d, Plot, LinearAxis
    from bokeh.models import PanTool, WheelZoomTool

    xdr = DataRange1d()
    ydr = DataRange1d()

    plot = Plot(x_range=xdr, y_range=ydr, min_border=80)

    extra = list()
    if not isinstance(xLabel, str) and not isinstance(yLabel, str):
        extra.append(xLabel)
        extra.append(yLabel)
        xLabel = 'x'
        yLabel = 'y'
    elif not isinstance(xLabel, str):
        extra.append(xLabel)
        xLabel = 'x'
    elif not isinstance(yLabel, str):
        extra.append(yLabel)
        yLabel = 'y'

    args = extra+list(args)
    for renderer in args:
        if not isinstance(renderer, list):
            plot.renderers.append(renderer)
        else:
            plot.renderers.extend(renderer)

    # axes
    xaxis = LinearAxis(axis_label=xLabel)
    plot.add_layout(xaxis, 'below')
    yaxis = LinearAxis(axis_label=yLabel)
    plot.add_layout(yaxis, 'left')
    # add grid to the plot
    # plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
    # plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))

    # interactive tools
    plot.add_tools(PanTool(), WheelZoomTool())  # , SaveTool())

    return plot


def save(plot, fname):
    # SaveTool https://github.com/bokeh/bokeh/blob/118b6a765ee79232b1fef0e82ed968a9dbb0e17f/examples/models/line.py
    from bokeh.io import save, output_file
    output_file(fname)
    save(plot)


def view(plot, show=False):
    from bokeh.plotting import curdoc
    from bokeh.client import push_session
    if show:
        session = push_session(curdoc())
        session.show(plot)
    else:
        curdoc().add_root(plot)
