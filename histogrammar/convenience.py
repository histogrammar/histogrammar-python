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

from histogrammar.defs import identity, unweighted
from histogrammar.primitives.average import Average
from histogrammar.primitives.bin import Bin
from histogrammar.primitives.count import Count
from histogrammar.primitives.deviate import Deviate
from histogrammar.primitives.sparselybin import SparselyBin
from histogrammar.primitives.categorize import Categorize
from histogrammar.primitives.select import Select


def Histogram(num, low, high, quantity=identity):
    """Create a conventional histogram that is capable of being filled and added.

    Parameters:
        num (int): the number of bins; must be at least one.
        low (float): the minimum-value edge of the first bin.
        high (float): the maximum-value edge of the last bin; must be strictly greater than `low`.
        quantity (function returning float or string): function that computes the quantity of interest from
            the data. pass on all values by default. If a string is given, quantity is set to identity(string),
            in which case that column is picked up from a pandas df.
    """
    return Bin.ing(num, low, high, quantity, Count.ing(), Count.ing(), Count.ing(), Count.ing())


def HistogramCut(num, low, high, quantity=identity, selection=unweighted):
    """Create a conventional histogram that is capable of being filled and added, with a selection cut.

    Parameters:
        num (int): the number of bins; must be at least one.
        low (float): the minimum-value edge of the first bin.
        high (float): the maximum-value edge of the last bin; must be strictly greater than `low`.
        quantity (function returning float or string): function that computes the quantity of interest from
            the data. pass on all values by default. If a string is given, quantity is set to identity(string),
            in which case that column is picked up from a pandas df.
        selection (function returning boolean): function that computes if data point is accepted or not.
            default is: lamba x: True
    """
    return Select.ing(selection, Bin.ing(num, low, high, quantity, Count.ing(), Count.ing(), Count.ing(), Count.ing()))


def SparselyHistogram(binWidth, quantity=identity, origin=0.0):
    """Create a sparsely binned histogram that is only capable of being added.

    Parameters:
        binWidth (float): the width of a bin.
        quantity (function returning float or string): function that computes the quantity of interest from
            the data. pass on all values by default. If a string is given, quantity is set to identity(string),
            in which case that column is picked up from a pandas df.
        origin (float): the left edge of the bin whose index is zero.
    """
    return SparselyBin.ing(binWidth, quantity, Count.ing(), Count.ing(), origin)


def CategorizeHistogram(quantity=identity):
    """Create a Categorize histogram for categorical features such as strings and booleans

    Parameters:
        quantity (function returning float or string): function that computes the quantity of interest from
            the data. pass on all values by default. If a string is given, quantity is set to identity(string),
            in which case that column is picked up from a pandas df.
    """
    return Categorize.ing(quantity, Count.ing())


def Profile(num, low, high, binnedQuantity, averagedQuantity):
    """Convenience function for creating binwise averages."""
    return Bin.ing(num, low, high, binnedQuantity, Average.ing(averagedQuantity))


def SparselyProfile(binWidth, binnedQuantity, averagedQuantity, origin=0.0):
    """Convenience function for creating sparsely binned binwise averages."""
    return SparselyBin.ing(binWidth, binnedQuantity, Average.ing(averagedQuantity), Count.ing(), origin)


def ProfileErr(num, low, high, binnedQuantity, averagedQuantity):
    """Convenience function for creating a profile plot

    This is a Profile with variances.
    """
    return Bin.ing(num, low, high, binnedQuantity, Deviate.ing(averagedQuantity))


def SparselyProfileErr(binWidth, binnedQuantity, averagedQuantity, origin=0.0):
    """Convenience function for creating a sparsely binned profile plot

    This is a Profile with variances.
    """
    return SparselyBin.ing(binWidth, binnedQuantity, Deviate.ing(averagedQuantity), Count.ing(), origin)


def TwoDimensionallyHistogram(xnum, xlow, xhigh, xquantity, ynum, ylow, yhigh, yquantity):
    """Convenience function for creating a conventional, two-dimensional histogram."""
    return Bin.ing(xnum, xlow, xhigh, xquantity, Bin.ing(ynum, ylow, yhigh, yquantity))


def TwoDimensionallySparselyHistogram(xbinWidth, xquantity, ybinWidth, yquantity, xorigin=0.0, yorigin=0.0):
    """Convenience function for creating a sparsely binned, two-dimensional histogram."""
    return SparselyBin.ing(xbinWidth, xquantity,
                           SparselyBin.ing(ybinWidth, yquantity, Count.ing(), Count.ing(), yorigin),
                           Count.ing(), xorigin)
