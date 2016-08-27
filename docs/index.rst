Histogrammar |version| for Python
=================================

All aggregation primitives descend from two classes, :doc:`Container <histogrammar.defs.Container>` and :doc:`Factory <histogrammar.defs.Factory>`. Container defines all the methods for the primitive to aggregate and contain data, while Factory has methods for making containers. (In other languages, the two roles are distinct.)

The "functions" passed to these primitives may be Python lambda functions, normally defined functions (with ``def``), or strings, which may be interpreted different ways by different back-ends. All primitives immediately wrap your functions as :doc:`UserFcn <histogrammar.util.UserFcn>`, which are serializable (with ``pickle``), may be cached (:doc:`CachedFcn <histogrammar.util.CachedFcn>`), and may have a name. Although the primitives wrap your function automatically, you may do it yourself to add features, like caching or a name. See :doc:`serializable <histogrammar.util.serializable>`, :doc:`cached <histogrammar.util.cached>`, and :doc:`named <histogrammar.util.named>`.
      
The primitive classes are listed below, grouped by kind. See the index for a list of all classes, members, and functions.

Zeroth kind: depend only on weights
-----------------------------------

:doc:`Count <histogrammar.primitives.count.Count>`: sum of weights
    Count entries by accumulating the sum of all observed weights or a sum of transformed weights (e.g. sum of squares of weights).

First kind: aggregate a data without sub-aggregators
----------------------------------------------------

:doc:`Sum <histogrammar.primitives.sum.Sum>`: sum of a given quantity
    Accumulate the (weighted) sum of a given quantity, calculated from the data.

:doc:`Average <histogrammar.primitives.average.Average>`: mean of a quantity
    Accumulate the weighted mean of a given quantity.

:doc:`Deviate <histogrammar.primitives.deviate.Deviate>`: mean and variance
    Accumulate the weighted mean and weighted variance of a given quantity.

:doc:`Minimize <histogrammar.primitives.minmax.Minimize>`: minimum value
    Find the minimum value of a given quantity. If no data are observed, the result is NaN.

:doc:`Maximize <histogrammar.primitives.minmax.Maximize>`: maximum value
    Find the maximum value of a given quantity. If no data are observed, the result is NaN.

:doc:`Bag <histogrammar.primitives.bag.Bag>`: accumulate values for scatter plots
    Accumulate raw numbers, vectors of numbers, or strings, with identical values merged.

Second kind: pass to different sub-aggregators based on values seen in data
---------------------------------------------------------------------------

:doc:`Bin <histogrammar.primitives.bin.Bin>`: regular binning for histograms
    Split a quantity into equally spaced bins between a low and high threshold and fill exactly one bin per datum.

:doc:`SparselyBin <histogrammar.primitives.sparselybin.SparselyBin>`: ignore zeros
    Split a quantity into equally spaced bins, creating them whenever their entries would be non-zero. Exactly one sub-aggregator is filled per datum.

:doc:`CentrallyBin <histogrammar.primitives.centrallybin.CentrallyBin>`: irregular but fully partitioning
    Split a quantity into bins defined by irregularly spaced bin centers, with exactly one sub-aggregator filled per datum (the closest one).

:doc:`IrregularlyBin <histogrammar.primitives.irregularlybin.IrregularlyBin>`: exclusive filling
    Accumulate a suite of aggregators, each between two thresholds, filling exactly one per datum.

:doc:`Categorize <histogrammar.primitives.categorize.Categorize>`: string-valued bins, bar charts
    Split a given quantity by its categorical value and fill only one category per datum.

:doc:`Fraction <histogrammar.primitives.fraction.Fraction>`: efficiency plots
    Accumulate two aggregators, one containing only entries that pass a given selection (numerator) and another that contains all entries (denominator).

:doc:`Stack <histogrammar.primitives.stack.Stack>`: cumulative filling
    Accumulates a suite of aggregators, each filtered with a tighter selection on the same quantity.

:doc:`Select <histogrammar.primitives.select.Select>`: apply a cut
    Filter or weight data according to a given selection.

Third kind: broadcast to every sub-aggregator, independent of data
------------------------------------------------------------------

:doc:`Label <histogrammar.primitives.collection.Label>`: directory with string-based keys
    Accumulate any number of aggregators of the same type and label them with strings. Every sub-aggregator is filled with every input datum.

:doc:`UntypedLabel <histogrammar.primitives.collection.UntypedLabel>`: directory of different types
    Accumulate any number of aggregators of any type and label them with strings. Every sub-aggregator is filled with every input datum.

:doc:`Index <histogrammar.primitives.collection.Index>`: list with integer keys
    Accumulate any number of aggregators of the same type in a list. Every sub-aggregator is filled with every input datum.

:doc:`Branch <histogrammar.primitives.collection.Branch>`: tuple of different types
    Accumulate aggregators of different types, indexed by i0 through i9. Every sub-aggregator is filled with every input datum.
