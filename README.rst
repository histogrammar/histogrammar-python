==================================
histogrammar Python implementation
==================================

histogrammar is a Python package for creating histograms. histogrammar has multiple histogram types,
supports numeric and categorical features, and works with Numpy arrays and Pandas and Spark dataframes.
Once a histogram is filled, it's easy to plot it, store it in JSON format (and retrieve it), or convert
it to Numpy arrays for further analysis.

At its core histogrammar is a suite of data aggregation primitives designed for use in parallel processing.
In the simplest case, you can use this to compute histograms, but the generality of the primitives
allows much more.

Several common histogram types can be plotted in Matplotlib, Bokeh and PyROOT with a single method call.
If Numpy or Pandas is available, histograms and other aggregators can be filled from arrays ten to a hundred times
more quickly via Numpy commands, rather than Python for loops. If PyROOT is available, histograms and other
aggregators can be filled from ROOT TTrees hundreds of times more quickly by JIT-compiling a specialized C++ filler.
Histograms and other aggregators may also be converted into CUDA code for inclusion in a GPU workflow. And if
PyCUDA is available, they can also be filled from Numpy arrays by JIT-compiling the CUDA code.
This Python implementation of histogrammar been tested to guarantee compatibility with its Scala implementation.

Latest Python release: v1.0.21 (Mar 2021).

Announcements
=============

Spark 3.0
---------

With Spark 3.0, based on Scala 2.12, make sure to pick up the correct histogrammar jar file:

.. code-block:: python

  spark = SparkSession.builder.config("spark.jars.packages", "io.github.histogrammar:histogrammar_2.12:1.0.11,io.github.histogrammar:histogrammar-sparksql_2.12:1.0.11").getOrCreate()

For Spark 2.X compiled against scala 2.11, in the string above simply replace "2.12" with "2.11".

February, 2021

Example notebooks
=================

.. list-table::
   :widths: 80 20
   :header-rows: 1

   * - Tutorial
     - Colab link
   * - `Basic tutorial <https://nbviewer.jupyter.org/github/histogrammar/histogrammar-python/blob/master/histogrammar/notebooks/histogrammar_tutorial_basic.ipynb>`_
     - |notebook_basic_colab|
   * - `Detailed example (featuring configuration, Apache Spark and more) <https://nbviewer.jupyter.org/github/histogrammar/histogrammar-python/blob/master/histogrammar/notebooks/histogrammar_tutorial_advanced.ipynb>`_
     - |notebook_advanced_colab|

Documentation
=============

See `histogrammar-docs <https://histogrammar.github.io/histogrammar-docs/>`_ for a complete introduction to `histogrammar`.
(A bit old but still good.) There you can also find documentation about the Scala implementation of `histogrammar`.

Check it out
============

The `historgrammar` library requires Python 3.6+ and is pip friendly. To get started, simply do:

.. code-block:: bash

  $ pip install histogrammar

or check out the code from our GitHub repository:

.. code-block:: bash

  $ git clone https://github.com/histogrammar/histogrammar-python
  $ pip install -e histogrammar-python

where in this example the code is installed in edit mode (option -e).

You can now use the package in Python with:

.. code-block:: python

  import histogrammar

**Congratulations, you are now ready to use the histogrammar library!**

Quick run
=========

As a quick example, you can do:

.. code-block:: python

  import pandas as pd
  import histogrammar as hg
  from histogrammar import resources

  # open synthetic data
  df = pd.read_csv(resources.data('test.csv.gz'), parse_dates=['date'])
  df.head()

  # create a histogram, tell it to look for column 'age'
  # fill the histogram with column 'age' and plot it
  hist = hg.Histogram(num=100, low=0, high=100, quantity='age')
  hist.fill.numpy(df)
  hist.plot.matplotlib()

  # generate histograms of all features in the dataframe using automatic binning
  # (importing histogrammar automatically adds this functionality to a pandas or spark dataframe)
  hists = df.hg_make_histograms()
  print(hists.keys())

  # multi-dimensional histograms are also supported. e.g. features longitude vs latitude
  hists = df.hg_make_histograms(features=['longitude:latitude'])
  ll = hists['longitude:latitude']
  ll.plot.matplotlib()

  # store histogram and retrieve it again
  ll.toJsonFile('longitude_latitude.json')
  ll2 = hg.Factory().fromJsonFile('longitude_latitude.json')

These examples also work with Spark dataframes (sdf):

.. code-block:: python

  from pyspark.sql.functions import col
  hist = hg.Histogram(num=100, low=0, high=100, quantity=col('age'))
  hist.fill.sparksql(sdf)

For more examples please see the example notebooks and tutorials.


Project contributors
====================

This package was originally authored by DIANA-HEP and is now maintained by volunteers.

Contact and support
===================

* Issues & Ideas & Support: https://github.com/histogrammar/histogrammar-python/issues

Please note that `histogrammar` is supported only on a best-effort basis.

License
=======
`histogrammar` is completely free, open-source and licensed under the `Apache-2.0 license <https://en.wikipedia.org/wiki/Apache_License>`_.

.. |notebook_basic_colab| image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Colab
    :target: https://colab.research.google.com/github/histogrammar/histogrammar-python/blob/master/histogrammar/notebooks/histogrammar_tutorial_basic.ipynb
.. |notebook_advanced_colab| image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Colab
    :target: https://colab.research.google.com/github/histogrammar/histogrammar-python/blob/master/histogrammar/notebooks/histogrammar_tutorial_advanced.ipynb
