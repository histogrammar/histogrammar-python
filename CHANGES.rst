=============
Release notes
=============

Version 1.0.29, June 2022
-------------------------
* Fix for machine-level rounding error, which can show up on in num_bins() call of Bin histogram.

Version 1.0.28, June 2022
-------------------------
* Multiple performance updates, to Bin, SparselyBin and Categorize histograms.
* SparselyBin, Categorize: optimized filling with 1-d and 2-d numpy arrays
* Bin, SparselyBin, Categorize: (fast) numpy arrays for bin-centers and bin-labels.
* Count: new, fast filling option when float weight is known.
* util.py: faster get_datatype() and get_ndim() functions.

Version 1.0.27, May 2022
------------------------
* Multiple performance updates, thanks to Simon Brugman.
* Use pandas functions to infer datatypes and return numpy arrays.
* Turn of unnecessary specialize function (slow) for Count objects.

Version 1.0.26, Apr 2022
------------------------
* Added tutorial notebook with exercises.
* Fixed 2d heatmap for categorical histograms, where one column was accidentally dropped.

Version 1.0.25, Apr 2021
------------------------
* Improve null handling in pandas dataframes, by inferring datatype using pandas' infer_dtype function.
* nans in bool columns get converted to "NaN", so the column keeps True and False values in Categorize.
* columns of type object get converted to strings using to_string(), of type string uses only_str().

Version 1.0.24, Apr 2021
------------------------
* Categorize histogram now handles nones and nans in friendlier way, they are converted to "NaN".
* make_histogram() now casts spark nulls to nan in case of numeric columns. scala interprets null as 0.
* SparselyBin histograms did not add up nanflow when added. Now fixed.
* Added unit test for doing checks on null conversion to nans
* Use new histogrammar-scala jar files, v1.0.20
* Added histogrammar-scala v1.0.20 jar files to tests/jars/
