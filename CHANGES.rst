=============
Release notes
=============

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