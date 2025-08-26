=============
Release notes
=============

Version 1.1.1, Aug 2024
-----------------------
* Compatibility with numpy v2.3: converting np.number to a dtype no longer allowed.

Version 1.1.0, Dec 2024
-----------------------
* Removed all ROOT, cpp and cuda code, it was no longer supported.

Version 1.0.34, Dec 2024
------------------------
* Fix typo in build pipeline Python versions config list.
* Fix error in SparselyBin __eq__ method.
* Fix test utility corner case error (test_numpy.twosigfigs function).
* Fix error in test context manager for pandas which prevented execution of tests.
* Fix error in expected bin count in test_numpy.test_n_bins test.
* Prevent logging zero execution time TestNumpy class.

* Remove Python 3.8 environment from build pipeline.
* Support numpy >= 2.0.0 (np.string_ -> np.bytes_, np.unicode_ -> np.str_).
* Remove uses of pd.util.testing.makeMixedDataFrame not available in pandas >= 2.0.0.
* Switch from 'pkg_resources' to 'importlib' module for resolving package files.
* Switch from 'distutils.spawn' to 'shutil.which' for finding nvcc command.

* Remove unused test_gpu.twosigfigs function.
* Refactor tests with Numpy() and Pandas() context managers to use single 'with' statement.

* Switch from setup.py to pyproject.toml
* Add numpy<2,pandas<2 test environment to build pipeline test matrix

Version 1.0.33, Dec 2022
------------------------
* fix of get_sub_hist() when Bin histogram is filled only with nans.

Version 1.0.32, Sep 2022
------------------------
* Support for decimal datetype in pandas and spark.

Version 1.0.31, Aug 2022
------------------------
* fix of spark df timestamp datatype detection (#59)
* fix for invalid bin_edges for SparselyBin histogram (#60)

Version 1.0.30, June 2022
-------------------------
* Fix for machine-level rounding error, which can show up on in num_bins() call of Bin histogram.
* supersedes broken v1.0.29

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
