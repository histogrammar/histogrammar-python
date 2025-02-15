"""Copyright Eskapade:
License Apache-2: https://github.com/KaveIO/Eskapade-Core/blob/master/LICENSE
Reference link:
https://github.com/KaveIO/Eskapade-Spark/blob/master/python/eskapadespark/links/spark_histogrammar_filler.py
All modifications copyright ING WBAA.
"""

import numpy as np
from tqdm import tqdm

import histogrammar as hg

from .histogram_filler_base import HistogramFillerBase

try:
    from pyspark.sql import DataFrame
    from pyspark.sql import functions as f
    from pyspark.sql.functions import approxCountDistinct
except (ModuleNotFoundError, AttributeError):
    pass


class SparkHistogrammar(HistogramFillerBase):
    """Fill histogrammar histograms with Spark.

    Algorithm to fill histogrammar style bin, sparse-bin and category histograms
    with Spark. Timestamp features are converted to nanoseconds before the binning
    is applied. Final histograms are stored in the datastore.
    """

    def __init__(
        self,
        features=None,
        binning="unit",
        bin_specs=None,
        time_axis="",
        var_dtype=None,
        read_key=None,
        store_key=None,
        nbins_1d=40,
        nbins_2d=20,
        nbins_3d=10,
        max_nunique=500,
    ):
        """Initialize module instance.

        Store and do basic check on the attributes HistogramFillerBase.

        :param list features: colums to pick up from input data. (default is all features)
            For multi-dimensional histograms, separate the column names with a :

            Example features list is:

            .. code-block:: python

                features = ['x', 'date', 'date:x', 'date:y', 'date:x:y']

        :param str binning: default binning to revert to in case bin_specs not supplied. options are:
            "unit" or "auto", default is "unit". When using "auto", semi-clever binning is automatically done.
        :param dict bin_specs: dictionaries used for rebinning numeric or timestamp features

            Example bin_specs dictionary is:

            .. code-block:: python

                bin_specs = {'x': {'binWidth': 1, 'origin': 0},
                             'y': {'num': 10, 'low': 0.0, 'high': 2.0},
                             'x:y': [{}, {'num': 5, 'low': 0.0, 'high': 1.0}],
                             'a': {'edges': [0, 2, 10, 11, 21, 101]},
                             'b': {'centers': [1, 6, 10.5, 16, 20, 100]},
                             'c': {'max': True},
                             'd': {'min': True},
                             'e': {'sum': True},
                             'f': {'average': True},
                             'a:f': [{'edges': [0, 10, 101]}, {'average': True}],
                             'g': {'thresholds': [0, 2, 10, 11, 21, 101]},
                             'h': {'bag': True},
                             }

            In the bin specs for x:y, x reverts to the 1-dim setting.

        :param str time_axis: name of datetime feature, used as time axis, eg 'date'. if True, will be guessed.
            If time_axis is set, if no features given, features becomes: ['date:x', 'date:y', 'date:z'] etc.
        :param dict var_dtype: dictionary with specified datatype per feature (optional)
        :param str read_key: key of input histogram-dict to read from data store .
            (only required when calling transform(datastore) as module)
        :param str store_key: key of output data to store in data store
            (only required when calling transform(datastore) as module)
        :param int nbins_1d: auto-binning number of bins for 1d histograms. default is 40.
        :param int nbins_2d: auto-binning number of bins for 2d histograms. default is 20.
        :param int nbins_3d: auto-binning number of bins for 3d histograms. default is 10.
        :param int max_nunique: auto-binning threshold for unique categorical values. default is 500.
        """
        HistogramFillerBase.__init__(
            self,
            features,
            binning,
            bin_specs,
            time_axis,
            var_dtype,
            read_key,
            store_key,
            nbins_1d,
            nbins_2d,
            nbins_3d,
            max_nunique,
        )
        self._unit_timestamp_specs = {
            k: float(self._unit_timestamp_specs[k]) for i, k in enumerate(self._unit_timestamp_specs)
        }

    def assert_dataframe(self, df):
        """Check that input data is a filled spark data frame.

        :param df: input (spark) data frame
        """
        if not isinstance(df, DataFrame):
            raise TypeError("retrieved object not of type Spark DataFrame")
        assert len(df.head(1)) != 0, "input dataframe is empty"
        return df

    def get_features(self, df):
        """Get columns of dataframe

        :param df: input spark dataframe
        """
        return df.columns

    def get_quantiles(self, df, quantiles=[0.05, 0.95], columns=[]):
        """return dict with quantiles for given columns

        :param df: input (spark) data frame
        :param quantiles: list of quantiles. default is [0.05, 0.95]
        :param columns: columns to select. default is all.
        """
        if len(columns) == 0:
            return {}
        qsl = df.approxQuantile(columns, quantiles, 0.25)
        return dict(zip(columns, qsl))

    def get_nunique(self, df, columns=[]):
        """return dict with number of unique entries for given columns

        :param df: input (spark) data frame
        :param columns: columns to select (optional)
        """
        if not columns:
            columns = df.columns
        qdf = df.agg(*(approxCountDistinct(f.col(c)).alias(c) for c in columns))
        return qdf.toPandas().T[0].to_dict()

    def get_data_type(self, df, col):
        """Get data type of dataframe column.

        :param df: input data frame
        :param str col: column
        """
        if col not in df.columns:
            raise KeyError(f'Column "{col:s}" not in input dataframe.')
        dt = dict(df.dtypes)[col]
        # spark conversions to numpy or python equivalent
        if dt == "string":
            dt = "str"
        elif dt in ["timestamp", "date"]:
            dt = np.datetime64
        elif dt == "boolean":
            dt = bool
        elif dt == "bigint":
            dt = np.int64
        elif dt.startswith("decimal("):
            return np.dtype(float, metadata={"decimal": True})

        return np.dtype(dt)

    def process_features(self, df, cols_by_type):
        """Process features before histogram filling.

        Specifically, in this case convert timestamp features to nanoseconds

        :param df: input data frame
        :return: output data frame with converted timestamp features
        :rtype: DataFrame
        """
        # make alias df for value counting (used below)
        idf = df.alias("")

        # timestamp variables are converted here to ns since 1970-1-1
        # histogrammar does not (yet) support long integers, so convert timestamps to float
        for col in cols_by_type["dt"]:
            self.logger.debug(f'Converting column "{col}" of type "{self.var_dtype[col]}" to nanosec.')
            # first cast to timestamp (in case column is stored as date)
            to_ns = f.col(col).cast("timestamp").cast("float") * 1e9
            idf = idf.withColumn(col, to_ns)

        # spark nulls are interpreted to 0 when cast to double in scala, done when given as input to numeric histograms
        # in columns that have them, replace by nones by nans
        for col in cols_by_type["num"]:
            if len(idf.where(f.col(col).isNull()).limit(1).collect()) > 0:
                self.logger.debug(f'In numeric column "{col}" converting each None to NaN.')
                idf = idf.withColumn(
                    col,
                    f.when(f.col(col).isNotNull(), f.col(col)).otherwise(float("nan")),
                )

        return idf

    def construct_empty_hist(self, df, features):
        """Create an (empty) histogram of right type.

        Create a multi-dim histogram by iterating through the features in
        reverse order and passing a single-dim hist as input to the next
        column.

        :param df: input dataframe
        :param list features: histogram features
        :return: created histogram
        :rtype: histogrammar.Count
        """
        hist = hg.Count()

        # create a multi-dim histogram by iterating through
        # the features in reverse order and passing a single-dim hist
        # as input to the next column
        revcols = list(reversed(features))
        for idx, col in enumerate(revcols):
            # histogram type depends on the data type
            dt = self.var_dtype[col]
            quant = df[col]
            hist = self.get_hist_bin(hist, features, quant, col, dt)

        return hist

    def fill_histograms(self, idf):
        """Fill the histograms

        :param idf: input data frame used for filling histogram
        """
        for cols in tqdm(self.features, ncols=100):
            self.logger.debug('Processing feature "{cols}".'.format(cols=":".join(cols)))
            self.fill_histogram(idf, cols)

    def fill_histogram(self, idf, features):
        """Fill input histogram with column(s) of input dataframe.

        :param idf: input data frame used for filling histogram
        :param list features: histogram column(s)
        """
        name = ":".join(features)
        if name not in self._hists:
            # create an (empty) histogram of right type
            self._hists[name] = self.construct_empty_hist(idf, features)
        hist = self._hists[name]

        # do the actual filling
        hist.fill.sparksql(idf)
        self._hists[name] = hist

    def _execute(self, df):
        df.persist()
        hists = super()._execute(df)
        df.unpersist()
        return hists
