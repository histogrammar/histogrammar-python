from os.path import abspath, dirname, join

import pandas as pd
import pytest

from histogrammar.dfinterface.spark_histogrammar import SparkHistogrammar
from histogrammar.dfinterface.make_histograms import make_histograms


try:
    from pyspark.sql import SparkSession
    from pyspark import __version__ as pyspark_version

    spark_found = True
except (ModuleNotFoundError, AttributeError):
    spark_found = False


def get_spark():
    if not spark_found:
        return None

    current_path = dirname(abspath(__file__))

    scala = '2.12' if int(pyspark_version[0]) >= 3 else '2.11'
    hist_spark_jar = join(current_path, f"jars/histogrammar-sparksql_{scala}-1.0.20.jar")
    hist_jar = join(current_path, f"jars/histogrammar_{scala}-1.0.20.jar")

    spark = (
        SparkSession.builder.master("local")
        .appName("histogrammar-pytest")
        .config("spark.jars", f"{hist_spark_jar},{hist_jar}")
        .config("spark.sql.execution.arrow.enabled", "false")
        .config("spark.sql.session.timeZone", "GMT")
        .getOrCreate()
    )
    return spark


@pytest.fixture
def spark_co():
    """
    :return: Spark configuration
    """
    spark = get_spark()
    return spark


@pytest.mark.spark
@pytest.mark.skipif(not spark_found, reason="spark not found")
@pytest.mark.filterwarnings(
    "ignore:createDataFrame attempted Arrow optimization because"
)
def test_get_histograms(spark_co):
    pytest.age["data"]["name"] = "b'age'"
    pytest.company["data"]["name"] = "b'company'"
    pytest.eyesColor["data"]["name"] = "b'eyeColor'"
    pytest.gender["data"]["name"] = "b'gender'"
    pytest.isActive["data"]["name"] = "b'isActive'"
    pytest.latitude["data"]["name"] = "b'latitude'"
    pytest.longitude["data"]["name"] = "b'longitude'"
    pytest.transaction["data"]["name"] = "b'transaction'"

    pytest.latitude_longitude["data"]["name"] = "b'latitude:longitude'"
    pytest.latitude_longitude["data"]["bins:name"] = "unit_func"

    spark = spark_co

    spark_df = spark.createDataFrame(pytest.test_df)

    spark_filler = SparkHistogrammar(
        features=[
            "date",
            "isActive",
            "age",
            "eyeColor",
            "gender",
            "company",
            "latitude",
            "longitude",
            ["isActive", "age"],
            ["latitude", "longitude"],
            "transaction",
        ],
        bin_specs={
            "transaction": {"num": 100, "low": -2000, "high": 2000},
            "longitude": {"bin_width": 5.0, "bin_offset": 0.0},
            "latitude": {"bin_width": 5.0, "bin_offset": 0.0},
        },
        read_key="input",
        store_key="output",
    )

    # test get_histograms() function call
    current_hists = spark_filler.get_histograms(spark_df)
    # current_hists = make_histograms(spark_df, features, bin_specs)
    assert current_hists["age"].toJson() == pytest.age
    assert current_hists["company"].toJson() == pytest.company
    assert current_hists["eyeColor"].toJson() == pytest.eyesColor
    assert current_hists["gender"].toJson() == pytest.gender
    assert current_hists["latitude"].toJson() == pytest.latitude
    assert current_hists["longitude"].toJson() == pytest.longitude
    assert current_hists["transaction"].toJson() == pytest.transaction

    # import json
    # with open('tests/popmon/hist/resource/transaction.json', 'w') as outfile:
    #     json.dump(current_hists["transaction"].toJson(), outfile, indent=4)


@pytest.mark.spark
@pytest.mark.skipif(not spark_found, reason="spark not found")
@pytest.mark.filterwarnings(
    "ignore:createDataFrame attempted Arrow optimization because"
)
def test_get_histograms_module(spark_co):
    pytest.age["data"]["name"] = "b'age'"
    pytest.company["data"]["name"] = "b'company'"
    pytest.eyesColor["data"]["name"] = "b'eyeColor'"
    pytest.gender["data"]["name"] = "b'gender'"
    pytest.isActive["data"]["name"] = "b'isActive'"
    pytest.latitude["data"]["name"] = "b'latitude'"
    pytest.longitude["data"]["name"] = "b'longitude'"

    pytest.latitude_longitude["data"]["name"] = "b'latitude:longitude'"
    pytest.latitude_longitude["data"]["bins:name"] = "unit_func"

    spark = spark_co

    spark_df = spark.createDataFrame(pytest.test_df)

    spark_filler = SparkHistogrammar(
        features=[
            "date",
            "isActive",
            "age",
            "eyeColor",
            "gender",
            "company",
            "latitude",
            "longitude",
            ["isActive", "age"],
            ["latitude", "longitude"],
        ],
        bin_specs={
            "longitude": {"bin_width": 5.0, "bin_offset": 0.0},
            "latitude": {"bin_width": 5.0, "bin_offset": 0.0},
        },
        read_key="input",
        store_key="output",
    )

    # test transform() function call
    datastore = spark_filler.transform(datastore={"input": spark_df})

    assert "output" in datastore
    current_hists = datastore["output"]
    assert current_hists["age"].toJson() == pytest.age
    assert current_hists["company"].toJson() == pytest.company
    assert current_hists["eyeColor"].toJson() == pytest.eyesColor
    assert current_hists["gender"].toJson() == pytest.gender
    assert current_hists["latitude"].toJson() == pytest.latitude
    assert current_hists["longitude"].toJson() == pytest.longitude
    # assert current_hists['date'].toJson() == pytest.date
    # assert current_hists['isActive'].toJson() == pytest.isActive
    # assert current_hists['isActive:age'].toJson() == pytest.isActive_age
    # assert current_hists['latitude:longitude'].toJson() == pytest.latitude_longitude


@pytest.mark.spark
@pytest.mark.skipif(not spark_found, reason="spark not found")
@pytest.mark.filterwarnings(
    "ignore:createDataFrame attempted Arrow optimization because"
)
def test_get_histograms_timestamp(spark_co):
    from pyspark.sql.functions import to_timestamp

    spark = spark_co

    data_date = [
        "2018-12-10 00:00:00",
        "2018-12-10 00:00:00",
        "2018-12-10 00:00:00",
        "2018-12-10 00:00:00",
        "2018-12-10 00:00:00",
        "2018-12-17 00:00:00",
        "2018-12-17 00:00:00",
        "2018-12-17 00:00:00",
        "2018-12-17 00:00:00",
        "2018-12-19 00:00:00",
    ]

    df = pd.DataFrame(data_date, columns=["dt"])
    sdf = spark.createDataFrame(df).withColumn(
        "dt", to_timestamp("dt", "yyyy-MM-dd HH:mm:ss")
    )
    expected = {
        "data": {
            "binWidth": 2592000000000000.0,
            "bins": {"108": 9.0, "109": 1.0},
            "bins:type": "Count",
            "entries": 10.0,
            "name": "b'dt'",
            "nanflow": 0.0,
            "nanflow:type": "Count",
            "origin": 1.2625632e18,
        },
        "type": "SparselyBin",
        "version": "1.0",
    }
    filler = SparkHistogrammar(features=["dt"])
    current_hists = filler.get_histograms(sdf)
    assert current_hists["dt"].toJson() == expected


@pytest.mark.spark
@pytest.mark.skipif(not spark_found, reason="spark not found")
@pytest.mark.filterwarnings(
    "ignore:createDataFrame attempted Arrow optimization because"
)
def test_get_histograms_date(spark_co):
    from pyspark.sql.functions import to_date

    spark = spark_co

    data_date = [
        "2018-12-10",
        "2018-12-10",
        "2018-12-10",
        "2018-12-10",
        "2018-12-10",
        "2018-12-17",
        "2018-12-17",
        "2018-12-17",
        "2018-12-17",
        "2018-12-19",
    ]

    df = pd.DataFrame(data_date, columns=["dt"])
    sdf = spark.createDataFrame(df).withColumn("dt", to_date("dt", "yyyy-MM-dd"))
    expected = {
        "data": {
            "binWidth": 2592000000000000.0,
            "bins": {"108": 9.0, "109": 1.0},
            "bins:type": "Count",
            "entries": 10.0,
            "name": "b'dt'",
            "nanflow": 0.0,
            "nanflow:type": "Count",
            "origin": 1.2625632e18,
        },
        "type": "SparselyBin",
        "version": "1.0",
    }
    filler = SparkHistogrammar(features=["dt"])
    current_hists = filler.get_histograms(sdf)
    assert current_hists["dt"].toJson() == expected


@pytest.mark.spark
@pytest.mark.skipif(not spark_found, reason="spark not found")
@pytest.mark.filterwarnings(
    "ignore:createDataFrame attempted Arrow optimization because"
)
def test_null_histograms(spark_co):
    spark = spark_co

    data = [(None, None, None, None), (1, None, None, 2), (None, True, "Jones", None), (3, True, "USA", 4),
            (4, False, "FL", 5)]
    columns = ["transaction", "isActive", "eyeColor", "t2"]
    sdf = spark.createDataFrame(data=data, schema=columns)

    hists = make_histograms(sdf, bin_specs={'transaction': {'num': 40, 'low': 0, 'high': 10}})

    assert 'transaction' in hists
    assert 'isActive' in hists
    assert 'eyeColor' in hists
    assert 't2' in hists

    h = hists['transaction']
    assert h.nanflow.entries == 2
    h = hists['t2']
    assert h.nanflow.entries == 2

    h = hists['isActive']
    assert 'NaN' in h.bins
    assert h.bins['NaN'].entries == 2

    h = hists['eyeColor']
    assert 'NaN' in h.bins
    assert h.bins['NaN'].entries == 2
