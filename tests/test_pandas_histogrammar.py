#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pytest

from histogrammar.dfinterface.pandas_histogrammar import PandasHistogrammar
from histogrammar.dfinterface.make_histograms import (
    get_bin_specs,
    get_time_axes,
    make_histograms,
)


def test_get_histograms():

    pandas_filler = PandasHistogrammar(
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
            "longitude": {"binWidth": 5, "origin": 0},
            "latitude": {"binWidth": 5, "origin": 0},
        },
    )
    current_hists = pandas_filler.get_histograms(pytest.test_df)

    assert current_hists["age"].toJson() == pytest.age
    assert current_hists["company"].toJson() == pytest.company
    assert current_hists["date"].toJson() == pytest.date
    assert current_hists["eyeColor"].toJson() == pytest.eyesColor
    assert current_hists["gender"].toJson() == pytest.gender
    assert current_hists["isActive"].toJson() == pytest.isActive
    assert current_hists["isActive:age"].toJson() == pytest.isActive_age
    assert current_hists["latitude"].toJson() == pytest.latitude
    assert current_hists["longitude"].toJson() == pytest.longitude
    assert current_hists["latitude:longitude"].toJson() == pytest.latitude_longitude


def test_make_histograms():

    features = [
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
    ]
    bin_specs = {
        "transaction": {"num": 100, "low": -2000, "high": 2000},
        "longitude": {"binWidth": 5, "origin": 0},
        "latitude": {"binWidth": 5, "origin": 0},
    }

    current_hists = make_histograms(
        pytest.test_df, features=features, binning="unit", bin_specs=bin_specs
    )

    assert current_hists["age"].toJson() == pytest.age
    assert current_hists["company"].toJson() == pytest.company
    assert current_hists["date"].toJson() == pytest.date
    assert current_hists["eyeColor"].toJson() == pytest.eyesColor
    assert current_hists["gender"].toJson() == pytest.gender
    assert current_hists["isActive"].toJson() == pytest.isActive
    assert current_hists["isActive:age"].toJson() == pytest.isActive_age
    assert current_hists["latitude"].toJson() == pytest.latitude
    assert current_hists["longitude"].toJson() == pytest.longitude
    assert current_hists["latitude:longitude"].toJson() == pytest.latitude_longitude
    assert current_hists["transaction"].toJson() == pytest.transaction


def test_make_histograms_no_time_axis():

    hists, features, bin_specs, time_axis, var_dtype = make_histograms(
        pytest.test_df, time_axis="", ret_specs=True,
    )

    assert len(hists) == 21
    assert len(features) == 21
    assert len(bin_specs) == 6
    assert len(var_dtype) == 21
    assert time_axis == ""
    assert "date" in hists
    h = hists["date"]
    assert h.binWidth == pytest.approx(751582381944448.0)
    for cols in features:
        cols = cols.split(":")
        assert len(cols) == 1
    for f, bs in bin_specs.items():
        assert isinstance(bs, dict)
    assert "age" in bin_specs
    dateage = bin_specs["age"]
    assert dateage["binWidth"] == 2.0
    assert dateage["origin"] == 9.5


def test_make_histograms_with_time_axis():

    hists, features, bin_specs, time_axis, var_dtype = make_histograms(
        pytest.test_df, time_axis=True, ret_specs=True, time_width=None, time_offset=None
    )

    assert len(hists) == 20
    assert len(features) == 20
    assert len(bin_specs) == 20
    assert len(var_dtype) == 21
    assert time_axis == "date"
    assert "date:age" in hists
    h = hists["date:age"]
    assert h.binWidth == pytest.approx(751582381944448.0)
    for cols in features:
        cols = cols.split(":")
        assert len(cols) == 2 and cols[0] == "date"
    for f, bs in bin_specs.items():
        assert len(bs) == 2
    assert "date:age" in bin_specs
    dateage = bin_specs["date:age"]
    assert dateage[0]["binWidth"] == pytest.approx(751582381944448.0)
    assert dateage[1]["binWidth"] == 2.0
    assert dateage[1]["origin"] == 9.5

    # test get_bin_specs 1
    bin_specs = get_bin_specs(hists)
    assert "date:age" in bin_specs
    dateage = bin_specs["date:age"]
    assert dateage[0]["binWidth"] == pytest.approx(751582381944448.0)
    assert dateage[1]["binWidth"] == 2.0
    assert dateage[1]["origin"] == 9.5

    # test get_bin_specs 2
    bin_specs = get_bin_specs(hists, skip_first_axis=True)
    assert "age" in bin_specs
    age = bin_specs["age"]
    assert age["binWidth"] == 2.0
    assert age["origin"] == 9.5

    # test get_bin_specs 3
    bin_specs = get_bin_specs(hists["date:age"])
    assert bin_specs[0]["binWidth"] == pytest.approx(751582381944448.0)
    assert bin_specs[1]["binWidth"] == 2.0
    assert bin_specs[1]["origin"] == 9.5

    # test get_bin_specs 4
    bin_specs = get_bin_specs(hists["date:age"], skip_first_axis=True)
    assert bin_specs["binWidth"] == 2.0
    assert bin_specs["origin"] == 9.5


def test_make_histograms_unit_binning():

    hists, features, bin_specs, time_axis, var_dtype = make_histograms(
        pytest.test_df, binning="unit", time_axis="", ret_specs=True
    )

    assert len(hists) == 21
    assert len(features) == 21
    assert len(bin_specs) == 0
    assert len(var_dtype) == 21
    assert time_axis == ""
    assert "date" in hists
    h = hists["date"]
    assert h.binWidth == 2592000000000000
    for cols in features:
        cols = cols.split(":")
        assert len(cols) == 1
    for f, bs in bin_specs.items():
        assert isinstance(bs, dict)
    assert "age" in hists
    h = hists["age"]
    assert h.binWidth == 1.0
    assert h.origin == 0.0


def test_get_histograms_module():

    pandas_filler = PandasHistogrammar(
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
            "longitude": {"binWidth": 5, "origin": 0},
            "latitude": {"binWidth": 5, "origin": 0},
        },
        read_key="input",
        store_key="output",
    )

    datastore = pandas_filler.transform(datastore={"input": pytest.test_df})

    assert "output" in datastore
    current_hists = datastore["output"]
    assert current_hists["age"].toJson() == pytest.age
    assert current_hists["company"].toJson() == pytest.company
    assert current_hists["date"].toJson() == pytest.date
    assert current_hists["eyeColor"].toJson() == pytest.eyesColor
    assert current_hists["gender"].toJson() == pytest.gender
    assert current_hists["isActive"].toJson() == pytest.isActive
    assert current_hists["isActive:age"].toJson() == pytest.isActive_age
    assert current_hists["latitude"].toJson() == pytest.latitude
    assert current_hists["longitude"].toJson() == pytest.longitude
    assert current_hists["latitude:longitude"].toJson() == pytest.latitude_longitude


def test_get_time_axes():
    time_axes = get_time_axes(pytest.test_df)
    np.testing.assert_array_equal(time_axes, ["date"])


def test_null_histograms():
    d = {'transaction': {0: np.nan, 1: 1.0, 2: np.nan, 3: 3.0, 4: 4.0},
         'isActive': {0: None, 1: None, 2: True, 3: True, 4: False},
         'eyeColor': {0: None, 1: None, 2: 'Jones', 3: 'USA', 4: 'FL'},
         't2': {0: np.nan, 1: 2.0, 2: np.nan, 3: 4.0, 4: 5.0},
         'foo': {0: np.nan, 1: np.nan, 2: np.nan, 3: True, 4: False},
         'bar': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
         'bla': {0: 1, 1: 2, 2: 3, 3: 4, 4: np.nan},
         'mixed': {0: 'a', 1: 'b', 2: 'c', 3: np.nan, 4: 1}}
    df = pd.DataFrame(d)
    df['bar'] = df['bar'].astype('category')

    hists = make_histograms(df, bin_specs={'transaction': {'num': 40, 'low': 0, 'high': 10}})

    assert 'transaction' in hists
    assert 'isActive' in hists
    assert 'eyeColor' in hists
    assert 't2' in hists
    assert 'foo' in hists
    assert 'bar' in hists
    assert 'bla' in hists
    assert 'bla' in hists
    assert 'mixed' in hists

    h = hists['transaction']
    assert h.nanflow.entries == 2

    h = hists['t2']
    assert h.nanflow.entries == 2

    h = hists['isActive']
    assert 'NaN' in h.bins
    assert h.bins['NaN'].entries == 2

    h = hists['eyeColor']
    assert 'None' in h.bins
    assert h.bins['None'].entries == 2

    h = hists['foo']
    assert 'NaN' in h.bins
    assert h.bins['NaN'].entries == 3

    h = hists['bar']
    assert 'NaN' not in h.bins

    h = hists['bla']
    assert h.nanflow.entries == 1

    h = hists['mixed']
    assert 'nan' in h.bins
    assert h.bins['nan'].entries == 1
