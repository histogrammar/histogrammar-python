from decimal import Decimal
from json import load
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from histogrammar import resources


def get_comparer_data():
    test_comparer_df = {}
    df = pd.DataFrame(
        data={
            "mae": [0.1, 0.11, 0.12, 0.2, 0.09],
            "mse": [0.1, 0.1, 0.1, 0.1, 0.1],
            "date": [2000, 2001, 2002, 2003, 2004],
        }
    )
    df.set_index("date", inplace=True)
    test_comparer_df["the_feature"] = df

    df = pd.DataFrame(
        data={
            "mae": [0.1, 0.11, 0.12, 0.2, 0.09],
            "date": [2000, 2001, 2002, 2003, 2004],
        }
    )
    df.set_index("date", inplace=True)
    test_comparer_df["dummy_feature"] = df

    return test_comparer_df


def get_ref_comparer_data():
    ref_data = pd.DataFrame()
    # we do not add "mse_std" on purpose to have some noise in the data
    ref_data["metric"] = ["mae_mean", "mae_std", "mae_pull", "mse_mean"]
    ref_data["value"] = [0.124, 0.0376, 0.0376, 0.09]
    ref_data["feature"] = "the_feature"
    ref_data["date"] = np.arange(ref_data.shape[0]) + 2010

    return ref_data


def pytest_configure():
    # attach common test data
    pytest.test_comparer_df = get_comparer_data()
    pytest.test_ref_comparer_df = get_ref_comparer_data()

    parent_path = Path(__file__).parent
    TEMPLATE_PATH = parent_path / "resources"
    CSV_FILE = "test.csv.gz"

    with (TEMPLATE_PATH / "age.json").open() as f:
        pytest.age = load(f)

    with (TEMPLATE_PATH / "company.json").open() as f:
        pytest.company = load(f)

    with (TEMPLATE_PATH / "date.json").open() as f:
        pytest.date = load(f)

    with (TEMPLATE_PATH / "eyesColor.json").open() as f:
        pytest.eyesColor = load(f)

    with (TEMPLATE_PATH / "gender.json").open() as f:
        pytest.gender = load(f)

    with (TEMPLATE_PATH / "isActive.json").open() as f:
        pytest.isActive = load(f)

    with (TEMPLATE_PATH / "isActive_age.json").open() as f:
        pytest.isActive_age = load(f)

    with (TEMPLATE_PATH / "latitude.json").open() as f:
        pytest.latitude = load(f)

    with (TEMPLATE_PATH / "longitude.json").open() as f:
        pytest.longitude = load(f)

    with (TEMPLATE_PATH / "latitude_longitude.json").open() as f:
        pytest.latitude_longitude = load(f)

    with (TEMPLATE_PATH / "transaction.json").open() as f:
        pytest.transaction = load(f)

    df = pd.read_csv(resources.data(CSV_FILE))
    df["date"] = pd.to_datetime(df["date"])

    # Decimal type
    df["amount"] = df["balance"].str.replace("$", "", regex=False).str.replace(",", "", regex=False).apply(Decimal)

    pytest.test_df = df
