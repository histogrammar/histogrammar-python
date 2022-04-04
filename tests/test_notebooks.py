import pytest
from pytest_notebook.nb_regression import NBRegressionFixture

from histogrammar.resources import notebook


@pytest.fixture(scope="module")
def nb_tester():
    """Test notebooks using pytest-notebook"""
    nb_regression = NBRegressionFixture(
        diff_ignore=(
            "/metadata/language_info",
            "/cells/*/execution_count",
            "/cells/*/outputs/*",
        ),
        exec_timeout=1800,
    )
    return nb_regression


def test_notebook_basic(nb_tester):
    nb_tester.check(notebook("histogrammar_tutorial_basic.ipynb"))


def test_notebook_advanced(nb_tester):
    nb_tester.check(notebook("histogrammar_tutorial_advanced.ipynb"))


def test_notebook_exercises(nb_tester):
    nb_tester.check(notebook("histogrammar_tutorial_exercises.ipynb"))
