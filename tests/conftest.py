import pytest
from pathlib import Path
import pandas as pd
import numpy as np

# MARKS


def pytest_configure(config):
    config.addinivalue_line("markers", "wip: WORK IN PROGRESS")


# TEST_ENVIRONMENT

CWD = Path(__file__).resolve()
FIXTURES_ROOT = CWD.parent / "fixtures"


@pytest.fixture
def test_data(request, tmp_path):
    if not hasattr(request, "param"):
        yield pd.read_csv(FIXTURES_ROOT / "griddata_default_colnames.csv")
    else:
        yield pd.read_csv(FIXTURES_ROOT / request.param)


@pytest.fixture
def random_npdata() -> np.ndarray:
    # uniques pair of coordinates
    n_uniques = 10
    x = np.random.rand(n_uniques) * 400000  # fake coordinates offsets
    y = np.random.rand(n_uniques) * 4000000
    z = np.array([])
    v = np.array([])
    num_pairs = []
    for i in range(n_uniques):
        # random number of measure per unique pair
        num_pairs.append(np.random.randint(1, 10))
        z = np.append(z, np.random.rand(num_pairs[i]) * 25)
        v = np.append(v, np.random.rand(num_pairs[i]) * 10000)

    x = x.repeat(num_pairs)
    y = y.repeat(num_pairs)

    data = np.column_stack((x, y, z, v))
    yield data
