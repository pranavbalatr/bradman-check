import pytest

from bradman import data, verdicts
from bradman.models import MODEL_DIR


@pytest.fixture(scope="session")
def innings():
    if not data.INNINGS_PATH.exists():
        pytest.skip("run `uv run python -m bradman fetch` first")
    return data.load()


@pytest.fixture(scope="session")
def bradman(innings):
    return data.player_innings(innings, "DG Bradman")


@pytest.fixture(scope="session")
def trained():
    if not (MODEL_DIR / "winprobnet.pt").exists() or not verdicts.PLAYERS_PATH.exists():
        pytest.skip("run `uv run python -m bradman train` first")
    from bradman import models

    em, net = models.load_models()
    table, cal = verdicts.load_tables()
    return em, net, table, cal
