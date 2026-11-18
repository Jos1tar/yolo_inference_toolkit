from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
FIXTURES = ROOT / "tests" / "fixtures"


def pytest_configure(config):
    config.addinivalue_line("markers", "models: needs exported models in models/ (run `make models`)")


@pytest.fixture(scope="session")
def models_dir() -> Path:
    return MODELS


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES
