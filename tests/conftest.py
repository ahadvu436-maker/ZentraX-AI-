import pytest

from configuration.settings import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(environment="testing", pipeline={"dataset_root": tmp_path / "versions"})
