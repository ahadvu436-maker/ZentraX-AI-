import pytest


pytestmark = pytest.mark.regression


def test_pytest_markers_are_registered():
    """Guards against accidental removal of the suite's marker configuration."""
    from pathlib import Path

    config = Path(__file__).resolve().parents[1] / "pytest.ini"
    assert config.exists(), "19_tests/pytest.ini is missing"


def test_project_root_fixture_is_available(project_root):
    assert project_root.exists()
    assert project_root.is_dir()
