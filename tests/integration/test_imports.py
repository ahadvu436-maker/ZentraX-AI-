import pytest


pytestmark = pytest.mark.integration


def test_configured_module_imports(import_targets, import_target_loader):
    if not import_targets:
        pytest.skip("Set ZENTRAX_IMPORT_TARGETS to comma-separated import paths")
    failures = []
    for target in import_targets:
        try:
            import_target_loader(target)
        except Exception as exc:
            failures.append(f"{target}: {type(exc).__name__}: {exc}")
    assert not failures, "Module import failures:\n" + "\n".join(failures)


def test_import_targets_are_unique(import_targets):
    assert len(import_targets) == len(set(import_targets)), "Duplicate ZENTRAX_IMPORT_TARGETS entries"
