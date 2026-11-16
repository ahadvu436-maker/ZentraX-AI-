import ast
import pytest

from fixtures.project_scan import iter_python_files


pytestmark = pytest.mark.security

DANGEROUS_CALLS = {"eval", "exec"}
WEAK_HASHES = {"md5", "sha1"}
SECRET_MARKERS = ("api_key", "secret_key", "password", "private_key", "access_token")


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def test_no_dynamic_eval_or_exec_in_application_source(project_root):
    findings = []
    for item in iter_python_files(project_root):
        if item.relative_path.startswith("19_tests/"):
            continue
        try:
            tree = ast.parse(item.source, filename=item.relative_path)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node.func) in DANGEROUS_CALLS:
                findings.append(f"{item.relative_path}:{getattr(node, 'lineno', '?')} {_call_name(node.func)}()")
    assert not findings, "Review dynamic code execution calls:\n" + "\n".join(findings)


def test_no_weak_hash_api_calls_for_security(project_root):
    findings = []
    for item in iter_python_files(project_root):
        if item.relative_path.startswith("19_tests/"):
            continue
        try:
            tree = ast.parse(item.source, filename=item.relative_path)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node.func).lower() in WEAK_HASHES:
                findings.append(f"{item.relative_path}:{getattr(node, 'lineno', '?')} {_call_name(node.func)}()")
    assert not findings, "Review weak hash usage; use modern password/KDF or SHA-256+ as appropriate:\n" + "\n".join(findings)


def test_no_obvious_hardcoded_secret_assignments(project_root, allowed_security_patterns):
    findings = []
    for item in iter_python_files(project_root):
        if item.relative_path.startswith("19_tests/"):
            continue
        if any(pattern in item.relative_path for pattern in allowed_security_patterns):
            continue
        try:
            tree = ast.parse(item.source, filename=item.relative_path)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                continue
            if len(value.value) < 8:
                continue
            for target in targets:
                names = []
                if isinstance(target, ast.Name):
                    names = [target.id.lower()]
                elif isinstance(target, ast.Attribute):
                    names = [target.attr.lower()]
                for name in names:
                    if any(marker in name for marker in SECRET_MARKERS):
                        findings.append(f"{item.relative_path}:{getattr(node, 'lineno', '?')} {name}")
    assert not findings, "Possible hardcoded secrets; move them to a secret manager/environment:\n" + "\n".join(findings)
