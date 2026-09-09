from pathlib import Path


def test_documentation_index_records_experimental_freeze():
    text = Path("docs/DOCUMENTATION_INDEX.md").read_text()
    assert "experimentally frozen at v1.3.7" in text
    assert "v1.3.1" in text
    assert "Rejected as effective debiaser" in text
    assert "Coat/Yahoo results must not be described" in text


def test_documentation_audit_preserves_release_boundaries():
    text = Path("docs/DOCUMENTATION_AUDIT.md").read_text()
    assert "Experimental scope is frozen at v1.3.7" in text
    assert "ESS" in text
    assert "observational-only identification" in text
    assert "Raw external datasets should not be bundled" in text


def test_readme_points_to_documentation_audit():
    text = Path("README.md").read_text()
    assert "Documentation and release status" in text
    assert "docs/DOCUMENTATION_INDEX.md" in text
    assert "docs/DOCUMENTATION_AUDIT.md" in text
