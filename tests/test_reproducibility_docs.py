from pathlib import Path


def test_reproducibility_guide_has_fresh_clone_path():
    text = Path("docs/REPRODUCIBILITY.md").read_text()
    assert "python -m pip install -e ." in text
    assert "pytest" in text
    assert "git status" in text


def test_reproducibility_guide_preserves_dataset_boundaries():
    text = Path("docs/REPRODUCIBILITY.md").read_text()
    assert "production contextual-bandit" in text
    assert "MNAR self-selected ratings" in text
    assert "Do not commit or redistribute these raw Yahoo R3 files" in text
    assert "sampling_data.txt` is not the full observational benchmark" in text


def test_reproducibility_guide_preserves_scientific_boundaries():
    text = Path("docs/REPRODUCIBILITY.md").read_text()
    assert "empirical reference" in text
    assert "analytically exact propensities" in text
    assert "ESS is described as a concentration diagnostic" in text
    assert "not observational-only identification" in text


def test_release_checklist_protects_final_claims():
    text = Path("docs/RELEASE_CHECKLIST.md").read_text()
    assert "Experimental scope remains frozen at v1.3.7" in text
    assert "OBD uses supplied propensity wording" in text
    assert "10% Yahoo calibration operating point is benchmark-specific" in text
    assert "Raw Yahoo R3 data are not committed or redistributed" in text


def test_readme_links_release_documents():
    text = Path("README.md").read_text()
    assert "docs/REPRODUCIBILITY.md" in text
    assert "docs/RELEASE_CHECKLIST.md" in text
