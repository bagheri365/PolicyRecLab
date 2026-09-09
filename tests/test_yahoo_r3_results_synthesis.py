from pathlib import Path


def test_yahoo_results_synthesis_preserves_identification_boundary():
    text = Path("docs/v1_3_7_yahoo_r3_results_synthesis.md").read_text()
    assert "not a production contextual-bandit log" in text
    assert "observational Yahoo R3 ratings alone do not nonparametrically identify" in text
    assert "10% randomized calibration is a defensible empirical operating point" in text
    assert "not a universal threshold" in text


def test_readme_links_yahoo_results_synthesis():
    text = Path("README.md").read_text()
    assert "Yahoo R3 MNAR calibration results" in text
    assert "docs/v1_3_7_yahoo_r3_results_synthesis.md" in text
    assert "structurally tied to the calibration subset" in text
