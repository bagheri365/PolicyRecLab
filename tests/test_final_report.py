from pathlib import Path


def test_final_report_preserves_core_research_rule():
    text = Path("docs/FINAL_REPORT.md").read_text()
    assert "Never report an offline policy value" in text
    assert "Identification comes first" in text
    assert "not nonparametrically identified" in text


def test_final_report_keeps_real_data_categories_distinct():
    text = Path("docs/FINAL_REPORT.md").read_text()
    assert "OBD is a production contextual-bandit benchmark" in text
    assert "Coat and Yahoo R3 are MNAR rating-exposure benchmarks" in text


def test_final_report_preserves_negative_and_caveated_results():
    text = Path("docs/FINAL_REPORT.md").read_text()
    assert "rejected as an effective debiaser" in text
    assert "does **not** demonstrate that the observational data alone identify" in text
    assert "10% randomized calibration is a defensible empirical operating point" in text
    assert "not a general law" in text


def test_readme_links_final_report_and_freezes_scope():
    text = Path("README.md").read_text()
    assert "docs/FINAL_REPORT.md" in text
    assert "experimental program is frozen at **v1.3.7**" in text
    assert "Close point-estimate agreement can coexist with weak overlap" in text
