from pathlib import Path


def test_readme_surfaces_research_question_and_headline_results():
    text = Path("README.md").read_text()
    assert "Research question" in text
    assert "## What this project shows" in text
    assert "## The reliability framework" in text
    assert "## Real-data evidence" in text


def test_readme_preserves_benchmark_boundaries():
    text = Path("README.md").read_text()
    assert "Contextual-bandit benchmark: Open Bandit Dataset" in text
    assert "MNAR exposure benchmark: Coat" in text
    assert "MNAR rating-exposure benchmark" in text
    assert "not a contextual-bandit log" in text


def test_readme_has_fast_reproduction_and_reading_paths():
    text = Path("README.md").read_text()
    assert "## Reproduce the project" in text
    assert "python -m pip install -e ." in text
    assert "## Read next" in text
    assert "docs/FINAL_REPORT.md" in text
    assert "docs/REPRODUCIBILITY.md" in text
