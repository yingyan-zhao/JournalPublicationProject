import importlib.util
import json
from pathlib import Path
import re
import sys

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/02_ExploreConcentrationAuthor.py"
)
SPEC = importlib.util.spec_from_file_location(
    "explore_author_concentration_graph5",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_graph5_interactive_html_contains_gap_data_and_hover_controls(tmp_path):
    data = pd.DataFrame(
        [
            {
                "previous_publication_year": 1980,
                "average_gap_years": 5.5,
                "number_of_authors": 97,
            },
            {
                "previous_publication_year": 2000,
                "average_gap_years": 5.4,
                "number_of_authors": 120,
            },
            {
                "previous_publication_year": 2020,
                "average_gap_years": 2.5,
                "number_of_authors": 124,
            },
        ]
    )
    output_path = tmp_path / "graph5.html"

    MODULE.write_interactive_first_to_second_gap_by_first_year(
        data,
        output_path,
        analysis_label="All fields",
        from_year=1980,
        to_year=2020,
    )

    document = output_path.read_text(encoding="utf-8")
    assert (
        "The observed gap to a second top-five publication has narrowed"
        in document
    )
    assert "Years between first and second publication" in document
    assert "const gapMin = 2;" in document
    assert "const gapMax = 7;" in document
    assert 'class: "gap-line"' in document
    assert 'class: "hover-marker"' in document
    assert 'class: "hit-line"' in document
    assert "pointermove" in document
    assert "number_of_authors" not in document
    assert "toLocaleString()} authors" in document
    assert "Recent cohorts may have incomplete follow-up" in document
    assert "legend" not in document.lower()
    assert "__SERIES_DATA__" not in document

    match = re.search(r"const data = (\[.*?\]);", document)
    assert match
    series = json.loads(match.group(1))
    assert series == [[1980, 5.5, 97], [2000, 5.4, 120], [2020, 2.5, 124]]
