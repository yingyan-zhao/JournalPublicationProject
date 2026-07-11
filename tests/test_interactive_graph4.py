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
    "explore_author_concentration",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_graph4_interactive_html_contains_cohort_data_and_hover_controls(tmp_path):
    rows = []
    for cohort_index, cohort in enumerate(
        ["1981-1990", "1991-2000", "2001-2010", "2011-2020"]
    ):
        rows.extend(
            [
                {
                    "entry_cohort": cohort,
                    "to_publication_number": 2,
                    "average_gap_years": 5.9 - cohort_index * 0.8,
                    "number_of_authors": 100 + cohort_index,
                },
                {
                    "entry_cohort": cohort,
                    "to_publication_number": 3,
                    "average_gap_years": 4.5 - cohort_index * 0.6,
                    "number_of_authors": 80 + cohort_index,
                },
            ]
        )
    output_path = tmp_path / "graph4.html"

    MODULE.write_interactive_publication_gaps_by_cohort(
        pd.DataFrame(rows),
        output_path,
        analysis_label="All fields",
        minimum_authors=30,
    )

    document = output_path.read_text(encoding="utf-8")
    assert "Observed publication gaps are shorter for newer cohorts" in document
    assert 'class: "hit-line"' in document
    assert "addPointMarker" in document
    assert 'add("circle"' in document
    assert 'add("rect"' in document
    assert 'add("polygon"' in document
    assert (
        '.point-marker[data-series="1981-1990"] '
        "{ fill: var(--cohort-1980s); }"
    ) in document
    assert (
        '.point-marker[data-series="2011-2020"] '
        "{ fill: var(--cohort-2010s); }"
    ) in document
    assert "pointermove" in document
    assert "tooltip.textContent" in document
    assert "authors)`" in document
    assert "__SERIES_DATA__" not in document

    match = re.search(r"const series = (\{.*?\});", document)
    assert match
    series = json.loads(match.group(1))
    assert list(series) == [
        "1981-1990",
        "1991-2000",
        "2001-2010",
        "2011-2020",
    ]
    assert series["2011-2020"][0] == [2, 3.5, 103]
