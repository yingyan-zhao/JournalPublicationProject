from pathlib import Path
import json
import re
import unicodedata

import os
import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

INPUT_DIR_Crossref = Path("data/processed/crossref_by_year")
OUTPUT_DIR_Crossref = Path("data/processed/crossref_by_year_cleaned")

CROSSREF_COLUMNS = [
    "DOI",
    "published_year",
    "published",
    "published-print",
    "published-online",
    "issued",
    "container-title",
    "short-container-title",
    "publisher",
    "volume",
    "issue",
    "page",
    "type",
    "abstract",
    "reference-count",
    "references-count",
    "is-referenced-by-count",
    "ISSN",
    "URL",
    "authors",
    "author_given",
    "author_family",
    "author_sequence",
    "author_affiliations",
    "author_orcids",
    "_query_journal",
    "_query_issn",
    "collection_date",
    "title",
]


def main() -> None:
    input_files = sorted(INPUT_DIR_Crossref.glob("Crossref_Works_*.csv"))
    if not input_files:
        raise FileNotFoundError(f"No yearly Crossref files found in {INPUT_DIR_Crossref}")

    OUTPUT_DIR_Crossref.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    for input_csv in input_files:
        crossref = pd.read_csv(input_csv)
        crossref_selected = clean_crossref_data(crossref)

        output_csv = OUTPUT_DIR_Crossref / input_csv.name.replace(
            "Crossref_Works_", "Crossref_Works_Cleaned_"
        )
        crossref_selected.to_csv(output_csv, index=False)

        total_rows += len(crossref_selected)
        print(f"Wrote {len(crossref_selected)} rows to {output_csv}")

    print(f"Wrote {total_rows} cleaned Crossref rows across {len(input_files)} yearly files.")


def clean_crossref_data(crossref: pd.DataFrame) -> pd.DataFrame:
    crossref_selected = keep_columns(crossref, CROSSREF_COLUMNS)

    # clean paper fields
    crossref_selected["doi"] = crossref_selected["DOI"].apply(clean_doi)
    crossref_selected = crossref_selected.drop(columns=["DOI"])

    # clean the title
    crossref_selected["title"] = crossref_selected["title"].apply(first_json_value)
    crossref_selected["title"] = crossref_selected["title"].replace("", pd.NA)
    crossref_selected = drop_blank_titles(crossref_selected)
    crossref_selected = drop_correction_titles(crossref_selected)
    crossref_selected = crossref_selected.rename(
        columns={"title": "crossref_title"}
    )
    # Keep Journal name and Journal issn
    crossref_selected = crossref_selected.rename(
        columns={"_query_journal": "crossref_journalname","_query_issn": "crossref_journalissn"}
    )

    return crossref_selected


def keep_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing_columns = [column for column in columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    return data[columns].copy()


def drop_blank_author_identifiers(data: pd.DataFrame) -> pd.DataFrame:
    author_name = data["author_name"]
    author_given = data["author_given"]
    author_family = data["author_family"]

    author_name_blank = author_name.isna() | (author_name.astype(str).str.strip() == "")
    author_given_blank = author_given.isna() | (author_given.astype(str).str.strip() == "")
    author_family_blank = author_family.isna() | (author_family.astype(str).str.strip() == "")

    keep_rows = ~(author_name_blank & author_given_blank & author_family_blank)
    return data.loc[keep_rows].copy()


def fill_author_name(row: pd.Series) -> str:
    author_name = clean_author_name(row.get("author_name", ""))
    if author_name:
        return author_name

    name_parts = [
        clean_author_name(row.get("author_given", "")),
        clean_author_name(row.get("author_family", "")),
    ]
    return " ".join(part for part in name_parts if part)


def first_json_value(value) -> str:
    parsed_value = parse_json_cell(value)
    if isinstance(parsed_value, list) and parsed_value:
        return clean_text(parsed_value[0])
    if isinstance(parsed_value, list):
        return ""
    return clean_text(value)


def join_json_values(value) -> str:
    parsed_value = parse_json_cell(value)
    if isinstance(parsed_value, list):
        return "; ".join(clean_text(item) for item in parsed_value if not is_blank(item))
    return clean_text(value)


def parse_json_cell(value):
    if pd.isna(value) or value == "":
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def clean_abstract(value) -> str:
    text = clean_text(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return " ".join(text.split())


def clean_doi(doi) -> str:
    if pd.isna(doi):
        return ""
    return (
        str(doi)
        .strip()
        .replace("https://doi.org/", "")
        .replace("http://doi.org/", "")
        .replace("http://dx.doi.org/", "")
    )


def clean_author_name(name) -> str:
    return clean_text(name)


def ascii_author_name(name) -> str:
    if pd.isna(name):
        return ""
    name = str(name)
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    return " ".join(name.split())


def first_name(name) -> str:
    parts = split_name(name)
    if len(parts) <= 1:
        return ""
    return " ".join(parts[:-1])


def last_name(name) -> str:
    parts = split_name(name)
    if not parts:
        return ""
    return parts[-1]


def split_name(name) -> list[str]:
    if pd.isna(name):
        return []
    return str(name).strip().split()


def drop_wrong_authors(data: pd.DataFrame) -> pd.DataFrame:
    correction_patterns = [
        "Editor",
        "Suggested by",
    ]

    pattern = "|".join(re.escape(phrase) for phrase in correction_patterns)

    correction_author = data["author_name"].fillna("").str.contains(
        pattern,
        case=False,
        regex=True,
    )

    return data.loc[~correction_author].copy()


def drop_correction_titles(data: pd.DataFrame) -> pd.DataFrame:
    correction_patterns = [
        "Correction:",
        "A Correction",
        "Correction to",
        "Erratum",
        "Corrigendum",
        "comment",
        "report of",
        "Editors' Introduction",
        "Editor's Introduction",
        "Foreword",
        "reply",
        "Editorial Announcement",
        "Ad Hoc Search Committee",
        "Executive Committee",
        "List of Online Reports",
        "Book Review",
        "Frontmatter of Econometrica",
        "Backmatter of Econometrica ",
        "Ad Hoc Committee",
        "Journal of Economic Perspectives",
        "American Economic Journal",
        "Journal of Economic Literature",
        "Committee on",
        "Index to Volume",
        "Recent Referees",
        "Minutes of the Annual Meeting",
        "Journal of Political Economy",
        "John Bates Clark Award",
        "Appendix",
        "American Economic Association",
        "American Economic Review",
        "Job Openings for Economists",
        "OUP accepted manuscript",
        "Minutes of the Annual Business Meeting"
    ]

    pattern = "|".join(re.escape(phrase) for phrase in correction_patterns)

    correction_title = data["title"].fillna("").str.contains(
        pattern,
        case=False,
        regex=True,
    )

    return data.loc[~correction_title].copy()


def select_affiliation(row: pd.Series) -> str:
    for column in [
        "raw_affiliation_university1",
        "raw_affiliation_university2",
    ]:
        value = row.get(column, "")
        if not is_blank(value):
            return str(value).strip()
    return ""


def is_blank(value) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def keep_university_affiliations(affiliation) -> str:
    return keep_affiliation_phrases(affiliation, r"\buniversity\b")


def keep_other_academic_affiliations(affiliation) -> str:
    return keep_affiliation_phrases(
        affiliation,
        r"\b(college|school|institute|mit|caltech|insead|cemfi|Federal Reserve|National Bureau|World Bank|European Central Bank)\b",
    )


def keep_affiliation_phrases(affiliation, pattern: str) -> str:
    if pd.isna(affiliation):
        return ""

    affiliation_pattern = re.compile(pattern, flags=re.IGNORECASE)
    affiliation_parts = [
        part.strip()
        for part in re.split(r"[;,]", str(affiliation))
        if part.strip()
    ]
    kept_parts = [
        part
        for part in affiliation_parts
        if affiliation_pattern.search(part)
    ]
    return "; ".join(kept_parts)

def drop_blank_titles(data: pd.DataFrame) -> pd.DataFrame:
    title_blank = data["title"].isna() | (data["title"].astype(str).str.strip() == "")
    keep_rows = ~(title_blank)
    return data.loc[keep_rows].copy()


if __name__ == "__main__":
    main()
