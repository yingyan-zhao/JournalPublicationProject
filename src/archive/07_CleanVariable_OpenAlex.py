from pathlib import Path
import json
import re
import unicodedata

import os
import pandas as pd

os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

INPUT_DIR_OpenAlex = Path("data/processed/openalex_by_year")
OUTPUT_DIR_OpenAlex = Path("data/processed/openalex_by_year_cleaned")

OPENALEX_COLUMNS = [
    "id",
    "doi",
    "title",
    "display_name",
    "publication_year",
    "publication_date",
    "type",
    "type_crossref",
    "authors",
    "author_ids",
    "raw_author_names",
    "orcid_ids",
    "author_positions",
    "author_institutions",
    "author_institution_ids",
    "raw_affiliation_strings",
    "institutions",
    "keywords",
    "concepts",
    "topics",
    "primary_topic",
    "cited_by_count",
    "counts_by_year",
    "collection_date",
    "_query_journal",
    "_query_issn",
]


def main() -> None:
    input_files = sorted(INPUT_DIR_OpenAlex.glob("OpenAlex_Works_*.csv"))
    if not input_files:
        raise FileNotFoundError(f"No yearly OpenAlex files found in {INPUT_DIR_OpenAlex}")

    OUTPUT_DIR_OpenAlex.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    for input_csv in input_files:
        openalex = pd.read_csv(input_csv)
        openalex_selected = clean_openalex_data(openalex)

        output_csv = OUTPUT_DIR_OpenAlex / input_csv.name.replace(
            "OpenAlex_Works_", "OpenAlex_Works_Cleaned_"
        )
        openalex_selected.to_csv(output_csv, index=False)

        total_rows += len(openalex_selected)
        print(f"Wrote {len(openalex_selected)} rows to {output_csv}")

    print(f"Wrote {total_rows} cleaned OpenAlex rows across {len(input_files)} yearly files.")


def clean_openalex_data(openalex: pd.DataFrame) -> pd.DataFrame:
    openalex_selected = keep_columns(openalex, OPENALEX_COLUMNS)
    openalex_selected = drop_blank_titles(openalex_selected)

    # clean the title
    openalex_selected["title"] = openalex_selected["title"].replace("", pd.NA)
    openalex_selected["title"] = openalex_selected["title"].fillna(openalex_selected["display_name"])
    openalex_selected = openalex_selected.drop(columns=["display_name"])
    openalex_selected = drop_correction_titles(openalex_selected)

    openalex_selected = openalex_selected.rename(
        columns={"title": "openalex_title"}
    )

    # clean DOI
    openalex_selected["doi"] = openalex_selected["doi"].apply(clean_doi)
    # drop publication year
    openalex_selected = openalex_selected.drop(columns=["publication_date"])
    openalex_selected = openalex_selected.rename(columns={"publication_year":"openalex_publication_year"})

    # clean number of citations
    openalex_selected = openalex_selected.drop(columns=["counts_by_year"])
    openalex_selected = openalex_selected.rename(
        columns={"cited_by_count": "openalex_cited_by_count"}
    )

    # Keep Journal name and Journal issn
    openalex_selected = openalex_selected.rename(
        columns={"_query_journal": "openalex_journalname","_query_issn": "openalex_journalissn"}
    )

    # clean topics/fields
    openalex_selected = add_primary_topic_columns(openalex_selected)
    openalex_selected = add_keyword_columns(openalex_selected)
    openalex_selected = add_concept_columns(openalex_selected)
    openalex_selected = openalex_selected.drop(columns=["primary_topic","keywords","concepts","topics"])

    return openalex_selected



def keep_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing_columns = [column for column in columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    return data[columns].copy()


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

def drop_blank_author_identifiers(data: pd.DataFrame) -> pd.DataFrame:
    author_name = data["author_name_ascii"]
    raw_author_name = data["raw_author_name_ascii"]
    author_name_blank = author_name.isna() | (author_name.astype(str).str.strip() == "")
    raw_author_name_blank = raw_author_name.isna() | (raw_author_name.astype(str).str.strip() == "")
    keep_rows = ~(author_name_blank & raw_author_name_blank)
    return data.loc[keep_rows].copy()


def drop_blank_titles(data: pd.DataFrame) -> pd.DataFrame:
    title_blank = data["title"].isna() | (data["title"].astype(str).str.strip() == "")
    display_name_blank = data["display_name"].isna() | (data["display_name"].astype(str).str.strip() == "")
    keep_rows = ~(title_blank & display_name_blank)
    return data.loc[keep_rows].copy()

def drop_wrong_authors(data: pd.DataFrame) -> pd.DataFrame:
    correction_patterns = [
        "Editor",
        "Suggested by",
    ]

    pattern = "|".join(re.escape(phrase) for phrase in correction_patterns)

    correction_title1 = data["author_name"].fillna("").str.contains(
        pattern,
        case=False,
        regex=True,
    )

    correction_title2 = data["raw_author_name"].fillna("").str.contains(
        pattern,
        case=False,
        regex=True,
    )

    return data.loc[~(correction_title1 | correction_title2)].copy()

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
        "Minutes of the Annual Business Meeting",
        "Front Matter",
        "The Econometric Society Annual Reports Econometrica"
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
        "author_institutions",
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


def clean_author_name(name) -> str:
    if pd.isna(name):
        return ""
    name = str(name).strip()
    return " ".join(name.split())


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


def keep_university_affiliations(affiliation) -> str:
    if pd.isna(affiliation):
        return ""

    academic_pattern = re.compile(r"\buniversity\b", flags=re.IGNORECASE)

    affiliation_parts = [
        part.strip()
        for part in re.split(r"[;,]", str(affiliation))
        if part.strip()
    ]
    university_parts = [
        part
        for part in affiliation_parts
        if academic_pattern.search(part)
    ]
    return "; ".join(university_parts)


def keep_other_academic_affiliations(affiliation) -> str:
    if pd.isna(affiliation):
        return ""

    academic_pattern = re.compile(
        r"\b(college|school|institute|mit|caltech|insead|cemfi|Federal Reserve|National Bureau|World Bank|European Central Bank)\b",
        flags=re.IGNORECASE,
    )

    affiliation_parts = [
        part.strip()
        for part in re.split(r"[;,]", str(affiliation))
        if part.strip()
    ]
    academic_parts = [
        part
        for part in affiliation_parts
        if academic_pattern.search(part)
    ]
    return "; ".join(academic_parts)


def add_primary_topic_columns(data: pd.DataFrame) -> pd.DataFrame:
    primary_topics = data["primary_topic"].apply(parse_json_cell)

    data["openalex_primary_domain"] = primary_topics.apply(
        lambda topic: nested_display_name(topic, "domain")
    )
    data["openalex_primary_field"] = primary_topics.apply(
        lambda topic: nested_display_name(topic, "field")
    )
    data["openalex_primary_subfield"] = primary_topics.apply(
        lambda topic: nested_display_name(topic, "subfield")
    )
    data["openalex_primary_topic"] = primary_topics.apply(display_name)
    return data


def add_keyword_columns(data: pd.DataFrame) -> pd.DataFrame:
    keywords = data["keywords"].apply(parse_json_cell)

    data["openalex_keywords"] = keywords.apply(format_scored_names)
    data["openalex_top3_keywords"] = keywords.apply(
        lambda value: format_scored_names(value, max_items=3)
    )
    return data


def add_concept_columns(data: pd.DataFrame) -> pd.DataFrame:
    concepts = data["concepts"].apply(parse_json_cell)

    data["openalex_concepts"] = concepts.apply(format_scored_names)
    data["openalex_top3_concepts"] = concepts.apply(
        lambda value: format_scored_names(value, max_items=3)
    )
    data["openalex_level0_concepts"] = concepts.apply(format_level0_concepts)
    return data


def parse_json_cell(value):
    if pd.isna(value) or value == "":
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def display_name(value) -> str:
    if isinstance(value, dict):
        return value.get("display_name", "") or ""
    return ""


def nested_display_name(value, key: str) -> str:
    if not isinstance(value, dict):
        return ""
    nested_value = value.get(key)
    return display_name(nested_value)


def format_scored_names(value, max_items: int | None = None) -> str:
    if not isinstance(value, list):
        return ""

    items = value[:max_items] if max_items else value
    formatted_items = []
    for item in items:
        if not isinstance(item, dict):
            continue

        name = display_name(item)
        score = item.get("score")
        if name and score is not None:
            formatted_items.append(f"{name} ({score:.3f})")
        elif name:
            formatted_items.append(name)

    return "; ".join(formatted_items)


def format_level0_concepts(value) -> str:
    if not isinstance(value, list):
        return ""

    level0_concepts = [
        item for item in value
        if isinstance(item, dict) and item.get("level") == 0
    ]
    return format_scored_names(level0_concepts)


def format_topics(value, max_topics: int | None = None) -> str:
    if not isinstance(value, list):
        return ""

    topics = value[:max_topics] if max_topics else value
    formatted_topics = []
    for topic in topics:
        if not isinstance(topic, dict):
            continue

        hierarchy = " > ".join(
            part
            for part in [
                nested_display_name(topic, "domain"),
                nested_display_name(topic, "field"),
                nested_display_name(topic, "subfield"),
                display_name(topic),
            ]
            if part
        )
        score = topic.get("score")
        if hierarchy and score is not None:
            formatted_topics.append(f"{hierarchy} ({score:.3f})")
        elif hierarchy:
            formatted_topics.append(hierarchy)

    return "; ".join(formatted_topics)


if __name__ == "__main__":
    main()
