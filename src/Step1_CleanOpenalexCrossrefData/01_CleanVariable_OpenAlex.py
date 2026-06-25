from pathlib import Path
import html
import json
import re
import unicodedata

import os
import pandas as pd

os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

INPUT_CSV_OpenAlex = Path("data/raw_csv/OpenAlex_Works.csv")
OUTPUT_CSV_OpenAlex = Path("data/processed/OpenAlex_Works_Cleaned.csv")
OPENALEX_COLUMNS_TO_DROP_AFTER_CLEANING = [
    "openalex_doi",
    "openalex_tag",
]
OPENALEX_IDS_TO_DROP = {
    "https://openalex.org/W1520927568",
    "https://openalex.org/W115369480",
    "https://openalex.org/W309465800",
    "https://openalex.org/W4249665525",
    "https://openalex.org/W4245342635",
    "https://openalex.org/W4246125188",
    "https://openalex.org/W4253097040",
    "https://openalex.org/W4239194262"
}
OPENALEX_UNION_COLUMNS_FOR_DUPLICATES = [
    "openalex_primary_domain",
    "openalex_primary_field",
    "openalex_primary_subfield",
    "openalex_primary_topic",
    "openalex_keywords",
    "openalex_top3_keywords",
    "openalex_concepts",
    "openalex_top3_concepts",
    "openalex_level0_concepts",
]

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
    "abstract_inverted_index",
    "cited_by_count",
    "counts_by_year",
    "collection_date",
    "_query_journal",
    "_query_issn",
]


def main() -> None:
    if not INPUT_CSV_OpenAlex.exists():
        raise FileNotFoundError(f"{INPUT_CSV_OpenAlex} does not exist.")

    openalex = pd.read_csv(INPUT_CSV_OpenAlex)
    openalex_selected = clean_openalex_data(openalex)

    OUTPUT_CSV_OpenAlex.parent.mkdir(parents=True, exist_ok=True)
    openalex_selected.to_csv(OUTPUT_CSV_OpenAlex, index=False)

    print(f"Read {len(openalex)} OpenAlex rows from {INPUT_CSV_OpenAlex}.")
    print(f"Wrote {len(openalex_selected)} cleaned OpenAlex rows to {OUTPUT_CSV_OpenAlex}.")
    print(f"Rows with abstract: {count_nonblank(openalex_selected, 'openalex_abstract')}.")
    print(f"Rows with JEL codes in abstract: {count_nonblank(openalex_selected, 'openalex_jel_codes')}.")
    print(f"Rows with duplicated DOI: {count_duplicate_rows(openalex_selected, 'openalex_doi_1')}.")
    print(f"Rows with duplicated DOI: {count_duplicate_rows(openalex_selected, 'openalex_doi_2')}.")
    print(f"Rows with duplicated DOI: {count_duplicate_rows(openalex_selected, 'openalex_doi_3')}.")
    print(f"Rows with duplicated title: {count_duplicate_rows(openalex_selected, 'openalex_title')}.")


def clean_openalex_data(openalex: pd.DataFrame) -> pd.DataFrame:
    openalex_selected = keep_columns(openalex, OPENALEX_COLUMNS)
    openalex_selected = drop_openalex_ids(openalex_selected, OPENALEX_IDS_TO_DROP)
    openalex_selected = drop_blank_titles(openalex_selected)

    # clean the title
    openalex_selected["title"] = openalex_selected["title"].replace("", pd.NA)
    openalex_selected["title"] = openalex_selected["title"].fillna(openalex_selected["display_name"])
    openalex_selected = openalex_selected.drop(columns=["display_name"])
    openalex_selected = drop_correction_titles(openalex_selected)
    openalex_selected["title"] = openalex_selected["title"].apply(clean_title)
    openalex_selected = rename_specific_openalex_titles(openalex_selected)
    openalex_selected = drop_blank_cleaned_titles(openalex_selected)

    openalex_selected["tag"] = duplicate_title_tag(openalex_selected, "title")
    openalex_selected = drop_duplicate_titles_with_blank_authors(openalex_selected)
    openalex_selected = use_longest_institutions_for_duplicate_titles(openalex_selected)

    openalex_selected = openalex_selected.rename(
        columns={"title": "openalex_title"}
    )

    # clean DOI
    openalex_selected["doi"] = openalex_selected["doi"].apply(clean_doi)
    # drop publication year
    openalex_selected = openalex_selected.drop(columns=["publication_date"])
    openalex_selected = openalex_selected.rename(columns={"publication_year":"openalex_publication_year"})

    # clean abstract
    openalex_selected["abstract"] = openalex_selected["abstract_inverted_index"].apply(
        abstract_from_inverted_index
    )
    openalex_selected["jel_codes"] = openalex_selected["abstract"].apply(extract_jel_codes)
    openalex_selected["abstract"] = openalex_selected["abstract"].apply(clean_abstract)
    openalex_selected = use_longest_abstract_for_duplicate_titles(openalex_selected)
    openalex_selected = use_longest_jel_codes_for_duplicate_titles(openalex_selected)
    openalex_selected = openalex_selected.drop(columns=["abstract_inverted_index"])

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
    openalex_selected = use_union_for_duplicate_titles(
        openalex_selected,
        OPENALEX_UNION_COLUMNS_FOR_DUPLICATES,
    )
    openalex_selected = openalex_selected.drop(columns=["primary_topic","keywords","concepts","topics"])
    openalex_selected = add_doi_versions_for_duplicate_titles(openalex_selected)
    openalex_selected = keep_one_observation_per_duplicated_title(openalex_selected)
    openalex_selected = add_openalex_prefix(openalex_selected)
    openalex_selected = drop_columns(openalex_selected, OPENALEX_COLUMNS_TO_DROP_AFTER_CLEANING)

    return openalex_selected


def rename_specific_openalex_titles(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    if "title" not in cleaned.columns or "publication_year" not in cleaned.columns:
        return cleaned

    title = cleaned["title"].fillna("").astype(str).str.strip()
    publication_year = cleaned["publication_year"].fillna("").astype(str).str.strip()
    target_row = (title == "Human Capital and Growth") & (publication_year == "2015")
    cleaned.loc[target_row, "title"] = "Human Capital and Growth 2015"
    return cleaned



def keep_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing_columns = [column for column in columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    return data[columns].copy()


def count_nonblank(data: pd.DataFrame, column: str) -> int:
    if column not in data.columns:
        return 0
    return int(data[column].fillna("").astype(str).str.strip().ne("").sum())


def count_duplicate_rows(data: pd.DataFrame, column: str) -> int:
    if column not in data.columns:
        return 0
    values = data[column].fillna("").astype(str).str.strip()
    values = values.loc[values != ""]
    return int(values.duplicated(keep=False).sum())


def drop_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    columns_to_drop = [
        column for column in columns
        if column in data.columns
    ]
    return data.drop(columns=columns_to_drop).copy()


def drop_openalex_ids(data: pd.DataFrame, openalex_ids: set[str]) -> pd.DataFrame:
    ids = data["id"].fillna("").astype(str).str.strip()
    return data.loc[~ids.isin(openalex_ids)].copy()


def duplicate_title_tag(data: pd.DataFrame, title_column: str) -> pd.Series:
    titles = data[title_column].fillna("").astype(str).str.strip()
    return titles.duplicated(keep=False).astype(int)


def drop_duplicate_titles_with_blank_authors(data: pd.DataFrame) -> pd.DataFrame:
    duplicate_title = data["tag"] == 1
    blank_authors = data["authors"].isna() | (data["authors"].astype(str).str.strip() == "")
    return data.loc[~(duplicate_title & blank_authors)].copy()


def keep_one_observation_per_duplicated_title(data: pd.DataFrame) -> pd.DataFrame:
    title_column = "title" if "title" in data.columns else "openalex_title"
    title_values = data[title_column].fillna("").astype(str).str.strip()
    duplicate_title = title_values.duplicated(keep=False)
    duplicate_rows = data.loc[duplicate_title].drop_duplicates(
        subset=[title_column],
        keep="first",
    )
    nonduplicate_rows = data.loc[~duplicate_title]
    cleaned = pd.concat([nonduplicate_rows, duplicate_rows], axis=0)
    return cleaned.sort_index().copy()


def add_doi_versions_for_duplicate_titles(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    title_column = "title" if "title" in cleaned.columns else "openalex_title"

    doi_versions_by_title = {
        title: ordered_unique_nonblank(group["doi"])
        for title, group in cleaned.groupby(title_column, sort=False)
    }
    max_versions = max(
        (len(values) for values in doi_versions_by_title.values()),
        default=0,
    )

    for version_number in range(1, max_versions + 1):
        column = f"openalex_doi_{version_number}"
        cleaned[column] = ""

    for title, doi_versions in doi_versions_by_title.items():
        title_rows = cleaned[title_column] == title
        for index, doi in enumerate(doi_versions, start=1):
            cleaned.loc[title_rows, f"openalex_doi_{index}"] = doi

    return cleaned


def ordered_unique_nonblank(values: pd.Series) -> list[str]:
    ordered_values = []
    seen = set()
    for value in values:
        text = clean_text(value)
        if text and text not in seen:
            ordered_values.append(text)
            seen.add(text)
    return ordered_values


def use_longest_institutions_for_duplicate_titles(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    duplicate_rows = cleaned["tag"] == 1
    longest_by_title = (
        cleaned.loc[duplicate_rows]
        .groupby("title")["author_institutions"]
        .transform(longest_text_value)
    )
    cleaned.loc[duplicate_rows, "author_institutions"] = longest_by_title
    return cleaned


def use_longest_abstract_for_duplicate_titles(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    duplicate_rows = cleaned["tag"] == 1
    title_column = "title" if "title" in cleaned.columns else "openalex_title"
    longest_by_title = (
        cleaned.loc[duplicate_rows]
        .groupby(title_column)["abstract"]
        .transform(longest_text_value)
    )
    cleaned.loc[duplicate_rows, "abstract"] = longest_by_title
    return cleaned


def use_longest_jel_codes_for_duplicate_titles(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    duplicate_rows = cleaned["tag"] == 1
    title_column = "title" if "title" in cleaned.columns else "openalex_title"
    longest_by_title = (
        cleaned.loc[duplicate_rows]
        .groupby(title_column)["jel_codes"]
        .transform(longest_text_value)
    )
    cleaned.loc[duplicate_rows, "jel_codes"] = longest_by_title
    return cleaned


def use_union_for_duplicate_titles(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    cleaned = data.copy()
    duplicate_rows = cleaned["tag"] == 1
    title_column = "title" if "title" in cleaned.columns else "openalex_title"

    for column in columns:
        if column not in cleaned.columns:
            continue
        union_by_title = (
            cleaned.loc[duplicate_rows]
            .groupby(title_column)[column]
            .transform(union_text_values)
        )
        cleaned.loc[duplicate_rows, column] = union_by_title

    return cleaned


def union_text_values(values: pd.Series) -> str:
    union_values = []
    seen = set()
    for value in values:
        for item in split_semicolon_values(value):
            if item and item not in seen:
                union_values.append(item)
                seen.add(item)
    return "; ".join(union_values)


def split_semicolon_values(value) -> list[str]:
    if pd.isna(value):
        return []
    return [
        clean_text(item)
        for item in str(value).split(";")
        if clean_text(item)
    ]


def longest_text_value(values: pd.Series) -> str:
    text_values = values.fillna("").astype(str).str.strip()
    if text_values.empty:
        return ""
    return text_values.loc[text_values.str.len().idxmax()]


def add_openalex_prefix(data: pd.DataFrame) -> pd.DataFrame:
    return data.rename(
        columns={
            column: f"openalex_{column}"
            for column in data.columns
            if not column.startswith("openalex_")
        }
    )


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


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def clean_title(value) -> str:
    text = clean_text(value)
    characters = [
        character if character.isalnum() else " "
        for character in text
    ]
    return clean_text("".join(characters))


def clean_abstract(value) -> str:
    text = clean_text(value)
    text = decode_html_entities(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = remove_jel_text(text)
    return clean_text(text)


def extract_jel_codes(value) -> str:
    text = clean_text(value)
    text = decode_html_entities(text)
    text = re.sub(r"<[^>]+>", " ", text)
    jel_contexts = re.findall(
        r"(?:JEL|JEL\s+classification|JEL\s+classifications|JEL\s+codes?|JEL\s+No\.?)"
        r"[^A-Za-z0-9]{0,50}"
        r"([A-Z][0-9]{2}(?:\s*[,;/]\s*[A-Z][0-9]{2})*)",
        text,
        flags=re.IGNORECASE,
    )
    codes = []
    for context in jel_contexts:
        codes.extend(re.findall(r"\b[A-Z][0-9]{2}\b", context.upper()))
    return "; ".join(unique_values(codes))


def decode_html_entities(text: str) -> str:
    previous = text
    for _ in range(3):
        decoded = html.unescape(previous)
        if decoded == previous:
            return decoded
        previous = decoded
    return previous


def remove_jel_text(text: str) -> str:
    patterns = [
        r"(?:JEL|JEL\s+classification|JEL\s+classifications|JEL\s+codes?|JEL\s+No\.?)"
        r"[^.;\n]*[A-Z][0-9]{2}(?:\s*[,;/]\s*[A-Z][0-9]{2})*\.?",
        r"Classification-JEL:[^.;\n]*\.?",
    ]
    for pattern in patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text


def unique_values(values) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        value = clean_text(value)
        if value and value not in seen:
            unique.append(value)
            seen.add(value)
    return unique

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


def drop_blank_cleaned_titles(data: pd.DataFrame) -> pd.DataFrame:
    title_blank = data["title"].isna() | (data["title"].astype(str).str.strip() == "")
    return data.loc[~title_blank].copy()

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
        "The Econometric Society Annual Reports Econometrica",
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
        "Backmatter of Econometrica",
        "Ad Hoc Committee",
        "Committee on",
        "Index to Volume",
        "Recent Referees",
        "Minutes of the Annual Meeting",
        "John Bates Clark Award",
        "Appendix",
        "Job Openings for Economists",
        "OUP accepted manuscript",
        "Accepted Manuscripts",
        "Acknowledgment of Referees",
        "Acknowledgement of Referees",
        "Acknowledgment to Referees",
        "Acknowledgements to Referees",
        "Abstracts",
        "Minutes of the Annual Business Meeting",
        "Front Matter",
        "The Econometric Society Annual Reports Econometrica",
        "Announcements",
        "Independent Auditors' Report",
        "The Marriage Squeeze Interpretation of Dowry Inflation: Response",
        "Forthcoming Papers",
        "Data on Time to First Decision",
        "Election of Fellows to the Econometric Society",
        "North American Summer Meeting of the Econometric Society",
        "Lucas Prize Announcement",
        "Back Cover",
        "News Notes",
        "Nobel Lecture:",
        "Meeting of the Econometric Society",
        "Submission of Manuscripts to Econometrica",
        "Submission Fees and Response Times in Academic Publishing",
        "Submission of Manuscripts",
        "Subscription Page",
        "Table of Content",
        "The Econometric Society Annual Reports",
        "The Quarterly Journal of Economics",
        "An Astonishing Sixty Years The Legacy of Hiroshima",
        "the Diamond Water Paradox",
        "General Information on the Association",
        "Information on the Association",
        "Private and Social Rates of Return to Education of Academicians Note",
        "Protectionism through Prostitution",
        "Voltaire on Labor Markets and Monetary Policy",
        "Private and Social Rates of Return to Education of Academicians Note",
        "Fellows of the Econometric Society",
        "Galileo on the Diamond/Water Paradox",
        "Independent Auditor's Report",
        "JPE Submissions",
        "JPE Turnaround Times",
        "JPE Turnaround Times, Previous Two Years",
        "Referee List",
        "Title Page",
        "Editors Introduction",
        "Editor s Introduction",
        "Editor s Note",
        "Report by the AEA Data Editor",
        "AEA Data and Code Availability Policy",
        "Note from the AEA Secretary Treasurer about the Proceedings Supplement",
        "INDEPENDENT AUDITOR S REPORT",
        "Independent Auditor s Report",
        "Behavior of the Firm Under Regulatory Constraint",
        "Auditors Report Audited Financial Statements",
        "INDEPENDENT AUDITOR S REPORT",
        "John Bates Clark Medalist"
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


def abstract_from_inverted_index(value) -> str:
    inverted_index = parse_json_cell(value)
    if not isinstance(inverted_index, dict):
        return ""

    positioned_words = []
    for word, positions in inverted_index.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int):
                positioned_words.append((position, str(word)))

    if not positioned_words:
        return ""

    positioned_words.sort(key=lambda item: item[0])
    return clean_text(" ".join(word for _, word in positioned_words))


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
