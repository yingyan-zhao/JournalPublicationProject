from pathlib import Path
import html
import json
import re
import unicodedata

import os
import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

INPUT_CSV_Crossref = Path("data/raw_csv/Crossref_Works.csv")
OUTPUT_CSV_Crossref = Path("data/processed/Crossref_Works_Cleaned.csv")
CROSSREF_COLUMNS_TO_DROP_AFTER_CLEANING = [
    "crossref_doi",
    "crossref_tag",
]
CROSSREF_DOIS_TO_DROP = {
    "10.1257/aer.97.3.1033",
    "10.1257/aer.15000025",
    "10.1257/aer.108.6.1598",
}
CROSSREF_EXACT_TITLES_TO_DROP = {
    "Index",
    "Announcement",
    "Back Matter",
    "Editorial",
    "Editorial Board",
    "Masthead",
    "Notice",
    "Travel Fund",
}

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
    if not INPUT_CSV_Crossref.exists():
        raise FileNotFoundError(f"{INPUT_CSV_Crossref} does not exist.")

    crossref = pd.read_csv(INPUT_CSV_Crossref)
    crossref_selected = clean_crossref_data(crossref)

    OUTPUT_CSV_Crossref.parent.mkdir(parents=True, exist_ok=True)
    crossref_selected.to_csv(OUTPUT_CSV_Crossref, index=False)

    print(f"Read {len(crossref)} Crossref rows from {INPUT_CSV_Crossref}.")
    print(f"Wrote {len(crossref_selected)} cleaned Crossref rows to {OUTPUT_CSV_Crossref}.")
    print(f"Rows with abstract: {count_nonblank(crossref_selected, 'crossref_abstract')}.")
    print(f"Rows with JEL codes in abstract: {count_nonblank(crossref_selected, 'crossref_jel_codes')}.")
    print(f"Rows with duplicated DOI: {count_duplicate_rows(crossref_selected, 'crossref_doi_1')}.")
    print(f"Rows with duplicated DOI: {count_duplicate_rows(crossref_selected, 'crossref_doi_2')}.")
    print(f"Rows with duplicated DOI: {count_duplicate_rows(crossref_selected, 'crossref_doi_3')}.")
    print(f"Rows with duplicated title: {count_duplicate_rows(crossref_selected, 'crossref_title')}.")


def clean_crossref_data(crossref: pd.DataFrame) -> pd.DataFrame:
    crossref_selected = keep_columns(crossref, CROSSREF_COLUMNS)

    # clean paper fields
    crossref_selected["doi"] = crossref_selected["DOI"].apply(clean_doi)
    crossref_selected = crossref_selected.drop(columns=["DOI"])
    crossref_selected = drop_dois(crossref_selected, CROSSREF_DOIS_TO_DROP)

    # clean the title
    crossref_selected["title"] = crossref_selected["title"].apply(first_json_value)
    crossref_selected["title"] = crossref_selected["title"].replace("", pd.NA)
    crossref_selected = drop_blank_titles(crossref_selected)
    crossref_selected = drop_correction_titles(crossref_selected)
    crossref_selected = drop_exact_titles(crossref_selected, CROSSREF_EXACT_TITLES_TO_DROP)
    crossref_selected["title"] = crossref_selected["title"].apply(clean_title)
    crossref_selected = rename_specific_crossref_titles(crossref_selected)
    crossref_selected = drop_blank_titles(crossref_selected)

    crossref_selected["tag"] = duplicate_title_tag(crossref_selected, "title")
    crossref_selected = drop_duplicate_titles_with_blank_authors(crossref_selected)
    crossref_selected = crossref_selected.rename(
        columns={"title": "crossref_title"}
    )

    # clean abstract and extract JEL codes embedded in it
    crossref_selected["jel_codes"] = crossref_selected["abstract"].apply(extract_jel_codes)
    crossref_selected["abstract"] = crossref_selected["abstract"].apply(clean_abstract)
    crossref_selected = use_longest_abstract_for_duplicate_titles(crossref_selected)
    crossref_selected = use_longest_jel_codes_for_duplicate_titles(crossref_selected)
    crossref_selected = use_longest_affiliations_for_duplicate_titles(crossref_selected)
    crossref_selected = add_doi_versions_for_titles(crossref_selected)
    crossref_selected = keep_one_observation_per_duplicated_title(crossref_selected)

    # Keep Journal name and Journal issn
    crossref_selected = crossref_selected.rename(
        columns={"_query_journal": "crossref_journalname","_query_issn": "crossref_journalissn"}
    )
    crossref_selected = add_crossref_prefix(crossref_selected)
    crossref_selected = drop_columns(crossref_selected, CROSSREF_COLUMNS_TO_DROP_AFTER_CLEANING)

    return crossref_selected


def rename_specific_crossref_titles(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    if "title" not in cleaned.columns or "published_year" not in cleaned.columns:
        return cleaned

    title = cleaned["title"].fillna("").astype(str).str.strip()
    publication_year = cleaned["published_year"].fillna("").astype(str).str.strip()
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


def duplicate_title_tag(data: pd.DataFrame, title_column: str) -> pd.Series:
    titles = data[title_column].fillna("").astype(str).str.strip()
    return titles.duplicated(keep=False).astype(int)


def drop_exact_titles(data: pd.DataFrame, titles_to_drop: set[str]) -> pd.DataFrame:
    titles = data["title"].fillna("").astype(str).str.strip()
    return data.loc[~titles.isin(titles_to_drop)].copy()


def drop_dois(data: pd.DataFrame, dois_to_drop: set[str]) -> pd.DataFrame:
    doi_values = data["doi"].apply(clean_doi).str.lower()
    return data.loc[~doi_values.isin(dois_to_drop)].copy()


def drop_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    columns_to_drop = [
        column for column in columns
        if column in data.columns
    ]
    return data.drop(columns=columns_to_drop).copy()


def drop_duplicate_titles_with_blank_authors(data: pd.DataFrame) -> pd.DataFrame:
    duplicate_title = data["tag"] == 1
    blank_authors = data["authors"].isna() | (data["authors"].astype(str).str.strip() == "")
    return data.loc[~(duplicate_title & blank_authors)].copy()


def use_longest_abstract_for_duplicate_titles(data: pd.DataFrame) -> pd.DataFrame:
    return use_longest_value_for_duplicate_titles(data, "abstract")


def use_longest_jel_codes_for_duplicate_titles(data: pd.DataFrame) -> pd.DataFrame:
    return use_longest_value_for_duplicate_titles(data, "jel_codes")


def use_longest_affiliations_for_duplicate_titles(data: pd.DataFrame) -> pd.DataFrame:
    return use_longest_value_for_duplicate_titles(data, "author_affiliations")


def use_longest_value_for_duplicate_titles(data: pd.DataFrame, value_column: str) -> pd.DataFrame:
    cleaned = data.copy()
    if value_column not in cleaned.columns:
        return cleaned

    duplicate_rows = cleaned["tag"] == 1
    title_column = "title" if "title" in cleaned.columns else "crossref_title"
    longest_by_title = (
        cleaned.loc[duplicate_rows]
        .groupby(title_column)[value_column]
        .transform(longest_text_value)
    )
    cleaned.loc[duplicate_rows, value_column] = longest_by_title
    return cleaned


def add_doi_versions_for_titles(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    title_column = "title" if "title" in cleaned.columns else "crossref_title"
    doi_versions_by_title = {
        title: ordered_unique_nonblank(group["doi"])
        for title, group in cleaned.groupby(title_column, sort=False)
    }
    max_versions = max(
        (len(values) for values in doi_versions_by_title.values()),
        default=0,
    )

    for version_number in range(1, max_versions + 1):
        cleaned[f"crossref_doi_{version_number}"] = ""

    for title, doi_versions in doi_versions_by_title.items():
        title_rows = cleaned[title_column] == title
        for index, doi in enumerate(doi_versions, start=1):
            cleaned.loc[title_rows, f"crossref_doi_{index}"] = doi

    return cleaned


def keep_one_observation_per_duplicated_title(data: pd.DataFrame) -> pd.DataFrame:
    title_column = "title" if "title" in data.columns else "crossref_title"
    title_values = data[title_column].fillna("").astype(str).str.strip()
    duplicate_title = title_values.duplicated(keep=False)
    duplicate_rows = data.loc[duplicate_title].drop_duplicates(
        subset=[title_column],
        keep="first",
    )
    nonduplicate_rows = data.loc[~duplicate_title]
    cleaned = pd.concat([nonduplicate_rows, duplicate_rows], axis=0)
    return cleaned.sort_index().copy()


def longest_text_value(values: pd.Series) -> str:
    text_values = values.fillna("").astype(str).str.strip()
    if text_values.empty:
        return ""
    return text_values.loc[text_values.str.len().idxmax()]


def ordered_unique_nonblank(values: pd.Series) -> list[str]:
    ordered_values = []
    seen = set()
    for value in values:
        text = clean_text(value)
        if text and text not in seen:
            ordered_values.append(text)
            seen.add(text)
    return ordered_values


def add_crossref_prefix(data: pd.DataFrame) -> pd.DataFrame:
    return data.rename(
        columns={
            column: f"crossref_{column}"
            for column in data.columns
            if not column.startswith("crossref_")
        }
    )


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
    text = html.unescape(text)
    text = remove_jel_text(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_jel_codes(value) -> str:
    text = clean_text(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
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


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return " ".join(text.split())


def clean_title(value) -> str:
    text = clean_text(value)
    characters = [
        character if character.isalnum() else " "
        for character in text
    ]
    return clean_text("".join(characters))


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

    correction_author = data["authors"].fillna("").str.contains(
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
