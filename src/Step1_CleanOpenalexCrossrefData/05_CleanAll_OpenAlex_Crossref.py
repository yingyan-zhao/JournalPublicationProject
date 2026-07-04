from pathlib import Path
import json
import os
import re
from typing import Any

import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

INPUT_CSV = Path("data/processed/OpenAlex_Crossref_All_Webscraped.csv")
OUTPUT_CSV = Path("data/processed/OpenAlex_Crossref_All_Webscraped_Cleaned.csv")

## ##############################################################################################################
# [05_CleanAll_OpenAlex_Crossref.py] the code does this:
# Step 1. Reads input data. Input: data/processed/OpenAlex_Crossref_All_Webscraped.csv
# Step 2. Cleans DOI values. It lowercases DOI, removes DOI URL prefixes, and removes trailing punctuation.
# Step 3. Removes duplicate DOI values within each row.
# Step 4. Creates doi_list. It combines DOI information from doi, openalex_doi_1,
# openalex_doi_2, openalex_doi_3, crossref_doi_1, crossref_doi_2,
# crossref_doi_3, and scrape_doi. The final doi_list stores unique DOI values
# separated by ";".
# Step 5. Creates DOI version columns. From doi_list, it creates doi_1, doi_2,
# doi_3, etc.
# Step 6. Builds cleaned journalname. It takes the first nonblank value from
# journalname, scrape_journalname, openalex_journalname, and crossref_journalname.
# Step 7. Builds cleaned journalissn. It takes openalex_journalissn; if blank, it
# takes crossref_journalissn.
# Step 8. Builds cleaned publication_year. It takes openalex_publication_year; if
# blank, it takes crossref_published_year.
# Step 9. Builds cleaned title. It takes the first nonblank value from title,
# scrape_title, openalex_title, and crossref_title.
# Step 10. Cleans title text. It removes endings such as Available for Purchase,
# Open Access, and trailing *.
# Step 11. Builds cleaned abstract. It takes scrape_abstract; if blank,
# openalex_abstract; if still blank, crossref_abstract.
# Step 12. Builds cleaned jel_codes. It takes scrape_jel_codes; if blank,
# openalex_jel_codes; if still blank, crossref_jel_codes.
# Step 13. Renames scrape JEL context. scrape_jel_context becomes jel_context.
# Step 14. Drops many intermediate/source columns. It drops merge fields, original
# DOI fields, OpenAlex technical columns, Crossref technical columns, scrape
# URL/status/error fields, and source columns already combined into cleaned variables.
# Step 15. Drops non-paper title rows. It removes rows whose cleaned title contains
# phrases in NONPAPER_TITLE_PATTERNS.
# Step 16. Drops selected Crossref author group rows. It drops rows where
# crossref_authors contains Opportunity Insights Team, Oregon Health Study Group,
# or the Seminar Dynamics Collective.
# Step 17. Prints cleaning summary. It reports input rows, output rows, rows with
# DOI list, number of DOI version columns, rows with abstract, rows with JEL codes,
# duplicated title rows, and duplicated doi_list rows.
# Step 18. Exports cleaned data. Output:
# data/processed/OpenAlex_Crossref_All_Webscraped_Cleaned.csv
## ##############################################################################################################

DROP_COLUMNS = [
    "record_status",
    "match_strategy",
    "doi",
    "openalex_id",
    "openalex_doi_1",
    "openalex_doi_2",
    "openalex_doi_3",
    "openalex_type",
    "openalex_type_crossref",
    "openalex_orcid_ids",
    "openalex_author_positions",
    "openalex_cited_by_count",
    "openalex_collection_date",
    "openalex_primary_domain",
    "openalex_primary_field",
    "openalex_primary_subfield",
    "openalex_top3_keywords",
    "openalex_top3_concepts",
    "openalex_level0_concepts",
    "crossref_published",
    "crossref_published-print",
    "crossref_published-online",
    "crossref_reference-count",
    "crossref_references-count",
    "crossref_is-referenced-by-count",
    "crossref_issued",
    "crossref_container-title",
    "crossref_short-container-title",
    "crossref_publisher",
    "crossref_volume",
    "crossref_issue",
    "crossref_page",
    "crossref_type",
    "crossref_URL",
    "crossref_ISSN",
    "crossref_author_sequence",
    "crossref_author_orcids",
    "crossref_collection_date",
    "crossref_doi_1",
    "crossref_doi_2",
    "crossref_doi_3",
    "scrape_doi",
    "scrape_doi_source",
    "scrape_url",
    "scrape_final_url",
    "scrape_http_status",
    "scrape_jel_source",
    "scrape_scraped_at",
    "scrape_error",
    "scrape_jel_source",
    "scrape_jel_context",
    "scrape_scraped_at",
    "scrape_error"
]



NONPAPER_TITLE_PATTERNS = [
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
    "John Bates Clark Medalist",
    "A PHENOMENOLOGICAL STUDY OF TEACHING ROLE PERCEPTIONS OF COLLEGE AND UNIVERSITY PROFESSORS",
    "International Bibliography of Economics"
]

DROP_CROSSREF_AUTHORS = {
    "Opportunity Insights Team",
    "Oregon Health Study Group",
    "the Seminar Dynamics Collective",
}



COALESCED_SOURCE_COLUMNS = [
    "openalex_journalname",
    "crossref_journalname",
    "scrape_journalname",
    "openalex_journalissn",
    "crossref_journalissn",
    "openalex_publication_year",
    "crossref_published_year",
    "scrape_title",
    "openalex_title",
    "crossref_title",
    "scrape_abstract",
    "openalex_abstract",
    "crossref_abstract",
    "scrape_jel_codes",
    "openalex_jel_codes",
    "crossref_jel_codes",
]

DOI_LIST_COLUMNS = [
    "doi",
    "openalex_doi_1",
    "openalex_doi_2",
    "openalex_doi_3",
    "crossref_doi_1",
    "crossref_doi_2",
    "crossref_doi_3",
    "scrape_doi",
]

## ##############################################################################################
# In [05_CleanAll_OpenAlex_Crossref.py], Step by step, it does this:
# Step 1. Reads input data Input: data/processed/OpenAlex_Crossref_All_Webscraped.csv
# Step 2. Cleans DOI values. It lowercases DOI, removes DOI URL prefixes, and removes trailing punctuation.
# Step 3. Removes duplicate DOI values within each row
# Step 4. Creates doi_list: It combines DOI information from: doi, openalex_doi_1, openalex_doi_2, openalex_doi_3, crossref_doi_1, crossref_doi_2, crossref_doi_3, and scrape_doi. The final doi_list stores unique DOI values separated by “;”
# Step 5. Creates DOI version columns From doi_list, it creates doi_1, doi_2, doi_3, etc.
# Step 6. Builds cleaned journalname. It takes the first nonblank value from: journalname, scrape_journalname, openalex_journalname, crossref_journalname.
# Step 7. Builds cleaned journalissn: It takes openalex_journalissn; if blank, it takes crossref_journalissn.
# Step 8. Builds cleaned publication_year: It takes openalex_publication_year; if blank, it takes crossref_published_year.
# Step 9. Builds cleaned title: It takes the first nonblank value from: title, scrape_title, openalex_title, crossref_title.
# Step 10. Cleans title text: It removes endings such as Available for Purchase, Open Access, and trailing *.
# Step 11. Builds cleaned abstract: It takes scrape_abstract; if blank, openalex_abstract; if still blank, crossref_abstract.
# Step 12. Builds cleaned jel_codes: It takes scrape_jel_codes; if blank, openalex_jel_codes; if still blank, crossref_jel_codes.
# Step 13. Renames scrape JEL context: scrape_jel_context becomes jel_context.
# Step 14. Drops many intermediate/source columns : It drops merge fields, original DOI fields, OpenAlex technical columns, Crossref technical columns, scrape URL/status/error fields, and source columns already combined into cleaned variables.
# Step 15. Drops non-paper title rows: It removes rows whose cleaned title contains phrases in NONPAPER_TITLE_PATTERNS
# Step 16. Drops selected Crossref author group rows. It drops rows where crossref_authors contains: Opportunity Insights Team, Oregon Health Study Group, or the Seminar Dynamics Collective.
# Step 17. Prints cleaning summary It reports input rows, output rows, rows with DOI list, number of DOI version columns, rows with abstract, rows with JEL codes, duplicated title rows, and duplicated doi_list rows.
# Step 18. Exports cleaned data Output: data/processed/OpenAlex_Crossref_All_Webscraped_Cleaned.csv
## ##############################################################################################

def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"{INPUT_CSV} does not exist.")

    data = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)
    cleaned = clean_all_data(data)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print("Clean all OpenAlex/Crossref/Webscrape summary:")
    print(f"  Input rows: {len(data)}")
    print(f"  Output rows: {len(cleaned)}")
    print(f"  Rows with DOI list: {count_nonblank(cleaned, 'doi_list')}")
    print(f"  DOI version columns: {count_doi_version_columns(cleaned)}")
    print(f"  Rows with abstract: {count_nonblank(cleaned, 'abstract')}")
    print(f"  Rows with JEL codes: {count_nonblank(cleaned, 'jel_codes')}")
    print(
        "  Rows with duplicated title: "
        f"{count_duplicate_rows(cleaned, 'title', normalize_title)}"
    )
    print(
        "  Rows with duplicated doi_list: "
        f"{count_duplicate_rows(cleaned, 'doi_list', normalize_doi_list)}"
    )

    cleaned.to_csv(OUTPUT_CSV, index=False)
    print(f"  Output CSV: {OUTPUT_CSV}")


def clean_all_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()

    cleaned["doi_list"] = data.apply(doi_list_from_row, axis=1)
    cleaned = add_doi_version_columns(cleaned)
    cleaned["journalname"] = coalesce_columns(
        data,
        ["journalname", "scrape_journalname", "openalex_journalname", "crossref_journalname"],
        clean_text,
    )
    cleaned["journalissn"] = coalesce_columns(
        data,
        ["openalex_journalissn", "crossref_journalissn"],
        clean_text,
    )
    cleaned["publication_year"] = coalesce_columns(
        data,
        ["openalex_publication_year", "crossref_published_year"],
        clean_year,
    )
    cleaned["title"] = coalesce_columns(
        data,
        ["title", "scrape_title", "openalex_title", "crossref_title"],
        clean_text,
    )
    cleaned["title"] = cleaned["title"].apply(clean_title)
    cleaned["abstract"] = coalesce_columns(
        data,
        ["scrape_abstract", "openalex_abstract", "crossref_abstract"],
        clean_text,
    )
    cleaned["jel_codes"] = coalesce_columns(
        data,
        ["scrape_jel_codes", "openalex_jel_codes", "crossref_jel_codes"],
        clean_text,
    )
    cleaned = cleaned.rename(columns={"scrape_jel_context": "jel_context"})

    columns_to_drop = DROP_COLUMNS + COALESCED_SOURCE_COLUMNS
    cleaned = cleaned.drop(columns=[column for column in columns_to_drop if column in cleaned.columns])

    before_nonpapers = len(cleaned)
    cleaned = drop_correction_titles(cleaned)
    dropped_nonpapers = before_nonpapers - len(cleaned)
    print(f"  Dropped non-paper title rows: {dropped_nonpapers}")

    before_crossref_author_groups = len(cleaned)
    cleaned = drop_crossref_author_groups(cleaned)
    dropped_crossref_author_groups = before_crossref_author_groups - len(cleaned)
    print(f"  Dropped Crossref author group rows: {dropped_crossref_author_groups}")
    return cleaned


def doi_list_from_row(row: pd.Series) -> str:
    dois = []
    seen = set()
    for column in DOI_LIST_COLUMNS:
        doi = clean_doi(row.get(column, ""))
        if doi == "" or doi in seen:
            continue
        dois.append(doi)
        seen.add(doi)
    return "; ".join(dois)


def add_doi_version_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    doi_versions = data["doi_list"].apply(split_doi_list)
    max_versions = int(doi_versions.apply(len).max()) if len(doi_versions) else 0

    for version_number in range(1, max_versions + 1):
        column = f"doi_{version_number}"
        data[column] = doi_versions.apply(
            lambda dois: dois[version_number - 1] if len(dois) >= version_number else ""
        )

    return data


def split_doi_list(value: Any) -> list[str]:
    return [clean_doi(doi) for doi in str(value or "").split(";") if clean_doi(doi)]


def coalesce_columns(data: pd.DataFrame, columns: list[str], cleaner) -> pd.Series:
    values = pd.Series([""] * len(data), index=data.index)
    for column in columns:
        if column not in data.columns:
            continue
        candidate = data[column].apply(cleaner)
        values = values.mask(values == "", candidate)
    return values


def coalesce_with_source(
    data: pd.DataFrame,
    columns_and_sources: list[tuple[str, str]],
    cleaner,
) -> tuple[pd.Series, pd.Series]:
    values = pd.Series([""] * len(data), index=data.index)
    sources = pd.Series([""] * len(data), index=data.index)
    for column, source in columns_and_sources:
        if column not in data.columns:
            continue
        candidate = data[column].apply(cleaner)
        use_candidate = (values == "") & (candidate != "")
        values = values.mask(use_candidate, candidate)
        sources = sources.mask(use_candidate, source)
    return values, sources


def get_column(data: pd.DataFrame, column: str) -> pd.Series:
    if column in data.columns:
        return data[column]
    return pd.Series([""] * len(data), index=data.index)


def publication_date(row: pd.Series) -> str:
    for column in ["crossref_published", "crossref_published-print", "crossref_published-online", "crossref_issued"]:
        value = row.get(column, "")
        date_text = date_from_json_cell(value)
        if date_text:
            return date_text
    year = clean_year(row.get("openalex_publication_year", ""))
    return year


def date_from_json_cell(value: Any) -> str:
    parsed = parse_json_cell(value)
    if not isinstance(parsed, dict):
        return ""
    date_parts = parsed.get("date-parts") or []
    if not date_parts or not date_parts[0]:
        return ""
    parts = [str(part).zfill(2) for part in date_parts[0] if str(part).strip()]
    if not parts:
        return ""
    parts[0] = str(int(parts[0]))
    return "-".join(parts)


def parse_json_cell(value: Any):
    text = clean_text(value)
    if text == "":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def clean_doi(value: Any) -> str:
    return (
        clean_text(value)
        .lower()
        .replace("https://doi.org/", "")
        .replace("http://doi.org/", "")
        .replace("http://dx.doi.org/", "")
        .rstrip(".,;")
    )


def clean_title(value: Any) -> str:
    title = clean_text(value)
    title = re.sub(r"\s+Available for Purchase$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+Open Access$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*\*$", "", title)
    return clean_text(title)


def clean_abstract(value: Any) -> str:
    text = clean_text(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^Abstract\s*:?\s*", "", text, flags=re.IGNORECASE)
    return clean_text(text)


def clean_keyword_list(value: Any) -> str:
    text = clean_text(value)
    if text == "":
        return ""
    pieces = re.split(r";|,", text)
    return "; ".join(unique_values(clean_text(piece) for piece in pieces))


def clean_jel_codes(value: Any) -> str:
    codes = re.findall(r"\b[A-Z][0-9]{2}\b", clean_text(value).upper())
    return "; ".join(unique_values(codes))


def clean_people_or_list(value: Any) -> str:
    text = clean_text(value)
    if text in {"[]", "{}"}:
        return ""
    parsed = parse_json_cell(text)
    if isinstance(parsed, list):
        return "; ".join(unique_values(clean_text(item) for item in parsed))
    return clean_text(text)


def clean_year(value: Any) -> str:
    text = clean_text(value)
    match = re.search(r"\b(19|20)\d{2}\b", text)
    if match:
        return match.group(0)
    return ""


def clean_integer(value: Any) -> str:
    text = clean_text(value)
    if text == "":
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return ""


def is_nonpaper_title(title: str) -> bool:
    normalized = clean_text(title).casefold()
    return any(pattern.casefold() in normalized for pattern in NONPAPER_TITLE_PATTERNS)


def drop_correction_titles(data: pd.DataFrame) -> pd.DataFrame:
    if "title" not in data.columns:
        return data.copy()
    correction_title = data["title"].fillna("").astype(str).apply(is_nonpaper_title)
    return data.loc[~correction_title].copy()


def drop_crossref_author_groups(data: pd.DataFrame) -> pd.DataFrame:
    if "crossref_authors" not in data.columns:
        return data.copy()
    drop_rows = data["crossref_authors"].apply(has_dropped_crossref_author)
    return data.loc[~drop_rows].copy()


def has_dropped_crossref_author(value: Any) -> bool:
    authors = [clean_text(author) for author in str(value or "").split(";")]
    return any(author in DROP_CROSSREF_AUTHORS for author in authors)


def normalize_title(value: Any) -> str:
    title = clean_text(value).lower()
    title = title.replace("&amp;", "and").replace("&", "and")
    title = "".join(character if character.isalnum() else " " for character in title)
    return " ".join(title.split())


def normalize_doi_list(value: Any) -> str:
    dois = split_doi_list(value)
    return "; ".join(dois)


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def unique_values(values) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        value = clean_text(value)
        if value and value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def count_nonblank(data: pd.DataFrame, column: str) -> int:
    if column not in data.columns:
        return 0
    return int(data[column].fillna("").astype(str).str.strip().ne("").sum())


def count_doi_version_columns(data: pd.DataFrame) -> int:
    return sum(1 for column in data.columns if re.fullmatch(r"doi_\d+", column))


def count_duplicate_rows(data: pd.DataFrame, column: str, cleaner) -> int:
    if column not in data.columns:
        return 0
    values = data[column].apply(cleaner)
    values = values.loc[values != ""]
    return int(values.duplicated(keep=False).sum())


if __name__ == "__main__":
    main()
