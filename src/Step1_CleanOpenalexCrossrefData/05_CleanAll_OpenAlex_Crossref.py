from pathlib import Path
import json
import os
import re
from typing import Any

import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

INPUT_CSV = Path("data/processed/OpenAlex_Crossref_All_Webscraped.csv")
OUTPUT_CSV = Path("data/processed/OpenAlex_Crossref_All_Webscraped_Cleaned.csv")

DROP_COLUMNS = [
    "record_status",
    "match_strategy",
    "openalex_id",
    "openalex_type",
    "openalex_type_crossref",
    "openalex_orcid_ids",
    "openalex_author_positions",
    "openalex_cited_by_count",
    "openalex_collection_date",
    "crossref_volume",
    "crossref_issue",
    "crossref_page",
    "crossref_type",
    "crossref_reference-count",
    "crossref_references-count",
    "crossref_is-referenced-by-count",
    "crossref_URL",
    "crossref_author_sequence",
    "crossref_author_orcids",
    "crossref_ISSN",
    "crossref_collection_date",
    "scrape_url	scrape_final_url",
    "scrape_http_status",
    "scrape_jel_source",
    "scrape_scraped_at",
    "scrape_error",
    "scrape_url",
    "scrape_final_url",
    "crossref_published",
    "crossref_published-print",
    "crossref_published-online",
    "crossref_issued",
    "crossref_container-title",
    "crossref_short-container-title",
    "crossref_publisher"
]

COALESCED_SOURCE_COLUMNS = [
    "openalex_doi",
    "crossref_doi",
    "scrape_doi",
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
    "scrape_jel_context"
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
    "The Quarterly Journal of Economics"
]


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"{INPUT_CSV} does not exist.")

    data = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)
    cleaned = clean_all_data(data)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT_CSV, index=False)

    print("Clean all OpenAlex/Crossref/Webscrape summary:")
    print(f"  Input rows: {len(data)}")
    print(f"  Output rows: {len(cleaned)}")
    print(f"  Rows with DOI: {count_nonblank(cleaned, 'doi')}")
    print(f"  Rows with abstract: {count_nonblank(cleaned, 'abstract')}")
    print(f"  Rows with JEL codes: {count_nonblank(cleaned, 'jel_codes')}")
    print(f"  Output CSV: {OUTPUT_CSV}")


def clean_all_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()

    cleaned["doi"] = coalesce_columns(
        data,
        ["doi", "scrape_doi", "openalex_doi", "crossref_doi"],
        clean_doi,
    )
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
    cleaned = cleaned.loc[~cleaned["title"].apply(is_nonpaper_title)].copy()
    dropped_nonpapers = before_nonpapers - len(cleaned)
    #
    print(f"  Dropped non-paper title rows: {dropped_nonpapers}")
    return cleaned


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


def normalize_title(value: Any) -> str:
    title = clean_text(value).lower()
    title = title.replace("&amp;", "and").replace("&", "and")
    title = "".join(character if character.isalnum() else " " for character in title)
    return " ".join(title.split())


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


if __name__ == "__main__":
    main()
