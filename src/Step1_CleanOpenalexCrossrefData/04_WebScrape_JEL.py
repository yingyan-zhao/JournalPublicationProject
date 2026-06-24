from pathlib import Path
import argparse
from datetime import datetime
import os
import re
import time
from typing import Any

from bs4 import BeautifulSoup
import pandas as pd
import requests


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

INPUT_CSV = Path("data/processed/OpenAlex_Crossref_All.csv")
OUTPUT_CSV = Path("data/processed/OpenAlex_Crossref_All_Webscraped.csv")

INPUT_COLUMN_CANDIDATES = {
    "title": ["title", "crossref_title", "openalex_title"],
    "doi": [
        "openalex_doi_1",
        "openalex_doi_2",
        "openalex_doi_3",
        "crossref_doi_1",
        "crossref_doi_2",
        "crossref_doi_3",
    ],
    "journalname": ["journalname", "crossref_journalname", "openalex_journalname"],
}

DOI_FALLBACK_COLUMNS = INPUT_COLUMN_CANDIDATES["doi"]

JEL_CODE_PATTERN = re.compile(r"\b[A-Z][0-9]{2}\b")
JEL_CONTEXT_PATTERN = re.compile(
    r"(?:JEL|JEL\s+classification|JEL\s+classifications|JEL\s+codes?)"
    r"[^A-Za-z0-9]{0,20}"
    r"([A-Z][0-9]{2}(?:\s*[,;/]\s*[A-Z][0-9]{2})*)",
    flags=re.IGNORECASE,
)

SCRAPE_OUTPUT_PREFIX = "scrape_"

OUTPUT_COLUMNS_BASE = [
    "title",
    "doi",
    "doi_source",
    "journalname",
    "scrape_url",
    "final_url",
    "http_status",
    "authors",
    "author_institutions",
    "abstract",
    "keywords",
    "jel_codes",
    "jel_source",
    "jel_context",
    "scraped_at",
    "error",
]
OUTPUT_COLUMNS = [
    column if column.startswith(SCRAPE_OUTPUT_PREFIX) else f"{SCRAPE_OUTPUT_PREFIX}{column}"
    for column in OUTPUT_COLUMNS_BASE
]


def main() -> None:
    args = parse_args()
    input_csv = resolve_input_csv(args.input_csv)
    data = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    validate_input_columns(data, input_csv)
    data = ensure_scrape_columns(data)

    rows_to_scrape = rows_needing_scrape(
        data=data,
        overwrite=args.overwrite,
    )
    if args.limit is not None:
        rows_to_scrape = rows_to_scrape[: args.limit]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Read full input dataset from {input_csv}.")
    print("Scraping row by row. Duplicate rows are not deduplicated.")
    print("DOI fallback order: " + " -> ".join(DOI_FALLBACK_COLUMNS))
    print(f"Input rows: {len(data)}")
    print(f"Rows to scrape in this run: {len(rows_to_scrape)}")
    print(f"Writing updated dataset to {args.output_csv}.")

    for position, row_index in enumerate(rows_to_scrape, start=1):
        record = scrape_record_from_row(data.loc[row_index])
        result = scrape_one_record(
            record=record,
            timeout=args.timeout,
            user_agent=args.user_agent,
        )
        for column, value in add_scrape_prefix(result).items():
            data.at[row_index, column] = value

        print(
            f"{position}/{len(rows_to_scrape)} "
            f"row={row_index} "
            f"status={result['http_status']} "
            f"jel={result['jel_codes']} "
            f"doi={result['doi']} "
            f"doi_source={result['doi_source']}"
        )

        if position % args.save_every == 0:
            data.to_csv(args.output_csv, index=False)
            print(f"  Saved progress to {args.output_csv}.")

        time.sleep(args.sleep)

    data.to_csv(args.output_csv, index=False)
    print(f"Wrote updated full dataset to {args.output_csv}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read the full OpenAlex/Crossref merged dataset and scrape website "
            "metadata into scrape_-prefixed columns on the same row."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--limit", type=int, help="Scrape only the first N eligible rows.")
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--save-every", type=int, default=25, help="Save progress after this many scraped rows.")
    parser.add_argument(
        "--user-agent",
        default=(
            "Mozilla/5.0 (compatible; JournalPublicationProject/1.0; "
            "mailto:yingyan_zhao@example.com)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rescrape rows even if scrape_scraped_at is already filled.",
    )
    return parser.parse_args()


def resolve_input_csv(input_csv: Path) -> Path:
    if input_csv.exists():
        return input_csv
    raise FileNotFoundError(f"Could not find the full merged input dataset: {input_csv}")


def validate_input_columns(data: pd.DataFrame, input_csv: Path) -> None:
    missing = []
    for output_column, candidates in INPUT_COLUMN_CANDIDATES.items():
        source_column = first_existing_column(data, candidates)
        if source_column == "":
            missing.append(f"{output_column} from {candidates}")

    if missing:
        raise ValueError(f"{input_csv} is missing required columns: {missing}")


def ensure_scrape_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    for column in OUTPUT_COLUMNS:
        if column not in data.columns:
            data[column] = ""
    return data


def rows_needing_scrape(data: pd.DataFrame, overwrite: bool) -> list[int]:
    rows = []
    for row_index, row in data.iterrows():
        record = scrape_record_from_row(row)
        if not record["doi_candidates"]:
            continue
        if not overwrite and row_has_scraped_information(row):
            continue
        rows.append(row_index)
    return rows


def scrape_record_from_row(row: pd.Series) -> dict[str, Any]:
    doi_candidates = doi_candidates_from_row(row)
    return {
        "title": coalesce_row_value(row, INPUT_COLUMN_CANDIDATES["title"], clean_text),
        "doi": doi_candidates[0]["doi"] if doi_candidates else "",
        "doi_source": doi_candidates[0]["doi_source"] if doi_candidates else "",
        "doi_candidates": doi_candidates,
        "journalname": coalesce_row_value(row, INPUT_COLUMN_CANDIDATES["journalname"], clean_text),
    }


def doi_candidates_from_row(row: pd.Series) -> list[dict[str, str]]:
    candidates = []
    seen = set()
    for column in DOI_FALLBACK_COLUMNS:
        doi = normalize_doi(row.get(column, ""))
        if not doi or doi in seen:
            continue
        candidates.append({"doi": doi, "doi_source": column})
        seen.add(doi)
    return candidates


def coalesce_row_value(row: pd.Series, columns: list[str], cleaner) -> str:
    for column in columns:
        value = cleaner(row.get(column, ""))
        if value:
            return value
    return ""


def first_existing_column(data: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in data.columns:
            return column
    return ""


def scrape_one_record(record: dict[str, str], timeout: float, user_agent: str) -> dict[str, str]:
    scraped_at = datetime.now().isoformat(timespec="seconds")
    result = {
        "title": "",
        "doi": "",
        "doi_source": "",
        "journalname": record["journalname"],
        "scrape_url": "",
        "final_url": "",
        "http_status": "",
        "authors": "",
        "author_institutions": "",
        "abstract": "",
        "keywords": "",
        "jel_codes": "",
        "jel_source": "",
        "jel_context": "",
        "scraped_at": scraped_at,
        "error": "",
    }

    errors = []
    for doi_candidate in record["doi_candidates"]:
        doi = doi_candidate["doi"]
        doi_source = doi_candidate["doi_source"]

        for scrape_url in candidate_urls(doi):
            result["doi"] = doi
            result["doi_source"] = doi_source
            result["scrape_url"] = scrape_url
            try:
                response = requests.get(
                    scrape_url,
                    headers=request_headers(user_agent),
                    timeout=timeout,
                    allow_redirects=True,
                )
                result["final_url"] = response.url
                result["http_status"] = str(response.status_code)

                if response.status_code >= 400:
                    errors.append(f"{doi_source} {scrape_url}: HTTP {response.status_code}")
                    continue

                metadata = extract_article_metadata(response.text)
                result["title"] = metadata["title"]
                result["authors"] = metadata["authors"]
                result["author_institutions"] = metadata["author_institutions"]
                result["abstract"] = metadata["abstract"]
                result["keywords"] = metadata["keywords"]
                result["jel_codes"] = metadata["jel_codes"]
                result["jel_source"] = metadata["jel_source"]
                result["jel_context"] = metadata["jel_context"]
                result["error"] = " | ".join(errors)

                if has_scraped_information(result):
                    if result["title"] == "":
                        result["title"] = record["title"]
                    return result

                errors.append(f"{doi_source} {scrape_url}: no requested metadata found")
            except requests.RequestException as error:
                errors.append(f"{doi_source} {scrape_url}: {type(error).__name__}: {error}")

    result["error"] = " | ".join(errors)
    if result["title"] == "":
        result["title"] = record["title"]
    return result


def row_has_scraped_information(row: pd.Series) -> bool:
    requested_columns = [
        "scrape_authors",
        "scrape_author_institutions",
        "scrape_abstract",
        "scrape_keywords",
        "scrape_title",
        "scrape_jel_codes",
    ]
    return any(clean_text(row.get(column, "")) for column in requested_columns)


def has_scraped_information(result: dict[str, str]) -> bool:
    requested_fields = [
        "authors",
        "author_institutions",
        "abstract",
        "keywords",
        "title",
        "jel_codes",
    ]
    return any(clean_text(result.get(field, "")) for field in requested_fields)


def add_scrape_prefix(row: dict[str, str]) -> dict[str, str]:
    return {
        prefixed_scrape_column(column): row.get(column, "")
        for column in OUTPUT_COLUMNS_BASE
    }


def prefixed_scrape_column(column: str) -> str:
    if column.startswith(SCRAPE_OUTPUT_PREFIX):
        return column
    return f"{SCRAPE_OUTPUT_PREFIX}{column}"


def candidate_urls(doi: str) -> list[str]:
    doi = normalize_doi(doi)
    candidates = [f"https://doi.org/{doi}"]

    if doi.startswith("10.1257/"):
        candidates.append(f"https://www.aeaweb.org/articles?id={doi}")
    elif doi.startswith("10.1086/"):
        candidates.append(f"https://www.journals.uchicago.edu/doi/{doi}")
    elif doi.startswith("10.1093/qje/"):
        candidates.append(f"https://academic.oup.com/qje/article-lookup/doi/{doi}")
    elif doi.startswith("10.1093/restud/"):
        candidates.append(f"https://academic.oup.com/restud/article-lookup/doi/{doi}")

    return list(dict.fromkeys(candidates))


def request_headers(user_agent: str) -> dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }


def extract_article_metadata(html: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = extract_title_from_html(soup)
    authors = extract_authors_from_html(soup)
    author_institutions = extract_author_institutions_from_html(soup)
    abstract = extract_abstract_from_html(soup)
    keywords = extract_keywords_from_html(soup)

    meta_text = metadata_text(soup)
    meta_codes, meta_context = extract_jel_from_text(meta_text)
    if meta_codes:
        return {
            "title": title,
            "authors": authors,
            "author_institutions": author_institutions,
            "abstract": abstract,
            "keywords": keywords,
            "jel_codes": "; ".join(meta_codes),
            "jel_source": "metadata",
            "jel_context": meta_context,
        }

    page_text = visible_page_text(soup)
    text_codes, text_context = extract_jel_from_text(page_text)
    if text_codes:
        return {
            "title": title,
            "authors": authors,
            "author_institutions": author_institutions,
            "abstract": abstract,
            "keywords": keywords,
            "jel_codes": "; ".join(text_codes),
            "jel_source": "page_text",
            "jel_context": text_context,
        }

    return {
        "title": title,
        "authors": authors,
        "author_institutions": author_institutions,
        "abstract": abstract,
        "keywords": keywords,
        "jel_codes": "",
        "jel_source": "",
        "jel_context": "",
    }


def extract_title_from_html(soup: BeautifulSoup) -> str:
    title = meta_content(
        soup,
        "citation_title",
        "dc.Title",
        "DC.Title",
        "og:title",
        "twitter:title",
    )
    if title:
        return clean_text(title)

    heading = soup.select_one("h1")
    if heading is not None:
        title = clean_text(heading.get_text(" ", strip=True))
        if title:
            return title

    if soup.title is not None:
        return clean_text(soup.title.get_text(" ", strip=True))

    return ""


def extract_authors_from_html(soup: BeautifulSoup) -> str:
    authors = meta_contents(
        soup,
        "citation_author",
        "dc.Creator",
        "DC.Creator",
        "author",
    )

    selectors = [
        ".al-authors-list .al-author-name",
        ".wi-authors .author-name",
        ".article-header .author-name",
        "[class*=author-name]",
        "[class*=authors] a",
    ]
    for selector in selectors:
        authors.extend(tag.get_text(" ", strip=True) for tag in soup.select(selector))

    return "; ".join(unique_values(authors))


def extract_author_institutions_from_html(soup: BeautifulSoup) -> str:
    institutions = meta_contents(
        soup,
        "citation_author_institution",
        "dc.Contributor",
        "DC.Contributor",
    )

    selectors = [
        ".al-authors-list .al-author-info",
        ".author-affiliation",
        ".affiliation",
        "[class*=affiliation]",
        "[class*=institution]",
    ]
    for selector in selectors:
        institutions.extend(tag.get_text(" ", strip=True) for tag in soup.select(selector))

    return "; ".join(unique_values(institutions))


def extract_abstract_from_html(soup: BeautifulSoup) -> str:
    abstract = meta_content(
        soup,
        "citation_abstract",
        "dc.Description",
        "DC.Description",
        "description",
        "og:description",
    )
    if abstract:
        return clean_abstract(abstract)

    selectors = [
        "section.article-information.abstract",
        "section.abstract",
        "div.abstract",
        "#abstract",
        ".abstract",
        "[class*=abstract]",
    ]
    for selector in selectors:
        tag = soup.select_one(selector)
        if tag is None:
            continue
        abstract = clean_abstract(tag.get_text(" ", strip=True))
        if abstract:
            return abstract

    return ""


def extract_keywords_from_html(soup: BeautifulSoup) -> str:
    keywords = []
    for tag in soup.find_all("meta"):
        name = clean_text(tag.get("name", "") or tag.get("property", "")).casefold()
        if "keyword" in name or name in {"citation_keywords", "dc.subject", "dc.Subject".casefold()}:
            keywords.extend(split_keywords(tag.get("content", "")))

    keyword_sections = [
        ".keywords",
        "section.keywords",
        "section.article-information.keywords",
        "[class*=keyword]",
    ]
    for selector in keyword_sections:
        tag = soup.select_one(selector)
        if tag is not None:
            text = remove_leading_label(tag.get_text(" ", strip=True), ["Keywords", "Keyword"])
            keywords.extend(split_keywords(text))

    return "; ".join(unique_values(keywords))


def meta_content(soup: BeautifulSoup, *names: str) -> str:
    name_set = {name.casefold() for name in names}
    for tag in soup.find_all("meta"):
        name = clean_text(tag.get("name", "") or tag.get("property", "")).casefold()
        if name in name_set:
            content = clean_text(tag.get("content", ""))
            if content:
                return content
    return ""


def meta_contents(soup: BeautifulSoup, *names: str) -> list[str]:
    name_set = {name.casefold() for name in names}
    values = []
    for tag in soup.find_all("meta"):
        name = clean_text(tag.get("name", "") or tag.get("property", "")).casefold()
        if name in name_set:
            content = clean_text(tag.get("content", ""))
            if content:
                values.append(content)
    return values


def metadata_text(soup: BeautifulSoup) -> str:
    pieces = []
    for tag in soup.find_all("meta"):
        name = clean_text(tag.get("name", "") or tag.get("property", ""))
        content = clean_text(tag.get("content", ""))
        if content:
            pieces.append(f"{name}: {content}")
    return " ".join(pieces)


def visible_page_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return clean_text(soup.get_text(" "))


def clean_abstract(value: Any) -> str:
    text = clean_text(value)
    text = remove_leading_label(text, ["Abstract"])
    return clean_text(text)


def remove_leading_label(text: str, labels: list[str]) -> str:
    text = clean_text(text)
    for label in labels:
        text = re.sub(rf"^{re.escape(label)}\s*:?\s*", "", text, flags=re.IGNORECASE)
    return clean_text(text)


def split_keywords(value: Any) -> list[str]:
    return [
        clean_text(keyword)
        for keyword in re.split(r";|,", str(value or ""))
        if clean_text(keyword)
    ]


def unique_values(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        value = clean_text(value)
        if value and value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def extract_jel_from_text(text: str) -> tuple[list[str], str]:
    text = clean_text(text)
    if text == "":
        return [], ""

    for match in JEL_CONTEXT_PATTERN.finditer(text):
        context = context_window(text, match.start(), match.end(), size=500)
        codes = unique_codes(JEL_CODE_PATTERN.findall(context.upper()))
        if codes:
            return codes, context

    jel_position = text.lower().find("jel")
    if jel_position != -1:
        window = text[jel_position : jel_position + 500]
        codes = unique_codes(JEL_CODE_PATTERN.findall(window.upper()))
        if codes:
            return codes, clean_text(window)

    return [], ""


def unique_codes(codes: list[str]) -> list[str]:
    seen = set()
    unique = []
    for code in codes:
        if code not in seen:
            unique.append(code)
            seen.add(code)
    return unique


def context_window(text: str, start: int, end: int, size: int = 160) -> str:
    left = max(0, start - size)
    right = min(len(text), end + size)
    return clean_text(text[left:right])


def normalize_doi(value: str) -> str:
    value = clean_text(value).lower()
    return (
        value
        .replace("https://doi.org/", "")
        .replace("http://doi.org/", "")
        .replace("http://dx.doi.org/", "")
    )


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


if __name__ == "__main__":
    main()
