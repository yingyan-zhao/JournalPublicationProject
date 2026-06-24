from pathlib import Path
import argparse
import csv
from datetime import datetime
import os
import re
import time
from typing import Any

from bs4 import BeautifulSoup
import pandas as pd
import requests


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

INPUT_CSV = Path("data/processed/openalex_crossref_repec_merged/OpenAlex_Crossref_RePEc_All.csv")
OUTPUT_CSV = Path("data/processed/webscrape_jel/WebScrape_JEL.csv")
URL_COLUMN = "URL"

JEL_CODE_PATTERN = re.compile(r"\b[A-Z][0-9]{2}\b")
JEL_CONTEXT_PATTERN = re.compile(
    r"(?:JEL|JEL\s+classification|JEL\s+classifications|JEL\s+codes?)"
    r"[^A-Za-z0-9]{0,20}"
    r"([A-Z][0-9]{2}(?:\s*[,;/]\s*[A-Z][0-9]{2})*)",
    flags=re.IGNORECASE,
)


def main() -> None:
    args = parse_args()
    records = read_url_records(args.input_csv, args.url_column)
    already_scraped = read_already_scraped_urls(args.output_csv)

    if args.overwrite:
        already_scraped = set()

    records_to_scrape = [
        record
        for record in records
        if record["url"] not in already_scraped
    ]
    if args.limit is not None:
        records_to_scrape = records_to_scrape[: args.limit]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = args.overwrite or not args.output_csv.exists()

    print(f"Found {len(records)} unique nonblank URLs in {args.input_csv}.")
    print(f"Already scraped URLs: {len(already_scraped)}.")
    print(f"URLs to scrape in this run: {len(records_to_scrape)}.")

    with args.output_csv.open(
        "w" if args.overwrite else "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()

        for index, record in enumerate(records_to_scrape, start=1):
            result = scrape_one_url(
                record=record,
                timeout=args.timeout,
                user_agent=args.user_agent,
            )
            writer.writerow(result)
            file.flush()

            print(
                f"{index}/{len(records_to_scrape)} "
                f"status={result['http_status']} "
                f"jel={result['jel_codes']} "
                f"url={record['url']}"
            )
            time.sleep(args.sleep)

    print(f"Wrote scrape results to {args.output_csv}.")


OUTPUT_COLUMNS = [
    "url",
    "doi",
    "title",
    "publication_year",
    "final_url",
    "http_status",
    "jel_codes",
    "jel_source",
    "jel_context",
    "scraped_at",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape DOI landing pages and extract JEL codes."
    )
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--url-column", default=URL_COLUMN)
    parser.add_argument("--limit", type=int, help="Scrape only the first N unscripted URLs.")
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=30.0)
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
        help="Overwrite the output file instead of resuming/appending.",
    )
    return parser.parse_args()


def read_url_records(input_csv: Path, url_column: str) -> list[dict[str, str]]:
    data = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    if url_column not in data.columns:
        raise ValueError(f"{input_csv} does not contain column {url_column!r}.")

    seen_urls = set()
    records = []
    for _, row in data.iterrows():
        url = clean_text(row.get(url_column, ""))
        if url == "" or url in seen_urls:
            continue

        records.append(
            {
                "url": url,
                "doi": normalize_doi(row.get("merge_doi", "")) or normalize_doi(url),
                "title": first_nonblank(
                    row,
                    ["openalex_title", "crossref_title", "repec_title"],
                ),
                "publication_year": first_nonblank(
                    row,
                    ["openalex_publication_year", "published_year", "repec_year"],
                ),
            }
        )
        seen_urls.add(url)

    return records


def read_already_scraped_urls(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()

    try:
        data = pd.read_csv(output_csv, usecols=["url"], dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return set()

    return set(data["url"].apply(clean_text).loc[lambda values: values != ""])


def scrape_one_url(record: dict[str, str], timeout: float, user_agent: str) -> dict[str, str]:
    scraped_at = datetime.now().isoformat(timespec="seconds")
    result = {
        "url": record["url"],
        "doi": record["doi"],
        "title": record["title"],
        "publication_year": record["publication_year"],
        "final_url": "",
        "http_status": "",
        "jel_codes": "",
        "jel_source": "",
        "jel_context": "",
        "scraped_at": scraped_at,
        "error": "",
    }

    errors = []
    for scrape_url in candidate_urls(record["url"], record["doi"]):
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
                errors.append(f"{scrape_url}: HTTP {response.status_code}")
                continue

            jel_codes, jel_source, jel_context = extract_jel_from_html(response.text)
            result["jel_codes"] = "; ".join(jel_codes)
            result["jel_source"] = jel_source
            result["jel_context"] = jel_context
            result["error"] = " | ".join(errors)
            return result
        except requests.RequestException as error:
            errors.append(f"{scrape_url}: {type(error).__name__}: {error}")

    result["error"] = " | ".join(errors)

    return result


def candidate_urls(url: str, doi: str) -> list[str]:
    candidates = [url]
    doi = normalize_doi(doi or url)

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


def extract_jel_from_html(html: str) -> tuple[list[str], str, str]:
    soup = BeautifulSoup(html, "html.parser")

    meta_text = metadata_text(soup)
    meta_codes, meta_context = extract_jel_from_text(meta_text)
    if meta_codes:
        return meta_codes, "metadata", meta_context

    page_text = visible_page_text(soup)
    text_codes, text_context = extract_jel_from_text(page_text)
    if text_codes:
        return text_codes, "page_text", text_context

    return [], "", ""


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


def first_nonblank(row: pd.Series, columns: list[str]) -> str:
    for column in columns:
        if column in row:
            value = clean_text(row.get(column, ""))
            if value != "":
                return value
    return ""


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


if __name__ == "__main__":
    main()
