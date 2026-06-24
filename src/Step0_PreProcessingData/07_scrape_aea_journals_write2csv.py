import argparse
import csv
from datetime import date, datetime
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup
import pandas as pd
import requests


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

BASE_URL = "https://www.aeaweb.org"
OUTPUT_CSV = Path("data/raw_csv/AEA_Journals_Papers.csv")

AEA_JOURNALS = {
    "aer": {
        "journal": "American Economic Review",
        "issues_url": "https://www.aeaweb.org/journals/aer/issues",
    },
    "aeri": {
        "journal": "American Economic Review: Insights",
        "issues_url": "https://www.aeaweb.org/journals/aeri/issues",
    },
    "app": {
        "journal": "American Economic Journal: Applied Economics",
        "issues_url": "https://www.aeaweb.org/journals/app/issues",
    },
    "pol": {
        "journal": "American Economic Journal: Economic Policy",
        "issues_url": "https://www.aeaweb.org/journals/pol/issues",
    },
    "mac": {
        "journal": "American Economic Journal: Macroeconomics",
        "issues_url": "https://www.aeaweb.org/journals/mac/issues",
    },
    "mic": {
        "journal": "American Economic Journal: Microeconomics",
        "issues_url": "https://www.aeaweb.org/journals/mic/issues",
    },
    "pandp": {
        "journal": "AEA Papers and Proceedings",
        "issues_url": "https://www.aeaweb.org/journals/pandp/issues",
    },
}

CSV_COLUMNS = [
    "doi",
    "title",
    "authors",
    "author_institutions",
    "author_institution_pairs",
    "journal",
    "journal_slug",
    "publication_year",
    "publication_month",
    "volume",
    "issue",
    "pages",
    "abstract",
    "keywords",
    "jel_codes",
    "jel_descriptions",
    "article_url",
    "issue_url",
    "issue_label",
    "collection_date",
    "scraped_at",
    "http_status",
    "error",
]

ARTICLE_LINK_PATTERN = re.compile(r"^/articles\?id=")
ISSUE_LINK_PATTERN = re.compile(r"^/issues/\d+$")
DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+", flags=re.IGNORECASE)
JEL_CODE_PATTERN = re.compile(r"\b[A-Z][0-9]{2}\b")
NONPAPER_TITLE_PATTERNS = [
    "front matter",
    "back matter",
    "report of independent auditor",
    "index",
    "erratum",
    "corrigendum",
    "correction",
    "announcement",
    "contents",
]


def main() -> None:
    args = parse_args()
    session = requests.Session()
    session.headers.update(request_headers(args.user_agent))

    journals = selected_journals(args.journals)
    already_scraped = read_already_scraped_dois(args.output_csv)
    if args.overwrite:
        already_scraped = set()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = args.overwrite or not args.output_csv.exists()

    total_issues = 0
    total_article_links = 0
    written_rows = 0
    skipped_existing_doi = 0
    skipped_duplicate_doi = 0
    failed_articles = 0
    seen_dois = set(already_scraped)
    collection_date = date.today().isoformat()

    with args.output_csv.open(
        "w" if args.overwrite else "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()

        for journal_slug, journal_info in journals.items():
            print(f"\nDiscovering issues for {journal_info['journal']}...")
            issues = discover_issue_links(
                session=session,
                journal_slug=journal_slug,
                journal_name=journal_info["journal"],
                issues_url=journal_info["issues_url"],
                from_year=args.from_year,
                to_year=args.to_year,
                timeout=args.timeout,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
            )
            if args.limit_issues is not None:
                issues = issues[: args.limit_issues]
            total_issues += len(issues)
            print(f"Found {len(issues)} issues for {journal_info['journal']}.")

            for issue_index, issue in enumerate(issues, start=1):
                article_links = discover_article_links(
                    session=session,
                    issue=issue,
                    timeout=args.timeout,
                    retries=args.retries,
                    retry_sleep=args.retry_sleep,
                )
                total_article_links += len(article_links)
                print(
                    f"  Issue {issue_index}/{len(issues)}: "
                    f"{issue['issue_label']} has {len(article_links)} article links."
                )

                for article in article_links:
                    if args.limit_articles is not None and written_rows >= args.limit_articles:
                        break

                    doi = normalize_doi(article.get("doi", ""))
                    if doi and doi in seen_dois:
                        if doi in already_scraped:
                            skipped_existing_doi += 1
                        else:
                            skipped_duplicate_doi += 1
                        continue

                    row = scrape_article(
                        session=session,
                        article=article,
                        issue=issue,
                        collection_date=collection_date,
                        timeout=args.timeout,
                        retries=args.retries,
                        retry_sleep=args.retry_sleep,
                    )
                    row_doi = normalize_doi(row.get("doi", ""))
                    if row_doi and row_doi in seen_dois:
                        skipped_duplicate_doi += 1
                        continue
                    if row_doi:
                        seen_dois.add(row_doi)
                    if row["error"]:
                        failed_articles += 1

                    writer.writerow(row)
                    file.flush()
                    written_rows += 1
                    time.sleep(args.sleep)

                if args.limit_articles is not None and written_rows >= args.limit_articles:
                    break
                time.sleep(args.sleep)

            if args.limit_articles is not None and written_rows >= args.limit_articles:
                break

    print("\nAEA scrape summary:")
    print(f"  Issues visited: {total_issues}")
    print(f"  Article links found: {total_article_links}")
    print(f"  Rows written: {written_rows}")
    print(f"  Skipped already-scraped DOIs: {skipped_existing_doi}")
    print(f"  Skipped duplicate DOIs in this run: {skipped_duplicate_doi}")
    print(f"  Rows with scrape errors: {failed_articles}")
    print(f"  Output CSV: {args.output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape AEA journal issue/article pages and write paper-level metadata to CSV."
    )
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument(
        "--journals",
        nargs="+",
        default=list(AEA_JOURNALS),
        choices=list(AEA_JOURNALS),
        help="Journal slugs to scrape.",
    )
    parser.add_argument("--from-year", type=int, help="Keep issues from this year or later.")
    parser.add_argument("--to-year", type=int, help="Keep issues from this year or earlier.")
    parser.add_argument("--limit-issues", type=int, help="Limit issues per journal for testing.")
    parser.add_argument("--limit-articles", type=int, help="Limit total article rows written for testing.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to wait between requests.")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3, help="Number of tries for each web request.")
    parser.add_argument("--retry-sleep", type=float, default=3.0, help="Seconds to wait between failed tries.")
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
        help="Overwrite output instead of appending/resuming by DOI.",
    )
    return parser.parse_args()


def selected_journals(journal_slugs: list[str]) -> dict[str, dict[str, str]]:
    return {slug: AEA_JOURNALS[slug] for slug in journal_slugs}


def discover_issue_links(
    session: requests.Session,
    journal_slug: str,
    journal_name: str,
    issues_url: str,
    from_year: int | None,
    to_year: int | None,
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> list[dict[str, str]]:
    try:
        response = request_with_retries(
            session=session,
            url=issues_url,
            timeout=timeout,
            retries=retries,
            retry_sleep=retry_sleep,
        )
    except requests.RequestException as error:
        print(f"  Could not discover issues for {journal_name}: {type(error).__name__}: {error}")
        return []
    soup = BeautifulSoup(response.text, "html.parser")

    issues = []
    seen_urls = set()
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if not ISSUE_LINK_PATTERN.match(href):
            continue

        issue_url = urljoin(BASE_URL, href)
        if issue_url in seen_urls:
            continue

        issue_label = clean_text(link.get_text(" ", strip=True))
        issue_year = year_from_text(issue_label)
        if from_year is not None and issue_year is not None and issue_year < from_year:
            continue
        if to_year is not None and issue_year is not None and issue_year > to_year:
            continue

        seen_urls.add(issue_url)
        issues.append(
            {
                "journal": journal_name,
                "journal_slug": journal_slug,
                "issue_url": issue_url,
                "issue_label": issue_label,
                "issue_year": str(issue_year or ""),
            }
        )

    return issues


def discover_article_links(
    session: requests.Session,
    issue: dict[str, str],
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> list[dict[str, str]]:
    try:
        response = request_with_retries(
            session=session,
            url=issue["issue_url"],
            timeout=timeout,
            retries=retries,
            retry_sleep=retry_sleep,
        )
    except requests.RequestException as error:
        print(
            f"  Could not fetch issue {issue.get('issue_label', '')}: "
            f"{type(error).__name__}: {error}"
        )
        return []
    soup = BeautifulSoup(response.text, "html.parser")

    articles = []
    seen_urls = set()
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if not ARTICLE_LINK_PATTERN.match(href):
            continue

        article_url = urljoin(BASE_URL, href)
        if article_url in seen_urls:
            continue

        title = clean_text(link.get_text(" ", strip=True))
        if is_nonpaper_title(title):
            continue

        doi = normalize_doi(doi_from_article_url(article_url))
        seen_urls.add(article_url)
        articles.append(
            {
                "article_url": article_url,
                "title_from_issue": title,
                "doi": doi,
            }
        )

    return articles


def scrape_article(
    session: requests.Session,
    article: dict[str, str],
    issue: dict[str, str],
    collection_date: str,
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    scraped_at = datetime.now().isoformat(timespec="seconds")
    base_row = {
        "doi": article.get("doi", ""),
        "title": article.get("title_from_issue", ""),
        "authors": "",
        "author_institutions": "",
        "author_institution_pairs": "",
        "journal": issue["journal"],
        "journal_slug": issue["journal_slug"],
        "publication_year": issue.get("issue_year", ""),
        "publication_month": month_from_text(issue.get("issue_label", "")),
        "volume": "",
        "issue": "",
        "pages": "",
        "abstract": "",
        "keywords": "",
        "jel_codes": "",
        "jel_descriptions": "",
        "article_url": article["article_url"],
        "issue_url": issue["issue_url"],
        "issue_label": issue["issue_label"],
        "collection_date": collection_date,
        "scraped_at": scraped_at,
        "http_status": "",
        "error": "",
    }

    try:
        response = request_with_retries(
            session=session,
            url=article["article_url"],
            timeout=timeout,
            retries=retries,
            retry_sleep=retry_sleep,
        )
        base_row["http_status"] = str(response.status_code)
    except requests.RequestException as error:
        base_row["error"] = f"{type(error).__name__}: {error}"
        return base_row

    soup = BeautifulSoup(response.text, "html.parser")
    title = clean_text(text_from_first(soup, "h1.title"))
    authors = "; ".join(
        clean_text(author.get_text(" ", strip=True))
        for author in soup.select("ul.attribution li.author")
        if clean_text(author.get_text(" ", strip=True))
    )
    author_institutions, author_institution_pairs = parse_author_institutions(soup, authors)
    abstract = section_text(soup, "section.article-information.abstract", "Abstract")
    keywords = parse_keywords(soup)
    citation = section_text(soup, "section.article-information.citation", "Citation")
    jel_codes, jel_descriptions = parse_jel_classification(soup)
    citation_fields = parse_citation(citation)
    doi = normalize_doi(citation_fields.get("doi", "") or article.get("doi", ""))

    base_row.update(
        {
            "doi": doi,
            "title": title or article.get("title_from_issue", ""),
            "authors": authors,
            "author_institutions": author_institutions,
            "author_institution_pairs": author_institution_pairs,
            "abstract": abstract,
            "keywords": keywords,
            "jel_codes": "; ".join(jel_codes),
            "jel_descriptions": "; ".join(jel_descriptions),
        }
    )
    for field in ["publication_year", "volume", "issue", "pages"]:
        if citation_fields.get(field):
            base_row[field] = citation_fields[field]

    return base_row


def parse_keywords(soup: BeautifulSoup) -> str:
    keyword_values = []
    for tag in soup.find_all("meta"):
        name = clean_text(tag.get("name", ""))
        if name != "keywords":
            continue
        content = clean_text(tag.get("content", ""))
        if content:
            keyword_values.extend(split_keywords(content))
    return "; ".join(unique_values(keyword_values))


def split_keywords(value: str) -> list[str]:
    return [
        clean_text(keyword)
        for keyword in value.split(",")
        if clean_text(keyword)
    ]


def parse_author_institutions(soup: BeautifulSoup, visible_authors: str) -> tuple[str, str]:
    metadata_authors = []
    institutions = []
    for tag in soup.find_all("meta"):
        name = clean_text(tag.get("name", ""))
        content = clean_text(tag.get("content", ""))
        if name == "citation_author":
            metadata_authors.append(content)
        elif name == "citation_author_institution":
            institutions.append(content)

    visible_author_list = [author for author in visible_authors.split("; ") if author]
    pair_authors = visible_author_list
    if len(pair_authors) != len(institutions) and metadata_authors:
        pair_authors = metadata_authors

    pairs = []
    for index, institution in enumerate(institutions):
        author = pair_authors[index] if index < len(pair_authors) else ""
        if author and institution:
            pairs.append(f"{author}: {institution}")
        elif institution:
            pairs.append(institution)

    return "; ".join(institutions), "; ".join(pairs)


def parse_jel_classification(soup: BeautifulSoup) -> tuple[list[str], list[str]]:
    section = soup.select_one("section.article-information.jel-classification")
    if section is None:
        return [], []

    codes = []
    descriptions = []
    code_tags = section.select("strong.code")

    for code_tag in code_tags:
        code = clean_text(code_tag.get_text(" ", strip=True))
        if not JEL_CODE_PATTERN.fullmatch(code):
            continue
        description = clean_text_after_code(code_tag)
        codes.append(code)
        descriptions.append(f"{code} {description}".strip())

    return unique_values(codes), unique_values(descriptions)


def clean_text_after_code(code_tag) -> str:
    pieces = []
    for sibling in code_tag.next_siblings:
        if getattr(sibling, "name", None) == "li":
            break
        pieces.append(clean_text(getattr(sibling, "get_text", lambda *args, **kwargs: str(sibling))(" ", strip=True)))
    return clean_text(" ".join(pieces))


def parse_citation(citation: str) -> dict[str, str]:
    fields = {}

    doi_match = DOI_PATTERN.search(citation)
    if doi_match:
        fields["doi"] = normalize_doi(doi_match.group(0).rstrip("."))

    year_match = re.search(r"\b(19|20)\d{2}\b", citation)
    if year_match:
        fields["publication_year"] = year_match.group(0)

    volume_issue_pages = re.search(
        r"\s(?P<volume>\d+)\s+\((?P<issue>[^)]+)\):\s+(?P<pages>[^.]+)\.\s+DOI:",
        citation,
    )
    if volume_issue_pages:
        fields["volume"] = clean_text(volume_issue_pages.group("volume"))
        fields["issue"] = clean_text(volume_issue_pages.group("issue"))
        fields["pages"] = clean_text(volume_issue_pages.group("pages"))

    return fields


def text_from_first(soup: BeautifulSoup, selector: str) -> str:
    tag = soup.select_one(selector)
    if tag is None:
        return ""
    return clean_text(tag.get_text(" ", strip=True))


def section_text(soup: BeautifulSoup, selector: str, heading: str) -> str:
    section = soup.select_one(selector)
    if section is None:
        return ""

    text = clean_text(section.get_text(" ", strip=True))
    if text.lower().startswith(heading.lower()):
        text = clean_text(text[len(heading) :])
    if heading.lower() == "citation":
        text = clean_text(text.split("Choose Format:", 1)[0])
    return text


def is_nonpaper_title(title: str) -> bool:
    normalized_title = clean_text(title).casefold()
    return any(pattern in normalized_title for pattern in NONPAPER_TITLE_PATTERNS)


def doi_from_article_url(article_url: str) -> str:
    query = parse_qs(urlparse(article_url).query)
    doi_values = query.get("id") or []
    if not doi_values:
        return ""
    return normalize_doi(doi_values[0])


def read_already_scraped_dois(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()
    try:
        data = pd.read_csv(output_csv, usecols=["doi"], dtype=str, keep_default_na=False)
    except (pd.errors.EmptyDataError, ValueError):
        return set()
    return set(data["doi"].apply(normalize_doi).loc[lambda values: values != ""])


def request_with_retries(
    session: requests.Session,
    url: str,
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> requests.Response:
    last_error = None
    attempts = max(1, retries)
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            if attempt == attempts:
                break
            print(
                f"  Request failed ({attempt}/{attempts}) for {url}: "
                f"{type(error).__name__}. Retrying in {retry_sleep:g}s."
            )
            time.sleep(retry_sleep)

    if last_error is not None:
        raise last_error
    raise requests.RequestException(f"Request failed for {url}")


def request_headers(user_agent: str) -> dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.aeaweb.org/journals",
    }


def normalize_doi(doi: Any) -> str:
    if doi is None:
        return ""
    return (
        str(doi)
        .strip()
        .lower()
        .replace("https://doi.org/", "")
        .replace("http://doi.org/", "")
        .replace("http://dx.doi.org/", "")
        .rstrip(".,;")
    )


def year_from_text(text: str) -> int | None:
    matches = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    if not matches:
        return None
    return int(matches[-1])


def month_from_text(text: str) -> str:
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    for month in months:
        if re.search(rf"\b{month}\b", text, flags=re.IGNORECASE):
            return month
    return ""


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def unique_values(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        value = clean_text(value)
        if value and value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


if __name__ == "__main__":
    main()
