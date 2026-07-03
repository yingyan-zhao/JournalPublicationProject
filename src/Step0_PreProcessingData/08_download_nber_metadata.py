import argparse
import csv
from datetime import date
import gzip
import os
from pathlib import Path
import re
from typing import Any

import pandas as pd
import requests


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

NBER_TSV_BASE_URL = "https://data.nber.org/nber_paper_chapter_metadata/tsv"
NBER_CSV_BASE_URL = "https://data.nber.org/nber_paper_chapter_metadata/csv"
RAW_DIR = Path("data/raw/nber_metadata")
OUTPUT_CSV = Path("data/raw_csv/NBER_Working_Papers_Metadata_After1995.csv")
FROM_YEAR = 1950

TSV_FILES = {
    "ref": "ref.tsv",
    "abstract": "abs.tsv",
    "authors": "auths.tsv",
    "jel": "jel.tsv",
    "published": "published.tsv",
    "program": "prog.tsv",
}
AUTHOR_PRINTER_VIEW_FILE = "working_papers_authors_printer_view.csv.gz"
AUTHOR_PRINTER_SCHEMA_FILE = "working_papers_authors_printer_view.schema.csv.gz"

OUTPUT_COLUMNS = [
    "paper",
    "doi",
    "title",
    "abstract",
    "authors",
    "author_institutions",
    "jel_codes",
    "keywords",
    "issue_date",
    "published_text",
    "collection_date",
]

ADDRESS_COLUMNS = [
    "ADDR1",
    "ADDR2",
    "ADDR3",
    "ADDR4",
    "ADDR5",
    "ADDR6",
    "City",
    "State",
    "Zip",
    "Country",
]


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading NBER metadata files...")
    tsv_paths = {
        name: download_file(
            url=f"{NBER_TSV_BASE_URL}/{filename}",
            output_path=raw_dir / filename,
            overwrite=args.overwrite,
            timeout=args.timeout,
        )
        for name, filename in TSV_FILES.items()
    }
    author_path = download_file(
        url=f"{NBER_CSV_BASE_URL}/{AUTHOR_PRINTER_VIEW_FILE}",
        output_path=raw_dir / AUTHOR_PRINTER_VIEW_FILE,
        overwrite=args.overwrite,
        timeout=args.timeout,
    )
    author_schema_path = download_file(
        url=f"{NBER_CSV_BASE_URL}/{AUTHOR_PRINTER_SCHEMA_FILE}",
        output_path=raw_dir / AUTHOR_PRINTER_SCHEMA_FILE,
        overwrite=args.overwrite,
        timeout=args.timeout,
    )

    print("Reading and merging metadata...")
    data = build_compiled_metadata(
        ref_path=tsv_paths["ref"],
        abstract_path=tsv_paths["abstract"],
        authors_path=tsv_paths["authors"],
        jel_path=tsv_paths["jel"],
        published_path=tsv_paths["published"],
        program_path=tsv_paths["program"],
        author_path=author_path,
        author_schema_path=author_schema_path,
        from_year=args.from_year,
        include_chapters=args.include_chapters,
    )

    data["collection_date"] = date.today().isoformat()
    data = data[OUTPUT_COLUMNS].copy()
    data.to_csv(args.output_csv, index=False, encoding="utf-8")

    print("\nNBER metadata summary:")
    print(f"  Rows written: {len(data)}")
    print(f"  First issue year kept: {args.from_year}")
    print(f"  Papers with DOI: {count_nonblank(data, 'doi')}")
    print(f"  Papers with abstract: {count_nonblank(data, 'abstract')}")
    print(f"  Papers with authors: {count_nonblank(data, 'authors')}")
    print(f"  Papers with author institutions/address: {count_nonblank(data, 'author_institutions')}")
    print(f"  Papers with JEL codes: {count_nonblank(data, 'jel_codes')}")
    print(f"  Papers with keywords: {count_nonblank(data, 'keywords')}")
    print(f"  Output CSV: {args.output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download official NBER working paper metadata and compile paper-level "
            "fields after a chosen issue year."
        )
    )
    parser.add_argument("--from-year", type=int, default=FROM_YEAR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-chapters",
        action="store_true",
        help="Also keep NBER book chapters. By default, only working paper IDs beginning with 'w' are kept.",
    )
    return parser.parse_args()


def download_file(url: str, output_path: Path, overwrite: bool, timeout: float) -> Path:
    if output_path.exists() and not overwrite:
        print(f"  Using existing file: {output_path}")
        return output_path

    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "JournalPublicationProject/1.0"},
    )
    response.raise_for_status()
    output_path.write_bytes(response.content)
    print(f"  Downloaded {url} -> {output_path} ({len(response.content):,} bytes)")
    return output_path


def build_compiled_metadata(
    ref_path: Path,
    abstract_path: Path,
    authors_path: Path,
    jel_path: Path,
    published_path: Path,
    program_path: Path,
    author_path: Path,
    author_schema_path: Path,
    from_year: int,
    include_chapters: bool,
) -> pd.DataFrame:
    ref = read_tsv(ref_path)
    abstracts = read_tsv(abstract_path)
    published = aggregate_by_paper(read_tsv(published_path), "published_text", "published_text")
    authors = aggregate_by_paper(read_tsv(authors_path), "name", "authors")
    jel = aggregate_by_paper(read_tsv(jel_path), "jel", "jel_codes")
    programs = aggregate_by_paper(read_tsv(program_path), "program", "keywords")
    author_institutions = author_institution_summary(author_path, author_schema_path)

    data = ref.merge(abstracts, on="paper", how="left")
    data = data.merge(authors, on="paper", how="left")
    data = data.merge(author_institutions, on="paper", how="left")
    data = data.merge(jel, on="paper", how="left")
    data = data.merge(programs, on="paper", how="left")
    data = data.merge(published, on="paper", how="left")

    data["issue_date"] = pd.to_datetime(data["issue_date"], errors="coerce")
    data = data.loc[data["issue_date"].dt.year >= from_year].copy()
    if not include_chapters:
        data = data.loc[data["paper"].fillna("").astype(str).str.match(r"^w\d+$")].copy()
    data["issue_date"] = data["issue_date"].dt.strftime("%Y-%m-%d").fillna("")

    for column in ["doi", "title", "abstract", "authors", "author_institutions", "jel_codes", "keywords", "published_text"]:
        if column not in data.columns:
            data[column] = ""
        data[column] = data[column].fillna("").apply(clean_text).replace({"NULL": "", "nan": ""})

    if "author" in data.columns:
        data["author"] = data["author"].fillna("").apply(clean_text)
        data.loc[data["authors"] == "", "authors"] = data.loc[data["authors"] == "", "author"]

    data["doi"] = data["doi"].apply(normalize_doi)
    data.loc[data["doi"] == "", "doi"] = data.loc[data["doi"] == "", "paper"].apply(nber_doi_from_paper)
    data = data.sort_values(["issue_date", "paper"]).reset_index(drop=True)
    return data


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="latin1",
        engine="python",
        quoting=csv.QUOTE_NONE,
    )


def aggregate_by_paper(data: pd.DataFrame, value_column: str, output_column: str) -> pd.DataFrame:
    if data.empty or value_column not in data.columns:
        return pd.DataFrame(columns=["paper", output_column])

    selected = data[["paper", value_column]].copy()
    selected[value_column] = selected[value_column].apply(clean_text).replace("NULL", "")
    selected = selected.loc[selected[value_column] != ""].copy()
    return (
        selected.groupby("paper", as_index=False)[value_column]
        .agg(lambda values: "; ".join(unique_values(values)))
        .rename(columns={value_column: output_column})
    )


def author_institution_summary(author_path: Path, schema_path: Path) -> pd.DataFrame:
    schema_columns = read_nber_schema_columns(schema_path)
    authors = pd.read_csv(
        author_path,
        compression="gzip",
        sep="¿",
        quotechar="¬",
        names=schema_columns,
        header=None,
        dtype=str,
        keep_default_na=False,
        encoding="latin1",
        engine="python",
    )

    authors["order_num_numeric"] = pd.to_numeric(authors.get("order_num", ""), errors="coerce")
    authors = authors.sort_values(["paper", "order_num_numeric", "name"]).copy()
    authors["name"] = authors["name"].apply(clean_text)
    authors["institution_address"] = authors.apply(author_address, axis=1)
    authors["author_institution_pair"] = authors.apply(author_institution_pair, axis=1)

    summary = (
        authors.groupby("paper", as_index=False)
        .agg(
            author_institutions=("author_institution_pair", lambda values: "; ".join(unique_values(values))),
        )
    )
    return summary


def read_nber_schema_columns(schema_path: Path) -> list[str]:
    with gzip.open(schema_path, "rt", encoding="latin1", newline="") as file:
        reader = csv.reader(file, delimiter="¿", quotechar="¬")
        return [row[0] for row in reader if row]


def author_address(row: pd.Series) -> str:
    pieces = []
    for column in ADDRESS_COLUMNS:
        value = clean_text(row.get(column, ""))
        if value and value != "\\N":
            pieces.append(value)
    return "; ".join(unique_values(pieces))


def author_institution_pair(row: pd.Series) -> str:
    name = clean_text(row.get("name", ""))
    address = clean_text(row.get("institution_address", ""))
    if name and address:
        return f"{name}: {address}"
    return address


def normalize_doi(value: Any) -> str:
    text = clean_text(value)
    if text == "NULL":
        return ""
    return (
        text.lower()
        .replace("https://doi.org/", "")
        .replace("http://doi.org/", "")
        .replace("http://dx.doi.org/", "")
        .rstrip(".,;")
    )


def nber_doi_from_paper(paper: Any) -> str:
    paper = clean_text(paper).lower()
    if re.fullmatch(r"w\d+", paper):
        return f"10.3386/{paper}"
    return ""


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text in {"\\N", "NULL", "nan", "NaN", "None"}:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


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
    return int(data[column].fillna("").astype(str).str.strip().ne("").sum())


if __name__ == "__main__":
    main()
