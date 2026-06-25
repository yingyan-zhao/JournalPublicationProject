from pathlib import Path
import os
import re

import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

OPENALEX_CROSSREF_WEBSCRAPED_INPUT_CSV = Path("data/processed/OpenAlex_Crossref_All_Webscraped_Cleaned.csv")
NBER_CLEANED_INPUT_CSV = Path("data/processed/NBER_Working_Papers_Metadata_After1995_Cleaned.csv")
MERGED_OUTPUT_CSV = Path("data/processed/OpenAlex_Crossref_Webscrape_NBER_Merged.csv")
PAPERS_TO_DROP = {"w7565", "w13800", "w21929", "w8649"}
NBER_COLUMNS_TO_DROP_AFTER_MERGE = [
    "nber_paper",
    "nber_doi",
    "nber_title",
    "nber_abstract",
    "nber_issue_date",
    "nber_published_text",
    "nber_collection_date",
    "nber_keywords",
    "nber_jel_codes"
]


def main() -> None:
    openalex_crossref = read_csv(OPENALEX_CROSSREF_WEBSCRAPED_INPUT_CSV)
    nber = read_csv(NBER_CLEANED_INPUT_CSV)
    nber["nber_title"] = nber["nber_title"].apply(keep_letters_and_numbers)
    nber = rename_specific_nber_titles(nber)
    nber_duplicate_title_stats = duplicate_title_stats(nber, "nber_title")

    merged = merge_with_nber_by_title(openalex_crossref, nber)

    MERGED_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print("OpenAlex/Crossref/Webscrape + NBER merge summary:")
    print(f"  Base input CSV: {OPENALEX_CROSSREF_WEBSCRAPED_INPUT_CSV}")
    print(f"  Base rows: {len(openalex_crossref)}")
    print(f"  Cleaned NBER CSV: {NBER_CLEANED_INPUT_CSV}")
    print(f"  Cleaned NBER rows: {len(nber)}")
    print(f"  NBER exact duplicate title rows: {nber_duplicate_title_stats['exact_duplicate_rows']}")
    print(f"  NBER exact duplicate title groups: {nber_duplicate_title_stats['exact_duplicate_groups']}")
    print(f"  NBER normalized duplicate title rows: {nber_duplicate_title_stats['normalized_duplicate_rows']}")
    print(f"  NBER normalized duplicate title groups: {nber_duplicate_title_stats['normalized_duplicate_groups']}")
    print(f"  Output CSV: {MERGED_OUTPUT_CSV}")
    print(f"  Output rows: {len(merged)}")
    print(
        "  Rows with duplicated title before export: "
        f"{count_duplicate_rows(merged, 'title', normalize_title)}"
    )
    print(
        "  Rows with duplicated doi_list before export: "
        f"{count_duplicate_rows(merged, 'doi_list', normalize_doi_list)}"
    )
    print(f"  Rows matched to NBER: {merged.attrs.get('nber_match_count', 0)}")
    filled_counts = merged.attrs.get("nber_filled_counts", {})
    print(f"  Abstracts filled from NBER: {filled_counts.get('abstract', 0)}")
    print(f"  JEL codes filled from NBER: {filled_counts.get('jel_codes', 0)}")
    print(f"  Keywords filled from NBER: {filled_counts.get('keywords', 0)}")

    merged.to_csv(MERGED_OUTPUT_CSV, index=False)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_nber_data(path: Path) -> pd.DataFrame:
    """Read NBER metadata, clean duplicate papers, and add nber_ before every column name."""
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    cleaned = clean_nber_data(data)
    return add_prefix_to_columns(cleaned, "nber_")


def clean_nber_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()

    before_drop_blank_titles = len(cleaned)
    cleaned = cleaned.loc[cleaned["title"].apply(normalize_title) != ""].copy()
    dropped_blank_titles = before_drop_blank_titles - len(cleaned)

    before_drop_papers = len(cleaned)
    cleaned = cleaned.loc[~cleaned["paper"].isin(PAPERS_TO_DROP)].copy()
    dropped_papers = before_drop_papers - len(cleaned)

    cleaned = rename_specific_nber_titles(cleaned)

    cleaned["normalized_title_for_dedup"] = cleaned["title"].apply(normalize_title)
    cleaned["issue_date_for_dedup"] = pd.to_datetime(
        cleaned["issue_date"],
        errors="coerce",
    )

    before_drop_duplicates = len(cleaned)
    blank_title_rows = cleaned.loc[cleaned["normalized_title_for_dedup"] == ""].copy()
    title_rows = cleaned.loc[cleaned["normalized_title_for_dedup"] != ""].copy()
    title_rows = title_rows.sort_values(
        ["normalized_title_for_dedup", "issue_date_for_dedup", "paper"],
        na_position="first",
    )
    title_rows = title_rows.drop_duplicates(
        subset=["normalized_title_for_dedup"],
        keep="last",
    )
    cleaned = pd.concat([title_rows, blank_title_rows], ignore_index=True)
    dropped_duplicate_titles = before_drop_duplicates - len(cleaned)

    cleaned = cleaned.drop(columns=["normalized_title_for_dedup", "issue_date_for_dedup"])
    cleaned = cleaned.sort_values(["issue_date", "paper"]).reset_index(drop=True)

    print("NBER cleaning summary:")
    print(f"  Raw rows: {len(data)}")
    print(f"  Dropped blank-title rows: {dropped_blank_titles}")
    print(f"  Dropped selected papers: {dropped_papers}")
    print(f"  Dropped duplicate-title rows: {dropped_duplicate_titles}")
    print(f"  Cleaned rows before prefix: {len(cleaned)}")
    return cleaned


def add_prefix_to_columns(data: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Return a copy whose columns all start with the requested prefix."""
    renamed = data.copy()
    renamed.columns = [
        column if column.startswith(prefix) else f"{prefix}{column}"
        for column in renamed.columns
    ]
    return renamed


def rename_specific_nber_titles(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    title_column = "nber_title" if "nber_title" in cleaned.columns else "title"
    issue_date_column = "nber_issue_date" if "nber_issue_date" in cleaned.columns else "issue_date"

    if title_column not in cleaned.columns or issue_date_column not in cleaned.columns:
        return cleaned

    title = cleaned[title_column].fillna("").astype(str).str.strip()
    issue_year = cleaned[issue_date_column].fillna("").astype(str).str[:4]
    target_row = (title == "Human Capital and Growth") & (issue_year == "2015")
    cleaned.loc[target_row, title_column] = "Human Capital and Growth 2015"
    return cleaned


def merge_with_nber_by_title(cleaned: pd.DataFrame, nber: pd.DataFrame) -> pd.DataFrame:
    cleaned_for_merge = cleaned.copy()
    nber_for_merge = nber.copy()

    cleaned_for_merge["merge_title"] = cleaned_for_merge["title"].apply(normalize_title)
    nber_for_merge["merge_title"] = nber_for_merge["nber_title"].apply(normalize_title)

    nber_for_merge = nber_for_merge.loc[nber_for_merge["merge_title"] != ""].copy()

    duplicated_nber_titles = int(nber_for_merge["merge_title"].duplicated(keep=False).sum())
    if duplicated_nber_titles:
        print(f"  Duplicated normalized NBER title rows before merge: {duplicated_nber_titles}")

    nber_for_merge = nber_for_merge.drop_duplicates(subset=["merge_title"], keep="first")

    merged = cleaned_for_merge.merge(
        nber_for_merge,
        on="merge_title",
        how="left",
    )
    nber_match_count = count_nonblank(merged, "nber_paper")
    merged = merged.drop(columns=["merge_title"])
    cleaned = clean_after_nber_merge(merged)
    cleaned.attrs["nber_match_count"] = nber_match_count
    return cleaned


def clean_after_nber_merge(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    before_counts = {
        "abstract": count_nonblank(cleaned, "abstract"),
        "jel_codes": count_nonblank(cleaned, "jel_codes"),
        "keywords": count_nonblank(cleaned, "keywords"),
    }

    cleaned["abstract"] = fill_blank_from_column(
        cleaned,
        target_column="abstract",
        source_column="nber_abstract",
    )
    cleaned["jel_codes"] = fill_blank_from_column(
        cleaned,
        target_column="jel_codes",
        source_column="nber_jel_codes",
    )
    filled_counts = {
        column: count_nonblank(cleaned, column) - before_count
        for column, before_count in before_counts.items()
    }

    columns_to_drop = [
        column for column in NBER_COLUMNS_TO_DROP_AFTER_MERGE
        if column in cleaned.columns
    ]
    cleaned = cleaned.drop(columns=columns_to_drop)
    cleaned.attrs["nber_filled_counts"] = filled_counts
    return cleaned


def fill_blank_from_column(
    data: pd.DataFrame,
    target_column: str,
    source_column: str,
) -> pd.Series:
    target = get_column(data, target_column).fillna("").astype(str)
    source = get_column(data, source_column).fillna("").astype(str)
    target_blank = target.str.strip() == ""
    source_nonblank = source.str.strip() != ""
    return target.mask(target_blank & source_nonblank, source)


def get_column(data: pd.DataFrame, column: str) -> pd.Series:
    if column in data.columns:
        return data[column]

    return pd.Series([""] * len(data), index=data.index)


def normalize_title(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def normalize_doi_list(value: object) -> str:
    dois = []
    seen = set()
    for doi in str(value or "").split(";"):
        cleaned_doi = clean_doi(doi)
        if cleaned_doi and cleaned_doi not in seen:
            dois.append(cleaned_doi)
            seen.add(cleaned_doi)
    return "; ".join(dois)


def clean_doi(value: object) -> str:
    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace("https://doi.org/", "")
        .replace("http://doi.org/", "")
        .replace("https://dx.doi.org/", "")
        .replace("http://dx.doi.org/", "")
        .replace("doi:", "")
        .rstrip(".,;")
    )


def keep_letters_and_numbers(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return " ".join(text.split())


def duplicate_title_stats(data: pd.DataFrame, title_column: str) -> dict[str, int]:
    title = get_column(data, title_column).fillna("").astype(str)
    exact_duplicate_rows = int(title.duplicated(keep=False).sum())
    exact_duplicate_groups = int(title.loc[title.duplicated(keep=False)].nunique())

    normalized_title = title.apply(normalize_title)
    normalized_duplicate_rows = int(normalized_title.duplicated(keep=False).sum())
    normalized_duplicate_groups = int(
        normalized_title.loc[normalized_title.duplicated(keep=False)].nunique()
    )

    return {
        "exact_duplicate_rows": exact_duplicate_rows,
        "exact_duplicate_groups": exact_duplicate_groups,
        "normalized_duplicate_rows": normalized_duplicate_rows,
        "normalized_duplicate_groups": normalized_duplicate_groups,
    }


def count_nonblank(data: pd.DataFrame, column: str) -> int:
    if column not in data.columns:
        return 0

    return int(data[column].fillna("").astype(str).str.strip().ne("").sum())


def count_duplicate_rows(data: pd.DataFrame, column: str, cleaner) -> int:
    if column not in data.columns:
        return 0

    values = data[column].apply(cleaner)
    values = values.loc[values != ""]
    return int(values.duplicated(keep=False).sum())


if __name__ == "__main__":
    main()
