from pathlib import Path
import os
import re

import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

CLEANED_INPUT_CSV = Path("data/processed/OpenAlex_Crossref_All_Webscraped_Cleaned.csv")
NBER_INPUT_CSV = Path("data/raw_csv/NBER_Working_Papers_Metadata_After1995.csv")
MERGED_OUTPUT_CSV = Path("data/processed/OpenAlex_Crossref_Webscrape_NBER_Merged.csv")
PAPERS_TO_DROP = {"w7565", "w13800", "w21929", "w8649"}
NBER_COLUMNS_TO_DROP_AFTER_MERGE = [
    "nber_paper",
    "nber_doi",
    "nber_title",
    "nber_issue_date",
    "nber_published_text",
    "nber_collection_date",
]


def main() -> None:
    cleaned = read_cleaned_openalex_crossref_data(CLEANED_INPUT_CSV)
    nber = read_nber_data(NBER_INPUT_CSV)
    merged = merge_with_nber_by_title(cleaned, nber)

    merged.to_csv(MERGED_OUTPUT_CSV, index=False)

    print("OpenAlex/Crossref/Webscrape + NBER merge summary:")
    print(f"  Base input CSV: {CLEANED_INPUT_CSV}")
    print(f"  Base rows: {len(cleaned)}")
    print(f"  Output CSV: {MERGED_OUTPUT_CSV}")
    print(f"  Output rows: {len(merged)}")
    print(f"  Rows matched to NBER: {merged.attrs.get('nber_match_count', 0)}")

def read_cleaned_openalex_crossref_data(path: Path) -> pd.DataFrame:
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

    columns_to_drop = [
        column for column in NBER_COLUMNS_TO_DROP_AFTER_MERGE
        if column in cleaned.columns
    ]
    return cleaned.drop(columns=columns_to_drop)


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


def count_nonblank(data: pd.DataFrame, column: str) -> int:
    if column not in data.columns:
        return 0

    return int(data[column].fillna("").astype(str).str.strip().ne("").sum())


if __name__ == "__main__":
    main()
