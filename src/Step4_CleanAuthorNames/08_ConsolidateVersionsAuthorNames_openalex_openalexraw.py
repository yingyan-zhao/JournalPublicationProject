from pathlib import Path
import os
import re

import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

INPUT_DIR = Path("data/processed/author_names")
OPENALEX_INPUT_CSV = INPUT_DIR / "JEL_Training_Data_OpenAlex_AuthorRows.csv"
OPENALEX_RAW_INPUT_CSV = INPUT_DIR / "JEL_Training_Data_OpenAlex_Raw_AuthorRows.csv"
OUTPUT_CSV = INPUT_DIR / "JEL_Training_Data_OpenAlex_OpenAlexRaw_AuthorRows_Merged.csv"


## #########################################################################
# In 08_ConsolidateVersionsAuthorNames_openalex_openalexraw.py.py, consolidate different versions of author names.
# Step 1. Merge "JEL_Training_Data_OpenAlex_AuthorRows.csv" with "JEL_Training_Data_OpenAlex_Raw_AuthorRows.csv" by doi_full and last name and first name.
# For the unmatched records, merge by doi_full and last name. Then, Keep all matched and unmatched observations.
# For doi_full = 10.3982/ecta7920, keep both openalex_only and openalex_raw_only.
# For doi_full = 10.3982/ecta9431 or 10.3982/ecta6754, keep openalex_only.
# For all other doi_full, if the doi_full has both openalex_only and openalex_raw_only, keep the one with openalex_raw_only.
# Step 2. Consolidate OpenAlex and OpenAlex_raw
# Create a new column openalex_names which consolidates both versions: openalex_authors and openalex_raw_author_names. Separate these two versions by ";"
# Create a new column openalex_last_name, it consolidates the information in openalex_author_last_name and openalex_raw_author_last_name
# Create a new column openalex_first_name1 = openalex_author_first_name and openalex_first_name2 = openalex_raw_author_first_name.
# Step 3. Keep doi_full, openalex_names, openalex_last_name, openalex_first_name1, openalex_first_name2, and Export as "JEL_Training_Data_OpenAlex_OpenAlexRaw_AuthorRows_Merged.csv"
## #########################################################################


def main() -> None:
    openalex = read_author_data(OPENALEX_INPUT_CSV)
    openalex_raw = read_author_data(OPENALEX_RAW_INPUT_CSV)

    openalex = add_merge_keys(
        openalex,
        first_name_column="openalex_author_first_name",
        last_name_column="openalex_author_last_name",
        first_name_key_column="openalex_merge_first_name",
    )
    openalex_raw = add_merge_keys(
        openalex_raw,
        first_name_column="openalex_raw_author_first_name",
        last_name_column="openalex_raw_author_last_name",
        first_name_key_column="openalex_raw_merge_first_name",
    )

    merged = merge_openalex_with_openalex_raw(openalex, openalex_raw)
    rows_before_filter = len(merged)
    merged = filter_openalex_raw_conflicting_only_rows(merged)
    merged = add_openalex_consolidated_name_columns(merged)
    output = keep_output_columns(merged)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_CSV, index=False)

    print("OpenAlex/OpenAlex raw author-name consolidation summary:")
    print(f"  OpenAlex input CSV: {OPENALEX_INPUT_CSV}")
    print(f"  OpenAlex rows: {len(openalex)}")
    print(f"  OpenAlex raw input CSV: {OPENALEX_RAW_INPUT_CSV}")
    print(f"  OpenAlex raw rows: {len(openalex_raw)}")
    print(f"  Output CSV: {OUTPUT_CSV}")
    print(f"  Output rows: {len(output)}")
    print(f"  Rows dropped by special openalex/raw rules: {rows_before_filter - len(merged)}")
    print_match_counts(merged)
    print_match_strategy_counts(merged)


def read_author_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    return pd.read_csv(path, dtype=str).fillna("")


def add_merge_keys(
    data: pd.DataFrame,
    first_name_column: str,
    last_name_column: str,
    first_name_key_column: str,
) -> pd.DataFrame:
    required_columns = ["doi_full", first_name_column, last_name_column]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in author data: {missing_columns}")

    data = data.copy()
    data[first_name_key_column] = data[first_name_column].apply(normalize_name_key)
    data["merge_last_name"] = data[last_name_column].apply(normalize_name_key)
    return data


def normalize_name_key(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    return re.sub(r"[^a-z0-9]", "", text)


def merge_openalex_with_openalex_raw(
    openalex: pd.DataFrame,
    openalex_raw: pd.DataFrame,
) -> pd.DataFrame:
    first_pass = openalex.merge(
        openalex_raw,
        left_on=["doi_full", "merge_last_name", "openalex_merge_first_name"],
        right_on=["doi_full", "merge_last_name", "openalex_raw_merge_first_name"],
        how="outer",
        indicator="first_pass_match_status",
    )
    first_pass["first_pass_match_status"] = first_pass[
        "first_pass_match_status"
    ].astype(str)

    matched_first_pass = first_pass.loc[
        first_pass["first_pass_match_status"].eq("both")
    ].copy()
    matched_first_pass["openalex_raw_match_status"] = "both"
    matched_first_pass["openalex_raw_match_strategy"] = "doi_last_first_name"

    unmatched_openalex = first_pass.loc[
        first_pass["first_pass_match_status"].eq("left_only"), openalex.columns
    ].copy()
    unmatched_openalex_raw = first_pass.loc[
        first_pass["first_pass_match_status"].eq("right_only"), openalex_raw.columns
    ].copy()

    second_pass = unmatched_openalex.merge(
        unmatched_openalex_raw,
        on=["doi_full", "merge_last_name"],
        how="outer",
        indicator="second_pass_match_status",
    )
    second_pass["second_pass_match_status"] = second_pass[
        "second_pass_match_status"
    ].astype(str)
    second_pass["openalex_raw_match_status"] = second_pass["second_pass_match_status"]
    second_pass["openalex_raw_match_strategy"] = second_pass[
        "second_pass_match_status"
    ].map(
        {
            "both": "doi_last_name",
            "left_only": "openalex_only",
            "right_only": "openalex_raw_only",
        }
    )

    matched_first_pass = matched_first_pass.drop(columns=["first_pass_match_status"])
    second_pass = second_pass.drop(columns=["second_pass_match_status"])
    return pd.concat([matched_first_pass, second_pass], ignore_index=True, sort=False)


def filter_openalex_raw_conflicting_only_rows(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    doi = data["doi_full"].fillna("").astype(str).str.strip()
    strategy = data["openalex_raw_match_strategy"].fillna("").astype(str)

    has_openalex_only = strategy.eq("openalex_only").groupby(doi).transform("any")
    has_openalex_raw_only = strategy.eq("openalex_raw_only").groupby(doi).transform("any")
    conflicting_doi = doi.ne("") & has_openalex_only & has_openalex_raw_only

    keep = pd.Series(True, index=data.index)
    keep_both_doi = {"10.3982/ecta7920"}
    keep_openalex_only_doi = {"10.3982/ecta9431", "10.3982/ecta6754"}

    drop_raw_only_for_openalex_only_doi = (
        doi.isin(keep_openalex_only_doi)
        & conflicting_doi
        & strategy.eq("openalex_raw_only")
    )
    keep = keep & ~drop_raw_only_for_openalex_only_doi

    drop_openalex_only_for_other_conflicts = (
        conflicting_doi
        & ~doi.isin(keep_both_doi)
        & ~doi.isin(keep_openalex_only_doi)
        & strategy.eq("openalex_only")
    )
    keep = keep & ~drop_openalex_only_for_other_conflicts

    return data.loc[keep].copy()


def add_openalex_consolidated_name_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["openalex_names"] = data.apply(
        lambda row: join_unique_values(
            row.get("openalex_authors", ""),
            row.get("openalex_raw_author_names", ""),
        ),
        axis=1,
    )
    data["openalex_last_name"] = coalesce_columns(
        data,
        ["openalex_author_last_name", "openalex_raw_author_last_name"],
    )
    data["openalex_first_name1"] = (
        data["openalex_author_first_name"].fillna("").astype(str).str.strip()
    )
    data["openalex_first_name2"] = (
        data["openalex_raw_author_first_name"].fillna("").astype(str).str.strip()
    )
    return data


def keep_output_columns(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "doi_full",
        "openalex_names",
        "openalex_last_name",
        "openalex_first_name1",
        "openalex_first_name2",
    ]
    missing_columns = [column for column in columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in OpenAlex merged data: {missing_columns}")
    return data[columns].copy()


def coalesce_columns(data: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = pd.Series([""] * len(data), index=data.index)
    for column in columns:
        if column not in data.columns:
            continue
        candidate = data[column].fillna("").astype(str).str.strip()
        values = values.mask(values.eq(""), candidate)
    return values


def join_unique_values(*values: object) -> str:
    unique_values = []
    seen_values = set()
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen_values:
            continue
        unique_values.append(text)
        seen_values.add(key)
    return ";".join(unique_values)


def print_match_counts(data: pd.DataFrame) -> None:
    counts = data["openalex_raw_match_status"].value_counts().to_dict()
    print("  Match counts:")
    print(f"    matched: {counts.get('both', 0)}")
    print(f"    only in OpenAlex: {counts.get('left_only', 0)}")
    print(f"    only in OpenAlex raw: {counts.get('right_only', 0)}")


def print_match_strategy_counts(data: pd.DataFrame) -> None:
    counts = data["openalex_raw_match_strategy"].value_counts().to_dict()
    print("  Match strategy counts:")
    print(f"    matched by doi + last + first: {counts.get('doi_last_first_name', 0)}")
    print(f"    matched by doi + last: {counts.get('doi_last_name', 0)}")
    print(f"    only in OpenAlex: {counts.get('openalex_only', 0)}")
    print(f"    only in OpenAlex raw: {counts.get('openalex_raw_only', 0)}")


if __name__ == "__main__":
    main()
