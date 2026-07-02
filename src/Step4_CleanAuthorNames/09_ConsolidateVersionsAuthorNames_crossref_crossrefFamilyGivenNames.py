from pathlib import Path
import os
import re

import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

INPUT_DIR = Path("data/processed/author_names")
CROSSREF_INPUT_CSV = INPUT_DIR / "JEL_Training_Data_Crossref_AuthorRows.csv"
CROSSREF_GIVEN_FAMILY_INPUT_CSV = (
    INPUT_DIR / "JEL_Training_Data_Crossref_GivenFamily_AuthorRows.csv"
)
CROSSREF_OUTPUT_CSV = (
    INPUT_DIR / "JEL_Training_Data_Crossref_CrossrefGivenFamily_AuthorRows_Merged.csv"
)


## #########################################################################
# In 09_ConsolidateVersionsAuthorNames_crossref_crossrefFamilyGivenNames.py, do the following only
# Step 1. Merge "JEL_Training_Data_Crossref_AuthorRows.csv" with "JEL_Training_Data_Crossref_GivenFamily_AuthorRows.csv" by doi_full and last name and first name.
# For the unmatched records, merge by doi_full and last name. Then, Keep all matched and unmatched observations.
# If the doi_full has both crossref_only and crossref_givenfamily_only, keep the one with crossref_only.
# Step 2. Consolidate crossref and crossref_FamilyGivenNames
# Create a new column crossref_last_name, it consolidates the information in crossref_author_last_name and crossref_author_family
# Create a new column crossref_first_name = crossref_author_first_name
# Step 3. Keep doi_full, crossref_last_name, crossref_first_name, crossref_authors, and Export as "JEL_Training_Data_Crossref_CrossrefGivenFamily_AuthorRows_Merged.csv"
## #########################################################################


def main() -> None:
    crossref = read_author_data(CROSSREF_INPUT_CSV)
    crossref_given_family = read_author_data(CROSSREF_GIVEN_FAMILY_INPUT_CSV)

    crossref = add_merge_keys(
        crossref,
        first_name_column="crossref_author_first_name",
        last_name_column="crossref_author_last_name",
        first_name_key_column="crossref_merge_first_name",
    )
    crossref_given_family = rename_crossref_given_family_columns(crossref_given_family)
    crossref_given_family = add_merge_keys(
        crossref_given_family,
        first_name_column="crossref_givenfamily_author_first_name",
        last_name_column="crossref_givenfamily_author_last_name",
        first_name_key_column="crossref_givenfamily_merge_first_name",
    )

    merged = merge_crossref_with_given_family(crossref, crossref_given_family)
    rows_before_filter = len(merged)
    merged = filter_crossref_givenfamily_conflicting_only_rows(merged)
    merged = add_crossref_consolidated_name_columns(merged)
    output = keep_crossref_consolidated_author_columns(merged)

    CROSSREF_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(CROSSREF_OUTPUT_CSV, index=False)

    print("Crossref/Crossref given-family author-name consolidation summary:")
    print(f"  Crossref input CSV: {CROSSREF_INPUT_CSV}")
    print(f"  Crossref rows: {len(crossref)}")
    print(f"  Crossref given-family input CSV: {CROSSREF_GIVEN_FAMILY_INPUT_CSV}")
    print(f"  Crossref given-family rows: {len(crossref_given_family)}")
    print(f"  Output CSV: {CROSSREF_OUTPUT_CSV}")
    print(f"  Output rows: {len(output)}")
    print(
        "  Rows dropped by special Crossref/given-family rules: "
        f"{rows_before_filter - len(merged)}"
    )
    print_crossref_givenfamily_match_counts(merged)
    print_crossref_givenfamily_strategy_counts(merged)


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


def rename_crossref_given_family_columns(data: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "crossref_author_first_name": "crossref_givenfamily_author_first_name",
        "crossref_author_first_name_length": (
            "crossref_givenfamily_author_first_name_length"
        ),
        "crossref_author_last_name": "crossref_givenfamily_author_last_name",
        "crossref_author_last_name_length": (
            "crossref_givenfamily_author_last_name_length"
        ),
    }
    return data.rename(columns=rename_map).copy()


def merge_crossref_with_given_family(
    crossref: pd.DataFrame,
    crossref_given_family: pd.DataFrame,
) -> pd.DataFrame:
    first_pass = crossref.merge(
        crossref_given_family,
        left_on=["doi_full", "merge_last_name", "crossref_merge_first_name"],
        right_on=["doi_full", "merge_last_name", "crossref_givenfamily_merge_first_name"],
        how="outer",
        indicator="first_pass_match_status",
    )
    first_pass["first_pass_match_status"] = first_pass[
        "first_pass_match_status"
    ].astype(str)

    matched_first_pass = first_pass.loc[
        first_pass["first_pass_match_status"].eq("both")
    ].copy()
    matched_first_pass["crossref_givenfamily_match_status"] = "both"
    matched_first_pass["crossref_givenfamily_match_strategy"] = "doi_last_first_name"

    unmatched_crossref = first_pass.loc[
        first_pass["first_pass_match_status"].eq("left_only"), crossref.columns
    ].copy()
    unmatched_crossref_given_family = first_pass.loc[
        first_pass["first_pass_match_status"].eq("right_only"),
        crossref_given_family.columns,
    ].copy()

    second_pass = unmatched_crossref.merge(
        unmatched_crossref_given_family,
        on=["doi_full", "merge_last_name"],
        how="outer",
        indicator="second_pass_match_status",
    )
    second_pass["second_pass_match_status"] = second_pass[
        "second_pass_match_status"
    ].astype(str)
    second_pass["crossref_givenfamily_match_status"] = second_pass[
        "second_pass_match_status"
    ]
    second_pass["crossref_givenfamily_match_strategy"] = second_pass[
        "second_pass_match_status"
    ].map(
        {
            "both": "doi_last_name",
            "left_only": "crossref_only",
            "right_only": "crossref_givenfamily_only",
        }
    )

    matched_first_pass = matched_first_pass.drop(columns=["first_pass_match_status"])
    second_pass = second_pass.drop(columns=["second_pass_match_status"])
    return pd.concat([matched_first_pass, second_pass], ignore_index=True, sort=False)


def filter_crossref_givenfamily_conflicting_only_rows(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    doi = data["doi_full"].fillna("").astype(str).str.strip()
    strategy = data["crossref_givenfamily_match_strategy"].fillna("").astype(str)

    has_crossref_only = strategy.eq("crossref_only").groupby(doi).transform("any")
    has_givenfamily_only = (
        strategy.eq("crossref_givenfamily_only").groupby(doi).transform("any")
    )
    conflicting_doi = doi.ne("") & has_crossref_only & has_givenfamily_only

    drop_givenfamily_only = conflicting_doi & strategy.eq("crossref_givenfamily_only")
    return data.loc[~drop_givenfamily_only].copy()


def add_crossref_consolidated_name_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["crossref_last_name"] = coalesce_columns(
        data,
        ["crossref_author_last_name", "crossref_givenfamily_author_last_name"],
    )
    data["crossref_first_name"] = (
        data["crossref_author_first_name"].fillna("").astype(str).str.strip()
    )
    return data


def keep_crossref_consolidated_author_columns(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "doi_full",
        "crossref_last_name",
        "crossref_first_name",
        "crossref_authors",
    ]
    missing_columns = [column for column in columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in Crossref merged data: {missing_columns}")
    return data[columns].copy()


def coalesce_columns(data: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = pd.Series([""] * len(data), index=data.index)
    for column in columns:
        if column not in data.columns:
            continue
        candidate = data[column].fillna("").astype(str).str.strip()
        values = values.mask(values.eq(""), candidate)
    return values


def print_crossref_givenfamily_match_counts(data: pd.DataFrame) -> None:
    counts = data["crossref_givenfamily_match_status"].value_counts().to_dict()
    print("  Match counts:")
    print(f"    matched: {counts.get('both', 0)}")
    print(f"    only in Crossref: {counts.get('left_only', 0)}")
    print(f"    only in Crossref given-family: {counts.get('right_only', 0)}")


def print_crossref_givenfamily_strategy_counts(data: pd.DataFrame) -> None:
    counts = data["crossref_givenfamily_match_strategy"].value_counts().to_dict()
    print("  Match strategy counts:")
    print(f"    matched by doi + last + first: {counts.get('doi_last_first_name', 0)}")
    print(f"    matched by doi + last: {counts.get('doi_last_name', 0)}")
    print(f"    only in Crossref: {counts.get('crossref_only', 0)}")
    print(
        "    only in Crossref given-family: "
        f"{counts.get('crossref_givenfamily_only', 0)}"
    )


if __name__ == "__main__":
    main()
