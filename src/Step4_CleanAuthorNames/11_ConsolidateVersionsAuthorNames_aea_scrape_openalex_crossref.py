from pathlib import Path
import os
import re

import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

INPUT_DIR = Path("data/processed/author_names")
AEA_SCRAPE_OPENALEX_INPUT_CSV = (
    INPUT_DIR / "JEL_Training_Data_AEA_Scrape_OpenAlex_AuthorRows_Merged.csv"
)
CROSSREF_INPUT_CSV = (
    INPUT_DIR / "JEL_Training_Data_Crossref_CrossrefGivenFamily_AuthorRows_Merged.csv"
)
OUTPUT_CSV = (
    INPUT_DIR / "JEL_Training_Data_AEA_Scrape_OpenAlex_Crossref_AuthorRows_Merged.csv"
)


## #########################################################################
# In 11_ConsolidateVersionsAuthorNames_aea_scrape_openalex_crossref.py, do the following only
# Step 1. Merge "JEL_Training_Data_AEA_Scrape_OpenAlex_AuthorRows_Merged.csv" with "JEL_Training_Data_Crossref_CrossrefGivenFamily_AuthorRows_Merged.csv" by doi_full and last name and first name.
# For the unmatched records, merge by doi_full and last name.
# Then, Keep all matched and unmatched observations.
# generate a label for doi_full who has both aea_scrape_openalex_only and crossref_only
# For doi_full who has both aea_scrape_openalex_only and crossref_only, keep the one with aea_scrape_openalex_only, drop the one with crossref_only
# Step 2. Generate a variable final_first_name, final_first_name will take the version of aea_scrape_openalex_first_name,
# if aea_scrape_openalex_first_name is missing or blank, take the value in crossref_first_name
# Generate a variable final_last_name, final_last_name will take the version of aea_scrape_openalex_last_name,
# if aea_scrape_openalex_last_name is missing or blank, take the value in crossref_last_name
# Step 3. Keep only the following doi_full, final_last_name, final_first_name, openalex_names, scrape_authors, aea_authors
# crossref_authors, and export the data.
## #########################################################################


def main() -> None:
    aea_scrape_openalex = read_author_data(AEA_SCRAPE_OPENALEX_INPUT_CSV)
    crossref = read_author_data(CROSSREF_INPUT_CSV)

    aea_scrape_openalex = add_compact_merge_keys(
        aea_scrape_openalex,
        first_name_column="aea_scrape_openalex_first_name",
        last_name_column="aea_scrape_openalex_last_name",
        first_name_key_column="aea_scrape_openalex_merge_first_name",
    )
    crossref = add_compact_merge_keys(
        crossref,
        first_name_column="crossref_first_name",
        last_name_column="crossref_last_name",
        first_name_key_column="crossref_merge_first_name",
    )

    merged = merge_aea_scrape_openalex_with_crossref(aea_scrape_openalex, crossref)
    merged = add_aea_scrape_openalex_crossref_doi_label(merged)
    rows_before_filter = len(merged)
    merged = filter_aea_scrape_openalex_crossref_conflicting_only_rows(merged)
    merged = add_final_name_columns(merged)
    output = keep_final_author_columns(merged)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_CSV, index=False)

    print("AEA/scrape/OpenAlex/Crossref author-name consolidation summary:")
    print(f"  AEA/scrape/OpenAlex input CSV: {AEA_SCRAPE_OPENALEX_INPUT_CSV}")
    print(f"  AEA/scrape/OpenAlex rows: {len(aea_scrape_openalex)}")
    print(f"  Crossref input CSV: {CROSSREF_INPUT_CSV}")
    print(f"  Crossref rows: {len(crossref)}")
    print(f"  Output CSV: {OUTPUT_CSV}")
    print(f"  Output rows: {len(output)}")
    print(
        "  Rows dropped by special AEA/scrape/OpenAlex/Crossref rules: "
        f"{rows_before_filter - len(merged)}"
    )
    print(
        "  DOI groups with both AEA/scrape/OpenAlex-only and Crossref-only: "
        f"{count_flagged_doi_groups(merged, 'doi_full_has_aea_scrape_openalex_only_and_crossref_only')}"
    )
    print_match_counts(merged)
    print_match_strategy_counts(merged)


def read_author_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    return pd.read_csv(path, dtype=str).fillna("")


def add_compact_merge_keys(
    data: pd.DataFrame,
    first_name_column: str,
    last_name_column: str,
    first_name_key_column: str,
) -> pd.DataFrame:
    required_columns = ["doi_full", first_name_column, last_name_column]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in compact author data: {missing_columns}")

    data = data.copy()
    data[first_name_key_column] = data[first_name_column].apply(normalize_name_key)
    data["merge_last_name"] = data[last_name_column].apply(normalize_name_key)
    return data


def normalize_name_key(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    return re.sub(r"[^a-z0-9]", "", text)


def merge_aea_scrape_openalex_with_crossref(
    aea_scrape_openalex: pd.DataFrame,
    crossref: pd.DataFrame,
) -> pd.DataFrame:
    first_pass = aea_scrape_openalex.merge(
        crossref,
        left_on=[
            "doi_full",
            "merge_last_name",
            "aea_scrape_openalex_merge_first_name",
        ],
        right_on=["doi_full", "merge_last_name", "crossref_merge_first_name"],
        how="outer",
        indicator="first_pass_match_status",
    )
    first_pass["first_pass_match_status"] = first_pass[
        "first_pass_match_status"
    ].astype(str)

    matched_first_pass = first_pass.loc[
        first_pass["first_pass_match_status"].eq("both")
    ].copy()
    matched_first_pass["aea_scrape_openalex_crossref_match_status"] = "both"
    matched_first_pass["aea_scrape_openalex_crossref_match_strategy"] = (
        "doi_last_first_name"
    )

    unmatched_base = first_pass.loc[
        first_pass["first_pass_match_status"].eq("left_only"),
        aea_scrape_openalex.columns,
    ].copy()
    unmatched_crossref = first_pass.loc[
        first_pass["first_pass_match_status"].eq("right_only"),
        crossref.columns,
    ].copy()

    second_pass = unmatched_base.merge(
        unmatched_crossref,
        on=["doi_full", "merge_last_name"],
        how="outer",
        indicator="second_pass_match_status",
    )
    second_pass["second_pass_match_status"] = second_pass[
        "second_pass_match_status"
    ].astype(str)
    second_pass["aea_scrape_openalex_crossref_match_status"] = second_pass[
        "second_pass_match_status"
    ]
    second_pass["aea_scrape_openalex_crossref_match_strategy"] = second_pass[
        "second_pass_match_status"
    ].map(
        {
            "both": "doi_last_name",
            "left_only": "aea_scrape_openalex_only",
            "right_only": "crossref_only",
        }
    )

    matched_first_pass = matched_first_pass.drop(columns=["first_pass_match_status"])
    second_pass = second_pass.drop(columns=["second_pass_match_status"])
    return pd.concat([matched_first_pass, second_pass], ignore_index=True, sort=False)


def add_aea_scrape_openalex_crossref_doi_label(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    doi = data["doi_full"].fillna("").astype(str).str.strip()
    strategy = data["aea_scrape_openalex_crossref_match_strategy"].fillna("").astype(str)

    has_base_only = strategy.eq("aea_scrape_openalex_only").groupby(doi).transform("any")
    has_crossref_only = strategy.eq("crossref_only").groupby(doi).transform("any")
    has_nonblank_doi = doi.ne("")

    data["doi_full_has_aea_scrape_openalex_only_and_crossref_only"] = (
        has_nonblank_doi & has_base_only & has_crossref_only
    ).astype(int)
    return data


def filter_aea_scrape_openalex_crossref_conflicting_only_rows(
    data: pd.DataFrame,
) -> pd.DataFrame:
    data = data.copy()
    strategy = data["aea_scrape_openalex_crossref_match_strategy"].fillna("").astype(str)
    conflict = data["doi_full_has_aea_scrape_openalex_only_and_crossref_only"].eq(1)
    drop_rows = conflict & strategy.eq("crossref_only")
    return data.loc[~drop_rows].copy()


def add_final_name_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["final_first_name"] = coalesce_columns(
        data,
        ["aea_scrape_openalex_first_name", "crossref_first_name"],
    )
    data["final_last_name"] = coalesce_columns(
        data,
        ["aea_scrape_openalex_last_name", "crossref_last_name"],
    )
    return data


def keep_final_author_columns(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "doi_full",
        "final_last_name",
        "final_first_name",
        "openalex_names",
        "scrape_authors",
        "aea_authors",
        "crossref_authors",
    ]
    missing_columns = [column for column in columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in final merged data: {missing_columns}")
    return data[columns].copy()


def coalesce_columns(data: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = pd.Series([""] * len(data), index=data.index)
    for column in columns:
        if column not in data.columns:
            continue
        candidate = data[column].fillna("").astype(str).str.strip()
        values = values.mask(values.eq(""), candidate)
    return values


def count_flagged_doi_groups(data: pd.DataFrame, flag_column: str) -> int:
    if flag_column not in data.columns:
        return 0
    flagged = data.loc[data[flag_column].eq(1), "doi_full"]
    return int(flagged.nunique())


def print_match_counts(data: pd.DataFrame) -> None:
    counts = data["aea_scrape_openalex_crossref_match_status"].value_counts().to_dict()
    print("  Match counts:")
    print(f"    matched: {counts.get('both', 0)}")
    print(f"    only in AEA/scrape/OpenAlex: {counts.get('left_only', 0)}")
    print(f"    only in Crossref: {counts.get('right_only', 0)}")


def print_match_strategy_counts(data: pd.DataFrame) -> None:
    counts = data["aea_scrape_openalex_crossref_match_strategy"].value_counts().to_dict()
    print("  Match strategy counts:")
    print(f"    matched by doi + last + first: {counts.get('doi_last_first_name', 0)}")
    print(f"    matched by doi + last: {counts.get('doi_last_name', 0)}")
    print(
        "    only in AEA/scrape/OpenAlex: "
        f"{counts.get('aea_scrape_openalex_only', 0)}"
    )
    print(f"    only in Crossref: {counts.get('crossref_only', 0)}")


if __name__ == "__main__":
    main()
