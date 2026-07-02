from pathlib import Path
import os
import re

import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

INPUT_DIR = Path("data/processed/author_names")
AEA_SCRAPE_INPUT_CSV = INPUT_DIR / "JEL_Training_Data_AEA_Scrape_AuthorRows_Merged.csv"
OPENALEX_INPUT_CSV = (
    INPUT_DIR / "JEL_Training_Data_OpenAlex_OpenAlexRaw_AuthorRows_Merged.csv"
)
OUTPUT_CSV = INPUT_DIR / "JEL_Training_Data_AEA_Scrape_OpenAlex_AuthorRows_Merged.csv"
LAST_NAME_OVERLAP_RATIO_THRESHOLD = 0.80


## #########################################################################
# In 10_ConsolidateVersionsAuthorNames_aea_scrape_openalex.py, do the following only
# Step 1. Merge "JEL_Training_Data_AEA_Scrape_AuthorRows_Merged.csv" with "JEL_Training_Data_OpenAlex_OpenAlexRaw_AuthorRows_Merged.csv" by doi_full and last name and first name.
# For the unmatched records, merge by doi_full and last name. Then, Keep all matched and unmatched observations.
# Step 2. For the unmatched ones, within the same doi_full, if last name has overlaps, then they are matched.
# generate a label indicating matching type: aea_scrape_only, openalex_only, doi_last_first_name, doi_last_name, doi_overlap_last_name
# For doi_full who has both openalex_only and aea_scrape_only, keep the one with aea_scrape_only, drop the one with openalex_only
# Step 3. Generate a variable aea_scrape_openalex_first_name, aea_scrape_openalex_first_name will take the version with the longest length among aea_scrape_first_name openalex_first_name1 openalex_first_name2.
# Generate a variable aea_scrape_openalex_last_name, aea_scrape_openalex_last_name will take the version of aea_scrape_last_name, if aea_scrape_last_name is missing or blank, take the value in openalex_last_name
# Step 4. Keep only the following doi_full, aea_scrape_openalex_last_name, aea_scrape_openalex_first_name, openalex_names, scrape_authors, aea_authors and export the data.
## #########################################################################


def main() -> None:
    aea_scrape = read_author_data(AEA_SCRAPE_INPUT_CSV)
    openalex = read_author_data(OPENALEX_INPUT_CSV)

    aea_scrape = add_compact_merge_keys(
        aea_scrape,
        first_name_columns=["aea_scrape_first_name"],
        last_name_column="aea_scrape_last_name",
        first_name_key_column="aea_scrape_merge_first_name",
    )
    openalex = add_compact_merge_keys(
        openalex,
        first_name_columns=["openalex_first_name1", "openalex_first_name2"],
        last_name_column="openalex_last_name",
        first_name_key_column="openalex_merge_first_name",
    )

    merged = merge_aea_scrape_with_openalex(aea_scrape, openalex)
    merged = add_aea_scrape_openalex_doi_label(merged)
    rows_before_filter = len(merged)
    merged = filter_aea_scrape_openalex_conflicting_only_rows(merged)
    merged = add_aea_scrape_openalex_first_name(merged)
    merged = add_aea_scrape_openalex_last_name(merged)
    output = keep_aea_scrape_openalex_author_columns(merged)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_CSV, index=False)

    print("AEA/scrape/OpenAlex author-name consolidation summary:")
    print(f"  AEA/scrape input CSV: {AEA_SCRAPE_INPUT_CSV}")
    print(f"  AEA/scrape rows: {len(aea_scrape)}")
    print(f"  OpenAlex input CSV: {OPENALEX_INPUT_CSV}")
    print(f"  OpenAlex rows: {len(openalex)}")
    print(f"  Output CSV: {OUTPUT_CSV}")
    print(f"  Output rows: {len(output)}")
    print(
        "  Rows dropped by special AEA/scrape/OpenAlex rules: "
        f"{rows_before_filter - len(merged)}"
    )
    print(
        "  DOI groups with both AEA/scrape-only and OpenAlex-only: "
        f"{count_flagged_doi_groups(merged, 'doi_full_has_openalex_only_and_aea_scrape_only')}"
    )
    print_match_counts(merged)
    print_match_strategy_counts(merged)


def read_author_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    return pd.read_csv(path, dtype=str).fillna("")


def add_compact_merge_keys(
    data: pd.DataFrame,
    first_name_columns: list[str],
    last_name_column: str,
    first_name_key_column: str,
) -> pd.DataFrame:
    required_columns = ["doi_full", last_name_column] + first_name_columns
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in compact author data: {missing_columns}")

    data = data.copy()
    data[first_name_key_column] = coalesce_columns(data, first_name_columns).apply(
        normalize_name_key
    )
    data["merge_last_name"] = data[last_name_column].apply(normalize_name_key)
    return data


def normalize_name_key(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    return re.sub(r"[^a-z0-9]", "", text)


def merge_aea_scrape_with_openalex(
    aea_scrape: pd.DataFrame,
    openalex: pd.DataFrame,
) -> pd.DataFrame:
    first_pass = aea_scrape.merge(
        openalex,
        left_on=["doi_full", "merge_last_name", "aea_scrape_merge_first_name"],
        right_on=["doi_full", "merge_last_name", "openalex_merge_first_name"],
        how="outer",
        indicator="first_pass_match_status",
    )
    first_pass["first_pass_match_status"] = first_pass[
        "first_pass_match_status"
    ].astype(str)

    matched_first_pass = first_pass.loc[
        first_pass["first_pass_match_status"].eq("both")
    ].copy()
    matched_first_pass["aea_scrape_openalex_match_status"] = "both"
    matched_first_pass["aea_scrape_openalex_match_strategy"] = "doi_last_first_name"

    unmatched_aea_scrape = first_pass.loc[
        first_pass["first_pass_match_status"].eq("left_only"), aea_scrape.columns
    ].copy()
    unmatched_openalex = first_pass.loc[
        first_pass["first_pass_match_status"].eq("right_only"), openalex.columns
    ].copy()

    second_pass = unmatched_aea_scrape.merge(
        unmatched_openalex,
        on=["doi_full", "merge_last_name"],
        how="outer",
        indicator="second_pass_match_status",
    )
    second_pass["second_pass_match_status"] = second_pass[
        "second_pass_match_status"
    ].astype(str)

    matched_second_pass = second_pass.loc[
        second_pass["second_pass_match_status"].eq("both")
    ].copy()
    matched_second_pass["aea_scrape_openalex_match_status"] = "both"
    matched_second_pass["aea_scrape_openalex_match_strategy"] = "doi_last_name"

    unmatched_aea_scrape = second_pass.loc[
        second_pass["second_pass_match_status"].eq("left_only"), aea_scrape.columns
    ].copy()
    unmatched_openalex = second_pass.loc[
        second_pass["second_pass_match_status"].eq("right_only"), openalex.columns
    ].copy()
    overlap_pass = overlap_match_by_last_name(unmatched_aea_scrape, unmatched_openalex)

    matched_first_pass = matched_first_pass.drop(columns=["first_pass_match_status"])
    matched_second_pass = matched_second_pass.drop(columns=["second_pass_match_status"])
    return pd.concat(
        [matched_first_pass, matched_second_pass, overlap_pass],
        ignore_index=True,
        sort=False,
    )


def overlap_match_by_last_name(
    aea_scrape: pd.DataFrame,
    openalex: pd.DataFrame,
) -> pd.DataFrame:
    matched_rows = []
    used_aea_indices = set()
    used_openalex_indices = set()

    for aea_index, aea_row in aea_scrape.iterrows():
        doi = str(aea_row.get("doi_full", "")).strip()
        aea_last_name = str(aea_row.get("merge_last_name", "")).strip()
        if not doi or not aea_last_name:
            continue

        candidates = openalex.loc[
            (openalex["doi_full"].fillna("").astype(str).str.strip() == doi)
            & ~openalex.index.isin(used_openalex_indices)
        ]
        if candidates.empty:
            continue

        best_index = None
        best_ratio = 0.0
        for openalex_index, openalex_row in candidates.iterrows():
            openalex_last_name = str(openalex_row.get("merge_last_name", "")).strip()
            overlap_ratio = last_name_overlap_ratio(aea_last_name, openalex_last_name)
            if overlap_ratio > best_ratio:
                best_index = openalex_index
                best_ratio = overlap_ratio

        if best_index is None or best_ratio < LAST_NAME_OVERLAP_RATIO_THRESHOLD:
            continue

        used_aea_indices.add(aea_index)
        used_openalex_indices.add(best_index)
        matched_rows.append(
            combined_row_from_two_sources(
                aea_scrape.loc[aea_index],
                openalex.loc[best_index],
                match_status="both",
                match_strategy="doi_overlap_last_name",
                overlap_score_value=round(best_ratio, 4),
            )
        )

    unmatched_rows = []
    for aea_index, row in aea_scrape.loc[~aea_scrape.index.isin(used_aea_indices)].iterrows():
        output_row = row.to_dict()
        output_row["aea_scrape_openalex_match_status"] = "left_only"
        output_row["aea_scrape_openalex_match_strategy"] = "aea_scrape_only"
        output_row["aea_scrape_openalex_last_name_overlap_ratio"] = ""
        unmatched_rows.append(output_row)

    for openalex_index, row in openalex.loc[
        ~openalex.index.isin(used_openalex_indices)
    ].iterrows():
        output_row = row.to_dict()
        output_row["aea_scrape_openalex_match_status"] = "right_only"
        output_row["aea_scrape_openalex_match_strategy"] = "openalex_only"
        output_row["aea_scrape_openalex_last_name_overlap_ratio"] = ""
        unmatched_rows.append(output_row)

    return pd.DataFrame(matched_rows + unmatched_rows)


def combined_row_from_two_sources(
    aea_row: pd.Series,
    openalex_row: pd.Series,
    match_status: str,
    match_strategy: str,
    overlap_score_value: float,
) -> dict:
    output_row = aea_row.to_dict()
    for column, value in openalex_row.to_dict().items():
        if column in output_row and nonblank_text(output_row[column]) != "":
            continue
        output_row[column] = value
    output_row["aea_scrape_openalex_match_status"] = match_status
    output_row["aea_scrape_openalex_match_strategy"] = match_strategy
    output_row["aea_scrape_openalex_last_name_overlap_ratio"] = overlap_score_value
    return output_row


def last_name_overlap_ratio(left: object, right: object) -> float:
    left_text = str(left or "").strip()
    right_text = str(right or "").strip()
    if not left_text or not right_text:
        return 0.0
    if left_text in right_text or right_text in left_text:
        return 1.0
    overlap_length = longest_common_substring_length(left_text, right_text)
    shorter_length = min(len(left_text), len(right_text))
    if shorter_length == 0:
        return 0.0
    return overlap_length / shorter_length


def longest_common_substring_length(left: str, right: str) -> int:
    if not left or not right:
        return 0
    previous_row = [0] * (len(right) + 1)
    best = 0
    for left_character in left:
        current_row = [0]
        for right_index, right_character in enumerate(right, start=1):
            if left_character == right_character:
                value = previous_row[right_index - 1] + 1
                best = max(best, value)
            else:
                value = 0
            current_row.append(value)
        previous_row = current_row
    return best


def add_aea_scrape_openalex_doi_label(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    doi = data["doi_full"].fillna("").astype(str).str.strip()
    strategy = data["aea_scrape_openalex_match_strategy"].fillna("").astype(str)

    has_aea_scrape_only = strategy.eq("aea_scrape_only").groupby(doi).transform("any")
    has_openalex_only = strategy.eq("openalex_only").groupby(doi).transform("any")
    has_nonblank_doi = doi.ne("")

    data["doi_full_has_openalex_only_and_aea_scrape_only"] = (
        has_nonblank_doi & has_aea_scrape_only & has_openalex_only
    ).astype(int)
    return data


def filter_aea_scrape_openalex_conflicting_only_rows(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    strategy = data["aea_scrape_openalex_match_strategy"].fillna("").astype(str)
    conflict = data["doi_full_has_openalex_only_and_aea_scrape_only"].eq(1)
    drop_rows = conflict & strategy.eq("openalex_only")
    return data.loc[~drop_rows].copy()


def add_aea_scrape_openalex_first_name(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["aea_scrape_openalex_first_name"] = data.apply(
        lambda row: longest_text(
            row.get("aea_scrape_first_name", ""),
            row.get("openalex_first_name1", ""),
            row.get("openalex_first_name2", ""),
        ),
        axis=1,
    )
    return data


def add_aea_scrape_openalex_last_name(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["aea_scrape_openalex_last_name"] = coalesce_columns(
        data,
        ["aea_scrape_last_name", "openalex_last_name"],
    )
    return data


def keep_aea_scrape_openalex_author_columns(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "doi_full",
        "aea_scrape_openalex_last_name",
        "aea_scrape_openalex_first_name",
        "openalex_names",
        "scrape_authors",
        "aea_authors",
    ]
    missing_columns = [column for column in columns if column not in data.columns]
    if missing_columns:
        raise ValueError(
            f"Missing columns in AEA/scrape/OpenAlex merged data: {missing_columns}"
        )
    return data[columns].copy()


def coalesce_columns(data: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = pd.Series([""] * len(data), index=data.index)
    for column in columns:
        if column not in data.columns:
            continue
        candidate = data[column].fillna("").astype(str).str.strip()
        values = values.mask(values.eq(""), candidate)
    return values


def longest_text(*values: object) -> str:
    cleaned_values = [str(value).strip() for value in values if not pd.isna(value)]
    cleaned_values = [value for value in cleaned_values if value]
    if not cleaned_values:
        return ""
    return max(cleaned_values, key=len)


def nonblank_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def count_flagged_doi_groups(data: pd.DataFrame, flag_column: str) -> int:
    if flag_column not in data.columns:
        return 0
    flagged = data.loc[data[flag_column].eq(1), "doi_full"]
    return int(flagged.nunique())


def print_match_counts(data: pd.DataFrame) -> None:
    counts = data["aea_scrape_openalex_match_status"].value_counts().to_dict()
    print("  Match counts:")
    print(f"    matched: {counts.get('both', 0)}")
    print(f"    only in AEA/scrape: {counts.get('left_only', 0)}")
    print(f"    only in OpenAlex: {counts.get('right_only', 0)}")


def print_match_strategy_counts(data: pd.DataFrame) -> None:
    counts = data["aea_scrape_openalex_match_strategy"].value_counts().to_dict()
    print("  Match strategy counts:")
    print(f"    matched by doi + last + first: {counts.get('doi_last_first_name', 0)}")
    print(f"    matched by doi + last: {counts.get('doi_last_name', 0)}")
    print(f"    matched by doi + overlapping last: {counts.get('doi_overlap_last_name', 0)}")
    print(f"    only in AEA/scrape: {counts.get('aea_scrape_only', 0)}")
    print(f"    only in OpenAlex: {counts.get('openalex_only', 0)}")


if __name__ == "__main__":
    main()
