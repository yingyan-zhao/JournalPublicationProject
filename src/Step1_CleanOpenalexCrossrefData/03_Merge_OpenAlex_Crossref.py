from pathlib import Path

import os
import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

INPUT_CSV_OpenAlex = Path("data/processed/OpenAlex_Works_Cleaned.csv")
INPUT_CSV_Crossref = Path("data/processed/Crossref_Works_Cleaned.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_CSV_All = OUTPUT_DIR / "OpenAlex_Crossref_All.csv"

OPENALEX_ROW_ID = "_openalex_row_id"
CROSSREF_ROW_ID = "_crossref_row_id"
MERGE_DOI = "_merge_doi"
MERGE_TITLE = "_merge_title"

OPENALEX_DOI_COLUMNS = [
    "openalex_doi_1",
    "openalex_doi_2",
    "openalex_doi_3",
]
CROSSREF_DOI_COLUMNS = [
    "crossref_doi_1",
    "crossref_doi_2",
    "crossref_doi_3",
]
OPENALEX_TITLE = "openalex_title"
CROSSREF_TITLE = "crossref_title"

DOI_MATCH_STAGES = [
    ("openalex_doi_1", "crossref_doi_1"),
    ("openalex_doi_1", "crossref_doi_2"),
    ("openalex_doi_1", "crossref_doi_3"),
    ("openalex_doi_2", "crossref_doi_1"),
    ("openalex_doi_2", "crossref_doi_2"),
    ("openalex_doi_2", "crossref_doi_3"),
    ("openalex_doi_3", "crossref_doi_1"),
    ("openalex_doi_3", "crossref_doi_2"),
    ("openalex_doi_3", "crossref_doi_3"),
]


def main() -> None:
    if not INPUT_CSV_OpenAlex.exists():
        raise FileNotFoundError(f"{INPUT_CSV_OpenAlex} does not exist.")
    if not INPUT_CSV_Crossref.exists():
        raise FileNotFoundError(f"{INPUT_CSV_Crossref} does not exist.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    openalex = pd.read_csv(INPUT_CSV_OpenAlex, dtype=str, keep_default_na=False)
    crossref = pd.read_csv(INPUT_CSV_Crossref, dtype=str, keep_default_na=False)

    check_required_columns(openalex, OPENALEX_DOI_COLUMNS, "OpenAlex")
    check_required_columns(crossref, CROSSREF_DOI_COLUMNS, "Crossref")
    check_required_columns(openalex, [OPENALEX_TITLE], "OpenAlex")
    check_required_columns(crossref, [CROSSREF_TITLE], "Crossref")

    all_records, stage_summaries = merge_by_doi_versions(openalex, crossref)
    all_records = add_combined_columns(all_records)
    all_records["_original_sort_order"] = range(len(all_records))
    all_records["title_duplicate_tag"] = duplicate_nonblank_tag(
        all_records,
        "title",
        normalize_title,
    )
    all_records = consolidate_duplicate_titles(all_records)
    all_records = order_output_columns(all_records, openalex.columns, crossref.columns)
    all_records = drop_temp_columns(all_records)

    print_merge_summary(openalex, crossref, all_records, stage_summaries)

    all_records.to_csv(OUTPUT_CSV_All, index=False)
    print(f"\nWrote all matched and unmatched rows to {OUTPUT_CSV_All}")
    print_final_duplicate_check(all_records)


def check_required_columns(data: pd.DataFrame, columns: list[str], dataset_name: str) -> None:
    missing_columns = [column for column in columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"{dataset_name} data missing columns: {missing_columns}")


def merge_by_doi_versions(
    openalex: pd.DataFrame,
    crossref: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, int | str]]]:
    openalex = openalex.copy()
    crossref = crossref.copy()

    openalex[OPENALEX_ROW_ID] = range(len(openalex))
    crossref[CROSSREF_ROW_ID] = range(len(crossref))

    remaining_openalex = openalex.copy()
    remaining_crossref = crossref.copy()
    matched_frames = []
    stage_summaries = []

    for openalex_doi_column, crossref_doi_column in DOI_MATCH_STAGES:
        matched, remaining_openalex, remaining_crossref, summary = match_one_doi_stage(
            remaining_openalex=remaining_openalex,
            remaining_crossref=remaining_crossref,
            openalex_doi_column=openalex_doi_column,
            crossref_doi_column=crossref_doi_column,
        )
        stage_summaries.append(summary)

        if not matched.empty:
            matched_frames.append(matched)

    matched_by_title, remaining_openalex, remaining_crossref, title_summary = match_by_title(
        remaining_openalex,
        remaining_crossref,
    )
    stage_summaries.append(title_summary)
    if not matched_by_title.empty:
        matched_frames.append(matched_by_title)

    matched_all = concat_frames(matched_frames)
    matched_all = add_record_status(matched_all, "matched")

    only_openalex = add_empty_crossref_columns(remaining_openalex, crossref.columns)
    only_openalex = add_record_status(only_openalex, "openalex_only")

    only_crossref = add_empty_openalex_columns(remaining_crossref, openalex.columns)
    only_crossref = add_record_status(only_crossref, "crossref_only")

    all_records = concat_frames([matched_all, only_openalex, only_crossref])
    all_records = order_output_columns(all_records, openalex.columns, crossref.columns)
    all_records = sort_records(all_records)

    return all_records, stage_summaries


def match_one_doi_stage(
    remaining_openalex: pd.DataFrame,
    remaining_crossref: pd.DataFrame,
    openalex_doi_column: str,
    crossref_doi_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int | str]]:
    if remaining_openalex.empty or remaining_crossref.empty:
        summary = stage_summary(
            openalex_doi_column,
            crossref_doi_column,
            openalex_matchable_rows=0,
            crossref_matchable_rows=0,
            ambiguous_openalex_rows=0,
            ambiguous_crossref_rows=0,
            matched_rows=0,
        )
        return pd.DataFrame(), remaining_openalex.copy(), remaining_crossref.copy(), summary

    openalex_with_key = remaining_openalex.copy()
    crossref_with_key = remaining_crossref.copy()
    openalex_with_key[MERGE_DOI] = openalex_with_key[openalex_doi_column].apply(normalize_doi)
    crossref_with_key[MERGE_DOI] = crossref_with_key[crossref_doi_column].apply(normalize_doi)

    openalex_unique, ambiguous_openalex_rows = keep_unique_nonblank_key(openalex_with_key, MERGE_DOI)
    crossref_unique, ambiguous_crossref_rows = keep_unique_nonblank_key(crossref_with_key, MERGE_DOI)

    matched = openalex_unique.merge(
        crossref_unique,
        on=MERGE_DOI,
        how="inner",
    )

    match_strategy = f"{openalex_doi_column}={crossref_doi_column}"
    if not matched.empty:
        matched["match_strategy"] = match_strategy

    matched_openalex_ids = set(matched[OPENALEX_ROW_ID].tolist()) if not matched.empty else set()
    matched_crossref_ids = set(matched[CROSSREF_ROW_ID].tolist()) if not matched.empty else set()

    next_remaining_openalex = remaining_openalex.loc[
        ~remaining_openalex[OPENALEX_ROW_ID].isin(matched_openalex_ids)
    ].copy()
    next_remaining_crossref = remaining_crossref.loc[
        ~remaining_crossref[CROSSREF_ROW_ID].isin(matched_crossref_ids)
    ].copy()

    summary = stage_summary(
        openalex_doi_column,
        crossref_doi_column,
        openalex_matchable_rows=len(openalex_unique),
        crossref_matchable_rows=len(crossref_unique),
        ambiguous_openalex_rows=ambiguous_openalex_rows,
        ambiguous_crossref_rows=ambiguous_crossref_rows,
        matched_rows=len(matched),
    )

    return matched, next_remaining_openalex, next_remaining_crossref, summary


def match_by_title(
    remaining_openalex: pd.DataFrame,
    remaining_crossref: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int | str]]:
    if remaining_openalex.empty or remaining_crossref.empty:
        summary = title_stage_summary(
            openalex_matchable_rows=0,
            crossref_matchable_rows=0,
            ambiguous_openalex_rows=0,
            ambiguous_crossref_rows=0,
            matched_rows=0,
        )
        return pd.DataFrame(), remaining_openalex.copy(), remaining_crossref.copy(), summary

    openalex_with_key = remaining_openalex.copy()
    crossref_with_key = remaining_crossref.copy()
    openalex_with_key[MERGE_TITLE] = openalex_with_key[OPENALEX_TITLE].apply(normalize_title)
    crossref_with_key[MERGE_TITLE] = crossref_with_key[CROSSREF_TITLE].apply(normalize_title)

    openalex_unique, ambiguous_openalex_rows = keep_unique_nonblank_key(
        openalex_with_key,
        MERGE_TITLE,
    )
    crossref_unique, ambiguous_crossref_rows = keep_unique_nonblank_key(
        crossref_with_key,
        MERGE_TITLE,
    )

    matched = openalex_unique.merge(
        crossref_unique,
        on=MERGE_TITLE,
        how="inner",
    )

    if not matched.empty:
        matched["match_strategy"] = "normalized_title_after_doi"

    matched_openalex_ids = set(matched[OPENALEX_ROW_ID].tolist()) if not matched.empty else set()
    matched_crossref_ids = set(matched[CROSSREF_ROW_ID].tolist()) if not matched.empty else set()

    next_remaining_openalex = remaining_openalex.loc[
        ~remaining_openalex[OPENALEX_ROW_ID].isin(matched_openalex_ids)
    ].copy()
    next_remaining_crossref = remaining_crossref.loc[
        ~remaining_crossref[CROSSREF_ROW_ID].isin(matched_crossref_ids)
    ].copy()

    summary = title_stage_summary(
        openalex_matchable_rows=len(openalex_unique),
        crossref_matchable_rows=len(crossref_unique),
        ambiguous_openalex_rows=ambiguous_openalex_rows,
        ambiguous_crossref_rows=ambiguous_crossref_rows,
        matched_rows=len(matched),
    )

    return matched, next_remaining_openalex, next_remaining_crossref, summary


def keep_unique_nonblank_key(data: pd.DataFrame, key_column: str) -> tuple[pd.DataFrame, int]:
    nonblank = data.loc[data[key_column] != ""].copy()
    duplicated_key = nonblank[key_column].duplicated(keep=False)
    unique_rows = nonblank.loc[~duplicated_key].copy()
    ambiguous_rows = int(duplicated_key.sum())
    return unique_rows, ambiguous_rows


def stage_summary(
    openalex_doi_column: str,
    crossref_doi_column: str,
    openalex_matchable_rows: int,
    crossref_matchable_rows: int,
    ambiguous_openalex_rows: int,
    ambiguous_crossref_rows: int,
    matched_rows: int,
) -> dict[str, int | str]:
    return {
        "match_stage": f"{openalex_doi_column}={crossref_doi_column}",
        "openalex_matchable_rows": openalex_matchable_rows,
        "crossref_matchable_rows": crossref_matchable_rows,
        "ambiguous_openalex_rows": ambiguous_openalex_rows,
        "ambiguous_crossref_rows": ambiguous_crossref_rows,
        "matched_rows": matched_rows,
    }


def title_stage_summary(
    openalex_matchable_rows: int,
    crossref_matchable_rows: int,
    ambiguous_openalex_rows: int,
    ambiguous_crossref_rows: int,
    matched_rows: int,
) -> dict[str, int | str]:
    return {
        "match_stage": "normalized_title_after_doi",
        "openalex_matchable_rows": openalex_matchable_rows,
        "crossref_matchable_rows": crossref_matchable_rows,
        "ambiguous_openalex_rows": ambiguous_openalex_rows,
        "ambiguous_crossref_rows": ambiguous_crossref_rows,
        "matched_rows": matched_rows,
    }


def add_empty_crossref_columns(data: pd.DataFrame, crossref_columns: pd.Index) -> pd.DataFrame:
    data = data.copy()
    for column in crossref_columns:
        if column not in data.columns:
            data[column] = ""
    data[CROSSREF_ROW_ID] = ""
    return data


def add_empty_openalex_columns(data: pd.DataFrame, openalex_columns: pd.Index) -> pd.DataFrame:
    data = data.copy()
    for column in openalex_columns:
        if column not in data.columns:
            data[column] = ""
    data[OPENALEX_ROW_ID] = ""
    return data


def add_record_status(data: pd.DataFrame, record_status: str) -> pd.DataFrame:
    data = data.copy()
    data["record_status"] = record_status
    if "match_strategy" not in data.columns:
        data["match_strategy"] = ""
    return data


def add_combined_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["doi"] = coalesce_columns(
        data,
        [
            "crossref_doi_1",
            "crossref_doi_2",
            "crossref_doi_3",
            "openalex_doi_1",
            "openalex_doi_2",
            "openalex_doi_3",
        ],
        normalize_doi,
    )
    data["journalname"] = coalesce_columns(
        data,
        ["crossref_journalname", "openalex_journalname"],
        clean_text,
    )
    data["title"] = coalesce_columns(
        data,
        ["crossref_title", "openalex_title"],
        clean_text,
    )
    return data


def coalesce_columns(data: pd.DataFrame, columns: list[str], cleaner) -> pd.Series:
    values = pd.Series([""] * len(data), index=data.index, dtype="object")
    for column in columns:
        if column not in data.columns:
            continue
        candidate = data[column].apply(cleaner)
        values = values.mask(values == "", candidate)
    return values


def consolidate_duplicate_titles(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["_normalized_output_title"] = data["title"].apply(normalize_title)
    duplicated_title = data["_normalized_output_title"].ne("") & data[
        "_normalized_output_title"
    ].duplicated(keep=False)

    duplicate_groups = data.loc[duplicated_title].groupby(
        "_normalized_output_title",
        sort=False,
    )
    consolidated_rows = [consolidate_duplicate_group(group) for _, group in duplicate_groups]
    nonduplicated_rows = data.loc[~duplicated_title].copy()

    if consolidated_rows:
        consolidated = pd.DataFrame(consolidated_rows, columns=data.columns)
        data = pd.concat([nonduplicated_rows, consolidated], ignore_index=True)
    else:
        data = nonduplicated_rows

    data = data.sort_values("_original_sort_order", kind="mergesort")
    return data.drop(columns=["_normalized_output_title", "_original_sort_order"])


def consolidate_duplicate_group(group: pd.DataFrame) -> pd.Series:
    row = group.iloc[0].copy()
    row["_original_sort_order"] = group["_original_sort_order"].min()

    for column in group.columns:
        if column in {"_normalized_output_title", "_original_sort_order"}:
            continue
        if column == "title_duplicate_tag":
            row[column] = 1
        else:
            row[column] = longest_text_value(group[column])

    return row


def longest_text_value(values: pd.Series) -> str:
    text_values = values.apply(clean_text)
    if text_values.empty:
        return ""
    return text_values.loc[text_values.str.len().idxmax()]


def normalize_doi(value) -> str:
    text = clean_text(value).lower()
    for prefix in [
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ]:
        if text.startswith(prefix):
            text = text.removeprefix(prefix)
    return text.strip()


def normalize_title(value) -> str:
    text = clean_text(value).lower()
    return "".join(character for character in text if character.isalnum())


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def order_output_columns(
    data: pd.DataFrame,
    openalex_columns: pd.Index,
    crossref_columns: pd.Index,
) -> pd.DataFrame:
    first_columns = [
        "record_status",
        "match_strategy",
        "doi",
        "journalname",
        "title",
        "title_duplicate_tag",
    ]
    ordered_columns = []
    for column in first_columns + list(openalex_columns) + list(crossref_columns):
        if column in data.columns and column not in ordered_columns:
            ordered_columns.append(column)

    remaining_columns = [column for column in data.columns if column not in ordered_columns]
    return data[ordered_columns + remaining_columns].copy()


def sort_records(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["_sort_openalex"] = pd.to_numeric(data[OPENALEX_ROW_ID], errors="coerce")
    data["_sort_crossref"] = pd.to_numeric(data[CROSSREF_ROW_ID], errors="coerce")
    data["_sort_openalex"] = data["_sort_openalex"].fillna(10**12)
    data["_sort_crossref"] = data["_sort_crossref"].fillna(10**12)
    data = data.sort_values(
        ["_sort_openalex", "_sort_crossref", "record_status"],
        kind="mergesort",
    )
    return data.drop(columns=["_sort_openalex", "_sort_crossref"])


def drop_temp_columns(data: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        column
        for column in [OPENALEX_ROW_ID, CROSSREF_ROW_ID, MERGE_DOI, MERGE_TITLE]
        if column in data.columns
    ]
    return data.drop(columns=columns_to_drop)


def concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty_frames = [frame for frame in frames if not frame.empty]
    if not nonempty_frames:
        return pd.DataFrame()
    return pd.concat(nonempty_frames, ignore_index=True)


def count_nonblank(data: pd.DataFrame, column: str) -> int:
    if column not in data.columns:
        return 0
    return int(data[column].fillna("").astype(str).str.strip().ne("").sum())


def duplicate_nonblank_rows(data: pd.DataFrame, column: str, cleaner=normalize_doi) -> int:
    if column not in data.columns:
        return 0
    values = data[column].apply(cleaner)
    values = values.loc[values != ""]
    return int(values.duplicated(keep=False).sum())


def duplicate_nonblank_groups(data: pd.DataFrame, column: str, cleaner=normalize_doi) -> int:
    if column not in data.columns:
        return 0
    values = data[column].apply(cleaner)
    values = values.loc[values != ""]
    duplicated_values = values.loc[values.duplicated(keep=False)]
    return int(duplicated_values.nunique())


def duplicate_nonblank_tag(data: pd.DataFrame, column: str, cleaner=normalize_doi) -> pd.Series:
    if column not in data.columns:
        return pd.Series([0] * len(data), index=data.index)
    values = data[column].apply(cleaner)
    duplicated = values.ne("") & values.duplicated(keep=False)
    return duplicated.astype(int)


def count_tagged_rows(data: pd.DataFrame, column: str) -> int:
    if column not in data.columns:
        return 0
    return int(pd.to_numeric(data[column], errors="coerce").fillna(0).sum())


def print_merge_summary(
    openalex: pd.DataFrame,
    crossref: pd.DataFrame,
    all_records: pd.DataFrame,
    stage_summaries: list[dict[str, int | str]],
) -> None:
    stage_summary_data = pd.DataFrame(stage_summaries)
    status_counts = all_records["record_status"].value_counts(dropna=False)

    print("\nOpenAlex/Crossref DOI-version merge summary:")
    print(f"  OpenAlex input rows: {len(openalex)}")
    print(f"  Crossref input rows: {len(crossref)}")
    print(f"  Output rows: {len(all_records)}")
    print(f"  Matched rows: {int(status_counts.get('matched', 0))}")
    print(f"  OpenAlex only rows: {int(status_counts.get('openalex_only', 0))}")
    print(f"  Crossref only rows: {int(status_counts.get('crossref_only', 0))}")
    print(f"  Rows with combined DOI: {count_nonblank(all_records, 'doi')}")
    print(f"  Rows with combined title: {count_nonblank(all_records, 'title')}")
    print(f"  Rows consolidated from duplicated titles: {count_tagged_rows(all_records, 'title_duplicate_tag')}")
    print(f"  Output duplicate DOI rows: {duplicate_nonblank_rows(all_records, 'doi')}")
    print(f"  Output duplicate DOI groups: {duplicate_nonblank_groups(all_records, 'doi')}")
    print(
        "  Output duplicate title rows: "
        f"{duplicate_nonblank_rows(all_records, 'title', normalize_title)}"
    )
    print(
        "  Output duplicate title groups: "
        f"{duplicate_nonblank_groups(all_records, 'title', normalize_title)}"
    )

    print("\nInput duplicate DOI rows by version column:")
    for column in OPENALEX_DOI_COLUMNS:
        print(f"  {column}: {duplicate_nonblank_rows(openalex, column)}")
    for column in CROSSREF_DOI_COLUMNS:
        print(f"  {column}: {duplicate_nonblank_rows(crossref, column)}")

    print("\nMatched rows by stage:")
    print(
        stage_summary_data[
            [
                "match_stage",
                "matched_rows",
                "ambiguous_openalex_rows",
                "ambiguous_crossref_rows",
            ]
        ].to_string(index=False)
    )


def print_final_duplicate_check(all_records: pd.DataFrame) -> None:
    print("\nFinal duplicate check:")
    print(f"  Rows with duplicated DOI: {duplicate_nonblank_rows(all_records, 'doi')}")
    print(
        "  Rows with duplicated title: "
        f"{duplicate_nonblank_rows(all_records, 'title', normalize_title)}"
    )


if __name__ == "__main__":
    main()
