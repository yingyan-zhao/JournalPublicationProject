from pathlib import Path
from difflib import SequenceMatcher
import os
import re
import unicodedata

import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

INPUT_CSV = (
    Path("data/processed/author_names")
    / "JEL_Training_Data_AEA_Scrape_OpenAlex_Crossref_AuthorRows_Merged.csv"
)
OUTPUT_DIR = Path("data/processed/author_names")
EXACT_AUTHOR_ROWS_OUTPUT_CSV = (
    OUTPUT_DIR / "JEL_Training_Data_AuthorRows_ExactAuthorGroups.csv"
)
FUZZY_AUTHOR_ROWS_OUTPUT_CSV = (
    OUTPUT_DIR / "JEL_Training_Data_AuthorRows_FuzzyAuthorGroups.csv"
)
FUZZY_MATCHES_OUTPUT_CSV = OUTPUT_DIR / "JEL_Training_Data_AuthorID_FuzzyMatches.csv"
PAPER_AUTHOR_OUTPUT_CSV = OUTPUT_DIR / "JEL_Training_Data_PaperAuthor_WithAuthorID.csv"
AUTHOR_CROSSWALK_OUTPUT_CSV = OUTPUT_DIR / "JEL_Training_Data_AuthorID_Crosswalk.csv"
MANUAL_REVIEW_OUTPUT_CSV = OUTPUT_DIR / "JEL_Training_Data_AuthorID_ManualReview.csv"

REQUIRED_COLUMNS = [
    "doi_full",
    "final_last_name",
    "final_first_name",
    "openalex_names",
    "scrape_authors",
    "aea_authors",
    "crossref_authors",
]


## #########################################################################
# In 12_CreateAuthorID.py, create IDs for authors.
# Step 1. Read in "JEL_Training_Data_AEA_Scrape_OpenAlex_Crossref_AuthorRows_Merged.csv"
# Step 2. Clean first and last names.
# converts names to ASCII
# removes periods, commas, semicolons, colons, numbers, non-letter characters
# collapses extra spaces
# creates lowercase matching keys with only letters and numbers
# Step 4. Create exact author groups by cleaned first + cleaned last. Export the csv
# Step 5. Create cautious fuzzy groups within same last name. Export the csv
# Step 6. Flag ambiguous groups for manual review.
# Step 7. Assign author_id.
# Step 8. Export author-level crosswalk and paper-author dataset.
## #########################################################################


def main() -> None:
    author_rows = read_author_rows(INPUT_CSV)
    author_rows = add_clean_name_columns(author_rows)

    exact_author_rows = add_exact_author_groups(author_rows)
    fuzzy_author_rows, fuzzy_matches = add_cautious_fuzzy_author_groups(
        exact_author_rows
    )
    paper_author_rows = assign_author_id(fuzzy_author_rows)
    paper_author_rows = add_manual_review_flags(paper_author_rows)
    author_crosswalk = create_author_crosswalk(paper_author_rows)
    manual_review = create_manual_review_data(paper_author_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exact_author_rows.to_csv(EXACT_AUTHOR_ROWS_OUTPUT_CSV, index=False)
    fuzzy_author_rows.to_csv(FUZZY_AUTHOR_ROWS_OUTPUT_CSV, index=False)
    fuzzy_matches.to_csv(FUZZY_MATCHES_OUTPUT_CSV, index=False)
    paper_author_rows.to_csv(PAPER_AUTHOR_OUTPUT_CSV, index=False)
    author_crosswalk.to_csv(AUTHOR_CROSSWALK_OUTPUT_CSV, index=False)
    manual_review.to_csv(MANUAL_REVIEW_OUTPUT_CSV, index=False)

    print("Create author ID summary:")
    print(f"  Input CSV: {INPUT_CSV}")
    print(f"  Rows: {len(author_rows)}")
    print(f"  Unique DOI values: {count_unique_nonblank(author_rows, 'doi_full')}")
    print(f"  Unique final last names: {count_unique_nonblank(author_rows, 'final_last_name')}")
    print(
        "  Unique final first + last name pairs: "
        f"{count_unique_name_pairs(author_rows)}"
    )
    print(f"  Unique cleaned last names: {count_unique_nonblank(author_rows, 'author_last_clean')}")
    print(
        "  Unique cleaned first + last name pairs: "
        f"{count_unique_clean_name_pairs(author_rows)}"
    )
    print(
        "  Exact author groups: "
        f"{count_unique_nonblank(exact_author_rows, 'author_exact_group_id')}"
    )
    print(
        "  Fuzzy author groups: "
        f"{count_unique_nonblank(fuzzy_author_rows, 'author_fuzzy_group_id')}"
    )
    print(f"  Fuzzy match links created: {len(fuzzy_matches)}")
    print(f"  Final author IDs: {count_unique_nonblank(paper_author_rows, 'author_id')}")
    print(f"  Author IDs flagged for manual review: {count_manual_review_author_ids(paper_author_rows)}")
    print(f"  Author rows flagged for manual review: {int(paper_author_rows['manual_review_flag'].sum())}")
    print(f"  Exact grouped author rows CSV: {EXACT_AUTHOR_ROWS_OUTPUT_CSV}")
    print(f"  Fuzzy grouped author rows CSV: {FUZZY_AUTHOR_ROWS_OUTPUT_CSV}")
    print(f"  Fuzzy matches diagnostic CSV: {FUZZY_MATCHES_OUTPUT_CSV}")
    print(f"  Paper-author output CSV: {PAPER_AUTHOR_OUTPUT_CSV}")
    print(f"  Author crosswalk output CSV: {AUTHOR_CROSSWALK_OUTPUT_CSV}")
    print(f"  Manual review diagnostic CSV: {MANUAL_REVIEW_OUTPUT_CSV}")


def read_author_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")

    data = pd.read_csv(path, dtype=str).fillna("")
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return data[REQUIRED_COLUMNS].copy()


def add_clean_name_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["author_first_clean"] = data["final_first_name"].apply(clean_name_display)
    data["author_last_clean"] = data["final_last_name"].apply(clean_name_display)
    data["author_full_clean"] = (
        data["author_first_clean"] + " " + data["author_last_clean"]
    ).str.strip()
    data["author_first_key"] = data["author_first_clean"].apply(clean_name_key)
    data["author_last_key"] = data["author_last_clean"].apply(clean_name_key)
    data["author_name_key"] = (
        data["author_last_key"] + "_" + data["author_first_key"]
    ).str.strip("_")
    return data


def add_exact_author_groups(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    unique_keys = sorted(
        key for key in data["author_name_key"].drop_duplicates().tolist() if key
    )
    exact_id_by_key = {
        key: f"AEX{index:06d}" for index, key in enumerate(unique_keys, start=1)
    }
    data["author_exact_group_id"] = data["author_name_key"].map(exact_id_by_key).fillna("")
    return data


def add_cautious_fuzzy_author_groups(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()
    exact_groups = (
        data.loc[data["author_name_key"].ne("")]
        .groupby("author_name_key", as_index=False)
        .agg(
            author_last_key=("author_last_key", "first"),
            author_first_key=("author_first_key", "first"),
            author_last_clean=("author_last_clean", longest_value),
            author_first_clean=("author_first_clean", longest_value),
            row_count=("author_name_key", "size"),
        )
    )

    union_find = UnionFind(exact_groups["author_name_key"].tolist())
    fuzzy_rows = []

    for _, last_name_group in exact_groups.groupby("author_last_key", sort=False):
        if len(last_name_group) < 2:
            continue

        first_keys = last_name_group["author_first_key"].tolist()
        records = last_name_group.to_dict("records")
        for left_index in range(len(records)):
            for right_index in range(left_index + 1, len(records)):
                left = records[left_index]
                right = records[right_index]
                reason = cautious_fuzzy_reason(
                    left["author_first_key"],
                    right["author_first_key"],
                    first_keys,
                )
                if not reason:
                    continue

                union_find.union(left["author_name_key"], right["author_name_key"])
                fuzzy_rows.append(
                    {
                        "left_author_name_key": left["author_name_key"],
                        "right_author_name_key": right["author_name_key"],
                        "author_last_key": left["author_last_key"],
                        "left_first_name": left["author_first_clean"],
                        "right_first_name": right["author_first_clean"],
                        "match_reason": reason,
                        "first_name_similarity": round(
                            name_similarity(
                                left["author_first_key"],
                                right["author_first_key"],
                            ),
                            4,
                        ),
                    }
                )

    root_by_key = {
        key: union_find.find(key)
        for key in exact_groups["author_name_key"].tolist()
    }
    unique_roots = sorted(set(root_by_key.values()))
    fuzzy_id_by_root = {
        root: f"AFZ{index:06d}" for index, root in enumerate(unique_roots, start=1)
    }
    data["author_fuzzy_group_id"] = data["author_name_key"].map(
        lambda key: fuzzy_id_by_root.get(root_by_key.get(key, ""), "")
    )

    fuzzy_key_set = set()
    for row in fuzzy_rows:
        fuzzy_key_set.add(row["left_author_name_key"])
        fuzzy_key_set.add(row["right_author_name_key"])
    data["author_group_method"] = data["author_name_key"].apply(
        lambda key: "cautious_fuzzy" if key in fuzzy_key_set else "exact"
    )

    fuzzy_matches = pd.DataFrame(
        fuzzy_rows,
        columns=[
            "left_author_name_key",
            "right_author_name_key",
            "author_last_key",
            "left_first_name",
            "right_first_name",
            "match_reason",
            "first_name_similarity",
        ],
    )
    return data, fuzzy_matches


def cautious_fuzzy_reason(
    left_first_key: str,
    right_first_key: str,
    first_keys_in_last_name_group: list[str],
) -> str:
    if not left_first_key or not right_first_key:
        return ""
    if left_first_key == right_first_key:
        return ""

    shorter, longer = sorted([left_first_key, right_first_key], key=len)
    if len(shorter) >= 3 and longer.startswith(shorter):
        return "first_name_prefix"

    if (
        left_first_key[0] == right_first_key[0]
        and min(len(left_first_key), len(right_first_key)) >= 4
        and name_similarity(left_first_key, right_first_key) >= 0.92
    ):
        return "high_first_name_similarity"

    if len(shorter) == 1 and longer.startswith(shorter):
        full_names_with_initial = {
            first_key
            for first_key in first_keys_in_last_name_group
            if len(first_key) > 1 and first_key.startswith(shorter)
        }
        if len(full_names_with_initial) == 1:
            return "unique_initial_with_same_last_name"

    return ""


def name_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def assign_author_id(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["author_id"] = data["author_fuzzy_group_id"]
    return data


def add_manual_review_flags(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["_manual_review_reasons_list"] = [[] for _ in range(len(data))]

    add_row_reason(data, data["author_first_key"].eq(""), "blank_first_name")
    add_row_reason(data, data["author_last_key"].eq(""), "blank_last_name")
    add_row_reason(
        data,
        data["author_first_key"].str.len().eq(1),
        "single_letter_first_name",
    )
    add_row_reason(
        data,
        data["author_group_method"].eq("cautious_fuzzy"),
        "cautious_fuzzy_group",
    )

    fuzzy_group_size = data.groupby("author_id")["author_name_key"].transform("nunique")
    add_row_reason(data, fuzzy_group_size.gt(1), "multiple_name_keys_in_author_id")

    same_initial_author_ids = data.groupby(
        ["author_last_key", data["author_first_key"].str[:1]]
    )["author_id"].transform("nunique")
    add_row_reason(
        data,
        data["author_first_key"].ne("") & same_initial_author_ids.gt(1),
        "same_last_name_and_first_initial_multiple_author_ids",
    )

    last_name_rows = data.groupby("author_last_key")["author_id"].transform("size")
    add_row_reason(data, last_name_rows.ge(50), "common_last_name")

    data["manual_review_reasons"] = data["_manual_review_reasons_list"].apply(
        lambda reasons: "; ".join(reasons)
    )
    data["manual_review_flag"] = data["manual_review_reasons"].ne("").astype(int)
    return data.drop(columns=["_manual_review_reasons_list"])


def add_row_reason(data: pd.DataFrame, mask: pd.Series, reason: str) -> None:
    for index in data.index[mask.fillna(False)]:
        data.at[index, "_manual_review_reasons_list"].append(reason)


def create_author_crosswalk(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.loc[data["author_id"].ne("")]
        .groupby("author_id", as_index=False)
        .agg(
            author_last_clean=("author_last_clean", longest_value),
            author_first_clean=("author_first_clean", longest_value),
            author_last_key=("author_last_key", first_nonblank),
            author_first_keys=("author_first_key", join_unique_values),
            author_name_keys=("author_name_key", join_unique_values),
            all_name_versions=("author_full_clean", join_unique_values),
            n_author_rows=("author_id", "size"),
            n_papers=("doi_full", count_unique_values),
            author_group_method=("author_group_method", join_unique_values),
            manual_review_flag=("manual_review_flag", "max"),
            manual_review_reasons=("manual_review_reasons", join_unique_values),
        )
    )


def create_manual_review_data(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "author_id",
        "doi_full",
        "final_last_name",
        "final_first_name",
        "author_last_clean",
        "author_first_clean",
        "author_last_key",
        "author_first_key",
        "author_name_key",
        "author_group_method",
        "manual_review_reasons",
        "openalex_names",
        "scrape_authors",
        "aea_authors",
        "crossref_authors",
    ]
    return (
        data.loc[data["manual_review_flag"].eq(1), columns]
        .sort_values(["author_last_key", "author_first_key", "author_id", "doi_full"])
        .copy()
    )


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        keep_root = min(left_root, right_root)
        other_root = max(left_root, right_root)
        self.parent[other_root] = keep_root


def clean_name_display(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = to_ascii(text)
    text = re.sub(r"[.,;:]+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^A-Za-z ]+", " ", text)
    return " ".join(text.split())


def clean_name_key(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    return re.sub(r"[^a-z0-9]", "", text)


def to_ascii(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def longest_value(values: pd.Series) -> str:
    cleaned_values = [str(value).strip() for value in values if str(value).strip()]
    if not cleaned_values:
        return ""
    return max(cleaned_values, key=len)


def first_nonblank(values: pd.Series) -> str:
    for value in values:
        text = str(value).strip()
        if text:
            return text
    return ""


def join_unique_values(values: pd.Series) -> str:
    seen = set()
    unique_values = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        unique_values.append(text)
        seen.add(text)
    return "; ".join(unique_values)


def count_unique_values(values: pd.Series) -> int:
    cleaned_values = [str(value).strip() for value in values if str(value).strip()]
    return len(set(cleaned_values))


def count_unique_nonblank(data: pd.DataFrame, column: str) -> int:
    values = data[column].fillna("").astype(str).str.strip()
    return int(values.loc[values.ne("")].nunique())


def count_unique_name_pairs(data: pd.DataFrame) -> int:
    names = pd.DataFrame(
        {
            "final_last_name": data["final_last_name"].fillna("").astype(str).str.strip(),
            "final_first_name": data["final_first_name"].fillna("").astype(str).str.strip(),
        }
    )
    names = names.loc[
        names["final_last_name"].ne("") | names["final_first_name"].ne("")
    ]
    return len(names.drop_duplicates())


def count_unique_clean_name_pairs(data: pd.DataFrame) -> int:
    names = data[["author_last_key", "author_first_key"]].copy()
    names = names.loc[
        names["author_last_key"].ne("") | names["author_first_key"].ne("")
    ]
    return len(names.drop_duplicates())


def count_manual_review_author_ids(data: pd.DataFrame) -> int:
    flagged = data.loc[data["manual_review_flag"].eq(1), "author_id"]
    return int(flagged.loc[flagged.ne("")].nunique())


if __name__ == "__main__":
    main()
