from pathlib import Path
import math
import os

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

INPUT_CSV = (
    Path("data/processed")
    / "JEL_Training_Data_PaperAuthor_WithAuthorID_JEL_Merged_Top5.csv"
)
OUTPUT_DIR = Path("outputs/figures/by_field")
ANNUAL_OUTPUT_CSV = (
    OUTPUT_DIR / "Graph1_FieldTop10AuthorShares_Annual.csv"
)
PERIOD_OUTPUT_CSV = (
    OUTPUT_DIR
    / "Graph1_FieldTop10AuthorShares_1995_2000_vs_2020_2025.csv"
)
GRAPH_OUTPUT_PNG = (
    OUTPUT_DIR
    / "Graph1_FieldTop10AuthorShares_1995_2000_vs_2020_2025.png"
)
NEW_AUTHOR_OUTPUT_CSV = (
    OUTPUT_DIR
    / "Graph2_FieldNewAuthorShares_1995_2000_vs_2020_2025.csv"
)
NEW_AUTHOR_GRAPH_OUTPUT_PNG = (
    OUTPUT_DIR
    / "Graph2_FieldNewAuthorShares_1995_2000_vs_2020_2025.png"
)
COAUTHOR_COMPOSITION_OUTPUT_CSV = (
    OUTPUT_DIR
    / "Graph3_FieldNewAuthorCoauthorComposition_1995_2000_vs_2020_2025.csv"
)
COAUTHOR_COMPOSITION_GRAPH_OUTPUT_PNG = (
    OUTPUT_DIR
    / "Graph3_FieldNewAuthorCoauthorComposition_1995_2000_vs_2020_2025.png"
)
PUBLICATION_GAP_OUTPUT_CSV = (
    OUTPUT_DIR
    / "Graph4_FieldFirstToSecondPublicationGap_1995_2000_vs_2010_2015.csv"
)
PUBLICATION_GAP_GRAPH_OUTPUT_PNG = (
    OUTPUT_DIR
    / "Graph4_FieldFirstToSecondPublicationGap_1995_2000_vs_2010_2015.png"
)

# Focus on the ten requested JEL fields.
FIELD_CODES = ("D", "C", "E", "J", "H", "O", "L", "G", "F", "I")
FIELD_NAMES = {
    "D": "Microeconomics",
    "C": "Quantitative methods",
    "E": "Macroeconomics",
    "J": "Labor economics",
    "H": "Public economics",
    "O": "Development and growth",
    "L": "Industrial organization",
    "G": "Financial economics",
    "F": "International economics",
    "I": "Health, education, and welfare",
}
PERIODS = {
    "1995-2000": (1995, 2000),
    "2020-2025": (2020, 2025),
}
PUBLICATION_GAP_PERIODS = {
    "1995-2000": (1995, 2000),
    "2010-2015": (2010, 2015),
}
PUBLICATION_GAP_FOLLOWUP_YEARS = 11
ROLLING_WINDOW_YEARS = 20
TOP_AUTHOR_PERCENTAGE = 10
BASELINE_COLOR = "#A8AFB7"
CHANGE_COLOR = "#087E73"
NEWCOMER_COLOR = "#E67E22"
LIGHT_TEAL_COLOR = "#8BC9C4"
EARLIER_CIRCLE_AREA = 68
LATER_CIRCLE_AREA = EARLIER_CIRCLE_AREA * 3
EARLIER_LEGEND_MARKER_SIZE = 7
LATER_LEGEND_MARKER_SIZE = EARLIER_LEGEND_MARKER_SIZE * math.sqrt(3)


#############################################################################
# In 03_ExploreConcentrationAuthorByFields.py, Starting from “JEL_Training_Data_PaperAuthor_WithAuthorID_JEL_Merged_Top5.csv”, we are going to explore patterns by fields. We are going to only focus on top 10 fields [D, C, E, J, H, O, L, G, F, I]. A paper can be in different categories, if it has multiple jel codes.
#
# Step 1. I want a horizontal dumbbell chart. Each line is a field. One each line there are two circles. One for the share of papers include top-ranked (top 10% of authors) authors in 2020-2025, and the other for the share of papers include top-ranked (top 10% of authors) authors in 1995-2000. Make the circles teal. Make 1995-200 circle lighter and 2020-2025 circle darker and 3 times bigger. Add an arrow, which shows the direction of changes from 1995-2000 number to 2020-2025 number. Add the percentage difference above the arrow. Rank the line by letters of each field.
#
# Step 2. I want a horizontal dumbbell chart. Each line is a field. One each line there are two circles.  One for the share of New authors among the total number of unique author names in 1995-2000, and the other for 2020-2025. Make the circles teal. Make the circles teal. Make 1995-200 circle lighter and 2020-2025 circle darker and 3 times bigger. Add an arrow, which shows the direction of changes from 1995-2000 number to 2020-2025 number. Add the percentage difference above the arrow. Rank the line by letters of each field.
#
# Step 3. I want a horizontal dumbbell chart. Each field will have three lines. One line for the share of New authors publishing their first top-five with experienced coauthors, one line for the share of new authors publishing with new comers and the other line for the share of solo-authored papers. On each line there are two circles, one is the share during 1995-2000 and the other one is the share during 2020-2025. Draw an arrow from the circle  of 1995-2000 to the circle of 2010-2015. Use teal for the share of New authors publishing their first top-five with experienced coauthors, use orange for the share of new authors publishing with new comers  and use grey for the share of solo-authored papers. The circle for 1995-2000 is lighter and the circle for 2020-2025 is darker. Also make the circle for 2020-2025   3 times larger. Add the change in percentage points over each line. Rank the dumbbell by letters of each field.
#
#
# Step 4. I want a horizontal dumbbell chart. Each line will be a field. The line measures the number of years.
# On each line there are two circles. One is the average gap of years between the first and second top-five publication in 1995-2000,
# the other is the average gap of years between the first and second top-five publication in  2010-2015.
# Draw an arrow from the circle  of 1995-2000 to the circle of 2010-2015. Rank the line by letters of each field.
# For each paper, we only look at the second paper that is published within 11 years after the first paper.
# Add a 95 confidence interval for each point.
#############################################################################


def main() -> None:
    # Step 1. Compare field-specific concentration in the two requested periods.
    data = read_input_data(INPUT_CSV)
    field_author_rows = prepare_field_author_rows(data)
    annual = calculate_annual_top_author_shares(field_author_rows)
    period_summary = summarize_period_shares(annual)

    write_csv(annual, ANNUAL_OUTPUT_CSV)
    write_csv(period_summary, PERIOD_OUTPUT_CSV)
    plot_period_comparison(period_summary, GRAPH_OUTPUT_PNG)

    multifield_papers = count_multifield_papers(data)
    print("Step 1 complete:")
    print(f"  Input paper-author rows: {len(data):,}")
    print(
        "  Unique papers represented in more than one requested field: "
        f"{multifield_papers:,}"
    )
    print(f"  Fields analyzed: {', '.join(FIELD_CODES)}")
    print(f"  Annual results: {ANNUAL_OUTPUT_CSV}")
    print(f"  Period comparison: {PERIOD_OUTPUT_CSV}")
    print(f"  Horizontal dumbbell graph: {GRAPH_OUTPUT_PNG}")

    # Step 2. Compare field-specific new-author shares in the two periods.
    new_author_summary = calculate_new_author_period_shares(field_author_rows)
    write_csv(new_author_summary, NEW_AUTHOR_OUTPUT_CSV)
    plot_new_author_period_comparison(
        new_author_summary,
        NEW_AUTHOR_GRAPH_OUTPUT_PNG,
    )
    print("Step 2 complete:")
    print(f"  New-author comparison: {NEW_AUTHOR_OUTPUT_CSV}")
    print(f"  Horizontal dumbbell graph: {NEW_AUTHOR_GRAPH_OUTPUT_PNG}")

    # Step 3. Compare how new authors enter each field in the two periods.
    coauthor_composition = calculate_new_author_coauthor_composition(
        field_author_rows
    )
    write_csv(coauthor_composition, COAUTHOR_COMPOSITION_OUTPUT_CSV)
    plot_new_author_coauthor_composition(
        coauthor_composition,
        COAUTHOR_COMPOSITION_GRAPH_OUTPUT_PNG,
    )
    print("Step 3 complete:")
    print(f"  Coauthor composition: {COAUTHOR_COMPOSITION_OUTPUT_CSV}")
    print(f"  Horizontal dumbbell graph: {COAUTHOR_COMPOSITION_GRAPH_OUTPUT_PNG}")

    # Step 4. Compare observed first-to-second publication gaps by field.
    publication_gaps = calculate_first_to_second_gap_by_field(field_author_rows)
    write_csv(publication_gaps, PUBLICATION_GAP_OUTPUT_CSV)
    plot_first_to_second_gap_by_field(
        publication_gaps,
        PUBLICATION_GAP_GRAPH_OUTPUT_PNG,
    )
    print("Step 4 complete:")
    print(f"  Publication-gap comparison: {PUBLICATION_GAP_OUTPUT_CSV}")
    print(f"  Horizontal dumbbell graph: {PUBLICATION_GAP_GRAPH_OUTPUT_PNG}")


def read_input_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def prepare_field_author_rows(data: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "doi_full",
        "publication_year",
        "author_id",
        "Final_jel_code",
    ]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    prepared = data[required_columns].copy()
    prepared["doi_full"] = clean_text_series(prepared["doi_full"])
    prepared["author_id"] = clean_text_series(prepared["author_id"])
    prepared["publication_year"] = pd.to_numeric(
        prepared["publication_year"],
        errors="coerce",
    )
    prepared["jel_field"] = prepared["Final_jel_code"].apply(split_jel_fields)
    prepared = prepared.explode("jel_field", ignore_index=True)
    prepared = prepared.loc[
        prepared["doi_full"].ne("")
        & prepared["author_id"].ne("")
        & prepared["publication_year"].notna()
        & prepared["jel_field"].isin(FIELD_CODES)
    ].copy()
    prepared["publication_year"] = prepared["publication_year"].astype(int)
    prepared["field_name"] = prepared["jel_field"].map(FIELD_NAMES)

    return prepared.drop_duplicates(
        ["jel_field", "doi_full", "author_id"],
        keep="first",
    ).reset_index(drop=True)


def clean_text_series(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip()


def split_jel_fields(value) -> list[str]:
    fields = []
    for part in str(value or "").split(";"):
        field = part.strip().upper()
        if field and field not in fields:
            fields.append(field)
    return fields


def count_multifield_papers(data: pd.DataFrame) -> int:
    paper_fields = (
        data[["doi_full", "Final_jel_code"]]
        .drop_duplicates("doi_full")
        .copy()
    )
    requested_field_count = paper_fields["Final_jel_code"].apply(
        lambda value: len(set(split_jel_fields(value)).intersection(FIELD_CODES))
    )
    return int(requested_field_count.gt(1).sum())


def calculate_annual_top_author_shares(
    field_author_rows: pd.DataFrame,
) -> pd.DataFrame:
    target_years = sorted(
        {
            year
            for start_year, end_year in PERIODS.values()
            for year in range(start_year, end_year + 1)
        }
    )
    result_rows = []

    for field_number, field in enumerate(FIELD_CODES, start=1):
        field_data = field_author_rows.loc[
            field_author_rows["jel_field"].eq(field)
        ].copy()
        for year in target_years:
            result_rows.append(
                calculate_field_year_share(field_data, field, year)
            )
        print(
            f"  Calculated field {field} "
            f"({field_number}/{len(FIELD_CODES)})",
            flush=True,
        )

    return pd.DataFrame(result_rows)


def calculate_field_year_share(
    field_data: pd.DataFrame,
    field: str,
    year: int,
) -> dict[str, object]:
    history_start_year = year - ROLLING_WINDOW_YEARS
    history_end_year = year - 1
    history = field_data.loc[
        field_data["publication_year"].between(
            history_start_year,
            history_end_year,
            inclusive="both",
        )
    ]
    current = field_data.loc[field_data["publication_year"].eq(year)]
    author_counts = (
        history.groupby("author_id")["doi_full"]
        .nunique()
        .sort_values(ascending=False, kind="mergesort")
    )
    top_authors, cutoff, nominal_count = select_top_authors(
        author_counts,
        TOP_AUTHOR_PERCENTAGE,
    )
    total_papers = int(current["doi_full"].nunique())
    papers_with_top_author = int(
        current.loc[current["author_id"].isin(top_authors), "doi_full"].nunique()
    )

    return {
        "jel_field": field,
        "field_name": FIELD_NAMES[field],
        "publication_year": year,
        "history_start_year": history_start_year,
        "history_end_year": history_end_year,
        "top_author_percentage": TOP_AUTHOR_PERCENTAGE,
        "number_of_eligible_authors": len(author_counts),
        "nominal_number_of_top_authors": nominal_count,
        "number_of_top_authors_including_ties": len(top_authors),
        "publication_count_cutoff": cutoff,
        "total_unique_papers": total_papers,
        "unique_papers_with_top_author": papers_with_top_author,
        "share_of_papers_with_top_author": (
            papers_with_top_author / total_papers if total_papers else 0.0
        ),
    }


def select_top_authors(
    author_counts: pd.Series,
    percentage: int,
) -> tuple[list[str], int | str, int]:
    if author_counts.empty:
        return [], "", 0
    nominal_count = max(1, math.ceil(len(author_counts) * percentage / 100))
    cutoff = int(author_counts.iloc[nominal_count - 1])
    selected = author_counts.loc[author_counts.ge(cutoff)]
    return selected.index.astype(str).tolist(), cutoff, nominal_count


def summarize_period_shares(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field in FIELD_CODES:
        field_annual = annual.loc[annual["jel_field"].eq(field)]
        row = {
            "jel_field": field,
            "field_name": FIELD_NAMES[field],
        }
        for period_label, (start_year, end_year) in PERIODS.items():
            period_data = field_annual.loc[
                field_annual["publication_year"].between(
                    start_year,
                    end_year,
                    inclusive="both",
                )
            ]
            total_papers = int(period_data["total_unique_papers"].sum())
            papers_with_top_author = int(
                period_data["unique_papers_with_top_author"].sum()
            )
            prefix = period_label.replace("-", "_")
            row[f"total_papers_{prefix}"] = total_papers
            row[f"papers_with_top_author_{prefix}"] = papers_with_top_author
            row[f"share_{prefix}"] = (
                papers_with_top_author / total_papers if total_papers else 0.0
            )
        row["change_percentage_points"] = 100 * (
            row["share_2020_2025"] - row["share_1995_2000"]
        )
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["recent_share_rank"] = (
        summary["share_2020_2025"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return summary.sort_values(
        ["recent_share_rank", "jel_field"],
        kind="mergesort",
    ).reset_index(drop=True)


def calculate_new_author_period_shares(
    field_author_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for field in FIELD_CODES:
        field_data = field_author_rows.loc[
            field_author_rows["jel_field"].eq(field)
        ].copy()
        first_field_year = field_data.groupby("author_id")["publication_year"].min()
        row = {
            "jel_field": field,
            "field_name": FIELD_NAMES[field],
        }

        for period_label, (start_year, end_year) in PERIODS.items():
            period_data = field_data.loc[
                field_data["publication_year"].between(
                    start_year,
                    end_year,
                    inclusive="both",
                )
            ]
            active_authors = set(period_data["author_id"])
            new_authors = {
                author_id
                for author_id in active_authors
                if start_year <= first_field_year.get(author_id, -1) <= end_year
            }
            prefix = period_label.replace("-", "_")
            row[f"total_unique_authors_{prefix}"] = len(active_authors)
            row[f"new_authors_{prefix}"] = len(new_authors)
            row[f"new_author_share_{prefix}"] = (
                len(new_authors) / len(active_authors) if active_authors else 0.0
            )

        row["change_percentage_points"] = 100 * (
            row["new_author_share_2020_2025"]
            - row["new_author_share_1995_2000"]
        )
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["recent_share_rank"] = (
        summary["new_author_share_2020_2025"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return summary.sort_values(
        ["recent_share_rank", "jel_field"],
        kind="mergesort",
    ).reset_index(drop=True)


def calculate_new_author_coauthor_composition(
    field_author_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for field in FIELD_CODES:
        field_data = field_author_rows.loc[
            field_author_rows["jel_field"].eq(field)
        ].copy()
        first_field_year = field_data.groupby("author_id")["publication_year"].min()
        paper_authors = (
            field_data.groupby("doi_full")["author_id"].apply(set).to_dict()
        )
        entry_papers = (
            field_data.loc[
                field_data["publication_year"].eq(
                    field_data["author_id"].map(first_field_year)
                )
            ]
            .groupby("author_id")["doi_full"]
            .apply(lambda values: set(values))
            .to_dict()
        )
        author_categories = {
            author_id: classify_entry_coauthors(
                author_id=author_id,
                entry_year=int(entry_year),
                entry_dois=entry_papers.get(author_id, set()),
                paper_authors=paper_authors,
                first_field_year=first_field_year,
            )
            for author_id, entry_year in first_field_year.items()
        }

        for period_label, (start_year, end_year) in PERIODS.items():
            new_authors = {
                author_id
                for author_id, entry_year in first_field_year.items()
                if start_year <= entry_year <= end_year
            }
            category_counts = pd.Series(
                [author_categories[author_id] for author_id in new_authors],
                dtype="object",
            ).value_counts()
            total_new_authors = len(new_authors)
            experienced_count = int(
                category_counts.get("experienced_coauthor", 0)
            )
            newcomer_count = int(
                category_counts.get("newcomer_coauthors_only", 0)
            )
            solo_count = int(category_counts.get("solo_authored", 0))
            rows.append(
                {
                    "jel_field": field,
                    "field_name": FIELD_NAMES[field],
                    "period": period_label,
                    "period_start_year": start_year,
                    "period_end_year": end_year,
                    "total_new_authors": total_new_authors,
                    "experienced_coauthor_count": experienced_count,
                    "newcomer_coauthors_only_count": newcomer_count,
                    "solo_authored_count": solo_count,
                    "experienced_coauthor_share": safe_share(
                        experienced_count,
                        total_new_authors,
                    ),
                    "newcomer_coauthors_only_share": safe_share(
                        newcomer_count,
                        total_new_authors,
                    ),
                    "solo_authored_share": safe_share(
                        solo_count,
                        total_new_authors,
                    ),
                }
            )

    period_order = {period: order for order, period in enumerate(PERIODS)}
    summary = pd.DataFrame(rows)
    summary["_period_order"] = summary["period"].map(period_order)
    return summary.sort_values(
        ["jel_field", "_period_order"],
        kind="mergesort",
    ).drop(columns="_period_order").reset_index(drop=True)


def classify_entry_coauthors(
    author_id: str,
    entry_year: int,
    entry_dois: set[str],
    paper_authors: dict[str, set[str]],
    first_field_year: pd.Series,
) -> str:
    has_coauthor = False
    has_newcomer_coauthor = False
    for doi in entry_dois:
        coauthors = paper_authors.get(doi, set()) - {author_id}
        if not coauthors:
            continue
        has_coauthor = True
        has_experienced_coauthor = any(
            first_field_year.get(coauthor, entry_year) < entry_year
            for coauthor in coauthors
        )
        if has_experienced_coauthor:
            return "experienced_coauthor"
        if any(first_field_year.get(coauthor) == entry_year for coauthor in coauthors):
            has_newcomer_coauthor = True

    if has_newcomer_coauthor:
        return "newcomer_coauthors_only"
    if has_coauthor:
        return "newcomer_coauthors_only"
    return "solo_authored"


def safe_share(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_gap_values(values: pd.Series) -> dict[str, float | int]:
    from scipy.stats import t as student_t

    gaps = pd.to_numeric(values, errors="coerce").dropna()
    count = int(len(gaps))
    if count == 0:
        return {
            "number_of_gap_observations": 0,
            "average_gap_years": pd.NA,
            "gap_standard_deviation": pd.NA,
            "gap_standard_error": pd.NA,
            "gap_ci_95_lower": pd.NA,
            "gap_ci_95_upper": pd.NA,
            "median_gap_years": pd.NA,
            "share_same_year": pd.NA,
        }

    mean = float(gaps.mean())
    median = float(gaps.median())
    share_same_year = float(gaps.eq(0).mean())
    if count == 1:
        standard_deviation = pd.NA
        standard_error = pd.NA
        lower = pd.NA
        upper = pd.NA
    else:
        standard_deviation = float(gaps.std(ddof=1))
        standard_error = standard_deviation / math.sqrt(count)
        critical_value = float(student_t.ppf(0.975, count - 1))
        margin = critical_value * standard_error
        lower = mean - margin
        upper = mean + margin

    return {
        "number_of_gap_observations": count,
        "average_gap_years": mean,
        "gap_standard_deviation": standard_deviation,
        "gap_standard_error": standard_error,
        "gap_ci_95_lower": lower,
        "gap_ci_95_upper": upper,
        "median_gap_years": median,
        "share_same_year": share_same_year,
    }


def calculate_first_to_second_gap_by_field(
    field_author_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    observation_end_year = int(field_author_rows["publication_year"].max())
    for field in FIELD_CODES:
        field_data = (
            field_author_rows.loc[field_author_rows["jel_field"].eq(field)]
            .drop_duplicates(["author_id", "doi_full"])
            .sort_values(
                ["author_id", "publication_year", "doi_full"],
                kind="mergesort",
            )
            .copy()
        )
        field_data["publication_number"] = (
            field_data.groupby("author_id").cumcount() + 1
        )
        first_publication_year = field_data.groupby("author_id")[
            "publication_year"
        ].min()
        second_publications = field_data.loc[
            field_data["publication_number"].eq(2)
        ].copy()
        second_publications["first_publication_year"] = second_publications[
            "author_id"
        ].map(first_publication_year)
        second_publications["gap_years"] = (
            second_publications["publication_year"]
            - second_publications["first_publication_year"]
        )

        for period_label, (start_year, end_year) in PUBLICATION_GAP_PERIODS.items():
            period_authors = first_publication_year.loc[
                first_publication_year.between(
                    start_year,
                    end_year,
                    inclusive="both",
                )
            ]
            complete_followup = (
                period_authors + PUBLICATION_GAP_FOLLOWUP_YEARS
            ).le(observation_end_year)
            cohort_authors = period_authors.loc[complete_followup]
            observed_second = second_publications.loc[
                second_publications["author_id"].isin(cohort_authors.index)
                & second_publications["gap_years"].between(
                    0,
                    PUBLICATION_GAP_FOLLOWUP_YEARS,
                    inclusive="both",
                )
            ]
            gap_values = observed_second["gap_years"]
            gap_statistics = summarize_gap_values(gap_values)
            authors_with_second = int(
                observed_second["author_id"].nunique()
            )
            rows.append(
                {
                    "jel_field": field,
                    "field_name": FIELD_NAMES[field],
                    "entry_period": period_label,
                    "entry_period_start_year": start_year,
                    "entry_period_end_year": end_year,
                    "followup_window_years": (
                        PUBLICATION_GAP_FOLLOWUP_YEARS
                    ),
                    "observation_end_year": observation_end_year,
                    "complete_followup_window_required": 1,
                    "entry_authors_before_followup_requirement": int(
                        len(period_authors)
                    ),
                    "total_entry_cohort_authors": int(len(cohort_authors)),
                    "authors_excluded_incomplete_followup": int(
                        len(period_authors) - len(cohort_authors)
                    ),
                    "authors_with_second_publication_within_window": (
                        authors_with_second
                    ),
                    "share_with_second_publication_within_window": safe_share(
                        authors_with_second,
                        int(len(cohort_authors)),
                    ),
                    **gap_statistics,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["jel_field", "entry_period"],
        kind="mergesort",
    ).reset_index(drop=True)


def plot_period_comparison(data: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    field_order = sorted(FIELD_CODES)
    plot_data = data.set_index("jel_field").reindex(field_order).reset_index()
    old_share = plot_data["share_1995_2000"].astype(float)
    new_share = plot_data["share_2020_2025"].astype(float)
    y_positions = list(range(len(plot_data)))
    old_color = LIGHT_TEAL_COLOR

    figure, axis = plt.subplots(figsize=(12, 7.8))
    for y_position, (_, row) in enumerate(plot_data.iterrows()):
        baseline = float(row["share_1995_2000"])
        recent = float(row["share_2020_2025"])
        change = float(row["change_percentage_points"])
        axis.annotate(
            "",
            xy=(recent, y_position),
            xytext=(baseline, y_position),
            arrowprops={
                "arrowstyle": "-|>",
                "color": CHANGE_COLOR,
                "linewidth": 1.5,
                "mutation_scale": 10,
                "shrinkA": 7,
                "shrinkB": 7,
            },
            zorder=2,
        )
        axis.scatter(
            baseline,
            y_position,
            s=EARLIER_CIRCLE_AREA,
            color=old_color,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        axis.scatter(
            recent,
            y_position,
            s=LATER_CIRCLE_AREA,
            color=CHANGE_COLOR,
            edgecolor="white",
            linewidth=1.0,
            zorder=4,
        )
        axis.text(
            baseline - 0.009,
            y_position,
            f"{baseline:.0%}",
            va="center",
            ha="right",
            color="#4A7C78",
            fontsize=9.5,
            zorder=5,
        )
        axis.text(
            recent + 0.009,
            y_position,
            f"{recent:.0%}",
            va="center",
            ha="left",
            color="#075F58",
            fontsize=9.5,
            fontweight="semibold",
            zorder=5,
        )
        axis.text(
            (baseline + recent) / 2,
            y_position - 0.18,
            f"{change:+.1f} pp",
            va="bottom",
            ha="center",
            color="#075F58",
            fontsize=8.5,
            fontweight="semibold",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
                "pad": 0.6,
            },
            zorder=5,
        )

    labels = [
        f"{row.jel_field}  {row.field_name}"
        for row in plot_data.itertuples(index=False)
    ]
    axis.set_yticks(y_positions, labels=labels)
    axis.invert_yaxis()
    axis.set_xlabel("Share of papers with at least one field-specific top-10% author")
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    maximum_share = float(
        plot_data[["share_1995_2000", "share_2020_2025"]].max().max()
    )
    axis.set_xlim(0, min(1.0, maximum_share + 0.10))
    axis.set_title(
        "Top-ranked authors appear on a growing share of papers across fields",
        loc="left",
        pad=38,
        fontsize=17,
        fontweight="semibold",
    )
    axis.text(
        0,
        1.015,
        (
            "Field-specific top 10% based on publication counts during the "
            "preceding 20 years"
        ),
        transform=axis.transAxes,
        color="#69727D",
        fontsize=11,
        ha="left",
        va="bottom",
    )
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=old_color,
                markeredgecolor="white",
                markersize=EARLIER_LEGEND_MARKER_SIZE,
                label="1995-2000 (lighter circle)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=CHANGE_COLOR,
                markeredgecolor="white",
                markersize=LATER_LEGEND_MARKER_SIZE,
                label="2020-2025 (3x-area, darker circle)",
            ),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
    )
    axis.grid(axis="x", color="#D8DDE3", linewidth=0.8, alpha=0.65, zorder=1)
    axis.grid(axis="y", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color("#B8C0C8")
    axis.tick_params(axis="y", length=0, colors="#34404B")
    axis.tick_params(axis="x", colors="#59636E")
    figure.text(
        0.01,
        0.012,
        (
            "Note: Rankings are recalculated annually within each field; ties "
            "at the top-10% cutoff are included. Period shares pool unique "
            "papers across six years. Papers with multiple JEL fields count "
            "once in each field. Arrows point from 1995-2000 to 2020-2025; "
            "labels above arrows report percentage-point changes."
        ),
        ha="left",
        va="bottom",
        fontsize=9,
        color="#69727D",
        wrap=True,
    )
    figure.subplots_adjust(left=0.29, right=0.97, top=0.84, bottom=0.20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_new_author_period_comparison(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    field_order = sorted(FIELD_CODES)
    plot_data = data.set_index("jel_field").reindex(field_order).reset_index()
    y_positions = list(range(len(plot_data)))

    figure, axis = plt.subplots(figsize=(12, 7.8))
    for y_position, (_, row) in enumerate(plot_data.iterrows()):
        baseline = float(row["new_author_share_1995_2000"])
        recent = float(row["new_author_share_2020_2025"])
        change = float(row["change_percentage_points"])
        close_values = abs(recent - baseline) < 0.03
        arrow_y_position = y_position - 0.11 if close_values else y_position
        arrow_properties = {
            "arrowstyle": "-|>",
            "color": CHANGE_COLOR,
            "linewidth": 1.5,
            "mutation_scale": 10,
            "shrinkA": 0 if close_values else 7,
            "shrinkB": 0 if close_values else 9,
        }
        if close_values:
            axis.plot(
                [baseline, baseline],
                [y_position, arrow_y_position],
                color=CHANGE_COLOR,
                linewidth=0.8,
                zorder=2,
            )
            axis.plot(
                [recent, recent],
                [y_position, arrow_y_position],
                color=CHANGE_COLOR,
                linewidth=0.8,
                zorder=2,
            )
        axis.annotate(
            "",
            xy=(recent, arrow_y_position),
            xytext=(baseline, arrow_y_position),
            arrowprops=arrow_properties,
            zorder=5 if close_values else 2,
        )
        axis.scatter(
            recent,
            y_position,
            s=LATER_CIRCLE_AREA,
            color=CHANGE_COLOR,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        axis.scatter(
            baseline,
            y_position,
            s=EARLIER_CIRCLE_AREA,
            color=LIGHT_TEAL_COLOR,
            edgecolor="white",
            linewidth=1.0,
            zorder=4,
        )
        old_is_right = baseline > recent
        axis.text(
            baseline + (0.012 if old_is_right else -0.012),
            y_position,
            f"{baseline:.0%}",
            va="center",
            ha="left" if old_is_right else "right",
            color="#4A7C78",
            fontsize=9.5,
            zorder=5,
        )
        axis.text(
            recent + (-0.012 if old_is_right else 0.012),
            y_position,
            f"{recent:.0%}",
            va="center",
            ha="right" if old_is_right else "left",
            color="#075F58",
            fontsize=9.5,
            fontweight="semibold",
            zorder=5,
        )
        axis.text(
            (baseline + recent) / 2,
            y_position - (0.28 if close_values else 0.18),
            f"{change:+.1f} pp",
            va="bottom",
            ha="center",
            color="#075F58",
            fontsize=8.5,
            fontweight="semibold",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
                "pad": 0.6,
            },
            zorder=5,
        )

    labels = [
        f"{row.jel_field}  {row.field_name}"
        for row in plot_data.itertuples(index=False)
    ]
    axis.set_yticks(y_positions, labels=labels)
    axis.invert_yaxis()
    axis.set_xlabel("Share of unique authors who are new to the field")
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    maximum_share = float(
        plot_data[
            ["new_author_share_1995_2000", "new_author_share_2020_2025"]
        ]
        .max()
        .max()
    )
    axis.set_xlim(0, min(1.0, maximum_share + 0.06))
    axis.set_title(
        "New-author shares declined in most fields",
        loc="left",
        pad=38,
        fontsize=17,
        fontweight="semibold",
    )
    axis.text(
        0,
        1.015,
        (
            "New author defined by the first observed Top Five publication "
            "within the JEL field"
        ),
        transform=axis.transAxes,
        color="#69727D",
        fontsize=11,
        ha="left",
        va="bottom",
    )
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=LIGHT_TEAL_COLOR,
                markeredgecolor="white",
                markersize=EARLIER_LEGEND_MARKER_SIZE,
                label="1995-2000 (lighter circle)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=CHANGE_COLOR,
                markeredgecolor="white",
                markersize=LATER_LEGEND_MARKER_SIZE,
                label="2020-2025 (3x-area, darker circle)",
            ),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
    )
    axis.grid(axis="x", color="#D8DDE3", linewidth=0.8, alpha=0.65, zorder=1)
    axis.grid(axis="y", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color("#B8C0C8")
    axis.tick_params(axis="y", length=0, colors="#34404B")
    axis.tick_params(axis="x", colors="#59636E")
    figure.text(
        0.01,
        0.012,
        (
            "Note: Authors are counted once within each field and period. "
            "A new author has no earlier observed Top Five publication in "
            "that field. Papers with multiple JEL fields contribute their "
            "authors once to each field. Arrows point from 1995-2000 to "
            "2020-2025; labels above arrows report percentage-point changes."
        ),
        ha="left",
        va="bottom",
        fontsize=9,
        color="#69727D",
        wrap=True,
    )
    figure.subplots_adjust(left=0.29, right=0.97, top=0.84, bottom=0.20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_new_author_coauthor_composition(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    category_specs = [
        (
            "experienced_coauthor_share",
            "Experienced coauthor",
            "Experienced",
            LIGHT_TEAL_COLOR,
            CHANGE_COLOR,
            "#075F58",
        ),
        (
            "newcomer_coauthors_only_share",
            "New coauthors only",
            "Newcomers",
            "#F6C28B",
            NEWCOMER_COLOR,
            "#A8520B",
        ),
        (
            "solo_authored_share",
            "Solo-authored",
            "Solo",
            "#D8DDE3",
            "#7E8790",
            "#59636E",
        ),
    ]
    field_order = sorted(FIELD_CODES)
    category_offsets = [-0.62, 0.0, 0.62]
    field_centers = {
        field: field_number * 2.45
        for field_number, field in enumerate(field_order)
    }

    figure, axis = plt.subplots(figsize=(13, 13.8))
    for field in field_order:
        old_row = data.loc[
            data["jel_field"].eq(field) & data["period"].eq("1995-2000")
        ]
        recent_row = data.loc[
            data["jel_field"].eq(field) & data["period"].eq("2020-2025")
        ]
        if old_row.empty or recent_row.empty:
            continue
        old_row = old_row.iloc[0]
        recent_row = recent_row.iloc[0]

        for category_number, category_spec in enumerate(category_specs):
            column, _, short_label, light_color, dark_color, text_color = (
                category_spec
            )
            y_position = (
                field_centers[field] + category_offsets[category_number]
            )
            old_value = float(old_row[column])
            recent_value = float(recent_row[column])
            draw_share_dumbbell(
                axis=axis,
                old_value=old_value,
                recent_value=recent_value,
                y_position=y_position,
                light_color=light_color,
                dark_color=dark_color,
                text_color=text_color,
            )
            axis.text(
                -0.012,
                y_position,
                short_label,
                ha="right",
                va="center",
                color="#59636E",
                fontsize=9,
            )

    field_labels = [
        f"{field}  {FIELD_NAMES[field]}" for field in field_order
    ]
    axis.set_yticks(
        [field_centers[field] for field in field_order],
        labels=field_labels,
    )
    axis.invert_yaxis()
    axis.set_xlim(-0.17, 1.0)
    axis.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel("Share of new authors")
    axis.set_title(
        "How new-author entry patterns shifted across fields",
        loc="left",
        pad=38,
        fontsize=17,
        fontweight="semibold",
    )
    axis.text(
        0,
        1.012,
        (
            "Coauthor composition in authors' first observed publication "
            "year within each JEL field"
        ),
        transform=axis.transAxes,
        color="#69727D",
        fontsize=11,
        ha="left",
        va="bottom",
    )
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=dark_color,
                markeredgecolor="white",
                markersize=8,
                label=label,
            )
            for _, label, _, _, dark_color, _ in category_specs
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.055),
        ncol=3,
    )
    category_legend = axis.get_legend()
    axis.add_artist(category_legend)
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#D8DDE3",
                markeredgecolor="white",
                markersize=EARLIER_LEGEND_MARKER_SIZE,
                label="1995-2000 (lighter circle)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#59636E",
                markeredgecolor="white",
                markersize=LATER_LEGEND_MARKER_SIZE,
                label="2020-2025 (3x-area, darker circle)",
            ),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.095),
        ncol=2,
    )
    axis.grid(axis="x", color="#D8DDE3", linewidth=0.8, alpha=0.65, zorder=1)
    axis.grid(axis="y", visible=False)
    axis.axvline(0, color="#B8C0C8", linewidth=0.8, zorder=1)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color("#B8C0C8")
    axis.tick_params(axis="y", length=0, colors="#34404B")
    axis.tick_params(axis="x", colors="#59636E")
    figure.text(
        0.01,
        0.012,
        (
            "Note: Each row shows a mutually exclusive share of new authors; "
            "the three categories sum to 100% within a field-period. "
            "Experience is defined within each field. If an "
            "author has multiple entry-year papers, any paper with an "
            "experienced coauthor places the author in that category; "
            "otherwise coauthored entry takes precedence over solo entry. "
            "Arrows point from 1995-2000 to 2020-2025; labels above the "
            "arrows report signed percentage-point changes."
        ),
        ha="left",
        va="bottom",
        fontsize=9,
        color="#69727D",
        wrap=True,
    )
    figure.subplots_adjust(left=0.31, right=0.98, top=0.91, bottom=0.16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def draw_share_dumbbell(
    axis,
    old_value: float,
    recent_value: float,
    y_position: float,
    light_color: str,
    dark_color: str,
    text_color: str,
) -> None:
    if not math.isclose(old_value, recent_value, abs_tol=0.002):
        axis.annotate(
            "",
            xy=(recent_value, y_position),
            xytext=(old_value, y_position),
            arrowprops={
                "arrowstyle": "-|>",
                "color": dark_color,
                "linewidth": 1.3,
                "mutation_scale": 9,
                "shrinkA": 6,
                "shrinkB": 6,
            },
            zorder=2,
        )
    axis.scatter(
        old_value,
        y_position,
        s=EARLIER_CIRCLE_AREA,
        color=light_color,
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    axis.scatter(
        recent_value,
        y_position,
        s=LATER_CIRCLE_AREA,
        color=dark_color,
        edgecolor="white",
        linewidth=0.9,
        zorder=4,
    )
    label_share_dumbbell_values(
        axis=axis,
        old_value=old_value,
        recent_value=recent_value,
        y_position=y_position,
        text_color=text_color,
    )
    axis.text(
        (old_value + recent_value) / 2,
        y_position - 0.18,
        f"{100 * (recent_value - old_value):+.1f} pp",
        ha="center",
        va="bottom",
        color=text_color,
        fontsize=7.8,
        fontweight="semibold",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.86,
            "pad": 0.6,
        },
        zorder=5,
    )


def label_share_dumbbell_values(
    axis,
    old_value: float,
    recent_value: float,
    y_position: float,
    text_color: str,
) -> None:
    if abs(old_value - recent_value) < 0.05:
        axis.text(
            old_value,
            y_position - 0.17,
            f"{old_value:.0%}",
            ha="center",
            va="bottom",
            color="#69727D",
            fontsize=8.5,
        )
        axis.text(
            recent_value,
            y_position + 0.17,
            f"{recent_value:.0%}",
            ha="center",
            va="top",
            color=text_color,
            fontsize=8.5,
            fontweight="semibold",
        )
        return

    old_is_right = old_value > recent_value
    axis.text(
        old_value + (0.012 if old_is_right else -0.012),
        y_position,
        f"{old_value:.0%}",
        ha="left" if old_is_right else "right",
        va="center",
        color="#69727D",
        fontsize=8.5,
    )
    axis.text(
        recent_value + (-0.012 if old_is_right else 0.012),
        y_position,
        f"{recent_value:.0%}",
        ha="right" if old_is_right else "left",
        va="center",
        color=text_color,
        fontsize=8.5,
        fontweight="semibold",
    )


def plot_first_to_second_gap_by_field(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    field_order = sorted(FIELD_CODES)
    old_data = (
        data.loc[data["entry_period"].eq("1995-2000")]
        .set_index("jel_field")
        .reindex(field_order)
    )
    later_data = (
        data.loc[data["entry_period"].eq("2010-2015")]
        .set_index("jel_field")
        .reindex(field_order)
    )
    old_gap = pd.to_numeric(old_data["average_gap_years"], errors="coerce")
    later_gap = pd.to_numeric(later_data["average_gap_years"], errors="coerce")
    old_ci_lower = pd.to_numeric(old_data["gap_ci_95_lower"], errors="coerce")
    old_ci_upper = pd.to_numeric(old_data["gap_ci_95_upper"], errors="coerce")
    later_ci_lower = pd.to_numeric(
        later_data["gap_ci_95_lower"],
        errors="coerce",
    )
    later_ci_upper = pd.to_numeric(
        later_data["gap_ci_95_upper"],
        errors="coerce",
    )
    y_positions = list(range(len(field_order)))

    figure, axis = plt.subplots(figsize=(12, 7.8))
    for y_position, field in zip(y_positions, field_order):
        old_value = float(old_gap.loc[field])
        later_value = float(later_gap.loc[field])
        old_lower = float(old_ci_lower.loc[field])
        old_upper = float(old_ci_upper.loc[field])
        later_lower = float(later_ci_lower.loc[field])
        later_upper = float(later_ci_upper.loc[field])
        if not math.isclose(old_value, later_value, abs_tol=0.01):
            axis.annotate(
                "",
                xy=(later_value, y_position),
                xytext=(old_value, y_position),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#59636E",
                    "linewidth": 1.5,
                    "mutation_scale": 10,
                    "shrinkA": 7,
                    "shrinkB": 7,
                },
                zorder=2,
            )
        axis.errorbar(
            old_value,
            y_position,
            xerr=[[old_value - old_lower], [old_upper - old_value]],
            fmt="none",
            ecolor=BASELINE_COLOR,
            elinewidth=1.2,
            capsize=3,
            capthick=1.2,
            zorder=3,
        )
        axis.errorbar(
            later_value,
            y_position,
            xerr=[[later_value - later_lower], [later_upper - later_value]],
            fmt="none",
            ecolor=CHANGE_COLOR,
            elinewidth=1.2,
            capsize=3,
            capthick=1.2,
            zorder=3,
        )
        axis.scatter(
            old_value,
            y_position,
            s=88,
            color=BASELINE_COLOR,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        axis.scatter(
            later_value,
            y_position,
            s=88,
            color=CHANGE_COLOR,
            edgecolor="white",
            linewidth=1.0,
            zorder=4,
        )
        label_gap_values(
            axis=axis,
            old_value=old_value,
            recent_value=later_value,
            y_position=y_position,
        )

    field_labels = [
        f"{field}  {FIELD_NAMES[field]}" for field in field_order
    ]
    axis.set_yticks(y_positions, labels=field_labels)
    axis.invert_yaxis()
    maximum_gap = float(pd.concat([old_ci_upper, later_ci_upper]).max())
    axis.set_xlim(0, math.ceil(maximum_gap + 1.0))
    axis.set_xlabel("Average years from first to second publication in the field")
    axis.set_title(
        "Time to a second publication varies across fields",
        loc="left",
        pad=38,
        fontsize=17,
        fontweight="semibold",
    )
    axis.text(
        0,
        1.015,
        (
            "Mean gap conditional on a second publication within 11 years, "
            "by field-entry period"
        ),
        transform=axis.transAxes,
        color="#69727D",
        fontsize=11,
        ha="left",
        va="bottom",
    )
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=BASELINE_COLOR,
                markeredgecolor="white",
                markersize=9,
                label="1995-2000 entry cohort",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=CHANGE_COLOR,
                markeredgecolor="white",
                markersize=9,
                label="2010-2015 entry cohort",
            ),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
    )
    axis.grid(axis="x", color="#D8DDE3", linewidth=0.8, alpha=0.65, zorder=1)
    axis.grid(axis="y", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color("#B8C0C8")
    axis.tick_params(axis="y", length=0, colors="#34404B")
    axis.tick_params(axis="x", colors="#59636E")
    figure.text(
        0.01,
        0.012,
        (
            "Note: Entry periods are defined by an author's first observed "
            "Top Five publication in the field. Every author has a complete "
            "11-year follow-up window. Means include only authors whose second "
            "publication in that field occurs within 11 years. Whiskers show "
            "95% Student's t confidence intervals for the mean."
        ),
        ha="left",
        va="bottom",
        fontsize=9,
        color="#69727D",
        wrap=True,
    )
    figure.subplots_adjust(left=0.31, right=0.98, top=0.84, bottom=0.20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def label_gap_values(
    axis,
    old_value: float,
    recent_value: float,
    y_position: int,
) -> None:
    if abs(old_value - recent_value) < 0.45:
        axis.text(
            old_value,
            y_position - 0.20,
            f"{old_value:.1f}",
            ha="center",
            va="bottom",
            color="#59636E",
            fontsize=9.5,
            fontweight="semibold",
        )
        axis.text(
            recent_value,
            y_position + 0.20,
            f"{recent_value:.1f}",
            ha="center",
            va="top",
            color="#075F58",
            fontsize=9.5,
            fontweight="semibold",
        )
        return

    old_alignment = "left" if old_value > recent_value else "right"
    recent_alignment = "right" if old_value > recent_value else "left"
    old_offset = 0.12 if old_value > recent_value else -0.12
    recent_offset = -0.12 if old_value > recent_value else 0.12
    axis.text(
        old_value + old_offset,
        y_position,
        f"{old_value:.1f}",
        ha=old_alignment,
        va="center",
        color="#59636E",
        fontsize=9.5,
        fontweight="semibold",
    )
    axis.text(
        recent_value + recent_offset,
        y_position,
        f"{recent_value:.1f}",
        ha=recent_alignment,
        va="center",
        color="#075F58",
        fontsize=9.5,
        fontweight="semibold",
    )


def write_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)


if __name__ == "__main__":
    main()
