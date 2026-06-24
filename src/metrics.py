from __future__ import annotations

import pandas as pd


def fractional_credit(df: pd.DataFrame) -> pd.Series:
    """Return fractional publication credit for each author-publication row."""
    if "n_authors" not in df:
        raise ValueError("Expected column 'n_authors'.")
    if (df["n_authors"] <= 0).any():
        raise ValueError("'n_authors' must be positive.")
    return 1 / df["n_authors"]


def author_productivity(
    df: pd.DataFrame,
    credit: str = "fractional",
    author_col: str = "author_id",
) -> pd.DataFrame:
    """Aggregate publication credit to the author level."""
    required = {author_col, "publication_id", "n_authors"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = df.copy()
    if credit == "fractional":
        work["credit"] = fractional_credit(work)
    elif credit == "full":
        work["credit"] = 1.0
    else:
        raise ValueError("credit must be either 'fractional' or 'full'.")

    out = (
        work.groupby(author_col, as_index=False)
        .agg(
            publication_credit=("credit", "sum"),
            publication_rows=("publication_id", "count"),
            distinct_publications=("publication_id", "nunique"),
        )
        .sort_values("publication_credit", ascending=False)
    )
    return out


def top_share(values: pd.Series, share: float) -> float:
    """Share of total output held by the top share of authors."""
    if not 0 < share <= 1:
        raise ValueError("share must be in (0, 1].")
    values = values.dropna().sort_values(ascending=False)
    if values.empty or values.sum() == 0:
        return 0.0
    n_top = max(1, int(len(values) * share))
    return float(values.head(n_top).sum() / values.sum())


def hhi(values: pd.Series) -> float:
    """Herfindahl-Hirschman Index for author publication shares."""
    values = values.dropna()
    total = values.sum()
    if values.empty or total == 0:
        return 0.0
    shares = values / total
    return float((shares**2).sum())


def gini(values: pd.Series) -> float:
    """Gini coefficient for nonnegative author productivity values."""
    values = values.dropna().sort_values().reset_index(drop=True)
    if values.empty:
        return 0.0
    if (values < 0).any():
        raise ValueError("Gini is only defined here for nonnegative values.")
    total = values.sum()
    if total == 0:
        return 0.0
    n = len(values)
    ranks = pd.Series(range(1, n + 1), index=values.index)
    return float((2 * (ranks * values).sum()) / (n * total) - (n + 1) / n)


def concentration_summary(productivity: pd.DataFrame) -> pd.DataFrame:
    """Return headline concentration statistics from author productivity."""
    values = productivity["publication_credit"]
    return pd.DataFrame(
        [
            {
                "n_authors": int(values.notna().sum()),
                "total_credit": float(values.sum()),
                "top_1pct_share": top_share(values, 0.01),
                "top_5pct_share": top_share(values, 0.05),
                "top_10pct_share": top_share(values, 0.10),
                "hhi": hhi(values),
                "gini": gini(values),
            }
        ]
    )

