# OpenAlex Pilot Report: 2020

Date run: 2026-05-12

## Command

```bash
python -m src.econ_pub_concentration.collect_openalex --from-year 2020 --to-year 2020 --processed-output data/processed/openalex_pilot_2020.csv --raw-output data/raw/openalex_pilot_2020.jsonl
```

## Files Created

- `data/raw/openalex_pilot_2020.jsonl`
- `data/processed/openalex_pilot_2020.csv`
- `outputs/pilot_2020/tables/author_productivity.csv`
- `outputs/pilot_2020/tables/concentration_summary.csv`
- `outputs/pilot_2020/tables/annual_concentration_summary.csv`

## OpenAlex Pull Results

- Raw OpenAlex works: 441
- Works with author rows in processed CSV: 387
- Author-paper rows: 1,128
- Unique authors in processed CSV: 871

Works without authors were mostly non-research records such as front matter, back matter, referee lists, annual reports, and journal administrative notes.

## Works With Authors by Journal

| Journal | Works |
|---|---:|
| American Economic Review | 123 |
| Review of Economic Studies | 84 |
| Econometrica | 78 |
| Journal of Political Economy | 61 |
| Quarterly Journal of Economics | 41 |

## Preliminary Concentration Output

These are smoke-test numbers, not final research estimates.

| Measure | Value |
|---|---:|
| Authors | 871 |
| Total fractional publication credit | 383.75 |
| Top 1% share | 0.0347 |
| Top 5% share | 0.1315 |
| Top 10% share | 0.2461 |
| HHI | 0.0016 |
| Gini | 0.2804 |

## Data Quality Notes

The pilot shows that OpenAlex is a good backbone source, but the raw pull is not yet a final research-article sample.

Issues found:

- Some raw records have no authors.
- Some records with authors are comments, replies, referee acknowledgments, or other non-standard items.
- American Economic Review likely includes Papers and Proceedings-style material under the same journal source.
- OpenAlex did not provide JEL codes, acknowledgments, first submission dates, accepted dates, or first working-paper dates.

## Next Cleaning Step

Create an article-level cleaning rule that flags or excludes:

- records without authors
- front matter and back matter
- referee acknowledgments and annual reports
- comments and replies, unless these are intentionally included
- correction, erratum, corrigendum, and editorial records

The final inclusion rule should be documented before running the full historical sample.

