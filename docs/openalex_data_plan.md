# OpenAlex Data Plan

OpenAlex will be the backbone source for the first version of the dataset.

## Role of OpenAlex

Use OpenAlex to collect:

- article title
- DOI
- journal
- publication year and date
- author names and OpenAlex author IDs
- author order
- institution names and OpenAlex institution IDs
- citation count at collection date
- abstract when available
- OpenAlex work URL
- OpenAlex Topics hierarchy: domain, field, subfield, and topic
- first-pass record classification for sample cleaning

## Top-Five Journal ISSNs

The collector uses print and online ISSNs where available:

- American Economic Review: `0002-8282`, `1944-7981`
- Journal of Political Economy: `0022-3808`, `1537-534X`
- Econometrica: `0012-9682`, `1468-0262`
- Review of Economic Studies: `0034-6527`, `1467-937X`
- Quarterly Journal of Economics: `0033-5533`, `1531-4650`

## What OpenAlex Will Not Fully Provide

OpenAlex is a strong starting point, but other sources will still be needed for:

- JEL codes
- article keywords, when missing
- acknowledgments or thanks sections
- first submission date
- accepted date
- first online working-paper date

These fields can be added later by matching the OpenAlex article records to publisher webpages, PDFs, Crossref, NBER, CEPR, SSRN, RePEc/IDEAS, IZA, and author websites.

## First-Pass Record Classification

The OpenAlex collector adds three cleaning fields:

- `record_classification`
- `include_in_main_sample`
- `classification_reason`

The first-pass classifier keeps likely research articles as `candidate_research_article` and flags records such as:

- `no_author_record`
- `front_back_matter`
- `referee_acknowledgment`
- `journal_report`
- `index`
- `forthcoming_list`
- `correction`
- `comment_reply`

This classification is meant to support auditing, not replace manual review. The final sample rule should be documented before the full historical analysis.

## OpenAlex Topics Hierarchy

OpenAlex maps works into a topic hierarchy:

```text
domain > field > subfield > topic
```

The collector stores the primary hierarchy in:

- `openalex_primary_domain`
- `openalex_primary_field`
- `openalex_primary_subfield`
- `openalex_primary_topic`

It also stores a compact semicolon-separated summary of all returned topics and scores in:

- `openalex_topics`

These OpenAlex topic fields should be treated as supplemental metadata. The project's main economics field classification will still follow JEL codes when available.

## Second-Stage Crossref Enrichment

After the OpenAlex pull, Crossref will be the next metadata source to check. The preferred matching key is DOI. Crossref can help verify or enrich:

- title
- DOI
- publisher metadata
- publication dates
- abstracts, when deposited
- references and reference counts, when available
- funding information, when deposited
- ORCID and affiliation-related metadata, when available

Crossref should be treated as an enrichment and verification source rather than the backbone source, because author and institution disambiguation are usually more useful in OpenAlex for this project.

## First Data Pull

Start with a limited pilot before collecting the full historical sample. For example:

```bash
python -m src.econ_pub_concentration.collect_openalex --from-year 2015 --to-year 2020 --mailto your_email@example.com
```

The script writes:

- `data/raw/openalex_works.jsonl`
- `data/processed/openalex_publications.csv`
