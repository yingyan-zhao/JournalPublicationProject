# Crossref Raw Field Inventory

File: `data/raw/Crossref_FullData_Jsonl.jsonl`
Records: 432
File size: 1.65 MB

## Top-Level Fields

- `DOI`: present 432/432; types str: 432
- `ISSN`: present 432/432; types list: 432; list length min/median/max: 1/2/2
- `URL`: present 432/432; types str: 432
- `_query_issn`: present 432/432; types str: 432
- `_query_journal`: present 432/432; types str: 432
- `abstract`: present 48/432; types str: 48
- `alternative-id`: present 392/432; types list: 392; list length min/median/max: 1/1/1
- `author`: present 378/432; types list: 378; list length min/median/max: 1/2/4
- `collection_date`: present 432/432; types str: 432
- `container-title`: present 432/432; types list: 432; list length min/median/max: 1/1/1
- `content-domain`: present 432/432; types dict: 432
- `created`: present 432/432; types dict: 432
- `deposited`: present 432/432; types dict: 432
- `indexed`: present 432/432; types dict: 432
- `is-referenced-by-count`: present 432/432; types int: 432
- `issn-type`: present 432/432; types list: 432; list length min/median/max: 1/2/2
- `issue`: present 432/432; types str: 432
- `issued`: present 432/432; types dict: 432
- `journal-issue`: present 432/432; types dict: 432
- `language`: present 432/432; types str: 432
- `license`: present 85/432; types list: 85; list length min/median/max: 1/1/1
- `link`: present 432/432; types list: 432; list length min/median/max: 1/1/2
- `member`: present 432/432; types str: 432
- `page`: present 432/432; types str: 432
- `prefix`: present 432/432; types str: 432
- `published`: present 432/432; types dict: 432
- `published-print`: present 432/432; types dict: 432
- `publisher`: present 432/432; types str: 432
- `reference`: present 319/432; types list: 319; list length min/median/max: 1/15/176
- `reference-count`: present 432/432; types int: 432
- `references-count`: present 432/432; types int: 432
- `relation`: present 6/432; types dict: 6
- `resource`: present 432/432; types dict: 432
- `score`: present 432/432; types float: 432
- `short-container-title`: present 432/432; types list: 432; list length min/median/max: 1/1/1
- `source`: present 432/432; types str: 432
- `subtitle`: present 3/432; types list: 3; list length min/median/max: 1/1/1
- `title`: present 432/432; types list: 432; list length min/median/max: 1/1/1
- `type`: present 432/432; types str: 432
- `volume`: present 432/432; types str: 432

## Nested Fields

- `author`: `affiliation`, `family`, `given`, `sequence`, `suffix`
- `content-domain`: `crossmark-restriction`, `domain`
- `created`: `date-parts`, `date-time`, `timestamp`
- `deposited`: `date-parts`, `date-time`, `timestamp`
- `indexed`: `date-parts`, `date-time`, `timestamp`, `version`
- `issn-type`: `type`, `value`
- `issued`: `date-parts`
- `journal-issue`: `issue`, `published-print`
- `license`: `URL`, `content-version`, `delay-in-days`, `start`
- `link`: `URL`, `content-type`, `content-version`, `intended-application`
- `published`: `date-parts`
- `published-print`: `date-parts`
- `reference`: `DOI`, `author`, `doi-asserted-by`, `edition`, `first-page`, `issue`, `journal-title`, `key`, `unstructured`, `volume`, `volume-title`, `year`
- `relation`: `has-preprint`
- `resource`: `primary`

## First Record Preview

- `DOI`: 10.1257/aer.90.4.927
- `title`: ['Meetings with Costly Participation']
- `container-title`: ['American Economic Review']
- `publisher`: American Economic Association
- `type`: journal-article
- `published`: {'date-parts': [[2000, 9, 1]]}
- `published-print`: {'date-parts': [[2000, 9, 1]]}
- `published-online`: None
- `reference-count`: 9
- `is-referenced-by-count`: 66
- `collection_date`: 2026-05-18
- `_query_journal`: American Economic Review
- `_query_issn`: 0002-8282
