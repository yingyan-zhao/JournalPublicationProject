# OpenAlex Raw Field Inventory

File: `data/raw/OpenAlex_FullData_Jsonl.jsonl`
Records: 424
File size: 5.40 MB

## Top-Level Fields

- `_query_issn`: present 424/424; types str: 424
- `_query_journal`: present 424/424; types str: 424
- `abstract_inverted_index`: present 424/424; types dict: 334, NoneType: 90
- `apc_list`: present 424/424; types NoneType: 263, dict: 161
- `apc_paid`: present 424/424; types NoneType: 424
- `authorships`: present 424/424; types list: 424; list length min/median/max: 0/2/4
- `awards`: present 424/424; types list: 424; list length min/median/max: 0/0/1
- `best_oa_location`: present 424/424; types NoneType: 381, dict: 43
- `biblio`: present 424/424; types dict: 424
- `citation_normalized_percentile`: present 424/424; types dict: 415, NoneType: 9
- `cited_by_count`: present 424/424; types int: 424
- `cited_by_percentile_year`: present 424/424; types dict: 363, NoneType: 61
- `collection_date`: present 424/424; types str: 424
- `concepts`: present 424/424; types list: 424; list length min/median/max: 1/11/31
- `content_urls`: present 424/424; types NoneType: 399, dict: 25
- `corresponding_author_ids`: present 424/424; types list: 424; list length min/median/max: 0/1/1
- `corresponding_institution_ids`: present 424/424; types list: 424; list length min/median/max: 0/1/4
- `countries_distinct_count`: present 424/424; types int: 424
- `counts_by_year`: present 424/424; types list: 424; list length min/median/max: 0/14/15
- `created_date`: present 424/424; types str: 424
- `display_name`: present 424/424; types str: 424
- `doi`: present 424/424; types str: 416, NoneType: 8
- `funders`: present 424/424; types list: 424; list length min/median/max: 0/0/3
- `fwci`: present 424/424; types float: 415, NoneType: 9
- `has_content`: present 424/424; types dict: 424
- `has_fulltext`: present 424/424; types bool: 424
- `id`: present 424/424; types str: 424
- `ids`: present 424/424; types dict: 424
- `indexed_in`: present 424/424; types list: 424; list length min/median/max: 0/1/2
- `institutions`: present 424/424; types list: 424; list length min/median/max: 0/0/0
- `institutions_distinct_count`: present 424/424; types int: 424
- `is_paratext`: present 424/424; types bool: 424
- `is_retracted`: present 424/424; types bool: 424
- `is_xpac`: present 424/424; types bool: 424
- `keywords`: present 424/424; types list: 424; list length min/median/max: 1/9/22
- `language`: present 424/424; types str: 424
- `locations`: present 424/424; types list: 424; list length min/median/max: 1/2/19
- `locations_count`: present 424/424; types int: 424
- `mesh`: present 424/424; types list: 424; list length min/median/max: 0/0/51
- `open_access`: present 424/424; types dict: 424
- `primary_location`: present 424/424; types dict: 424
- `primary_topic`: present 424/424; types dict: 415, NoneType: 9
- `publication_date`: present 424/424; types str: 424
- `publication_year`: present 424/424; types int: 424
- `referenced_works`: present 424/424; types list: 424; list length min/median/max: 0/14/177
- `referenced_works_count`: present 424/424; types int: 424
- `related_works`: present 424/424; types list: 424; list length min/median/max: 0/10/20
- `sustainable_development_goals`: present 424/424; types list: 424; list length min/median/max: 0/1/2
- `title`: present 424/424; types str: 424
- `topics`: present 424/424; types list: 424; list length min/median/max: 0/3/3
- `type`: present 424/424; types str: 424
- `updated_date`: present 424/424; types str: 424

## Nested Fields

- `apc_list`: `currency`, `value`, `value_usd`
- `authorships`: `affiliations`, `author`, `author_position`, `countries`, `institutions`, `is_corresponding`, `raw_affiliation_strings`, `raw_author_name`, `raw_orcid`
- `awards`: `display_name`, `funder_award_id`, `funder_display_name`, `funder_id`, `id`
- `best_oa_location`: `id`, `is_accepted`, `is_oa`, `is_published`, `landing_page_url`, `license`, `license_id`, `pdf_url`, `raw_source_name`, `raw_type`, `source`, `version`
- `biblio`: `first_page`, `issue`, `last_page`, `volume`
- `citation_normalized_percentile`: `is_in_top_10_percent`, `is_in_top_1_percent`, `value`
- `cited_by_percentile_year`: `max`, `min`
- `concepts`: `display_name`, `id`, `level`, `score`, `wikidata`
- `content_urls`: `grobid_xml`, `pdf`
- `counts_by_year`: `cited_by_count`, `year`
- `funders`: `display_name`, `id`, `ror`
- `has_content`: `grobid_xml`, `pdf`
- `ids`: `doi`, `mag`, `openalex`, `pmid`
- `keywords`: `display_name`, `id`, `score`
- `locations`: `id`, `is_accepted`, `is_oa`, `is_published`, `landing_page_url`, `license`, `license_id`, `pdf_url`, `raw_source_name`, `raw_type`, `source`, `version`
- `mesh`: `descriptor_name`, `descriptor_ui`, `is_major_topic`, `qualifier_name`, `qualifier_ui`
- `open_access`: `any_repository_has_fulltext`, `is_oa`, `oa_status`, `oa_url`
- `primary_location`: `id`, `is_accepted`, `is_oa`, `is_published`, `landing_page_url`, `license`, `license_id`, `pdf_url`, `raw_source_name`, `raw_type`, `source`, `version`
- `primary_topic`: `display_name`, `domain`, `field`, `id`, `score`, `subfield`
- `sustainable_development_goals`: `display_name`, `id`, `score`
- `topics`: `display_name`, `domain`, `field`, `id`, `score`, `subfield`

## First Record Preview

- `id`: https://openalex.org/W2135410596
- `doi`: https://doi.org/10.1257/aer.90.1.166
- `title`: ERC: A Theory of Equity, Reciprocity, and Competition
- `display_name`: ERC: A Theory of Equity, Reciprocity, and Competition
- `publication_year`: 2000
- `publication_date`: 2000-03-01
- `type`: article
- `cited_by_count`: 5511
- `collection_date`: 2026-05-15
- `_query_journal`: American Economic Review
- `_query_issn`: 0002-8282
