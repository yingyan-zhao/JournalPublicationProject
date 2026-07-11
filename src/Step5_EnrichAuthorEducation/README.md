# ORCID-Only PhD Enrichment

`01_EnrichAuthorPhD.py` uses only the ORCID Public API to enrich:

`data/processed/author_names/JEL_Training_Data_AuthorList.csv`

For each author, the program:

1. searches ORCID by first and last name;
2. retrieves each candidate's public person and education records;
3. keeps education entries explicitly identified as a PhD, DPhil, doctorate,
   Doctor of Philosophy, or an equivalent doctoral title;
4. takes `organization.name` as the PhD institution;
5. takes `end-date.year` as the PhD graduation year;
6. scores the match and sends uncertain cases to manual review.

It does not query RePEc, Google Scholar, faculty webpages, CVs, search engines,
or university alumni records.

## ORCID Credentials

Register free read-public API credentials following the official ORCID Public
API documentation: <https://info.orcid.org/documentation/features/public-api/>.

The program automatically loads a local project-root `.env` file. Use this
format (an `.env.example` template is included):

```bash
ORCID_CLIENT_ID="APP-..."
ORCID_CLIENT_SECRET="..."
CONTACT_EMAIL="your_email@example.com"
```

The real `.env` file is excluded by `.gitignore`. Shell environment variables
are also supported and take priority over values in `.env`. Do not place the
client secret directly inside the Python file or commit it to Git.

## Pilot

Start with a small run:

```bash
python src/Step5_EnrichAuthorEducation/01_EnrichAuthorPhD.py --limit 25
```

The final output still contains every input author. Authors outside the pilot
are labeled `not_processed`.

After inspecting the pilot, resume with:

```bash
python src/Step5_EnrichAuthorEducation/01_EnrichAuthorPhD.py
```

The JSON Lines checkpoint is written after every author, so an interrupted run
continues without repeating completed ORCID requests.

ORCID currently documents a daily quota for the registered Public API. Because
one author can require a search call plus person and education calls for
multiple candidates, the complete 19,000-author list may require several daily
runs. The checkpoint makes those batches cumulative.

Useful options:

```text
--limit 100              Process at most 100 new authors.
--max-candidates 10      Maximum ORCID profiles fetched for one name query.
--request-delay 1.0      Delay between ORCID requests.
--retry-unresolved       Retry earlier no-result and ambiguous authors.
--overwrite              Start a new ORCID-only checkpoint.
```

Authors affected by a temporary ORCID error are retried automatically on the
next run. Cached successful API responses are reused.

## Matching And Confidence

Each doctoral education candidate receives a confidence score based on:

- similarity between the project name and public ORCID name: 55%;
- uniqueness of the ORCID name search: 20%;
- presence of a PhD institution: 15%;
- presence of an ORCID education end year: 10%.

The combined value is multiplied by an ORCID source factor of `0.95`. The
default auto-accept threshold is `0.85`, and the best candidate must lead a
candidate from another ORCID profile by at least `0.12`.

The program does not automatically choose among:

- similarly scored ORCID profiles for the same name;
- conflicting doctoral records on one ORCID profile;
- candidates below the confidence threshold;
- search results truncated before a usable profile is found.

These cases are exported for manual review.

## Outputs

Outputs are written under `data/processed/author_education/`:

- `JEL_Training_Data_AuthorList_WithPhD.csv`: all input authors plus accepted
  PhD institution, graduation year, confidence, match status, and ORCID ID.
- `JEL_Training_Data_AuthorPhD_ManualReview.csv`: ambiguous, low-confidence,
  incomplete, and no-result cases with ranked ORCID candidates.
- `JEL_Training_Data_AuthorPhD_ORCID_Evidence.csv`: one row per public doctoral
  education record retrieved from ORCID.
- `JEL_Training_Data_AuthorPhD_ORCID_Checkpoint.jsonl`: resumable processing
  state, including ORCID source errors.

Principal new final columns are:

```text
phd_institution
phd_year
phd_confidence
phd_match_status
phd_candidate_institution
phd_candidate_year
phd_degree_title
phd_department
orcid_id_matched
orcid_candidate_id
orcid_profile_url
orcid_search_candidate_count
orcid_fetched_candidate_count
orcid_phd_record_count
phd_manual_review_reason
orcid_source_errors
orcid_retrieved_at
```

## Coverage Limitation

ORCID records are created and maintained by researchers. Education entries can
be absent, private, incomplete, or use a role title that does not identify the
degree. Therefore, `not_found` means only that the ORCID Public API did not
provide a usable public doctoral record; it does not mean the researcher has
no PhD.
