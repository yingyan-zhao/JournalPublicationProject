import importlib.util
import os
from pathlib import Path
import sys

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/Step5_EnrichAuthorEducation/01_EnrichAuthorPhD.py"
)
SPEC = importlib.util.spec_from_file_location("enrich_author_phd", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def author(first="Claudia", last="Goldin", author_id="A1"):
    return {
        "author_id": author_id,
        "final_first_name": first,
        "final_last_name": last,
    }


def education(institution="University of Chicago", year="1972"):
    return {
        "degree_title": "Ph.D. in Economics",
        "department": "Department of Economics",
        "institution": institution,
        "year": year,
        "institution_city": "Chicago",
        "institution_country": "US",
        "education_put_code": "101",
    }


def evidence(
    *,
    orcid_id="0000-0001-2345-6789",
    candidate_name="Claudia Goldin",
    institution="University of Chicago",
    year="1972",
    search_count=1,
):
    return MODULE.create_evidence(
        author=author(),
        orcid_id=orcid_id,
        candidate_name=candidate_name,
        name_similarity=MODULE.compare_person_names(author(), candidate_name),
        search_candidate_count=search_count,
        education=education(institution, year),
        direct_orcid_id=False,
    )


def resolve(items, search_count=1, truncated=False):
    return MODULE.resolve_evidence(
        items,
        search_candidate_count=search_count,
        search_truncated=truncated,
        min_auto_confidence=0.85,
        ambiguity_margin=0.12,
    )


def test_doctoral_degree_titles_are_identified():
    accepted = [
        "Ph.D. in Economics",
        "DPhil Economics",
        "Doctor of Philosophy",
        "Doctorate in Economics",
        "Dr. rer. pol.",
    ]
    rejected = ["B.A. Economics", "Master of Science", "Research Fellow"]

    assert all(MODULE.is_doctoral_degree(value) for value in accepted)
    assert not any(MODULE.is_doctoral_degree(value) for value in rejected)


def test_parse_orcid_educations_keeps_only_doctoral_records():
    payload = {
        "affiliation-group": [
            {
                "summaries": [
                    {
                        "education-summary": {
                            "put-code": 101,
                            "role-title": "Ph.D. in Economics",
                            "department-name": "Department of Economics",
                            "end-date": {"year": {"value": "2006"}},
                            "organization": {
                                "name": "Harvard University",
                                "address": {"city": "Cambridge", "country": "US"},
                            },
                        }
                    }
                ]
            },
            {
                "summaries": [
                    {
                        "education-summary": {
                            "put-code": 102,
                            "role-title": "B.A. Economics",
                            "end-date": {"year": {"value": "2000"}},
                            "organization": {"name": "Yale University"},
                        }
                    }
                ]
            },
        ]
    }

    records = MODULE.parse_orcid_educations(payload)

    assert records == [
        {
            "degree_title": "Ph.D. in Economics",
            "department": "Department of Economics",
            "institution": "Harvard University",
            "year": "2006",
            "institution_city": "Cambridge",
            "institution_country": "US",
            "education_put_code": "101",
        }
    ]


def test_name_similarity_accepts_initial_but_requires_last_name():
    assert MODULE.compare_person_names(author(), "Claudia Dale Goldin") > 0.90
    assert MODULE.compare_person_names(author(), "C. Goldin") > 0.90
    assert MODULE.compare_person_names(author(), "Goldin, Claudia") == 1.0
    assert MODULE.compare_person_names(author(), "Claudia Smith") < 0.72


def test_one_complete_orcid_candidate_is_accepted():
    result = resolve([evidence()])

    assert result.accepted
    assert result.status == "matched_high_confidence"
    assert result.institution == "University of Chicago"
    assert result.year == "1972"
    assert result.confidence >= 0.90


def test_institution_without_public_end_year_is_retained_as_partial_match():
    result = resolve([evidence(year="")])

    assert result.accepted
    assert result.status == "matched_institution_only"
    assert result.institution == "University of Chicago"
    assert result.year == ""


def test_similarly_scored_orcid_profiles_are_ambiguous():
    first = evidence(
        orcid_id="0000-0001-2345-6789",
        institution="University of Chicago",
        search_count=2,
    )
    second = evidence(
        orcid_id="0000-0002-3456-7890",
        institution="Harvard University",
        search_count=2,
    )

    result = resolve([first, second], search_count=2)

    assert not result.accepted
    assert result.status == "ambiguous"
    assert result.reason == "multiple_competing_orcid_profiles"


def test_conflicting_doctoral_records_on_one_orcid_are_ambiguous():
    first = evidence(institution="University of Chicago", year="1972")
    second = evidence(institution="Harvard University", year="1975")
    second.education_put_code = "202"

    result = resolve([first, second])

    assert not result.accepted
    assert result.reason == "one_orcid_profile_has_conflicting_doctoral_records"


def test_truncated_orcid_search_is_not_auto_accepted():
    result = resolve([evidence(search_count=20)], search_count=20, truncated=True)

    assert not result.accepted
    assert result.status == "ambiguous"
    assert result.reason == (
        "orcid_search_truncated_before_all_candidates_were_evaluated"
    )


def test_incomplete_doctoral_record_is_kept_for_manual_review():
    incomplete = evidence(institution="", year="1972")

    result = resolve([incomplete])

    assert not result.accepted
    assert result.status == "incomplete_orcid_record"
    assert result.candidates == [incomplete]


class FakeOrcidClient:
    def post_json(self, url, *, data, headers):
        return {"access_token": "test-token"}

    def get_json(
        self,
        url,
        *,
        headers,
        cache_namespace,
        cache_key,
        params=None,
    ):
        if url.endswith("/search/"):
            return {
                "num-found": 1,
                "result": [
                    {"orcid-identifier": {"path": "0000-0001-2345-6789"}}
                ],
            }
        if url.endswith("/person"):
            return {
                "name": {
                    "given-names": {"value": "Claudia"},
                    "family-name": {"value": "Goldin"},
                }
            }
        if url.endswith("/educations"):
            return {
                "affiliation-group": [
                    {
                        "summaries": [
                            {
                                "education-summary": {
                                    "put-code": 101,
                                    "role-title": "Ph.D. in Economics",
                                    "end-date": {"year": {"value": "1972"}},
                                    "organization": {
                                        "name": "University of Chicago",
                                        "address": {
                                            "city": "Chicago",
                                            "country": "US",
                                        },
                                    },
                                }
                            }
                        ]
                    }
                ]
            }
        raise AssertionError(f"Unexpected fake URL: {url}")


def test_orcid_source_uses_search_person_and_education_endpoints():
    source = MODULE.OrcidSource(
        client_id="APP-TEST",
        client_secret="secret",
        client=FakeOrcidClient(),
        max_candidates=10,
    )

    collection = source.collect(author())

    assert collection.search_candidate_count == 1
    assert collection.fetched_candidate_count == 1
    assert len(collection.evidence) == 1
    assert collection.evidence[0].phd_institution == "University of Chicago"
    assert collection.evidence[0].phd_year == "1972"


def test_outputs_contain_final_match_and_manual_review(tmp_path):
    authors = pd.DataFrame(
        [
            author(author_id="A1"),
            author(first="No", last="PublicPhD", author_id="A2"),
        ]
    )
    matched_evidence = evidence()
    matched_collection = MODULE.CollectionResult([matched_evidence], 1, 1, False)
    matched_resolution = resolve([matched_evidence])
    unresolved_collection = MODULE.CollectionResult([], 1, 1, False)
    unresolved_resolution = resolve([], search_count=1)
    records = {
        "A1": MODULE.checkpoint_record(
            authors.iloc[0].to_dict(),
            matched_collection,
            [],
            matched_resolution,
        ),
        "A2": MODULE.checkpoint_record(
            authors.iloc[1].to_dict(),
            unresolved_collection,
            [],
            unresolved_resolution,
        ),
    }

    MODULE.write_outputs(authors, records, tmp_path)

    final = pd.read_csv(tmp_path / MODULE.FINAL_OUTPUT_NAME, dtype=str).fillna("")
    review = pd.read_csv(
        tmp_path / MODULE.MANUAL_REVIEW_OUTPUT_NAME,
        dtype=str,
    ).fillna("")
    evidence_output = pd.read_csv(
        tmp_path / MODULE.EVIDENCE_OUTPUT_NAME,
        dtype=str,
    ).fillna("")
    assert final.loc[final["author_id"] == "A1", "phd_institution"].item() == (
        "University of Chicago"
    )
    assert final.loc[final["author_id"] == "A1", "orcid_id_matched"].item() == (
        "0000-0001-2345-6789"
    )
    assert review["author_id"].tolist() == ["A2"]
    assert evidence_output["author_id"].tolist() == ["A1"]


def test_checkpoint_retries_source_errors_and_optionally_unresolved():
    errored = {"source_errors": ["temporary"], "resolution": {"accepted": False}}
    unresolved = {"source_errors": [], "resolution": {"accepted": False}}
    accepted = {"source_errors": [], "resolution": {"accepted": True}}

    assert not MODULE.checkpoint_is_complete(errored, retry_unresolved=False)
    assert MODULE.checkpoint_is_complete(unresolved, retry_unresolved=False)
    assert not MODULE.checkpoint_is_complete(unresolved, retry_unresolved=True)
    assert MODULE.checkpoint_is_complete(accepted, retry_unresolved=True)


def test_local_env_loader_reads_quotes_and_preserves_existing_values(
    tmp_path,
    monkeypatch,
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        'ORCID_CLIENT_ID="APP-LOCAL"\n'
        'ORCID_CLIENT_SECRET="local-secret"\n'
        'CONTACT_EMAIL=“local@example.com”\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("ORCID_CLIENT_ID", raising=False)
    monkeypatch.delenv("ORCID_CLIENT_SECRET", raising=False)
    monkeypatch.setenv("CONTACT_EMAIL", "shell@example.com")

    MODULE.load_local_env(env_file)

    assert os.environ["ORCID_CLIENT_ID"] == "APP-LOCAL"
    assert os.environ["ORCID_CLIENT_SECRET"] == "local-secret"
    assert os.environ["CONTACT_EMAIL"] == "shell@example.com"
