"""Enrich researchers with PhD institution and graduation year from ORCID.

The program uses only the ORCID Public API. It searches by researcher name,
reads structured education records, keeps doctoral degrees, scores identity
matches, and leaves uncertain cases for manual review.

Required environment variables:
    ORCID_CLIENT_ID
    ORCID_CLIENT_SECRET

ORCID records are self-maintained and may be incomplete. A blank result means
that no usable public doctoral education record was found; it does not prove
that the researcher has no PhD.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import time
import unicodedata
from typing import Any, Iterable

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = (
    PROJECT_ROOT
    / "data/processed/author_names/JEL_Training_Data_AuthorList.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/author_education"

FINAL_OUTPUT_NAME = "JEL_Training_Data_AuthorList_WithPhD.csv"
MANUAL_REVIEW_OUTPUT_NAME = "JEL_Training_Data_AuthorPhD_ManualReview.csv"
EVIDENCE_OUTPUT_NAME = "JEL_Training_Data_AuthorPhD_ORCID_Evidence.csv"
CHECKPOINT_NAME = "JEL_Training_Data_AuthorPhD_ORCID_Checkpoint.jsonl"

REQUIRED_AUTHOR_COLUMNS = ["author_id", "final_last_name", "final_first_name"]
LOCAL_ENV_KEYS = {
    "ORCID_CLIENT_ID",
    "ORCID_CLIENT_SECRET",
    "CONTACT_EMAIL",
}
SCHEMA_VERSION = 2

ORCID_TOKEN_URL = "https://orcid.org/oauth/token"
ORCID_API_ROOT = "https://pub.orcid.org/v3.0"
ORCID_PROFILE_ROOT = "https://orcid.org"
ORCID_ID_PATTERN = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\b(?:19\d{2}|20\d{2})\b")
DOCTORAL_DEGREE_PATTERN = re.compile(
    r"(?:"
    r"\bph\.?\s*d\.?\b|"
    r"\bd\.?\s*phil\.?\b|"
    r"\bdoctor(?:ate|al)\b|"
    r"\bdoctor\s+of\s+philosophy\b|"
    r"\bdoctor\s+of\s+economics\b|"
    r"\bdoctorat\b|"
    r"\bdr\.?\s*rer\.?\s*pol\.?\b"
    r")",
    flags=re.IGNORECASE,
)

EVIDENCE_COLUMNS = [
    "author_id",
    "query_name",
    "orcid_id",
    "orcid_profile_url",
    "orcid_candidate_name",
    "name_similarity",
    "orcid_search_candidate_count",
    "degree_title",
    "department",
    "phd_institution",
    "phd_year",
    "institution_city",
    "institution_country",
    "education_put_code",
    "evidence_confidence",
    "evidence_status",
    "retrieved_at",
]

MANUAL_REVIEW_COLUMNS = [
    "author_id",
    "final_last_name",
    "final_first_name",
    "phd_match_status",
    "manual_review_reason",
    "candidate_rank",
    "orcid_id",
    "orcid_candidate_name",
    "name_similarity",
    "candidate_institution",
    "candidate_year",
    "candidate_degree_title",
    "candidate_department",
    "candidate_confidence",
    "orcid_profile_url",
    "orcid_search_candidate_count",
    "source_errors",
]


@dataclass
class Evidence:
    author_id: str
    query_name: str
    orcid_id: str
    orcid_profile_url: str
    orcid_candidate_name: str
    name_similarity: float
    orcid_search_candidate_count: int
    degree_title: str
    department: str
    phd_institution: str
    phd_year: str
    institution_city: str
    institution_country: str
    education_put_code: str
    evidence_confidence: float
    evidence_status: str
    retrieved_at: str


@dataclass
class CollectionResult:
    evidence: list[Evidence]
    search_candidate_count: int
    fetched_candidate_count: int
    search_truncated: bool


@dataclass
class Resolution:
    accepted: bool
    status: str
    reason: str
    confidence: float
    institution: str
    year: str
    selected: Evidence | None
    candidates: list[Evidence]


class OrcidApiError(RuntimeError):
    """A recoverable ORCID API or network error."""


class JsonCache:
    """Small file cache that avoids repeating successful ORCID API calls."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def get(self, namespace: str, key: str) -> Any | None:
        path = self._path(namespace, key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, namespace: str, key: str, value: Any) -> None:
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(value, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_path.replace(path)

    def _path(self, namespace: str, key: str) -> Path:
        digest = sha256(key.encode("utf-8")).hexdigest()
        return self.root / namespace / f"{digest}.json"


class OrcidHttpClient:
    def __init__(
        self,
        *,
        cache: JsonCache,
        user_agent: str,
        timeout: float,
        delay: float,
        max_retries: int,
        max_retry_after: float,
    ) -> None:
        self.cache = cache
        self.timeout = timeout
        self.delay = delay
        self.max_retries = max_retries
        self.max_retry_after = max_retry_after
        self.last_request_at = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def get_json(
        self,
        url: str,
        *,
        headers: dict[str, str],
        cache_namespace: str,
        cache_key: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        cached = self.cache.get(cache_namespace, cache_key)
        if cached is not None:
            return cached
        response = self.request("GET", url, headers=headers, params=params)
        payload = response_json(response)
        self.cache.put(cache_namespace, cache_key, payload)
        return payload

    def post_json(
        self,
        url: str,
        *,
        data: dict[str, Any],
        headers: dict[str, str],
    ) -> Any:
        response = self.request("POST", url, data=data, headers=headers)
        return response_json(response)

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        for attempt in range(self.max_retries + 1):
            self._pace()
            try:
                response = self.session.request(
                    method,
                    url,
                    timeout=self.timeout,
                    **kwargs,
                )
            except requests.RequestException as error:
                if attempt == self.max_retries:
                    raise OrcidApiError(f"ORCID request failed: {error}") from error
                time.sleep(min(2**attempt, 8))
                continue

            if response.status_code == 429:
                wait_seconds = parse_retry_after(response.headers.get("Retry-After"))
                if wait_seconds is None:
                    wait_seconds = float(2**attempt)
                if wait_seconds > self.max_retry_after:
                    raise OrcidApiError(
                        "ORCID rate limited the request and requested a "
                        f"{wait_seconds:.0f}-second delay."
                    )
                if attempt == self.max_retries:
                    raise OrcidApiError("ORCID rate limit persisted after all retries.")
                time.sleep(max(1.0, wait_seconds))
                continue

            if response.status_code >= 500 and attempt < self.max_retries:
                time.sleep(min(2**attempt, 8))
                continue

            try:
                response.raise_for_status()
            except requests.HTTPError as error:
                raise OrcidApiError(
                    f"ORCID returned HTTP {response.status_code}."
                ) from error
            return response

        raise OrcidApiError("ORCID request failed without returning a response.")

    def _pace(self) -> None:
        elapsed = time.monotonic() - self.last_request_at
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request_at = time.monotonic()


class OrcidSource:
    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        client: OrcidHttpClient,
        max_candidates: int,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.client = client
        self.max_candidates = max_candidates
        self._access_token = ""

    def collect(self, author: dict[str, str]) -> CollectionResult:
        direct_orcid_id = normalize_orcid_id(author.get("orcid_id", ""))
        if direct_orcid_id:
            orcid_ids = [direct_orcid_id]
            search_candidate_count = 1
            search_truncated = False
        else:
            query = build_orcid_query(author)
            payload = self.client.get_json(
                f"{ORCID_API_ROOT}/search/",
                headers=self.api_headers(),
                params={"q": query, "rows": self.max_candidates},
                cache_namespace="orcid_search",
                cache_key=f"{query}|rows={self.max_candidates}",
            )
            orcid_ids = extract_orcid_ids(payload)
            search_candidate_count = extract_orcid_search_count(payload, len(orcid_ids))
            search_truncated = search_candidate_count > len(orcid_ids)

        evidence: list[Evidence] = []
        for orcid_id in orcid_ids:
            person_payload = self.client.get_json(
                f"{ORCID_API_ROOT}/{orcid_id}/person",
                headers=self.api_headers(),
                cache_namespace="orcid_person",
                cache_key=orcid_id,
            )
            candidate_name = extract_orcid_person_name(person_payload)
            if direct_orcid_id:
                name_similarity = 1.0
            elif candidate_name:
                name_similarity = compare_person_names(author, candidate_name)
            else:
                name_similarity = 0.70
            if name_similarity < 0.70:
                continue

            education_payload = self.client.get_json(
                f"{ORCID_API_ROOT}/{orcid_id}/educations",
                headers=self.api_headers(),
                cache_namespace="orcid_educations",
                cache_key=orcid_id,
            )
            for education in parse_orcid_educations(education_payload):
                evidence.append(
                    create_evidence(
                        author=author,
                        orcid_id=orcid_id,
                        candidate_name=candidate_name,
                        name_similarity=name_similarity,
                        search_candidate_count=search_candidate_count,
                        education=education,
                        direct_orcid_id=bool(direct_orcid_id),
                    )
                )

        return CollectionResult(
            evidence=deduplicate_evidence(evidence),
            search_candidate_count=search_candidate_count,
            fetched_candidate_count=len(orcid_ids),
            search_truncated=search_truncated,
        )

    def api_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/vnd.orcid+json",
            "Authorization": f"Bearer {self.access_token()}",
        }

    def access_token(self) -> str:
        if self._access_token:
            return self._access_token
        payload = self.client.post_json(
            ORCID_TOKEN_URL,
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "client_credentials",
                "scope": "/read-public",
            },
            headers={"Accept": "application/json"},
        )
        token = clean_text(payload.get("access_token")) if isinstance(payload, dict) else ""
        if not token:
            raise OrcidApiError("ORCID did not return a Public API access token.")
        self._access_token = token
        return token


def main() -> None:
    load_local_env(PROJECT_ROOT / ".env")
    args = parse_args()
    client_id = os.getenv("ORCID_CLIENT_ID", "").strip()
    client_secret = os.getenv("ORCID_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise RuntimeError(
            "Set ORCID_CLIENT_ID and ORCID_CLIENT_SECRET before running this script."
        )

    authors = read_author_list(args.input_csv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = JsonCache(output_dir / "cache/orcid")
    client = OrcidHttpClient(
        cache=cache,
        user_agent=build_user_agent(args.mailto),
        timeout=args.timeout,
        delay=args.request_delay,
        max_retries=args.max_retries,
        max_retry_after=args.max_retry_after,
    )
    source = OrcidSource(
        client_id=client_id,
        client_secret=client_secret,
        client=client,
        max_candidates=args.max_candidates,
    )

    checkpoint_path = output_dir / CHECKPOINT_NAME
    if args.overwrite and checkpoint_path.exists():
        checkpoint_path.unlink()
    checkpoint_records = load_checkpoint(checkpoint_path)

    authors_to_process = select_authors_to_process(
        authors,
        checkpoint_records,
        retry_unresolved=args.retry_unresolved,
        start_after_author_id=args.start_after_author_id,
        limit=args.limit,
    )
    print_start_summary(authors, authors_to_process, checkpoint_path)

    total = len(authors_to_process)
    for position, author in enumerate(authors_to_process.to_dict("records"), start=1):
        collection, source_errors = collect_one_author(source, author)
        resolution = resolve_evidence(
            collection.evidence,
            search_candidate_count=collection.search_candidate_count,
            search_truncated=collection.search_truncated,
            min_auto_confidence=args.min_auto_confidence,
            ambiguity_margin=args.ambiguity_margin,
        )
        record = checkpoint_record(
            author,
            collection,
            source_errors,
            resolution,
        )
        append_jsonl(checkpoint_path, record)
        checkpoint_records[author["author_id"]] = record
        print_author_progress(position, total, author, resolution, source_errors)

    write_outputs(authors, checkpoint_records, output_dir)
    print_output_summary(authors, checkpoint_records, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Get researchers' PhD institutions and years from ORCID."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-after-author-id", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--retry-unresolved",
        action="store_true",
        help="Reprocess prior not-found, low-confidence, and ambiguous authors.",
    )
    parser.add_argument("--min-auto-confidence", type=float, default=0.85)
    parser.add_argument("--ambiguity-margin", type=float, default=0.12)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--request-delay", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-retry-after", type=float, default=120.0)
    parser.add_argument("--mailto", default=os.getenv("CONTACT_EMAIL", ""))
    return parser.parse_args()


def collect_one_author(
    source: OrcidSource,
    author: dict[str, str],
) -> tuple[CollectionResult, list[str]]:
    first_name = clean_text(author.get("final_first_name"))
    last_name = clean_text(author.get("final_last_name"))
    direct_orcid_id = normalize_orcid_id(author.get("orcid_id", ""))
    if not direct_orcid_id and (not first_name or not last_name):
        return empty_collection(), ["Identity lookup requires both first and last name."]
    try:
        return source.collect(author), []
    except (OrcidApiError, requests.RequestException, ValueError) as error:
        return empty_collection(), [str(error)]


def empty_collection() -> CollectionResult:
    return CollectionResult([], 0, 0, False)


def parse_orcid_educations(payload: Any) -> list[dict[str, str]]:
    """Return only public ORCID education entries identified as doctorates."""
    records: list[dict[str, str]] = []
    for item in walk_dicts(payload):
        summary = item.get("education-summary")
        if not isinstance(summary, dict):
            continue
        degree_title = clean_text(summary.get("role-title"))
        if not is_doctoral_degree(degree_title):
            continue

        organization = summary.get("organization") or {}
        if not isinstance(organization, dict):
            organization = {}
        address = organization.get("address") or {}
        if not isinstance(address, dict):
            address = {}
        records.append(
            {
                "degree_title": degree_title,
                "department": clean_text(summary.get("department-name")),
                "institution": clean_text(organization.get("name")),
                "year": extract_orcid_date_year(summary.get("end-date")),
                "institution_city": clean_text(address.get("city")),
                "institution_country": clean_text(address.get("country")),
                "education_put_code": clean_text(summary.get("put-code")),
            }
        )
    return deduplicate_education_records(records)


def is_doctoral_degree(value: Any) -> bool:
    return bool(DOCTORAL_DEGREE_PATTERN.search(clean_text(value)))


def create_evidence(
    *,
    author: dict[str, str],
    orcid_id: str,
    candidate_name: str,
    name_similarity: float,
    search_candidate_count: int,
    education: dict[str, str],
    direct_orcid_id: bool,
) -> Evidence:
    institution = clean_text(education.get("institution"))
    year = normalize_year(education.get("year"))
    evidence_status = "complete" if institution and year else "incomplete"
    confidence = score_orcid_evidence(
        name_similarity=name_similarity,
        search_candidate_count=search_candidate_count,
        has_institution=bool(institution),
        has_year=bool(year),
        direct_orcid_id=direct_orcid_id,
    )
    return Evidence(
        author_id=clean_text(author.get("author_id")),
        query_name=author_full_name(author),
        orcid_id=orcid_id,
        orcid_profile_url=f"{ORCID_PROFILE_ROOT}/{orcid_id}",
        orcid_candidate_name=candidate_name,
        name_similarity=round(clamp(name_similarity), 4),
        orcid_search_candidate_count=max(1, int(search_candidate_count)),
        degree_title=clean_text(education.get("degree_title")),
        department=clean_text(education.get("department")),
        phd_institution=institution,
        phd_year=year,
        institution_city=clean_text(education.get("institution_city")),
        institution_country=clean_text(education.get("institution_country")),
        education_put_code=clean_text(education.get("education_put_code")),
        evidence_confidence=confidence,
        evidence_status=evidence_status,
        retrieved_at=utc_now(),
    )


def score_orcid_evidence(
    *,
    name_similarity: float,
    search_candidate_count: int,
    has_institution: bool,
    has_year: bool,
    direct_orcid_id: bool,
) -> float:
    """Score one ORCID candidate on a transparent zero-to-one scale."""
    if direct_orcid_id:
        uniqueness = 1.0
    else:
        uniqueness = max(0.35, 1.0 - 0.15 * (max(search_candidate_count, 1) - 1))
    raw_score = 0.55 * clamp(name_similarity)
    raw_score += 0.20 * uniqueness
    raw_score += 0.15 if has_institution else 0.0
    raw_score += 0.10 if has_year else 0.0
    return round(min(0.99, 0.95 * raw_score), 4)


def resolve_evidence(
    evidence: list[Evidence],
    *,
    search_candidate_count: int,
    search_truncated: bool,
    min_auto_confidence: float,
    ambiguity_margin: float,
) -> Resolution:
    usable = [item for item in evidence if clean_text(item.phd_institution)]
    usable = deduplicate_evidence(usable)
    usable.sort(key=lambda item: item.evidence_confidence, reverse=True)

    if not usable:
        if evidence:
            reason = "doctoral_orcid_record_missing_institution"
            status = "incomplete_orcid_record"
        elif search_truncated:
            reason = "orcid_search_truncated_without_public_phd_match"
            status = "ambiguous"
        elif search_candidate_count > 0:
            reason = "orcid_profiles_found_without_public_phd_record"
            status = "not_found"
        else:
            reason = "no_matching_public_orcid_profile"
            status = "not_found"
        incomplete_candidates = sorted(
            evidence,
            key=lambda item: item.evidence_confidence,
            reverse=True,
        )
        confidence = (
            incomplete_candidates[0].evidence_confidence
            if incomplete_candidates
            else 0.0
        )
        return Resolution(
            False,
            status,
            reason,
            confidence,
            "",
            "",
            None,
            incomplete_candidates,
        )

    top = usable[0]
    top_orcid_records = [item for item in usable if item.orcid_id == top.orcid_id]
    if doctoral_records_conflict(top_orcid_records):
        return Resolution(
            False,
            "ambiguous",
            "one_orcid_profile_has_conflicting_doctoral_records",
            top.evidence_confidence,
            "",
            "",
            None,
            usable,
        )

    if search_truncated:
        return Resolution(
            False,
            "ambiguous",
            "orcid_search_truncated_before_all_candidates_were_evaluated",
            top.evidence_confidence,
            "",
            "",
            None,
            usable,
        )

    different_orcid_candidates = [
        item for item in usable[1:] if item.orcid_id != top.orcid_id
    ]
    if different_orcid_candidates:
        runner_up = different_orcid_candidates[0]
        if top.evidence_confidence - runner_up.evidence_confidence < ambiguity_margin:
            return Resolution(
                False,
                "ambiguous",
                "multiple_competing_orcid_profiles",
                top.evidence_confidence,
                "",
                "",
                None,
                usable,
            )

    if top.evidence_confidence < min_auto_confidence:
        return Resolution(
            False,
            "low_confidence",
            "orcid_confidence_below_auto_accept_threshold",
            top.evidence_confidence,
            "",
            "",
            None,
            usable,
        )

    status = "matched_high_confidence" if top.phd_year else "matched_institution_only"
    reason = "clear_orcid_doctoral_record" if top.phd_year else "orcid_year_not_public"
    return Resolution(
        True,
        status,
        reason,
        top.evidence_confidence,
        top.phd_institution,
        top.phd_year,
        top,
        usable,
    )


def doctoral_records_conflict(records: list[Evidence]) -> bool:
    institutions = {
        normalize_key(item.phd_institution)
        for item in records
        if clean_text(item.phd_institution)
    }
    years = {
        normalize_year(item.phd_year)
        for item in records
        if normalize_year(item.phd_year)
    }
    return len(institutions) > 1 or len(years) > 1


def read_author_list(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing author-list CSV: {path}")
    data = pd.read_csv(path, dtype=str, low_memory=False).fillna("")
    missing = [column for column in REQUIRED_AUTHOR_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Author-list CSV is missing columns: {missing}")
    data["author_id"] = data["author_id"].apply(clean_text)
    if (data["author_id"] == "").any():
        raise ValueError("author_id contains blank values.")
    if data["author_id"].duplicated().any():
        examples = data.loc[
            data["author_id"].duplicated(keep=False), "author_id"
        ].head().tolist()
        raise ValueError(f"author_id is not unique; examples: {examples}")
    if "orcid_id" in data.columns:
        data["orcid_id"] = data["orcid_id"].apply(normalize_orcid_id)
    return data.sort_values("author_id", kind="mergesort").reset_index(drop=True)


def select_authors_to_process(
    authors: pd.DataFrame,
    checkpoint_records: dict[str, dict[str, Any]],
    *,
    retry_unresolved: bool,
    start_after_author_id: str,
    limit: int | None,
) -> pd.DataFrame:
    selected = authors.copy()
    if start_after_author_id:
        selected = selected.loc[selected["author_id"] > start_after_author_id]
    complete_ids = {
        author_id
        for author_id, record in checkpoint_records.items()
        if checkpoint_is_complete(record, retry_unresolved=retry_unresolved)
    }
    selected = selected.loc[~selected["author_id"].isin(complete_ids)]
    if limit is not None:
        selected = selected.head(limit)
    return selected


def write_outputs(
    authors: pd.DataFrame,
    checkpoint_records: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    final_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    manual_rows: list[dict[str, Any]] = []

    for author in authors.to_dict("records"):
        record = checkpoint_records.get(clean_text(author.get("author_id")))
        if record is None:
            final_rows.append(not_processed_row(author))
            continue

        resolution = resolution_from_dict(record.get("resolution") or {})
        evidence = [Evidence(**item) for item in record.get("evidence", [])]
        evidence_rows.extend(asdict(item) for item in evidence)
        top = resolution.selected or (resolution.candidates[0] if resolution.candidates else None)
        selected = resolution.selected if resolution.accepted else None

        final_rows.append(
            {
                **author,
                "phd_institution": resolution.institution,
                "phd_year": resolution.year,
                "phd_confidence": round(resolution.confidence, 4),
                "phd_match_status": resolution.status,
                "phd_candidate_institution": top.phd_institution if top else "",
                "phd_candidate_year": top.phd_year if top else "",
                "phd_degree_title": selected.degree_title if selected else "",
                "phd_department": selected.department if selected else "",
                "orcid_id_matched": selected.orcid_id if selected else "",
                "orcid_candidate_id": top.orcid_id if top else "",
                "orcid_profile_url": selected.orcid_profile_url if selected else "",
                "orcid_search_candidate_count": record.get(
                    "orcid_search_candidate_count", 0
                ),
                "orcid_fetched_candidate_count": record.get(
                    "orcid_fetched_candidate_count", 0
                ),
                "orcid_phd_record_count": len(evidence),
                "phd_manual_review_reason": "" if resolution.accepted else resolution.reason,
                "orcid_source_errors": "; ".join(record.get("source_errors", [])),
                "orcid_retrieved_at": record.get("processed_at", ""),
            }
        )

        if not resolution.accepted:
            if resolution.candidates:
                for rank, candidate in enumerate(resolution.candidates, start=1):
                    manual_rows.append(
                        manual_review_row(
                            author,
                            resolution,
                            candidate,
                            rank,
                            record,
                        )
                    )
            else:
                manual_rows.append(
                    manual_review_row(author, resolution, None, 0, record)
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(final_rows).to_csv(output_dir / FINAL_OUTPUT_NAME, index=False)
    pd.DataFrame(evidence_rows, columns=EVIDENCE_COLUMNS).to_csv(
        output_dir / EVIDENCE_OUTPUT_NAME,
        index=False,
    )
    pd.DataFrame(manual_rows, columns=MANUAL_REVIEW_COLUMNS).to_csv(
        output_dir / MANUAL_REVIEW_OUTPUT_NAME,
        index=False,
    )


def not_processed_row(author: dict[str, str]) -> dict[str, Any]:
    return {
        **author,
        "phd_institution": "",
        "phd_year": "",
        "phd_confidence": "",
        "phd_match_status": "not_processed",
        "phd_candidate_institution": "",
        "phd_candidate_year": "",
        "phd_degree_title": "",
        "phd_department": "",
        "orcid_id_matched": "",
        "orcid_candidate_id": "",
        "orcid_profile_url": "",
        "orcid_search_candidate_count": "",
        "orcid_fetched_candidate_count": "",
        "orcid_phd_record_count": "",
        "phd_manual_review_reason": "",
        "orcid_source_errors": "",
        "orcid_retrieved_at": "",
    }


def manual_review_row(
    author: dict[str, str],
    resolution: Resolution,
    candidate: Evidence | None,
    rank: int,
    record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "author_id": clean_text(author.get("author_id")),
        "final_last_name": clean_text(author.get("final_last_name")),
        "final_first_name": clean_text(author.get("final_first_name")),
        "phd_match_status": resolution.status,
        "manual_review_reason": resolution.reason,
        "candidate_rank": rank if candidate else "",
        "orcid_id": candidate.orcid_id if candidate else "",
        "orcid_candidate_name": candidate.orcid_candidate_name if candidate else "",
        "name_similarity": candidate.name_similarity if candidate else "",
        "candidate_institution": candidate.phd_institution if candidate else "",
        "candidate_year": candidate.phd_year if candidate else "",
        "candidate_degree_title": candidate.degree_title if candidate else "",
        "candidate_department": candidate.department if candidate else "",
        "candidate_confidence": candidate.evidence_confidence if candidate else "",
        "orcid_profile_url": candidate.orcid_profile_url if candidate else "",
        "orcid_search_candidate_count": record.get("orcid_search_candidate_count", 0),
        "source_errors": "; ".join(record.get("source_errors", [])),
    }


def checkpoint_record(
    author: dict[str, str],
    collection: CollectionResult,
    source_errors: list[str],
    resolution: Resolution,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "author": author,
        "evidence": [asdict(item) for item in collection.evidence],
        "orcid_search_candidate_count": collection.search_candidate_count,
        "orcid_fetched_candidate_count": collection.fetched_candidate_count,
        "orcid_search_truncated": collection.search_truncated,
        "source_errors": source_errors,
        "resolution": resolution_to_dict(resolution),
        "processed_at": utc_now(),
    }


def resolution_to_dict(resolution: Resolution) -> dict[str, Any]:
    return {
        "accepted": resolution.accepted,
        "status": resolution.status,
        "reason": resolution.reason,
        "confidence": resolution.confidence,
        "institution": resolution.institution,
        "year": resolution.year,
        "selected": asdict(resolution.selected) if resolution.selected else None,
        "candidates": [asdict(item) for item in resolution.candidates],
    }


def resolution_from_dict(value: dict[str, Any]) -> Resolution:
    selected_value = value.get("selected")
    selected = Evidence(**selected_value) if isinstance(selected_value, dict) else None
    candidates = [
        Evidence(**item)
        for item in value.get("candidates", [])
        if isinstance(item, dict)
    ]
    return Resolution(
        accepted=bool(value.get("accepted", False)),
        status=clean_text(value.get("status")) or "not_found",
        reason=clean_text(value.get("reason")),
        confidence=parse_float(value.get("confidence"), 0.0),
        institution=clean_text(value.get("institution")),
        year=normalize_year(value.get("year")),
        selected=selected,
        candidates=candidates,
    )


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid checkpoint JSON on line {line_number}: {path}"
                ) from error
            if record.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(
                    "The existing checkpoint was created by a different pipeline. "
                    "Run this ORCID-only version with --overwrite."
                )
            author_id = clean_text((record.get("author") or {}).get("author_id"))
            if author_id:
                records[author_id] = record
    return records


def checkpoint_is_complete(
    record: dict[str, Any],
    *,
    retry_unresolved: bool,
) -> bool:
    resolution = record.get("resolution") or {}
    if bool(resolution.get("accepted", False)):
        return True
    if record.get("source_errors"):
        return False
    return not retry_unresolved


def build_orcid_query(author: dict[str, str]) -> str:
    first_name = escape_orcid_query_value(author.get("final_first_name"))
    last_name = escape_orcid_query_value(author.get("final_last_name"))
    return f'given-names:"{first_name}" AND family-name:"{last_name}"'


def escape_orcid_query_value(value: Any) -> str:
    return clean_text(value).replace("\\", "\\\\").replace('"', '\\"')


def extract_orcid_ids(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    output: list[str] = []
    for result in payload.get("result", []) or []:
        if not isinstance(result, dict):
            continue
        identifier = result.get("orcid-identifier") or {}
        if not isinstance(identifier, dict):
            continue
        orcid_id = normalize_orcid_id(identifier.get("path"))
        if orcid_id and orcid_id not in output:
            output.append(orcid_id)
    return output


def extract_orcid_search_count(payload: Any, fallback: int) -> int:
    if not isinstance(payload, dict):
        return fallback
    try:
        return max(int(payload.get("num-found", fallback)), fallback)
    except (TypeError, ValueError):
        return fallback


def extract_orcid_person_name(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    name = payload.get("name") or {}
    if not isinstance(name, dict):
        return ""
    given_name = nested_value(name.get("given-names"), "value")
    family_name = nested_value(name.get("family-name"), "value")
    return clean_text(f"{given_name} {family_name}")


def extract_orcid_date_year(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    return normalize_year(nested_value(value.get("year"), "value"))


def normalize_orcid_id(value: Any) -> str:
    match = ORCID_ID_PATTERN.search(clean_text(value))
    return match.group().upper() if match else ""


def compare_person_names(author: dict[str, str], candidate_name: str) -> float:
    query_first = name_tokens(author.get("final_first_name"))
    query_last = name_tokens(author.get("final_last_name"))
    if not query_last or not candidate_name:
        return 0.0

    if "," in candidate_name:
        last_part, first_part = candidate_name.split(",", maxsplit=1)
        candidate_last = name_tokens(last_part)
        candidate_first = name_tokens(first_part)
    else:
        candidate_tokens = name_tokens(candidate_name)
        candidate_last = candidate_tokens[-max(1, len(query_last)) :]
        candidate_first = candidate_tokens[: -len(candidate_last)]

    query_last_key = "".join(query_last)
    candidate_last_key = "".join(candidate_last)
    if query_last_key == candidate_last_key:
        last_score = 1.0
    elif query_last_key in candidate_last_key or candidate_last_key in query_last_key:
        last_score = 0.90
    else:
        last_score = SequenceMatcher(None, query_last_key, candidate_last_key).ratio()
    if last_score < 0.72:
        return round(0.68 * last_score, 4)

    query_first_key = query_first[0] if query_first else ""
    first_score = max(
        (given_name_similarity(query_first_key, value) for value in candidate_first),
        default=0.0,
    )
    return round(clamp(0.68 * last_score + 0.32 * first_score), 4)


def given_name_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if left[0] == right[0] and (len(left) == 1 or len(right) == 1):
        return 0.82
    if left.startswith(right) or right.startswith(left):
        return 0.90
    return SequenceMatcher(None, left, right).ratio()


def deduplicate_education_records(
    records: Iterable[dict[str, str]],
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for record in records:
        key = (
            normalize_key(record.get("institution")),
            normalize_year(record.get("year")),
            normalize_key(record.get("degree_title")),
            clean_text(record.get("education_put_code")),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append({name: clean_text(value) for name, value in record.items()})
    return output


def deduplicate_evidence(evidence: Iterable[Evidence]) -> list[Evidence]:
    output: list[Evidence] = []
    seen: set[tuple[str, ...]] = set()
    for item in evidence:
        key = (
            item.author_id,
            item.orcid_id,
            normalize_key(item.phd_institution),
            normalize_year(item.phd_year),
            normalize_key(item.degree_title),
            item.education_put_code,
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_dicts(child)


def nested_value(value: Any, key: str) -> str:
    return clean_text(value.get(key)) if isinstance(value, dict) else ""


def name_tokens(value: Any) -> list[str]:
    text = unicodedata.normalize("NFKD", clean_text(value))
    ascii_text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.findall(r"[a-z]+", ascii_text)


def normalize_key(value: Any) -> str:
    return "".join(name_tokens(value))


def normalize_year(value: Any) -> str:
    match = YEAR_PATTERN.search(clean_text(value))
    if not match:
        return ""
    year = int(match.group())
    if 1900 <= year <= datetime.now().year + 1:
        return str(year)
    return ""


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return " ".join(str(value).replace("\u00a0", " ").split())


def load_local_env(path: Path) -> None:
    """Load approved local settings without replacing shell environment values."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            if value.startswith("export "):
                value = value.removeprefix("export ").strip()
            if "=" not in value:
                raise ValueError(f"Invalid .env line {line_number}: expected KEY=VALUE")
            key, setting = value.split("=", maxsplit=1)
            key = key.strip()
            if key not in LOCAL_ENV_KEYS:
                continue
            setting = strip_matching_quotes(setting.strip())
            os.environ.setdefault(key, setting)


def strip_matching_quotes(value: str) -> str:
    quote_pairs = {'"': '"', "'": "'", "“": "”", "‘": "’"}
    if len(value) >= 2 and value[0] in quote_pairs:
        if value[-1] == quote_pairs[value[0]]:
            return value[1:-1]
    return value


def author_full_name(author: dict[str, str]) -> str:
    return clean_text(
        f"{clean_text(author.get('final_first_name'))} "
        f"{clean_text(author.get('final_last_name'))}"
    )


def response_json(response: requests.Response) -> Any:
    try:
        return response.json()
    except requests.JSONDecodeError as error:
        raise OrcidApiError("ORCID returned invalid JSON.") from error


def parse_retry_after(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def build_user_agent(mailto: str) -> str:
    contact = clean_text(mailto) or "contact-not-provided"
    return f"JournalPublicationProject-ORCID/1.0 ({contact})"


def print_start_summary(
    authors: pd.DataFrame,
    authors_to_process: pd.DataFrame,
    checkpoint_path: Path,
) -> None:
    print("ORCID PhD enrichment:")
    print(f"  Input authors: {len(authors):,}")
    print(f"  Authors to process now: {len(authors_to_process):,}")
    print("  Source: ORCID Public API only")
    print(f"  Resume checkpoint: {checkpoint_path}")
    print("  A checkpoint record is appended after every author.")


def print_author_progress(
    position: int,
    total: int,
    author: dict[str, str],
    resolution: Resolution,
    source_errors: list[str],
) -> None:
    print(
        f"[{position:,}/{total:,}] {author.get('author_id', '')} "
        f"{author_full_name(author)}: {resolution.status} "
        f"(confidence={resolution.confidence:.4f})"
    )
    for error in source_errors:
        print(f"    ORCID warning: {error}")


def print_output_summary(
    authors: pd.DataFrame,
    checkpoint_records: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    statuses: dict[str, int] = {}
    for record in checkpoint_records.values():
        status = clean_text((record.get("resolution") or {}).get("status"))
        statuses[status] = statuses.get(status, 0) + 1
    print("ORCID PhD enrichment output summary:")
    print(f"  Total rows in final author CSV: {len(authors):,}")
    print(f"  Authors processed: {len(checkpoint_records):,}")
    for status, count in sorted(statuses.items()):
        print(f"  {status}: {count:,}")
    print(f"  Final CSV: {output_dir / FINAL_OUTPUT_NAME}")
    print(f"  Manual-review CSV: {output_dir / MANUAL_REVIEW_OUTPUT_NAME}")
    print(f"  ORCID evidence CSV: {output_dir / EVIDENCE_OUTPUT_NAME}")


if __name__ == "__main__":
    main()
