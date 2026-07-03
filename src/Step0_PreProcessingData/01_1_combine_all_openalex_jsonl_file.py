from pathlib import Path
import json
import os
import re
from typing import Any


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

INPUT_BY_YEAR_DIR = Path("data/raw/openalex_by_year")
OUTPUT_JSONL = Path("data/raw/OpenAlex_FullData_Jsonl.jsonl")
YEAR_FILE_PATTERN = "OpenAlex_Year*_Jsonl.jsonl"


def main() -> None:
    yearly_files = sorted(
        INPUT_BY_YEAR_DIR.glob(YEAR_FILE_PATTERN),
        key=year_from_path,
    )
    if not yearly_files:
        raise FileNotFoundError(
            f"No yearly OpenAlex JSONL files found in {INPUT_BY_YEAR_DIR} "
            f"matching {YEAR_FILE_PATTERN}"
        )

    rows_written = combine_jsonl_files(yearly_files, OUTPUT_JSONL)

    print("OpenAlex yearly JSONL combine summary:")
    print(f"  Input folder: {INPUT_BY_YEAR_DIR}")
    print(f"  Year files found: {len(yearly_files)}")
    print(f"  Output JSONL: {OUTPUT_JSONL}")
    print(f"  Unique records written: {rows_written}")


def combine_jsonl_files(yearly_files: list[Path], output_jsonl: Path) -> int:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    rows_written = 0

    with output_jsonl.open("w", encoding="utf-8") as output_file:
        for yearly_file in yearly_files:
            file_rows = 0
            file_rows_written = 0

            for record in read_jsonl(yearly_file):
                file_rows += 1
                work_id = str(record.get("id", "")).strip()
                if work_id and work_id in seen_ids:
                    continue

                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                rows_written += 1
                file_rows_written += 1
                if work_id:
                    seen_ids.add(work_id)

            print(
                f"  {yearly_file.name}: read {file_rows}, "
                f"wrote {file_rows_written} new records."
            )

    return rows_written


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {error}"
                ) from error
    return records


def year_from_path(path: Path) -> int:
    match = re.search(r"OpenAlex_Year(\d{4})_Jsonl\.jsonl$", path.name)
    if match:
        return int(match.group(1))
    return 9999


if __name__ == "__main__":
    main()
