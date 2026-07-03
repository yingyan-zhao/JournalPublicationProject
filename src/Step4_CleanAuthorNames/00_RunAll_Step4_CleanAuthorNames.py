from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")
STEP_DIR = PROJECT_ROOT / "src" / "Step4_CleanAuthorNames"

SCRIPTS_TO_RUN = [
    STEP_DIR / "01_CleanAuthorNames_aea.py",
    STEP_DIR / "02_CleanAuthorNames_scrape.py",
    STEP_DIR / "03_CleanAuthorNames_openalex.py",
    STEP_DIR / "04_CleanAuthorNames_openalex_raw.py",
    STEP_DIR / "05_CleanAuthorNames_crossref.py",
    STEP_DIR / "06_CleanAuthroNames_crossref_FamilyGivenNames.py",
    STEP_DIR / "07_ConsolidateVersionsAuthorNames_aea_scrape.py",
    STEP_DIR / "08_ConsolidateVersionsAuthorNames_openalex_openalexraw.py",
    STEP_DIR / "09_ConsolidateVersionsAuthorNames_crossref_crossrefFamilyGivenNames.py",
    STEP_DIR / "10_ConsolidateVersionsAuthorNames_aea_scrape_openalex.py",
    STEP_DIR / "11_ConsolidateVersionsAuthorNames_aea_scrape_openalex_crossref.py",
    STEP_DIR / "12_CreateAuthorID.py",
]


def main() -> None:
    check_scripts_exist(SCRIPTS_TO_RUN)

    print("Running Step 4: Clean and consolidate author names")
    for script in SCRIPTS_TO_RUN:
        run_script(script)

    print()
    print("Step 4 author-name cleaning pipeline finished successfully.")


def check_scripts_exist(scripts: list[Path]) -> None:
    missing_scripts = [script for script in scripts if not script.exists()]
    if missing_scripts:
        missing_text = "\n".join(str(script) for script in missing_scripts)
        raise FileNotFoundError(f"Missing Step 4 script(s):\n{missing_text}")


def run_script(script: Path) -> None:
    print()
    print(f"Running {script.name}...")
    subprocess.run(
        [sys.executable, str(script)],
        cwd=PROJECT_ROOT,
        check=True,
    )
    print(f"Finished {script.name}.")


if __name__ == "__main__":
    main()
