from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")
STEP_DIR = PROJECT_ROOT / "src" / "Step1_CleanOpenalexCrossrefData"

SCRIPTS_TO_RUN = [
    STEP_DIR / "01_CleanVariable_OpenAlex.py",
    STEP_DIR / "02_CleanVariable_Crossref.py",
    STEP_DIR / "03_Merge_OpenAlex_Crossref.py",
    STEP_DIR / "04_WebScrape_JEL.py",
    STEP_DIR / "05_CleanAll_OpenAlex_Crossref.py",
]


def main() -> None:
    check_scripts_exist(SCRIPTS_TO_RUN)

    print("Running Step 1: Clean OpenAlex/Crossref data")
    for script in SCRIPTS_TO_RUN:
        run_script(script)

    print()
    print("Step 1 finished successfully.")


def check_scripts_exist(scripts: list[Path]) -> None:
    missing_scripts = [script for script in scripts if not script.exists()]
    if missing_scripts:
        missing_text = "\n".join(str(script) for script in missing_scripts)
        raise FileNotFoundError(f"Missing Step 1 script(s):\n{missing_text}")


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
