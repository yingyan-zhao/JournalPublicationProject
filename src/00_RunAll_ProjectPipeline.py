from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

SCRIPTS_TO_RUN = [
    PROJECT_ROOT
    / "src"
    / "Step1_CleanOpenalexCrossrefData"
    / "00_RunAll_Step1_CleanOpenAlexCrossrefData.py",
    PROJECT_ROOT
    / "src"
    / "Step2_MergeAllDatasets"
    / "00_RunAll_Step2_MergeAllDatasets.py",
    PROJECT_ROOT
    / "src"
    / "Step3_TrainingModelClassifyJELCodes"
    / "00_RunAll_Step3_TrainingModelClassifyJELCodes.py",
]


def main() -> None:
    check_scripts_exist(SCRIPTS_TO_RUN)

    print("Running full JournalPublicationProject pipeline")
    for script in SCRIPTS_TO_RUN:
        run_script(script)

    print()
    print("Full project pipeline finished successfully.")


def check_scripts_exist(scripts: list[Path]) -> None:
    missing_scripts = [script for script in scripts if not script.exists()]
    if missing_scripts:
        missing_text = "\n".join(str(script) for script in missing_scripts)
        raise FileNotFoundError(f"Missing pipeline script(s):\n{missing_text}")


def run_script(script: Path) -> None:
    print()
    print(f"Running {script.relative_to(PROJECT_ROOT)}...")
    subprocess.run(
        [sys.executable, str(script)],
        cwd=PROJECT_ROOT,
        check=True,
    )
    print(f"Finished {script.relative_to(PROJECT_ROOT)}.")


if __name__ == "__main__":
    main()
