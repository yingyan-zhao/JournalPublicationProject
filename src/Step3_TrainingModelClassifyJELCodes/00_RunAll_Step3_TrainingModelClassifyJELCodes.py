from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")
STEP_DIR = PROJECT_ROOT / "src" / "Step3_TrainingModelClassifyJELCodes"

SCRIPTS_TO_RUN = [
    STEP_DIR / "01_Training_jel_codes_DataPreparation.py",
    STEP_DIR / "02_TrainPredict_Single_jel_codes_TFIDF.py",
    STEP_DIR / "03_TrainPredict_Single_jel_codes_specter2.py",
    STEP_DIR / "04_TrainPredict_Single_jel_codes_scibert.py",
    STEP_DIR / "05_Ensemble_Single_jel_codes.py",
    STEP_DIR / "06_TrainPredict_Multi_jel_codes_TFIDF.py",
    STEP_DIR / "07_TrainPredict_Multi_jel_codes_specter2.py",
    STEP_DIR / "08_TrainPredict_Multi_jel_codes_scibert.py",
    STEP_DIR / "09_Ensemble_Multi_jel_codes.py",
    STEP_DIR / "10_Multi_jel_codes_CleanPredictedJELdata.py",
]


def main() -> None:
    check_scripts_exist(SCRIPTS_TO_RUN)

    print("Running Step 3: Train, ensemble, and clean JEL prediction models")
    for script in SCRIPTS_TO_RUN:
        run_script(script)

    print()
    print("Step 3 JEL prediction pipeline finished successfully.")


def check_scripts_exist(scripts: list[Path]) -> None:
    missing_scripts = [script for script in scripts if not script.exists()]
    if missing_scripts:
        missing_text = "\n".join(str(script) for script in missing_scripts)
        raise FileNotFoundError(f"Missing Step 3 script(s):\n{missing_text}")


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
