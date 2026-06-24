from collections import Counter
from pathlib import Path
import os

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

TFIDF_VALIDATION_CSV = Path("data/processed/JEL_Codes_1_TFIDF_Validation_Predictions.csv")
SPECTER2_VALIDATION_CSV = Path("data/processed/JEL_Codes_1_SPECTER2_Validation_Predictions.csv")
SCIBERT_VALIDATION_CSV = Path("data/processed/JEL_Codes_1_SciBERT_Validation_Predictions.csv")

ENSEMBLE_VALIDATION_OUTPUT_CSV = Path("data/processed/JEL_Codes_1_Ensemble_Validation_Predictions.csv")
ENSEMBLE_REPORT_OUTPUT_TXT = Path("data/processed/JEL_Codes_1_Ensemble_Report.txt")


def main() -> None:
    check_required_files()

    tfidf = read_validation_predictions(TFIDF_VALIDATION_CSV)
    specter2 = read_validation_predictions(SPECTER2_VALIDATION_CSV)
    scibert = read_validation_predictions(SCIBERT_VALIDATION_CSV)

    validation = merge_validation_predictions(tfidf, specter2, scibert)
    validation = add_ensemble_predictions(validation)

    accuracy = accuracy_score(
        validation["jel_codes_1_true"],
        validation["ensemble_jel_codes_1_predicted"],
    )
    report = classification_report(
        validation["jel_codes_1_true"],
        validation["ensemble_jel_codes_1_predicted"],
        zero_division=0,
    )

    ENSEMBLE_VALIDATION_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    validation.to_csv(ENSEMBLE_VALIDATION_OUTPUT_CSV, index=False)
    write_report(validation, accuracy, report)

    print("JEL code first-letter ensemble summary:")
    print(f"  Validation rows used: {len(validation)}")
    print(f"  Ensemble accuracy: {accuracy:.4f}")
    print(f"  Output CSV: {ENSEMBLE_VALIDATION_OUTPUT_CSV}")
    print(f"  Report: {ENSEMBLE_REPORT_OUTPUT_TXT}")
    print("  Agreement counts:")
    print(validation["ensemble_agreement"].value_counts().sort_index().to_string())


def check_required_files() -> None:
    missing_files = [
        path for path in [
            TFIDF_VALIDATION_CSV,
            SPECTER2_VALIDATION_CSV,
            SCIBERT_VALIDATION_CSV,
        ]
        if not path.exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Run all three model scripts first. Missing validation files: "
            + ", ".join(str(path) for path in missing_files)
        )


def read_validation_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def merge_validation_predictions(
    tfidf: pd.DataFrame,
    specter2: pd.DataFrame,
    scibert: pd.DataFrame,
) -> pd.DataFrame:
    validation = tfidf.merge(
        specter2,
        on=["validation_row_id", "jel_codes_1_true"],
        how="inner",
    )
    validation = validation.merge(
        scibert,
        on=["validation_row_id", "jel_codes_1_true"],
        how="inner",
    )
    return validation


def add_ensemble_predictions(validation: pd.DataFrame) -> pd.DataFrame:
    result = validation.copy()
    ensemble_values = result.apply(ensemble_one_row, axis=1, result_type="expand")
    ensemble_values.columns = [
        "ensemble_jel_codes_1_predicted",
        "ensemble_source",
        "ensemble_confidence",
        "ensemble_agreement",
    ]
    return pd.concat([result, ensemble_values], axis=1)


def ensemble_one_row(row: pd.Series) -> tuple[str, str, float, int]:
    predictions = [
        row["tfidf_jel_codes_1_predicted"],
        row["specter2_jel_codes_1_predicted"],
        row["scibert_jel_codes_1_predicted"],
    ]
    confidences = [
        to_float(row["tfidf_confidence"]),
        to_float(row["specter2_confidence"]),
        to_float(row["scibert_confidence"]),
    ]
    model_names = ["tfidf", "specter2", "scibert"]

    counts = Counter(predictions)
    label, agreement = counts.most_common(1)[0]
    if agreement >= 2:
        source = "unanimous" if agreement == 3 else "majority_vote"
        label_confidences = [
            confidence
            for prediction, confidence in zip(predictions, confidences)
            if prediction == label
        ]
        confidence = max(label_confidences)
        return label, source, round(confidence, 4), agreement

    best_index = max(range(len(confidences)), key=lambda index: confidences[index])
    return (
        predictions[best_index],
        f"highest_confidence_{model_names[best_index]}",
        round(confidences[best_index], 4),
        1,
    )


def to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def write_report(validation: pd.DataFrame, accuracy: float, report: str) -> None:
    ENSEMBLE_REPORT_OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with ENSEMBLE_REPORT_OUTPUT_TXT.open("w", encoding="utf-8") as file:
        file.write("JEL code first-letter ensemble report\n")
        file.write("=====================================\n\n")
        file.write("Rule:\n")
        file.write("  1. If at least two models agree, use the agreed label.\n")
        file.write("  2. If all three disagree, use the label from the highest-confidence model.\n\n")
        file.write(f"Validation rows: {len(validation)}\n")
        file.write(f"Ensemble accuracy: {accuracy:.4f}\n\n")
        file.write("Agreement counts:\n")
        file.write(validation["ensemble_agreement"].value_counts().sort_index().to_string())
        file.write("\n\n")
        file.write("Source counts:\n")
        file.write(validation["ensemble_source"].value_counts().to_string())
        file.write("\n\n")
        file.write("Classification report:\n")
        file.write(report)
        file.write("\n")


if __name__ == "__main__":
    main()
