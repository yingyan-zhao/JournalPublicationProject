from collections import defaultdict
from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

TRAINING_INPUT_CSV = Path("data/trainingmodel/JEL_Training_Data_With_JEL.csv")
PREDICTION_INPUT_CSV = Path("data/trainingmodel/JEL_Training_Data_Without_JEL.csv")

TFIDF_VALIDATION_CSV = Path("data/trainingmodel/JEL_Codes_1_TFIDF_Validation_Predictions.csv")
SPECTER2_VALIDATION_CSV = Path("data/trainingmodel/JEL_Codes_1_SPECTER2_Validation_Predictions.csv")
SCIBERT_VALIDATION_CSV = Path("data/trainingmodel/JEL_Codes_1_SciBERT_Validation_Predictions.csv")

TFIDF_PREDICTION_CSV = Path("data/trainingmodel/JEL_Training_Data_Without_JEL_Predicted.csv")
SPECTER2_PREDICTION_CSV = Path("data/trainingmodel/JEL_Training_Data_Without_JEL_Predicted_SPECTER2.csv")
SCIBERT_PREDICTION_CSV = Path("data/trainingmodel/JEL_Training_Data_Without_JEL_Predicted_SciBERT.csv")

PREDICTION_OUTPUT_CSV = Path("data/trainingmodel/JEL_Training_Data_Without_JEL_Predicted_Ensemble.csv")
COMBINED_OUTPUT_CSV = Path("data/trainingmodel/JEL_Training_Data_With_Observed_And_Predicted_Ensemble.csv")
VALIDATION_OUTPUT_CSV = Path("data/trainingmodel/JEL_Codes_1_Ensemble_Validation_Predictions.csv")
CONFUSION_MATRIX_OUTPUT_CSV = Path("data/trainingmodel/JEL_Codes_1_Ensemble_Confusion_Matrix.csv")
ACCURACY_OUTPUT_CSV = Path("data/trainingmodel/JEL_Codes_1_Ensemble_Accuracy_Summary.csv")
ENSEMBLE_TUNING_OUTPUT_CSV = Path("data/trainingmodel/JEL_Codes_1_Ensemble_SciBERT_Weight_Tuning.csv")
REPORT_OUTPUT_TXT = Path("data/trainingmodel/JEL_Codes_1_Ensemble_Report.txt")

LABEL_COLUMN = "jel_code_1"
PREDICTED_LABEL_COLUMN = "jel_code_1_predicted_ensemble"
PREDICTED_CONFIDENCE_COLUMN = "jel_code_1_predicted_ensemble_confidence"
JEL_SOURCE_COLUMN = "jel_code_1_ensemble_source"
SCIBERT_WEIGHT_OPTIONS = [1.0, 1.5, 2.0, 2.5]

MODEL_SPECS = [
    {
        "name": "tfidf",
        "validation_csv": TFIDF_VALIDATION_CSV,
        "prediction_csv": TFIDF_PREDICTION_CSV,
        "validation_prediction_column": "tfidf_jel_code_1_predicted",
        "validation_confidence_column": "tfidf_confidence",
        "prediction_column": "jel_code_1_predicted_tfidf",
        "prediction_confidence_column": "jel_code_1_predicted_tfidf_confidence",
    },
    {
        "name": "specter2",
        "validation_csv": SPECTER2_VALIDATION_CSV,
        "prediction_csv": SPECTER2_PREDICTION_CSV,
        "validation_prediction_column": "specter2_jel_code_1_predicted",
        "validation_confidence_column": "specter2_confidence",
        "prediction_column": "jel_code_1_predicted_specter2",
        "prediction_confidence_column": "jel_code_1_predicted_specter2_confidence",
    },
    {
        "name": "scibert",
        "validation_csv": SCIBERT_VALIDATION_CSV,
        "prediction_csv": SCIBERT_PREDICTION_CSV,
        "validation_prediction_column": "scibert_jel_code_1_predicted",
        "validation_confidence_column": "scibert_confidence",
        "prediction_column": "jel_code_1_predicted_scibert",
        "prediction_confidence_column": "jel_code_1_predicted_scibert_confidence",
    },
]


def main() -> None:
    training_data = read_data(TRAINING_INPUT_CSV)
    prediction_data = read_data(PREDICTION_INPUT_CSV)

    ensemble_tuning = tune_scibert_weight()
    best_scibert_weight = float(ensemble_tuning.iloc[0]["scibert_weight"])
    print_ensemble_tuning(ensemble_tuning)

    validation = build_validation_ensemble(scibert_weight=best_scibert_weight)
    prediction = build_prediction_ensemble(
        prediction_data,
        scibert_weight=best_scibert_weight,
    )

    validation_accuracy = accuracy_score(
        validation["jel_code_1_true"],
        validation[PREDICTED_LABEL_COLUMN],
    )
    any_code_match = predicted_label_in_jel_code_full(
        validation["jel_code_full"],
        validation[PREDICTED_LABEL_COLUMN],
    )
    any_code_accuracy = float(any_code_match.mean())
    validation["ensemble_prediction_in_jel_code_full"] = any_code_match.astype(int)
    validation_report = classification_report(
        validation["jel_code_1_true"],
        validation[PREDICTED_LABEL_COLUMN],
        zero_division=0,
    )
    model_accuracies = model_validation_accuracies(validation)
    model_any_code_accuracies = model_any_code_accuracies_from_validation(validation)
    accuracy_summary = build_accuracy_summary(
        ensemble_accuracy=validation_accuracy,
        ensemble_any_code_accuracy=any_code_accuracy,
        model_accuracies=model_accuracies,
        model_any_code_accuracies=model_any_code_accuracies,
    )

    predicted_data = prediction_data.copy()
    predicted_data[PREDICTED_LABEL_COLUMN] = prediction[PREDICTED_LABEL_COLUMN]
    predicted_data[PREDICTED_CONFIDENCE_COLUMN] = prediction[PREDICTED_CONFIDENCE_COLUMN].round(4)
    predicted_data[JEL_SOURCE_COLUMN] = "predicted_ensemble"
    predicted_data["ensemble_scibert_weight"] = best_scibert_weight
    for model_spec in MODEL_SPECS:
        model_name = model_spec["name"]
        predicted_data[f"{model_name}_prediction_used"] = prediction[f"{model_name}_prediction"]
        predicted_data[f"{model_name}_confidence_used"] = prediction[f"{model_name}_confidence"].round(4)

    observed_data = training_data.copy()
    observed_data[PREDICTED_LABEL_COLUMN] = observed_data[LABEL_COLUMN]
    observed_data[PREDICTED_CONFIDENCE_COLUMN] = ""
    observed_data[JEL_SOURCE_COLUMN] = "observed"
    combined = pd.concat([observed_data, predicted_data], ignore_index=True, sort=False)

    PREDICTION_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    predicted_data.to_csv(PREDICTION_OUTPUT_CSV, index=False)
    combined.to_csv(COMBINED_OUTPUT_CSV, index=False)
    validation.to_csv(VALIDATION_OUTPUT_CSV, index=False)
    confusion = write_confusion_matrix(validation["jel_code_1_true"], validation[PREDICTED_LABEL_COLUMN])
    write_accuracy_summary(accuracy_summary)
    write_ensemble_tuning(ensemble_tuning)
    write_report(
        validation_accuracy=validation_accuracy,
        any_code_accuracy=any_code_accuracy,
        validation_report=validation_report,
        model_accuracies=model_accuracies,
        model_any_code_accuracies=model_any_code_accuracies,
        accuracy_summary=accuracy_summary,
        ensemble_tuning=ensemble_tuning,
        best_scibert_weight=best_scibert_weight,
        confusion=confusion,
        validation_rows=len(validation),
        prediction_rows=len(predicted_data),
    )

    print("JEL code first-letter ensemble summary:")
    print(f"  Training input CSV: {TRAINING_INPUT_CSV}")
    print(f"  Prediction input CSV: {PREDICTION_INPUT_CSV}")
    print(f"  Validation rows: {len(validation)}")
    print(f"  Rows predicted: {len(predicted_data)}")
    print(f"  Best SciBERT ensemble weight: {best_scibert_weight}")
    for model_name, accuracy in model_accuracies.items():
        print(f"  {model_name} validation accuracy: {accuracy:.4f}")
        print(f"  {model_name} any-code accuracy: {model_any_code_accuracies[model_name]:.4f}")
    print(f"  Ensemble validation accuracy: {validation_accuracy:.4f}")
    print(f"  Ensemble any-code accuracy: {any_code_accuracy:.4f}")
    print(f"  Prediction output CSV: {PREDICTION_OUTPUT_CSV}")
    print(f"  Combined output CSV: {COMBINED_OUTPUT_CSV}")
    print(f"  Validation prediction CSV: {VALIDATION_OUTPUT_CSV}")
    print(f"  Confusion matrix CSV: {CONFUSION_MATRIX_OUTPUT_CSV}")
    print(f"  Accuracy summary CSV: {ACCURACY_OUTPUT_CSV}")
    print(f"  SciBERT weight tuning CSV: {ENSEMBLE_TUNING_OUTPUT_CSV}")
    print(f"  Model report: {REPORT_OUTPUT_TXT}")


def read_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def tune_scibert_weight() -> pd.DataFrame:
    rows = []
    for scibert_weight in SCIBERT_WEIGHT_OPTIONS:
        validation = build_validation_ensemble(scibert_weight=scibert_weight)
        accuracy = accuracy_score(
            validation["jel_code_1_true"],
            validation[PREDICTED_LABEL_COLUMN],
        )
        any_code_match = predicted_label_in_jel_code_full(
            validation["jel_code_full"],
            validation[PREDICTED_LABEL_COLUMN],
        )
        rows.append(
            {
                "scibert_weight": scibert_weight,
                "accuracy": accuracy,
                "any_code_accuracy": float(any_code_match.mean()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["accuracy", "any_code_accuracy", "scibert_weight"],
        ascending=[False, False, True],
        kind="mergesort",
    )


def print_ensemble_tuning(ensemble_tuning: pd.DataFrame) -> None:
    print("SciBERT ensemble weight tuning:")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 120):
        print(ensemble_tuning.to_string(index=False))


def write_ensemble_tuning(ensemble_tuning: pd.DataFrame) -> None:
    ENSEMBLE_TUNING_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    ensemble_tuning.to_csv(ENSEMBLE_TUNING_OUTPUT_CSV, index=False)


def build_validation_ensemble(scibert_weight: float) -> pd.DataFrame:
    merged = None
    for model_spec in MODEL_SPECS:
        model_validation = read_model_validation(model_spec)
        if merged is None:
            merged = model_validation
        else:
            merged = merged.merge(
                model_validation,
                on=["validation_row_id", "jel_code_1_true", "jel_code_full"],
                how="inner",
            )

    if merged is None or merged.empty:
        raise ValueError("No validation predictions were available to ensemble.")

    ensemble = merged.copy()
    ensemble["ensemble_scibert_weight"] = scibert_weight
    ensemble[[PREDICTED_LABEL_COLUMN, PREDICTED_CONFIDENCE_COLUMN]] = ensemble.apply(
        lambda row: pd.Series(ensemble_vote_from_row(row, scibert_weight=scibert_weight)),
        axis=1,
    )
    return ensemble


def read_model_validation(model_spec: dict) -> pd.DataFrame:
    data = read_data(model_spec["validation_csv"])
    required_columns = [
        "validation_row_id",
        "jel_code_1_true",
        "jel_code_full",
        model_spec["validation_prediction_column"],
        model_spec["validation_confidence_column"],
    ]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"{model_spec['validation_csv']} is missing columns: {missing_columns}")

    model_name = model_spec["name"]
    selected = data[required_columns].copy()
    selected = selected.rename(
        columns={
            model_spec["validation_prediction_column"]: f"{model_name}_prediction",
            model_spec["validation_confidence_column"]: f"{model_name}_confidence",
        }
    )
    return selected


def build_prediction_ensemble(prediction_data: pd.DataFrame, scibert_weight: float) -> pd.DataFrame:
    ensemble = pd.DataFrame(index=prediction_data.index)
    expected_rows = len(prediction_data)

    for model_spec in MODEL_SPECS:
        model_prediction = read_data(model_spec["prediction_csv"])
        if len(model_prediction) != expected_rows:
            raise ValueError(
                f"{model_spec['prediction_csv']} has {len(model_prediction)} rows, "
                f"but {PREDICTION_INPUT_CSV} has {expected_rows} rows."
            )
        missing_columns = [
            column for column in [
                model_spec["prediction_column"],
                model_spec["prediction_confidence_column"],
            ]
            if column not in model_prediction.columns
        ]
        if missing_columns:
            raise ValueError(f"{model_spec['prediction_csv']} is missing columns: {missing_columns}")

        model_name = model_spec["name"]
        ensemble[f"{model_name}_prediction"] = model_prediction[model_spec["prediction_column"]]
        ensemble[f"{model_name}_confidence"] = model_prediction[model_spec["prediction_confidence_column"]]

    ensemble["ensemble_scibert_weight"] = scibert_weight
    ensemble[[PREDICTED_LABEL_COLUMN, PREDICTED_CONFIDENCE_COLUMN]] = ensemble.apply(
        lambda row: pd.Series(ensemble_vote_from_row(row, scibert_weight=scibert_weight)),
        axis=1,
    )
    return ensemble


def ensemble_vote_from_row(row: pd.Series, scibert_weight: float) -> tuple[str, float]:
    votes = []
    for model_spec in MODEL_SPECS:
        model_name = model_spec["name"]
        prediction = clean_text(row.get(f"{model_name}_prediction", ""))
        confidence = to_float(row.get(f"{model_name}_confidence", 0.0))
        if prediction:
            weight = model_vote_weight(model_name, scibert_weight)
            votes.append((prediction, confidence, weight))

    if not votes:
        return "", 0.0

    vote_count = defaultdict(int)
    confidence_sum = defaultdict(float)
    for prediction, confidence, weight in votes:
        vote_count[prediction] += weight
        confidence_sum[prediction] += confidence * weight

    best_label = sorted(
        vote_count,
        key=lambda label: (
            vote_count[label],
            confidence_sum[label],
            -ord(label[0]) if label else 0,
        ),
        reverse=True,
    )[0]
    average_confidence = confidence_sum[best_label] / vote_count[best_label]
    return best_label, average_confidence


def model_vote_weight(model_name: str, scibert_weight: float) -> float:
    if model_name == "scibert":
        return scibert_weight
    return 1.0


def model_validation_accuracies(validation: pd.DataFrame) -> dict[str, float]:
    accuracies = {}
    true_labels = validation["jel_code_1_true"]
    for model_spec in MODEL_SPECS:
        model_name = model_spec["name"]
        prediction_column = f"{model_name}_prediction"
        accuracies[model_name] = accuracy_score(true_labels, validation[prediction_column])
    return accuracies


def model_any_code_accuracies_from_validation(validation: pd.DataFrame) -> dict[str, float]:
    accuracies = {}
    for model_spec in MODEL_SPECS:
        model_name = model_spec["name"]
        prediction_column = f"{model_name}_prediction"
        any_code_match = predicted_label_in_jel_code_full(
            validation["jel_code_full"],
            validation[prediction_column],
        )
        accuracies[model_name] = float(any_code_match.mean())
    return accuracies


def build_accuracy_summary(
    ensemble_accuracy: float,
    ensemble_any_code_accuracy: float,
    model_accuracies: dict[str, float],
    model_any_code_accuracies: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for model_spec in MODEL_SPECS:
        model_name = model_spec["name"]
        rows.append(
            {
                "model": model_name,
                "accuracy": model_accuracies.get(model_name, np.nan),
                "any_code_accuracy": model_any_code_accuracies.get(model_name, np.nan),
            }
        )

    rows.append(
        {
            "model": "ensemble",
            "accuracy": ensemble_accuracy,
            "any_code_accuracy": ensemble_any_code_accuracy,
        }
    )
    return pd.DataFrame(rows)


def write_accuracy_summary(accuracy_summary: pd.DataFrame) -> None:
    ACCURACY_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    accuracy_summary.to_csv(ACCURACY_OUTPUT_CSV, index=False)


def predicted_label_in_jel_code_full(
    jel_code_full: pd.Series,
    predicted_labels: pd.Series,
) -> pd.Series:
    predicted = predicted_labels.fillna("").astype(str).str.strip()
    full_codes = jel_code_full.fillna("").astype(str)
    return pd.Series(
        [
            prediction != "" and prediction in split_jel_code_full(codes)
            for prediction, codes in zip(predicted, full_codes)
        ],
        index=jel_code_full.index,
    )


def split_jel_code_full(value: str) -> set[str]:
    return {
        code.strip()
        for code in str(value).split(";")
        if code.strip()
    }


def write_confusion_matrix(true_labels: pd.Series, predicted_labels: pd.Series) -> pd.DataFrame:
    labels_order = sorted(true_labels.unique())
    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels_order)
    confusion = pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in labels_order],
        columns=[f"pred_{label}" for label in labels_order],
    )
    CONFUSION_MATRIX_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    confusion.to_csv(CONFUSION_MATRIX_OUTPUT_CSV)
    return confusion


def write_report(
    validation_accuracy: float,
    any_code_accuracy: float,
    validation_report: str,
    model_accuracies: dict[str, float],
    model_any_code_accuracies: dict[str, float],
    accuracy_summary: pd.DataFrame,
    ensemble_tuning: pd.DataFrame,
    best_scibert_weight: float,
    confusion: pd.DataFrame,
    validation_rows: int,
    prediction_rows: int,
) -> None:
    REPORT_OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_OUTPUT_TXT.open("w", encoding="utf-8") as file:
        file.write("JEL code first-letter ensemble model\n")
        file.write("====================================\n\n")
        file.write("Model: majority vote ensemble with confidence tie-break\n")
        file.write(f"Label column: {LABEL_COLUMN}\n")
        file.write(f"SciBERT weight options: {SCIBERT_WEIGHT_OPTIONS}\n")
        file.write(f"Best SciBERT weight: {best_scibert_weight}\n")
        file.write(f"Validation rows: {validation_rows}\n")
        file.write(f"Prediction rows: {prediction_rows}\n\n")
        file.write("SciBERT weight tuning:\n")
        file.write(f"  Tuning results CSV: {ENSEMBLE_TUNING_OUTPUT_CSV}\n")
        file.write(ensemble_tuning.to_string(index=False))
        file.write("\n\n")
        file.write("Input models:\n")
        for model_spec in MODEL_SPECS:
            model_name = model_spec["name"]
            file.write(
                f"  {model_name}: "
                f"accuracy={model_accuracies.get(model_name, np.nan):.4f}, "
                f"any-code accuracy={model_any_code_accuracies.get(model_name, np.nan):.4f}\n"
            )
        file.write("\n")
        file.write(f"Ensemble validation accuracy: {validation_accuracy:.4f}\n")
        file.write(f"Ensemble any-code accuracy: {any_code_accuracy:.4f}\n")
        file.write(f"Accuracy summary CSV: {ACCURACY_OUTPUT_CSV}\n")
        file.write("Accuracy summary:\n")
        file.write(accuracy_summary.to_string(index=False))
        file.write("\n\n")
        file.write(f"Confusion matrix CSV: {CONFUSION_MATRIX_OUTPUT_CSV}\n")
        file.write("Confusion matrix:\n")
        file.write(confusion.to_string())
        file.write("\n\n")
        file.write(f"Validation prediction CSV: {VALIDATION_OUTPUT_CSV}\n\n")
        file.write("Validation classification report:\n")
        file.write(validation_report)
        file.write("\n")


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def to_float(value) -> float:
    try:
        if pd.isna(value):
            return 0.0
        text = str(value).strip()
        if text == "":
            return 0.0
        return float(text)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
