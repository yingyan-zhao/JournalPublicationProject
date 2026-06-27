from collections import defaultdict
from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

TRAINING_INPUT_CSV = Path("data/processed/JEL_Training_Data_With_JEL.csv")
PREDICTION_INPUT_CSV = Path("data/processed/JEL_Training_Data_Without_JEL.csv")

TFIDF_VALIDATION_CSV = Path("data/processed/JEL_Codes_Multi_TFIDF_Validation_Predictions.csv")
SPECTER2_VALIDATION_CSV = Path("data/processed/JEL_Codes_Multi_SPECTER2_Validation_Predictions.csv")
SCIBERT_VALIDATION_CSV = Path("data/processed/JEL_Codes_Multi_SciBERT_Validation_Predictions.csv")

TFIDF_PREDICTION_CSV = Path("data/processed/JEL_Training_Data_Without_JEL_Predicted_Multi_TFIDF.csv")
SPECTER2_PREDICTION_CSV = Path("data/processed/JEL_Training_Data_Without_JEL_Predicted_Multi_SPECTER2.csv")
SCIBERT_PREDICTION_CSV = Path("data/processed/JEL_Training_Data_Without_JEL_Predicted_Multi_SciBERT.csv")

PREDICTION_OUTPUT_CSV = Path("data/processed/JEL_Training_Data_Without_JEL_Predicted_Multi_Ensemble.csv")
COMBINED_OUTPUT_CSV = Path("data/processed/JEL_Training_Data_With_Observed_And_Predicted_Multi_Ensemble.csv")
VALIDATION_OUTPUT_CSV = Path("data/processed/JEL_Codes_Multi_Ensemble_Validation_Predictions.csv")
REPORT_OUTPUT_TXT = Path("data/processed/JEL_Codes_Multi_Ensemble_Report.txt")

LABEL_COLUMN = "jel_code_full"
PREDICTED_LABEL_COLUMN = "jel_code_full_predicted_ensemble"
PREDICTED_CONFIDENCE_COLUMN = "jel_code_full_predicted_ensemble_confidence"
JEL_SOURCE_COLUMN = "jel_code_full_ensemble_source"
MIN_MODEL_VOTES = 2

MODEL_SPECS = [
    {
        "name": "tfidf",
        "validation_csv": TFIDF_VALIDATION_CSV,
        "prediction_csv": TFIDF_PREDICTION_CSV,
        "validation_prediction_column": "jel_code_full_predicted_tfidf",
        "validation_confidence_column": "tfidf_max_confidence",
        "prediction_column": "jel_code_full_predicted_tfidf",
        "prediction_confidence_column": "jel_code_full_predicted_tfidf_max_confidence",
    },
    {
        "name": "specter2",
        "validation_csv": SPECTER2_VALIDATION_CSV,
        "prediction_csv": SPECTER2_PREDICTION_CSV,
        "validation_prediction_column": "jel_code_full_predicted_specter2",
        "validation_confidence_column": "specter2_max_confidence",
        "prediction_column": "jel_code_full_predicted_specter2",
        "prediction_confidence_column": "jel_code_full_predicted_specter2_max_confidence",
    },
    {
        "name": "scibert",
        "validation_csv": SCIBERT_VALIDATION_CSV,
        "prediction_csv": SCIBERT_PREDICTION_CSV,
        "validation_prediction_column": "jel_code_full_predicted_scibert",
        "validation_confidence_column": "scibert_max_confidence",
        "prediction_column": "jel_code_full_predicted_scibert",
        "prediction_confidence_column": "jel_code_full_predicted_scibert_max_confidence",
    },
]


def main() -> None:
    training_data = read_data(TRAINING_INPUT_CSV)
    prediction_data = read_data(PREDICTION_INPUT_CSV)

    label_binarizer = build_label_binarizer(training_data[LABEL_COLUMN])
    validation = build_validation_ensemble(label_binarizer)
    prediction = build_prediction_ensemble(prediction_data)

    true_validation_matrix = label_binarizer.transform(
        validation["jel_code_full_true"].apply(split_label_cell)
    )
    ensemble_validation_matrix = label_binarizer.transform(
        validation[PREDICTED_LABEL_COLUMN].apply(split_label_cell)
    )
    ensemble_metrics = multilabel_metrics(true_validation_matrix, ensemble_validation_matrix)
    model_metrics = model_validation_metrics(validation, label_binarizer)

    predicted_data = prediction_data.copy()
    predicted_data[PREDICTED_LABEL_COLUMN] = prediction[PREDICTED_LABEL_COLUMN]
    predicted_data[PREDICTED_CONFIDENCE_COLUMN] = prediction[PREDICTED_CONFIDENCE_COLUMN].round(4)
    predicted_data[JEL_SOURCE_COLUMN] = "predicted_multi_ensemble"
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
    write_report(
        ensemble_metrics=ensemble_metrics,
        model_metrics=model_metrics,
        validation_rows=len(validation),
        prediction_rows=len(predicted_data),
        label_classes=label_binarizer.classes_,
    )

    print("Multi-label JEL ensemble summary:")
    print(f"  Training input CSV: {TRAINING_INPUT_CSV}")
    print(f"  Prediction input CSV: {PREDICTION_INPUT_CSV}")
    print(f"  Validation rows: {len(validation)}")
    print(f"  Rows predicted: {len(predicted_data)}")
    for model_name, metrics in model_metrics.items():
        print(f"  {model_name} micro F1: {metrics['micro_f1']:.4f}")
        print(f"  {model_name} macro F1: {metrics['macro_f1']:.4f}")
        print(f"  {model_name} hamming loss: {metrics['hamming_loss']:.4f}")
        print(f"  {model_name} subset accuracy: {metrics['subset_accuracy']:.4f}")
    print(f"  Ensemble micro F1: {ensemble_metrics['micro_f1']:.4f}")
    print(f"  Ensemble macro F1: {ensemble_metrics['macro_f1']:.4f}")
    print(f"  Ensemble precision: {ensemble_metrics['micro_precision']:.4f}")
    print(f"  Ensemble recall: {ensemble_metrics['micro_recall']:.4f}")
    print(f"  Ensemble hamming loss: {ensemble_metrics['hamming_loss']:.4f}")
    print(f"  Ensemble subset accuracy: {ensemble_metrics['subset_accuracy']:.4f}")
    print(f"  Prediction output CSV: {PREDICTION_OUTPUT_CSV}")
    print(f"  Combined output CSV: {COMBINED_OUTPUT_CSV}")
    print(f"  Validation prediction CSV: {VALIDATION_OUTPUT_CSV}")
    print(f"  Model report: {REPORT_OUTPUT_TXT}")


def read_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def build_label_binarizer(label_column: pd.Series) -> MultiLabelBinarizer:
    label_binarizer = MultiLabelBinarizer()
    label_binarizer.fit(label_column.apply(split_label_cell))
    return label_binarizer


def build_validation_ensemble(label_binarizer: MultiLabelBinarizer) -> pd.DataFrame:
    merged = None
    for model_spec in MODEL_SPECS:
        model_validation = read_model_validation(model_spec)
        if merged is None:
            merged = model_validation
        else:
            merged = merged.merge(
                model_validation,
                on=["validation_row_id", "jel_code_full_true"],
                how="inner",
            )

    if merged is None or merged.empty:
        raise ValueError("No validation predictions were available to ensemble.")

    ensemble = merged.copy()
    ensemble[[PREDICTED_LABEL_COLUMN, PREDICTED_CONFIDENCE_COLUMN]] = ensemble.apply(
        lambda row: pd.Series(ensemble_multilabel_vote_from_row(row)),
        axis=1,
    )
    true_matrix = label_binarizer.transform(ensemble["jel_code_full_true"].apply(split_label_cell))
    predicted_matrix = label_binarizer.transform(ensemble[PREDICTED_LABEL_COLUMN].apply(split_label_cell))
    ensemble["exact_match"] = (true_matrix == predicted_matrix).all(axis=1).astype(int)
    return ensemble


def read_model_validation(model_spec: dict) -> pd.DataFrame:
    data = read_data(model_spec["validation_csv"])
    required_columns = [
        "validation_row_id",
        "jel_code_full_true",
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


def build_prediction_ensemble(prediction_data: pd.DataFrame) -> pd.DataFrame:
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
            column
            for column in [
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

    ensemble[[PREDICTED_LABEL_COLUMN, PREDICTED_CONFIDENCE_COLUMN]] = ensemble.apply(
        lambda row: pd.Series(ensemble_multilabel_vote_from_row(row)),
        axis=1,
    )
    return ensemble


def ensemble_multilabel_vote_from_row(row: pd.Series) -> tuple[str, float]:
    model_predictions = []
    label_votes = defaultdict(int)
    label_confidence_sum = defaultdict(float)

    for model_spec in MODEL_SPECS:
        model_name = model_spec["name"]
        labels = split_label_cell(row.get(f"{model_name}_prediction", ""))
        confidence = to_float(row.get(f"{model_name}_confidence", 0.0))
        if not labels:
            continue
        model_predictions.append((labels, confidence))
        for label in labels:
            label_votes[label] += 1
            label_confidence_sum[label] += confidence

    if not model_predictions:
        return "", 0.0

    selected_labels = [
        label
        for label, vote_count in label_votes.items()
        if vote_count >= MIN_MODEL_VOTES
    ]
    if selected_labels:
        selected_labels = sorted(selected_labels)
        confidence = average_selected_label_confidence(
            selected_labels,
            label_votes,
            label_confidence_sum,
        )
        return "; ".join(selected_labels), confidence

    fallback_labels, fallback_confidence = max(
        model_predictions,
        key=lambda labels_and_confidence: (
            labels_and_confidence[1],
            len(labels_and_confidence[0]),
            "; ".join(labels_and_confidence[0]),
        ),
    )
    return "; ".join(sorted(fallback_labels)), fallback_confidence


def average_selected_label_confidence(
    selected_labels: list[str],
    label_votes: dict[str, int],
    label_confidence_sum: dict[str, float],
) -> float:
    confidence_values = [
        label_confidence_sum[label] / label_votes[label]
        for label in selected_labels
        if label_votes[label] > 0
    ]
    if not confidence_values:
        return 0.0
    return float(np.mean(confidence_values))


def model_validation_metrics(
    validation: pd.DataFrame,
    label_binarizer: MultiLabelBinarizer,
) -> dict[str, dict[str, float]]:
    metrics = {}
    true_matrix = label_binarizer.transform(validation["jel_code_full_true"].apply(split_label_cell))
    for model_spec in MODEL_SPECS:
        model_name = model_spec["name"]
        prediction_column = f"{model_name}_prediction"
        predicted_matrix = label_binarizer.transform(validation[prediction_column].apply(split_label_cell))
        metrics[model_name] = multilabel_metrics(true_matrix, predicted_matrix)
    return metrics


def multilabel_metrics(true_matrix: np.ndarray, predicted_matrix: np.ndarray) -> dict[str, float]:
    return {
        "micro_f1": f1_score(true_matrix, predicted_matrix, average="micro", zero_division=0),
        "macro_f1": f1_score(true_matrix, predicted_matrix, average="macro", zero_division=0),
        "micro_precision": precision_score(true_matrix, predicted_matrix, average="micro", zero_division=0),
        "macro_precision": precision_score(true_matrix, predicted_matrix, average="macro", zero_division=0),
        "micro_recall": recall_score(true_matrix, predicted_matrix, average="micro", zero_division=0),
        "macro_recall": recall_score(true_matrix, predicted_matrix, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(true_matrix, predicted_matrix),
        "subset_accuracy": float((true_matrix == predicted_matrix).all(axis=1).mean()),
    }


def split_label_cell(value: object) -> list[str]:
    if pd.isna(value):
        return []
    labels = []
    seen = set()
    for label in str(value).split(";"):
        cleaned = clean_text(label)
        if cleaned and cleaned not in seen:
            labels.append(cleaned)
            seen.add(cleaned)
    return labels


def write_report(
    ensemble_metrics: dict[str, float],
    model_metrics: dict[str, dict[str, float]],
    validation_rows: int,
    prediction_rows: int,
    label_classes: np.ndarray,
) -> None:
    REPORT_OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_OUTPUT_TXT.open("w", encoding="utf-8") as file:
        file.write("Multi-label JEL ensemble model\n")
        file.write("==============================\n\n")
        file.write("Model: label-wise majority vote ensemble with confidence fallback\n")
        file.write(f"Label column: {LABEL_COLUMN}\n")
        file.write(f"Minimum model votes per label: {MIN_MODEL_VOTES}\n")
        file.write(f"Labels: {list(label_classes)}\n")
        file.write(f"Validation rows: {validation_rows}\n")
        file.write(f"Prediction rows: {prediction_rows}\n\n")
        file.write("Input models:\n")
        for model_spec in MODEL_SPECS:
            model_name = model_spec["name"]
            metrics = model_metrics.get(model_name, {})
            file.write(f"  {model_name}:\n")
            file.write(f"    Micro F1: {metrics.get('micro_f1', np.nan):.4f}\n")
            file.write(f"    Macro F1: {metrics.get('macro_f1', np.nan):.4f}\n")
            file.write(f"    Micro precision: {metrics.get('micro_precision', np.nan):.4f}\n")
            file.write(f"    Micro recall: {metrics.get('micro_recall', np.nan):.4f}\n")
            file.write(f"    Hamming loss: {metrics.get('hamming_loss', np.nan):.4f}\n")
            file.write(f"    Subset accuracy: {metrics.get('subset_accuracy', np.nan):.4f}\n")
        file.write("\n")
        file.write("Ensemble validation metrics:\n")
        file.write(f"  Micro F1: {ensemble_metrics['micro_f1']:.4f}\n")
        file.write(f"  Macro F1: {ensemble_metrics['macro_f1']:.4f}\n")
        file.write(f"  Micro precision: {ensemble_metrics['micro_precision']:.4f}\n")
        file.write(f"  Macro precision: {ensemble_metrics['macro_precision']:.4f}\n")
        file.write(f"  Micro recall: {ensemble_metrics['micro_recall']:.4f}\n")
        file.write(f"  Macro recall: {ensemble_metrics['macro_recall']:.4f}\n")
        file.write(f"  Hamming loss: {ensemble_metrics['hamming_loss']:.4f}\n")
        file.write(f"  Subset accuracy: {ensemble_metrics['subset_accuracy']:.4f}\n")
        file.write(f"\nValidation prediction CSV: {VALIDATION_OUTPUT_CSV}\n")


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
