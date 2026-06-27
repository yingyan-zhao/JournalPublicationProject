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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

TRAINING_INPUT_CSV = Path("data/processed/JEL_Training_Data_With_JEL.csv")
PREDICTION_INPUT_CSV = Path("data/processed/JEL_Training_Data_Without_JEL.csv")

PREDICTION_OUTPUT_CSV = Path("data/processed/JEL_Training_Data_Without_JEL_Predicted_Multi_SciBERT.csv")
COMBINED_OUTPUT_CSV = Path("data/processed/JEL_Training_Data_With_Observed_And_Predicted_Multi_SciBERT.csv")
VALIDATION_OUTPUT_CSV = Path("data/processed/JEL_Codes_Multi_SciBERT_Validation_Predictions.csv")
TUNING_RESULTS_OUTPUT_CSV = Path("data/processed/JEL_Codes_Multi_SciBERT_Tuning_Results.csv")
THRESHOLD_TUNING_OUTPUT_CSV = Path("data/processed/JEL_Codes_Multi_SciBERT_Threshold_Tuning_Results.csv")
REPORT_OUTPUT_TXT = Path("data/processed/JEL_Codes_Multi_SciBERT_Report.txt")
MODEL_OUTPUT_DIR = Path("data/processed/JEL_Codes_Multi_SciBERT_Model")

MODEL_NAME = "allenai/scibert_scivocab_uncased"
TEXT_COLUMNS = ["title", "keywords", "abstract"]
LABEL_COLUMN = "jel_code_full"
PREDICTED_LABEL_COLUMN = "jel_code_full_predicted_scibert"
PREDICTED_CONFIDENCE_COLUMN = "jel_code_full_predicted_scibert_max_confidence"
JEL_SOURCE_COLUMN = "jel_code_full_scibert_source"

RANDOM_STATE = 2026
TEST_SIZE = 0.2
MAX_LENGTH = 512
PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 8
TUNING_THRESHOLD = 0.5
THRESHOLD_OPTIONS = [0.2, 0.3, 0.4, 0.5, 0.6]

TUNING_GRID = [
    {
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
    }
    # Restore these full grids when you are ready for a long run:
    # for learning_rate in [1e-5, 2e-5, 3e-5]
    # for num_train_epochs in [2, 3]
    # for weight_decay in [0, 0.01]
    # for warmup_ratio in [0, 0.1]
    for learning_rate in [1e-5]
    for num_train_epochs in [2]
    for weight_decay in [0.01]
    for warmup_ratio in [0.1]
]


def main() -> None:
    transformers, torch = import_finetuning_dependencies()
    print("MPS available:", torch.backends.mps.is_available())
    print("MPS built:", torch.backends.mps.is_built())

    training_data = keep_labeled_rows(read_data(TRAINING_INPUT_CSV), LABEL_COLUMN)
    prediction_data = read_data(PREDICTION_INPUT_CSV)

    training_data["validation_row_id"] = training_data.index.astype(str)
    text = combine_text_columns(training_data, TEXT_COLUMNS)
    prediction_text = combine_text_columns(prediction_data, TEXT_COLUMNS).tolist()
    label_lists = training_data[LABEL_COLUMN].apply(split_label_cell)

    label_binarizer = MultiLabelBinarizer()
    labels = label_binarizer.fit_transform(label_lists).astype(float)

    train_text, test_text, train_labels, test_labels, row_id_train, row_id_test = train_test_split(
        text.tolist(),
        labels,
        training_data["validation_row_id"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=training_data["jel_code_1"],
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = MultiLabelTextDataset(train_text, train_labels, tokenizer, torch)
    test_dataset = MultiLabelTextDataset(test_text, test_labels, tokenizer, torch)

    best_trainer, best_params, tuning_results = tune_scibert(
        transformers=transformers,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        label_classes=label_binarizer.classes_,
    )
    tuning_results = sort_tuning_results(pd.DataFrame(tuning_results))
    print_tuning_results(tuning_results)
    write_tuning_results(tuning_results)

    validation_output = best_trainer.predict(test_dataset)
    validation_probabilities = sigmoid(validation_output.predictions)
    threshold_results = tune_thresholds(test_labels, validation_probabilities, THRESHOLD_OPTIONS)
    print_threshold_results(threshold_results)
    write_threshold_results(threshold_results)
    best_threshold = float(threshold_results.iloc[0]["threshold"])

    validation_predictions = probabilities_to_multilabel(validation_probabilities, threshold=best_threshold)
    metrics = multilabel_metrics(test_labels, validation_predictions)

    prediction_dataset = TextOnlyDataset(prediction_text, tokenizer, torch)
    prediction_output = best_trainer.predict(prediction_dataset)
    prediction_probabilities = sigmoid(prediction_output.predictions)
    predicted_matrix = probabilities_to_multilabel(prediction_probabilities, threshold=best_threshold)
    predicted_labels = format_multilabel_predictions(label_binarizer, predicted_matrix)
    predicted_confidence = prediction_probabilities.max(axis=1)

    predicted_data = prediction_data.copy()
    predicted_data[PREDICTED_LABEL_COLUMN] = predicted_labels
    predicted_data[PREDICTED_CONFIDENCE_COLUMN] = predicted_confidence.round(4)
    predicted_data[JEL_SOURCE_COLUMN] = "predicted_multi_scibert"

    observed_data = training_data.copy()
    observed_data[PREDICTED_LABEL_COLUMN] = observed_data[LABEL_COLUMN]
    observed_data[PREDICTED_CONFIDENCE_COLUMN] = ""
    observed_data[JEL_SOURCE_COLUMN] = "observed"
    combined = pd.concat([observed_data, predicted_data], ignore_index=True, sort=False)

    PREDICTION_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    predicted_data.to_csv(PREDICTION_OUTPUT_CSV, index=False)
    combined.to_csv(COMBINED_OUTPUT_CSV, index=False)
    write_validation_predictions(
        row_ids=row_id_test,
        true_matrix=test_labels,
        predicted_matrix=validation_predictions,
        probabilities=validation_probabilities,
        label_binarizer=label_binarizer,
    )
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    best_trainer.save_model(MODEL_OUTPUT_DIR)
    write_report(
        metrics=metrics,
        label_counts=label_counts(label_lists),
        label_classes=label_binarizer.classes_,
        training_rows=len(training_data),
        prediction_rows=len(predicted_data),
        best_params=best_params,
        tuning_results=tuning_results,
        threshold=best_threshold,
        threshold_results=threshold_results,
    )

    print("Multi-label SciBERT prediction summary:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Training rows with labels: {len(training_data)}")
    print(f"  Rows predicted: {len(predicted_data)}")
    print(f"  Tuning runs: {len(tuning_results)}")
    print(f"  Best tuning parameters: {best_params}")
    print(f"  Best prediction threshold: {best_threshold}")
    print(f"  Micro F1: {metrics['micro_f1']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Precision: {metrics['micro_precision']:.4f}")
    print(f"  Recall: {metrics['micro_recall']:.4f}")
    print(f"  Hamming loss: {metrics['hamming_loss']:.4f}")
    print(f"  Subset accuracy: {metrics['subset_accuracy']:.4f}")
    print(f"  Prediction output CSV: {PREDICTION_OUTPUT_CSV}")
    print(f"  Combined output CSV: {COMBINED_OUTPUT_CSV}")
    print(f"  Validation prediction CSV: {VALIDATION_OUTPUT_CSV}")
    print(f"  Tuning results CSV: {TUNING_RESULTS_OUTPUT_CSV}")
    print(f"  Threshold tuning results CSV: {THRESHOLD_TUNING_OUTPUT_CSV}")
    print(f"  Model report: {REPORT_OUTPUT_TXT}")
    print(f"  Saved model directory: {MODEL_OUTPUT_DIR}")


class MultiLabelTextDataset:
    def __init__(self, texts, labels, tokenizer, torch):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.torch = torch

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        item = {key: self.torch.tensor(value) for key, value in encoded.items()}
        item["labels"] = self.torch.tensor(self.labels[index], dtype=self.torch.float)
        return item


class TextOnlyDataset:
    def __init__(self, texts, tokenizer, torch):
        self.texts = texts
        self.tokenizer = tokenizer
        self.torch = torch

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        return {key: self.torch.tensor(value) for key, value in encoded.items()}


def import_finetuning_dependencies():
    try:
        import accelerate  # noqa: F401
        import torch
        import transformers
    except ImportError as error:
        raise ImportError(
            "This script needs torch, transformers, and accelerate. Install them first, for example: "
            "pip install torch transformers accelerate"
        ) from error
    return transformers, torch


def compute_metrics(eval_output) -> dict[str, float]:
    logits = eval_output.predictions if hasattr(eval_output, "predictions") else eval_output[0]
    labels = eval_output.label_ids if hasattr(eval_output, "label_ids") else eval_output[1]
    probabilities = sigmoid(logits)
    predictions = probabilities_to_multilabel(probabilities, threshold=TUNING_THRESHOLD)
    return multilabel_metrics(labels, predictions)


def tune_scibert(
    transformers,
    train_dataset: MultiLabelTextDataset,
    test_dataset: MultiLabelTextDataset,
    label_classes: np.ndarray,
):
    best_trainer = None
    best_params = None
    best_result = None
    tuning_results = []

    for run_number, params in enumerate(TUNING_GRID, start=1):
        print(f"  SciBERT multi-label tuning run {run_number}/{len(TUNING_GRID)}: {params}")
        run_output_dir = MODEL_OUTPUT_DIR / f"run_{run_number:03d}"
        model = build_model(transformers, label_classes)
        training_args = build_training_args(transformers, run_output_dir, params)
        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        metrics = trainer.evaluate()
        result = {
            **params,
            "micro_f1": float(metrics.get("eval_micro_f1", 0.0)),
            "macro_f1": float(metrics.get("eval_macro_f1", 0.0)),
            "micro_precision": float(metrics.get("eval_micro_precision", 0.0)),
            "micro_recall": float(metrics.get("eval_micro_recall", 0.0)),
            "hamming_loss": float(metrics.get("eval_hamming_loss", 1.0)),
            "subset_accuracy": float(metrics.get("eval_subset_accuracy", 0.0)),
        }
        tuning_results.append(result)
        print(
            "    "
            f"micro_f1={result['micro_f1']:.4f}, "
            f"macro_f1={result['macro_f1']:.4f}, "
            f"precision={result['micro_precision']:.4f}, "
            f"recall={result['micro_recall']:.4f}, "
            f"hamming_loss={result['hamming_loss']:.4f}, "
            f"subset_accuracy={result['subset_accuracy']:.4f}"
        )

        if best_result is None or compare_metric_rows(result, best_result):
            best_result = result
            best_params = params
            best_trainer = trainer

    if best_trainer is None or best_params is None:
        raise ValueError("No SciBERT tuning run completed.")
    return best_trainer, best_params, tuning_results


def compare_metric_rows(candidate: dict, incumbent: dict) -> bool:
    candidate_key = (
        candidate["micro_f1"],
        candidate["macro_f1"],
        candidate["micro_precision"],
        candidate["micro_recall"],
        -candidate["hamming_loss"],
        candidate["subset_accuracy"],
    )
    incumbent_key = (
        incumbent["micro_f1"],
        incumbent["macro_f1"],
        incumbent["micro_precision"],
        incumbent["micro_recall"],
        -incumbent["hamming_loss"],
        incumbent["subset_accuracy"],
    )
    return candidate_key > incumbent_key


def build_model(transformers, label_classes: np.ndarray):
    return transformers.AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_classes),
        problem_type="multi_label_classification",
        id2label={index: label for index, label in enumerate(label_classes)},
        label2id={label: index for index, label in enumerate(label_classes)},
    )


def build_training_args(transformers, output_dir: Path, params: dict):
    return transformers.TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=params["learning_rate"],
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        num_train_epochs=params["num_train_epochs"],
        weight_decay=params["weight_decay"],
        warmup_ratio=params["warmup_ratio"],
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        logging_steps=50,
        seed=RANDOM_STATE,
    )


def read_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def keep_labeled_rows(data: pd.DataFrame, label_column: str) -> pd.DataFrame:
    labels = data[label_column].fillna("").astype(str).str.strip()
    return data.loc[labels != ""].copy()


def combine_text_columns(data: pd.DataFrame, columns: list[str]) -> pd.Series:
    text_parts = pd.DataFrame(index=data.index)
    for column in columns:
        if column in data.columns:
            text_parts[column] = data[column].fillna("").astype(str)
        else:
            text_parts[column] = ""
    return text_parts.apply(
        lambda row: " ".join(value.strip() for value in row if value.strip()),
        axis=1,
    )


def split_label_cell(value: object) -> list[str]:
    if pd.isna(value):
        return []
    labels = []
    seen = set()
    for label in str(value).split(";"):
        cleaned = label.strip()
        if cleaned and cleaned not in seen:
            labels.append(cleaned)
            seen.add(cleaned)
    return labels


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def probabilities_to_multilabel(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    predicted = (probabilities >= threshold).astype(int)
    empty_rows = predicted.sum(axis=1) == 0
    if empty_rows.any():
        top_labels = probabilities[empty_rows].argmax(axis=1)
        predicted[empty_rows, top_labels] = 1
    return predicted


def tune_thresholds(
    true_matrix: np.ndarray,
    probabilities: np.ndarray,
    thresholds: list[float],
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        predicted_matrix = probabilities_to_multilabel(probabilities, threshold=threshold)
        rows.append({"threshold": threshold, **multilabel_metrics(true_matrix, predicted_matrix)})
    return sort_threshold_results(pd.DataFrame(rows))


def sort_tuning_results(results: pd.DataFrame) -> pd.DataFrame:
    return results.sort_values(
        [
            "micro_f1",
            "macro_f1",
            "micro_precision",
            "micro_recall",
            "hamming_loss",
            "subset_accuracy",
        ],
        ascending=[False, False, False, False, True, False],
        kind="mergesort",
    )


def sort_threshold_results(results: pd.DataFrame) -> pd.DataFrame:
    return sort_tuning_results(results)


def print_tuning_results(results: pd.DataFrame) -> None:
    print()
    print("Multi-label SciBERT tuning metrics by parameter combination:")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print(results.to_string(index=False))
    print()


def write_tuning_results(results: pd.DataFrame) -> None:
    TUNING_RESULTS_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(TUNING_RESULTS_OUTPUT_CSV, index=False)


def print_threshold_results(results: pd.DataFrame) -> None:
    print()
    print("Multi-label SciBERT threshold tuning results:")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print(results.to_string(index=False))
    print()


def write_threshold_results(results: pd.DataFrame) -> None:
    THRESHOLD_TUNING_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(THRESHOLD_TUNING_OUTPUT_CSV, index=False)


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


def format_multilabel_predictions(
    label_binarizer: MultiLabelBinarizer,
    predicted_matrix: np.ndarray,
) -> list[str]:
    labels = label_binarizer.inverse_transform(predicted_matrix)
    return ["; ".join(label_set) for label_set in labels]


def label_counts(label_lists: pd.Series) -> pd.Series:
    counts = {}
    for labels in label_lists:
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
    return pd.Series(counts).sort_index()


def write_validation_predictions(
    row_ids: pd.Series,
    true_matrix: np.ndarray,
    predicted_matrix: np.ndarray,
    probabilities: np.ndarray,
    label_binarizer: MultiLabelBinarizer,
) -> None:
    validation = pd.DataFrame(
        {
            "validation_row_id": row_ids.astype(str).to_list(),
            "jel_code_full_true": format_multilabel_predictions(label_binarizer, true_matrix),
            "jel_code_full_predicted_scibert": format_multilabel_predictions(label_binarizer, predicted_matrix),
            "scibert_max_confidence": probabilities.max(axis=1).round(4),
            "exact_match": (true_matrix == predicted_matrix).all(axis=1).astype(int),
        }
    )
    for index, label in enumerate(label_binarizer.classes_):
        validation[f"true_{label}"] = true_matrix[:, index]
        validation[f"predicted_{label}"] = predicted_matrix[:, index]
        validation[f"probability_{label}"] = probabilities[:, index].round(4)

    validation.to_csv(VALIDATION_OUTPUT_CSV, index=False)


def write_report(
    metrics: dict[str, float],
    label_counts: pd.Series,
    label_classes: np.ndarray,
    training_rows: int,
    prediction_rows: int,
    best_params: dict,
    tuning_results: pd.DataFrame,
    threshold: float,
    threshold_results: pd.DataFrame,
) -> None:
    REPORT_OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_OUTPUT_TXT.open("w", encoding="utf-8") as file:
        file.write("Multi-label JEL SciBERT prediction model\n")
        file.write("========================================\n\n")
        file.write("Model: Fine-tuned SciBERT multi-label classifier\n")
        file.write(f"Base model: {MODEL_NAME}\n")
        file.write(f"Label column: {LABEL_COLUMN}\n")
        file.write(f"Labels: {list(label_classes)}\n")
        file.write(f"Text columns: {TEXT_COLUMNS}\n")
        file.write(f"Max length: {MAX_LENGTH}\n")
        file.write(f"Train batch size: {PER_DEVICE_TRAIN_BATCH_SIZE}\n")
        file.write(f"Eval batch size: {PER_DEVICE_EVAL_BATCH_SIZE}\n")
        file.write(f"Prediction threshold: {threshold}\n")
        file.write(f"Threshold options: {THRESHOLD_OPTIONS}\n")
        file.write(f"Training rows: {training_rows}\n")
        file.write(f"Prediction rows: {prediction_rows}\n\n")
        file.write("Tuning grid:\n")
        file.write(f"  Runs: {len(tuning_results)}\n")
        file.write(f"  Best parameters: {best_params}\n")
        file.write(f"  Tuning results CSV: {TUNING_RESULTS_OUTPUT_CSV}\n")
        file.write(tuning_results.to_string(index=False))
        file.write("\n\n")
        file.write("Threshold tuning:\n")
        file.write(f"  Best threshold: {threshold}\n")
        file.write(f"  Threshold tuning results CSV: {THRESHOLD_TUNING_OUTPUT_CSV}\n")
        file.write(threshold_results.to_string(index=False))
        file.write("\n\n")
        file.write("Label counts:\n")
        file.write(label_counts.to_string())
        file.write("\n\n")
        file.write("Validation metrics:\n")
        file.write(f"  Micro F1: {metrics['micro_f1']:.4f}\n")
        file.write(f"  Macro F1: {metrics['macro_f1']:.4f}\n")
        file.write(f"  Micro precision: {metrics['micro_precision']:.4f}\n")
        file.write(f"  Macro precision: {metrics['macro_precision']:.4f}\n")
        file.write(f"  Micro recall: {metrics['micro_recall']:.4f}\n")
        file.write(f"  Macro recall: {metrics['macro_recall']:.4f}\n")
        file.write(f"  Hamming loss: {metrics['hamming_loss']:.4f}\n")
        file.write(f"  Subset accuracy: {metrics['subset_accuracy']:.4f}\n")
        file.write(f"\nValidation prediction CSV: {VALIDATION_OUTPUT_CSV}\n")


if __name__ == "__main__":
    main()
