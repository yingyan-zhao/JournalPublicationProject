from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

TRAINING_INPUT_CSV = Path("data/trainingmodel/JEL_Training_Data_With_JEL.csv")
PREDICTION_INPUT_CSV = Path("data/trainingmodel/JEL_Training_Data_Without_JEL.csv")
PREDICTION_OUTPUT_CSV = Path("data/trainingmodel/JEL_Training_Data_Without_JEL_Predicted_SciBERT.csv")
COMBINED_OUTPUT_CSV = Path("data/trainingmodel/JEL_Training_Data_With_Observed_And_Predicted_SciBERT.csv")
VALIDATION_OUTPUT_CSV = Path("data/trainingmodel/JEL_Codes_1_SciBERT_Validation_Predictions.csv")
CONFUSION_MATRIX_OUTPUT_CSV = Path("data/trainingmodel/JEL_Codes_1_SciBERT_Confusion_Matrix.csv")
TUNING_RESULTS_OUTPUT_CSV = Path("data/trainingmodel/JEL_Codes_1_SciBERT_Tuning_Results.csv")
REPORT_OUTPUT_TXT = Path("data/trainingmodel/JEL_Codes_1_SciBERT_Report.txt")
MODEL_OUTPUT_DIR = Path("data/trainingmodel/JEL_Codes_1_SciBERT_Model")

MODEL_NAME = "allenai/scibert_scivocab_uncased"
TEXT_COLUMNS = [
    "title",
    "keywords",
    "abstract",
]
LABEL_COLUMN = "jel_code_1"
PREDICTED_LABEL_COLUMN = "jel_code_1_predicted_scibert"
PREDICTED_CONFIDENCE_COLUMN = "jel_code_1_predicted_scibert_confidence"
JEL_SOURCE_COLUMN = "jel_code_1_scibert_source"
RANDOM_STATE = 2026
MAX_LENGTH = 512
PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 8
TUNING_GRID = [
    {
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
    }
    for learning_rate in [2e-5, 3e-5, 5e-5]
    for num_train_epochs in [2, 3, 4]
    for weight_decay in [0, 0.05]
    for warmup_ratio in [0]
]


def main() -> None:
    transformers, torch = import_finetuning_dependencies()
    print("MPS available:", torch.backends.mps.is_available())
    print("MPS built:", torch.backends.mps.is_built())

    training_data = keep_labeled_rows(read_data(TRAINING_INPUT_CSV), LABEL_COLUMN)
    training_data["validation_row_id"] = training_data.index.astype(str)
    prediction_data = read_data(PREDICTION_INPUT_CSV)

    text = combine_text_columns(training_data, TEXT_COLUMNS)
    labels = training_data[LABEL_COLUMN].astype(str).str.strip()
    row_ids = training_data["validation_row_id"]

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    train_text, test_text, train_labels, test_labels, row_id_train, row_id_test = train_test_split(
        text.tolist(),
        encoded_labels,
        row_ids,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=encoded_labels,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = TextClassificationDataset(train_text, train_labels, tokenizer, torch)
    test_dataset = TextClassificationDataset(test_text, test_labels, tokenizer, torch)

    best_trainer, best_params, tuning_results = tune_scibert(
        transformers=transformers,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        label_classes=label_encoder.classes_,
    )
    tuning_results = sorted_tuning_results(pd.DataFrame(tuning_results))
    print_tuning_results(tuning_results)
    write_tuning_results(tuning_results)

    validation_output = best_trainer.predict(test_dataset)
    validation_probabilities = softmax(validation_output.predictions)
    validation_ids = validation_probabilities.argmax(axis=1)
    validation_predictions = label_encoder.inverse_transform(validation_ids)
    validation_confidence = validation_probabilities.max(axis=1)
    validation_true_labels = label_encoder.inverse_transform(test_labels)
    validation_accuracy = accuracy_score(validation_true_labels, validation_predictions)
    validation_jel_code_full = training_data.loc[row_id_test.index, "jel_code_full"]
    any_code_match = predicted_label_in_jel_code_full(
        jel_code_full=validation_jel_code_full,
        predicted_labels=validation_predictions,
    )
    any_code_accuracy = float(any_code_match.mean())
    validation_report = classification_report(
        validation_true_labels,
        validation_predictions,
        zero_division=0,
    )

    full_training_dataset = TextClassificationDataset(text.tolist(), encoded_labels, tokenizer, torch)
    final_trainer = train_final_scibert(
        transformers=transformers,
        train_dataset=full_training_dataset,
        label_classes=label_encoder.classes_,
        best_params=best_params,
    )

    prediction_text = combine_text_columns(prediction_data, TEXT_COLUMNS).tolist()
    prediction_dataset = TextOnlyDataset(prediction_text, tokenizer, torch)
    prediction_output = final_trainer.predict(prediction_dataset)
    prediction_probabilities = softmax(prediction_output.predictions)
    prediction_ids = prediction_probabilities.argmax(axis=1)
    predicted_labels = label_encoder.inverse_transform(prediction_ids)
    predicted_confidence = prediction_probabilities.max(axis=1)

    predicted_data = prediction_data.copy()
    predicted_data[PREDICTED_LABEL_COLUMN] = predicted_labels
    predicted_data[PREDICTED_CONFIDENCE_COLUMN] = predicted_confidence.round(4)
    predicted_data[JEL_SOURCE_COLUMN] = "predicted_scibert"

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
        true_labels=validation_true_labels,
        jel_code_full=validation_jel_code_full,
        predicted_labels=validation_predictions,
        confidence=validation_confidence,
        any_code_match=any_code_match,
    )
    write_confusion_matrix(validation_true_labels, validation_predictions)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    final_trainer.save_model(MODEL_OUTPUT_DIR)
    write_report(
        validation_accuracy=validation_accuracy,
        any_code_accuracy=any_code_accuracy,
        validation_report=validation_report,
        label_counts=labels.value_counts().sort_index(),
        label_classes=label_encoder.classes_,
        best_params=best_params,
        tuning_results=tuning_results,
        validation_rows=len(test_labels),
    )

    print("SciBERT single-label fine-tuning summary:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Label column: {LABEL_COLUMN}")
    print(f"  Training rows: {len(training_data)}")
    print(f"  Rows predicted: {len(predicted_data)}")
    print(f"  Tuning runs: {len(tuning_results)}")
    print(f"  Best tuning parameters: {best_params}")
    print(f"  Validation rows: {len(test_labels)}")
    print(f"  Validation split accuracy: {validation_accuracy:.4f}")
    print(f"  Validation split any-code accuracy: {any_code_accuracy:.4f}")
    print(f"  Prediction output CSV: {PREDICTION_OUTPUT_CSV}")
    print(f"  Combined output CSV: {COMBINED_OUTPUT_CSV}")
    print(f"  Validation prediction CSV: {VALIDATION_OUTPUT_CSV}")
    print(f"  Confusion matrix CSV: {CONFUSION_MATRIX_OUTPUT_CSV}")
    print(f"  Tuning results CSV: {TUNING_RESULTS_OUTPUT_CSV}")
    print(f"  Model report: {REPORT_OUTPUT_TXT}")
    print(f"  Saved model directory: {MODEL_OUTPUT_DIR}")


class TextClassificationDataset:
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
        item["labels"] = self.torch.tensor(self.labels[index], dtype=self.torch.long)
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


def compute_metrics(eval_output, label_names: np.ndarray) -> dict[str, float]:
    logits, labels = eval_output
    probabilities = softmax(logits)
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
    }


def tune_scibert(
    transformers,
    train_dataset: TextClassificationDataset,
    test_dataset: TextClassificationDataset,
    label_classes: np.ndarray,
):
    best_trainer = None
    best_params = None
    best_accuracy = -1.0
    tuning_results = []

    for run_number, params in enumerate(TUNING_GRID, start=1):
        print(f"  SciBERT tuning run {run_number}/{len(TUNING_GRID)}: {params}")
        run_output_dir = MODEL_OUTPUT_DIR / f"run_{run_number:03d}"
        model = build_model(transformers, label_classes)
        training_args = build_training_args(transformers, run_output_dir, params)
        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=lambda output: compute_metrics(output, label_classes),
        )
        trainer.train()
        metrics = trainer.evaluate()
        accuracy = float(metrics.get("eval_accuracy", 0.0))
        tuning_results.append({**params, "accuracy": accuracy})
        print(f"    Validation accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            best_trainer = trainer

    if best_trainer is None or best_params is None:
        raise ValueError("No SciBERT tuning run completed.")
    return best_trainer, best_params, tuning_results


def train_final_scibert(
    transformers,
    train_dataset: TextClassificationDataset,
    label_classes: np.ndarray,
    best_params: dict,
):
    print("  Training final SciBERT model on all labeled observations.")
    model = build_model(transformers, label_classes)
    training_args = build_final_training_args(
        transformers,
        MODEL_OUTPUT_DIR / "final_model",
        best_params,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()
    return trainer


def sorted_tuning_results(tuning_results: pd.DataFrame) -> pd.DataFrame:
    if tuning_results.empty:
        return tuning_results
    return tuning_results.sort_values(
        ["accuracy", "learning_rate", "num_train_epochs", "weight_decay", "warmup_ratio"],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    )


def print_tuning_results(tuning_results: pd.DataFrame) -> None:
    print("SciBERT tuning accuracy by parameter combination:")
    if tuning_results.empty:
        print("  No tuning results.")
        return
    display = tuning_results.copy()
    display.insert(0, "rank", range(1, len(display) + 1))
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 160):
        print(display.to_string(index=False))


def write_tuning_results(tuning_results: pd.DataFrame) -> None:
    TUNING_RESULTS_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    tuning_results.to_csv(TUNING_RESULTS_OUTPUT_CSV, index=False)


def build_model(transformers, label_classes: np.ndarray):
    return transformers.AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_classes),
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
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        seed=RANDOM_STATE,
    )


def build_final_training_args(transformers, output_dir: Path, params: dict):
    return transformers.TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="no",
        save_strategy="epoch",
        learning_rate=params["learning_rate"],
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        num_train_epochs=params["num_train_epochs"],
        weight_decay=params["weight_decay"],
        warmup_ratio=params["warmup_ratio"],
        load_best_model_at_end=False,
        logging_steps=50,
        seed=RANDOM_STATE,
    )


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


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


def predicted_label_in_jel_code_full(
    jel_code_full: pd.Series,
    predicted_labels,
) -> pd.Series:
    predicted = pd.Series(predicted_labels, index=jel_code_full.index).fillna("").astype(str).str.strip()
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


def write_confusion_matrix(
    true_labels,
    predicted_labels,
) -> None:
    labels_order = sorted(pd.Series(true_labels).unique())
    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels_order)
    confusion = pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in labels_order],
        columns=[f"pred_{label}" for label in labels_order],
    )
    CONFUSION_MATRIX_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    confusion.to_csv(CONFUSION_MATRIX_OUTPUT_CSV)


def write_report(
    validation_accuracy: float,
    any_code_accuracy: float,
    validation_report: str,
    label_counts: pd.Series,
    label_classes: np.ndarray,
    best_params: dict,
    tuning_results: pd.DataFrame,
    validation_rows: int,
) -> None:
    REPORT_OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_OUTPUT_TXT.open("w", encoding="utf-8") as file:
        file.write("JEL first-letter prediction model\n")
        file.write("=================================\n\n")
        file.write("Model: Fine-tuned SciBERT single-label classifier\n")
        file.write(f"Base model: {MODEL_NAME}\n")
        file.write(f"Label column: {LABEL_COLUMN}\n")
        file.write(f"Labels: {list(label_classes)}\n")
        file.write(f"Text columns: {TEXT_COLUMNS}\n")
        file.write(f"Max length: {MAX_LENGTH}\n")
        file.write(f"Train batch size: {PER_DEVICE_TRAIN_BATCH_SIZE}\n")
        file.write(f"Eval batch size: {PER_DEVICE_EVAL_BATCH_SIZE}\n\n")
        file.write("Tuning grid:\n")
        file.write(f"  Runs: {len(TUNING_GRID)}\n")
        file.write(f"  Best parameters: {best_params}\n")
        file.write(f"  Tuning results CSV: {TUNING_RESULTS_OUTPUT_CSV}\n")
        file.write("  Results:\n")
        if tuning_results.empty:
            file.write("    No tuning results.\n")
        else:
            display = tuning_results.copy()
            display.insert(0, "rank", range(1, len(display) + 1))
            file.write(display.to_string(index=False))
            file.write("\n")
        file.write("\n")
        file.write("Model selection:\n")
        file.write("  Used one train/validation split for tuning.\n")
        file.write("  After choosing best parameters, trained one final model on all labeled observations.\n\n")
        file.write("Label counts:\n")
        file.write(label_counts.to_string())
        file.write("\n\n")
        file.write(f"Validation rows: {validation_rows}\n")
        file.write(f"Validation split accuracy: {validation_accuracy:.4f}\n")
        file.write(f"Validation split any-code accuracy: {any_code_accuracy:.4f}\n")
        file.write(f"Confusion matrix CSV: {CONFUSION_MATRIX_OUTPUT_CSV}\n")
        file.write("\n")
        file.write("Validation split classification report:\n")
        file.write(validation_report)
        file.write("\n")


def write_validation_predictions(
    row_ids: pd.Series,
    true_labels,
    jel_code_full: pd.Series,
    predicted_labels,
    confidence,
    any_code_match: pd.Series,
) -> None:
    validation = pd.DataFrame(
        {
            "validation_row_id": row_ids.astype(str).to_list(),
            "jel_code_1_true": true_labels,
            "jel_code_full": jel_code_full.astype(str).to_list(),
            "scibert_jel_code_1_predicted": predicted_labels,
            "scibert_confidence": pd.Series(confidence).round(4),
            "scibert_prediction_in_jel_code_full": any_code_match.astype(int).to_list(),
        }
    )
    validation.to_csv(VALIDATION_OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()
