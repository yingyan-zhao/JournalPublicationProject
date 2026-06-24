from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

TRAINING_INPUT_CSV = Path("data/processed/JEL_Training_Data_With_JEL.csv")
PREDICTION_INPUT_CSV = Path("data/processed/JEL_Training_Data_Without_JEL.csv")
PREDICTION_OUTPUT_CSV = Path("data/processed/JEL_Training_Data_Without_JEL_Predicted_SciBERT.csv")
COMBINED_OUTPUT_CSV = Path("data/processed/JEL_Training_Data_With_Observed_And_Predicted_SciBERT.csv")
VALIDATION_OUTPUT_CSV = Path("data/processed/JEL_Codes_1_SciBERT_Validation_Predictions.csv")
REPORT_OUTPUT_TXT = Path("data/processed/JEL_Codes_1_SciBERT_Report.txt")
MODEL_OUTPUT_DIR = Path("data/processed/JEL_Codes_1_SciBERT_Model")

MODEL_NAME = "allenai/scibert_scivocab_uncased"
TEXT_COLUMNS = [
    "title",
    "abstract",
    "openalex_keywords",
    "openalex_top3_keywords",
    "openalex_concepts",
    "openalex_top3_concepts",
    "openalex_level0_concepts",
    "scrape_keywords",
    "nber_keywords",
    "repec_keywords",
    "aea_keywords",
]
LABEL_COLUMN = "jel_codes_1"
PREDICTED_LABEL_COLUMN = "jel_codes_1_predicted"
PREDICTED_CONFIDENCE_COLUMN = "jel_codes_1_predicted_confidence"
JEL_SOURCE_COLUMN = "jel_codes_1_source"
RANDOM_STATE = 2026
MAX_LENGTH = 512


def main() -> None:
    transformers, torch = import_finetuning_dependencies()

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
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_encoder.classes_),
        id2label={index: label for index, label in enumerate(label_encoder.classes_)},
        label2id={label: index for index, label in enumerate(label_encoder.classes_)},
    )

    train_dataset = TextClassificationDataset(train_text, train_labels, tokenizer, torch)
    test_dataset = TextClassificationDataset(test_text, test_labels, tokenizer, torch)

    training_args = transformers.TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        seed=RANDOM_STATE,
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=lambda output: compute_metrics(output, label_encoder.classes_),
    )
    trainer.train()

    test_output = trainer.predict(test_dataset)
    test_probabilities = softmax(test_output.predictions)
    test_predicted_ids = test_probabilities.argmax(axis=1)
    test_predicted_labels = label_encoder.inverse_transform(test_predicted_ids)
    test_true_labels = label_encoder.inverse_transform(test_labels)
    test_confidence = test_probabilities.max(axis=1)
    validation_accuracy = accuracy_score(test_true_labels, test_predicted_labels)
    validation_report = classification_report(
        test_true_labels,
        test_predicted_labels,
        zero_division=0,
    )

    prediction_text = combine_text_columns(prediction_data, TEXT_COLUMNS).tolist()
    prediction_dataset = TextOnlyDataset(prediction_text, tokenizer, torch)
    prediction_output = trainer.predict(prediction_dataset)
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
        true_labels=test_true_labels,
        predicted_labels=test_predicted_labels,
        confidence=test_confidence,
    )
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    trainer.save_model(MODEL_OUTPUT_DIR)
    write_report(
        validation_accuracy=validation_accuracy,
        validation_report=validation_report,
        label_counts=labels.value_counts().sort_index(),
        label_classes=label_encoder.classes_,
    )

    print("SciBERT single-label fine-tuning summary:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Label column: {LABEL_COLUMN}")
    print(f"  Training rows: {len(training_data)}")
    print(f"  Rows predicted: {len(predicted_data)}")
    print(f"  Validation accuracy: {validation_accuracy:.4f}")
    print(f"  Prediction output CSV: {PREDICTION_OUTPUT_CSV}")
    print(f"  Combined output CSV: {COMBINED_OUTPUT_CSV}")
    print(f"  Validation prediction CSV: {VALIDATION_OUTPUT_CSV}")
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
    missing_columns = [column for column in columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing text columns: {missing_columns}")
    text_parts = data[columns].fillna("").astype(str)
    return text_parts.apply(
        lambda row: " ".join(value.strip() for value in row if value.strip()),
        axis=1,
    )


def write_report(
    validation_accuracy: float,
    validation_report: str,
    label_counts: pd.Series,
    label_classes: np.ndarray,
) -> None:
    REPORT_OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_OUTPUT_TXT.open("w", encoding="utf-8") as file:
        file.write("JEL first-letter prediction model\n")
        file.write("=================================\n\n")
        file.write("Model: Fine-tuned SciBERT single-label classifier\n")
        file.write(f"Base model: {MODEL_NAME}\n")
        file.write(f"Label column: {LABEL_COLUMN}\n")
        file.write(f"Labels: {list(label_classes)}\n")
        file.write(f"Text columns: {TEXT_COLUMNS}\n\n")
        file.write("Label counts:\n")
        file.write(label_counts.to_string())
        file.write("\n\n")
        file.write(f"Validation accuracy: {validation_accuracy:.4f}\n\n")
        file.write("Validation classification report:\n")
        file.write(validation_report)
        file.write("\n")


def write_validation_predictions(
    row_ids: pd.Series,
    true_labels,
    predicted_labels,
    confidence,
) -> None:
    validation = pd.DataFrame(
        {
            "validation_row_id": row_ids.astype(str).to_list(),
            "jel_codes_1_true": true_labels,
            "scibert_jel_codes_1_predicted": predicted_labels,
            "scibert_confidence": pd.Series(confidence).round(4),
        }
    )
    validation.to_csv(VALIDATION_OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()
