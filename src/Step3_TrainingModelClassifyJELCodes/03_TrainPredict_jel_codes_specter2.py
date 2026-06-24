from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

TRAINING_INPUT_CSV = Path("data/processed/JEL_Training_Data_With_JEL.csv")
PREDICTION_INPUT_CSV = Path("data/processed/JEL_Training_Data_Without_JEL.csv")
PREDICTION_OUTPUT_CSV = Path("data/processed/JEL_Training_Data_Without_JEL_Predicted_SPECTER2.csv")
COMBINED_OUTPUT_CSV = Path("data/processed/JEL_Training_Data_With_Observed_And_Predicted_SPECTER2.csv")
VALIDATION_OUTPUT_CSV = Path("data/processed/JEL_Codes_1_SPECTER2_Validation_Predictions.csv")
REPORT_OUTPUT_TXT = Path("data/processed/JEL_Codes_1_SPECTER2_LogisticRegression_Report.txt")
MODEL_OUTPUT = Path("data/processed/JEL_Codes_1_SPECTER2_LogisticRegression.joblib")

MODEL_NAME = os.environ.get("SPECTER2_MODEL_PATH", "allenai/specter2_base")
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
BATCH_SIZE = 16
MAX_LENGTH = 512


def main() -> None:
    transformers, torch = import_transformer_dependencies()

    training_data = keep_labeled_rows(read_data(TRAINING_INPUT_CSV), LABEL_COLUMN)
    training_data["validation_row_id"] = training_data.index.astype(str)
    prediction_data = read_data(PREDICTION_INPUT_CSV)

    hf_token = huggingface_token()
    print(f"  Using SPECTER2 model/path: {MODEL_NAME}")
    print(f"  Python sees Hugging Face token: {bool(hf_token)}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    embedding_model = transformers.AutoModel.from_pretrained(MODEL_NAME, token=hf_token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model.to(device)
    embedding_model.eval()

    training_text = combine_text_columns(training_data, TEXT_COLUMNS)
    prediction_text = combine_text_columns(prediction_data, TEXT_COLUMNS)

    training_embeddings = encode_texts(
        texts=training_text.tolist(),
        tokenizer=tokenizer,
        model=embedding_model,
        torch=torch,
        device=device,
        label="training",
    )
    labels = training_data[LABEL_COLUMN].astype(str).str.strip()
    row_ids = training_data["validation_row_id"]

    x_train, x_test, y_train, y_test, row_id_train, row_id_test = train_test_split(
        training_embeddings,
        labels,
        row_ids,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    classifier = build_classifier()
    classifier.fit(x_train, y_train)

    validation_predictions = classifier.predict(x_test)
    validation_confidence = classifier.predict_proba(x_test).max(axis=1)
    validation_accuracy = accuracy_score(y_test, validation_predictions)
    validation_report = classification_report(
        y_test,
        validation_predictions,
        zero_division=0,
    )

    prediction_embeddings = encode_texts(
        texts=prediction_text.tolist(),
        tokenizer=tokenizer,
        model=embedding_model,
        torch=torch,
        device=device,
        label="prediction",
    )
    predicted_labels = classifier.predict(prediction_embeddings)
    predicted_confidence = classifier.predict_proba(prediction_embeddings).max(axis=1)

    predicted_data = prediction_data.copy()
    predicted_data[PREDICTED_LABEL_COLUMN] = predicted_labels
    predicted_data[PREDICTED_CONFIDENCE_COLUMN] = predicted_confidence.round(4)
    predicted_data[JEL_SOURCE_COLUMN] = "predicted_specter2"

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
        true_labels=y_test,
        predicted_labels=validation_predictions,
        confidence=validation_confidence,
    )
    joblib.dump(
        {
            "classifier": classifier,
            "model_name": MODEL_NAME,
            "text_columns": TEXT_COLUMNS,
            "label_column": LABEL_COLUMN,
        },
        MODEL_OUTPUT,
    )
    write_report(validation_accuracy, validation_report, labels.value_counts().sort_index())

    print("SPECTER2 embedding + Logistic Regression summary:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Training rows: {len(training_data)}")
    print(f"  Rows predicted: {len(predicted_data)}")
    print(f"  Validation accuracy: {validation_accuracy:.4f}")
    print(f"  Prediction output CSV: {PREDICTION_OUTPUT_CSV}")
    print(f"  Combined output CSV: {COMBINED_OUTPUT_CSV}")
    print(f"  Validation prediction CSV: {VALIDATION_OUTPUT_CSV}")
    print(f"  Model report: {REPORT_OUTPUT_TXT}")


def import_transformer_dependencies():
    try:
        import torch
        import transformers
    except ImportError as error:
        raise ImportError(
            "This script needs torch and transformers. Install them first, for example: "
            "pip install torch transformers"
        ) from error
    return transformers, torch


def huggingface_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def encode_texts(texts: list[str], tokenizer, model, torch, device, label: str) -> np.ndarray:
    embeddings = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Encoding {len(texts)} {label} texts in {total_batches} batches...")

    for batch_number, start in enumerate(range(0, len(texts), BATCH_SIZE), start=1):
        if batch_number == 1 or batch_number % 10 == 0 or batch_number == total_batches:
            print(f"    {label}: batch {batch_number}/{total_batches}")

        batch_texts = texts[start : start + BATCH_SIZE]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
        batch_embeddings = output.last_hidden_state[:, 0, :].detach().cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)


def build_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


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


def write_report(validation_accuracy: float, validation_report: str, label_counts: pd.Series) -> None:
    REPORT_OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_OUTPUT_TXT.open("w", encoding="utf-8") as file:
        file.write("JEL code first-letter prediction model\n")
        file.write("======================================\n\n")
        file.write("Model: SPECTER2 embeddings + Logistic Regression\n")
        file.write(f"Embedding model: {MODEL_NAME}\n")
        file.write(f"Label column: {LABEL_COLUMN}\n")
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
    true_labels: pd.Series,
    predicted_labels,
    confidence,
) -> None:
    validation = pd.DataFrame(
        {
            "validation_row_id": row_ids.astype(str).to_list(),
            "jel_codes_1_true": true_labels.astype(str).to_list(),
            "specter2_jel_codes_1_predicted": predicted_labels,
            "specter2_confidence": pd.Series(confidence).round(4),
        }
    )
    validation.to_csv(VALIDATION_OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()
