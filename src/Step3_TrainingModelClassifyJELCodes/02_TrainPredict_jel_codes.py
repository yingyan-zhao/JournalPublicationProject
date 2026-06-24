from pathlib import Path
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

TRAINING_INPUT_CSV = Path("data/processed/JEL_Training_Data_With_JEL.csv")
PREDICTION_INPUT_CSV = Path("data/processed/JEL_Training_Data_Without_JEL.csv")
PREDICTION_OUTPUT_CSV = Path("data/processed/JEL_Training_Data_Without_JEL_Predicted.csv")
COMBINED_OUTPUT_CSV = Path("data/processed/JEL_Training_Data_With_Observed_And_Predicted.csv")
VALIDATION_OUTPUT_CSV = Path("data/processed/JEL_Codes_1_TFIDF_Validation_Predictions.csv")
REPORT_OUTPUT_TXT = Path("data/processed/JEL_Codes_1_Model_Report.txt")
MODEL_OUTPUT = Path("data/processed/JEL_Codes_1_TFIDF_LogisticRegression.joblib")

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


def main() -> None:
    training_data = read_data(TRAINING_INPUT_CSV)
    prediction_data = read_data(PREDICTION_INPUT_CSV)

    training_data = keep_labeled_rows(training_data, LABEL_COLUMN)
    training_data["validation_row_id"] = training_data.index.astype(str)
    train_text = combine_text_columns(training_data, TEXT_COLUMNS)
    labels = training_data[LABEL_COLUMN].astype(str).str.strip()
    row_ids = training_data["validation_row_id"]

    x_train, x_test, y_train, y_test, row_id_train, row_id_test = train_test_split(
        train_text,
        labels,
        row_ids,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    model = build_model()
    model.fit(x_train, y_train)

    validation_predictions = model.predict(x_test)
    validation_confidence = model.predict_proba(x_test).max(axis=1)
    validation_accuracy = accuracy_score(y_test, validation_predictions)
    validation_report = classification_report(
        y_test,
        validation_predictions,
        zero_division=0,
    )

    prediction_text = combine_text_columns(prediction_data, TEXT_COLUMNS)
    predicted_labels = model.predict(prediction_text)
    predicted_confidence = model.predict_proba(prediction_text).max(axis=1)

    predicted_data = prediction_data.copy()
    predicted_data[PREDICTED_LABEL_COLUMN] = predicted_labels
    predicted_data[PREDICTED_CONFIDENCE_COLUMN] = predicted_confidence.round(4)
    predicted_data[JEL_SOURCE_COLUMN] = "predicted"

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
    joblib.dump(model, MODEL_OUTPUT)
    write_report(
        validation_accuracy=validation_accuracy,
        validation_report=validation_report,
        label_counts=labels.value_counts().sort_index(),
    )

    print("JEL code first-letter prediction summary:")
    print(f"  Training input CSV: {TRAINING_INPUT_CSV}")
    print(f"  Prediction input CSV: {PREDICTION_INPUT_CSV}")
    print(f"  Training rows with labels: {len(training_data)}")
    print(f"  Rows predicted: {len(predicted_data)}")
    print(f"  Validation accuracy: {validation_accuracy:.4f}")
    print(f"  Prediction output CSV: {PREDICTION_OUTPUT_CSV}")
    print(f"  Combined output CSV: {COMBINED_OUTPUT_CSV}")
    print(f"  Validation prediction CSV: {VALIDATION_OUTPUT_CSV}")
    print(f"  Model report: {REPORT_OUTPUT_TXT}")
    print(f"  Saved model: {MODEL_OUTPUT}")


def read_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    return pd.read_csv(path, dtype=str, keep_default_na=False)


def keep_labeled_rows(data: pd.DataFrame, label_column: str) -> pd.DataFrame:
    labels = data[label_column].fillna("").astype(str).str.strip()
    return data.loc[labels != ""].copy()


def combine_text_columns(data: pd.DataFrame, columns: list[str]) -> pd.Series:
    missing_columns = [
        column for column in columns
        if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing text columns: {missing_columns}")

    text_parts = data[columns].fillna("").astype(str)
    return text_parts.apply(
        lambda row: " ".join(value.strip() for value in row if value.strip()),
        axis=1,
    )


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    max_features=50000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def write_report(
    validation_accuracy: float,
    validation_report: str,
    label_counts: pd.Series,
) -> None:
    REPORT_OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_OUTPUT_TXT.open("w", encoding="utf-8") as file:
        file.write("JEL code first-letter prediction model\n")
        file.write("======================================\n\n")
        file.write("Model: TF-IDF text features + Logistic Regression\n")
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
            "tfidf_jel_codes_1_predicted": predicted_labels,
            "tfidf_confidence": pd.Series(confidence).round(4),
        }
    )
    validation.to_csv(VALIDATION_OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()
