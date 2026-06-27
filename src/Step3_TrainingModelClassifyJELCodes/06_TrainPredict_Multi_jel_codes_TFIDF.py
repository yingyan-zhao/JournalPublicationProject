from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

TRAINING_INPUT_CSV = Path("data/processed/JEL_Training_Data_With_JEL.csv")
PREDICTION_INPUT_CSV = Path("data/processed/JEL_Training_Data_Without_JEL.csv")

PREDICTION_OUTPUT_CSV = Path("data/processed/JEL_Training_Data_Without_JEL_Predicted_Multi_TFIDF.csv")
COMBINED_OUTPUT_CSV = Path("data/processed/JEL_Training_Data_With_Observed_And_Predicted_Multi_TFIDF.csv")
VALIDATION_OUTPUT_CSV = Path("data/processed/JEL_Codes_Multi_TFIDF_Validation_Predictions.csv")
TUNING_RESULTS_OUTPUT_CSV = Path("data/processed/JEL_Codes_Multi_TFIDF_Tuning_Results.csv")
THRESHOLD_TUNING_OUTPUT_CSV = Path("data/processed/JEL_Codes_Multi_TFIDF_Threshold_Tuning_Results.csv")
REPORT_OUTPUT_TXT = Path("data/processed/JEL_Codes_Multi_TFIDF_Report.txt")
MODEL_OUTPUT = Path("data/processed/JEL_Codes_Multi_TFIDF_OneVsRest_LogisticRegression.joblib")

TEXT_COLUMNS = ["title", "keywords", "abstract"]
LABEL_COLUMN = "jel_code_full"
PREDICTED_LABEL_COLUMN = "jel_code_full_predicted_tfidf"
PREDICTED_CONFIDENCE_COLUMN = "jel_code_full_predicted_tfidf_max_confidence"
JEL_SOURCE_COLUMN = "jel_code_full_tfidf_source"

RANDOM_STATE = 2026
TEST_SIZE = 0.2
THRESHOLD_OPTIONS = [0.2, 0.3, 0.4, 0.5, 0.6]

PARAM_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "tfidf__min_df": [2, 3, 5],
    "tfidf__max_df": [0.9, 0.95, 1.0],
    "tfidf__max_features": [30000, 50000, 100000],
    "classifier__estimator__C": [0.3, 1, 3],
    "classifier__estimator__class_weight": ["balanced", None],
}

SCORING = {
    "micro_f1": make_scorer(f1_score, average="micro", zero_division=0),
    "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
    "micro_precision": make_scorer(precision_score, average="micro", zero_division=0),
    "micro_recall": make_scorer(recall_score, average="micro", zero_division=0),
    "hamming_loss": make_scorer(hamming_loss, greater_is_better=False),
    "subset_accuracy": make_scorer(accuracy_score),
}


def main() -> None:
    training_data = keep_labeled_rows(read_data(TRAINING_INPUT_CSV), LABEL_COLUMN)
    prediction_data = read_data(PREDICTION_INPUT_CSV)

    training_data["validation_row_id"] = training_data.index.astype(str)
    train_text = combine_text_columns(training_data, TEXT_COLUMNS)
    prediction_text = combine_text_columns(prediction_data, TEXT_COLUMNS)
    label_lists = training_data[LABEL_COLUMN].apply(split_label_cell)

    label_binarizer = MultiLabelBinarizer()
    labels = label_binarizer.fit_transform(label_lists)

    x_train, x_test, y_train, y_test, row_id_train, row_id_test = train_test_split(
        train_text,
        labels,
        training_data["validation_row_id"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=training_data["jel_code_1"],
    )

    model = tune_model(x_train, y_train)
    tuning_results = grid_search_results(model)
    best_grid_micro_f1 = float(tuning_results.iloc[0]["mean_test_micro_f1"])
    print_tuning_results(tuning_results)
    write_tuning_results(tuning_results)

    validation_probabilities = predict_probabilities(model, x_test)
    threshold_results = tune_thresholds(y_test, validation_probabilities, THRESHOLD_OPTIONS)
    print_threshold_results(threshold_results)
    write_threshold_results(threshold_results)
    best_threshold = float(threshold_results.iloc[0]["threshold"])
    validation_predictions = probabilities_to_multilabel(
        validation_probabilities,
        threshold=best_threshold,
    )
    metrics = multilabel_metrics(y_test, validation_predictions)

    prediction_probabilities = predict_probabilities(model, prediction_text)
    predicted_matrix = probabilities_to_multilabel(
        prediction_probabilities,
        threshold=best_threshold,
    )
    predicted_labels = format_multilabel_predictions(label_binarizer, predicted_matrix)
    predicted_confidence = prediction_probabilities.max(axis=1)

    predicted_data = prediction_data.copy()
    predicted_data[PREDICTED_LABEL_COLUMN] = predicted_labels
    predicted_data[PREDICTED_CONFIDENCE_COLUMN] = predicted_confidence.round(4)
    predicted_data[JEL_SOURCE_COLUMN] = "predicted_multi_tfidf"

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
        true_matrix=y_test,
        predicted_matrix=validation_predictions,
        probabilities=validation_probabilities,
        label_binarizer=label_binarizer,
    )
    joblib.dump(
        {
            "model": model,
            "label_binarizer": label_binarizer,
            "text_columns": TEXT_COLUMNS,
            "label_column": LABEL_COLUMN,
            "prediction_threshold": best_threshold,
            "threshold_options": THRESHOLD_OPTIONS,
            "best_params": model.best_params_,
        },
        MODEL_OUTPUT,
    )
    write_report(
        metrics=metrics,
        label_counts=label_counts(label_lists),
        label_classes=label_binarizer.classes_,
        training_rows=len(training_data),
        prediction_rows=len(predicted_data),
        best_params=model.best_params_,
        best_score=best_grid_micro_f1,
        tuning_results=tuning_results,
        threshold=best_threshold,
        threshold_results=threshold_results,
    )

    print("Multi-label JEL TF-IDF prediction summary:")
    print(f"  Training input CSV: {TRAINING_INPUT_CSV}")
    print(f"  Prediction input CSV: {PREDICTION_INPUT_CSV}")
    print(f"  Training rows with labels: {len(training_data)}")
    print(f"  Rows predicted: {len(predicted_data)}")
    print(f"  Grid-search candidates: {len(model.cv_results_['params'])}")
    print(f"  Best grid-search micro F1: {best_grid_micro_f1:.4f}")
    print(f"  Best parameters: {model.best_params_}")
    print(f"  Threshold options: {THRESHOLD_OPTIONS}")
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
    print(f"  Saved model: {MODEL_OUTPUT}")


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
                    stop_words="english",
                    use_idf=True,
                    norm="l2",
                    analyzer="word",
                ),
            ),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=3000,
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    )
                ),
            ),
        ]
    )


def tune_model(x_train: pd.Series, y_train: np.ndarray) -> GridSearchCV:
    search = GridSearchCV(
        estimator=build_model(),
        param_grid=PARAM_GRID,
        scoring=SCORING,
        cv=3,
        n_jobs=1,
        refit=best_grid_search_index,
        verbose=2,
    )
    search.fit(x_train, y_train)
    return search


def best_grid_search_index(cv_results: dict) -> int:
    results = prepare_grid_search_results(pd.DataFrame(cv_results))
    return int(results.index[0])


def grid_search_results(search: GridSearchCV) -> pd.DataFrame:
    results = prepare_grid_search_results(pd.DataFrame(search.cv_results_))
    fold_columns = [
        column
        for column in results.columns
        if column.startswith("split") and "_test_" in column
    ]
    columns = [
        "rank_test_micro_f1",
        "mean_test_micro_f1",
        "std_test_micro_f1",
        "mean_test_macro_f1",
        "std_test_macro_f1",
        "mean_test_micro_precision",
        "std_test_micro_precision",
        "mean_test_micro_recall",
        "std_test_micro_recall",
        "mean_test_hamming_loss",
        "std_test_hamming_loss",
        "mean_test_subset_accuracy",
        "std_test_subset_accuracy",
        "param_tfidf__ngram_range",
        "param_tfidf__min_df",
        "param_tfidf__max_df",
        "param_tfidf__max_features",
        "param_classifier__estimator__C",
        "param_classifier__estimator__class_weight",
    ]
    available_columns = [column for column in columns + fold_columns if column in results.columns]
    results = results[available_columns].copy()
    return sort_tuning_results(results)


def prepare_grid_search_results(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    hamming_columns = [
        column
        for column in results.columns
        if column == "mean_test_hamming_loss"
        or column.startswith("split") and column.endswith("_test_hamming_loss")
    ]
    for column in hamming_columns:
        results[column] = -results[column]
    return sort_tuning_results(results)


def sort_tuning_results(results: pd.DataFrame) -> pd.DataFrame:
    return results.sort_values(
        [
            "mean_test_micro_f1",
            "mean_test_macro_f1",
            "mean_test_micro_precision",
            "mean_test_micro_recall",
            "mean_test_hamming_loss",
            "mean_test_subset_accuracy",
        ],
        ascending=[False, False, False, False, True, False],
        kind="mergesort",
    )


def print_tuning_results(results: pd.DataFrame) -> None:
    print()
    print("Multi-label TF-IDF tuning metrics by parameter combination:")
    display_columns = [
        "rank_test_micro_f1",
        "mean_test_micro_f1",
        "std_test_micro_f1",
        "mean_test_macro_f1",
        "mean_test_micro_precision",
        "mean_test_micro_recall",
        "mean_test_hamming_loss",
        "mean_test_subset_accuracy",
        "param_tfidf__ngram_range",
        "param_tfidf__min_df",
        "param_tfidf__max_df",
        "param_tfidf__max_features",
        "param_classifier__estimator__C",
        "param_classifier__estimator__class_weight",
    ]
    display_columns = [column for column in display_columns if column in results.columns]
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print(results[display_columns].to_string(index=False))
    print()


def write_tuning_results(results: pd.DataFrame) -> None:
    TUNING_RESULTS_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(TUNING_RESULTS_OUTPUT_CSV, index=False)


def predict_probabilities(model: Pipeline, text: pd.Series) -> np.ndarray:
    probabilities = model.predict_proba(text)
    if isinstance(probabilities, list):
        probabilities = np.column_stack([values[:, 1] for values in probabilities])
    return np.asarray(probabilities)


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
        metrics = multilabel_metrics(true_matrix, predicted_matrix)
        rows.append(
            {
                "threshold": threshold,
                **metrics,
            }
        )

    results = pd.DataFrame(rows)
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


def print_threshold_results(results: pd.DataFrame) -> None:
    print()
    print("Multi-label TF-IDF threshold tuning results:")
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
            "jel_code_full_predicted_tfidf": format_multilabel_predictions(label_binarizer, predicted_matrix),
            "tfidf_max_confidence": probabilities.max(axis=1).round(4),
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
    best_score: float,
    tuning_results: pd.DataFrame,
    threshold: float,
    threshold_results: pd.DataFrame,
) -> None:
    REPORT_OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_OUTPUT_TXT.open("w", encoding="utf-8") as file:
        file.write("Multi-label JEL TF-IDF prediction model\n")
        file.write("=======================================\n\n")
        file.write("Model: TF-IDF + One-vs-Rest Logistic Regression\n")
        file.write(f"Label column: {LABEL_COLUMN}\n")
        file.write(f"Text columns: {TEXT_COLUMNS}\n")
        file.write(f"Prediction threshold: {threshold}\n")
        file.write(f"Threshold options: {THRESHOLD_OPTIONS}\n")
        file.write(f"Training rows: {training_rows}\n")
        file.write(f"Prediction rows: {prediction_rows}\n")
        file.write(f"Labels: {list(label_classes)}\n\n")
        file.write("Grid search:\n")
        file.write(f"  Cross-validation folds: 3\n")
        file.write(f"  Candidate models: {len(tuning_results)}\n")
        file.write(f"  Best cross-validation micro F1: {best_score:.4f}\n")
        file.write(f"  Best parameters: {best_params}\n")
        file.write(f"  Tuning results CSV: {TUNING_RESULTS_OUTPUT_CSV}\n\n")
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
        file.write("\nTop tuning results:\n")
        file.write(tuning_results.head(25).to_string(index=False))
        file.write("\n")


if __name__ == "__main__":
    main()
