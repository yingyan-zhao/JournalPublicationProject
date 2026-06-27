from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

TRAINING_INPUT_CSV = Path("data/trainingmodel/JEL_Training_Data_With_JEL.csv")
PREDICTION_INPUT_CSV = Path("data/trainingmodel/JEL_Training_Data_Without_JEL.csv")

PREDICTION_OUTPUT_CSV = Path("data/trainingmodel/JEL_Training_Data_Without_JEL_Predicted.csv")
COMBINED_OUTPUT_CSV = Path("data/trainingmodel/JEL_Training_Data_With_Observed_And_Predicted.csv")
VALIDATION_OUTPUT_CSV = Path("data/trainingmodel/JEL_Codes_1_TFIDF_Validation_Predictions.csv")
CONFUSION_MATRIX_OUTPUT_CSV = Path("data/trainingmodel/JEL_Codes_1_TFIDF_Confusion_Matrix.csv")
TUNING_RESULTS_OUTPUT_CSV = Path("data/trainingmodel/JEL_Codes_1_TFIDF_Tuning_Results.csv")

REPORT_OUTPUT_TXT = Path("data/trainingmodel/JEL_Codes_1_TFIDF_Model_Report.txt")
MODEL_OUTPUT = Path("data/trainingmodel/JEL_Codes_1_TFIDF_LogisticRegression.joblib")

TEXT_COLUMNS = ["title", "keywords", "abstract"]

LABEL_COLUMN = "jel_code_1"
PREDICTED_LABEL_COLUMN = "jel_code_1_predicted_tfidf"
PREDICTED_CONFIDENCE_COLUMN = "jel_code_1_predicted_tfidf_confidence"
JEL_SOURCE_COLUMN = "jel_code_1_tfidf_source"
RANDOM_STATE = 2026
CV_FOLDS = 5
GRID_SEARCH_N_JOBS = -1

TFIDF_PARAM_GRID = {
    "tfidf__ngram_range": [ (1, 3), (1, 4), (1, 5)],
    "tfidf__min_df": [ 6, 7, 8],
    "tfidf__max_df": [0.9, 0.95],
    "tfidf__max_features": [30000],
}


def main() -> None:
    training_data = read_data(TRAINING_INPUT_CSV)
    prediction_data = read_data(PREDICTION_INPUT_CSV)

    training_data = keep_labeled_rows(training_data, LABEL_COLUMN)
    training_data["validation_row_id"] = training_data.index.astype(str)
    train_text = combine_text_columns(training_data, TEXT_COLUMNS)
    labels = training_data[LABEL_COLUMN].astype(str).str.strip()
    row_ids = training_data["validation_row_id"]

    cv = make_cross_validator(labels)
    model = build_model()
    search = tune_model(
        model=model,
        texts=train_text,
        labels=labels,
        cv=cv,
    )
    tuning_results = grid_search_results(search)
    print_tuning_results(tuning_results)
    write_tuning_results(tuning_results)
    best_model_for_validation = build_model()
    best_model_for_validation.set_params(**search.best_params_)
    cv_scores, validation_predictions, validation_confidence = cross_validate_model(
        model=best_model_for_validation,
        texts=train_text,
        labels=labels,
        cv=cv,
    )
    validation_accuracy = cv_scores.mean()
    validation_accuracy_std = cv_scores.std()
    any_code_match = predicted_label_in_jel_code_full(
        jel_code_full=training_data["jel_code_full"],
        predicted_labels=validation_predictions,
    )
    any_code_accuracy = float(any_code_match.mean())
    validation_report = classification_report(
        labels,
        validation_predictions,
        zero_division=0,
    )

    final_model = search.best_estimator_

    prediction_text = combine_text_columns(prediction_data, TEXT_COLUMNS)
    predicted_labels = final_model.predict(prediction_text)
    predicted_confidence = final_model.predict_proba(prediction_text).max(axis=1)

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
        row_ids=row_ids,
        true_labels=labels,
        jel_code_full=training_data["jel_code_full"],
        predicted_labels=validation_predictions,
        confidence=validation_confidence,
        any_code_match=any_code_match,
    )
    write_confusion_matrix(labels, validation_predictions)
    joblib.dump(final_model, MODEL_OUTPUT)
    write_report(
        validation_accuracy=validation_accuracy,
        validation_accuracy_std=validation_accuracy_std,
        any_code_accuracy=any_code_accuracy,
        validation_report=validation_report,
        label_counts=labels.value_counts().sort_index(),
        cv_scores=cv_scores,
        best_params=search.best_params_,
        grid_search_best_score=search.best_score_,
        tuning_results=tuning_results,
    )

    print("JEL code first-letter prediction summary:")
    print(f"  Training input CSV: {TRAINING_INPUT_CSV}")
    print(f"  Prediction input CSV: {PREDICTION_INPUT_CSV}")
    print(f"  Training rows with labels: {len(training_data)}")
    print(f"  Rows predicted: {len(predicted_data)}")
    print(f"  Grid-search candidates: {len(search.cv_results_['params'])}")
    print(f"  Best grid-search accuracy: {search.best_score_:.4f}")
    print(f"  Best TF-IDF parameters: {search.best_params_}")
    print(f"  Cross-validation folds: {len(cv_scores)}")
    print(f"  Cross-validation accuracy mean: {validation_accuracy:.4f}")
    print(f"  Cross-validation accuracy std: {validation_accuracy_std:.4f}")
    print(f"  Cross-validation any-code accuracy: {any_code_accuracy:.4f}")
    print(f"  Prediction output CSV: {PREDICTION_OUTPUT_CSV}")
    print(f"  Combined output CSV: {COMBINED_OUTPUT_CSV}")
    print(f"  Validation prediction CSV: {VALIDATION_OUTPUT_CSV}")
    print(f"  Confusion matrix CSV: {CONFUSION_MATRIX_OUTPUT_CSV}")
    print(f"  Tuning results CSV: {TUNING_RESULTS_OUTPUT_CSV}")
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


def make_cross_validator(labels: pd.Series) -> StratifiedKFold:
    smallest_label_count = int(labels.value_counts().min())
    n_splits = min(CV_FOLDS, smallest_label_count)
    if n_splits < 2:
        raise ValueError("At least two observations per JEL label are needed for cross-validation.")

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def tune_model(
    model: Pipeline,
    texts: pd.Series,
    labels: pd.Series,
    cv: StratifiedKFold,
) -> GridSearchCV:
    search = GridSearchCV(
        estimator=model,
        param_grid=TFIDF_PARAM_GRID,
        scoring="accuracy",
        cv=cv,
        n_jobs=GRID_SEARCH_N_JOBS,
        refit=True,
        verbose=2,
    )
    with parallel_backend("threading"):
        search.fit(texts, labels)
    return search


def grid_search_results(search: GridSearchCV) -> pd.DataFrame:
    results = pd.DataFrame(search.cv_results_)
    columns = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "param_tfidf__ngram_range",
        "param_tfidf__min_df",
        "param_tfidf__max_df",
        "param_tfidf__max_features",
    ]
    fold_columns = [
        column for column in results.columns
        if column.startswith("split") and column.endswith("_test_score")
    ]
    available_columns = [column for column in columns + fold_columns if column in results.columns]
    results = results[available_columns].copy()
    return results.sort_values(
        ["rank_test_score", "mean_test_score"],
        ascending=[True, False],
    )


def print_tuning_results(results: pd.DataFrame) -> None:
    print("TF-IDF tuning accuracy by parameter combination:")
    for _, row in results.iterrows():
        print(
            "  "
            f"rank={int(row['rank_test_score'])}, "
            f"accuracy={row['mean_test_score']:.4f}, "
            f"std={row['std_test_score']:.4f}, "
            f"ngram={row['param_tfidf__ngram_range']}, "
            f"min_df={row['param_tfidf__min_df']}, "
            f"max_df={row['param_tfidf__max_df']}, "
            f"max_features={row['param_tfidf__max_features']}"
        )


def write_tuning_results(results: pd.DataFrame) -> None:
    TUNING_RESULTS_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(TUNING_RESULTS_OUTPUT_CSV, index=False)


def cross_validate_model(
    model: Pipeline,
    texts: pd.Series,
    labels: pd.Series,
    cv: StratifiedKFold,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    text_values = texts.to_numpy()
    label_values = labels.to_numpy()
    predictions = np.empty(len(labels), dtype=object)
    confidence = np.zeros(len(labels), dtype=float)
    scores = []

    for fold_number, (train_index, test_index) in enumerate(
        cv.split(text_values, label_values),
        start=1,
    ):
        fold_model = clone(model)
        fold_model.fit(text_values[train_index], label_values[train_index])

        fold_predictions = fold_model.predict(text_values[test_index])
        fold_confidence = fold_model.predict_proba(text_values[test_index]).max(axis=1)
        predictions[test_index] = fold_predictions
        confidence[test_index] = fold_confidence

        fold_accuracy = accuracy_score(label_values[test_index], fold_predictions)
        scores.append(fold_accuracy)
        print(f"  CV fold {fold_number}/{cv.get_n_splits()}: accuracy {fold_accuracy:.4f}")

    return np.array(scores), predictions, confidence


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
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
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
    true_labels: pd.Series,
    predicted_labels,
) -> None:
    labels_order = sorted(true_labels.unique())
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
    validation_accuracy_std: float,
    any_code_accuracy: float,
    validation_report: str,
    label_counts: pd.Series,
    cv_scores: np.ndarray,
    best_params: dict,
    grid_search_best_score: float,
    tuning_results: pd.DataFrame,
) -> None:
    REPORT_OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_OUTPUT_TXT.open("w", encoding="utf-8") as file:
        file.write("JEL code first-letter prediction model\n")
        file.write("======================================\n\n")
        file.write("Model: TF-IDF text features + Logistic Regression\n")
        file.write("Evaluation: stratified cross-validation\n")
        file.write(f"Label column: {LABEL_COLUMN}\n")
        file.write(f"Text columns: {TEXT_COLUMNS}\n\n")
        file.write("Label counts:\n")
        file.write(label_counts.to_string())
        file.write("\n\n")
        file.write("TF-IDF grid search:\n")
        file.write(f"  Candidate models: {np.prod([len(values) for values in TFIDF_PARAM_GRID.values()])}\n")
        file.write(f"  Best grid-search accuracy: {grid_search_best_score:.4f}\n")
        file.write(f"  Best parameters: {best_params}\n")
        file.write(f"  Tuning results CSV: {TUNING_RESULTS_OUTPUT_CSV}\n")
        file.write("  Tuning results:\n")
        file.write(tuning_results.to_string(index=False))
        file.write("\n\n")
        file.write(f"Cross-validation folds: {len(cv_scores)}\n")
        file.write(f"Cross-validation accuracy mean: {validation_accuracy:.4f}\n")
        file.write(f"Cross-validation accuracy std: {validation_accuracy_std:.4f}\n")
        file.write(f"Cross-validation any-code accuracy: {any_code_accuracy:.4f}\n")
        file.write(f"Confusion matrix CSV: {CONFUSION_MATRIX_OUTPUT_CSV}\n")
        file.write("Cross-validation fold accuracies:\n")
        for fold_number, score in enumerate(cv_scores, start=1):
            file.write(f"  Fold {fold_number}: {score:.4f}\n")
        file.write("\n")
        file.write("Out-of-fold classification report:\n")
        file.write(validation_report)
        file.write("\n")


def write_validation_predictions(
    row_ids: pd.Series,
    true_labels: pd.Series,
    jel_code_full: pd.Series,
    predicted_labels,
    confidence,
    any_code_match: pd.Series,
) -> None:
    validation = pd.DataFrame(
        {
            "validation_row_id": row_ids.astype(str).to_list(),
            "jel_code_1_true": true_labels.astype(str).to_list(),
            "jel_code_full": jel_code_full.astype(str).to_list(),
            "tfidf_jel_code_1_predicted": predicted_labels,
            "tfidf_confidence": pd.Series(confidence).round(4),
            "tfidf_prediction_in_jel_code_full": any_code_match.astype(int).to_list(),
        }
    )
    validation.to_csv(VALIDATION_OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()
