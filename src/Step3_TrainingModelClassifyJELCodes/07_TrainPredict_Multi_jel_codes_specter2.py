from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

TRAINING_INPUT_CSV = Path("data/processed/JEL_Training_Data_With_JEL.csv")
PREDICTION_INPUT_CSV = Path("data/processed/JEL_Training_Data_Without_JEL.csv")

PREDICTION_OUTPUT_CSV = Path("data/processed/JEL_Training_Data_Without_JEL_Predicted_Multi_SPECTER2.csv")
COMBINED_OUTPUT_CSV = Path("data/processed/JEL_Training_Data_With_Observed_And_Predicted_Multi_SPECTER2.csv")
VALIDATION_OUTPUT_CSV = Path("data/processed/JEL_Codes_Multi_SPECTER2_Validation_Predictions.csv")
TUNING_RESULTS_OUTPUT_CSV = Path("data/processed/JEL_Codes_Multi_SPECTER2_Tuning_Results.csv")
THRESHOLD_TUNING_OUTPUT_CSV = Path("data/processed/JEL_Codes_Multi_SPECTER2_Threshold_Tuning_Results.csv")
REPORT_OUTPUT_TXT = Path("data/processed/JEL_Codes_Multi_SPECTER2_Report.txt")
MODEL_OUTPUT = Path("data/processed/JEL_Codes_Multi_SPECTER2_OneVsRest_LogisticRegression.joblib")

TRAINING_EMBEDDINGS_OUTPUT = Path("data/processed/JEL_Codes_Multi_SPECTER2_Training_Embeddings.npy")
PREDICTION_EMBEDDINGS_OUTPUT = Path("data/processed/JEL_Codes_Multi_SPECTER2_Prediction_Embeddings.npy")

MODEL_OPTIONS = [
    "allenai/specter2",
    "allenai/specter2_base",
    "allenai/scibert_scivocab_uncased",
]
SPECTER2_BASE_MODEL = "allenai/specter2_base"
POOLING_OPTIONS = ["cls", "mean"]

TEXT_COLUMNS = ["title", "keywords", "abstract"]
LABEL_COLUMN = "jel_code_full"
PREDICTED_LABEL_COLUMN = "jel_code_full_predicted_specter2"
PREDICTED_CONFIDENCE_COLUMN = "jel_code_full_predicted_specter2_max_confidence"
JEL_SOURCE_COLUMN = "jel_code_full_specter2_source"

RANDOM_STATE = 2026
TEST_SIZE = 0.2
CV_FOLDS = 3
BATCH_SIZE = 16
MAX_LENGTH = 512
THRESHOLD_OPTIONS = [0.2, 0.3, 0.4, 0.5, 0.6]

CLASSIFIER_PARAM_GRID = {
    "scaler": [StandardScaler(), "passthrough"],
    "classifier__estimator__C": [0.1, 0.3, 1, 3, 10],
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
    transformers, torch = import_transformer_dependencies()

    training_data = keep_labeled_rows(read_data(TRAINING_INPUT_CSV), LABEL_COLUMN)
    prediction_data = read_data(PREDICTION_INPUT_CSV)

    training_data["validation_row_id"] = training_data.index.astype(str)
    training_text = combine_text_columns(training_data, TEXT_COLUMNS)
    prediction_text = combine_text_columns(prediction_data, TEXT_COLUMNS)
    label_lists = training_data[LABEL_COLUMN].apply(split_label_cell)

    label_binarizer = MultiLabelBinarizer()
    labels = label_binarizer.fit_transform(label_lists)

    train_index, test_index = train_test_split(
        np.arange(len(training_data)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=training_data["jel_code_1"],
    )

    print("Multi-label SPECTER2 setup:")
    hf_token = huggingface_token()
    print(f"  Model options: {MODEL_OPTIONS}")
    print(f"  Python sees Hugging Face token: {bool(hf_token)}")
    print(f"  Text columns: {TEXT_COLUMNS}")
    print(f"  Label column: {LABEL_COLUMN}")
    print(f"  Threshold options: {THRESHOLD_OPTIONS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    best_model_name = ""
    best_pooling = ""
    best_search = None
    best_training_embeddings = None
    best_prediction_embeddings = None
    best_grid_micro_f1 = -1.0
    tuning_result_frames = []

    for model_name in MODEL_OPTIONS:
        print(f"\nTrying embedding model: {model_name}")
        tokenizer, embedding_model = load_embedding_components(
            transformers=transformers,
            model_name=model_name,
            hf_token=hf_token,
        )
        embedding_model.to(device)
        embedding_model.eval()

        training_embeddings_by_pooling = encode_texts(
            texts=training_text.tolist(),
            tokenizer=tokenizer,
            model=embedding_model,
            torch=torch,
            device=device,
            label=f"training ({model_name})",
        )
        prediction_embeddings_by_pooling = encode_texts(
            texts=prediction_text.tolist(),
            tokenizer=tokenizer,
            model=embedding_model,
            torch=torch,
            device=device,
            label=f"prediction ({model_name})",
        )
        save_embeddings(training_embeddings_by_pooling, prediction_embeddings_by_pooling, model_name)

        for pooling in POOLING_OPTIONS:
            print(f"  Tuning classifier with {pooling} pooling...")
            search = tune_classifier(
                embeddings=training_embeddings_by_pooling[pooling][train_index],
                labels=labels[train_index],
                cv=cv,
            )
            result_frame = grid_search_results(search, model_name, pooling)
            tuning_result_frames.append(result_frame)
            model_score = float(result_frame.iloc[0]["mean_test_micro_f1"])
            print(f"    {pooling} best grid-search micro F1: {model_score:.4f}")

            if model_score > best_grid_micro_f1:
                best_grid_micro_f1 = model_score
                best_model_name = model_name
                best_pooling = pooling
                best_search = search
                best_training_embeddings = training_embeddings_by_pooling[pooling]
                best_prediction_embeddings = prediction_embeddings_by_pooling[pooling]

        del embedding_model

    if best_search is None or best_training_embeddings is None or best_prediction_embeddings is None:
        raise ValueError("No SPECTER2/embedding model completed tuning.")

    tuning_results = combine_tuning_results(tuning_result_frames)
    print_tuning_results(tuning_results)
    write_tuning_results(tuning_results)

    validation_probabilities = predict_probabilities(best_search, best_training_embeddings[test_index])
    threshold_results = tune_thresholds(labels[test_index], validation_probabilities, THRESHOLD_OPTIONS)
    print_threshold_results(threshold_results)
    write_threshold_results(threshold_results)
    best_threshold = float(threshold_results.iloc[0]["threshold"])

    validation_predictions = probabilities_to_multilabel(validation_probabilities, threshold=best_threshold)
    metrics = multilabel_metrics(labels[test_index], validation_predictions)

    prediction_probabilities = predict_probabilities(best_search, best_prediction_embeddings)
    predicted_matrix = probabilities_to_multilabel(prediction_probabilities, threshold=best_threshold)
    predicted_labels = format_multilabel_predictions(label_binarizer, predicted_matrix)
    predicted_confidence = prediction_probabilities.max(axis=1)

    predicted_data = prediction_data.copy()
    predicted_data[PREDICTED_LABEL_COLUMN] = predicted_labels
    predicted_data[PREDICTED_CONFIDENCE_COLUMN] = predicted_confidence.round(4)
    predicted_data[JEL_SOURCE_COLUMN] = "predicted_multi_specter2"

    observed_data = training_data.copy()
    observed_data[PREDICTED_LABEL_COLUMN] = observed_data[LABEL_COLUMN]
    observed_data[PREDICTED_CONFIDENCE_COLUMN] = ""
    observed_data[JEL_SOURCE_COLUMN] = "observed"
    combined = pd.concat([observed_data, predicted_data], ignore_index=True, sort=False)

    PREDICTION_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    predicted_data.to_csv(PREDICTION_OUTPUT_CSV, index=False)
    combined.to_csv(COMBINED_OUTPUT_CSV, index=False)
    write_validation_predictions(
        row_ids=training_data.iloc[test_index]["validation_row_id"],
        true_matrix=labels[test_index],
        predicted_matrix=validation_predictions,
        probabilities=validation_probabilities,
        label_binarizer=label_binarizer,
    )
    joblib.dump(
        {
            "classifier": best_search,
            "label_binarizer": label_binarizer,
            "model_name": best_model_name,
            "model_options": MODEL_OPTIONS,
            "text_columns": TEXT_COLUMNS,
            "label_column": LABEL_COLUMN,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "pooling": best_pooling,
            "pooling_options": POOLING_OPTIONS,
            "prediction_threshold": best_threshold,
            "threshold_options": THRESHOLD_OPTIONS,
            "best_params": best_search.best_params_,
        },
        MODEL_OUTPUT,
    )
    write_report(
        metrics=metrics,
        label_counts=label_counts(label_lists),
        label_classes=label_binarizer.classes_,
        training_rows=len(training_data),
        prediction_rows=len(predicted_data),
        best_model_name=best_model_name,
        best_pooling=best_pooling,
        best_params=best_search.best_params_,
        best_score=best_grid_micro_f1,
        tuning_results=tuning_results,
        threshold=best_threshold,
        threshold_results=threshold_results,
    )

    print("Multi-label SPECTER2 prediction summary:")
    print(f"  Training rows with labels: {len(training_data)}")
    print(f"  Rows predicted: {len(predicted_data)}")
    print(f"  Best embedding model: {best_model_name}")
    print(f"  Best pooling: {best_pooling}")
    print(f"  Best grid-search micro F1: {best_grid_micro_f1:.4f}")
    print(f"  Best classifier parameters: {best_search.best_params_}")
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


def load_tokenizer(transformers, model_name: str, hf_token: str | None):
    try:
        return transformers.AutoTokenizer.from_pretrained(model_name, token=hf_token)
    except ValueError:
        print(
            "  Fast tokenizer failed to load. Trying the slow tokenizer. "
            "If this also fails, install sentencepiece and tiktoken."
        )
        try:
            return transformers.AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token,
                use_fast=False,
            )
        except ValueError as slow_error:
            raise ValueError(
                f"Could not load tokenizer for {model_name}. "
                "Install tokenizer dependencies with: pip install sentencepiece tiktoken"
            ) from slow_error


def load_embedding_components(transformers, model_name: str, hf_token: str | None):
    if model_name == "allenai/specter2":
        return load_specter2_adapter_model(transformers, hf_token)

    tokenizer = load_tokenizer(transformers, model_name, hf_token)
    model = transformers.AutoModel.from_pretrained(model_name, token=hf_token)
    return tokenizer, model


def load_specter2_adapter_model(transformers, hf_token: str | None):
    try:
        from adapters import AutoAdapterModel
    except ImportError as error:
        raise ImportError(
            "allenai/specter2 is an adapter for allenai/specter2_base, not a standalone "
            "Transformers model. Install adapter support with: pip install adapters"
        ) from error

    tokenizer = load_tokenizer(transformers, SPECTER2_BASE_MODEL, hf_token)
    model = AutoAdapterModel.from_pretrained(SPECTER2_BASE_MODEL, token=hf_token)
    model.load_adapter(
        "allenai/specter2",
        source="hf",
        load_as="specter2",
        set_active=True,
    )
    activate_adapter(model, "specter2")
    return tokenizer, model


def activate_adapter(model, adapter_name: str) -> None:
    if hasattr(model, "set_active_adapters"):
        model.set_active_adapters(adapter_name)
    if hasattr(model, "active_adapters"):
        try:
            model.active_adapters = adapter_name
        except AttributeError:
            pass
    model._active_adapter_name = adapter_name


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


def encode_texts(texts: list[str], tokenizer, model, torch, device, label: str) -> dict[str, np.ndarray]:
    embeddings_by_pooling = {pooling: [] for pooling in POOLING_OPTIONS}
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
            output = model_forward(model, encoded)

        hidden_state = output.last_hidden_state
        if "cls" in POOLING_OPTIONS:
            embeddings_by_pooling["cls"].append(hidden_state[:, 0, :].detach().cpu().numpy())
        if "mean" in POOLING_OPTIONS:
            embeddings_by_pooling["mean"].append(
                mean_pool_embeddings(
                    hidden_state=hidden_state,
                    attention_mask=encoded["attention_mask"],
                    torch=torch,
                )
            )

    return {
        pooling: np.vstack(embedding_parts)
        for pooling, embedding_parts in embeddings_by_pooling.items()
    }


def mean_pool_embeddings(hidden_state, attention_mask, torch) -> np.ndarray:
    mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    masked_hidden_state = hidden_state * mask
    summed = masked_hidden_state.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return (summed / counts).detach().cpu().numpy()


def model_forward(model, encoded):
    adapter_name = getattr(model, "_active_adapter_name", "")
    if adapter_name:
        try:
            from adapters import AdapterSetup

            with AdapterSetup(adapter_name):
                return model(**encoded)
        except ImportError:
            pass
    return model(**encoded)


def save_embeddings(
    training_embeddings_by_pooling: dict[str, np.ndarray],
    prediction_embeddings_by_pooling: dict[str, np.ndarray],
    model_name: str,
) -> None:
    TRAINING_EMBEDDINGS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for pooling, training_embeddings in training_embeddings_by_pooling.items():
        training_path = embedding_output_path(TRAINING_EMBEDDINGS_OUTPUT, model_name, pooling)
        prediction_path = embedding_output_path(PREDICTION_EMBEDDINGS_OUTPUT, model_name, pooling)
        np.save(training_path, training_embeddings)
        np.save(prediction_path, prediction_embeddings_by_pooling[pooling])
        print(f"  Saved {pooling} training embeddings: {training_path}")
        print(f"  Saved {pooling} prediction embeddings: {prediction_path}")


def embedding_output_path(path: Path, model_name: str, pooling: str) -> Path:
    model_slug = safe_filename_part(model_name)
    return path.with_name(f"{path.stem}_{model_slug}_{pooling}{path.suffix}")


def safe_filename_part(value: str) -> str:
    return value.replace("/", "_").replace(":", "_").replace(" ", "_")


def build_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(
                        max_iter=3000,
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    )
                ),
            ),
        ]
    )


def tune_classifier(embeddings: np.ndarray, labels: np.ndarray, cv: KFold) -> GridSearchCV:
    search = GridSearchCV(
        estimator=build_classifier(),
        param_grid=CLASSIFIER_PARAM_GRID,
        scoring=SCORING,
        cv=cv,
        n_jobs=1,
        refit=best_grid_search_index,
        verbose=1,
    )
    search.fit(embeddings, labels)
    return search


def best_grid_search_index(cv_results: dict) -> int:
    results = prepare_grid_search_results(pd.DataFrame(cv_results))
    return int(results.index[0])


def grid_search_results(search: GridSearchCV, model_name: str, pooling: str) -> pd.DataFrame:
    results = prepare_grid_search_results(pd.DataFrame(search.cv_results_))
    results["embedding_model"] = model_name
    results["pooling"] = pooling
    fold_columns = [
        column
        for column in results.columns
        if column.startswith("split") and "_test_" in column
    ]
    columns = [
        "embedding_model",
        "pooling",
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
        "param_scaler",
        "param_classifier__estimator__C",
        "param_classifier__estimator__class_weight",
    ]
    available_columns = [column for column in columns + fold_columns if column in results.columns]
    return sort_tuning_results(results[available_columns].copy())


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


def combine_tuning_results(result_frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not result_frames:
        return pd.DataFrame()
    return sort_tuning_results(pd.concat(result_frames, ignore_index=True, sort=False))


def print_tuning_results(results: pd.DataFrame) -> None:
    print()
    print("Multi-label SPECTER2 tuning metrics by parameter combination:")
    if results.empty:
        print("  No tuning results.")
        return
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 220):
        print(results.to_string(index=False))
    print()


def write_tuning_results(results: pd.DataFrame) -> None:
    TUNING_RESULTS_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(TUNING_RESULTS_OUTPUT_CSV, index=False)


def predict_probabilities(model, embeddings: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(embeddings)
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
        rows.append({"threshold": threshold, **multilabel_metrics(true_matrix, predicted_matrix)})
    return sort_threshold_results(pd.DataFrame(rows))


def sort_threshold_results(results: pd.DataFrame) -> pd.DataFrame:
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
    print("Multi-label SPECTER2 threshold tuning results:")
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
            "jel_code_full_predicted_specter2": format_multilabel_predictions(label_binarizer, predicted_matrix),
            "specter2_max_confidence": probabilities.max(axis=1).round(4),
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
    best_model_name: str,
    best_pooling: str,
    best_params: dict,
    best_score: float,
    tuning_results: pd.DataFrame,
    threshold: float,
    threshold_results: pd.DataFrame,
) -> None:
    REPORT_OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_OUTPUT_TXT.open("w", encoding="utf-8") as file:
        file.write("Multi-label JEL SPECTER2 prediction model\n")
        file.write("=========================================\n\n")
        file.write("Model: SPECTER2 embeddings + One-vs-Rest Logistic Regression\n")
        file.write(f"Embedding model options: {MODEL_OPTIONS}\n")
        file.write(f"Best embedding model: {best_model_name}\n")
        file.write(f"Pooling options: {POOLING_OPTIONS}\n")
        file.write(f"Best pooling: {best_pooling}\n")
        file.write(f"Label column: {LABEL_COLUMN}\n")
        file.write(f"Text columns: {TEXT_COLUMNS}\n")
        file.write(f"Max length: {MAX_LENGTH}\n")
        file.write(f"Batch size: {BATCH_SIZE}\n")
        file.write(f"Prediction threshold: {threshold}\n")
        file.write(f"Threshold options: {THRESHOLD_OPTIONS}\n")
        file.write(f"Training rows: {training_rows}\n")
        file.write(f"Prediction rows: {prediction_rows}\n")
        file.write(f"Labels: {list(label_classes)}\n\n")
        file.write("Grid search:\n")
        file.write(f"  Candidate rows: {len(tuning_results)}\n")
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
