from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

TRAINING_INPUT_CSV = Path("data/trainingmodel/JEL_Training_Data_With_JEL.csv")
PREDICTION_INPUT_CSV = Path("data/trainingmodel/JEL_Training_Data_Without_JEL.csv")

PREDICTION_OUTPUT_CSV = Path("data/trainingmodel/JEL_Training_Data_Without_JEL_Predicted_SPECTER2.csv")
COMBINED_OUTPUT_CSV = Path("data/trainingmodel/JEL_Training_Data_With_Observed_And_Predicted_SPECTER2.csv")
VALIDATION_OUTPUT_CSV = Path("data/trainingmodel/JEL_Codes_1_SPECTER2_Validation_Predictions.csv")
CONFUSION_MATRIX_OUTPUT_CSV = Path("data/trainingmodel/JEL_Codes_1_SPECTER2_Confusion_Matrix.csv")
TUNING_RESULTS_OUTPUT_CSV = Path("data/trainingmodel/JEL_Codes_1_SPECTER2_Tuning_Results.csv")
REPORT_OUTPUT_TXT = Path("data/trainingmodel/JEL_Codes_1_SPECTER2_LogisticRegression_Report.txt")
MODEL_OUTPUT = Path("data/trainingmodel/JEL_Codes_1_SPECTER2_LogisticRegression.joblib")

TRAINING_EMBEDDINGS_OUTPUT = Path("data/trainingmodel/JEL_Codes_1_SPECTER2_Training_Embeddings.npy")
PREDICTION_EMBEDDINGS_OUTPUT = Path("data/trainingmodel/JEL_Codes_1_SPECTER2_Prediction_Embeddings.npy")

MODEL_OPTIONS = [
    "allenai/specter2",
    "allenai/specter2_base",
    "allenai/scibert_scivocab_uncased",
]
SPECTER2_BASE_MODEL = "allenai/specter2_base"
TEXT_COLUMNS = ["title", "keywords", "abstract"]
LABEL_COLUMN = "jel_code_1"
PREDICTED_LABEL_COLUMN = "jel_code_1_predicted_specter2"
PREDICTED_CONFIDENCE_COLUMN = "jel_code_1_predicted_specter2_confidence"
JEL_SOURCE_COLUMN = "jel_code_1_specter2_source"

RANDOM_STATE = 2026
CV_FOLDS = 5
BATCH_SIZE = 32
MAX_LENGTH = 512
POOLING_OPTIONS = ["cls",
                   "mean"
]

CLASSIFIER_PARAM_GRID = {
    "scaler": [StandardScaler(), "passthrough"],
    "classifier__C": [0.1, 0.3, 1],
    "classifier__class_weight": ["balanced", None],
}


def main() -> None:
    transformers, torch = import_transformer_dependencies()

    training_data = keep_labeled_rows(read_data(TRAINING_INPUT_CSV), LABEL_COLUMN)
    training_data["validation_row_id"] = training_data.index.astype(str)
    prediction_data = read_data(PREDICTION_INPUT_CSV)

    labels = training_data[LABEL_COLUMN].fillna("").astype(str).str.strip()
    row_ids = training_data["validation_row_id"]
    training_text = combine_text_columns(training_data, TEXT_COLUMNS)
    prediction_text = combine_text_columns(prediction_data, TEXT_COLUMNS)

    print("SPECTER2 setup:")
    hf_token = huggingface_token()
    print(f"  Model options: {MODEL_OPTIONS}")
    print(f"  Python sees Hugging Face token: {bool(hf_token)}")
    print(f"  Text columns: {TEXT_COLUMNS}")
    print(f"  Label column: {LABEL_COLUMN}")

    print("PyTorch version:", torch.__version__)
    print("MPS available:", torch.backends.mps.is_available())
    print("MPS built:", torch.backends.mps.is_built())

    device = select_torch_device(torch)
    print(f"  Device: {device}")

    cv = make_cross_validator(labels)
    base_classifier = build_classifier()
    best_model_name = ""
    best_pooling = ""
    best_search = None
    best_training_embeddings = None
    best_prediction_embeddings = None
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
        save_embeddings(
            training_embeddings_by_pooling=training_embeddings_by_pooling,
            prediction_embeddings_by_pooling=prediction_embeddings_by_pooling,
            model_name=model_name,
        )

        model_best_pooling, search = tune_pooling_and_classifier(
            classifier=base_classifier,
            embeddings_by_pooling=training_embeddings_by_pooling,
            labels=labels,
            cv=cv,
            model_name=model_name,
            tuning_result_frames=tuning_result_frames,
        )
        print(
            f"  Best for {model_name}: pooling={model_best_pooling}, "
            f"accuracy={search.best_score_:.4f}, params={search.best_params_}"
        )

        if best_search is None or search.best_score_ > best_search.best_score_:
            best_model_name = model_name
            best_pooling = model_best_pooling
            best_search = search
            best_training_embeddings = training_embeddings_by_pooling[model_best_pooling]
            best_prediction_embeddings = prediction_embeddings_by_pooling[model_best_pooling]

        del embedding_model

    if best_search is None or best_training_embeddings is None or best_prediction_embeddings is None:
        raise ValueError("No SPECTER2/embedding model completed tuning.")

    search = best_search
    tuning_results = combine_tuning_results(tuning_result_frames)
    print_tuning_results(tuning_results)
    write_tuning_results(tuning_results)
    training_embeddings = best_training_embeddings
    prediction_embeddings = best_prediction_embeddings

    best_classifier_for_validation = build_classifier()
    best_classifier_for_validation.set_params(**search.best_params_)
    cv_scores, validation_predictions, validation_confidence = cross_validate_classifier(
        classifier=best_classifier_for_validation,
        embeddings=training_embeddings,
        labels=labels,
        cv=cv,
    )
    validation_accuracy = float(cv_scores.mean())
    validation_accuracy_std = float(cv_scores.std())
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

    final_classifier = search.best_estimator_
    predicted_labels = final_classifier.predict(prediction_embeddings)
    predicted_confidence = final_classifier.predict_proba(prediction_embeddings).max(axis=1)

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
        row_ids=row_ids,
        true_labels=labels,
        jel_code_full=training_data["jel_code_full"],
        predicted_labels=validation_predictions,
        confidence=validation_confidence,
        any_code_match=any_code_match,
    )
    write_confusion_matrix(labels, validation_predictions)
    joblib.dump(
        {
            "classifier": final_classifier,
            "model_name": best_model_name,
            "model_options": MODEL_OPTIONS,
            "text_columns": TEXT_COLUMNS,
            "label_column": LABEL_COLUMN,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "pooling": best_pooling,
            "pooling_options": POOLING_OPTIONS,
            "best_params": search.best_params_,
        },
        MODEL_OUTPUT,
    )
    write_report(
        validation_accuracy=validation_accuracy,
        validation_accuracy_std=validation_accuracy_std,
        any_code_accuracy=any_code_accuracy,
        validation_report=validation_report,
        label_counts=labels.value_counts().sort_index(),
        cv_scores=cv_scores,
        best_model_name=best_model_name,
        best_pooling=best_pooling,
        best_params=search.best_params_,
        grid_search_best_score=float(search.best_score_),
        tuning_results=tuning_results,
    )

    print("SPECTER2 embedding + Logistic Regression summary:")
    print(f"  Training rows with labels: {len(training_data)}")
    print(f"  Rows predicted: {len(predicted_data)}")
    print(f"  Model options tried: {MODEL_OPTIONS}")
    print(f"  Classifier grid candidates per model/pooling: {len(search.cv_results_['params'])}")
    print(f"  Best grid-search accuracy: {search.best_score_:.4f}")
    print(f"  Best embedding model: {best_model_name}")
    print(f"  Best pooling: {best_pooling}")
    print(f"  Best classifier parameters: {search.best_params_}")
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


def select_torch_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def huggingface_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def load_tokenizer(transformers, model_name: str, hf_token: str | None):
    try:
        return transformers.AutoTokenizer.from_pretrained(model_name, token=hf_token)
    except ValueError as error:
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
            cls_embeddings = hidden_state[:, 0, :].detach().cpu().numpy()
            embeddings_by_pooling["cls"].append(cls_embeddings)
        if "mean" in POOLING_OPTIONS:
            mean_embeddings = mean_pool_embeddings(
                hidden_state=hidden_state,
                attention_mask=encoded["attention_mask"],
                torch=torch,
            )
            embeddings_by_pooling["mean"].append(mean_embeddings)

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
    return (
        value.replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
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


def build_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def tune_pooling_and_classifier(
    classifier: Pipeline,
    embeddings_by_pooling: dict[str, np.ndarray],
    labels: pd.Series,
    cv: StratifiedKFold,
    model_name: str,
    tuning_result_frames: list[pd.DataFrame],
) -> tuple[str, GridSearchCV]:
    best_pooling = ""
    best_search = None
    for pooling in POOLING_OPTIONS:
        print(f"  Tuning classifier with {pooling} pooling...")
        search = tune_classifier(
            classifier=classifier,
            embeddings=embeddings_by_pooling[pooling],
            labels=labels,
            cv=cv,
        )
        tuning_result_frames.append(grid_search_results(search, model_name, pooling))
        print(f"    {pooling} best grid-search accuracy: {search.best_score_:.4f}")
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_pooling = pooling
            best_search = search

    if best_search is None:
        raise ValueError("No pooling option was tuned.")
    return best_pooling, best_search


def tune_classifier(
    classifier: Pipeline,
    embeddings: np.ndarray,
    labels: pd.Series,
    cv: StratifiedKFold,
) -> GridSearchCV:
    search = GridSearchCV(
        estimator=classifier,
        param_grid=CLASSIFIER_PARAM_GRID,
        scoring="accuracy",
        cv=cv,
        n_jobs=1,
        refit=True,
        verbose=1,
    )
    search.fit(embeddings, labels)
    return search


def grid_search_results(search: GridSearchCV, model_name: str, pooling: str) -> pd.DataFrame:
    results = pd.DataFrame(search.cv_results_)
    results["embedding_model"] = model_name
    results["pooling"] = pooling
    results["model_option"] = model_name
    results["pooling_option"] = pooling
    results["class_weight"] = results["param_classifier__class_weight"].apply(format_parameter_value)
    columns = [
        "model_option",
        "pooling_option",
        "embedding_model",
        "pooling",
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "param_scaler",
        "param_classifier__C",
        "param_classifier__class_weight",
        "class_weight",
    ]
    fold_columns = [
        column for column in results.columns
        if column.startswith("split") and column.endswith("_test_score")
    ]
    available_columns = [column for column in columns + fold_columns if column in results.columns]
    return results[available_columns].copy()


def combine_tuning_results(result_frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not result_frames:
        return pd.DataFrame()
    results = pd.concat(result_frames, ignore_index=True, sort=False)
    return results.sort_values(
        ["mean_test_score", "std_test_score"],
        ascending=[False, True],
    )


def print_tuning_results(results: pd.DataFrame) -> None:
    print("SPECTER2 tuning accuracy by parameter combination:")
    if results.empty:
        print("  No tuning results.")
        return
    for _, row in results.iterrows():
        print(
            "  "
            f"accuracy={row['mean_test_score']:.4f}, "
            f"std={row['std_test_score']:.4f}, "
            f"model_option={row['model_option']}, "
            f"pooling_option={row['pooling_option']}, "
            f"scaler={row['param_scaler']}, "
            f"C={row['param_classifier__C']}, "
            f"class_weight={row['class_weight']}"
        )


def write_tuning_results(results: pd.DataFrame) -> None:
    TUNING_RESULTS_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(TUNING_RESULTS_OUTPUT_CSV, index=False)


def format_parameter_value(value) -> str:
    if pd.isna(value):
        return "None"
    text = str(value).strip()
    if text == "":
        return "None"
    return text


def cross_validate_classifier(
    classifier: Pipeline,
    embeddings: np.ndarray,
    labels: pd.Series,
    cv: StratifiedKFold,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label_values = labels.to_numpy()
    predictions = np.empty(len(labels), dtype=object)
    confidence = np.zeros(len(labels), dtype=float)
    scores = []

    for fold_number, (train_index, test_index) in enumerate(
        cv.split(embeddings, label_values),
        start=1,
    ):
        fold_classifier = clone(classifier)
        fold_classifier.fit(embeddings[train_index], label_values[train_index])

        fold_predictions = fold_classifier.predict(embeddings[test_index])
        fold_confidence = fold_classifier.predict_proba(embeddings[test_index]).max(axis=1)
        predictions[test_index] = fold_predictions
        confidence[test_index] = fold_confidence

        fold_accuracy = accuracy_score(label_values[test_index], fold_predictions)
        scores.append(fold_accuracy)
        print(f"  CV fold {fold_number}/{cv.get_n_splits()}: accuracy {fold_accuracy:.4f}")

    return np.array(scores), predictions, confidence


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
            "specter2_jel_code_1_predicted": predicted_labels,
            "specter2_confidence": pd.Series(confidence).round(4),
            "specter2_prediction_in_jel_code_full": any_code_match.astype(int).to_list(),
        }
    )
    validation.to_csv(VALIDATION_OUTPUT_CSV, index=False)


def write_report(
    validation_accuracy: float,
    validation_accuracy_std: float,
    any_code_accuracy: float,
    validation_report: str,
    label_counts: pd.Series,
    cv_scores: np.ndarray,
    best_model_name: str,
    best_pooling: str,
    best_params: dict,
    grid_search_best_score: float,
    tuning_results: pd.DataFrame,
) -> None:
    REPORT_OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_OUTPUT_TXT.open("w", encoding="utf-8") as file:
        file.write("JEL code first-letter prediction model\n")
        file.write("======================================\n\n")
        file.write("Model: SPECTER2 embeddings + Logistic Regression\n")
        file.write(f"Embedding model options: {MODEL_OPTIONS}\n")
        file.write(f"Best embedding model: {best_model_name}\n")
        file.write(f"Label column: {LABEL_COLUMN}\n")
        file.write(f"Text columns: {TEXT_COLUMNS}\n")
        file.write(f"Max length: {MAX_LENGTH}\n")
        file.write(f"Batch size: {BATCH_SIZE}\n\n")
        file.write(f"Pooling options: {POOLING_OPTIONS}\n")
        file.write(f"Best pooling: {best_pooling}\n\n")
        file.write("Label counts:\n")
        file.write(label_counts.to_string())
        file.write("\n\n")
        file.write("Logistic Regression grid search:\n")
        file.write(f"  Candidate models: {np.prod([len(values) for values in CLASSIFIER_PARAM_GRID.values()])}\n")
        file.write(f"  Best grid-search accuracy: {grid_search_best_score:.4f}\n")
        file.write(f"  Best parameters: {best_params}\n")
        file.write(f"  Tuning results CSV: {TUNING_RESULTS_OUTPUT_CSV}\n")
        file.write("  Tuning results:\n")
        if tuning_results.empty:
            file.write("    No tuning results.\n")
        else:
            file.write(tuning_results.to_string(index=False))
            file.write("\n")
        file.write("\n")
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


if __name__ == "__main__":
    main()
