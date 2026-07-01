from pathlib import Path
import os

import pandas as pd


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

TRAINING_INPUT_CSV = Path("data/trainingmodel/JEL_Training_Data_With_JEL.csv")
PREDICTION_INPUT_CSV = Path("data/trainingmodel/JEL_Training_Data_Without_JEL.csv")

OUTPUT_WITH_OBSERVED_AND_PREDICTED = Path(
    "data/trainingmodel/JEL_Training_Data_With_Observed_And_Predicted.csv"
)
OUTPUT_WITHOUT_JEL_PREDICTED = Path(
    "data/trainingmodel/JEL_Training_Data_Without_JEL_Predicted.csv"
)
OUTPUT_COMPLETE = Path(
    "data/trainingmodel/JEL_Training_Data_Complete_Observed_And_Predicted.csv"
)

REQUIRED_WITHOUT_JEL_PREDICTION_COLUMNS = [
    "tfidf_predicted_jel_code_full",
    "tfidf_max_confidence",
    "specter2_predicted_jel_code_full",
    "specter2_max_confidence",
    "scibert_predicted_jel_code_full",
    "scibert_max_confidence",
    "ensemble_predicted_jel_code_full",
    "ensemble_max_confidence",
]


MODEL_FILES = [
    {
        "model": "tfidf",
        "validation_csv": Path("data/trainingmodel/JEL_Codes_Multi_TFIDF_Validation_Predictions.csv"),
        "combined_csv": Path("data/trainingmodel/JEL_Training_Data_With_Observed_And_Predicted_Multi_TFIDF.csv"),
        "prediction_csv": Path("data/trainingmodel/JEL_Training_Data_Without_JEL_Predicted_Multi_TFIDF.csv"),
        "prediction_column": "jel_code_full_predicted_tfidf",
        "confidence_column": "jel_code_full_predicted_tfidf_max_confidence",
        "validation_confidence_column": "tfidf_max_confidence",
        "source_column": "jel_code_full_tfidf_source",
    },
    {
        "model": "specter2",
        "validation_csv": Path("data/trainingmodel/JEL_Codes_Multi_SPECTER2_Validation_Predictions.csv"),
        "combined_csv": Path("data/trainingmodel/JEL_Training_Data_With_Observed_And_Predicted_Multi_SPECTER2.csv"),
        "prediction_csv": Path("data/trainingmodel/JEL_Training_Data_Without_JEL_Predicted_Multi_SPECTER2.csv"),
        "prediction_column": "jel_code_full_predicted_specter2",
        "confidence_column": "jel_code_full_predicted_specter2_max_confidence",
        "validation_confidence_column": "specter2_max_confidence",
        "source_column": "jel_code_full_specter2_source",
    },
    {
        "model": "scibert",
        "validation_csv": Path("data/trainingmodel/JEL_Codes_Multi_SciBERT_Validation_Predictions.csv"),
        "combined_csv": Path("data/trainingmodel/JEL_Training_Data_With_Observed_And_Predicted_Multi_SciBERT.csv"),
        "prediction_csv": Path("data/trainingmodel/JEL_Training_Data_Without_JEL_Predicted_Multi_SciBERT.csv"),
        "prediction_column": "jel_code_full_predicted_scibert",
        "confidence_column": "jel_code_full_predicted_scibert_max_confidence",
        "validation_confidence_column": "scibert_max_confidence",
        "source_column": "jel_code_full_scibert_source",
    },
    {
        "model": "ensemble",
        "validation_csv": Path("data/trainingmodel/JEL_Codes_Multi_Ensemble_Validation_Predictions.csv"),
        "combined_csv": Path("data/trainingmodel/JEL_Training_Data_With_Observed_And_Predicted_Multi_Ensemble.csv"),
        "prediction_csv": Path("data/trainingmodel/JEL_Training_Data_Without_JEL_Predicted_Multi_Ensemble.csv"),
        "prediction_column": "jel_code_full_predicted_ensemble",
        "confidence_column": "jel_code_full_predicted_ensemble_confidence",
        "validation_confidence_column": "jel_code_full_predicted_ensemble_confidence",
        "source_column": "jel_code_full_ensemble_source",
    },
]


def main() -> None:
    observed = read_data(TRAINING_INPUT_CSV)
    without_jel = read_data(PREDICTION_INPUT_CSV)

    observed["validation_row_id"] = observed.index.astype(str)
    without_jel["prediction_row_id"] = without_jel.index.astype(str)

    observed_output = observed.copy()
    without_jel_output = without_jel.copy()

    print("Cleaning multi-label predicted JEL outputs:")
    print(f"  Observed input rows: {len(observed_output)}")
    print(f"  Without-JEL input rows: {len(without_jel_output)}")

    for model_spec in MODEL_FILES:
        model_name = model_spec["model"]
        observed_model = observed_predictions_for_model(model_spec, expected_rows=len(observed_output))
        prediction_model = prediction_rows_for_model(model_spec, expected_rows=len(without_jel_output))

        observed_output = observed_output.merge(
            observed_model,
            on="validation_row_id",
            how="left",
        )
        without_jel_output = without_jel_output.merge(
            prediction_model,
            on="prediction_row_id",
            how="left",
        )

        print(f"  {model_name} observed prediction rows merged: {len(observed_model)}")
        print(f"  {model_name} without-JEL prediction rows merged: {len(prediction_model)}")

    observed_output = observed_output.drop(columns=["validation_row_id"])
    without_jel_output = without_jel_output.drop(columns=["prediction_row_id"])
    require_columns(
        without_jel_output,
        REQUIRED_WITHOUT_JEL_PREDICTION_COLUMNS,
        OUTPUT_WITHOUT_JEL_PREDICTED,
    )

    OUTPUT_WITH_OBSERVED_AND_PREDICTED.parent.mkdir(parents=True, exist_ok=True)
    observed_output.to_csv(OUTPUT_WITH_OBSERVED_AND_PREDICTED, index=False)
    without_jel_output.to_csv(OUTPUT_WITHOUT_JEL_PREDICTED, index=False)
    complete_output = combine_observed_and_predicted(observed_output, without_jel_output)
    complete_output.to_csv(OUTPUT_COMPLETE, index=False)

    print()
    print("Final multi-label prediction datasets written:")
    print(f"  Observed and predicted CSV: {OUTPUT_WITH_OBSERVED_AND_PREDICTED}")
    print(f"  Rows: {len(observed_output)}")
    print(f"  Columns: {len(observed_output.columns)}")
    print(f"  Without-JEL predicted CSV: {OUTPUT_WITHOUT_JEL_PREDICTED}")
    print(f"  Rows: {len(without_jel_output)}")
    print(f"  Columns: {len(without_jel_output.columns)}")
    print(f"  Complete observed/predicted CSV: {OUTPUT_COMPLETE}")
    print(f"  Rows: {len(complete_output)}")
    print(f"  Columns: {len(complete_output.columns)}")
    print("  Required without-JEL prediction columns:")
    for column in REQUIRED_WITHOUT_JEL_PREDICTION_COLUMNS:
        print(f"    {column}")


def read_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path)


def combine_observed_and_predicted(
    observed_output: pd.DataFrame,
    without_jel_output: pd.DataFrame,
) -> pd.DataFrame:
    observed = observed_output.copy()
    without_jel = without_jel_output.copy()

    observed["jel_observed"] = 1
    without_jel["jel_observed"] = 0

    columns = list(observed.columns)
    for column in without_jel.columns:
        if column not in columns:
            columns.append(column)

    return pd.concat(
        [
            observed.reindex(columns=columns),
            without_jel.reindex(columns=columns),
        ],
        ignore_index=True,
        sort=False,
    )


def validation_predictions_for_model(model_spec: dict[str, object]) -> pd.DataFrame:
    model_name = str(model_spec["model"])
    path = Path(model_spec["validation_csv"])
    data = read_data(path)

    required_columns = ["validation_row_id", str(model_spec["prediction_column"])]
    require_columns(data, required_columns, path)

    selected = pd.DataFrame()
    selected["validation_row_id"] = data["validation_row_id"].astype(str)
    selected[f"{model_name}_predicted_jel_code_full"] = data[str(model_spec["prediction_column"])]

    confidence_column = str(model_spec["validation_confidence_column"])
    if confidence_column in data.columns:
        selected[f"{model_name}_max_confidence"] = data[confidence_column]

    return selected


def observed_predictions_for_model(model_spec: dict[str, object], expected_rows: int) -> pd.DataFrame:
    combined_path = Path(model_spec["combined_csv"])
    if combined_path.exists():
        combined = read_data(combined_path)
        has_full_observed_predictions = (
            str(model_spec["prediction_column"]) in combined.columns
            and str(model_spec["confidence_column"]) in combined.columns
            and str(model_spec["source_column"]) in combined.columns
            and combined[str(model_spec["source_column"])].eq("observed_with_model_prediction").any()
        )
        if has_full_observed_predictions:
            return observed_predictions_from_combined_file(
                combined=combined,
                model_spec=model_spec,
                expected_rows=expected_rows,
                path=combined_path,
            )

    return validation_predictions_for_model(model_spec)


def observed_predictions_from_combined_file(
    combined: pd.DataFrame,
    model_spec: dict[str, object],
    expected_rows: int,
    path: Path,
) -> pd.DataFrame:
    model_name = str(model_spec["model"])
    prediction_column = str(model_spec["prediction_column"])
    confidence_column = str(model_spec["confidence_column"])

    if len(combined) < expected_rows:
        raise ValueError(f"{path} has fewer rows than the observed input file.")

    observed_rows = combined.head(expected_rows).copy()
    selected = pd.DataFrame()
    selected["validation_row_id"] = observed_rows.index.astype(str)
    selected[f"{model_name}_predicted_jel_code_full"] = observed_rows[prediction_column]
    selected[f"{model_name}_max_confidence"] = observed_rows[confidence_column]

    return selected


def prediction_rows_for_model(model_spec: dict[str, object], expected_rows: int) -> pd.DataFrame:
    model_name = str(model_spec["model"])
    path = Path(model_spec["prediction_csv"])
    data = read_data(path)

    if len(data) != expected_rows:
        raise ValueError(
            f"{path} has {len(data)} rows, but the base without-JEL file has {expected_rows} rows."
        )

    required_columns = [str(model_spec["prediction_column"])]
    require_columns(data, required_columns, path)

    selected = pd.DataFrame()
    selected["prediction_row_id"] = data.index.astype(str)
    selected[f"{model_name}_predicted_jel_code_full"] = data[str(model_spec["prediction_column"])]

    confidence_column = str(model_spec["confidence_column"])
    if confidence_column in data.columns:
        selected[f"{model_name}_max_confidence"] = data[confidence_column]

    source_column = str(model_spec["source_column"])
    if source_column in data.columns:
        selected[f"{model_name}_source"] = data[source_column]

    return selected


def require_columns(data: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing_columns = [column for column in columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {missing_columns}")


if __name__ == "__main__":
    main()
