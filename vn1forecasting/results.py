from datetime import timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def save_predictions_in_custom_format(
    test_predictions: pd.DataFrame, test_samples: List[Dict[str, Any]], output_path: str
) -> pd.DataFrame:
    """
    Save predictions in the desired format:
    unique_id (Client-Warehouse-Product), ds (future dates), pred (predictions).

    Args:
    - test_predictions: DataFrame containing predictions and metadata.
    - test_samples: List of dictionaries containing the test sample metadata (dates).
    - output_path: Path to save the output file (CSV).

    Returns:
    - formatted_df: DataFrame with columns unique_id, ds, and pred.
    """
    results = []

    # Iterate over rows in test_predictions DataFrame
    for index, row in test_predictions.iterrows():
        # Create unique_id
        unique_id = f"{row['Client']}-{row['Warehouse']}-{row['Product']}"

        # Compute future start date from the corresponding test sample
        sample = test_samples[index]
        future_start_date = sample["cursor_date"]

        # Iterate through predictions for the current row
        for i, prediction in enumerate(row["Predictions"]):
            future_date = future_start_date + timedelta(weeks=i)  # Compute future date
            results.append({"unique_id": unique_id, "ds": future_date, "senoni": prediction})

    # Convert to DataFrame
    formatted_df = pd.DataFrame(results)

    # Save to CSV
    formatted_df.to_csv(output_path, index=False)
    print(f"Predictions saved in custom format to {output_path}")

    return formatted_df


def read_and_prepare_data(file_path: str, value_name: str = "y") -> pd.DataFrame:
    """Reads data in wide format and converts it to long format with `unique_id`, `ds`, and `y` columns."""
    df = pd.read_csv(file_path)
    df["unique_id"] = df[["Client", "Warehouse", "Product"]].astype(str).agg("-".join, axis=1)
    df = df.drop(["Client", "Warehouse", "Product"], axis=1)
    df = df.melt(id_vars=["unique_id"], var_name="ds", value_name=value_name)
    df["ds"] = pd.to_datetime(df["ds"])
    return df.sort_values(by=["unique_id", "ds"])


def get_competition_forecasts(forecast_paths: List[tuple[str, str]]) -> pd.DataFrame:
    """Reads competition forecasts and merges them into a single DataFrame."""
    fcst_dfs = [read_and_prepare_data(file_path, place) for file_path, place in forecast_paths]
    merged_df = pd.concat(fcst_dfs, axis=1)
    return merged_df.loc[:, ~merged_df.columns.duplicated()]


def evaluate_forecasts(
    actual_path: str, formatted_df: pd.DataFrame, forecast_paths: List[tuple[str, str]]
) -> pd.DataFrame:
    """
    Evaluate forecasts against actual data and compute scores for each model.

    Args:
        actual_path (str): Path to the actual sales data.
        formatted_df (pd.DataFrame): DataFrame containing `unique_id`, `ds`, and forecast columns.
        forecast_paths (list): List of tuples with forecast file paths and their respective model names.

    Returns:
        pd.DataFrame: DataFrame with models and their scores, sorted by score.
    """
    # Prepare competition forecasts
    fcst_df_comp = get_competition_forecasts(forecast_paths)

    # Merge formatted DataFrame with forecasts
    res = formatted_df.iloc[:, :3].merge(fcst_df_comp, on=["unique_id", "ds"], how="left")

    # Read and prepare actual data
    actual = read_and_prepare_data(actual_path)
    result = actual[["unique_id", "ds", "y"]].merge(res, on=["unique_id", "ds"], how="left")

    # Verify unique_id consistency
    assert set(res["unique_id"].unique()) == set(result["unique_id"].unique()), "Some unique_ids are missing"

    # Compute scores for each model
    scores = {
        model: round(
            (np.nansum(np.abs(result[model] - result["y"])) + np.abs(np.nansum(result[model] - result["y"])))
            / result["y"].sum(),
            4,
        )
        for model in res.columns
        if model not in ["unique_id", "ds"]
    }

    # Create and sort score DataFrame
    score_df = pd.DataFrame(scores.items(), columns=["model", "score"]).sort_values(by="score").reset_index(drop=True)
    return score_df
