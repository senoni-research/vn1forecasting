from typing import Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch import Tensor


class SampleDict(TypedDict):
    sales: NDArray[np.float32]
    price: NDArray[np.float32]
    price_validity: NDArray[np.int32]
    wom: NDArray[np.int32]
    woy: NDArray[np.int32]
    moy: NDArray[np.int32]
    qoy: NDArray[np.int32]
    past_length: int
    future_length: int
    target: Optional[NDArray[np.float32]]
    cursor_date: pd.Timestamp
    product: int
    Client: int
    Warehouse: int
    rolling_4w_sales: NDArray[np.float32]
    rolling_13w_sales: NDArray[np.float32]


class DataPreprocessor:
    def __init__(self) -> None:
        self.client_encoder = LabelEncoder()
        self.warehouse_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()
        self.normalization_params: Dict[int, Dict[str, MinMaxScaler]] = {}

    def read_and_prepare_data(self, phase: int, value_name: str, use: str) -> DataFrame:
        """
        Load and prepare data by melting it and adding metadata.
        """
        df = pd.read_csv(f"../data/phase_{phase}_{value_name}.csv")
        df = df.melt(id_vars=["Client", "Warehouse", "Product"], var_name="Date", value_name=value_name)
        df["Date"] = pd.to_datetime(df["Date"])
        df["use"] = use
        return df.sort_values(by=["Client", "Warehouse", "Product", "Date"])

    def remove_leading_zeros(self, group: DataFrame, num_leading_zeros: int = 13) -> DataFrame:
        """Remove leading zeros from sales data."""
        first_nonzero_idx = group[group["Sales"] != 0].index.min()

        if pd.isna(first_nonzero_idx):
            return group

        position_of_first_nonzero = group.index.get_loc(first_nonzero_idx)
        start_pos = max(position_of_first_nonzero - num_leading_zeros, 0)
        keep_index_labels = group.index[start_pos:]

        return group.loc[keep_index_labels]

    def inverse_transform(
        self,
        product_id: int,
        sales: Optional[Union[float, NDArray[np.float32]]] = None,
        price: Optional[Union[float, NDArray[np.float32]]] = None,
    ) -> Tuple[Optional[NDArray[np.float32]], Optional[NDArray[np.float32]]]:
        """Inverse transform normalized values back to original scale."""
        scaler = self.normalization_params[product_id]
        original_sales: Optional[NDArray[np.float32]] = None
        original_price: Optional[NDArray[np.float32]] = None

        if sales is not None:
            scaled_sales = scaler["sales_scaler"].inverse_transform(np.array(sales, dtype=np.float32).reshape(-1, 1))
            original_sales = np.expm1(scaled_sales).astype(np.float32)

        if price is not None:
            original_price = (
                scaler["price_scaler"]
                .inverse_transform(np.array(price, dtype=np.float32).reshape(-1, 1))
                .astype(np.float32)
            )

        return original_sales, original_price

    def fit_minmax_per_product_and_normalize(self, df: DataFrame) -> DataFrame:
        """Normalize data using min-max scaling per product."""

        def normalize_group(group: DataFrame) -> DataFrame:
            product_id = group.name

            group["Sales"] = group["Sales"].apply(lambda x: np.log1p(x) if x >= 0 else 0)

            train_data = group[group["use"] == "train"]
            train_data = pd.concat(
                [
                    train_data,
                    pd.DataFrame({"Sales": [0], "Price": [0], "rolling_4w_sales": [0], "rolling_13w_sales": [0]}),
                ],
                ignore_index=True,
            )

            sales_scaler = MinMaxScaler(feature_range=(0.1, 0.9))
            price_scaler = MinMaxScaler(feature_range=(0.1, 0.9))
            rolling_scaler = MinMaxScaler(feature_range=(0.1, 0.9))

            sales_scaler.fit(train_data[["Sales"]])
            price_scaler.fit(train_data[["Price"]])
            rolling_scaler.fit(train_data[["rolling_4w_sales", "rolling_13w_sales"]])

            self.normalization_params[product_id] = {"sales_scaler": sales_scaler, "price_scaler": price_scaler}

            group["Sales"] = sales_scaler.transform(group[["Sales"]])
            group["Price"] = price_scaler.transform(group[["Price"]])
            group[["rolling_4w_sales", "rolling_13w_sales"]] = rolling_scaler.transform(
                group[["rolling_4w_sales", "rolling_13w_sales"]]
            )

            return group

        return df.groupby("Product", group_keys=False).apply(normalize_group)

    def preprocess_data(self) -> DataFrame:
        """Main preprocessing pipeline."""
        # Read and merge data
        sales_0 = self.read_and_prepare_data(0, "Sales", "train")
        price_0 = self.read_and_prepare_data(0, "Price", "train")
        sales_1 = self.read_and_prepare_data(1, "Sales", "train")
        price_1 = self.read_and_prepare_data(1, "Price", "train")
        sales_2 = self.read_and_prepare_data(2, "Sales", "test")

        sales = pd.concat([sales_0, sales_1, sales_2]).reset_index(drop=True)
        price = pd.concat([price_0, price_1]).reset_index(drop=True)
        df = sales.merge(price, on=["Client", "Warehouse", "Product", "Date", "use"], how="left")
        df = df.sort_values(by=["Client", "Warehouse", "Product", "Date"])

        # Remove leading zeros
        df = df.groupby(["Client", "Warehouse", "Product"], group_keys=False).apply(self.remove_leading_zeros)
        df = df.sort_values(by=["Client", "Warehouse", "Product", "Date"]).reset_index(drop=True)

        # Encode categorical columns
        df["Client"] = self.client_encoder.fit_transform(df["Client"])
        df["Warehouse"] = self.warehouse_encoder.fit_transform(df["Warehouse"])
        df["Product"] = self.product_encoder.fit_transform(df["Product"])

        def calc_rolling_positive_sales(x: Series, window: int) -> Series:
            """Calculate rolling means on positive sales only."""
            positive_sales = x.copy()
            positive_sales[x <= 0] = np.nan
            return positive_sales.shift(1).rolling(window=window, min_periods=1).sum()

        # Calculate rolling means
        df["rolling_4w_sales"] = df.groupby(["Client", "Warehouse", "Product"])["Sales"].transform(
            lambda x: calc_rolling_positive_sales(x, 4)
        )
        df["rolling_13w_sales"] = df.groupby(["Client", "Warehouse", "Product"])["Sales"].transform(
            lambda x: calc_rolling_positive_sales(x, 13)
        )

        # Normalize data
        df["Price"] = df["Price"].replace(0, np.nan)
        df = self.fit_minmax_per_product_and_normalize(df)

        return df


def generate_time_series_samples(
    df: DataFrame,
    n_samples: Optional[int] = None,
    mode: str = "train",
    train_valid_split: float = 0.8,
    max_past_weeks: int = 52,
    max_future_weeks: int = 13,
    phase: str = "init",
) -> Union[Tuple[List[SampleDict], List[SampleDict]], List[SampleDict]]:
    """Generate time-series samples for training, validation, or testing."""
    samples: List[SampleDict] = []
    unique_groups = df[["Client", "Warehouse", "Product"]].drop_duplicates()
    max_date = df["Date"].max()
    min_date = df["Date"].min()

    if mode == "test":
        groups_to_process = unique_groups.iterrows()
        n_samples = len(unique_groups)
    else:
        groups_to_process = (
            unique_groups.sample(n_samples, replace=True).iterrows() if n_samples else unique_groups.iterrows()
        )

    for _, group in groups_to_process:
        group_df = df[
            (df["Client"] == group["Client"])
            & (df["Warehouse"] == group["Warehouse"])
            & (df["Product"] == group["Product"])
        ]

        product_id = group["Product"]

        if mode == "test":
            cursor_date = group_df[group_df["use"] == "test"]["Date"].min()
            if pd.isna(cursor_date):
                continue
        else:
            group_df.loc[group_df["use"] == "test", "Sales"] = np.nan
            valid_cursor_start = min_date + pd.Timedelta(weeks=max_past_weeks)
            valid_cursor_end = max_date - pd.Timedelta(weeks=max_future_weeks)
            valid_dates = group_df["Date"][
                (group_df["Date"] >= valid_cursor_start) & (group_df["Date"] <= valid_cursor_end)
            ].values
            if len(valid_dates) == 0:
                continue
            cursor_date = pd.to_datetime(np.random.choice(valid_dates))

        # Define history and future periods
        history_start = max(min_date, cursor_date - pd.Timedelta(weeks=max_past_weeks))
        history_end = cursor_date - pd.Timedelta(weeks=1)
        future_start = cursor_date
        future_end = min(max_date, cursor_date + pd.Timedelta(weeks=max_future_weeks - 1))

        # Extract historical and future data
        history = group_df[(group_df["Date"] >= history_start) & (group_df["Date"] <= history_end)].sort_values("Date")
        future = group_df[(group_df["Date"] >= future_start) & (group_df["Date"] <= future_end)].sort_values("Date")

        # Ensure there are at least 26 positive sales values in history
        if phase == "init" and (history["Sales"] > 0).sum() < 26 and mode == "train":
            continue

        if mode == "train":
            remove_price_proba = 0.25
            # Randomly replace x% of Price values in history with NaN
            mask_history = np.random.rand(len(history)) < remove_price_proba
            history.loc[mask_history, "Price"] = np.nan

            if np.random.rand() < {"init": 1.0, "core": 1.0, "tune": 0.5, "finish": 0}[phase]:
                # Randomly replace x% of Price values in future with NaN
                mask_future = np.random.rand(len(future)) < remove_price_proba
                future.loc[mask_future, "Price"] = np.nan
            else:
                # Replace all Price values in future with np.nan
                future["Price"] = np.nan
        if mode == "test":
            future["Price"] = np.nan

        # Skip if there's no valid historical data
        if history["Sales"].notna().sum() == 0 and mode == "train":
            continue

        # Prepare data for the sample
        sales_values = history["Sales"].values
        price_values = np.concatenate([history["Price"].values, future["Price"].values])
        price_validity_mask = np.logical_not(np.isnan(price_values)).astype(int)
        target_values = future["Sales"].values if mode == "train" else None

        # Extract and pad date features
        dates = pd.concat([history["Date"], future["Date"]])
        wom = dates.dt.day // 7
        woy = dates.dt.isocalendar().week - 1
        moy = dates.dt.month - 1
        qoy = (dates.dt.month - 1) // 3

        padded_sales = np.full(max_past_weeks, np.nan)
        padded_sales[-len(sales_values) :] = sales_values

        padded_price = np.full(max_past_weeks + max_future_weeks, np.nan)
        padded_price[-len(price_values) :] = price_values

        padded_price_validity = np.zeros(max_past_weeks + max_future_weeks, dtype=int)
        padded_price_validity[-len(price_values) :] = price_validity_mask

        padded_wom = np.zeros(max_past_weeks + max_future_weeks, dtype=int)
        padded_wom[-len(wom) :] = wom.values

        padded_woy = np.zeros(max_past_weeks + max_future_weeks, dtype=int)
        padded_woy[-len(woy) :] = woy.values

        padded_moy = np.zeros(max_past_weeks + max_future_weeks, dtype=int)
        padded_moy[-len(moy) :] = moy.values

        padded_qoy = np.zeros(max_past_weeks + max_future_weeks, dtype=int)
        padded_qoy[-len(qoy) :] = qoy.values

        # Record lengths
        past_length = len(sales_values)
        future_length = len(target_values) if mode == "train" and target_values is not None else max_future_weeks

        # rolling_4w_sales, rolling_13w_sales
        rolling_4w_sales_values = history["rolling_4w_sales"].values
        padded_rolling_4w_sales = np.full(max_past_weeks, np.nan)
        padded_rolling_4w_sales[-len(rolling_4w_sales_values) :] = rolling_4w_sales_values

        rolling_13w_sales_values = history["rolling_13w_sales"].values
        padded_rolling_13w_sales = np.full(max_past_weeks, np.nan)
        padded_rolling_13w_sales[-len(rolling_13w_sales_values) :] = rolling_13w_sales_values

        # Append sample
        samples.append(
            {
                "sales": padded_sales,
                "price": padded_price,
                "price_validity": padded_price_validity,
                "wom": padded_wom,
                "woy": padded_woy,
                "moy": padded_moy,
                "qoy": padded_qoy,
                "past_length": past_length,
                "future_length": future_length,
                "target": target_values,
                "cursor_date": cursor_date,
                "product": product_id,
                "Client": group["Client"],
                "Warehouse": group["Warehouse"],
                "rolling_4w_sales": padded_rolling_4w_sales,
                "rolling_13w_sales": padded_rolling_13w_sales,
            }
        )

    if mode == "train":
        train_samples = [
            sample
            for sample in samples
            if hash(f"{sample['cursor_date']}-{sample['Client']}-{sample['product']}-{sample['Warehouse']}") % 100
            < train_valid_split * 100
        ]
        valid_samples = [
            sample
            for sample in samples
            if hash(f"{sample['cursor_date']}-{sample['Client']}-{sample['product']}-{sample['Warehouse']}") % 100
            >= train_valid_split * 100
        ]
        return train_samples, valid_samples
    else:
        return samples


BatchOutput = Union[
    Tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ],
    Tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ],
]


def prepare_batch_data(
    samples: List[SampleDict],
    max_past_weeks: int = 52,
    max_future_weeks: int = 13,
    decoder_history_length: int = 13,
    mode: str = "train",
    device: str = "cpu",
) -> BatchOutput:
    """
    Prepare batch data for training, validation, or inference.

    Args:
        samples: List of sample dictionaries containing time series data
        max_past_weeks: Maximum number of past weeks to consider
        max_future_weeks: Maximum number of future weeks to predict
        decoder_history_length: Number of past sales values to use
        mode: 'train' or 'test'
        device: PyTorch device to use

    Returns:
        If mode == 'train':
            Tuple of tensors (sales, price, decoder_input, wom, woy, moy, qoy,
                            sales_padding_mask, price_padding_mask, price_validity_mask,
                            client, warehouse, product, target, rolling_4w_sales,
                            rolling_13w_sales)
        If mode != 'train':
            Same tuple without the target tensor
    """
    # Extract batch size
    batch_size = len(samples)

    # Initialize arrays for batch data
    sales_arr: NDArray[np.float32] = np.array([sample["sales"] for sample in samples], dtype=np.float32)
    price_arr: NDArray[np.float32] = np.array([sample["price"] for sample in samples], dtype=np.float32)
    wom_arr: NDArray[np.int32] = np.array([sample["wom"] for sample in samples], dtype=np.int32)
    woy_arr: NDArray[np.int32] = np.array([sample["woy"] for sample in samples], dtype=np.int32)
    moy_arr: NDArray[np.int32] = np.array([sample["moy"] for sample in samples], dtype=np.int32)
    qoy_arr: NDArray[np.int32] = np.array([sample["qoy"] for sample in samples], dtype=np.int32)
    client_arr: NDArray[np.int32] = np.array([sample["Client"] for sample in samples], dtype=np.int32)
    warehouse_arr: NDArray[np.int32] = np.array([sample["Warehouse"] for sample in samples], dtype=np.int32)
    product_arr: NDArray[np.int32] = np.array([sample["product"] for sample in samples], dtype=np.int32)
    price_validity_arr: NDArray[np.int32] = np.array([sample["price_validity"] for sample in samples], dtype=np.int32)
    past_lengths: NDArray[np.int32] = np.array([sample["past_length"] for sample in samples], dtype=np.int32)
    future_lengths: NDArray[np.int32] = np.array([sample["future_length"] for sample in samples], dtype=np.int32)
    rolling_4w_sales_arr: NDArray[np.float32] = np.array(
        [sample["rolling_4w_sales"] for sample in samples], dtype=np.float32
    )
    rolling_13w_sales_arr: NDArray[np.float32] = np.array(
        [sample["rolling_13w_sales"] for sample in samples], dtype=np.float32
    )

    if mode == "train":
        targets_arr: NDArray[np.float32] = np.array(
            [sample["target"] for sample in samples if sample["target"] is not None], dtype=np.float32
        )

    # Extract decoder inputs
    decoder_inputs_list: List[NDArray[np.float32]] = []
    for sample in samples:
        last_sales = sample["sales"][-decoder_history_length:]
        last_sales = np.nan_to_num(last_sales, nan=0.0)
        decoder_inputs_list.append(last_sales)
    decoder_inputs_arr: NDArray[np.float32] = np.array(decoder_inputs_list, dtype=np.float32)

    # Convert to tensors
    sales: Tensor = torch.FloatTensor(sales_arr).unsqueeze(-1).to(device)
    price: Tensor = torch.FloatTensor(price_arr).unsqueeze(-1).to(device)
    wom: Tensor = torch.LongTensor(wom_arr).to(device)
    woy: Tensor = torch.LongTensor(woy_arr).to(device)
    moy: Tensor = torch.LongTensor(moy_arr).to(device)
    qoy: Tensor = torch.LongTensor(qoy_arr).to(device)
    client: Tensor = torch.LongTensor(client_arr).to(device)
    warehouse: Tensor = torch.LongTensor(warehouse_arr).to(device)
    product: Tensor = torch.LongTensor(product_arr).to(device)
    price_validity_mask: Tensor = torch.FloatTensor(price_validity_arr).to(device)
    rolling_4w_sales: Tensor = torch.FloatTensor(rolling_4w_sales_arr).unsqueeze(-1).to(device)
    rolling_13w_sales: Tensor = torch.FloatTensor(rolling_13w_sales_arr).unsqueeze(-1).to(device)
    decoder_inputs: Tensor = torch.FloatTensor(decoder_inputs_arr).unsqueeze(-1).to(device)

    # Padding masks
    sales_padding_mask: Tensor = torch.zeros(batch_size, max_past_weeks).to(device)
    price_padding_mask: Tensor = torch.zeros(batch_size, max_past_weeks + max_future_weeks).to(device)

    for i in range(batch_size):
        sales_padding_mask[i, max_past_weeks - past_lengths[i] :] = 1
        price_padding_mask[i, max_past_weeks - past_lengths[i] : max_past_weeks + future_lengths[i]] = 1

    # Prepare decoder input
    future_decoder_input: Tensor = torch.zeros(batch_size, max_future_weeks, 1).to(device)
    # decoder_input = torch.cat([decoder_inputs, future_decoder_input], dim=1)
    decoder_input: Tensor = future_decoder_input

    if mode == "train":
        target: Tensor = torch.FloatTensor(targets_arr).to(device)
        return (
            sales,
            price,
            decoder_input,
            wom,
            woy,
            moy,
            qoy,
            sales_padding_mask,
            price_padding_mask,
            price_validity_mask,
            client,
            warehouse,
            product,
            target,
            rolling_4w_sales,
            rolling_13w_sales,
        )
    else:
        return (
            sales,
            price,
            decoder_input,
            wom,
            woy,
            moy,
            qoy,
            sales_padding_mask,
            price_padding_mask,
            price_validity_mask,
            client,
            warehouse,
            product,
            rolling_4w_sales,
            rolling_13w_sales,
        )
