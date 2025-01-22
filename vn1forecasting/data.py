import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch

class DataPreprocessor:
    def __init__(self):
        self.client_encoder = LabelEncoder()
        self.warehouse_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()
        self.normalization_params = {}

    def read_and_prepare_data(self, phase, value_name, use):
        """
        Load and prepare data by melting it and adding metadata.

        Args:
        - phase: 0, 1, or 2
        - value_name: 'Sales' or 'Price'
        - use: 'train' or 'test'

        Returns:
        - DataFrame: Prepared data.
        """
        df = pd.read_csv(f"../data/phase_{phase}_{value_name}.csv")
        df = df.melt(id_vars=["Client", "Warehouse", "Product"], var_name="Date", value_name=value_name)
        df["Date"] = pd.to_datetime(df["Date"])
        df["use"] = use
        return df.sort_values(by=["Client", "Warehouse", "Product", "Date"])

    def remove_leading_zeros(self, group, num_leading_zeros=13):
        """
        Retain up to `num_leading_zeros` zeros prior to the first nonzero sale.
        
        Args:
            group (DataFrame): Grouped DataFrame for a specific (Client, Warehouse, Product).
            num_leading_zeros (int): How many leading zeros to keep before the first nonzero row.
        
        Returns:
            DataFrame: Group with partial retention of leading zeros.
        """
        first_nonzero_idx = group[group['Sales'] != 0].index.min()
        
        # If the product never sells, return as is
        if pd.isna(first_nonzero_idx):
            return group  
        
        # Get the integer position of the first nonzero index within this group
        position_of_first_nonzero = group.index.get_loc(first_nonzero_idx)
        
        # Compute new start position, ensuring we don't go below 0
        start_pos = max(position_of_first_nonzero - num_leading_zeros, 0)
        
        # Convert that position back into the actual group index label
        keep_index_labels = group.index[start_pos:]
        
        # Return the subset of the group from 'keep_index_labels' onward
        return group.loc[keep_index_labels]

    def fit_minmax_per_product_and_normalize(self, df):
        """
        Normalize Sales and Price using MinMaxScaler per product.

        Args:
        - df: DataFrame with 'Sales', 'Price', and 'Product' columns.

        Returns:
        - df: Normalized DataFrame.
        - normalization_params: Dictionary with scalers for each product.
        """
        def normalize_group(group):
            product_id = group.name

            # Apply log transformation to Sales
            group['Sales'] = group['Sales'].apply(lambda x: np.log1p(x) if x >= 0 else 0)

            train_data = group[group['use'] == 'train']
            train_data = pd.concat([train_data, pd.DataFrame({'Sales': [0], 'Price': [0], 'rolling_4w_sales': [0], 'rolling_4w_sales': [0]})], ignore_index=True)

            # Create scalers
            sales_scaler = MinMaxScaler(feature_range=(0.1, 0.9))
            price_scaler = MinMaxScaler(feature_range=(0.1, 0.9))
            rolling_scaler = MinMaxScaler(feature_range=(0.1, 0.9))

            # Fit scalers on training data
            sales_scaler.fit(train_data[['Sales']])
            price_scaler.fit(train_data[['Price']])
            rolling_scaler.fit(train_data[['rolling_4w_sales', 'rolling_13w_sales']])

            # Store scalers
            self.normalization_params[product_id] = {
                'sales_scaler': sales_scaler,
                'price_scaler': price_scaler
            }

            # Transform both training and test data
            group['Sales'] = sales_scaler.transform(group[['Sales']])
            group['Price'] = price_scaler.transform(group[['Price']])
            group[['rolling_4w_sales', 'rolling_13w_sales']] = rolling_scaler.transform(group[['rolling_4w_sales', 'rolling_13w_sales']])

            return group

        # Normalize each group by Product
        df = df.groupby('Product', group_keys=False).apply(normalize_group)
        return df

    def inverse_transform(self, product_id, sales=None, price=None):
        """
        Perform inverse transformation for Sales and/or Price.

        Args:
        - product_id: Product ID for scaler lookup.
        - sales: Normalized sales values to inverse transform.
        - price: Normalized price values to inverse transform.

        Returns:
        - original_sales: Inverse transformed Sales (if sales provided).
        - original_price: Inverse transformed Price (if price provided).
        """
        scaler = self.normalization_params[product_id]
        original_sales, original_price = None, None

        if sales is not None:
            scaled_sales = scaler['sales_scaler'].inverse_transform(np.array(sales).reshape(-1, 1))
            original_sales = np.expm1(scaled_sales)  # Reverse log1p

        if price is not None:
            original_price = scaler['price_scaler'].inverse_transform(np.array(price).reshape(-1, 1))

        return original_sales, original_price
        
    def preprocess_data(self):
        """
        Main preprocessing pipeline to load, clean, encode, and normalize data.

        Returns:
        - df: Final preprocessed DataFrame.
        """
        # Read and merge data
        sales_0 = self.read_and_prepare_data(0, 'Sales', 'train')
        price_0 = self.read_and_prepare_data(0, 'Price', 'train')
        sales_1 = self.read_and_prepare_data(1, 'Sales', 'train')
        price_1 = self.read_and_prepare_data(1, 'Price', 'train')
        sales_2 = self.read_and_prepare_data(2, 'Sales', 'test')

        sales = pd.concat([sales_0, sales_1, sales_2]).reset_index(drop=True)
        price = pd.concat([price_0, price_1]).reset_index(drop=True)
        df = sales.merge(price, on=["Client", "Warehouse", "Product", "Date", "use"], how="left")
        df = df.sort_values(by=["Client", "Warehouse", "Product", "Date"])

        # Remove leading zeros
        df = df.groupby(['Client', 'Warehouse', 'Product'], group_keys=False).apply(self.remove_leading_zeros)
        df = df.sort_values(by=["Client", "Warehouse", "Product", "Date"]).reset_index(drop=True)

        # Encode categorical columns
        df['Client'] = self.client_encoder.fit_transform(df['Client'])
        df['Warehouse'] = self.warehouse_encoder.fit_transform(df['Warehouse'])
        df['Product'] = self.product_encoder.fit_transform(df['Product'])

        # Calculate rolling means on positive sales only
        def calc_rolling_positive_sales(x, window):
            # Create mask for positive sales
            positive_sales = x.copy()
            positive_sales[x <= 0] = np.nan
            # Calculate rolling sum on positive sales only
            return positive_sales.shift(1).rolling(window=window, min_periods=1).sum()

        # last 4 weeks' sales (excluding current week, positive sales only)
        df['rolling_4w_sales'] = df.groupby(
            ['Client', 'Warehouse', 'Product']
        )['Sales'].transform(lambda x: calc_rolling_positive_sales(x, 4))

        # last 13 weeks' sales (excluding current week, positive sales only)
        df['rolling_13w_sales'] = df.groupby(
            ['Client', 'Warehouse', 'Product']
        )['Sales'].transform(lambda x: calc_rolling_positive_sales(x, 13))

        # Normalize data
        df['Price'] = df['Price'].replace(0, np.nan)
        df = self.fit_minmax_per_product_and_normalize(df)

        return df


def generate_time_series_samples(
    df, 
    n_samples=None, 
    mode='train', 
    train_valid_split=0.8, 
    max_past_weeks=52, 
    max_future_weeks=13,
    phase='init'
):
    """
    Generate time-series samples for training, validation, or testing.

    Args:
    - df: Preprocessed DataFrame containing 'Client', 'Warehouse', 'Product', 'Date', 'Sales', and 'Price'.
    - n_samples: Number of samples to generate (only used for training/validation).
    - mode: 'train' or 'test'. Determines behavior for sample generation.
    - train_valid_split: Proportion of training samples if mode='train'.
    - max_past_weeks: Maximum length of historical data.
    - max_future_weeks: Maximum length of future data.

    Returns:
    - train_samples: Tuple of arrays (only if mode='train').
    - valid_samples: Tuple of arrays (only if mode='train').
    - test_samples: List of dictionaries (only if mode='test').
    """
    samples = []  # Store generated samples
    unique_groups = df[['Client', 'Warehouse', 'Product']].drop_duplicates()
    max_date = df['Date'].max()
    min_date = df['Date'].min()

    if mode == 'test':
        # In test mode, process all unique groups
        groups_to_process = unique_groups.iterrows()
        n_samples = len(unique_groups)  # Process all combinations
    else:
        # For train/validation, randomly sample groups
        groups_to_process = (
            unique_groups.sample(n_samples, replace=True).iterrows()
            if n_samples else unique_groups.iterrows()
        )

    for _, group in groups_to_process:
        # Extract data for the selected group
        group_df = df[
            (df['Client'] == group['Client']) & 
            (df['Warehouse'] == group['Warehouse']) & 
            (df['Product'] == group['Product'])
        ]

        product_id = group['Product']

        if mode == 'test':
            # Test mode: Fixed cursor date is one week before test start date
            cursor_date = group_df[group_df['use'] == 'test']['Date'].min()
            if pd.isna(cursor_date):
                continue
        else:
            # Train mode: Randomly sample cursor date within valid range
            group_df.loc[group_df['use'] == 'test', 'Sales'] = np.nan
            valid_cursor_start = min_date + pd.Timedelta(weeks=max_past_weeks)
            valid_cursor_end = max_date - pd.Timedelta(weeks=max_future_weeks)
            valid_dates = group_df['Date'][
                (group_df['Date'] >= valid_cursor_start) & (group_df['Date'] <= valid_cursor_end)
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
        history = group_df[(group_df['Date'] >= history_start) & (group_df['Date'] <= history_end)].sort_values('Date')
        future = group_df[(group_df['Date'] >= future_start) & (group_df['Date'] <= future_end)].sort_values('Date')

        # Ensure there are at least 26 positive sales values in history
        if phase == 'init' and (history['Sales'] > 0).sum() < 26 and mode == 'train':
                continue

        if mode == 'train':
            remove_price_proba = 0.25
            # Randomly replace x% of Price values in history with NaN
            mask_history = np.random.rand(len(history)) < remove_price_proba
            history.loc[mask_history, 'Price'] = np.nan

            if np.random.rand() < {'init': 1.0, 'core': 1.0, 'tune': 0.5, 'finish': 0}[phase]:
                # Randomly replace x% of Price values in future with NaN
                mask_future = np.random.rand(len(future)) < remove_price_proba
                future.loc[mask_future, 'Price'] = np.nan
            else:
                # Replace all Price values in future with np.nan
                future['Price'] = np.nan
        if mode == 'test':
            future['Price'] = np.nan

        # Skip if there's no valid historical data
        if history['Sales'].notna().sum() == 0 and mode == 'train':
            continue

        # Prepare data for the sample
        sales_values = history['Sales'].values
        price_values = np.concatenate([history['Price'].values, future['Price'].values])
        price_validity_mask = np.logical_not(np.isnan(price_values)).astype(int)
        target_values = future['Sales'].values if mode == 'train' else None

        # Extract and pad date features
        dates = pd.concat([history['Date'], future['Date']])
        wom = dates.dt.day // 7
        woy = dates.dt.isocalendar().week - 1
        moy = dates.dt.month - 1
        qoy = (dates.dt.month - 1) // 3

        padded_sales = np.full(max_past_weeks, np.nan)
        padded_sales[-len(sales_values):] = sales_values

        padded_price = np.full(max_past_weeks + max_future_weeks, np.nan)
        padded_price[-len(price_values):] = price_values

        padded_price_validity = np.zeros(max_past_weeks + max_future_weeks, dtype=int)
        padded_price_validity[-len(price_values):] = price_validity_mask

        padded_wom = np.zeros(max_past_weeks + max_future_weeks, dtype=int)
        padded_wom[-len(wom):] = wom.values

        padded_woy = np.zeros(max_past_weeks + max_future_weeks, dtype=int)
        padded_woy[-len(woy):] = woy.values

        padded_moy = np.zeros(max_past_weeks + max_future_weeks, dtype=int)
        padded_moy[-len(moy):] = moy.values

        padded_qoy = np.zeros(max_past_weeks + max_future_weeks, dtype=int)
        padded_qoy[-len(qoy):] = qoy.values

        # Record lengths
        past_length = len(sales_values)
        future_length = len(target_values) if mode == 'train' else max_future_weeks

        # rolling_4w_sales, rolling_13w_sales
        rolling_4w_sales_values = history['rolling_4w_sales'].values
        padded_rolling_4w_sales = np.full(max_past_weeks, np.nan)
        padded_rolling_4w_sales[-len(rolling_4w_sales_values):] = rolling_4w_sales_values

        rolling_13w_sales_values = history['rolling_13w_sales'].values
        padded_rolling_13w_sales = np.full(max_past_weeks, np.nan)
        padded_rolling_13w_sales[-len(rolling_13w_sales_values):] = rolling_13w_sales_values

        # Append sample
        samples.append({
            'sales': padded_sales,
            'price': padded_price,
            'price_validity': padded_price_validity,
            'wom': padded_wom,
            'woy': padded_woy,
            'moy': padded_moy,
            'qoy': padded_qoy,
            'past_length': past_length,
            'future_length': future_length,
            'target': target_values,
            'cursor_date': cursor_date,
            'product': product_id,
            'Client': group['Client'],
            'Warehouse': group['Warehouse'],
            'rolling_4w_sales': padded_rolling_4w_sales,
            'rolling_13w_sales': padded_rolling_13w_sales,
        })

    if mode == 'train':
        # Split into training and validation sets
        train_samples = [
            sample for sample in samples
            if hash(f"{sample['cursor_date']}-{sample['Client']}-{sample['product']}-{sample['Warehouse']}") % 100 < train_valid_split * 100
        ]
        valid_samples = [
            sample for sample in samples
            if hash(f"{sample['cursor_date']}-{sample['Client']}-{sample['product']}-{sample['Warehouse']}") % 100 >= train_valid_split * 100
        ]
        return train_samples, valid_samples
    else:
        # Return test samples
        return samples


def prepare_batch_data(
    samples, 
    max_past_weeks=52, 
    max_future_weeks=13, 
    decoder_history_length=13,  # Number of past sales values to use
    mode='train',
    device='cpu'
):
    """
    Prepare batch data for training, validation, or inference from generated samples.
    """
    # Extract batch size
    batch_size = len(samples)

    # Initialize arrays for batch data
    sales = np.array([sample['sales'] for sample in samples])
    price = np.array([sample['price'] for sample in samples])
    wom = np.array([sample['wom'] for sample in samples])
    woy = np.array([sample['woy'] for sample in samples])
    moy = np.array([sample['moy'] for sample in samples])
    qoy = np.array([sample['qoy'] for sample in samples])
    client = np.array([sample['Client'] for sample in samples])
    warehouse = np.array([sample['Warehouse'] for sample in samples])
    product = np.array([sample['product'] for sample in samples])
    price_validity = np.array([sample['price_validity'] for sample in samples])
    past_lengths = np.array([sample['past_length'] for sample in samples])
    future_lengths = np.array([sample['future_length'] for sample in samples])
    targets = np.array([sample['target'] for sample in samples]) if mode == 'train' else None
    rolling_4w_sales = np.array([sample['rolling_4w_sales'] for sample in samples])
    rolling_13w_sales = np.array([sample['rolling_13w_sales'] for sample in samples])

    # Extract the last `decoder_history_length` sales values for decoder input
    decoder_inputs = []
    for sample in samples:
        last_sales = sample['sales'][-decoder_history_length:]
        last_sales = np.nan_to_num(last_sales, nan=0.)
        decoder_inputs.append(last_sales)
    decoder_inputs = np.array(decoder_inputs)  # Shape: (batch_size, decoder_history_length)

    # Prepare tensors
    sales = torch.FloatTensor(sales).unsqueeze(-1).to(device)
    price = torch.FloatTensor(price).unsqueeze(-1).to(device)
    wom = torch.LongTensor(wom).to(device)
    woy = torch.LongTensor(woy).to(device)
    moy = torch.LongTensor(moy).to(device)
    qoy = torch.LongTensor(qoy).to(device)
    client = torch.LongTensor(client).to(device)
    warehouse = torch.LongTensor(warehouse).to(device)
    product = torch.LongTensor(product).to(device)
    price_validity_mask = torch.FloatTensor(price_validity).to(device)
    rolling_4w_sales = torch.FloatTensor(rolling_4w_sales).unsqueeze(-1).to(device)
    rolling_13w_sales = torch.FloatTensor(rolling_13w_sales).unsqueeze(-1).to(device)

    # Prepare decoder inputs
    decoder_inputs = torch.FloatTensor(decoder_inputs).unsqueeze(-1).to(device)  # Shape: (batch_size, decoder_history_length, 1)

    # Padding masks
    sales_padding_mask = torch.zeros(batch_size, max_past_weeks).to(device)
    price_padding_mask = torch.zeros(batch_size, max_past_weeks + max_future_weeks).to(device)
    for i in range(batch_size):
        sales_padding_mask[i, max_past_weeks - past_lengths[i]:] = 1
        price_padding_mask[i, max_past_weeks - past_lengths[i]:max_past_weeks + future_lengths[i]] = 1

    # Prepare decoder input based on the chosen strategy
    future_decoder_input = torch.zeros(batch_size, max_future_weeks, 1).to(device)
    # decoder_input = torch.cat([decoder_inputs, future_decoder_input], dim=1)  # Shape: (batch_size, decoder_history_length + max_future_weeks, 1)
    decoder_input = future_decoder_input
    if mode == 'train':
        target = torch.FloatTensor(targets).to(device)
        return (
            sales, price, decoder_input, wom, woy, moy, qoy, 
            sales_padding_mask, price_padding_mask, price_validity_mask,
            client, warehouse, product, target, rolling_4w_sales, rolling_13w_sales
        )
    else:
        return (
            sales, price, decoder_input, wom, woy, moy, qoy, 
            sales_padding_mask, price_padding_mask, price_validity_mask,
            client, warehouse, product, rolling_4w_sales, rolling_13w_sales
        )
