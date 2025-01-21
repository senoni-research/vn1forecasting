import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import timedelta


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

            # Create scalers
            sales_scaler = MinMaxScaler()
            price_scaler = MinMaxScaler()
            rolling_scaler = MinMaxScaler()

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


class MultiTimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                 num_wom, num_woy, num_moy, num_qoy, date_embedding_dim, num_clients, num_warehouses, 
                 num_products, category_embedding_dim, dropout=0.1):
        super(MultiTimeSeriesTransformer, self).__init__()

        # Define total embedding dimension
        self.total_embedding_dim = d_model + 4 * date_embedding_dim + 3 * category_embedding_dim

        # Sales Encoder
        self.sales_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.total_embedding_dim, nhead=nhead, 
                                       dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_encoder_layers
        )

        # Price Encoder
        self.price_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.total_embedding_dim, nhead=nhead, 
                                       dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_encoder_layers
        )

        # Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.total_embedding_dim, nhead=nhead, 
                                       dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_decoder_layers
        )

        # Input Embeddings
        self.sales_embedding = nn.Linear(input_dim + 2, d_model)
        self.price_embedding = nn.Linear(input_dim, d_model)
        self.future_embedding = nn.Linear(input_dim, d_model)

        # Date Feature Embeddings
        self.wom_embedding = nn.Embedding(num_wom, date_embedding_dim)
        self.woy_embedding = nn.Embedding(num_woy, date_embedding_dim)
        self.moy_embedding = nn.Embedding(num_moy, date_embedding_dim)
        self.qoy_embedding = nn.Embedding(num_qoy, date_embedding_dim)

        # Categorical Feature Embeddings
        self.client_embedding = nn.Embedding(num_clients, category_embedding_dim)
        self.warehouse_embedding = nn.Embedding(num_warehouses, category_embedding_dim)
        self.product_embedding = nn.Embedding(num_products, category_embedding_dim)

        # Output Layer
        self.output_layer = nn.Linear(self.total_embedding_dim, 1)

    def forward(self, sales, price, decoder_input, wom, woy, moy, qoy, sales_padding_mask, price_padding_mask,
                price_validity_mask, client, warehouse, product, rolling_4w_sales, rolling_13w_sales):
        # Replace NaN values in inputs with neutral values
        sales = torch.nan_to_num(sales, nan=0.)
        price = torch.nan_to_num(price, nan=0.)
        rolling_4w_sales = torch.nan_to_num(rolling_4w_sales, nan=0.)
        rolling_13w_sales = torch.nan_to_num(rolling_13w_sales, nan=0.)

        # Embed time-series inputs
        sales_emb = self.sales_embedding(torch.cat([sales, rolling_4w_sales, rolling_13w_sales], dim=-1))
        price_emb = self.price_embedding(price)
        decoder_emb = self.future_embedding(decoder_input)

        # Embed date features
        wom_emb = self.wom_embedding(wom)
        woy_emb = self.woy_embedding(woy)
        moy_emb = self.moy_embedding(moy)
        qoy_emb = self.qoy_embedding(qoy)

        # Embed categorical features
        client_emb = self.client_embedding(client)
        warehouse_emb = self.warehouse_embedding(warehouse)
        product_emb = self.product_embedding(product)

        # Concatenate embeddings
        cat_emb = torch.cat([client_emb, warehouse_emb, product_emb], dim=-1)  # Combine categorical embeddings

        sales_emb = torch.cat([sales_emb, wom_emb[:, :sales_emb.size(1)], woy_emb[:, :sales_emb.size(1)], 
                               moy_emb[:, :sales_emb.size(1)], qoy_emb[:, :sales_emb.size(1)], 
                               cat_emb.unsqueeze(1).repeat(1, sales_emb.size(1), 1)], dim=-1)
        
        price_emb = torch.cat([price_emb, wom_emb, woy_emb, moy_emb, qoy_emb, 
                               cat_emb.unsqueeze(1).repeat(1, price_emb.size(1), 1)], dim=-1)
        
        decoder_emb = torch.cat([decoder_emb, wom_emb[:, -decoder_emb.size(1):], 
                                 woy_emb[:, -decoder_emb.size(1):], moy_emb[:, -decoder_emb.size(1):], 
                                 qoy_emb[:, -decoder_emb.size(1):], 
                                 cat_emb.unsqueeze(1).repeat(1, decoder_emb.size(1), 1)], dim=-1)

        # Create masks
        final_price_mask = price_padding_mask * price_validity_mask

        # Ensure consistent types for masks
        causal_mask_sales = self.create_causal_mask(sales_emb.size(1)).to(sales_emb.device).bool()
        causal_mask_price = self.create_causal_mask(price_emb.size(1)).to(price_emb.device).bool()
        sales_padding_mask = sales_padding_mask.bool()
        final_price_mask = final_price_mask.bool()

        # Encode sales and price
        sales_encoded = self.sales_encoder(
            sales_emb.permute(1, 0, 2),
            mask=causal_mask_sales,
            src_key_padding_mask=(~sales_padding_mask)
        )
        price_encoded = self.price_encoder(
            price_emb.permute(1, 0, 2),
            mask=causal_mask_price,
            src_key_padding_mask=(~final_price_mask)
        )

        # Combine sales and price encodings for cross-attention
        memory = torch.cat((sales_encoded, price_encoded), dim=0)

        # Decode future sales
        decoder_output = self.decoder(
            decoder_emb.permute(1, 0, 2),
            memory,
            tgt_mask=None,
            tgt_key_padding_mask=None
        )

        # Output predictions
        output = self.output_layer(decoder_output.permute(1, 0, 2))
        future_predictions = output[:, -13:, :]  # Shape: (batch_size, max_future_weeks, 1)
        return future_predictions

    @staticmethod
    def create_causal_mask(seq_len):
        """Create a causal mask to prevent attention to future time steps."""
        return torch.tril(torch.ones(seq_len, seq_len))
    
    @staticmethod
    def masked_loss(predictions, target, loss_fn=nn.MSELoss(reduction='none')):
        """
        Compute masked loss, ignoring NaN values explicitly.
        - predictions: Model predictions (batch_size, seq_len).
        - target: True values (batch_size, seq_len) (may contain NaN).
        - mask: Binary mask indicating valid target values (1 for valid, 0 for NaN).
        - loss_fn: Loss function (default: MSELoss with no reduction).
        """
        # Replace NaNs in the target with zero (they will be ignored by the mask)
        mask = (~torch.isnan(target)).float()
        target = torch.where(mask > 0, target, torch.zeros_like(target))

        # Compute element-wise loss
        elementwise_loss = loss_fn(predictions, target)  # (batch_size, seq_len)

        # Apply the mask to exclude invalid values
        masked_loss = elementwise_loss * mask

        # Normalize by the number of valid entries (add epsilon to avoid division by zero)
        return masked_loss.sum() / (mask.sum() + 1e-8)


def validate_model_with_loss(
    model, val_samples, batch_size, loss_fn, max_past_weeks=52, max_future_weeks=13, device='cpu'
):
    """
    Compute validation loss and predictions.

    Args:
    - model: Trained transformer model.
    - val_samples: List of validation samples (from generate_time_series_samples).
    - batch_size: Batch size for validation.
    - loss_fn: Loss function (e.g., nn.MSELoss with reduction='none').
    - max_past_weeks: Maximum length of historical data.
    - max_future_weeks: Maximum length of future data.

    Returns:
    - val_loss: Average validation loss.
    - all_predictions: Predictions for the entire validation set.
    - all_targets: Ground-truth target values for the entire validation set.
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        # Iterate over batches
        for i in range(0, len(val_samples), batch_size):
            # Extract batch samples
            batch_samples = val_samples[i:i + batch_size]

            # Prepare batch data using the unified function
            batch_data = prepare_batch_data(
                batch_samples, 
                max_past_weeks=max_past_weeks, 
                max_future_weeks=max_future_weeks, 
                mode='train',  # Validation mode includes targets
                device=device
            )

            # Unpack batch data
            sales, price, decoder_input, wom, woy, moy, qoy, sales_padding_mask, price_padding_mask, price_validity_mask, client, warehouse, product, target, rolling_4w_sales, rolling_13w_sales = batch_data

            # Forward pass
            predictions = model(
                sales, price, decoder_input, wom, woy, moy, qoy, 
                sales_padding_mask, price_padding_mask, price_validity_mask,
                client, warehouse, product, rolling_4w_sales, rolling_13w_sales, 
            ).squeeze(-1)

            # Compute loss
            batch_loss = model.masked_loss(predictions, target, loss_fn)
            total_loss += batch_loss.item()

            # Store predictions and targets for metrics
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    # Normalize total loss by number of batches
    val_loss = total_loss / (len(val_samples) / batch_size)

    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return val_loss, all_predictions, all_targets


def run_inference_on_test(model, test_samples, batch_size, preprocessor, device='cpu'):
    """
    Run inference on the test set, inverse-transform predictions, and decode label-encoded variables.

    Args:
    - model: Trained transformer model.
    - test_samples: List of dictionaries containing test data.
    - batch_size: Batch size for inference.
    - preprocessor: Instance of DataPreprocessor to use for inverse-transforming and decoding.

    Returns:
    - predictions: DataFrame with inverse-transformed predictions and original identifiers.
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(0, len(test_samples), batch_size):
            # Prepare batch data
            batch_samples = test_samples[i:i + batch_size]

            # Prepare batch tensors
            batch_data = prepare_batch_data(
                batch_samples,
                max_past_weeks=52,
                max_future_weeks=13,
                mode='test',  # No targets for inference
                device=device
            )

            # Unpack batch data
            sales, price, decoder_input, wom, woy, moy, qoy, sales_padding_mask, price_padding_mask, price_validity_mask, client, warehouse, product, rolling_4w_sales, rolling_13w_sales = batch_data

            # Forward pass
            preds = model(
                sales, price, decoder_input, wom, woy, moy, qoy, 
                sales_padding_mask, price_padding_mask, price_validity_mask,
                client, warehouse, product, rolling_4w_sales, rolling_13w_sales
            ).squeeze(-1)  # Remove last dimension
            preds = preds.cpu().numpy()  # Convert to numpy array

            # Inverse transform predictions and decode label-encoded variables
            for j, prediction in enumerate(preds):
                sample = batch_samples[j]
                product_id = sample['product']  # Retrieve product ID

                # Inverse-transform the predictions
                original_preds, _ = preprocessor.inverse_transform(product_id, sales=prediction)

                # Decode label-encoded variables
                original_client = preprocessor.client_encoder.inverse_transform([sample['Client']])[0]
                original_warehouse = preprocessor.warehouse_encoder.inverse_transform([sample['Warehouse']])[0]
                original_product = preprocessor.product_encoder.inverse_transform([sample['product']])[0]

                # Append results
                predictions.append({
                    'Client': original_client,
                    'Warehouse': original_warehouse,
                    'Product': original_product,
                    'Predictions': original_preds.flatten(),  # Flatten for readability
                })

    # Convert predictions to a DataFrame for easy analysis
    return pd.DataFrame(predictions)


def save_predictions_in_custom_format(test_predictions, test_samples, output_path):
    """
    Save predictions in the desired format:
    unique_id (Client-Warehouse-Product), ds (future dates), pred (predictions).

    Args:
    - test_predictions: DataFrame containing predictions and metadata.
    - test_samples: List of dictionaries containing the test sample metadata (dates).
    - preprocessor: Instance of DataPreprocessor to decode label-encoded variables.
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
        future_start_date = sample['cursor_date']

        # Iterate through predictions for the current row
        for i, prediction in enumerate(row['Predictions']):
            future_date = future_start_date + timedelta(weeks=i)  # Compute future date
            results.append({
                'unique_id': unique_id,
                'ds': future_date,
                'senoni': prediction
            })

    # Convert to DataFrame
    formatted_df = pd.DataFrame(results)

    # Save to CSV
    formatted_df.to_csv(output_path, index=False)
    print(f"Predictions saved in custom format to {output_path}")

    return formatted_df


def plot_predictions_vs_actual_with_price(
    sample, scalers, preprocessor, val_predictions=None, val_targets=None
):
    """
    Plot predictions vs actual sales for a single sequence, along with historical and future prices and sales,
    after inverse-transforming the data using the stored scalers.

    Args:
    - sample: A single sample dictionary containing sales, price, and product information.
    - scalers: Dictionary of normalization scalers for each product.
    - preprocessor: Instance of DataPreprocessor to use its inverse_transform function.
    - val_predictions: Predictions from the model (optional, for plotting future predictions).
    - val_targets: Actual target sales (optional, for comparison with predictions).
    """
    import matplotlib.pyplot as plt

    # Extract product ID and data
    product_id = sample['product']
    sales = sample['sales']
    price = sample['price']
    target = sample['target']  # Use 'target' from the sample if val_targets is not provided
    prediction = val_predictions if val_predictions is not None else None

    # Inverse transform sales, prices, and targets
    original_sales, _ = preprocessor.inverse_transform(product_id=product_id, sales=sales)
    _, original_prices = preprocessor.inverse_transform(product_id=product_id, price=price)
    original_predictions, _ = (
        preprocessor.inverse_transform(product_id=product_id, sales=prediction)
        if prediction is not None
        else (None, None)
    )
    original_targets, _ = (
        preprocessor.inverse_transform(product_id=product_id, sales=target)
        if target is not None
        else (None, None)
    )

    # Create the timeline
    history_len = len(sales)
    future_len = len(prediction) if prediction is not None else (len(target) if target is not None else 0)

    # Plot historical and future price data
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax1.plot(range(history_len), original_prices[:history_len], label='Historical Price', color='red', linestyle='--')
    ax1.plot(range(history_len, history_len + future_len), original_prices[history_len:], label='Future Price', color='red', linestyle='-')
    ax1.set_ylabel("Price", color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    # Plot historical sales
    ax2 = ax1.twinx()
    ax2.plot(range(history_len), original_sales.flatten(), label='Historical Sales', color='blue', linestyle='-')

    # Plot actual and predicted future sales
    if original_targets is not None:
        ax2.plot(range(history_len, history_len + future_len), original_targets.flatten(), label='Actual Future Sales', color='green')
    if original_predictions is not None:
        ax2.plot(range(history_len, history_len + future_len), original_predictions.flatten(), label='Predicted Future Sales', color='orange', linestyle='--')
    ax2.set_ylabel("Sales", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Add labels and legend
    ax1.set_xlabel("Weeks")
    ax1.set_title(f"Predictions vs Actual Sales with Historical and Future Prices (Product ID: {product_id})")
    ax1.grid()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Show the plot
    plt.show()