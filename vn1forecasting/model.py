import torch
import torch.nn as nn

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

    @staticmethod
    def masked_competition_loss(predictions, target):
        """
        Custom loss matching the competition metric:
        
            score = ( sum |pred_i - true_i| + | sum(pred_i - true_i) | ) / sum(true_i)
        
        - Ignores NaN target values (via a mask).
        - predictions: (batch_size, seq_len, 1) or (batch_size, seq_len)
        - target: (batch_size, seq_len)
        """

        # Squeeze last dim if predictions are (B, T, 1)
        if predictions.dim() == 3 and predictions.size(-1) == 1:
            predictions = predictions.squeeze(-1)  # (B, T)

        # Create mask for valid target entries
        mask = ~torch.isnan(target)

        # Replace NaNs in target with zero
        target_filled = torch.where(mask, target, torch.zeros_like(target))

        # Also zero out predictions in invalid places so they don't affect sums
        pred_filled = torch.where(mask, predictions, torch.zeros_like(predictions))

        # 1) Sum of absolute errors
        sum_abs_errors = (pred_filled - target_filled).abs().sum()

        # 2) Net error (then take absolute value)
        net_error = (pred_filled - target_filled).sum().abs()

        # 3) Denominator: sum of actuals (only valid entries)
        sum_of_actuals = target_filled.sum()

        # 4) Combine
        total_error = sum_abs_errors + net_error
        competition_loss_value = total_error / (sum_of_actuals + 1e-8)

        return competition_loss_value