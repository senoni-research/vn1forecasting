import torch
import torch.nn as nn

import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import timedelta


def freeze_module_by_name(model, module_name_substring):
    """
    Freeze all parameters whose name contains 'module_name_substring'.
    i.e., param.requires_grad = False
    """
    for name, param in model.named_parameters():
        if module_name_substring in name:
            param.requires_grad = False


def unfreeze_module_by_name(model, module_name_substring):
    """
    Unfreeze all parameters whose name contains 'module_name_substring'.
    i.e., param.requires_grad = True
    """
    for name, param in model.named_parameters():
        if module_name_substring in name:
            param.requires_grad = True
            

def train_model(
    model,
    preprocessed_df,
    device,
    generate_time_series_samples,
    prepare_batch_data,
    validate_model_with_loss,
    phases_config=[
        ('init', 1, 24, 50000, 1e-3),
        ('core', 51, 512, 200000, 1e-3),
        ('core', 51, 512, 1, 1e-4),
        ('core', 51, 512, 1, 1e-5),
        ('tune', 51, 512, 200000, 1e-5),
        ('finish', 51, 512, 200000, 1e-5)
    ],
    patience=10
):
    for phase, n_epochs, batch_size, n_samples, lr in phases_config:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # ---------------------------------------------------------------------------
        # 1) Phase-Specific Freezing
        # ---------------------------------------------------------------------------
        if phase == 'tune':
            # Freeze all embeddings: 
            freeze_module_by_name(model, 'embedding')
            freeze_module_by_name(model, 'sales_encoder')
            print("Froze all embedding layers.")
            print("Froze the sales encoder.")

        if phase == 'finish':
            # Freeze the price encoder:
            freeze_module_by_name(model, 'price_encoder')
            unfreeze_module_by_name(model, 'sales_encoder')
            print("Froze the price encoder.")
            print("Unfroze the sales encoder.")

        # ---------------------------------------------------------------------------
        # 2) Sample Generation
        # ---------------------------------------------------------------------------
        if phase in ['init', 'tune', 'finish']:
            train_samples, valid_samples = [], []
        _train_samples, _valid_samples = generate_time_series_samples(
            preprocessed_df, n_samples, train_valid_split=0.8, phase=phase
        )
        train_samples += _train_samples
        valid_samples += _valid_samples
        print(f"Training samples: {len(train_samples)}")

        # Initialize early stopping variables
        best_val_loss = float('inf')
        best_model_state = None
        patience = 10
        patience_counter = 0
        
        # ---------------------------------------------------------------------------
        # 3) Train/Eval Loop for this Phase
        # ---------------------------------------------------------------------------
        for epoch in range(n_epochs):
            model.train()
            total_train_loss = 0

            # Shuffle train samples
            np.random.shuffle(train_samples)

            # Process training batches
            for i in range(0, len(train_samples), batch_size):
                # Prepare batch data
                batch_samples = train_samples[i:i + batch_size]
                batch_data = prepare_batch_data(batch_samples, mode='train', device=device)
                (sales, price, decoder_input, wom, woy, moy, qoy,
                sales_padding_mask, price_padding_mask, price_validity_mask,
                client, warehouse, product, target, rolling_4w_sales,
                rolling_13w_sales) = batch_data

                # Forward pass
                predictions = model(
                    sales, price, decoder_input, wom, woy, moy, qoy,
                    sales_padding_mask, price_padding_mask, price_validity_mask,
                    client, warehouse, product, rolling_4w_sales, rolling_13w_sales
                ).squeeze(-1)

                # Compute loss
                loss = model.masked_competition_loss(predictions, target)

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            # Validation
            val_loss, val_predictions, val_targets = validate_model_with_loss(
                model, valid_samples, batch_size, prepare_batch_data, device=device
            )
            print(f"Epoch {epoch}, Train Loss: {total_train_loss / (len(train_samples) // batch_size):.4f}, "
                f"Validation Loss: {val_loss:.4f}")

            # Additional metrics (MSE, MAE)
            valid_mask = ~np.isnan(val_targets)
            val_mse = np.mean((val_predictions[valid_mask] - val_targets[valid_mask]) ** 2)
            val_mae = np.mean(np.abs(val_predictions[valid_mask] - val_targets[valid_mask]))
            print(f"Validation MSE: {val_mse:.4f}")
            print(f"Validation MAE: {val_mae:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs - no val loss improvement for {patience} epochs")
                model.load_state_dict(best_model_state)
                break

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.2e}")

        # Ensure best model for next phase
        model.load_state_dict(best_model_state)

    return model, valid_samples, val_predictions, val_targets


def validate_model_with_loss(
    model, val_samples, batch_size, prepare_batch_data, max_past_weeks=52, max_future_weeks=13, device='cpu'
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
            batch_loss = model.masked_competition_loss(predictions, target)
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


def run_inference_on_test(model, test_samples, batch_size, prepare_batch_data, preprocessor, device='cpu'):
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


