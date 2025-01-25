import copy
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, TypeVar, cast

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch.nn import Module

T = TypeVar("T")


def freeze_module_by_name(model: Module, module_name_substring: str) -> None:
    """
    Freeze all parameters whose name contains 'module_name_substring'.
    i.e., param.requires_grad = False
    """
    for name, param in model.named_parameters():
        if module_name_substring in name:
            param.requires_grad = False


def unfreeze_module_by_name(model: Module, module_name_substring: str) -> None:
    """
    Unfreeze all parameters whose name contains 'module_name_substring'.
    i.e., param.requires_grad = True
    """
    for name, param in model.named_parameters():
        if module_name_substring in name:
            param.requires_grad = True


def train_model(
    model: Module,
    preprocessed_df: pd.DataFrame,
    device: torch.device,
    generate_time_series_samples: Callable[..., Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]],
    prepare_batch_data: Callable[..., Tuple[torch.Tensor, ...]],
    validate_model_with_loss: Callable[..., Tuple[float, NDArray[np.float64], NDArray[np.float64]]],
    phases_config: List[Tuple[str, int, int, int, float]] = [
        ("init", 1, 24, 50000, 1e-3),
        ("core", 51, 512, 200000, 1e-3),
        ("core", 51, 512, 1, 1e-4),
        ("core", 51, 512, 1, 1e-5),
        ("tune", 51, 512, 200000, 1e-5),
        ("finish", 51, 512, 200000, 1e-5),
    ],
    patience: int = 10,
) -> Tuple[Module, List[Dict[str, Any]], NDArray[np.float64], NDArray[np.float64]]:
    """
    Train the model through multiple phases with different configurations.

    Args:
        model: The neural network model to train
        preprocessed_df: Preprocessed DataFrame containing the training data
        device: Device to run the training on (CPU/GPU)
        generate_time_series_samples: Function to generate training and validation samples
        prepare_batch_data: Function to prepare batch data for training
        validate_model_with_loss: Function to validate the model and compute loss
        phases_config: List of phase configurations (phase_name, n_epochs, batch_size, n_samples, learning_rate)
        patience: Number of epochs to wait for improvement before early stopping

    Returns:
        Tuple containing:
        - Trained model
        - Validation samples
        - Validation predictions
        - Validation targets
    """
    current_best_state: Optional[Dict[str, torch.Tensor]] = None
    val_predictions: NDArray[np.float64] = np.array([])
    val_targets: NDArray[np.float64] = np.array([])

    for phase, n_epochs, batch_size, n_samples, lr in phases_config:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # ---------------------------------------------------------------------------
        # 1) Phase-Specific Freezing
        # ---------------------------------------------------------------------------
        if phase == "tune":
            freeze_module_by_name(model, "embedding")
            freeze_module_by_name(model, "sales_encoder")
            print("Froze all embedding layers.")
            print("Froze the sales encoder.")

        if phase == "finish":
            freeze_module_by_name(model, "price_encoder")
            unfreeze_module_by_name(model, "sales_encoder")
            print("Froze the price encoder.")
            print("Unfroze the sales encoder.")

        # ---------------------------------------------------------------------------
        # 2) Sample Generation
        # ---------------------------------------------------------------------------
        if phase in ["init", "tune", "finish"]:
            train_samples, valid_samples = [], []
        _train_samples, _valid_samples = generate_time_series_samples(
            preprocessed_df, n_samples, train_valid_split=0.8, phase=phase
        )
        train_samples += _train_samples
        valid_samples += _valid_samples
        print(f"Training samples: {len(train_samples)}")

        # Initialize early stopping variables
        best_val_loss = float("inf")
        best_model_state: Optional[Dict[str, torch.Tensor]] = None
        patience_counter = 0

        # ---------------------------------------------------------------------------
        # 3) Train/Eval Loop for this Phase
        # ---------------------------------------------------------------------------
        for epoch in range(n_epochs):
            model.train()
            total_train_loss = 0

            # Shuffle train samples
            train_samples_arr = np.array(train_samples, dtype=object)
            np.random.shuffle(train_samples_arr)
            train_samples = cast(List[Dict[str, Any]], train_samples_arr.tolist())

            # Process training batches
            for i in range(0, len(train_samples), batch_size):
                batch_samples = train_samples[i : i + batch_size]
                batch_data = prepare_batch_data(batch_samples, mode="train", device=device)
                (
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
                ) = batch_data

                predictions = model(
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
                ).squeeze(-1)

                if phase == "finish":
                    loss = model.masked_competition_loss(predictions, target)
                else:
                    loss = model.masked_loss(predictions, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            val_loss, val_predictions, val_targets = validate_model_with_loss(
                model, valid_samples, batch_size, prepare_batch_data, phase, device=device
            )
            print(
                f"Epoch {epoch}, Train Loss: {total_train_loss / (len(train_samples) // batch_size):.4f}, "
                f"Validation Loss: {val_loss:.4f}"
            )

            valid_mask = ~np.isnan(val_targets)
            val_mse = np.mean((val_predictions[valid_mask] - val_targets[valid_mask]) ** 2)
            val_mae = np.mean(np.abs(val_predictions[valid_mask] - val_targets[valid_mask]))
            print(f"Validation MSE: {val_mse:.4f}")
            print(f"Validation MAE: {val_mae:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                current_best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs - no val loss improvement for {patience} epochs")
                if current_best_state is not None:
                    model.load_state_dict(cast(Mapping[str, Any], current_best_state))
                break

            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Current learning rate: {current_lr:.2e}")

        if current_best_state is not None:
            model.load_state_dict(cast(Mapping[str, Any], current_best_state))

    return model, valid_samples, val_predictions, val_targets


def validate_model_with_loss(
    model: Module,
    val_samples: List[Dict[str, Any]],
    batch_size: int,
    prepare_batch_data: Callable[..., Tuple[torch.Tensor, ...]],
    phase: str,
    max_past_weeks: int = 52,
    max_future_weeks: int = 13,
    device: str = "cpu",
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute validation loss and predictions.

    Args:
        model: Trained transformer model
        val_samples: List of validation samples
        batch_size: Batch size for validation
        prepare_batch_data: Function to prepare batch data
        max_past_weeks: Maximum length of historical data
        max_future_weeks: Maximum length of future data
        device: Device to run validation on

    Returns:
        Tuple containing:
        - val_loss: Average validation loss
        - all_predictions: Predictions for the entire validation set
        - all_targets: Ground-truth target values
    """
    model.eval()
    total_loss = 0
    all_predictions: List[NDArray[np.float64]] = []
    all_targets: List[NDArray[np.float64]] = []

    with torch.no_grad():
        for i in range(0, len(val_samples), batch_size):
            batch_samples = val_samples[i : i + batch_size]
            batch_data = prepare_batch_data(
                batch_samples,
                max_past_weeks=max_past_weeks,
                max_future_weeks=max_future_weeks,
                mode="train",
                device=device,
            )

            (
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
            ) = batch_data

            predictions = model(
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
            ).squeeze(-1)

            if phase == "finish":
                batch_loss = model.masked_competition_loss(predictions, target)
            else:
                batch_loss = model.masked_loss(predictions, target)
            total_loss += batch_loss.item()

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    val_loss = total_loss / (len(val_samples) / batch_size)
    all_predictions_array = np.concatenate(all_predictions, axis=0)
    all_targets_array = np.concatenate(all_targets, axis=0)

    return val_loss, all_predictions_array, all_targets_array


def run_inference_on_test(
    model: Module,
    test_samples: List[Dict[str, Any]],
    batch_size: int,
    prepare_batch_data: Callable[..., Tuple[torch.Tensor, ...]],
    preprocessor: Any,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Run inference on the test set, inverse-transform predictions, and decode label-encoded variables.

    Args:
        model: Trained transformer model
        test_samples: List of dictionaries containing test data
        batch_size: Batch size for inference
        prepare_batch_data: Function to prepare batch data
        preprocessor: Instance of DataPreprocessor for inverse-transforming and decoding
        device: Device to run inference on

    Returns:
        DataFrame with inverse-transformed predictions and original identifiers
    """
    model.eval()
    predictions: List[Dict[str, Any]] = []

    with torch.no_grad():
        for i in range(0, len(test_samples), batch_size):
            batch_samples = test_samples[i : i + batch_size]
            batch_data = prepare_batch_data(
                batch_samples,
                max_past_weeks=52,
                max_future_weeks=13,
                mode="test",
                device=device,
            )

            (
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
            ) = batch_data

            preds = model(
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
            ).squeeze(-1)
            preds = preds.cpu().numpy()

            for j, prediction in enumerate(preds):
                sample = batch_samples[j]
                product_id = sample["product"]
                original_preds, _ = preprocessor.inverse_transform(product_id, sales=prediction)
                original_client = preprocessor.client_encoder.inverse_transform([sample["Client"]])[0]
                original_warehouse = preprocessor.warehouse_encoder.inverse_transform([sample["Warehouse"]])[0]
                original_product = preprocessor.product_encoder.inverse_transform([sample["product"]])[0]

                predictions.append(
                    {
                        "Client": original_client,
                        "Warehouse": original_warehouse,
                        "Product": original_product,
                        "Predictions": original_preds.flatten(),
                    }
                )

    return pd.DataFrame(predictions)
