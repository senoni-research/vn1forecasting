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