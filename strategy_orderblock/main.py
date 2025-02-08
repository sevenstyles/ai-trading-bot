import pandas as pd
import orderblock_detector
import visualization
import data_loader  # Import the data_loader module

if __name__ == "__main__":
    symbol = "SOLUSDT"
    interval = "1h"
    limit = 500
    swing_length = 50  # This corresponds to the "size" parameter in the Pine Script
    mitigation_type = 'highlow'
    volatility_type = 'atr'
    atr_period = 200
    atr_multiplier = 2

    df = data_loader.load_binance_data(symbol, interval, limit, volatility_type, atr_period, atr_multiplier)
    if df is None:
        print("Failed to load data. Exiting.")
        exit()

    order_blocks = orderblock_detector.identify_order_blocks(df, swing_length=swing_length, volatility_type=volatility_type, order_block_mitigation=mitigation_type)

    print(f"Number of order blocks detected: {len(order_blocks)}")
    print(order_blocks)

    visualization.visualize_order_blocks(df, order_blocks, filename="btc_order_blocks.png", mitigation_type=mitigation_type)
    print("Order blocks visualized and saved to btc_order_blocks.png") 