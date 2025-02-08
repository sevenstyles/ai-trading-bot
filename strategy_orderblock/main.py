import pandas as pd
import orderblock_detector
import visualization
import data_loader  

if __name__ == "__main__":
    symbol = "ETHUSDT"
    interval = "1h"
    limit = 1000
    swing_length = 50  
    mitigation_type = 'highlow'
    volatility_type = 'atr'
    atr_period = 200
    atr_multiplier = 2

    df = data_loader.load_binance_data(symbol, interval, limit, volatility_type, atr_period, atr_multiplier)
    if df is None:
        print("Failed to load data. Exiting.")
        exit()

    swing_order_blocks, internal_order_blocks = orderblock_detector.identify_order_blocks(df, swing_length=5, internal_length=2, volatility_type=volatility_type, atr_period=atr_period, atr_multiplier=atr_multiplier)

    print(f"Number of swing order blocks detected: {len(swing_order_blocks)}")
    print(swing_order_blocks)
    print(f"Number of internal order blocks detected: {len(internal_order_blocks)}")
    print(internal_order_blocks)


    visualization.visualize_order_blocks(df, swing_order_blocks, internal_order_blocks, filename="btc_order_blocks.png", mitigation_type=mitigation_type)
    print("Order blocks visualized and saved to btc_order_blocks.png")