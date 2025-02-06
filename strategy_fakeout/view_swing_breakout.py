import matplotlib.pyplot as plt

# Simulated candle data for illustration
candles = {
    'index': list(range(15)),
    'open':   [100, 101, 103, 102, 101, 100, 99, 100, 106, 105, 104, 103, 102, 101, 100],
    'high':   [100, 102, 104, 105, 103, 102, 101, 102, 107, 105, 104, 103, 102, 101, 100],
    'low':    [99, 100, 102, 101, 100, 99, 98, 99, 105, 104, 103, 102, 101, 100, 99],
    'close':  [100, 101, 103, 104, 102, 101, 100, 101, 107, 104, 103, 102, 101, 100, 99]
}

# Define special candle positions based on our rule:
swing_index = 3         # Swing candle where swing high is formed
breakout_index = 8      # Breakout candle (must occur at least 5 candles after swing candle)
confirmation_index = 9  # Confirmation candle

# Extract the swing high level from the swing candle
swing_high = candles['high'][swing_index]

plt.figure(figsize=(10, 6))
plt.plot(candles['index'], candles['close'], marker='o', label='Close Price')

# Draw horizontal line at the swing high level
plt.axhline(swing_high, color='r', linestyle='--', label=f"Swing High ({swing_high})")

# Mark the swing, breakout and confirmation candles
plt.scatter(swing_index, candles['close'][swing_index], color='orange', s=100, label='Swing Candle')
plt.scatter(breakout_index, candles['close'][breakout_index], color='green', s=100, label='Breakout Candle')
plt.scatter(confirmation_index, candles['close'][confirmation_index], color='blue', s=100, label='Confirmation Candle')

# Annotate the gap between the swing candle and the breakout candle
gap = breakout_index - swing_index
plt.text((swing_index + breakout_index) / 2, swing_high + 1, f"{gap} candle gap", 
         ha='center', color='purple', fontsize=12)

plt.xlabel("Candle Index")
plt.ylabel("Price")
plt.title("Visualization of Swing High and Breakout with Minimum 5 Candle Gap")
plt.legend()
plt.grid(True)

# Instead of showing the plot interactively, save it to a file
plt.savefig("swing_breakout.png")
print("Chart saved to swing_breakout.png") 