import requests
import datetime
import json
from openai import OpenAI  # Import the new OpenAI client
from tabulate import tabulate

# Fetch data from Binance API
url = 'https://api.binance.com/api/v3/klines'
params = {
    'symbol': 'BTCUSDT',
    'interval': '15m',
    'limit': 100
}
response = requests.get(url, params=params)
data = response.json()

# Parse and organize data
ohlc_data = []
for candle in data:
    ohlc = {
        'Open Time': datetime.datetime.fromtimestamp(candle[0]/1000).strftime('%Y-%m-%d %H:%M:%S'),
        'Open': float(candle[1]),
        'High': float(candle[2]),
        'Low': float(candle[3]),
        'Close': float(candle[4]),
        'Volume': float(candle[5])
    }
    ohlc_data.append(ohlc)

# Convert data to a table string
table = tabulate(ohlc_data, headers='keys', tablefmt='grid')

# Set up OpenAI client for DeepSeek API
client = OpenAI(
    api_key='sk-b63a8aeb069d46d7bba8be80092bbab7',  # Replace with your actual API key
    base_url='https://api.deepseek.com'  # DeepSeek API base URL
)

# Prepare messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f"Analyse the following data and provide a trade entry, exit and stop loss level.\n\nData:\n{table}"}
]

# Send chat completion request
try:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False
    )
    # Extract and save the response
    with open('response.txt', 'w') as file:
        file.write(response.choices[0].message.content)
    print("Response saved to response.txt")
except Exception as e:
    print(f"An error occurred: {e}")