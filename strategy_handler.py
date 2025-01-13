import os
import json
import logging
import re
from openai import OpenAI
from tabulate import tabulate

# Get trading strategy from DeepSeek API
def get_trading_strategy(ohlc_data):
    try:
        client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url='https://api.deepseek.com'
        )

        with open('deepseek_prompt.txt', 'r') as f:
            system_message = f.read().strip()

        table = tabulate(ohlc_data, headers='keys', tablefmt='grid')
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"""Analyze this BTCUSDT OHLC data and provide a trade strategy.
            Requirements:
            1. Based on the price data, determine if this should be a LONG or SHORT trade
            2. If there is no suitable trade, do not place a trade.
            
            Data:
            {table}
            
            Respond with ONLY the raw JSON object - no markdown, no code blocks."""}
        ]

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.1,
            stream=False
        )
        
        ds_response = response.choices[0].message.content.strip()
        ds_response = re.sub(r'^```json\s*', '', ds_response)
        ds_response = re.sub(r'\s*```$', '', ds_response)
        
        strategy = json.loads(ds_response)
        logging.info(f"Received strategy: {strategy}")
        return strategy
    
    except Exception as e:
        logging.error(f"Error getting trading strategy: {e}")
        raise 