import requests
import os

def send_telegram_alert(row):
    message = (
        f"🚨 {row['symbol']} {row['direction']} TRIGGERED\n"
        f"Entry: {row['actual_entry']}\n"
        f"Stop: {row['stop_price']}\n"
        f"TP1: {row['tp1']}\n" 
        f"TP2: {row['tp2']}\n"
        f"Time: {row['trigger_time']}"
    )
    
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    params = {
        'chat_id': chat_id,
        'text': message
    }
    
    requests.post(url, params=params) 