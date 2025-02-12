from binance import Client
from config import API_KEY, API_SECRET

def test_binance_connection():
    try:
        client = Client(API_KEY, API_SECRET, testnet=True)
        server_time = client.get_server_time()
        print(f"Binance Testnet Server Time: {server_time}")
        print("Binance API connection successful!")
        return True
    except Exception as e:
        print(f"Binance API connection failed: {e}")
        return False

if __name__ == "__main__":
    test_binance_connection()