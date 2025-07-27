import requests
def get_sentiment(symbol, token):
    url = f'https://finnhub.io/api/v1/stock/social-sentiment?symbol={symbol}&token={token}'
    return requests.get(url).json()
