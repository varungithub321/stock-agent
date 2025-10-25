import sys
import json
import argparse
from datetime import datetime

try:
    import yfinance as yf
except ModuleNotFoundError:
    print("Package 'yfinance' is not installed. Install it with: pip install yfinance")
    sys.exit(1)

def get_stock_data(tickers):
    print("Running fetch_prices.py...")
    results = []
    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).history(period="1d")
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            continue

        print(f"{ticker} → Rows: {len(data)}")
        if data.empty:
            print(f"{ticker} has no data!")
            continue

        latest = data.tail(1)
        close = latest["Close"].values[0]
        open_ = latest["Open"].values[0]
        change = ((close - open_) / open_) * 100
        results.append({
            "ticker": ticker,
            # ensure native Python floats (not numpy types) for JSON-friendliness
            "price": float(round(close, 2)),
            "change_pct": float(round(change, 2)),
            "timestamp": datetime.now().isoformat()
        })

    return results


def _print_human(results):
    print("Fetched prices:")
    for stock in results:
        print(stock)


def _cli():
    parser = argparse.ArgumentParser(prog="fetch_prices.py", description="Fetch latest stock prices")
    parser.add_argument("-t", "--tickers", help="Comma-separated tickers (default: AAPL,NVDA,GOOGL,TSLA,IBIT)",
                        default="AAPL,NVDA,GOOGL,TSLA,IBIT")
    parser.add_argument("--json", help="Print JSON output", action="store_true")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    results = get_stock_data(tickers)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        _print_human(results)


if __name__ == "__main__":
    _cli()