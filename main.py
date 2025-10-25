from src.fetch_prices import get_stock_data
import argparse
import json


def main():
    parser = argparse.ArgumentParser(prog="main.py", description="Project entrypoint: fetch prices")
    parser.add_argument("-t", "--tickers", help="Comma-separated tickers", default="AAPL,NVDA,GOOGL,TSLA,IBIT")
    parser.add_argument("--json", help="Print JSON output", action="store_true")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    results = get_stock_data(tickers)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("Fetched prices:")
        for r in results:
            print(r)


if __name__ == "__main__":
    main()
