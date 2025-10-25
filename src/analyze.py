import sys
import json
import argparse
from typing import List, Dict, Any, Tuple
from datetime import datetime

from openai import OpenAI
from dotenv import load_dotenv
import os
import tiktoken

from fetch_prices import get_stock_data
from fetch_news import get_stock_news

# Constants for cost estimation
GPT35_INPUT_COST = 0.0015 / 1000  # $0.0015 per 1K input tokens
GPT35_OUTPUT_COST = 0.002 / 1000   # $0.002 per 1K output tokens
MAX_TOKENS_OUTPUT = 60  # Limit response length

def count_tokens(text: str) -> int:
    """Count tokens in a text string using tiktoken."""
    try:
        encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoder.encode(text))
    except Exception as e:
        print(f"Warning: Could not count tokens accurately: {e}")
        # Fallback: rough estimate based on words
        return len(text.split()) * 1.3

def analyze_stock(ticker: str, price_data: Dict[str, Any], news_articles: List[Dict[str, Any]], client: OpenAI) -> str:
    """Generate a natural language summary of a stock's performance and news."""
    
    # Extract just the most relevant news title
    top_news = news_articles[0]["title"] if news_articles else "No recent news"
    
    # Create minimal prompt
    prompt = f"${ticker} {price_data['change_pct']:+.2f}% to ${price_data['price']}. Recent: {top_news}. Write 1 clear sentence connecting price and news."
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using GPT-3.5-turbo for wider availability
            messages=[
                {"role": "system", "content": "You are a stock market analyst writing brief, accurate summaries of stock performance and news."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating summary for {ticker}: {str(e)}"

def estimate_cost(tickers: List[str], price_data: Dict[str, Dict], news_data: Dict[str, List]) -> Tuple[float, int, int]:
    """Estimate API cost and token usage for analysis."""
    total_input_tokens = 0
    total_output_tokens = MAX_TOKENS_OUTPUT * len(tickers)  # Maximum possible output
    
    for ticker in tickers:
        if ticker in price_data and ticker in news_data:
            # Simulate prompt construction and count tokens
            mock_prompt = f"${ticker} {price_data[ticker]['change_pct']:+.2f}% to ${price_data[ticker]['price']}. "
            if news_data[ticker]:
                mock_prompt += f"Recent: {news_data[ticker][0]['title']}"
            total_input_tokens += count_tokens(mock_prompt)
    
    # Calculate costs
    input_cost = total_input_tokens * GPT35_INPUT_COST
    output_cost = total_output_tokens * GPT35_OUTPUT_COST
    total_cost = input_cost + output_cost
    
    return total_cost, total_input_tokens, total_output_tokens

def analyze_stocks(tickers: List[str], estimate_only: bool = False) -> List[Dict[str, Any]]:
    """Analyze multiple stocks, fetching prices, news, and generating summaries."""
    
    # Load OpenAI key
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment or .env file")
        sys.exit(1)
    
    # Get price and news data
    price_data = {stock["ticker"]: stock for stock in get_stock_data(tickers)}
    news_data = {ticker: get_stock_news(ticker, days_back=1, max_articles=1) for ticker in tickers}
    
    # If estimation mode, calculate and return costs
    if estimate_only:
        cost, input_tokens, output_tokens = estimate_cost(tickers, price_data, news_data)
        print(f"\nEstimated Analysis Cost:")
        print(f"Input tokens:  {input_tokens:,}")
        print(f"Output tokens: {output_tokens:,}")
        print(f"Total cost:    ${cost:.4f}")
        return []
    
    # Debug info (safely)
    key_len = len(api_key) if api_key else 0
    key_start = api_key[:5] if api_key else ''
    key_end = api_key[-4:] if api_key and len(api_key) > 4 else ''
    print(f"Debug: API key length: {key_len}, starts with: {key_start}..., ends with: ...{key_end}")
    
    if not api_key.startswith('sk-'):
        print("Error: OpenAI API key should start with 'sk-'")
        sys.exit(1)
    
    client = OpenAI(api_key=api_key)
    
    # Get price data for all tickers
    price_data = {stock["ticker"]: stock for stock in get_stock_data(tickers)}
    
    results = []
    for ticker in tickers:
        if ticker not in price_data:
            print(f"No price data found for {ticker}")
            continue
            
        # Get news for this ticker
        news = get_stock_news(ticker, days_back=2, max_articles=3)
        
        # Generate summary
        summary = analyze_stock(ticker, price_data[ticker], news, client)
        
        results.append({
            "ticker": ticker,
            "price": price_data[ticker]["price"],
            "change_pct": price_data[ticker]["change_pct"],
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        })
    
    return results

def _print_human(results: List[Dict[str, Any]]) -> None:
    """Print results in a human-readable format."""
    print("\nStock Analysis:")
    print("=" * 80)
    for r in results:
        print(f"\n{r['ticker']} (${r['price']}, {r['change_pct']:+.2f}%):")
        print(f"{r['summary']}")
    print("\n" + "=" * 80)

def _cli():
    """Handle command-line interface."""
    parser = argparse.ArgumentParser(description="Analyze stocks with price and news summaries")
    parser.add_argument("tickers", help="Comma-separated list of stock tickers (e.g., AAPL,MSFT,GOOGL)")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--estimate", action="store_true", help="Estimate API cost without running analysis")
    args = parser.parse_args()
    
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        parser.error("Please provide at least one ticker symbol")
    
    results = analyze_stocks(tickers, estimate_only=args.estimate)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        _print_human(results)

if __name__ == "__main__":
    _cli()