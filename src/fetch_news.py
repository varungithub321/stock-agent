import sys
import json
import argparse
from newsapi import NewsApiClient
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

def get_stock_news(ticker: str, days_back: int = 3, max_articles: int = 5) -> List[Dict[Any, Any]]:
    """
    Fetch recent news articles for a given stock ticker.
    
    Args:
        ticker: Stock symbol to fetch news for (e.g., 'AAPL')
        days_back: How many days of news to fetch (default: 3)
        max_articles: Maximum number of articles to return (default: 5)
    
    Returns:
        List of dicts containing article info (title, description, url, published_at)
    """
    print(f"Fetching news for {ticker}...")
    
    try:
        load_dotenv()
        api_key = os.getenv("NEWSAPI_KEY")
        if not api_key:
            print("Error: NEWSAPI_KEY not found in environment or .env file")
            return []
            
        newsapi = NewsApiClient(api_key=api_key)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Build search query (company name OR ticker)
        query = f"({ticker} OR {ticker} stock)"
        
        articles = newsapi.get_everything(
            q=query,
            language="en",
            sort_by="relevancy",
            from_param=start_date.strftime("%Y-%m-%d"),
            to=end_date.strftime("%Y-%m-%d"),
            page_size=max_articles
        )
        
        if not articles["articles"]:
            print(f"No recent news found for {ticker}")
            return []
            
        # Extract relevant fields from each article
        results = []
        for article in articles["articles"]:
            results.append({
                "title": article["title"],
                "description": article["description"],
                "url": article["url"],
                "published_at": article["publishedAt"],
                "source": article["source"]["name"]
            })
            
        return results
        
    except Exception as e:
        print(f"Error fetching news for {ticker}: {str(e)}")
        return []

def _print_human(articles: List[Dict[Any, Any]]) -> None:
    """Print articles in a human-readable format."""
    if not articles:
        return
        
    for i, article in enumerate(articles, 1):
        print(f"\nArticle {i}:")
        print(f"Title: {article['title']}")
        print(f"Source: {article['source']}")
        print(f"Published: {article['published_at']}")
        print(f"URL: {article['url']}")
        if article['description']:
            print(f"Summary: {article['description']}\n")


def _cli():
    """Handle command-line interface."""
    parser = argparse.ArgumentParser(description="Fetch recent news for a stock ticker")
    parser.add_argument("ticker", help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("--days", type=int, default=3, help="Days of news to fetch (default: 3)")
    parser.add_argument("--max", type=int, default=5, help="Maximum articles to fetch (default: 5)")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    args = parser.parse_args()
    
    articles = get_stock_news(args.ticker, args.days, args.max)
    if args.json:
        print(json.dumps(articles, indent=2))
    else:
        _print_human(articles)


if __name__ == "__main__":
    _cli()