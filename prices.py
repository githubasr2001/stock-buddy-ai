import os
from dotenv import load_dotenv
import google.generativeai as genai
import yfinance as yf
from pydantic import BaseModel, Field
from typing import List

# Load API Keys from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Error configuring Gemini API: {str(e)}")
    exit(1)

# Define Pydantic Model for Stock Data
class StockInfo(BaseModel):
    symbol: str = Field(..., description="Stock ticker symbol")
    name: str = Field(..., description="Company name")
    price: float = Field(..., description="Current stock price")
    currency: str = Field(..., description="Currency in which the stock is traded")

# Fetch Stock Prices
def get_stock_price(symbols: List[str]) -> List[StockInfo]:
    stock_data = []
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            stock_info = StockInfo(
                symbol=symbol,
                name=info.get("longName", "Unknown"),
                price=info.get("currentPrice", 0.0),
                currency=info.get("currency", "USD")
            )
            stock_data.append(stock_info)
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
    return stock_data

# Use Gemini AI to Extract Stock Symbols from User Query
def get_stock_symbols(query: str) -> List[str]:
    try:
        # Using gemini-1.5-pro which is confirmed available
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config={
                "temperature": 0.1,
                "top_p": 1,
                "max_output_tokens": 100,
            }
        )
        
        prompt = f"""
        Extract stock ticker symbols from this query:
        Query: "{query}"
        Return only a comma-separated list of stock symbols without any additional text.
        For example: AAPL, MSFT, GOOGL
        """
        
        response = model.generate_content(prompt)
        
        if not response.text:
            print("No response received from Gemini API")
            return []
            
        symbols = response.text.strip().split(",")
        return [s.strip().upper() for s in symbols if s.strip()]
    except Exception as e:
        print(f"Error extracting stock symbols: {str(e)}")
        return []

# Main Function
def main():
    try:
        user_query = input("Enter your stock query: ")
        if not user_query:
            print("Please enter a valid query")
            return
            
        symbols = get_stock_symbols(user_query)
        if not symbols:
            print("No stock symbols found in query")
            return
            
        stock_prices = get_stock_price(symbols)
        if not stock_prices:
            print("No stock data retrieved")
            return
            
        print("\nStock Prices:")
        print("-" * 50)
        for stock in stock_prices:
            print(f"{stock.name} ({stock.symbol}): {stock.price} {stock.currency}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()