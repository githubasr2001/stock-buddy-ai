import os
from dotenv import load_dotenv
import google.generativeai as genai
import yfinance as yf
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Optional
import streamlit as st
from streamlit_chat import message
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class StockInfo(BaseModel):
    """Pydantic model for stock information"""
    symbol: str = Field(..., description="Stock ticker symbol")
    name: str = Field(..., description="Company name")
    price: float = Field(..., description="Current stock price")
    currency: str = Field(..., description="Currency in which the stock is traded")

class StockDataManager:
    """Handles all stock data operations"""
    @staticmethod
    def get_current_stock_price(symbols: List[str]) -> List[StockInfo]:
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
                st.warning(f"Error fetching data for {symbol}: {str(e)}")
        return stock_data

    @staticmethod
    def get_historical_data(symbols: List[str], years: int = 5) -> Dict[str, pd.Series]:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        historical_data = {}
        
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                df = stock.history(start=start_date, end=end_date)
                if not df.empty:
                    historical_data[symbol] = df['Close']
            except Exception as e:
                st.warning(f"Error fetching historical data for {symbol}: {str(e)}")
        
        return historical_data

class Visualization:
    """Handles all visualization tasks"""
    @staticmethod
    def create_stock_plot(historical_data: Dict[str, pd.Series]) -> go.Figure:
        fig = go.Figure()
        for symbol, data in historical_data.items():
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data,
                name=symbol,
                mode='lines',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Stock Price History",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )
        return fig

class AIHandler:
    """Manages AI interactions"""
    def __init__(self, api_key: str):
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-pro",
                generation_config={
                    "temperature": 0.1,
                    "top_p": 1,
                    "max_output_tokens": 1000,
                }
            )
        except Exception as e:
            st.error(f"Error configuring Gemini API: {str(e)}")
            st.stop()

    def extract_symbols(self, query: str) -> List[str]:
        symbol_prompt = f"""
        Extract stock ticker symbols from this query:
        Query: "{query}"
        Return only a comma-separated list of stock symbols without any additional text.
        For example: AAPL, MSFT, GOOGL
        """
        response = self.model.generate_content(symbol_prompt)
        symbols = response.text.strip().split(",") if response.text else []
        return [s.strip().upper() for s in symbols if s.strip()]

    def generate_response(self, query: str, stock_data: str) -> str:
        prompt = f"""
        Given this stock data: {stock_data},
        generate a friendly response to: "{query}"
        """
        response = self.model.generate_content(prompt)
        return response.text if response.text else "I processed your request but couldn't generate a response."

class StockBuddy:
    """Main application class"""
    def __init__(self):
        self.ai_handler = AIHandler(GEMINI_API_KEY)
        self.stock_manager = StockDataManager()
        self.visualizer = Visualization()

    def process_query(self, query: str) -> Tuple[str, Optional[go.Figure]]:
        try:
            symbols = self.ai_handler.extract_symbols(query)
            if not symbols:
                return "I couldn't find any stock symbols in your query. Please mention specific companies or ticker symbols.", None

            historical_requested = any(word in query.lower() for word in ["past", "history", "historical", "years", "compare"])
            
            if historical_requested:
                years = next((int(word) for word in query.split() if word.isdigit()), 5)
                historical_data = self.stock_manager.get_historical_data(symbols, years)
                if not historical_data:
                    return "Sorry, I couldn't retrieve historical data for those stocks.", None
                
                plot = self.visualizer.create_stock_plot(historical_data)
                current_prices = self.stock_manager.get_current_stock_price(symbols)
                stock_info = ', '.join([f'{s.name} ({s.symbol}): {s.price} {s.currency}' for s in current_prices])
                
                response = self.ai_handler.generate_response(query, stock_info)
                return response, plot
            else:
                stock_data = self.stock_manager.get_current_stock_price(symbols)
                if not stock_data:
                    return "Sorry, I couldn't retrieve stock data for those symbols.", None
                
                stock_info = ', '.join([f'{s.name} ({s.symbol}): {s.price} {s.currency}' for s in stock_data])
                response = self.ai_handler.generate_response(query, stock_info)
                return response, None

        except Exception as e:
            return f"An error occurred: {str(e)}", None

def setup_page():
    """Configure Streamlit page settings and styling"""
    st.set_page_config(
        page_title="Stock Buddy",
        page_icon="💹",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .main { background-color: #f0f2f6; padding: 20px; }
    .stTextInput > div > div > input {
        border-radius: 20px;
        padding: 10px 15px;
    }
    .stButton > button {
        border-radius: 20px;
        background-color: #4CAF50;
        color: white;
        width: 100%;
    }
    .stButton > button:hover { background-color: #45a049; }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar content"""
    with st.sidebar:
        st.image("stocks.jpg", width=100)
        st.title("Stock Buddy")
        st.markdown("Your friendly stock price assistant!")
        st.markdown("---")
        st.info("Ask me about current or historical stock prices!")
        st.markdown("""
        Examples:
        - What's Tesla's stock price?
        - Compare Apple and Tesla over 5 years
        - Show me historical prices for GOOGL and MSFT
        """)

def main():
    setup_page()
    render_sidebar()
    
    st.title("💹 Stock Buddy - Your Market Assistant")
    st.markdown("Chat with me about stock prices below!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm Stock Buddy. Ask me about current or historical stock prices!", "plot": None}
        ]

    # Display chat messages
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            message(msg["content"], is_user=True, key=f"user_{i}")
        else:
            message(msg["content"], key=f"assistant_{i}")
            if msg.get("plot"):
                st.plotly_chart(msg["plot"], use_container_width=True)

    # Input area
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input("Ask me anything about stocks:", 
                                  placeholder="Compare Apple and Tesla over 5 years")
    with col2:
        send_button = st.button("Send", use_container_width=True)

    stock_buddy = StockBuddy()

    if send_button and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input, "plot": None})
        
        with st.spinner("Checking the markets..."):
            response, plot = stock_buddy.process_query(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response, "plot": plot})
        
        st.rerun()

    if st.button("Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm Stock Buddy. Ask me about current or historical stock prices!", "plot": None}
        ]
        st.rerun()

if __name__ == "__main__":
    main()