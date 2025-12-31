"""
Financial Analysis Agent using LangGraph and Azure OpenAI
"""

import os
from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
import logging
from utils.config import load_env_variables, config
from dotenv import load_dotenv  
load_dotenv()

env_name = load_env_variables()
app_config = config[env_name]

azure_api_key = app_config.AZURE_OPENAI_API_KEY
azure_endpoint = app_config.AZURE_OPENAI_ENDPOINT   
azure_deployment = app_config.AZURE_OPENAI_DEPLOYMENT_NAME
azure_api_version = app_config.AZURE_OPENAI_API_VERSION
tavily_api_key = app_config.TAVILY_API_KEY

from financial_tools import (
    get_stock_data,
    calculate_moving_averages,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    get_support_resistance,
    analyze_volume_trend,
    get_comprehensive_analysis
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialAgentConfig:
    """Configuration for Financial Agent"""
    
    def __init__(
        self,
        azure_endpoint: str,
        azure_api_key: str,
        azure_deployment: str,
        azure_api_version: str = "2024-02-15-preview",
        tavily_api_key: str = None
    ):
        self.azure_endpoint = azure_endpoint
        self.azure_api_key = azure_api_key
        self.azure_deployment = azure_deployment
        self.azure_api_version = azure_api_version
        self.tavily_api_key = tavily_api_key


class FinancialAgent:
    """
    Financial Analysis Agent with market news and technical analysis capabilities
    """
    
    def __init__(self, config: FinancialAgentConfig):
        """Initialize the financial agent with Azure OpenAI"""
        
        # System prompt as a message
        self.system_message = SystemMessage(content="""You are an expert financial analyst assistant with deep knowledge of stock markets, technical analysis, and financial indicators.

Your capabilities include:
1. **Technical Analysis**: Calculate and interpret moving averages, RSI, MACD, Bollinger Bands, support/resistance levels
2. **Market Data**: Retrieve real-time stock prices, volumes, and company information
3. **Market News**: Search for latest market news and events (when Tavily is available)
4. **Investment Insights**: Provide actionable insights based on technical indicators

Guidelines:
- Always use the appropriate tools to get accurate, real-time data
- For stock queries, start with get_stock_data or get_comprehensive_analysis
- Combine multiple indicators for better analysis
- Explain technical terms in simple language
- Provide both bullish and bearish perspectives
- Always mention that this is not financial advice
- When asked about news, use the tavily_search_results_json tool
- Be concise but thorough in your analysis

Tool Usage:
- get_stock_data: Get basic stock information
- calculate_moving_averages: Analyze trend using SMAs
- calculate_rsi: Check if stock is overbought/oversold
- calculate_macd: Identify momentum and trend changes
- calculate_bollinger_bands: Assess volatility and price extremes
- get_support_resistance: Find key price levels
- analyze_volume_trend: Understand buying/selling pressure
- get_comprehensive_analysis: Get complete technical analysis
- tavily_search_results_json: Search for market news and events

Always cite your sources and be transparent about limitations.""")
        
        # Initialize Azure OpenAI with system message
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.azure_api_key,
            azure_deployment=config.azure_deployment,
            api_version=config.azure_api_version,
            temperature=0.7,
            streaming=True
        )
        
        # Initialize tools
        self.tools = [
            get_stock_data,
            calculate_moving_averages,
            calculate_rsi,
            calculate_macd,
            calculate_bollinger_bands,
            get_support_resistance,
            analyze_volume_trend,
            get_comprehensive_analysis
        ]
        
        # Add Tavily search if API key provided
        if config.tavily_api_key:
            try:
                tavily_search = TavilySearchResults(
                    api_key=config.tavily_api_key,
                    max_results=3,
                    search_depth='advanced',
                    max_tokens=2000
                )
                self.tools.append(tavily_search)
                logger.info("Tavily search tool added successfully")
            except Exception as e:
                logger.warning(f"Could not initialize Tavily search: {e}")
        
        # Create the agent with system message prepended
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools
        )
        
        logger.info("Financial Agent initialized successfully")
    
    def query(self, question: str, stream: bool = False):
        """
        Query the financial agent
        
        Args:
            question: User's question
            stream: Whether to stream the response
        
        Returns:
            Generator of messages if stream=True, else final response
        """
        # Prepend system message to the input
        inputs = {
            "messages": [
                self.system_message,
                HumanMessage(content=question)
            ]
        }
        
        if stream:
            return self.agent.stream(inputs, stream_mode="values")
        else:
            result = None
            for s in self.agent.stream(inputs, stream_mode="values"):
                result = s
            return result
    
    def analyze_stock(self, ticker: str) -> dict:
        """
        Perform comprehensive stock analysis
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Analysis results
        """
        question = f"Provide a comprehensive technical analysis for {ticker}. Include all key indicators and give me your overall assessment."
        return self.query(question, stream=False)
    
    def get_market_news(self, query: str = "latest stock market news") -> dict:
        """
        Get latest market news
        
        Args:
            query: News search query
        
        Returns:
            News results
        """
        question = f"Search for: {query}. Summarize the key findings."
        return self.query(question, stream=False)
    
    def compare_stocks(self, tickers: list) -> dict:
        """
        Compare multiple stocks
        
        Args:
            tickers: List of ticker symbols
        
        Returns:
            Comparison results
        """
        ticker_str = ", ".join(tickers)
        question = f"Compare these stocks: {ticker_str}. Analyze their technical indicators and tell me which one looks more promising."
        return self.query(question, stream=False)
    
    def print_stream(self, stream):
        """Pretty print streaming results"""
        for s in stream:
            message = s["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                if hasattr(message, 'pretty_print'):
                    message.pretty_print()
                else:
                    print(message)


# Convenience function
def create_financial_agent(
    azure_endpoint: str = None,
    azure_api_key: str = None,
    azure_deployment: str = None,
    tavily_api_key: str = None
) -> FinancialAgent:
    """
    Create a financial agent with environment variables or provided credentials
    
    Args:
        azure_endpoint: Azure OpenAI endpoint
        azure_api_key: Azure OpenAI API key
        azure_deployment: Azure OpenAI deployment name
        tavily_api_key: Tavily API key for web search
    
    Returns:
        Configured FinancialAgent
    """
    # Use provided values or fall back to environment variables
    endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
    deployment = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
    tavily_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
    
    if not endpoint or not api_key or not deployment:
        raise ValueError("Azure OpenAI credentials are required. Please provide them via parameters or environment variables.")
    
    config = FinancialAgentConfig(
        azure_endpoint=endpoint,
        azure_api_key=api_key,
        azure_deployment=deployment,
        tavily_api_key=tavily_key
    )
    
    return FinancialAgent(config)


# Example usage
if __name__ == "__main__":
    # Create agent
    agent = create_financial_agent()
    
    # Example queries
    print("=== Stock Analysis ===")
    result = agent.analyze_stock("AAPL")
    agent.print_stream([result])
    
    print("\n=== Market News ===")
    news = agent.get_market_news("Apple stock latest news")
    agent.print_stream([news])
    
    print("\n=== Stock Comparison ===")
    comparison = agent.compare_stocks(["AAPL", "MSFT", "GOOGL"])
    agent.print_stream([comparison])
    
    print("\n=== Custom Query ===")
    stream = agent.query("What are the best tech stocks to watch right now based on technical indicators?", stream=True)
    agent.print_stream(stream)