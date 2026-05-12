from abc import ABC, abstractmethod
from typing import Dict, Any

class IDataProvider(ABC):
    """Interface for financial data providers"""
    
    @abstractmethod
    def get_financial_metrics(self, ticker: str) -> Dict[str, Any]:
        """Get basic financial metrics for a ticker"""
        pass
    
    @abstractmethod
    def get_price_data(self, ticker: str) -> Dict[str, Any]:
        """Get price and technical data"""
        pass

    @abstractmethod
    def get_professional_analyst_data(self, ticker: str) -> Dict[str, Any]:
        """Get price and technical data"""
        pass

    @abstractmethod
    def get_daily_close_prices(self, tickers: list, start: str, end: str) -> Any:
        """Get daily close prices for multiple tickers"""
        pass