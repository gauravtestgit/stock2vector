from dataclasses import dataclass
from typing import Optional

@dataclass
class FinancialMetrics:
    # Valuation
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    fund_family: Optional[str] = None
    category: Optional[str] = None
    quote_type: Optional[str] = None
    long_name: Optional[str] = None
    enterprise_value : Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    
    
    # Profitability
    total_revenue: Optional[float] = None
    net_income: Optional[float] = None
    current_revenue : Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    # Growth
    yearly_revenue_growth: Optional[float] = None
    quarterly_revenue_growth: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None
    
    # Financial Health
    free_cash_flow: Optional[float] = None
    total_debt : Optional[float] = None
    total_cash : Optional[float] = None
    shares_outstanding : Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    
    
    # Technical
    current_price: Optional[float] = None
    beta: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None