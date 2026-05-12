import yfinance as yf
import logging
from typing import Dict, Any
from ...interfaces.data_provider import IDataProvider
from ...models.financial_metrics import FinancialMetrics
from ...utils.rate_limit_tracker import rate_tracker
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress noisy yfinance/urllib3 logs (HTTP 401, 404, delisted warnings)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('peewee').setLevel(logging.CRITICAL)

class YahooFinanceProvider(IDataProvider):
    """Yahoo Finance data provider implementation"""
    def get_revenue_trend(self, stock: yf.Ticker, info: Dict = None) -> Dict:
        """Get revenue trend from yfinance - optimized with parallel DataFrame fetching"""
        
        # Parallel DataFrame fetching
        def fetch_dataframe(attr_name):
            return getattr(stock, attr_name)
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all DataFrame fetch operations in parallel
            future_to_attr = {
                executor.submit(fetch_dataframe, 'quarterly_income_stmt'): 'quarterly_income',
                executor.submit(fetch_dataframe, 'income_stmt'): 'annual_income',
                executor.submit(fetch_dataframe, 'cashflow'): 'cashflow',
                executor.submit(fetch_dataframe, 'quarterly_financials'): 'quarterly_financials',
                executor.submit(fetch_dataframe, 'financials'): 'annual_financials'
            }
            
            # Collect results as they complete
            dataframes = {}
            for future in as_completed(future_to_attr):
                attr_name = future_to_attr[future]
                try:
                    dataframes[attr_name] = future.result()
                except Exception as e:
                    print(f"Error fetching {attr_name}: {e}")
                    dataframes[attr_name] = None
        
        # Extract DataFrames
        quarterly_income = dataframes.get('quarterly_income')
        annual_income = dataframes.get('annual_income')
        cashflow = dataframes.get('cashflow')
        quarterly_financials = dataframes.get('quarterly_financials')
        annual_financials = dataframes.get('annual_financials')
        
        # Extract revenue data (serialize DataFrames to dicts for JSON compatibility)
        def serialize_dataframe(df):
            if df is None or df.empty:
                return {}
            df_dict = df.to_dict()
            serialized = {}
            for row_key, row_data in df_dict.items():
                serialized[str(row_key)] = {str(col_key): (str(value) if hasattr(value, 'strftime') else value) for col_key, value in row_data.items()}
            return serialized
        
        revenue_data = {
            'quarterly_income_stmt': serialize_dataframe(quarterly_income),
            'annual_income_stmt': serialize_dataframe(annual_income),
            'quarterly_financial_stmt': serialize_dataframe(quarterly_financials),
            'annual_financial_stmt': serialize_dataframe(annual_financials),
            'cashflow': serialize_dataframe(cashflow)
        }
    
        # Annual revenue
        if annual_income is not None and not annual_income.empty and 'Total Revenue' in annual_income.index:
            annual_revenue = annual_income.loc['Total Revenue'].dropna()
            revenue_data['annual_revenue'] = {str(k): v for k, v in annual_revenue.to_dict().items()}
            
            # Calculate growth rates
            revenue_values = annual_revenue.values
            if len(revenue_values) >= 2:
                recent_growth = (revenue_values[0] - revenue_values[1]) / revenue_values[1] * 100
                revenue_data['recent_annual_growth'] = recent_growth
        
        # Quarterly revenue
        if quarterly_income is not None and not quarterly_income.empty and 'Total Revenue' in quarterly_income.index:
            quarterly_revenue = quarterly_income.loc['Total Revenue'].dropna()
            revenue_data['quarterly_revenue'] = {str(k): v for k, v in quarterly_revenue.to_dict().items()}
            
            # Calculate QoQ growth
            revenue_values = quarterly_revenue.values
            if len(revenue_values) >= 2:
                qoq_growth = (revenue_values[0] - revenue_values[1]) / revenue_values[1] * 100
                revenue_data['recent_quarterly_growth'] = qoq_growth
        
        # Method 3: From info (current metrics) - use passed info parameter
        if info is None:
            info = stock.info
        revenue_data['current_revenue'] = info.get('totalRevenue', 0)
        revenue_data['revenue_growth'] = info.get('revenueGrowth', 0)
        revenue_data['quarterly_revenue_growth'] = info.get('quarterlyRevenueGrowth', 0)
        
        return revenue_data
    
    def get_financial_metrics(self, ticker: str) -> Dict[str, Any]:
        """Get financial metrics from Yahoo Finance"""
        try:
            rate_tracker.track_request(ticker)
            stock = yf.Ticker(ticker)
            info = stock.info
            revenue_data = self.get_revenue_trend(stock, info)
            cashflow = stock.cashflow
            
            fcf = 0
            if not cashflow.empty and 'Free Cash Flow' in cashflow.index:
                fcf_data = cashflow.loc['Free Cash Flow'].dropna()
                if len(fcf_data) > 0:
                    fcf = fcf_data.iloc[0]

            # Extract dividend information
            dividends = stock.dividends
            dividend_info = {
                'dividend_yield': info.get('dividendYield', 0),
                'dividend_rate': info.get('dividendRate', 0),
                'payout_ratio': info.get('payoutRatio', 0),
                'ex_dividend_date': info.get('exDividendDate'),
                'five_year_avg_dividend_yield': info.get('fiveYearAvgDividendYield', 0),
                'trailing_annual_dividend_rate': info.get('trailingAnnualDividendRate', 0),
                'trailing_annual_dividend_yield': info.get('trailingAnnualDividendYield', 0)
            }
            
            # Get recent dividend history
            if not dividends.empty:
                recent_dividends = dividends.tail(12)  # Last 12 dividend payments
                dividend_info['recent_dividends'] = {str(k): v for k, v in recent_dividends.to_dict().items()}
                dividend_info['last_dividend_amount'] = dividends.iloc[-1] if len(dividends) > 0 else 0
                dividend_info['last_dividend_date'] = str(dividends.index[-1]) if len(dividends) > 0 else None

            # Debug: Print available price fields for ETFs
            price_fields = ['currentPrice', 'regularMarketPrice', 'navPrice', 'previousClose', 'ask', 'bid', 'open']
            available_prices = {field: info.get(field) for field in price_fields if info.get(field)}
            # print(f"Yahoo Provider - Available price fields for {ticker}: {available_prices}")
            
            # Forward-looking financial metrics (not analyst opinions)
            forward_metrics = {
                'forward_pe': info.get('forwardPE', 0),
                'forward_eps': info.get('forwardEps', 0) or info.get('epsForward', 0),
                'current_year_eps': info.get('epsCurrentYear', 0),
                'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth', 0),
                'earnings_timestamp': info.get('earningsTimestamp'),
                'earnings_date': None
            }
            
            # Convert earnings timestamp to readable date
            if forward_metrics['earnings_timestamp']:
                from datetime import datetime
                forward_metrics['earnings_date'] = datetime.fromtimestamp(forward_metrics['earnings_timestamp']).strftime('%Y-%m-%d')
            
            return {
                'market_cap': info.get('marketCap', 0),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'fund_family': info.get('fundFamily', ''),
                'category': info.get('category', ''),
                'quote_type': info.get('quoteType', ''),
                'long_name': info.get('longName', ''),
                'business_summary': info.get('longBusinessSummary', ''),
                'enterprise_value': info.get('enterpriseValue', 0),
                'ev_ebitda_multiple': info.get('enterpriseToEbitda', 0),
                'total_revenue': info.get('totalRevenue', 0),
                'net_income': info.get('netIncomeToCommon', 0),
                'current_revenue': f"${revenue_data.get('current_revenue'):,.2f}",
                'yearly_revenue_growth': revenue_data.get('revenue_growth'),
                'quarterly_revenue_growth': revenue_data.get('quarterly_revenue_growth'),
                'calculated_annual_growth': revenue_data.get('recent_annual_growth', 0),
                'calculated_quarterly_growth': revenue_data.get('recent_quarterly_growth', 0),
                'revenue_data_statements': revenue_data,  # Include full revenue trend data
                'free_cash_flow': fcf,
                'total_debt': info.get('totalDebt', 0) or 0,
                'total_cash': info.get('totalCash', 0) or 0,
                'shares_outstanding': info.get('sharesOutstanding', 0),
                'float_shares': info.get('floatShares', 0),
                'current_price': info.get('currentPrice') or info.get('regularMarketPrice') or info.get('navPrice') or info.get('previousClose') or info.get('ask') or info.get('bid') or 0,
                'beta': info.get('beta', 1.0),
                'pe_ratio': info.get('trailingPE'),
                'pb_ratio': info.get('priceToBook'),
                'ps_ratio': info.get('priceToSalesTrailing12Months'),
                'trailing_eps': info.get('trailingEps', 0),
                'forward_eps': info.get('forwardEps', 0),
                'peg_ratio': info.get('pegRatio'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'debt_to_equity': info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0,
                'current_ratio': info.get('currentRatio'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'dividend_yield': info.get('dividendYield'),
                'payout_ratio': info.get('payoutRatio'),
                'dividend_info': dividend_info,  # Include full dividend data
                'profit_margin': info.get('profitMargins', 0) * 100,
                'quick_ratio': info.get('quickRatio', 0),
                'book_value_per_share': info.get('bookValue', 0),
                'cash_per_share': info.get('totalCashPerShare', 0),                
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                # Add forward metrics to financial_metrics
                **forward_metrics
            }
        except Exception as e:
            rate_tracker.check_rate_limit_error(str(e), ticker)
            return {'error': str(e)}
    
    def get_price_data(self, ticker: str) -> Dict[str, Any]:
        """Get price and technical data"""
        try:
            rate_tracker.track_request(ticker)
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            
            # Get last 30 days for chart
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            chart_hist = stock.history(start=start_date, end=end_date)
            
            chart_data = {}
            if not chart_hist.empty:
                chart_data = {
                    'prices': chart_hist['Close'].tolist(),
                    'dates': [date.strftime('%m/%d') for date in chart_hist.index]
                }
            
            return {
                'price_history': hist,
                'current_price': hist['Close'].iloc[-1] if not hist.empty else None,
                'volume': hist['Volume'].iloc[-1] if not hist.empty else None,
                'chart_data': chart_data
            }
        except Exception as e:
            rate_tracker.check_rate_limit_error(str(e), ticker)
            return {'error': str(e)}
    
    def get_professional_analyst_data(self, ticker: str) -> Dict:
        """Fetch ONLY analyst opinions and price targets (not financial metrics)"""
        data = {}
        try:
            rate_tracker.track_request(ticker)
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Price targets (analyst opinions)
            data['target_price'] = info.get('targetMeanPrice', 0)
            data['target_high'] = info.get('targetHighPrice', 0)
            data['target_low'] = info.get('targetLowPrice', 0)
            data['target_median_price'] = info.get('targetMedianPrice', 0)
            
            # Analyst recommendations
            data['recommendation'] = info.get('recommendationMean')
            data['recommendation_key'] = info.get('recommendationKey')
            data['analyst_count'] = info.get('numberOfAnalystOpinions', 0)
            
            # Growth estimates (analyst projections)
            data['earnings_growth'] = info.get('earningsGrowth', 0)
            data['revenue_growth'] = info.get('revenueGrowth', 0)
            
            # Recent upgrades/downgrades with dated price targets
            try:
                upgrades_df = stock.upgrades_downgrades
                if upgrades_df is not None and not upgrades_df.empty:
                    recent = upgrades_df.head(10)
                    data['recent_ratings'] = []
                    for date_idx, row in recent.iterrows():
                        data['recent_ratings'].append({
                            'date': str(date_idx)[:10],
                            'firm': row.get('Firm', 'N/A'),
                            'to_grade': row.get('ToGrade', 'N/A'),
                            'from_grade': row.get('FromGrade', 'N/A'),
                            'action': row.get('Action', 'N/A'),
                            'price_target_action': row.get('priceTargetAction', 'N/A'),
                            'current_price_target': row.get('currentPriceTarget', None),
                            'prior_price_target': row.get('priorPriceTarget', None)
                        })
                    # Latest rating date
                    data['latest_rating_date'] = str(upgrades_df.index[0])[:10]
            except Exception:
                data['recent_ratings'] = []
            
        except Exception as e:
            rate_tracker.check_rate_limit_error(str(e), ticker)
            data = {}
        
        return data
    
    def get_daily_close_prices(self, tickers: list, start: str, end: str) -> Dict[str, Any]:
        """Download daily close prices for multiple tickers, return successful and failed."""
        import pandas as pd
        import time
        successful = {}
        failed = []
        for ticker in tickers:
            try:
                rate_tracker.track_request(ticker)
                df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
                if len(df) > 1:
                    successful[ticker] = df["Close"][ticker]
                else:
                    failed.append(ticker)
                time.sleep(0.2)
            except Exception as e:
                is_rate_limited = rate_tracker.check_rate_limit_error(str(e), ticker)
                if is_rate_limited:
                    try:
                        rate_tracker.track_request(ticker)
                        df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
                        if len(df) > 1:
                            successful[ticker] = df["Close"][ticker]
                        else:
                            failed.append(ticker)
                    except Exception:
                        failed.append(ticker)
                else:
                    failed.append(ticker)
        prices = pd.DataFrame(successful)
        return {"prices": prices, "failed": failed}

    def get_management_data(self, ticker: str) -> Dict[str, Any]:
        """Get management data from Yahoo Finance"""
        try:
            rate_tracker.track_request(ticker)
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract available management-related data
            management_data = {
                'company_officers': info.get('companyOfficers', []),
                'governance_epoch_date': info.get('governanceEpochDate'),
                'compensation_risk': info.get('compensationRisk'),
                'shareholder_rights_risk': info.get('shareHolderRightsRisk'),
                'board_risk': info.get('boardRisk'),
                'audit_risk': info.get('auditRisk'),
                'overall_risk': info.get('overallRisk'),
                'held_percent_insiders': info.get('heldPercentInsiders'),
                'held_percent_institutions': info.get('heldPercentInstitutions'),
                'float_shares': info.get('floatShares'),
                'shares_outstanding': info.get('sharesOutstanding'),
                'implied_shares_outstanding': info.get('impliedSharesOutstanding')
            }
            
            # Process company officers data
            officers_summary = []
            for officer in management_data.get('company_officers', [])[:5]:  # Top 5 officers
                officers_summary.append({
                    'name': officer.get('name', 'N/A'),
                    'title': officer.get('title', 'N/A'),
                    'age': officer.get('age'),
                    'total_pay': officer.get('totalPay'),
                    'exercised_value': officer.get('exercisedValue'),
                    'unexercised_value': officer.get('unexercisedValue')
                })
            
            return {
                'ticker': ticker,
                'management_data': management_data,
                'officers_summary': officers_summary,
                'data_source': 'Yahoo Finance'
            }
            
        except Exception as e:
            rate_tracker.check_rate_limit_error(str(e), ticker)
            return {'error': str(e)}