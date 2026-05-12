import yfinance as yf
import pandas as pd

etfs = ['XLK', 'XLE', 'XLF', 'XLV', 'SPY', 'QQQ', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']

print("Testing ETF downloads from yfinance:")
print()
for ticker in etfs:
    try:
        df = yf.download(ticker, start='2025-04-01', end='2026-05-10', interval='1d', progress=False, auto_adjust=True)
        if df.empty:
            print(f"  {ticker:6s}: EMPTY")
        else:
            # Check column structure
            if isinstance(df.columns, pd.MultiIndex):
                print(f"  {ticker:6s}: {len(df)} rows, MultiIndex columns: {list(df.columns.get_level_values(0).unique())}")
                # Try to get Close
                try:
                    close = df["Close"][ticker]
                    print(f"          Close[{ticker}]: {len(close)} values, first={close.iloc[0]:.2f}")
                except:
                    print(f"          Close[{ticker}]: FAILED")
                    # Try without ticker level
                    try:
                        close = df["Close"]
                        print(f"          Close (no ticker): {type(close)}, shape={close.shape}")
                    except Exception as e:
                        print(f"          Close fallback failed: {e}")
            else:
                print(f"  {ticker:6s}: {len(df)} rows, columns: {list(df.columns)}")
                if "Close" in df.columns:
                    print(f"          Close: first={df['Close'].iloc[0]:.2f}, last={df['Close'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"  {ticker:6s}: ERROR - {e}")
