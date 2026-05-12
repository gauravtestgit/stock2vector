import time
from collections import defaultdict
import threading

class RateLimitTracker:
    def __init__(self):
        self.request_count = 0
        self.rate_limit_errors = 0
        self.all_errors = []
        self.request_times = []
        self.ticker_requests = defaultdict(int)
        self.error_patterns = defaultdict(int)
        self.last_rate_limit_time = 0
        self.rate_limit_lock = threading.Lock()
    
    def track_request(self, ticker=""):
        self.request_count += 1
        current_time = time.time()
        self.request_times.append(current_time)
        if ticker:
            self.ticker_requests[ticker] += 1
        
        if self.request_count % 100 == 0:
            recent_rate = self._calculate_recent_rate()
            print(f"🔄 YFinance: {self.request_count} requests, Rate: {recent_rate:.1f}/min, Errors: {self.rate_limit_errors}")
    
    def check_rate_limit_error(self, error_msg: str, ticker: str = "") -> bool:
        """Check for rate limit error and pause if detected. Returns True if rate limited."""
        # Log all errors for analysis
        self.all_errors.append(f"{ticker}: {error_msg}")
        self.error_patterns[error_msg.lower()[:50]] += 1
        
        # Check for rate limit indicators
        rate_limit_indicators = ['too many requests', '429', 'rate limit', 'throttle', 'quota exceeded', 'monthly limit', 'daily limit']
        if any(indicator in error_msg.lower() for indicator in rate_limit_indicators):
            with self.rate_limit_lock:
                self.rate_limit_errors += 1
                current_time = time.time()
                
                # Only pause if we haven't paused recently (avoid multiple threads all pausing)
                if current_time - self.last_rate_limit_time > 30:
                    self.last_rate_limit_time = current_time
                    wait_time = 60
                    print(f"\n⚠️ RATE LIMIT HIT! Pausing all threads for {wait_time}s...")
                    print(f"   Total rate limit errors: {self.rate_limit_errors}")
                    print(f"   Ticker: {ticker}, Error: {error_msg}")
                    time.sleep(wait_time)
                    print(f"✅ Resuming after {wait_time}s pause\n")
                else:
                    # Another thread already paused recently, just wait a bit
                    time.sleep(5)
                
                return True
        else:
            # Check for internal throttling patterns
            # Only log unexpected errors, suppress common non-critical ones
            suppressed_patterns = ['no data found', 'possibly delisted', 'quote not found', 'nonetype']
            if not any(pattern in error_msg.lower() for pattern in suppressed_patterns):
                print(f"🔍 YFinance Error ({ticker}): {error_msg}")
        
        return False
    
    def get_error_summary(self):
        return {
            'total_requests': self.request_count,
            'rate_limit_errors': self.rate_limit_errors,
            'all_errors': self.all_errors[-10:],  # Last 10 errors
            'request_rate_per_min': self._calculate_recent_rate(),
            'top_error_patterns': dict(sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[:5]),
            'ticker_request_counts': dict(self.ticker_requests)
        }
    
    def _calculate_recent_rate(self):
        """Calculate requests per minute for last 60 seconds"""
        if not self.request_times:
            return 0.0
        
        current_time = time.time()
        recent_requests = [t for t in self.request_times if current_time - t <= 60]
        return len(recent_requests)

    def print_analysis(self):
        """Print detailed analysis of request patterns"""
        print("\n📊 YFinance Request Analysis:")
        print(f"Total Requests: {self.request_count}")
        print(f"Current Rate: {self._calculate_recent_rate():.1f} requests/minute")
        print(f"Rate Limit Errors: {self.rate_limit_errors}")
        print(f"Other Errors: {len(self.all_errors) - self.rate_limit_errors}")
        
        if self.error_patterns:
            print("\nTop Error Patterns:")
            for pattern, count in sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"  • {pattern}: {count} times")
        
        if self.ticker_requests:
            failed_tickers = [ticker for ticker, count in self.ticker_requests.items() if count > 1]
            if failed_tickers:
                print(f"\nTickers with multiple requests (potential retries): {failed_tickers}")

# Global instance
rate_tracker = RateLimitTracker()