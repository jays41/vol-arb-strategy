import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


class DeltaHedger:
    """
    Delta hedging calculations for options positions.
    Calculates exact number of shares needed to neutralize delta exposure.
    """
    
    def __init__(self, spot_price, risk_free_rate, dividend_yield=0.0):
        """
        Initialize the delta hedger.
        
        Parameters:
        -----------
        spot_price : float
            Current spot price
        risk_free_rate : float
            Risk-free rate (annualized)
        dividend_yield : float
            Dividend yield (annualized)
        """
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
    
    def black_scholes_call(self, S, K, T, r, sigma, q=0.0):
        """Calculate Black-Scholes call price."""
        if T <= 0:
            return max(S - K, 0)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    
    def black_scholes_put(self, S, K, T, r, sigma, q=0.0):
        """Calculate Black-Scholes put price."""
        if T <= 0:
            return max(K - S, 0)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        return put_price
    
    def calculate_call_delta(self, S, K, T, r, sigma, q=0.0):
        """Calculate call delta (sensitivity to spot price change)."""
        if T <= 0:
            return 1.0 if S > K else 0.0
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return np.exp(-q * T) * norm.cdf(d1)
    
    def calculate_put_delta(self, S, K, T, r, sigma, q=0.0):
        """Calculate put delta (sensitivity to spot price change)."""
        if T <= 0:
            return -1.0 if S < K else 0.0
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return -np.exp(-q * T) * norm.cdf(-d1)
    
    def calculate_straddle_delta(self, S, K, T, r, sigma, q=0.0):
        """
        Calculate delta for a long straddle (long call + long put).
        
        Parameters:
        -----------
        S : float
            Current spot price
        K : float
            Strike price
        T : float
            Time to expiration (years)
        r : float
            Risk-free rate
        sigma : float
            Implied volatility
        q : float
            Dividend yield
        
        Returns:
        --------
        float : Straddle delta
        """
        call_delta = self.calculate_call_delta(S, K, T, r, sigma, q)
        put_delta = self.calculate_put_delta(S, K, T, r, sigma, q)
        straddle_delta = call_delta + put_delta
        return straddle_delta
    
    def calculate_hedge_position(self, S, K, T, sigma):
        """
        Calculate exact number of shares needed to hedge delta.
        
        Parameters:
        -----------
        S : float
            Current spot price
        K : float
            Strike price
        T : float
            Time to expiration (years)
        sigma : float
            Implied volatility
        
        Returns:
        --------
        dict : Hedge position details
        """
        straddle_delta = self.calculate_straddle_delta(S, K, T, self.risk_free_rate, sigma, self.dividend_yield)
        
        # To neutralize delta, we need to short straddle_delta shares
        # If straddle_delta is positive, we short (sell) that many shares
        # If straddle_delta is negative, we long (buy) that many shares
        hedge_units = -straddle_delta
        hedge_direction = 'LONG' if hedge_units > 0 else 'SHORT'
        hedge_cost = abs(hedge_units) * S
        
        # Verify delta neutrality
        residual_delta = straddle_delta + (-hedge_units)
        is_delta_neutral = abs(residual_delta) < 0.001
        
        return {
            'straddle_delta': straddle_delta,
            'hedge_units': hedge_units,
            'hedge_direction': hedge_direction,
            'hedge_cost': hedge_cost if hedge_direction == 'LONG' else -hedge_cost,
            'residual_delta': residual_delta,
            'is_delta_neutral': is_delta_neutral
        }
    
    def needs_rehedge(self, old_delta, new_delta, rehedge_threshold=0.10):
        """
        Determine if position needs rehedging.
        
        Parameters:
        -----------
        old_delta : float
            Previous delta
        new_delta : float
            Current delta
        rehedge_threshold : float
            Delta change threshold to trigger rehedge
        
        Returns:
        --------
        bool : True if rehedge needed
        """
        delta_change = abs(new_delta - old_delta)
        return delta_change >= rehedge_threshold
    
    def analyze_rehedge_points(self, K, T, sigma, spot_range=0.10):
        """
        Analyze rehedge requirements across spot price range.
        
        Parameters:
        -----------
        K : float
            Strike price
        T : float
            Time to expiration
        sigma : float
            Implied volatility
        spot_range : float
            Range of spot prices to analyze (e.g., 0.10 = ±10%)
        
        Returns:
        --------
        DataFrame : Rehedge analysis
        """
        # Generate spot prices from -spot_range% to +spot_range%
        spot_moves = np.linspace(-spot_range, spot_range, 21)
        spots = self.spot_price * (1 + spot_moves)
        
        rehedge_data = []
        initial_delta = self.calculate_straddle_delta(self.spot_price, K, T, self.risk_free_rate, sigma, self.dividend_yield)
        
        for spot in spots:
            current_delta = self.calculate_straddle_delta(spot, K, T, self.risk_free_rate, sigma, self.dividend_yield)
            delta_change = abs(current_delta - initial_delta)
            needs_rehedge = delta_change >= 0.10
            
            # New hedge position at this spot
            new_hedge = self.calculate_hedge_position(spot, K, T, sigma)
            
            rehedge_data.append({
                'spot_price': spot,
                'spot_move_pct': spot_moves[len(rehedge_data)]*100,
                'delta': current_delta,
                'delta_change': delta_change,
                'needs_rehedge': needs_rehedge,
                'new_hedge_units': new_hedge['hedge_units'],
                'new_hedge_direction': new_hedge['hedge_direction']
            })
        
        return pd.DataFrame(rehedge_data)
    
    def calculate_gamma_pnl(self, K, T, sigma, spot_move):
        """
        Calculate P&L from gamma exposure (profit from spot movement).
        
        Parameters:
        -----------
        K : float
            Strike price
        T : float
            Time to expiration
        sigma : float
            Implied volatility
        spot_move : float
            Spot price movement in dollars
        
        Returns:
        --------
        float : P&L from gamma
        """
        d1 = (np.log(self.spot_price / K) + (self.risk_free_rate - self.dividend_yield + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        gamma = np.exp(-self.dividend_yield * T) * norm.pdf(d1) / (self.spot_price * sigma * np.sqrt(T))
        gamma_pnl = 0.5 * gamma * (spot_move ** 2)
        return gamma_pnl
    
    def calculate_vega_pnl(self, K, T, current_iv, forecast_iv):
        """
        Calculate P&L if IV moves to forecast level.
        
        Parameters:
        -----------
        K : float
            Strike price
        T : float
            Time to expiration
        current_iv : float
            Current implied volatility
        forecast_iv : float
            Forecasted implied volatility
        
        Returns:
        --------
        dict : Vega P&L analysis
        """
        # Current straddle value
        call_curr = self.black_scholes_call(self.spot_price, K, T, self.risk_free_rate, current_iv, self.dividend_yield)
        put_curr = self.black_scholes_put(self.spot_price, K, T, self.risk_free_rate, current_iv, self.dividend_yield)
        straddle_curr = call_curr + put_curr
        
        # Value if IV moves to forecast
        call_forecast = self.black_scholes_call(self.spot_price, K, T, self.risk_free_rate, forecast_iv, self.dividend_yield)
        put_forecast = self.black_scholes_put(self.spot_price, K, T, self.risk_free_rate, forecast_iv, self.dividend_yield)
        straddle_forecast = call_forecast + put_forecast
        
        vega_pnl = straddle_forecast - straddle_curr
        vega_pnl_pct = (vega_pnl / straddle_curr * 100) if straddle_curr != 0 else 0
        
        # Calculate actual vega (sensitivity per 1% IV change)
        d1 = (np.log(self.spot_price / K) + (self.risk_free_rate - self.dividend_yield + 0.5 * current_iv**2) * T) / (current_iv * np.sqrt(T))
        vega = self.spot_price * np.exp(-self.dividend_yield * T) * norm.pdf(d1) * np.sqrt(T) / 100
        
        return {
            'current_value': straddle_curr,
            'forecast_value': straddle_forecast,
            'vega_pnl': vega_pnl,
            'vega_pnl_pct': vega_pnl_pct,
            'vega_per_1pct': vega,
            'iv_move': (forecast_iv - current_iv) * 100
        }
    
    def calculate_theta_pnl(self, K, T, sigma, days=1):
        """
        Calculate daily theta decay (time value loss).
        
        Parameters:
        -----------
        K : float
            Strike price
        T : float
            Time to expiration
        sigma : float
            Implied volatility
        days : int
            Number of days to calculate theta for
        
        Returns:
        --------
        float : Theta P&L for specified days
        """
        # Theta is typically calculated per day and is negative for long options
        d1 = (np.log(self.spot_price / K) + (self.risk_free_rate - self.dividend_yield + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Call theta
        call_theta = (-self.spot_price * np.exp(-self.dividend_yield * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                      + self.dividend_yield * self.spot_price * np.exp(-self.dividend_yield * T) * norm.cdf(d1)
                      - self.risk_free_rate * K * np.exp(-self.risk_free_rate * T) * norm.cdf(d2))
        
        # Put theta
        put_theta = (-self.spot_price * np.exp(-self.dividend_yield * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                     - self.dividend_yield * self.spot_price * np.exp(-self.dividend_yield * T) * norm.cdf(-d1)
                     + self.risk_free_rate * K * np.exp(-self.risk_free_rate * T) * norm.cdf(-d2))
        
        straddle_theta = (call_theta + put_theta) / 365 * days
        return straddle_theta
    
    def simulate_hedge_pnl(self, K, T, current_iv, forecast_iv, spot_moves=None):
        """
        Simulate P&L across different spot moves and IV scenarios.
        
        Parameters:
        -----------
        K : float
            Strike price
        T : float
            Time to expiration
        current_iv : float
            Current IV
        forecast_iv : float
            Forecasted IV
        spot_moves : array-like
            Spot moves to test (in %)
        
        Returns:
        --------
        DataFrame : P&L simulation results
        """
        if spot_moves is None:
            spot_moves = np.linspace(-0.10, 0.10, 21)  # -10% to +10%
        
        vega_analysis = self.calculate_vega_pnl(K, T, current_iv, forecast_iv)
        vega_pnl = vega_analysis['vega_pnl']
        
        results = []
        for move in spot_moves:
            spot_change = self.spot_price * move
            gamma_pnl = self.calculate_gamma_pnl(K, T, current_iv, spot_change)
            theta_pnl = self.calculate_theta_pnl(K, T, current_iv, days=1)
            
            # Total P&L = gamma (from spot move) + vega (from IV change) + theta (daily decay)
            total_pnl = gamma_pnl + vega_pnl + theta_pnl
            
            results.append({
                'spot_move_pct': move * 100,
                'new_spot': self.spot_price * (1 + move),
                'gamma_pnl': gamma_pnl,
                'vega_pnl': vega_pnl,
                'theta_pnl': theta_pnl,
                'total_pnl': total_pnl
            })
        
        return pd.DataFrame(results)
    
    def plot_rehedge_requirements(self, K, T, sigma, spot_range=0.10):
        """Plot rehedge triggers across spot price range."""
        df = self.analyze_rehedge_points(K, T, sigma, spot_range)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Delta and delta change
        ax1.plot(df['spot_price'], df['delta'], 'b-', linewidth=2, label='Straddle Delta')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax1.fill_between(df['spot_price'], -0.10, 0.10, alpha=0.2, color='green', label='No Rehedge Zone (±0.10)')
        ax1.set_ylabel('Delta', fontsize=10)
        ax1.set_title('Straddle Delta vs Spot Price', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Rehedge triggers
        rehedge_mask = df['needs_rehedge']
        ax2.scatter(df[rehedge_mask]['spot_price'], df[rehedge_mask]['delta_change'], 
                   color='red', s=100, label='Rehedge Required', zorder=5)
        ax2.scatter(df[~rehedge_mask]['spot_price'], df[~rehedge_mask]['delta_change'], 
                   color='green', s=50, label='No Rehedge', alpha=0.5)
        ax2.axhline(y=0.10, color='red', linestyle='--', alpha=0.5, label='Rehedge Threshold')
        ax2.set_xlabel('Spot Price', fontsize=10)
        ax2.set_ylabel('Delta Change from Initial', fontsize=10)
        ax2.set_title('Rehedge Triggers', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_hedge_pnl_breakdown(self, K, T, current_iv, forecast_iv):
        """Plot P&L breakdown (gamma, vega, theta)."""
        df = self.simulate_hedge_pnl(K, T, current_iv, forecast_iv)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Stacked area chart
        ax.fill_between(df['spot_move_pct'], 0, df['gamma_pnl'], alpha=0.6, label='Gamma P&L')
        ax.fill_between(df['spot_move_pct'], df['gamma_pnl'], 
                       df['gamma_pnl'] + df['vega_pnl'], alpha=0.6, label='Vega P&L')
        ax.fill_between(df['spot_move_pct'], df['gamma_pnl'] + df['vega_pnl'], 
                       df['total_pnl'], alpha=0.6, label='Theta P&L')
        
        ax.plot(df['spot_move_pct'], df['total_pnl'], 'k-', linewidth=2, label='Total P&L')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Spot Move (%)', fontsize=10)
        ax.set_ylabel('P&L', fontsize=10)
        ax.set_title('Delta-Hedged Straddle P&L Breakdown (Gamma + Vega + Theta)', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def print_hedge_summary(self, K, T, sigma, forecast_iv=None):
        """Print summary of hedge position and exposure."""
        hedge = self.calculate_hedge_position(self.spot_price, K, T, sigma)
        
        print("\n" + "="*80)
        print("DELTA HEDGE SUMMARY")
        print("="*80)
        
        print(f"\nPosition Details:")
        print(f"  Spot Price:              ${self.spot_price:.2f}")
        print(f"  Strike:                  ${K:.2f}")
        print(f"  Time to Expiration:      {T:.3f} years ({T*12:.0f} months)")
        print(f"  Implied Volatility:      {sigma:.2%}")
        
        print(f"\nDelta Analysis:")
        print(f"  Straddle Delta:          {hedge['straddle_delta']:+.6f}")
        print(f"  Hedge Direction:         {hedge['hedge_direction']}")
        print(f"  Hedge Units (shares):    {abs(hedge['hedge_units']):.6f}")
        print(f"  Hedge Cost:              ${abs(hedge['hedge_cost']):.2f}")
        print(f"  Residual Delta:          {hedge['residual_delta']:+.6f}")
        print(f"  Delta Neutral:           {'✓ YES' if hedge['is_delta_neutral'] else '✗ NO'}")
        
        if forecast_iv:
            vega_analysis = self.calculate_vega_pnl(K, T, sigma, forecast_iv)
            print(f"\nVega Exposure (if IV → {forecast_iv:.2%}):")
            print(f"  Current Value:           ${vega_analysis['current_value']:.4f}")
            print(f"  Forecast Value:          ${vega_analysis['forecast_value']:.4f}")
            print(f"  Vega P&L:                ${vega_analysis['vega_pnl']:+.4f} ({vega_analysis['vega_pnl_pct']:+.1f}%)")
        
        print("\n" + "="*80)


# Example usage
if __name__ == '__main__':
    # Example parameters
    spot_price = 100
    strike = 100
    risk_free_rate = 0.05
    dividend_yield = 0.02
    sigma = 0.25
    T = 0.5  # 6 months
    
    # Create hedger
    hedger = DeltaHedger(spot_price, risk_free_rate, dividend_yield)
    
    # Calculate hedge position
    hedge = hedger.calculate_hedge_position(spot_price, strike, T, sigma)
    
    print("Hedge Position:")
    print(f"  Straddle Delta: {hedge['straddle_delta']:.6f}")
    print(f"  Hedge Units: {abs(hedge['hedge_units']):.6f} shares ({hedge['hedge_direction']})")
    print(f"  Delta Neutral: {'✓' if hedge['is_delta_neutral'] else '✗'}")
    
    # Analyze rehedge points
    print("\nRehedge Analysis:")
    rehedge_df = hedger.analyze_rehedge_points(strike, T, sigma, spot_range=0.10)
    print(rehedge_df[rehedge_df['needs_rehedge']].to_string(index=False))
    
    # Print summary
    hedger.print_hedge_summary(strike, T, sigma, forecast_iv=0.30)
