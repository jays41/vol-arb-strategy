import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import brentq
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


class ImpliedVolSurface:
    """
    Class to calculate and visualize implied volatility surfaces for options.
    Uses Black-Scholes model to back out implied volatility from option prices.
    """
    
    def __init__(self, spot_price, risk_free_rate, dividend_yield=0.0):
        """
        Initialize the IV Surface calculator.
        
        Parameters:
        -----------
        spot_price : float
            Current spot price of the underlying
        risk_free_rate : float
            Risk-free rate (annualized)
        dividend_yield : float
            Dividend yield (annualized)
        """
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
    
    def black_scholes_call(self, S, K, T, r, sigma, q=0.0):
        """
        Calculate Black-Scholes call option price.
        
        Parameters:
        -----------
        S : float
            Spot price
        K : float
            Strike price
        T : float
            Time to expiration (in years)
        r : float
            Risk-free rate
        sigma : float
            Volatility
        q : float
            Dividend yield
        
        Returns:
        --------
        float : Call option price
        """
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    
    def black_scholes_put(self, S, K, T, r, sigma, q=0.0):
        """
        Calculate Black-Scholes put option price.
        """
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        return put_price
    
    def vega(self, S, K, T, r, sigma, q=0.0):
        """
        Calculate vega (sensitivity to volatility change).
        """
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega_value = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        return vega_value
    
    def implied_volatility(self, market_price, K, T, option_type='call', initial_guess=0.2):
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Parameters:
        -----------
        market_price : float
            Market price of the option
        K : float
            Strike price
        T : float
            Time to expiration (in years)
        option_type : str
            'call' or 'put'
        initial_guess : float
            Initial guess for volatility
        
        Returns:
        --------
        float : Implied volatility
        """
        if option_type == 'call':
            objective = lambda sigma: self.black_scholes_call(
                self.spot_price, K, T, self.risk_free_rate, sigma, self.dividend_yield
            ) - market_price
        else:
            objective = lambda sigma: self.black_scholes_put(
                self.spot_price, K, T, self.risk_free_rate, sigma, self.dividend_yield
            ) - market_price
        
        try:
            iv = brentq(objective, 0.001, 5.0)
            return iv
        except ValueError:
            return np.nan
    
    def generate_surface_data(self, strikes, maturities, market_prices, option_type='call'):
        """
        Generate implied volatility surface data from market prices.
        
        Parameters:
        -----------
        strikes : array-like
            Array of strike prices
        maturities : array-like
            Array of time to maturities (in years)
        market_prices : 2D array
            Matrix of option prices (rows=strikes, cols=maturities)
        option_type : str
            'call' or 'put'
        
        Returns:
        --------
        tuple : (strikes, maturities, iv_surface)
        """
        strikes = np.asarray(strikes)
        maturities = np.asarray(maturities)
        
        iv_surface = np.zeros((len(strikes), len(maturities)))
        
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                price = market_prices[i, j]
                iv = self.implied_volatility(price, K, T, option_type)
                iv_surface[i, j] = iv
        
        return strikes, maturities, iv_surface
    
    def plot_surface_3d(self, strikes, maturities, iv_surface, title='Implied Volatility Surface'):
        """
        Create a 3D plot of the implied volatility surface.
        """
        X, Y = np.meshgrid(maturities, strikes)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X, Y, iv_surface, cmap='viridis', alpha=0.8, edgecolor='none')
        
        ax.set_xlabel('Time to Maturity (years)', fontsize=10)
        ax.set_ylabel('Strike Price', fontsize=10)
        ax.set_zlabel('Implied Volatility', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        fig.colorbar(surf, ax=ax, label='IV', shrink=0.5)
        plt.tight_layout()
        
        return fig, ax
    
    def plot_surface_contour(self, strikes, maturities, iv_surface, title='IV Surface Contour'):
        """
        Create a contour plot of the implied volatility surface.
        """
        X, Y = np.meshgrid(maturities, strikes)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        contour = ax.contourf(X, Y, iv_surface, levels=20, cmap='viridis')
        contour_lines = ax.contour(X, Y, iv_surface, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax.clabel(contour_lines, inline=True, fontsize=8)
        
        ax.set_xlabel('Time to Maturity (years)', fontsize=10)
        ax.set_ylabel('Strike Price', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        cbar = fig.colorbar(contour, ax=ax, label='IV')
        plt.tight_layout()
        
        return fig, ax
    
    def plot_term_structure(self, strikes, maturities, iv_surface):
        """
        Plot IV term structure for different strikes (IV vs Time to Maturity).
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Select specific strikes to plot
        strike_indices = [0, len(strikes)//4, len(strikes)//2, 3*len(strikes)//4, -1]
        
        for idx in strike_indices:
            strike = strikes[idx]
            ivs = iv_surface[idx, :]
            label = f'Strike: {strike:.0f}'
            ax.plot(maturities, ivs, marker='o', label=label, linewidth=2)
        
        ax.set_xlabel('Time to Maturity (years)', fontsize=10)
        ax.set_ylabel('Implied Volatility', fontsize=10)
        ax.set_title('IV Term Structure', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig, ax
    
    def plot_smile(self, maturities, strikes, iv_surface):
        """
        Plot volatility smile for different maturities (IV vs Strike).
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Select specific maturities to plot
        maturity_indices = [0, len(maturities)//3, 2*len(maturities)//3, -1]
        
        for idx in maturity_indices:
            maturity = maturities[idx]
            ivs = iv_surface[:, idx]
            label = f'T = {maturity:.2f} years'
            ax.plot(strikes, ivs, marker='s', label=label, linewidth=2)
        
        ax.set_xlabel('Strike Price', fontsize=10)
        ax.set_ylabel('Implied Volatility', fontsize=10)
        ax.set_title('Volatility Smile', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig, ax
    
    def calculate_straddle_cost(self, K, T, sigma):
        """
        Calculate the cost of a straddle (long call + long put at same strike).
        
        Parameters:
        -----------
        K : float
            Strike price
        T : float
            Time to expiration
        sigma : float
            Implied volatility
        
        Returns:
        --------
        float : Total straddle cost
        """
        call_price = self.black_scholes_call(self.spot_price, K, T, self.risk_free_rate, sigma, self.dividend_yield)
        put_price = self.black_scholes_put(self.spot_price, K, T, self.risk_free_rate, sigma, self.dividend_yield)
        straddle_cost = call_price + put_price
        return straddle_cost
    
    def calculate_straddle_greeks(self, K, T, sigma):
        """
        Calculate Greeks for a long straddle position.
        
        Parameters:
        -----------
        K : float
            Strike price
        T : float
            Time to expiration
        sigma : float
            Implied volatility
        
        Returns:
        --------
        dict : Greeks (delta, gamma, vega, theta)
        """
        d1 = (np.log(self.spot_price / K) + (self.risk_free_rate - self.dividend_yield + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Deltas (call delta + put delta)
        call_delta = np.exp(-self.dividend_yield * T) * norm.cdf(d1)
        put_delta = -np.exp(-self.dividend_yield * T) * norm.cdf(-d1)
        straddle_delta = call_delta + put_delta
        
        # Gamma (same for call and put)
        gamma = np.exp(-self.dividend_yield * T) * norm.pdf(d1) / (self.spot_price * sigma * np.sqrt(T))
        
        # Vega (same for call and put, per 1% change)
        vega = self.spot_price * np.exp(-self.dividend_yield * T) * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Theta (call theta + put theta)
        call_theta = (-self.spot_price * np.exp(-self.dividend_yield * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                      + self.dividend_yield * self.spot_price * np.exp(-self.dividend_yield * T) * norm.cdf(d1)
                      - self.risk_free_rate * K * np.exp(-self.risk_free_rate * T) * norm.cdf(d2)) / 365
        
        put_theta = (-self.spot_price * np.exp(-self.dividend_yield * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                     - self.dividend_yield * self.spot_price * np.exp(-self.dividend_yield * T) * norm.cdf(-d1)
                     + self.risk_free_rate * K * np.exp(-self.risk_free_rate * T) * norm.cdf(-d2)) / 365
        
        straddle_theta = call_theta + put_theta
        
        return {
            'delta': straddle_delta,
            'gamma': gamma,
            'vega': vega,
            'theta': straddle_theta,
            'cost': self.calculate_straddle_cost(K, T, sigma)
        }
    
    def find_optimal_straddle(self, strikes, maturities, iv_surface, metric='cheapest'):
        """
        Find the optimal straddle position across the IV surface.
        
        Parameters:
        -----------
        strikes : array-like
            Array of strike prices
        maturities : array-like
            Array of time to maturities
        iv_surface : 2D array
            IV surface matrix
        metric : str
            'cheapest' - minimum straddle cost
            'highest_gamma' - maximum gamma (best for direction move)
            'best_vega_carry' - high vega relative to theta decay
        
        Returns:
        --------
        dict : Optimal straddle details
        """
        results = []
        
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                sigma = iv_surface[i, j]
                if np.isnan(sigma):
                    continue
                
                greeks = self.calculate_straddle_greeks(K, T, sigma)
                
                # Calculate custom metrics
                atm_distance = abs(K - self.spot_price) / self.spot_price
                vega_theta_ratio = abs(greeks['vega'] / greeks['theta']) if greeks['theta'] != 0 else 0
                
                results.append({
                    'strike': K,
                    'maturity': T,
                    'iv': sigma,
                    'atm_distance': atm_distance,
                    'cost': greeks['cost'],
                    'delta': greeks['delta'],
                    'gamma': greeks['gamma'],
                    'vega': greeks['vega'],
                    'theta': greeks['theta'],
                    'vega_theta_ratio': vega_theta_ratio,
                    'strike_idx': i,
                    'maturity_idx': j
                })
        
        df_results = pd.DataFrame(results)
        
        if metric == 'cheapest':
            optimal = df_results.loc[df_results['cost'].idxmin()]
        elif metric == 'highest_gamma':
            optimal = df_results.loc[df_results['gamma'].idxmax()]
        elif metric == 'best_vega_carry':
            optimal = df_results.loc[df_results['vega_theta_ratio'].idxmax()]
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return optimal, df_results
    
    def plot_straddle_surface(self, strikes, maturities, iv_surface, metric='cost'):
        """
        Plot straddle metrics (cost, gamma, vega) across the surface.
        
        Parameters:
        -----------
        strikes : array-like
            Array of strike prices
        maturities : array-like
            Array of time to maturities
        iv_surface : 2D array
            IV surface matrix
        metric : str
            'cost', 'gamma', 'vega', or 'theta'
        
        Returns:
        --------
        tuple : (fig, ax)
        """
        metric_surface = np.zeros((len(strikes), len(maturities)))
        
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                sigma = iv_surface[i, j]
                if np.isnan(sigma):
                    continue
                
                greeks = self.calculate_straddle_greeks(K, T, sigma)
                
                if metric == 'cost':
                    metric_surface[i, j] = greeks['cost']
                elif metric == 'gamma':
                    metric_surface[i, j] = greeks['gamma']
                elif metric == 'vega':
                    metric_surface[i, j] = greeks['vega']
                elif metric == 'theta':
                    metric_surface[i, j] = greeks['theta']
        
        X, Y = np.meshgrid(maturities, strikes)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X, Y, metric_surface, cmap='RdYlGn', alpha=0.8, edgecolor='none')
        
        ax.set_xlabel('Time to Maturity (years)', fontsize=10)
        ax.set_ylabel('Strike Price', fontsize=10)
        ax.set_zlabel(f'Straddle {metric.capitalize()}', fontsize=10)
        ax.set_title(f'Straddle {metric.capitalize()} Across Surface', fontsize=12, fontweight='bold')
        
        fig.colorbar(surf, ax=ax, label=metric.capitalize(), shrink=0.5)
        plt.tight_layout()
        
        return fig, ax
    
    def plot_optimal_straddles(self, strikes, maturities, iv_surface):
        """
        Highlight optimal straddles across different metrics on a 2D contour.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        X, Y = np.meshgrid(maturities, strikes)
        
        metrics = ['cheapest', 'highest_gamma', 'best_vega_carry']
        metric_labels = ['Cost', 'Gamma', 'Vega/Theta Ratio']
        
        # Plot cost surface with optimal
        cost_surface = np.zeros((len(strikes), len(maturities)))
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                sigma = iv_surface[i, j]
                if not np.isnan(sigma):
                    cost_surface[i, j] = self.calculate_straddle_cost(K, T, sigma)
        
        ax = axes[0, 0]
        contour = ax.contourf(X, Y, cost_surface, levels=20, cmap='viridis')
        optimal_cheap, _ = self.find_optimal_straddle(strikes, maturities, iv_surface, 'cheapest')
        ax.plot(optimal_cheap['maturity'], optimal_cheap['strike'], 'r*', markersize=20, label='Optimal')
        ax.set_xlabel('Time to Maturity')
        ax.set_ylabel('Strike Price')
        ax.set_title('Straddle Cost (Cheapest)')
        ax.legend()
        fig.colorbar(contour, ax=ax)
        
        # Plot gamma surface with optimal
        gamma_surface = np.zeros((len(strikes), len(maturities)))
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                sigma = iv_surface[i, j]
                if not np.isnan(sigma):
                    gamma_surface[i, j] = self.calculate_straddle_greeks(K, T, sigma)['gamma']
        
        ax = axes[0, 1]
        contour = ax.contourf(X, Y, gamma_surface, levels=20, cmap='plasma')
        optimal_gamma, _ = self.find_optimal_straddle(strikes, maturities, iv_surface, 'highest_gamma')
        ax.plot(optimal_gamma['maturity'], optimal_gamma['strike'], 'r*', markersize=20, label='Optimal')
        ax.set_xlabel('Time to Maturity')
        ax.set_ylabel('Strike Price')
        ax.set_title('Straddle Gamma (Highest)')
        ax.legend()
        fig.colorbar(contour, ax=ax)
        
        # Plot vega/theta ratio with optimal
        vega_theta_surface = np.zeros((len(strikes), len(maturities)))
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                sigma = iv_surface[i, j]
                if not np.isnan(sigma):
                    greeks = self.calculate_straddle_greeks(K, T, sigma)
                    vega_theta_surface[i, j] = abs(greeks['vega'] / greeks['theta']) if greeks['theta'] != 0 else 0
        
        ax = axes[1, 0]
        contour = ax.contourf(X, Y, vega_theta_surface, levels=20, cmap='cool')
        optimal_vega, _ = self.find_optimal_straddle(strikes, maturities, iv_surface, 'best_vega_carry')
        ax.plot(optimal_vega['maturity'], optimal_vega['strike'], 'r*', markersize=20, label='Optimal')
        ax.set_xlabel('Time to Maturity')
        ax.set_ylabel('Strike Price')
        ax.set_title('Vega/Theta Ratio (Best Carry)')
        ax.legend()
        fig.colorbar(contour, ax=ax)
        
        # Summary table
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_data = [
            ['Metric', 'Strike', 'Maturity', 'Cost'],
            ['Cheapest', f'{optimal_cheap["strike"]:.2f}', f'{optimal_cheap["maturity"]:.3f}', f'{optimal_cheap["cost"]:.4f}'],
            ['High Gamma', f'{optimal_gamma["strike"]:.2f}', f'{optimal_gamma["maturity"]:.3f}', f'{optimal_gamma["cost"]:.4f}'],
            ['Best V/T', f'{optimal_vega["strike"]:.2f}', f'{optimal_vega["maturity"]:.3f}', f'{optimal_vega["cost"]:.4f}'],
        ]
        
        table = ax.table(cellText=summary_data, cellLoc='center', loc='center', 
                        colWidths=[0.25, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Format header row
        for i in range(4):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Optimal Straddles Summary', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    def calculate_straddle_pnl_at_forecast(self, K, T, current_iv, forecast_iv, spot_move=0):
        """
        Calculate straddle P&L if implied vol moves to forecast level.
        
        Parameters:
        -----------
        K : float
            Strike price
        T : float
            Time to expiration
        current_iv : float
            Current implied volatility
        forecast_iv : float
            Expected/forecasted implied volatility
        spot_move : float
            Optional spot price move (in %)
        
        Returns:
        --------
        dict : P&L analysis
        """
        # Current straddle cost
        current_cost = self.calculate_straddle_cost(K, T, current_iv)
        
        # Value if IV moves to forecast but spot stays same
        forecast_value = self.calculate_straddle_cost(K, T, forecast_iv)
        iv_pnl = forecast_value - current_cost
        
        # Value if spot moves
        spot_move_pnl = 0
        if spot_move != 0:
            new_spot = self.spot_price * (1 + spot_move)
            # Approximate P&L from spot move using gamma
            greeks = self.calculate_straddle_greeks(K, T, forecast_iv)
            spot_change = new_spot - self.spot_price
            spot_move_pnl = greeks['delta'] * spot_change + 0.5 * greeks['gamma'] * (spot_change ** 2)
        
        total_pnl = iv_pnl + spot_move_pnl
        
        return {
            'current_cost': current_cost,
            'forecast_value': forecast_value,
            'iv_pnl': iv_pnl,
            'iv_pnl_pct': (iv_pnl / current_cost * 100) if current_cost != 0 else 0,
            'spot_move_pnl': spot_move_pnl,
            'total_pnl': total_pnl,
            'total_pnl_pct': (total_pnl / current_cost * 100) if current_cost != 0 else 0,
            'current_iv': current_iv,
            'forecast_iv': forecast_iv,
            'iv_difference': forecast_iv - current_iv,
            'iv_difference_pct': ((forecast_iv - current_iv) / current_iv * 100) if current_iv != 0 else 0
        }
    
    def evaluate_straddles_vs_forecast(self, strikes, maturities, iv_surface, forecast_iv):
        """
        Evaluate all optimal straddles against forecasted volatility.
        
        Parameters:
        -----------
        strikes : array-like
            Array of strike prices
        maturities : array-like
            Array of time to maturities
        iv_surface : 2D array
            IV surface matrix
        forecast_iv : float
            Forecasted implied volatility
        
        Returns:
        --------
        dict : Comparison of optimal straddles vs forecast
        """
        optimal_cheap, _ = self.find_optimal_straddle(strikes, maturities, iv_surface, 'cheapest')
        optimal_gamma, _ = self.find_optimal_straddle(strikes, maturities, iv_surface, 'highest_gamma')
        optimal_vega, _ = self.find_optimal_straddle(strikes, maturities, iv_surface, 'best_vega_carry')
        
        # Calculate P&L for each strategy
        cheap_pnl = self.calculate_straddle_pnl_at_forecast(
            optimal_cheap['strike'], optimal_cheap['maturity'], 
            optimal_cheap['iv'], forecast_iv
        )
        
        gamma_pnl = self.calculate_straddle_pnl_at_forecast(
            optimal_gamma['strike'], optimal_gamma['maturity'], 
            optimal_gamma['iv'], forecast_iv
        )
        
        vega_pnl = self.calculate_straddle_pnl_at_forecast(
            optimal_vega['strike'], optimal_vega['maturity'], 
            optimal_vega['iv'], forecast_iv
        )
        
        return {
            'forecast_iv': forecast_iv,
            'cheapest': {**optimal_cheap, 'pnl': cheap_pnl},
            'highest_gamma': {**optimal_gamma, 'pnl': gamma_pnl},
            'best_vega_carry': {**optimal_vega, 'pnl': vega_pnl}
        }
    
    def plot_forecast_comparison(self, strikes, maturities, iv_surface, forecast_iv):
        """
        Plot comparison of current IV surface vs forecast level.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        X, Y = np.meshgrid(maturities, strikes)
        
        # Current IV surface
        ax = axes[0]
        contour1 = ax.contourf(X, Y, iv_surface, levels=20, cmap='RdYlBu_r')
        contour_lines1 = ax.contour(X, Y, iv_surface, levels=10, colors='black', alpha=0.2, linewidths=0.5)
        ax.clabel(contour_lines1, inline=True, fontsize=8)
        ax.set_xlabel('Time to Maturity (years)')
        ax.set_ylabel('Strike Price')
        ax.set_title('Current IV Surface')
        cbar1 = fig.colorbar(contour1, ax=ax, label='IV')
        
        # IV difference from forecast
        iv_diff = iv_surface - forecast_iv
        ax = axes[1]
        contour2 = ax.contourf(X, Y, iv_diff, levels=20, cmap='RdBu_r', 
                               vmin=-np.nanmax(np.abs(iv_diff)), vmax=np.nanmax(np.abs(iv_diff)))
        contour_lines2 = ax.contour(X, Y, iv_diff, levels=10, colors='black', alpha=0.2, linewidths=0.5)
        ax.clabel(contour_lines2, inline=True, fontsize=8)
        ax.axhline(y=self.spot_price, color='green', linestyle='--', linewidth=2, label='ATM Strike')
        ax.set_xlabel('Time to Maturity (years)')
        ax.set_ylabel('Strike Price')
        ax.set_title(f'IV Difference (Current - Forecast {forecast_iv:.1%})')
        cbar2 = fig.colorbar(contour2, ax=ax, label='IV Difference')
        
        # Add text annotation
        fig.text(0.5, 0.02, f'Blue = Market IV below forecast (BUY straddles) | Red = Market IV above forecast (SELL straddles)',
                ha='center', fontsize=10, style='italic')
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        return fig
    
    def print_forecast_analysis(self, strikes, maturities, iv_surface, forecast_iv):
        """
        Print detailed analysis comparing optimal straddles to forecast.
        """
        results = self.evaluate_straddles_vs_forecast(strikes, maturities, iv_surface, forecast_iv)
        
        print("\n" + "="*80)
        print("STRADDLE ANALYSIS vs FORECASTED VOLATILITY")
        print("="*80)
        print(f"\nForecasted IV: {forecast_iv:.2%}\n")
        
        strategies = ['cheapest', 'highest_gamma', 'best_vega_carry']
        strategy_names = ['CHEAPEST STRADDLE', 'HIGHEST GAMMA STRADDLE', 'BEST VEGA/THETA CARRY']
        
        for strategy, name in zip(strategies, strategy_names):
            data = results[strategy]
            pnl = data['pnl']
            
            print(f"\n{name}")
            print("-" * 80)
            print(f"  Position Details:")
            print(f"    Strike:        ${data['strike']:.2f}")
            print(f"    Maturity:      {data['maturity']:.3f} years ({data['maturity']*12:.0f} months)")
            print(f"    Current IV:    {data['iv']:.2%}")
            print(f"    Forecast IV:   {forecast_iv:.2%}")
            print(f"    IV Difference: {pnl['iv_difference']:.2%} ({pnl['iv_difference_pct']:+.1f}%)")
            
            print(f"\n  Trade Setup:")
            if pnl['iv_difference'] < 0:
                print(f"    STATUS:        UNDERPRICED (Market IV < Forecast) → BUY STRADDLE")
            else:
                print(f"    STATUS:        OVERPRICED (Market IV > Forecast) → SELL STRADDLE")
            
            print(f"\n  P&L Analysis (if forecast is correct):")
            print(f"    Entry Cost:    ${pnl['current_cost']:.4f}")
            print(f"    Exit Value:    ${pnl['forecast_value']:.4f}")
            print(f"    IV P&L:        ${pnl['iv_pnl']:.4f} ({pnl['iv_pnl_pct']:+.1f}%)")
            print(f"    Payoff Ratio:  {abs(pnl['total_pnl_pct']):.1f}% return on initial investment")
            
            print(f"\n  Greeks at Entry:")
            greeks = data['greeks'] if 'greeks' in data else self.calculate_straddle_greeks(
                data['strike'], data['maturity'], data['iv']
            )
            print(f"    Delta:  {data['delta']:.4f}")
            print(f"    Gamma:  {data['gamma']:.6f}")
            print(f"    Vega:   {data['vega']:.4f}")
            print(f"    Theta:  {data['theta']:.4f}")
        
        print("\n" + "="*80)


# Example usage
if __name__ == '__main__':
    # Initialize parameters
    spot_price = 100
    risk_free_rate = 0.05
    dividend_yield = 0.02
    
    # Create IV surface calculator
    iv_calc = ImpliedVolSurface(spot_price, risk_free_rate, dividend_yield)
    
    # Generate sample data
    strikes = np.linspace(80, 120, 15)
    maturities = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0])  # quarters, semi, 9 months, 1yr, 1.5yr, 2yr
    
    # Create synthetic market prices (assuming true vol of 0.25 with some smile)
    true_vol = 0.25
    X, Y = np.meshgrid(maturities, strikes)
    
    # Smile factor: IV increases away from ATM
    moneyness = strikes / spot_price
    smile_factor = 0.8 + 0.4 * ((moneyness - 1) ** 2)
    
    market_prices = np.zeros((len(strikes), len(maturities)))
    for i, K in enumerate(strikes):
        for j, T in enumerate(maturities):
            sigma = true_vol * smile_factor[i]
            market_prices[i, j] = iv_calc.black_scholes_call(spot_price, K, T, risk_free_rate, sigma, dividend_yield)
    
    # Calculate implied volatility surface
    strikes_out, maturities_out, iv_surface = iv_calc.generate_surface_data(
        strikes, maturities, market_prices, option_type='call'
    )
    
    # Create visualizations
    fig1, ax1 = iv_calc.plot_surface_3d(strikes_out, maturities_out, iv_surface)
    fig1.savefig('iv_surface_3d.png', dpi=150, bbox_inches='tight')
    
    fig2, ax2 = iv_calc.plot_surface_contour(strikes_out, maturities_out, iv_surface)
    fig2.savefig('iv_surface_contour.png', dpi=150, bbox_inches='tight')
    
    fig3, ax3 = iv_calc.plot_term_structure(strikes_out, maturities_out, iv_surface)
    fig3.savefig('iv_term_structure.png', dpi=150, bbox_inches='tight')
    
    fig4, ax4 = iv_calc.plot_smile(maturities_out, strikes_out, iv_surface)
    fig4.savefig('iv_smile.png', dpi=150, bbox_inches='tight')
    
    print("IV Surface Analysis Complete!")
    print(f"IV Surface shape: {iv_surface.shape}")
    print(f"Min IV: {np.nanmin(iv_surface):.4f}")
    print(f"Max IV: {np.nanmax(iv_surface):.4f}")
    print(f"Mean IV: {np.nanmean(iv_surface):.4f}")
    print("\n" + "="*70)
    print("STRADDLE ANALYSIS")
    print("="*70 + "\n")
    
    # Find optimal straddles using different metrics
    print("1. CHEAPEST STRADDLE")
    optimal_cheap, all_straddles = iv_calc.find_optimal_straddle(
        strikes_out, maturities_out, iv_surface, metric='cheapest'
    )
    print(f"   Strike: ${optimal_cheap['strike']:.2f}")
    print(f"   Maturity: {optimal_cheap['maturity']:.3f} years ({optimal_cheap['maturity']*12:.0f} months)")
    print(f"   IV: {optimal_cheap['iv']:.2%}")
    print(f"   Cost: ${optimal_cheap['cost']:.4f}")
    print(f"   Greeks: Δ={optimal_cheap['delta']:.4f}, Γ={optimal_cheap['gamma']:.6f}, ν={optimal_cheap['vega']:.4f}, Θ={optimal_cheap['theta']:.4f}")
    
    print("\n2. HIGHEST GAMMA STRADDLE (Best for directional move)")
    optimal_gamma, _ = iv_calc.find_optimal_straddle(
        strikes_out, maturities_out, iv_surface, metric='highest_gamma'
    )
    print(f"   Strike: ${optimal_gamma['strike']:.2f}")
    print(f"   Maturity: {optimal_gamma['maturity']:.3f} years ({optimal_gamma['maturity']*12:.0f} months)")
    print(f"   IV: {optimal_gamma['iv']:.2%}")
    print(f"   Cost: ${optimal_gamma['cost']:.4f}")
    print(f"   Greeks: Δ={optimal_gamma['delta']:.4f}, Γ={optimal_gamma['gamma']:.6f}, ν={optimal_gamma['vega']:.4f}, Θ={optimal_gamma['theta']:.4f}")
    
    print("\n3. BEST VEGA/THETA RATIO (Best carry strategy)")
    optimal_vega, _ = iv_calc.find_optimal_straddle(
        strikes_out, maturities_out, iv_surface, metric='best_vega_carry'
    )
    print(f"   Strike: ${optimal_vega['strike']:.2f}")
    print(f"   Maturity: {optimal_vega['maturity']:.3f} years ({optimal_vega['maturity']*12:.0f} months)")
    print(f"   IV: {optimal_vega['iv']:.2%}")
    print(f"   Cost: ${optimal_vega['cost']:.4f}")
    print(f"   Greeks: Δ={optimal_vega['delta']:.4f}, Γ={optimal_vega['gamma']:.6f}, ν={optimal_vega['vega']:.4f}, Θ={optimal_vega['theta']:.4f}")
    print(f"   Vega/Theta Ratio: {optimal_vega['vega_theta_ratio']:.2f}")
    
    # Plot straddle surfaces
    fig5, ax5 = iv_calc.plot_straddle_surface(strikes_out, maturities_out, iv_surface, metric='cost')
    fig5.savefig('straddle_cost_surface.png', dpi=150, bbox_inches='tight')
    
    fig6, ax6 = iv_calc.plot_straddle_surface(strikes_out, maturities_out, iv_surface, metric='gamma')
    fig6.savefig('straddle_gamma_surface.png', dpi=150, bbox_inches='tight')
    
    fig7, ax7 = iv_calc.plot_optimal_straddles(strikes_out, maturities_out, iv_surface)
    fig7.savefig('optimal_straddles_summary.png', dpi=150, bbox_inches='tight')
    
    print("\n" + "="*70)
    print("All analysis plots saved!")
    print("="*70)
    
    # FORECAST COMPARISON
    # Let's assume we forecast higher volatility (30% vs current 25%)
    forecast_volatility = 0.30
    
    print("\n\nCOMPARING TO FORECASTED VOLATILITY...")
    iv_calc.print_forecast_analysis(strikes_out, maturities_out, iv_surface, forecast_volatility)
    
    # Plot forecast comparison
    fig8, ax8 = iv_calc.plot_forecast_comparison(strikes_out, maturities_out, iv_surface, forecast_volatility)
    fig8.savefig('forecast_comparison.png', dpi=150, bbox_inches='tight')
