import numpy as np
import pandas as pd
from datetime import datetime
from implied_vol_surface import ImpliedVolSurface
from Delta_Hedging import DeltaHedger


class VolArbTradeExecutor:
    """
    Complete workflow for initiating delta-neutral vol arb straddle positions.
    Identifies optimal straddle, calculates required hedges, and provides trading instructions.
    """
    
    def __init__(self, spot_price: float, risk_free_rate: float, iv_calculator: 'ImpliedVolSurface', hedger: 'DeltaHedger', dividend_yield: float = 0.0):
        """
        Initialize the trade executor.
        
        Parameters:
        -----------
        spot_price : float
            Current spot price
        risk_free_rate : float
            Risk-free rate (annualized)
        iv_calculator : ImpliedVolSurface
            ImpliedVolSurface object for IV calculations
        dividend_yield : float
            Dividend yield (annualized)
        """
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        
        self.iv_calculator = iv_calculator
        self.hedger = hedger
        
        self.active_trade = None
    
    def initiate_straddle(self, strikes, maturities, iv_surface, forecast_iv, strategy='cheapest'):
        """
        Initiate a delta-neutral straddle position.
        
        Parameters:
        -----------
        strikes : array-like
            Array of strike prices
        maturities : array-like
            Array of maturities
        iv_surface : 2D array
            IV surface
        forecast_iv : float
            Forecasted implied volatility
        strategy : str
            'cheapest', 'highest_gamma', or 'best_vega_carry'
        
        Returns:
        --------
        dict : Complete trade details
        """
        # Find optimal straddle
        optimal_straddle, _ = self.iv_calculator.find_optimal_straddle(
            strikes, maturities, iv_surface, metric=strategy
        )
        
        strike = optimal_straddle['strike']
        maturity = optimal_straddle['maturity']
        current_iv = optimal_straddle['iv']
        
        # Get straddle details
        greeks = self.iv_calculator.calculate_straddle_greeks(strike, maturity, current_iv)
        straddle_cost = greeks['cost']
        
        # Calculate delta hedge
        hedge_position = self.hedger.calculate_hedge_position(self.spot_price, strike, maturity, current_iv)
        
        # Calculate vega and theta exposure
        vega_analysis = self.hedger.calculate_vega_pnl(strike, maturity, current_iv, forecast_iv)
        theta_daily = self.hedger.calculate_theta_pnl(strike, maturity, current_iv, days=1)
        
        # Build trade structure
        trade = {
            'trade_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'strategy': strategy,
            'forecast_iv': forecast_iv,
            
            # Straddle details
            'straddle': {
                'strike': strike,
                'maturity': maturity,
                'maturity_months': maturity * 12,
                'current_iv': current_iv,
                'cost': straddle_cost,
                'entry_delta': greeks['delta'],
                'gamma': greeks['gamma'],
                'vega': greeks['vega'],
                'theta': greeks['theta'],
            },
            
            # Hedge details
            'hedge': {
                'units': hedge_position['hedge_units'],
                'direction': hedge_position['hedge_direction'],
                'cost': hedge_position['hedge_cost'],
                'is_delta_neutral': hedge_position['is_delta_neutral'],
            },
            
            # P&L exposure
            'pnl_exposure': {
                'vega_pnl_if_forecast': vega_analysis['vega_pnl'],
                'vega_pnl_pct': vega_analysis['vega_pnl_pct'],
                'theta_daily': theta_daily,
                'gamma': greeks['gamma'],
            }
        }
        
        self.active_trade = trade
        return trade
    
    def print_trade_blotter(self, trade):
        """
        Print formatted trade blotter with execution instructions.
        """
        print("\n" + "="*100)
        print("VOL ARB STRADDLE POSITION - EXECUTION BLOTTER")
        print("="*100)
        print(f"Trade Date: {trade['trade_date']}")
        print(f"Strategy: {trade['strategy'].upper()}")
        
        print(f"\n{'STRADDLE POSITION':^100}")
        print("-" * 100)
        print(f"  Strike Price:        ${trade['straddle']['strike']:.2f}")
        print(f"  Expiration:          {trade['straddle']['maturity']:.3f} years ({trade['straddle']['maturity_months']:.0f} months)")
        print(f"  Current IV:          {trade['straddle']['current_iv']:.2%}")
        print(f"  Forecast IV:         {trade['forecast_iv']:.2%}")
        print(f"  Straddle Cost:       ${trade['straddle']['cost']:.4f} per contract")
        
        print(f"\n{'STRADDLE GREEKS AT ENTRY':^100}")
        print("-" * 100)
        print(f"  Delta (Δ):           {trade['straddle']['entry_delta']:.6f}")
        print(f"  Gamma (Γ):           {trade['straddle']['gamma']:.8f}")
        print(f"  Vega (ν):            {trade['straddle']['vega']:.4f}")
        print(f"  Theta (Θ):           {trade['straddle']['theta']:.4f} per day")
        
        print(f"\n{'DELTA HEDGE EXECUTION':^100}")
        print("-" * 100)
        print(f"  Straddle Delta:      {trade['straddle']['entry_delta']:.6f}")
        print(f"  Hedge Direction:     {trade['hedge']['direction']}")
        print(f"  Hedge Units:         {abs(trade['hedge']['units']):.6f} shares of underlying")
        print(f"  Hedge Cost:          ${abs(trade['hedge']['cost']):.2f}")
        print(f"  Delta Neutral:       {'✓ YES' if trade['hedge']['is_delta_neutral'] else '✗ NO'}")
        
        print(f"\n{'ACTION ITEMS':^100}")
        print("-" * 100)
        
        if trade['straddle']['entry_delta'] > 0:
            print(f"  1. BUY 1 Straddle (ATM Call + ATM Put)")
            print(f"     └─ Strike: ${trade['straddle']['strike']:.2f}")
            print(f"     └─ Expiry: {trade['straddle']['maturity_months']:.0f} months")
            print(f"     └─ Max Risk: ${trade['straddle']['cost']:.4f} per contract")
        else:
            print(f"  1. SELL 1 Straddle (Short ATM Call + Short ATM Put)")
            print(f"     └─ Strike: ${trade['straddle']['strike']:.2f}")
            print(f"     └─ Expiry: {trade['straddle']['maturity_months']:.0f} months")
            print(f"     └─ Max Profit: ${trade['straddle']['cost']:.4f} per contract")
        
        if trade['hedge']['direction'] == 'LONG':
            print(f"\n  2. BUY {abs(trade['hedge']['units']):.6f} shares of underlying")
            print(f"     └─ At: ${self.spot_price:.2f}")
            print(f"     └─ Cost: ${abs(trade['hedge']['cost']):.2f}")
        else:
            print(f"\n  2. SHORT {abs(trade['hedge']['units']):.6f} shares of underlying")
            print(f"     └─ At: ${self.spot_price:.2f}")
            print(f"     └─ Proceeds: ${abs(trade['hedge']['cost']):.2f}")
        
        print(f"\n  3. After execution, Delta = {trade['straddle']['entry_delta'] + (-trade['hedge']['units']):.6f} (≈ 0 = delta neutral)")
        
        print(f"\n{'P&L EXPOSURE (After Delta Hedge)':^100}")
        print("-" * 100)
        print(f"  Gamma P&L:           +/- {trade['pnl_exposure']['gamma']:.8f} per $1 spot move²")
        print(f"  Vega P&L:            ${trade['pnl_exposure']['vega_pnl_if_forecast']:.4f} if IV → {trade['forecast_iv']:.2%}")
        print(f"                       ({trade['pnl_exposure']['vega_pnl_pct']:+.1f}% of initial cost)")
        print(f"  Theta P&L:           ${trade['pnl_exposure']['theta_daily']:.4f} per day (decay)")
        print(f"  Delta P&L:           $0 (hedged)")
        
        print(f"\n{'PROFIT SCENARIOS':^100}")
        print("-" * 100)
        
        # Scenario 1: IV rises to forecast
        if trade['pnl_exposure']['vega_pnl_if_forecast'] > 0:
            print(f"  ✓ IF IV rises to {trade['forecast_iv']:.2%}:")
            print(f"    └─ Vega P&L: +${trade['pnl_exposure']['vega_pnl_if_forecast']:.4f}")
            print(f"    └─ Return: +{trade['pnl_exposure']['vega_pnl_pct']:.1f}%")
        else:
            print(f"  ✗ IF IV falls below {trade['forecast_iv']:.2%}:")
            print(f"    └─ Vega P&L: ${trade['pnl_exposure']['vega_pnl_if_forecast']:.4f}")
            print(f"    └─ Loss: {trade['pnl_exposure']['vega_pnl_pct']:.1f}%")
        
        # Scenario 2: Spot moves
        spot_moves_test = [5, 10]
        print(f"\n  Spot Move P&L (assuming no IV change, 1 day theta decay):")
        for move in spot_moves_test:
            gamma_pnl = 0.5 * trade['pnl_exposure']['gamma'] * (move ** 2)
            total = gamma_pnl + trade['pnl_exposure']['theta_daily']
            print(f"    ├─ Spot +{move}%: Gamma +${gamma_pnl:.4f}, Theta ${trade['pnl_exposure']['theta_daily']:.4f} = ${total:.4f}")
        
        print(f"\n{'REHEDGING STRATEGY':^100}")
        print("-" * 100)
        print(f"  • Rehedge when spot moves ±2-5%")
        print(f"  • Rehedge when delta changes by ±0.10")
        print(f"  • Adjust hedge size: {abs(trade['hedge']['units']):.6f} * (new_delta / {abs(trade['straddle']['entry_delta']):.6f})")
        
        print("\n" + "="*100)
    
    def print_hedge_execution_guide(self, trade):
        """
        Print step-by-step hedge execution guide.
        """
        print("\n" + "="*100)
        print("DELTA HEDGE EXECUTION GUIDE")
        print("="*100)
        
        strike = trade['straddle']['strike']
        hedge_units = trade['hedge']['units']
        hedge_direction = trade['hedge']['direction']
        
        print(f"\nSTEP 1: ESTABLISH STRADDLE")
        print(f"  • Buy 1x ATM Straddle (Call + Put)")
        print(f"  • Strike: ${strike:.2f}")
        print(f"  • Cost: ${trade['straddle']['cost']:.4f} per contract")
        print(f"  • Entry Delta of Straddle: {trade['straddle']['entry_delta']:.6f}")
        
        print(f"\nSTEP 2: CALCULATE HEDGE")
        print(f"  • Straddle Delta = {trade['straddle']['entry_delta']:.6f}")
        print(f"  • To neutralize: Hedge = -{trade['straddle']['entry_delta']:.6f}")
        print(f"  • Hedge Position = {abs(hedge_units):.6f} shares")
        
        print(f"\nSTEP 3: EXECUTE HEDGE")
        if hedge_direction == 'LONG':
            print(f"  ACTION: BUY {abs(hedge_units):.6f} shares of underlying")
        else:
            print(f"  ACTION: SHORT {abs(hedge_units):.6f} shares of underlying")
        print(f"  • Current Spot Price: ${self.spot_price:.2f}")
        print(f"  • Hedge Cost: ${abs(trade['hedge']['cost']):.2f}")
        
        print(f"\nSTEP 4: VERIFY DELTA NEUTRAL")
        residual_delta = trade['straddle']['entry_delta'] + (-hedge_units)
        print(f"  • Straddle Delta:        {trade['straddle']['entry_delta']:+.6f}")
        print(f"  • Hedge Delta:           {-hedge_units:+.6f}")
        print(f"  • Total Portfolio Delta: {residual_delta:+.6f}")
        print(f"  • Status: {'✓ DELTA NEUTRAL' if abs(residual_delta) < 0.001 else '✗ NOT NEUTRAL'}")
        
        print(f"\nSTEP 5: ONGOING MANAGEMENT")
        print(f"  Monitor these metrics daily:")
        print(f"  • Spot Price (for rehedging triggers)")
        print(f"  • IV Level (track P&L)")
        print(f"  • Days to Expiration (theta decay)")
        print(f"  • Portfolio Delta (should stay near 0)")
        
        print(f"\nSTEP 6: REHEDGING RULES")
        print(f"  Rehedge when:")
        print(f"  • Spot moves by ±2-5% (typical trigger: ±{0.025*self.spot_price:.2f})")
        print(f"  • Delta changes by ±0.10 or more")
        print(f"  • Daily gamma P&L accumulation > transaction costs")
        
        print(f"\nSTEP 7: EXIT STRATEGY")
        print(f"  Close position when:")
        print(f"  • IV reaches forecast level ({trade['forecast_iv']:.2%})")
        print(f"  • Target P&L is hit")
        print(f"  • Stop loss triggered (if needed)")
        print(f"  • Days to expiration < 7 days (gamma risk increases)")
        
        print("\n" + "="*100)
    
    def calculate_exit_signals(self, trade, days_elapsed=0, current_spot=None, current_iv=None):
        """
        Calculate exit signals based on various criteria.
        
        Parameters:
        -----------
        trade : dict
            Trade details from initiate_straddle()
        days_elapsed : int
            Days since trade initiation
        current_spot : float
            Current spot price (uses initial if None)
        current_iv : float
            Current IV (uses initial if None)
        
        Returns:
        --------
        dict : Exit signals and recommendations
        """
        if current_spot is None:
            current_spot = self.spot_price
        if current_iv is None:
            current_iv = trade['straddle']['current_iv']
        
        strike = trade['straddle']['strike']
        maturity = trade['straddle']['maturity']
        initial_cost = trade['straddle']['cost']
        forecast_iv = trade['forecast_iv']
        original_vega = trade['straddle']['vega']
        
        # Calculate time remaining
        days_total = maturity * 365
        days_remaining = days_total - days_elapsed
        pct_time_elapsed = (days_elapsed / days_total) * 100 if days_total > 0 else 0
        
        # Calculate current straddle value
        current_straddle_value = self.iv_calculator.calculate_straddle_cost(strike, days_remaining/365, current_iv)
        current_pnl = current_straddle_value - initial_cost
        current_pnl_pct = (current_pnl / initial_cost) * 100 if initial_cost != 0 else 0
        
        # Calculate current vega (vega decays with time)
        if days_remaining > 0:
            current_vega = self.iv_calculator.calculate_straddle_greeks(strike, days_remaining/365, current_iv)['vega']
            vega_remaining_pct = (current_vega / original_vega) * 100 if original_vega > 0 else 0
        else:
            current_vega = 0
            vega_remaining_pct = 0
        
        # Exit signal 1: IV reaches forecast
        iv_exit_triggered = current_iv >= forecast_iv if trade['pnl_exposure']['vega_pnl_if_forecast'] > 0 else current_iv <= forecast_iv
        
        # Exit signal 2: Maximum profit target (50% of max profit)
        max_profit_target = initial_cost * 0.5
        profit_target_triggered = current_pnl >= max_profit_target
        
        # Exit signal 3: Stop loss (30% of initial cost)
        stop_loss_level = -initial_cost * 0.3
        stop_loss_triggered = current_pnl <= stop_loss_level
        
        # Exit signal 4: Days to expiration < 7 days (gamma blows up)
        expiration_exit_triggered = days_remaining < 7
        
        # Exit signal 5: Half-life theta decay (optimal exit for vega trades)
        theta_decay_half_life = -initial_cost * 0.5  # Exit if lost half the position to theta
        theta_exit_triggered = current_pnl <= theta_decay_half_life and days_remaining > 7
        
        # Exit signal 6: VEGA DECAY - No longer a vol trade if vega < 40% of original
        vega_decay_exit_triggered = vega_remaining_pct < 40 and days_remaining > 0
        
        # Calculate breakeven IV
        if trade['pnl_exposure']['vega_pnl_if_forecast'] > 0:
            # Long vega position - need IV to rise
            vega_per_1pct = trade['straddle']['vega'] / 100
            iv_move_needed_pct = abs(initial_cost) / vega_per_1pct if vega_per_1pct > 0 else 0
            breakeven_iv = current_iv + (iv_move_needed_pct / 100)
        else:
            # Short vega position - need IV to fall
            vega_per_1pct = trade['straddle']['vega'] / 100
            iv_move_needed_pct = abs(initial_cost) / vega_per_1pct if vega_per_1pct > 0 else 0
            breakeven_iv = current_iv - (iv_move_needed_pct / 100)
        
        return {
            'days_elapsed': days_elapsed,
            'days_remaining': days_remaining,
            'pct_time_elapsed': pct_time_elapsed,
            'current_spot': current_spot,
            'current_iv': current_iv,
            'current_straddle_value': current_straddle_value,
            'current_pnl': current_pnl,
            'current_pnl_pct': current_pnl_pct,
            'original_vega': original_vega,
            'current_vega': current_vega,
            'vega_remaining_pct': vega_remaining_pct,
            'iv_exit_triggered': iv_exit_triggered,
            'profit_target_triggered': profit_target_triggered,
            'stop_loss_triggered': stop_loss_triggered,
            'expiration_exit_triggered': expiration_exit_triggered,
            'theta_exit_triggered': theta_exit_triggered,
            'vega_decay_exit_triggered': vega_decay_exit_triggered,
            'breakeven_iv': breakeven_iv,
            'profit_target_level': max_profit_target,
            'stop_loss_level': stop_loss_level,
            'any_exit_signal': iv_exit_triggered or profit_target_triggered or stop_loss_triggered or expiration_exit_triggered or vega_decay_exit_triggered
        }
    
    def print_exit_strategy(self, trade):
        """
        Print detailed exit strategy and triggers.
        """
        print("\n" + "="*100)
        print("EXIT STRATEGY & TRIGGERS")
        print("="*100)
        
        strike = trade['straddle']['strike']
        maturity = trade['straddle']['maturity']
        initial_cost = trade['straddle']['cost']
        forecast_iv = trade['forecast_iv']
        days_total = maturity * 365
        
        print(f"\n{'PRIMARY EXIT TRIGGERS':^100}")
        print("-" * 100)
        
        print(f"\n1. IV REACHES FORECAST LEVEL: {forecast_iv:.2%}")
        print(f"   Current IV: {trade['straddle']['current_iv']:.2%}")
        print(f"   ├─ Target: IV moves to {forecast_iv:.2%}")
        print(f"   ├─ Expected P&L: ${trade['pnl_exposure']['vega_pnl_if_forecast']:.4f}")
        print(f"   ├─ Expected Return: {trade['pnl_exposure']['vega_pnl_pct']:+.1f}%")
        print(f"   └─ Action: CLOSE entire straddle position")
        
        print(f"\n2. PROFIT TARGET: ${initial_cost * 0.5:.4f} (50% of initial cost)")
        print(f"   Initial Cost: ${initial_cost:.4f}")
        print(f"   ├─ Target P&L: ${initial_cost * 0.5:.4f}")
        print(f"   ├─ Target Return: +50%")
        print(f"   └─ Action: TAKE PROFITS - close position")
        
        print(f"\n3. STOP LOSS: -${initial_cost * 0.3:.4f} (30% of initial cost)")
        print(f"   Initial Cost: ${initial_cost:.4f}")
        print(f"   ├─ Stop Loss P&L: -${initial_cost * 0.3:.4f}")
        print(f"   ├─ Stop Loss Return: -30%")
        print(f"   └─ Action: CUT LOSSES - immediately close position")
        
        print(f"\n4. EXPIRATION APPROACHING: < 7 Days to Expiration")
        print(f"   Total Duration: {days_total:.0f} days ({maturity:.2f} years)")
        print(f"   ├─ Days Remaining: < 7 days")
        print(f"   ├─ Risk: Gamma explodes near expiration")
        print(f"   ├─ Time decay accelerates")
        print(f"   └─ Action: CLOSE position regardless of P&L")
        
        print(f"\n{'SECONDARY EXIT TRIGGERS':^100}")
        print("-" * 100)
        
        print(f"\n5. VEGA DECAY: Vega drops below 40% of original")
        print(f"   Entry Vega:        {trade['straddle']['vega']:.4f}")
        print(f"   ├─ Vega at entry represents your vol exposure")
        print(f"   ├─ Exit trigger: Current vega < 40% of original")
        print(f"   ├─ Reason: No longer trading volatility, just theta decay")
        print(f"   ├─ Calculation: Vega ∝ √(days_remaining)")
        print(f"   └─ Action: EXIT - position is no longer a vol trade")
        
        print(f"\n6. THETA DECAY: Lost 50% of position to time decay")
        print(f"   Initial Cost: ${initial_cost:.4f}")
        print(f"   ├─ Trigger P&L: -${initial_cost * 0.5:.4f}")
        print(f"   ├─ Daily Theta: ${trade['pnl_exposure']['theta_daily']:.4f}")
        print(f"   ├─ Days to trigger: {abs((-initial_cost * 0.5) / trade['pnl_exposure']['theta_daily']):.0f} days")
        print(f"   └─ Action: EXIT - theta is working against you faster than vega can compensate")
        
        print(f"\n{'OPTIMAL EXIT SCENARIOS':^100}")
        print("-" * 100)
        
        print(f"\nSCENARIO A: Thesis is Correct (IV rises)")
        print(f"  • Entry: Buy straddle at {trade['straddle']['current_iv']:.2%} IV")
        print(f"  • Target: IV rises to {forecast_iv:.2%}")
        print(f"  • Exit: Close straddle when IV ≥ {forecast_iv:.2%}")
        print(f"  • Expected P&L: ${trade['pnl_exposure']['vega_pnl_if_forecast']:.4f} (+{trade['pnl_exposure']['vega_pnl_pct']:.1f}%)")
        print(f"  • Recommended: Use this as primary exit")
        
        print(f"\nSCENARIO B: Quick Profit Lock-in")
        print(f"  • Entry: Buy straddle at ${initial_cost:.4f}")
        print(f"  • Target: Capture 25-50% profit")
        print(f"  • Exit: Close when P&L = ${initial_cost * 0.25:.4f} - ${initial_cost * 0.5:.4f}")
        print(f"  • Rationale: Bird in hand - secure profits early")
        print(f"  • Timeframe: Usually 5-15 days for IV pop")
        
        print(f"\nSCENARIO C: Thesis is Wrong (IV falls)")
        print(f"  • Signal: IV falls below {trade['straddle']['current_iv'] - (forecast_iv - trade['straddle']['current_iv']):.2%}")
        print(f"  • Exit: Close position to prevent further losses")
        print(f"  • Max Loss: ${initial_cost:.4f} if held to expiration OTM")
        print(f"  • Recommended: Stop loss at -30% = -${initial_cost * 0.3:.4f}")
        
        print(f"\n{'MONITORING CHECKLIST':^100}")
        print("-" * 100)
        print(f"  ☐ Check IV daily - compare to forecast ({forecast_iv:.2%})")
        print(f"  ☐ Monitor P&L - take profits at 50% ({initial_cost * 0.5:.4f})")
        print(f"  ☐ Check days to expiration - exit if < 7 days")
        print(f"  ☐ Monitor delta - rehedge if changes by ±0.10")
        print(f"  ☐ Calculate theta decay - track daily loss")
        print(f"  ☐ Spot price moves - track gamma P&L")
        print(f"  ☐ Set stop loss alert at -${initial_cost * 0.3:.4f}")
        
        print("\n" + "="*100)
    
    def print_risk_summary(self, trade):
        """
        Print risk summary for the trade.
        """
        print("\n" + "="*100)
        print("RISK SUMMARY")
        print("="*100)
        
        print(f"\nMAX LOSS / MAX GAIN:")
        print(f"  • Max Loss (if straddle expires OTM): ${trade['straddle']['cost']:.4f}")
        print(f"  • Max Gain (if held to expiry): Unlimited (via gamma)")
        
        print(f"\nTHETA DECAY (Time Risk):")
        print(f"  • Daily decay: ${trade['pnl_exposure']['theta_daily']:.4f}")
        print(f"  • Monthly decay: ${trade['pnl_exposure']['theta_daily'] * 30:.2f}")
        print(f"  • Total until expiry: ${trade['pnl_exposure']['theta_daily'] * (trade['straddle']['maturity'] * 365):.2f}")
        print(f"  • Risk: Losing money if IV stays constant")
        
        print(f"\nGAMMA RISK (Realized Volatility):")
        print(f"  • Gamma: {trade['pnl_exposure']['gamma']:.8f}")
        print(f"  • P&L per 1% spot move: ${0.5 * trade['pnl_exposure']['gamma'] * 1:.4f}")
        print(f"  • Breakeven spot move: {np.sqrt(abs(trade['pnl_exposure']['theta_daily'] * 365 / (0.5 * trade['pnl_exposure']['gamma']))):.2f}%")
        print(f"  • Risk: Need IV to exceed realized volatility")
        
        print(f"\nVEGA RISK (Implied Vol):")
        print(f"  • Vega Exposure: {trade['straddle']['vega']:.4f}")
        print(f"  • Current IV vs Forecast: {(trade['forecast_iv'] - trade['straddle']['current_iv'])*100:.0f} basis points")
        if trade['pnl_exposure']['vega_pnl_if_forecast'] > 0:
            print(f"  • P&L if IV realizes forecast: +${trade['pnl_exposure']['vega_pnl_if_forecast']:.4f}")
            print(f"  • Risk: LOW (vega bet is correct direction)")
        else:
            print(f"  • P&L if IV realizes forecast: ${trade['pnl_exposure']['vega_pnl_if_forecast']:.4f}")
            print(f"  • Risk: HIGH (vega bet is wrong direction)")
        
        print(f"\nHEDGE SLIPPAGE:")
        print(f"  • Rehedging will incur transaction costs")
        print(f"  • Bid-ask spread on underlying: Assume 0.01-0.05%")
        print(f"  • Bid-ask spread on options: Assume 1-2% of premium")
        print(f"  • Commission: Variable by broker")
        
        print("\n" + "="*100)


def run_vol_arb_workflow(strikes, maturities, market_prices, spot_price, risk_free_rate, 
                         dividend_yield, forecast_iv, strategy='cheapest'):
    """
    Complete workflow: Build IV surface → Find optimal straddle → Initiate trade → Calculate hedge
    
    Parameters:
    -----------
    strikes : array-like
        Strike prices
    maturities : array-like
        Time to maturities
    market_prices : 2D array
        Option prices matrix
    spot_price : float
        Current spot price
    risk_free_rate : float
        Risk-free rate
    dividend_yield : float
        Dividend yield
    forecast_iv : float
        Forecasted IV
    strategy : str
        Trading strategy ('cheapest', 'highest_gamma', 'best_vega_carry')
    """
    
    print("\n" + "="*100)
    print("VOL ARB TRADING WORKFLOW")
    print("="*100)
    
    # Step 1: Build IV surface
    print("\n[STEP 1] Building Implied Volatility Surface...")
    iv_calc = ImpliedVolSurface(spot_price, risk_free_rate, dividend_yield)
    strikes_out, maturities_out, iv_surface = iv_calc.generate_surface_data(
        strikes, maturities, market_prices, option_type='call'
    )
    print(f"✓ IV Surface built: {iv_surface.shape}")
    print(f"  Min IV: {np.nanmin(iv_surface):.2%}, Max IV: {np.nanmax(iv_surface):.2%}")
    
    # Step 2: Create executor and initiate trade
    print(f"\n[STEP 2] Initiating Trade: {strategy.upper()} Strategy...")
    executor = VolArbTradeExecutor(spot_price, risk_free_rate, dividend_yield)
    trade = executor.initiate_straddle(strikes_out, maturities_out, iv_surface, forecast_iv, strategy=strategy)
    print(f"✓ Trade initiated")
    
    # Step 3: Print trade blotter
    print(f"\n[STEP 3] Trade Blotter & Execution Instructions...")
    executor.print_trade_blotter(trade)
    
    # Step 4: Print hedge execution guide
    print(f"\n[STEP 4] Hedge Execution Guide...")
    executor.print_hedge_execution_guide(trade)
    
    # Step 5: Print risk summary
    print(f"\n[STEP 5] Risk Analysis...")
    executor.print_risk_summary(trade)
    
    # Step 6: Print exit strategy
    print(f"\n[STEP 6] Exit Strategy...")
    executor.print_exit_strategy(trade)
    
    return executor, trade
