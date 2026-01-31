import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
from preprocess_data import get_log_returns, parse_data
from garch import garch_modelling
from implied_vol_surface import ImpliedVolSurface
from Delta_Hedging import DeltaHedger
from transactionCosts import TransactionCost
from regime_identifier import RegimeBlockerXGB
from scipy.stats import norm

warnings.filterwarnings('ignore', category=RuntimeWarning)

DE_MEAN = "AR"
MODEL = "EGARCH"
DISTRIBUTION = {"GARCH": "normal", "EGARCH": "t"}[MODEL]
validity_checks = False

def load_options_data(filepath, ticker=None):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df['exdate'] = pd.to_datetime(df['exdate'])
    
    if ticker:
        df = df[df['ticker'] == ticker].copy()
        if len(df) == 0:
            print(f"  [!] Warning: No data found for ticker '{ticker}'")
            print(f"  Available tickers: {sorted(df['ticker'].unique())[:10]}... ({len(df['ticker'].unique())} total)")
    
    # Use mid_price if available, otherwise calculate from bid/offer
    if 'mid_price' in df.columns:
        df['market_price'] = df['mid_price']
    else:
        df['market_price'] = (df['best_bid'] + df['best_offer']) / 2
    
    # Convert strike price from cents/thousands to dollars
    # Check if strikes are very large (>10000 suggests they're in cents)
    if df['strike_price'].median() > 10000:
        df['strike_price'] = df['strike_price'] / 1000
    elif df['strike_price'].median() > 1000:
        df['strike_price'] = df['strike_price'] / 100
    
    # Calculate time to maturity in years
    df['maturity'] = df['days_to_expiry'] / 365
    
    return df

def get_iv_surface_for_date(options_df, date, spot_price, risk_free_rate=0.02, dividend_yield=0.02):
    date_options = options_df[options_df['date'] == date].copy()
    
    if len(date_options) == 0:
        return None, None, None
    
    calls = date_options[date_options['cp_flag'] == 'C'].copy()
    
    if len(calls) < 10:  # Need minimum data
        return None, None, None
    
    # Filter out only the most problematic options that cause numerical issues
    # 1. Remove expired or very short-dated options (< 2 days) - these cause div/0
    calls = calls[calls['days_to_expiry'] >= 2].copy()
    
    # 2. Remove options with zero or negative prices (invalid data)
    calls = calls[calls['market_price'] > 0.01].copy()
    
    # 3. Remove extreme OTM options to avoid numerical instability (strike > 3x spot)
    calls = calls[calls['strike_price'] < spot_price * 3.0].copy()
    
    if len(calls) < 10:  # Recheck after filtering
        return None, None, None
    
    strikes = sorted(calls['strike_price'].unique())
    maturities = sorted(calls['maturity'].unique())
    
    # Build price matrix
    market_prices = np.full((len(strikes), len(maturities)), np.nan)
    
    for i, K in enumerate(strikes):
        for j, T in enumerate(maturities):
            price_data = calls[(calls['strike_price'] == K) & (calls['maturity'] == T)]
            if len(price_data) > 0:
                market_prices[i, j] = price_data['market_price'].iloc[0]
    
    # Filter out strikes/maturities with too much missing data
    valid_strikes_mask = ~np.isnan(market_prices).all(axis=1)
    valid_maturities_mask = ~np.isnan(market_prices).all(axis=0)
    
    strikes = np.array(strikes)[valid_strikes_mask]
    maturities = np.array(maturities)[valid_maturities_mask]
    market_prices = market_prices[valid_strikes_mask][:, valid_maturities_mask]
    
    if len(strikes) < 5 or len(maturities) < 2:
        return None, None, None
    
    return strikes, maturities, market_prices

def get_atm_option_for_dte(options_df, date, spot_price, target_dte=45, dte_range=(30, 45)):
    date_options = options_df[options_df['date'] == date].copy()
    
    if len(date_options) == 0:
        return None
    
    date_options = date_options[
        (date_options['days_to_expiry'] >= dte_range[0]) & 
        (date_options['days_to_expiry'] <= dte_range[1])
    ].copy()
    
    if len(date_options) == 0:
        return None
    
    date_options['dte_diff'] = abs(date_options['days_to_expiry'] - target_dte)
    target_dte_actual = date_options.loc[date_options['dte_diff'].idxmin(), 'days_to_expiry']
    target_options = date_options[date_options['days_to_expiry'] == target_dte_actual].copy()
    
    strikes = target_options['strike_price'].unique()
    atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
    
    calls = target_options[(target_options['strike_price'] == atm_strike) & (target_options['cp_flag'] == 'C')]
    puts = target_options[(target_options['strike_price'] == atm_strike) & (target_options['cp_flag'] == 'P')]
    
    if len(calls) == 0 or len(puts) == 0:
        return None
    
    call_price = calls['market_price'].iloc[0]
    put_price = puts['market_price'].iloc[0]
    
    return {
        'strike': atm_strike,
        'dte': target_dte_actual,
        'maturity': target_dte_actual / 365,
        'call_price': call_price,
        'put_price': put_price,
        'straddle_price': call_price + put_price,
        'exdate': calls['exdate'].iloc[0]
    }


def get_iv_for_option(iv_calc, strike, maturity, call_price, put_price):
    try:
        call_iv = iv_calc.implied_volatility(call_price, strike, maturity, 'call')
        put_iv = iv_calc.implied_volatility(put_price, strike, maturity, 'put')
        # Average of call and put IV
        if not np.isnan(call_iv) and not np.isnan(put_iv):
            return (call_iv + put_iv) / 2
        elif not np.isnan(call_iv):
            return call_iv
        elif not np.isnan(put_iv):
            return put_iv
        return np.nan
    except:
        return np.nan


def rolling_window_backtest(ticker, train_window=126, refit_frequency=21, 
                            starting_capital=100000, position_size=8000,
                            start_date=None, end_date=None,
                            use_regime_blocker=True, verbose=False):
    
    print("="*80)
    print("SHORT VOL STRATEGY - 45 DTE STRADDLE BACKTEST")
    print("="*80)
    print("\n  Strategy Rules:")
    print("    Entry: Market IV > GARCH + 2%, IV > 75th pctl, 30-45 DTE, no stress")
    print("    Position: SELL ATM straddle, $8k size, delta hedge daily")
    print("    Exit Day 7: IV -3% take profit, IV +2% stop loss")
    print("    Exit Day 10: IV -2% take profit, IV +3% stop loss")
    print("    Exit Day 14: Mandatory exit OR 2x premium stop loss")
    
    ticker_upper = ticker.upper()
    ticker_upper = ticker.upper()
    ticker_lower = ticker.lower()
    sp_data_path = f"{ticker_lower}_stock_prices_2020_2024.csv"
    options_data_path = f"{ticker_lower}_options_2020_2024.csv"
    
    print(f"\n[1/6] Loading data for {ticker_upper}...")
    
    stock_data = pd.read_csv(sp_data_path, parse_dates=['date'])
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data = stock_data.set_index('date').sort_index()
    
    print(f"  [i] Using {ticker_upper} stock and options data")
    options_data = load_options_data(options_data_path, ticker=ticker_upper)
    print(f"  * Loaded {len(stock_data)} days of {ticker_upper} stock data")
    print(f"  * Loaded {len(options_data)} option quotes for {ticker_upper}")
    
    trading_dates = sorted(options_data['date'].unique())
    trading_dates_set = set(trading_dates)
    
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        trading_dates = [d for d in trading_dates if d >= start_date]
        print(f"  [i] Start date filter: {start_date.date()}")
    
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        trading_dates = [d for d in trading_dates if d <= end_date]
        print(f"  [i] End date filter: {end_date.date()}")
    
    print(f"  * {len(trading_dates)} trading dates with options data")
    print(f"  [i] Training window: {train_window} days, will start backtesting from day {train_window + 1}")
    
    if len(trading_dates) <= train_window:
        print(f"  [!] Error: Not enough data! Need >{train_window} days, have {len(trading_dates)}")
        return None
    
    print(f"\n[2/6] Computing historical IV percentiles...")
    
    regime_blocker = None
    if use_regime_blocker:
        print(f"\n[3/6] Initializing regime blocker...")
        backtest_start_date = trading_dates[train_window]
        historical_stock_data = stock_data.loc[:backtest_start_date].iloc[:-1]
        log_returns_historical = parse_data(historical_stock_data.reset_index(), price_col='prc')
        
        print(f"  [i] Training regime blocker on data up to {backtest_start_date.date()} ({len(log_returns_historical)} days)")
        print(f"  [i] This ensures no look-ahead bias - blocker never sees backtest period data")
        
        try:
            regime_blocker = RegimeBlockerXGB(
                returns=log_returns_historical,
                stress_vol_percentile=90.0,
                stress_drawdown_threshold=-0.05,
                calm_vol_percentile=25.0,
                block_threshold=0.5,
                random_state=42,
                verbose=False
            )
            print("  * Regime blocker initialized successfully (trained on pre-backtest data only)")
        except Exception as e:
            print(f"  [!] Warning: Could not initialize regime blocker: {e}")
            print("  * Continuing without regime blocking")
            regime_blocker = None
    else:
        print(f"\n[3/6] Regime blocker disabled (use_regime_blocker=False)")
    
    results = []
    trade_log = []
    portfolio_value = starting_capital
    cash = starting_capital
    active_position = None
    tc_calc = TransactionCost()
    iv_history = []
    
    print("\n[4/6] Starting rolling window backtest...")
    print(f"  Starting Capital: ${starting_capital:,.2f}")
    print(f"  Position Size: ${position_size:,.2f}")
    last_garch_fit_date = None
    garch_forecast = None
    
    processed_count = 0
    skipped_count = 0
    blocked_count = 0
    trades_entered = 0
    trades_exited = 0
    total_dates = len(trading_dates)
    total_hedge_costs = 0
    total_hedge_pnl = 0
    
    forecast_history = []
    market_iv_history = []
    
    def calculate_vol_risk_premium(forecast_hist, market_hist, min_samples=60):
        if len(forecast_hist) < min_samples:
            # Not enough data: use default equity risk premium
            return 0.03  # 3% is typical for equities
        
        # Use last 60-126 observations for rolling calibration
        lookback = min(126, len(forecast_hist))
        recent_forecasts = np.array(forecast_hist[-lookback:])
        recent_markets = np.array(market_hist[-lookback:])
        
        # Calculate median difference (robust to outliers)
        risk_premium = np.median(recent_markets - recent_forecasts)
        
        # Sanity check: risk premium should be 0-10%
        risk_premium = np.clip(risk_premium, 0.0, 0.10)
        
        return risk_premium
    
    def ensemble_vol_forecast(returns, base_forecast):
        forecasts = [base_forecast]
        
        if len(returns) >= 21:
            recent_rv = returns.tail(21).std() * np.sqrt(252)
            forecasts.append(recent_rv)
        
        if len(returns) >= 30:
            ewma_vol = returns.ewm(span=30).std().iloc[-1] * np.sqrt(252)
            forecasts.append(ewma_vol)
        
        return np.median(forecasts)
    
    for i, current_date in enumerate(trading_dates):
        # progress indicator
        if i % 20 == 0:
            progress_pct = (i / total_dates) * 100
            print(f"\r  Progress: {i}/{total_dates} dates ({progress_pct:.1f}%) - Trades: {trades_entered} entered, {trades_exited} exited, Blocked: {blocked_count}", end="", flush=True)
        
        if i < train_window:
            continue  # Need minimum training data
        
        if current_date not in stock_data.index:
            skipped_count += 1
            if verbose:
                print(f"  [{current_date.date()}] No stock data for this date")
            continue
        spot_price = stock_data.loc[current_date, 'prc']
        
        # Refit GARCH model periodically
        days_since_fit = (current_date - last_garch_fit_date).days if last_garch_fit_date else refit_frequency + 1
        
        if days_since_fit >= refit_frequency or garch_forecast is None:
            if verbose:
                print(f"\n  [{current_date.date()}] Refitting GARCH model (window: {train_window} days)...")
            
            # Get historical returns for training
            historical_data = stock_data.loc[:current_date].iloc[-train_window-1:-1]
            log_returns = parse_data(historical_data, price_col='prc')
            
            try:
                _, sigma_forecast_base = garch_modelling(log_returns, DE_MEAN, MODEL, DISTRIBUTION, validity_checks=False)
                sigma_forecast_ensemble = ensemble_vol_forecast(log_returns, sigma_forecast_base)
                vol_risk_premium = calculate_vol_risk_premium(forecast_history, market_iv_history)
                garch_forecast = sigma_forecast_ensemble + vol_risk_premium
                
                last_garch_fit_date = current_date
                
                if verbose:
                    print(f"\n  [{current_date.date()}] Refit Model:")
                    print(f"    Base EGARCH:     {sigma_forecast_base:.2%}")
                    print(f"    Ensemble:        {sigma_forecast_ensemble:.2%}")
                    print(f"    Risk Premium:    {vol_risk_premium:.2%}")
                    print(f"    Final Forecast:  {garch_forecast:.2%}")
                
            except Exception as e:
                if verbose:
                    print(f"\n  [{current_date.date()}] GARCH fit failed: {e}, skipping...")
                continue
        
        if active_position is not None:
            days_held = (current_date - active_position['entry_date']).days
            
            current_option = get_atm_option_for_dte(
                options_data, current_date, spot_price,
                target_dte=active_position['dte_remaining'],
                dte_range=(max(1, active_position['dte_remaining'] - 5), active_position['dte_remaining'] + 5)
            )
            
            date_options = options_data[options_data['date'] == current_date]
            strike_options = date_options[date_options['strike_price'] == active_position['strike']]
            
            current_iv = None
            current_straddle_value = None
            
            if len(strike_options) > 0:
                # Find options with similar expiry
                calls = strike_options[strike_options['cp_flag'] == 'C']
                puts = strike_options[strike_options['cp_flag'] == 'P']
                
                if len(calls) > 0 and len(puts) > 0:
                    # Get option closest to original expiry
                    calls = calls.iloc[(calls['exdate'] - active_position['exdate']).abs().argsort()[:1]]
                    puts = puts.iloc[(puts['exdate'] - active_position['exdate']).abs().argsort()[:1]]
                    
                    if len(calls) > 0 and len(puts) > 0:
                        current_call_price = calls['market_price'].iloc[0]
                        current_put_price = puts['market_price'].iloc[0]
                        current_straddle_value = (current_call_price + current_put_price) * 100 * active_position['num_straddles']
                        
                        # Calculate current IV
                        current_dte = max(1, active_position['entry_dte'] - days_held)
                        current_maturity = current_dte / 365
                        
                        iv_calc = ImpliedVolSurface(
                            spot_price=spot_price,
                            risk_free_rate=0.02,
                            dividend_yield=0.02,
                            verbose=False
                        )
                        current_iv = get_iv_for_option(iv_calc, active_position['strike'], current_maturity, 
                                                       current_call_price, current_put_price)
            
            # If we couldn't get current values, estimate them
            if current_straddle_value is None or current_iv is None:
                # Skip this day for position management
                active_position['dte_remaining'] = max(1, active_position['entry_dte'] - days_held)
                continue
            
            # Track IV change
            iv_change = current_iv - active_position['entry_iv']
            iv_change_pct = iv_change * 100  # Convert to percentage points
            
            # Daily delta hedging with proper P&L calculation
            hedger = DeltaHedger(spot_price, risk_free_rate=0.02, dividend_yield=0.02)
            current_maturity = max(0.01, (active_position['entry_dte'] - days_held) / 365)
            
            hedge_result = hedger.calculate_hedge_position(
                spot_price, active_position['strike'], current_maturity, current_iv
            )
            
            d1 = (np.log(spot_price / active_position['strike']) + 
                  (0.02 - 0.02 + 0.5 * current_iv**2) * current_maturity) / \
                 (current_iv * np.sqrt(current_maturity))
            
            # Straddle delta = call delta + put delta
            call_delta = np.exp(-0.02 * current_maturity) * norm.cdf(d1)
            put_delta = -np.exp(-0.02 * current_maturity) * norm.cdf(-d1)
            straddle_delta = call_delta + put_delta
            
            # Straddle gamma = 2 * single option gamma (call and put have same gamma)
            single_gamma = np.exp(-0.02 * current_maturity) * norm.pdf(d1) / \
                           (spot_price * current_iv * np.sqrt(current_maturity))
            straddle_gamma = 2 * single_gamma
            
            prev_spot = active_position.get('prev_spot', active_position['entry_spot'])
            spot_change = spot_price - prev_spot
            prev_delta = active_position.get('prev_delta', 0)
            prev_gamma = active_position.get('prev_gamma', straddle_gamma)
            
            delta_hedge_pnl = -prev_delta * spot_change * active_position['num_straddles']
            gamma_pnl = -0.5 * prev_gamma * (spot_change ** 2) * active_position['num_straddles']
            theta_pnl = -hedger.calculate_theta_pnl(active_position['strike'], current_maturity, 
                                                   current_iv, days=1) * active_position['num_straddles'] * 100
            
            if 'prev_hedge_units' in active_position:
                hedge_adjustment = abs(hedge_result['hedge_units'] - active_position['prev_hedge_units'])
                hedge_rebalance_cost = hedge_adjustment * spot_price * 0.001
            else:
                hedge_rebalance_cost = 0
            
            daily_hedge_pnl = delta_hedge_pnl + gamma_pnl + theta_pnl - hedge_rebalance_cost
            active_position['cumulative_hedge_pnl'] = active_position.get('cumulative_hedge_pnl', 0) + daily_hedge_pnl
            active_position['cumulative_gamma_pnl'] = active_position.get('cumulative_gamma_pnl', 0) + gamma_pnl
            active_position['cumulative_theta_pnl'] = active_position.get('cumulative_theta_pnl', 0) + theta_pnl
            active_position['cumulative_delta_pnl'] = active_position.get('cumulative_delta_pnl', 0) + delta_hedge_pnl
            active_position['total_hedge_cost'] = active_position.get('total_hedge_cost', 0) + hedge_rebalance_cost
            total_hedge_costs += hedge_rebalance_cost
            
            # Update position tracking for next day
            active_position['prev_hedge_units'] = hedge_result['hedge_units']
            active_position['prev_spot'] = spot_price
            active_position['prev_delta'] = straddle_delta
            active_position['prev_gamma'] = straddle_gamma
            
            should_exit = False
            exit_reason = None
            option_pnl = active_position['entry_credit'] - current_straddle_value
            
            # Position-based stop loss (lose 2x premium collected)
            if option_pnl < -2 * active_position['entry_credit']:
                should_exit = True
                exit_reason = f"Position stop loss (2x premium): P&L ${option_pnl:.0f}"
            elif days_held >= 14:
                # Day 14: EXIT regardless
                should_exit = True
                exit_reason = "Day 14 mandatory exit"
            elif days_held >= 10:
                # Day 10-13: REVERSED - profit when IV drops, stop when IV spikes
                if iv_change_pct <= -2.0:
                    should_exit = True
                    exit_reason = f"Day {days_held} take profit (IV {iv_change_pct:.1f}%)"
                elif iv_change_pct >= 3.0:
                    should_exit = True
                    exit_reason = f"Day {days_held} stop loss (IV +{iv_change_pct:.1f}%)"
            elif days_held >= 7:
                # Day 7-9: REVERSED - profit when IV drops significantly
                if iv_change_pct <= -3.0:
                    should_exit = True
                    exit_reason = f"Day {days_held} take profit (IV {iv_change_pct:.1f}%)"
                elif iv_change_pct >= 2.0:
                    should_exit = True
                    exit_reason = f"Day {days_held} stop loss (IV +{iv_change_pct:.1f}%)"
            
            if should_exit:
                cumulative_hedge_pnl = active_position.get('cumulative_hedge_pnl', 0)
                exit_cost = tc_calc.calculate(
                    price=current_straddle_value / (100 * active_position['num_straddles']),
                    contracts=active_position['num_straddles'] * 2,
                    ticker=ticker_upper
                )
                
                entry_cost_tc = active_position.get('entry_tc', 0)
                total_hedge_rebalance_costs = active_position.get('total_hedge_cost', 0)
                gross_pnl = option_pnl + cumulative_hedge_pnl
                net_pnl = gross_pnl - entry_cost_tc - exit_cost
                
                portfolio_value += net_pnl
                trades_exited += 1
                
                trade_log.append({
                    'entry_date': active_position['entry_date'],
                    'exit_date': current_date,
                    'days_held': days_held,
                    'strike': active_position['strike'],
                    'entry_iv': active_position['entry_iv'],
                    'exit_iv': current_iv,
                    'iv_change': iv_change,
                    'iv_change_pct': iv_change_pct,
                    'entry_credit': active_position['entry_credit'],
                    'exit_cost': current_straddle_value,
                    'option_pnl': option_pnl,
                    'gamma_pnl': active_position.get('cumulative_gamma_pnl', 0),
                    'theta_pnl': active_position.get('cumulative_theta_pnl', 0),
                    'delta_hedge_pnl': active_position.get('cumulative_delta_pnl', 0),
                    'hedge_pnl': cumulative_hedge_pnl,
                    'gross_pnl': gross_pnl,
                    'entry_tc': entry_cost_tc,
                    'exit_tc': exit_cost,
                    'hedge_rebalance_costs': total_hedge_rebalance_costs,
                    'net_pnl': net_pnl,
                    'exit_reason': exit_reason,
                    'garch_forecast': active_position['garch_forecast'],
                    'num_straddles': active_position['num_straddles']
                })
                
                if verbose:
                    print(f"\n  [{current_date.date()}] EXIT: {exit_reason} | P&L: ${net_pnl:.2f}")
                
                active_position = None
            else:
                active_position['dte_remaining'] = max(1, active_position['entry_dte'] - days_held)
                active_position['current_iv'] = current_iv
                active_position['current_value'] = current_straddle_value
            
            results.append({
                'date': current_date,
                'spot_price': spot_price,
                'forecast_iv': garch_forecast,
                'market_iv': current_iv if current_iv else np.nan,
                'position_status': 'HOLDING' if active_position else 'EXITED',
                'days_held': days_held,
                'iv_change_pct': iv_change_pct,
                'portfolio_value': portfolio_value,
                'trade_pnl_dollars': net_pnl if should_exit else 0,
                'traded': should_exit
            })
            processed_count += 1
            continue
        
        strikes, maturities, market_prices = get_iv_surface_for_date(
            options_data, current_date, spot_price
        )
        
        if strikes is None:
            skipped_count += 1
            if verbose:
                print(f"  [{current_date.date()}] Insufficient options data, skipping...")
            continue
        
        try:
            iv_calc = ImpliedVolSurface(
                spot_price=spot_price,
                risk_free_rate=0.02,
                dividend_yield=0.02,
                strikes=strikes,
                maturities=maturities,
                market_prices=market_prices,
                verbose=False
            )
            
            if iv_calc.iv_surface is None:
                skipped_count += 1
                continue
            
            atm_option = get_atm_option_for_dte(
                options_data, current_date, spot_price,
                target_dte=45, dte_range=(30, 45)
            )
            
            if atm_option is None:
                skipped_count += 1
                if verbose:
                    print(f"  [{current_date.date()}] No 30-45 DTE options available")
                continue
            
            market_iv = get_iv_for_option(
                iv_calc, atm_option['strike'], atm_option['maturity'],
                atm_option['call_price'], atm_option['put_price']
            )
            
            if np.isnan(market_iv):
                skipped_count += 1
                continue
            
            iv_history.append(market_iv)
            if len(iv_history) > 252:
                iv_history = iv_history[-252:]
            
            if garch_forecast is not None:
                forecast_history.append(garch_forecast)
                market_iv_history.append(market_iv)
                if len(forecast_history) > 252:
                    forecast_history = forecast_history[-252:]
                    market_iv_history = market_iv_history[-252:]
            
            iv_diff = market_iv - garch_forecast
            garch_signal = iv_diff > 0.02
            if len(iv_history) >= 20:  # Need some history
                iv_percentile = (np.array(iv_history) < market_iv).sum() / len(iv_history) * 100
                iv_expensive = iv_percentile > 75  # Top 25% - expensive
            else:
                iv_expensive = True  # Not enough history, assume expensive
                iv_percentile = 50
            
            # 3. 30-45 DTE options available (already checked above)
            dte_available = True
            
            # 4. No stress regime
            is_blocked = False
            if regime_blocker is not None:
                try:
                    is_blocked = regime_blocker.isBlocked(current_date.strftime('%Y-%m-%d'))
                except:
                    is_blocked = False
            
            if garch_signal and iv_expensive and dte_available and not is_blocked:
                signal = 'SELL'
            else:
                signal = 'HOLD'
                if is_blocked:
                    blocked_count += 1
            
            entry_notes = []
            if not garch_signal:
                entry_notes.append(f"Market IV not expensive (diff {iv_diff:.1%} < 2%)")
            if not iv_expensive:
                entry_notes.append(f"IV percentile {iv_percentile:.0f} < 75")
            if is_blocked:
                entry_notes.append("Stress regime")
            
            if signal == 'SELL':
                straddle_price_per_unit = atm_option['straddle_price'] * 100
                num_straddles = max(1, int(position_size / straddle_price_per_unit))
                entry_credit = straddle_price_per_unit * num_straddles
                max_notional_risk = starting_capital * 0.15
                if entry_credit > max_notional_risk:
                    if verbose:
                        print(f"\n  [{current_date.date()}] SKIPPED: Position too large (${entry_credit:.0f} > ${max_notional_risk:.0f})")
                    signal = 'HOLD'  # Override signal
                    continue
                
                # Entry transaction costs
                entry_tc = tc_calc.calculate(
                    price=atm_option['straddle_price'],
                    contracts=num_straddles * 2,
                    ticker=ticker_upper
                )
                
                hedger = DeltaHedger(spot_price, risk_free_rate=0.02, dividend_yield=0.02)
                hedge_result = hedger.calculate_hedge_position(
                    spot_price, atm_option['strike'], atm_option['maturity'], market_iv
                )
                
                straddle_delta = hedger.calculate_straddle_delta(
                    spot_price, atm_option['strike'], atm_option['maturity'], 
                    0.02, market_iv, 0.02
                )
                d1 = (np.log(spot_price / atm_option['strike']) + 
                      (0.02 - 0.02 + 0.5 * market_iv**2) * atm_option['maturity']) / \
                     (market_iv * np.sqrt(atm_option['maturity']))
                straddle_gamma = np.exp(-0.02 * atm_option['maturity']) * norm.pdf(d1) / \
                                 (spot_price * market_iv * np.sqrt(atm_option['maturity'])) * 2
                active_position = {
                    'entry_date': current_date,
                    'strike': atm_option['strike'],
                    'entry_dte': atm_option['dte'],
                    'dte_remaining': atm_option['dte'],
                    'exdate': atm_option['exdate'],
                    'entry_iv': market_iv,
                    'garch_forecast': garch_forecast,
                    'entry_credit': entry_credit,
                    'entry_tc': entry_tc,
                    'num_straddles': num_straddles,
                    'entry_spot': spot_price,
                    'prev_spot': spot_price,
                    'prev_hedge_units': hedge_result['hedge_units'],
                    'prev_delta': straddle_delta,
                    'prev_gamma': straddle_gamma,
                    'cumulative_hedge_pnl': 0,
                    'cumulative_gamma_pnl': 0,
                    'cumulative_theta_pnl': 0,
                    'cumulative_delta_pnl': 0,
                    'total_hedge_cost': 0,
                    'is_short': True
                }
                
                trades_entered += 1
                
                if verbose:
                    print(f"\n  [{current_date.date()}] ENTRY (SELL): Strike ${atm_option['strike']:.0f}, "
                          f"DTE {atm_option['dte']}, IV {market_iv:.1%}, GARCH {garch_forecast:.1%}, "
                          f"Credit ${entry_credit:.0f}")
            
            results.append({
                'date': current_date,
                'spot_price': spot_price,
                'forecast_iv': garch_forecast,
                'market_iv': market_iv,
                'iv_spread': iv_diff,
                'iv_percentile': iv_percentile if len(iv_history) >= 20 else np.nan,
                'signal': signal,
                'blocked': is_blocked,
                'entry_notes': '; '.join(entry_notes) if entry_notes else '',
                'atm_strike': atm_option['strike'],
                'atm_dte': atm_option['dte'],
                'straddle_price': atm_option['straddle_price'],
                'portfolio_value': portfolio_value,
                'trade_pnl_dollars': 0,
                'traded': signal == 'SELL'
            })
            processed_count += 1
            
        except Exception as e:
            skipped_count += 1
            if verbose:
                print(f"  [{current_date.date()}] Error: {str(e)}")
            continue
    
    print(f"\n\n[5/6] Processing results...")
    print(f"  Processed: {processed_count}, Skipped: {skipped_count}, Blocked: {blocked_count}")
    print(f"  Trades entered: {trades_entered}, Trades exited: {trades_exited}")
    
    if len(forecast_history) > 0 and len(market_iv_history) > 0:
        print(f"\n  Forecast Quality Analysis:")
        forecast_arr = np.array(forecast_history)
        market_arr = np.array(market_iv_history)
        
        forecast_error = forecast_arr - market_arr
        print(f"    Mean Forecast:          {forecast_arr.mean():.2%}")
        print(f"    Mean Market IV:         {market_arr.mean():.2%}")
        print(f"    Mean Forecast Error:    {forecast_error.mean():.2%}")
        print(f"    RMSE:                   {np.sqrt((forecast_error**2).mean()):.2%}")
        print(f"    Correlation:            {np.corrcoef(forecast_arr, market_arr)[0,1]:.3f}")
        print(f"    Calibrated Risk Prem:   {calculate_vol_risk_premium(forecast_history, market_iv_history):.2%}")
    
    df_results = pd.DataFrame(results)
    df_trades = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    
    if len(df_results) == 0:
        print("  [!] No valid backtest results")
        return None
    
    print(f"  * Generated {len(df_results)} daily observations")
    print(f"  * Completed {len(df_trades)} round-trip trades")
    print(f"  * Regime blocker prevented {blocked_count} entries during stress periods")
    
    print("\n[6/6] Performance Analysis...")
    
    if len(df_trades) > 0:
        print(f"\n  Trade Statistics:")
        print(f"    Total Trades:     {len(df_trades):>6}")
        print(f"    Winning Trades:   {(df_trades['net_pnl'] > 0).sum():>6}")
        print(f"    Losing Trades:    {(df_trades['net_pnl'] < 0).sum():>6}")
        print(f"    Win Rate:         {(df_trades['net_pnl'] > 0).mean()*100:>6.1f}%")
        
        print(f"\n  Holding Period:")
        print(f"    Mean:             {df_trades['days_held'].mean():>6.1f} days")
        print(f"    Median:           {df_trades['days_held'].median():>6.1f} days")
        print(f"    Min:              {df_trades['days_held'].min():>6} days")
        print(f"    Max:              {df_trades['days_held'].max():>6} days")
        
        print(f"\n  IV Change (Entry to Exit):")
        print(f"    Mean:             {df_trades['iv_change_pct'].mean():>7.2f}%")
        print(f"    Median:           {df_trades['iv_change_pct'].median():>7.2f}%")
        print(f"    Std:              {df_trades['iv_change_pct'].std():>7.2f}%")
        
        print(f"\n  P&L Statistics:")
        print(f"    Total Gross P&L:  ${df_trades['gross_pnl'].sum():>10,.2f}")
        print(f"    Total Net P&L:    ${df_trades['net_pnl'].sum():>10,.2f}")
        print(f"    Mean Trade P&L:   ${df_trades['net_pnl'].mean():>10,.2f}")
        print(f"    Median Trade P&L: ${df_trades['net_pnl'].median():>10,.2f}")
        print(f"    Best Trade:       ${df_trades['net_pnl'].max():>10,.2f}")
        print(f"    Worst Trade:      ${df_trades['net_pnl'].min():>10,.2f}")
        
        print(f"\n  Exit Reasons:")
        for reason, count in df_trades['exit_reason'].value_counts().items():
            print(f"    {reason}: {count}")
        
        print(f"\n  P&L Breakdown (SHORT VOL):")
        print(f"    Option P&L:         ${df_trades['option_pnl'].sum():>10,.2f}  (profit when IV drops)")
        print(f"    Gamma P&L:          ${df_trades['gamma_pnl'].sum():>10,.2f}  (cost from spot moves)")
        print(f"    Theta P&L:          ${df_trades['theta_pnl'].sum():>10,.2f}  (GAIN from decay)")
        print(f"    Delta Hedge P&L:    ${df_trades['delta_hedge_pnl'].sum():>10,.2f}")
        print(f"    Total Hedge P&L:    ${df_trades['hedge_pnl'].sum():>10,.2f}")
        
        print(f"\n  Transaction Costs:")
        print(f"    Entry Costs:        ${df_trades['entry_tc'].sum():>10,.2f}")
        print(f"    Exit Costs:         ${df_trades['exit_tc'].sum():>10,.2f}")
        print(f"    Hedge Rebal Costs:  ${df_trades['hedge_rebalance_costs'].sum():>10,.2f}")
    
    # Entry criteria distribution
    if 'signal' in df_results.columns:
        print(f"\n  Signal Distribution:")
        for sig in ['SELL', 'HOLD']:
            count = (df_results['signal'] == sig).sum()
            pct = count / len(df_results) * 100
            print(f"    {sig}: {count:>4} ({pct:>5.1f}%)")
    
    if 'iv_spread' in df_results.columns:
        iv_spreads = df_results['iv_spread'].dropna()
        if len(iv_spreads) > 0:
            print(f"\n  Market IV - GARCH Spread:")
            print(f"    Mean:             {iv_spreads.mean():>7.2%}")
            print(f"    Median:           {iv_spreads.median():>7.2%}")
            print(f"    % Above 2%:       {(iv_spreads > 0.02).mean()*100:>6.1f}%")
    
    print(f"\n  Final Portfolio Value: ${portfolio_value:,.2f}")
    print(f"  Total Return:          {(portfolio_value - starting_capital) / starting_capital * 100:.2f}%")
    
    print("\n  Backtest complete!")
    print("="*80)
    
    df_results.attrs['trade_log'] = df_trades
    
    return df_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    else:
        ticker = 'AAPL'
        print(f"No ticker specified, using default: {ticker}")
        print(f"Usage: python main_short_vol.py <TICKER>")
        print(f"Example: python main_short_vol.py AAPL\n")
    
    results = rolling_window_backtest(
        ticker=ticker,
        train_window=126,
        refit_frequency=21,
        starting_capital=100000,
        position_size=8000,
        use_regime_blocker=True,
        start_date=None,
        end_date=None,
        verbose=True
    )
    
    if results is not None:
        results.to_csv(f"backtest_results_{ticker.lower()}_SHORT_VOL.csv", index=False)
        print(f"\nDaily results saved to backtest_results_{ticker.lower()}_SHORT_VOL.csv")
        
        trade_log = results.attrs.get('trade_log', pd.DataFrame())
        if len(trade_log) > 0:
            trade_log.to_csv(f"trade_log_{ticker.lower()}_SHORT_VOL.csv", index=False)
            print(f"Trade log saved to trade_log_{ticker.lower()}_SHORT_VOL.csv")