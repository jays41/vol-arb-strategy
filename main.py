from preprocess_data import get_log_returns
from garch import garch_modelling


DE_MEAN = "AR"   # arch_model mean specification: "Constant" or "AR" (demeaning built in)
MODEL = "GARCH"  # vol model: "GARCH", "EGARCH"
DISTRIBUTION = {"GARCH": "normal", "EGARCH": "t"}[MODEL]
validity_checks = True

if __name__ == "__main__":

    log_returns = get_log_returns("vol-arb-strategy/s&p_data.csv")

    print(log_returns)

    garch_results, sigma_forecast = garch_modelling(log_returns, DE_MEAN, MODEL, DISTRIBUTION, validity_checks)

# Next Steps:
#   - Convert variance forecasts, i.e. IV comparison
#   - Build a delta-hedged vol PnL
#   - Add rolling re-estimation
#   - Test signal stability over regimes