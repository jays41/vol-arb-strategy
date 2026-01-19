from preprocess_data import get_log_returns
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import het_arch
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera
from garch import garch_modeling


DE_MEAN = "AR"   # arch_model mean specification: "Constant" or "AR" (demeaning built in)
MODEL = "GARCH"  # vol model: "GARCH", "EGARCH"
DISTRIBUTION = {"GARCH": "normal", "EGARCH": "t"}[MODEL]
validity_checks = True

if __name__ == "__main__":

    log_returns = get_log_returns("vol-arb-strategy/s&p_data.csv")

    print(log_returns)

    garch_results, sigma_forecast = garch_modeling(log_returns)
