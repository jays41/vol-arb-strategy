import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import het_arch
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera

def garch_modelling(log_returns, DE_MEAN, MODEL, DISTRIBUTION, validity_checks):

    # ADF test
    adf_result = adfuller(log_returns)
    print("ADF statistic:", adf_result[0])
    print("p-value:", adf_result[1])
    
    # ACF plot
    # plot_acf(log_returns, lags=30)
    # plt.title("ACF of Returns")
    # plt.show()

    # ARCH LM test >> if p-value is small, there is ARCH effect
    arch_test = het_arch(log_returns)
    print("ARCH LM p-value:", arch_test[1])

    if arch_test[1] > 0.05:
        print("No ARCH effect detected in the series.")
        exit()

    if DE_MEAN == "Constant":
        garch = arch_model(
            log_returns * 100,      # scale improves numerical stability
            mean="Constant",
            vol=MODEL,
            p=1,
            q=1,
            dist=DISTRIBUTION
        )

    elif DE_MEAN == "AR":
        garch = arch_model(
            log_returns * 100,
            mean="AR",     # <-- AR mean
            lags=1,        # AR(1)
            vol=MODEL,
            p=1,
            q=1,
            dist=DISTRIBUTION
        )

    garch_res = garch.fit(update_freq=5)

    # print(garch_res.summary())

    # Validity checks
    if validity_checks:
        print("\n Validity Checks: \n")

        alpha = garch_res.params.get('alpha[1]', 0)
        beta = garch_res.params.get('beta[1]', 0)
        res_check = alpha + beta
        if res_check >= 1:
            print("⚠️ WARNING: Model may be non-stationary (alpha + beta >= 1).")
        else:
            print("✅ Stationarity test passed.")

        std_resid = garch_res.std_resid

        rolling_var = std_resid.rolling(250).var()
        deviation = np.abs(rolling_var - 1)
        if deviation.dropna().max() > 0.25:
            print("⚠️ WARNING: Rolling variance of standardized residuals unstable.")
        else:
            print("✅ Rolling Variance test passed.")

        lb_test = acorr_ljungbox(std_resid, lags=[10, 20], return_df=True)
        if lb_test.iloc[0, 0] < 0.05:
            print("⚠️ WARNING: Ljung-Box test indicates remaining autocorrelation in standardized residuals.")
        else:
            print("✅ Ljung-Box test passed.")

        lb_test_sq = acorr_ljungbox(std_resid**2, lags=[10, 20], return_df=True)
        if lb_test_sq.iloc[0, 0] < 0.05:
            print("⚠️ WARNING: Ljung-Box Squared test indicates remaining autocorrelation in standardized residuals.")
        else:
            print("✅ Ljung-Box Squared test passed.")

        if MODEL != "EGARCH":
            std_resid_clean = std_resid.dropna()
            arch_test_resid = het_arch(std_resid_clean)
            if arch_test_resid[1] < 0.05:
                print("⚠️ WARNING: ARCH effect remains in standardized residuals.")
            else:
                print("✅ ARCH-ML Passed.")

        jb_stat, jb_p = jarque_bera(std_resid)
        if jb_p < 0.05:
            print("⚠️ WARNING: Standardized residuals are not normally distributed.")
        else:
            print("✅ JB test passed.")


    # cond_vol = garch_res.conditional_volatility
    # plt.plot(cond_vol)
    # plt.title("Conditional Volatility")
    # plt.show()

    # plot_acf(cond_vol, lags=30)
    # plt.title("ACF of Conditional Volatility")
    # plt.show()

    forecast = garch_res.forecast(horizon=1)
    sigma_forecast = forecast.variance.iloc[-1, 0] ** 0.5
    print("Next period volatility forecast:", sigma_forecast)

    return garch_res, sigma_forecast


