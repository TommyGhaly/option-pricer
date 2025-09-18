#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../include/models.h"
#include "../include/option_pricer.h"

namespace py = pybind11;

PYBIND11_MODULE(option_pricer, m) {
    m.doc() = "Option Pricing Module using C++ and pybind11";

    // Black-Scholes functions
    m.def("black_scholes", &black_scholes, "Calculate European option price using Black-Scholes formula",
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("q"), py::arg("T"), py::arg("sigma"), py::arg("is_call"));

    // Binomial Tree functions
    m.def("binomial_tree_euro", &binomial_tree_euro, "Price European option using Binomial Tree model",
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("q"), py::arg("sigma"), py::arg("T"), py::arg("N"), py::arg("is_call"));
    m.def("binomial_tree_american", &binomial_tree_american, "Price American option using Binomial Tree model",
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("q"), py::arg("sigma"), py::arg("T"), py::arg("N"), py::arg("is_call"));

    // Monte Carlo functions
    m.def("monte_carlo_american_option", &monte_carlo_american_option, "Price American option using Monte Carlo simulation",
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("q"), py::arg("sigma"), py::arg("T"),
          py::arg("steps"), py::arg("num_simulations"), py::arg("is_call"));
    m.def("monte_carlo_asian_option", &monte_carlo_asian_option, "Price Asian option using Monte Carlo simulation",
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("q"), py::arg("sigma"), py::arg("T"),
          py::arg("steps"), py::arg("num_simulations"), py::arg("is_call"));

    // Greeks functions
    m.def("delta", &delta, "Calculate Delta of the option",
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("q"), py::arg("T"), py::arg("sigma"), py::arg("is_call"));
    m.def("gamma", [](double S, double K, double r, double q, double T, double sigma) {
        return gamma(S, K, r, q, T, sigma);
    }, "Calculate Gamma of the option",
    py::arg("S"), py::arg("K"), py::arg("r"), py::arg("q"), py::arg("T"), py::arg("sigma"));
    m.def("theta", &theta, "Calculate Theta of the option",
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("q"), py::arg("T"), py::arg("sigma"), py::arg("is_call"));
    m.def("vega", &vega, "Calculate Vega of the option",
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("q"), py::arg("T"), py::arg("sigma"));
    m.def("rho", &rho, "Calculate Rho of the option",
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("q"), py::arg("T"), py::arg("sigma"), py::arg("is_call"));
    m.def("vanna", &vanna, "Calculate Vanna of the option",
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("q"), py::arg("T"), py::arg("sigma"));
    m.def("charm", &charm, "Calculate Charm of the option",
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("q"), py::arg("T"), py::arg("sigma"), py::arg("is_call"));
    m.def("vomma", &vomma, "Calculate Vomma of the option",
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("q"), py::arg("T"), py::arg("sigma"));
    m.def("veta", &veta, "Calculate Veta of the option",
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("q"), py::arg("T"), py::arg("sigma"));

    // Heston Model function - corrected signature
    m.def("heston_model", &heston_model, "Price option using Heston model",
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("q"), py::arg("T"),
          py::arg("v0"), py::arg("kappa"), py::arg("theta"), py::arg("sigma"),
          py::arg("rho"), py::arg("lambda"), py::arg("is_call"));

    // Jump Diffusion Model function
    m.def("jump_diffusion", &jump_diffusion, "Price option using Merton's Jump Diffusion model",
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("q"), py::arg("sigma"),
          py::arg("lambda"), py::arg("mu_j"), py::arg("sigma_j"), py::arg("T"), py::arg("is_call"));

    // Local Volatility Model functions
    m.def("european_local_vol_fdm", &european_local_vol_fdm, "Price European option using Local Volatility model with FDM",
          py::arg("S0"), py::arg("K"), py::arg("r"), py::arg("q"), py::arg("T"),
          py::arg("implied_vol"), py::arg("is_call"), py::arg("N_S") = 200, py::arg("N_T") = 100);
    m.def("american_local_vol_fdm", &american_local_vol_fdm, "Price American option using Local Volatility model with FDM",
          py::arg("S0"), py::arg("K"), py::arg("r"), py::arg("q"), py::arg("T"),
          py::arg("implied_vol"), py::arg("is_call"), py::arg("N_S") = 200, py::arg("N_T") = 100);

    // SABR Model functions
    m.def("SABRImpliedVol", &SABRImpliedVol, "Calculate SABR implied volatility",
          py::arg("F"), py::arg("K"), py::arg("T"), py::arg("alpha"), py::arg("beta"), py::arg("rho"), py::arg("nu"));
    m.def("SABROptionPrice", &SABROptionPrice, "Price option using SABR model",
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("T"),
          py::arg("F"), py::arg("alpha"), py::arg("beta"), py::arg("rho"), py::arg("nu"), py::arg("isCall"));

    // SABR Calibration - needs special handling for array parameters
    m.def("CalibratesSABR", [](py::array_t<double> strikes, py::array_t<double> marketVols,
                               double F, double T, double fixedBeta) {
        auto strikes_buf = strikes.request();
        auto vols_buf = marketVols.request();

        if (strikes_buf.ndim != 1 || vols_buf.ndim != 1) {
            throw std::runtime_error("Input arrays must be 1-dimensional");
        }

        if (strikes_buf.size != vols_buf.size) {
            throw std::runtime_error("Strikes and market vols must have same size");
        }

        double* strikes_ptr = static_cast<double*>(strikes_buf.ptr);
        double* vols_ptr = static_cast<double*>(vols_buf.ptr);
        int n = strikes_buf.size;

        double alpha, rho, nu;
        CalibratesSABR(strikes_ptr, vols_ptr, n, F, T, fixedBeta, alpha, rho, nu);

        return py::make_tuple(alpha, rho, nu);
    }, "Calibrate SABR model parameters",
       py::arg("strikes"), py::arg("marketVols"), py::arg("F"), py::arg("T"), py::arg("fixedBeta"));

    m.def("SABRDelta", &SABRDelta, "Calculate Delta using SABR model",
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("T"),
          py::arg("F"), py::arg("alpha"), py::arg("beta"), py::arg("rho"), py::arg("nu"), py::arg("isCall"));
    m.def("SABRGamma", &SABRGamma, "Calculate Gamma using SABR model",
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("T"),
          py::arg("F"), py::arg("alpha"), py::arg("beta"), py::arg("rho"), py::arg("nu"), py::arg("isCall"));
    m.def("SABRVega", &SABRVega, "Calculate Vega using SABR model",
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("T"),
          py::arg("F"), py::arg("alpha"), py::arg("beta"), py::arg("rho"), py::arg("nu"), py::arg("isCall"));
    m.def("SABRVolga", &SABRVolga, "Calculate Volga using SABR model",
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("T"),
          py::arg("F"), py::arg("alpha"), py::arg("beta"), py::arg("rho"), py::arg("nu"));
    m.def("SABRVanna", &SABRVanna, "Calculate Vanna using SABR model",
          py::arg("S"), py::arg("K"), py::arg("r"), py::arg("T"),
          py::arg("F"), py::arg("alpha"), py::arg("beta"), py::arg("rho"), py::arg("nu"));
}
