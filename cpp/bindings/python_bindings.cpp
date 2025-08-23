#include <pybind11/pybind11.h>
#include "../core/black_scholes.h"
#include "../core/binomial_tree.h"
#include "../core/greeks.h"
#include "../core/monte_carlo.h"

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
    m.def("gamma", &gamma, "Calculate Gamma of the option",
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
}
