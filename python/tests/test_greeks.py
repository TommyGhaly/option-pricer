import ctypes

options = ctypes.CDLL("../../cpp/core/liboptions.so")


options.delta_wrapper.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                  ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
options.delta_wrapper.restype = ctypes.c_double

options.gamma_wrapper.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                  ctypes.c_double, ctypes.c_double, ctypes.c_double]
options.gamma_wrapper.restype = ctypes.c_double

options.theta_wrapper.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                  ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
options.theta_wrapper.restype = ctypes.c_double

options.vega_wrapper.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                 ctypes.c_double, ctypes.c_double, ctypes.c_double]
options.vega_wrapper.restype = ctypes.c_double

options.rho_wrapper.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                 ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
options.rho_wrapper.restype = ctypes.c_double


def test_greeks() -> None:
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2

    delta_call = options.delta_wrapper(S, K, T, r, 0.0, sigma, 1)
    delta_put = options.delta_wrapper(S, K, T, r, 0.0, sigma, 0)
    print(f"Delta Call: {delta_call}, Delta Put: {delta_put}")

    gamma = options.gamma_wrapper(S, K, T, r, 0.0, sigma)
    print(f"Gamma: {gamma}")


    theta_call = options.theta_wrapper(S, K, T, r, 0.0, sigma, 1)
    theta_put = options.theta_wrapper(S, K, T, r, 0.0, sigma, 0)
    print(f"Theta Call: {theta_call}, Theta Put: {theta_put}")


    vega = options.vega_wrapper(S, K, T, r, 0.0, sigma)
    print(f"Vega: {vega}")

    rho_call = options.rho_wrapper(S, K, T, r, 0.0, sigma, 1)
    rho_put = options.rho_wrapper(S, K, T, r, 0.0, sigma, 0)
    print(f"Rho Call: {rho_call}, Rho Put: {rho_put}")

if __name__ == "__main__":
    test_greeks()
    print("All tests passed.")
