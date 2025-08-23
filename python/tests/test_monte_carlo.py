import ctypes



options = ctypes.CDLL("../../cpp/core/liboptions.so")


options.mc_american_option.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                         ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int]
options.mc_american_option.restype = ctypes.c_double


options.mc_asian_option.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                      ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int]
options.mc_asian_option.restype = ctypes.c_double


# Call the functions
american_price = options.mc_american_option(100, 100, 0.05, 0.0, 1, 0.2, 10000, 100, 1)
asian_price = options.mc_asian_option(100, 100, 0.05, 0.0, 1, 0.2, 10000, 100, 1)
print("American option price:", american_price)
print("Asian option price:", asian_price)
