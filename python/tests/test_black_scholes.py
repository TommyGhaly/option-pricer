import ctypes


options = ctypes.CDLL("../../cpp/core/liboptions.so")


# Example: set argtypes and restype for black_scholes
options.bs_option_price.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                         ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
options.bs_option_price.restype = ctypes.c_double

# Call the function
price = options.bs_option_price(100, 100, 0.05, 0.0, 1, 0.2, 1)
print("Call price:", price)
