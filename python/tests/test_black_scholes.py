import importlib.util

# Import your module (using the method that worked)

so_file = "/Users/tommyghaly/Desktop/quant_finance_journey/option_pricer/option_pricer.cpython-39-darwin.so"
spec = importlib.util.spec_from_file_location("option_pricer", so_file)
option_pricer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(option_pricer)



print(f'Black-Scholes Call Price: {option_pricer.black_scholes(100, 100, 0.05, 0.2, 1)}')
print(f'Black-Scholes Put Price: {option_pricer.black_scholes(100, 100, 0.05, 0.2, 0)}')
