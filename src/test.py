import numpy
import talib

print(numpy.__version__)  # Should be 1.21.4 or similar
print(talib.get_functions())  # Should print available TA-Lib functions without errors
