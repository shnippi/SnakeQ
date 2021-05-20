import numpy as np

limited = 5
side_padding = np.array([1 for i in range(limited)])
up_padding = np.array([side_padding for x in range(limited)])
print(up_padding)