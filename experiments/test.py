import numpy as np
data = np.load('../data/raw/respiratory_only.npy')
print(data.shape)
print(data.dtype)
print(data.min(), data.max())