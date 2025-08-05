import numpy as np

a = np.random.randn(10, 5)
b = np.random.randn(10)
print(a.shape)
print(b.shape)
c2 = (a.T / b).T
print(c2.shape)
c1 = a / b[:, np.newaxis]
print(c1.shape)
print(c1 == c2)
